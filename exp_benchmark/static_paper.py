"""Source-pinned static EMA measurements for the paper's mixed and single attributes."""

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import uuid

import numpy as np

from .aggregate import qps_at_recall
from .dynamic.gt import array_digest, read_fvecs
from .dynamic.runtime import (
    DEFAULT_LIMITS, OutputLock, atomic_json, choose_core, environment_config,
    file_info, immutable_json, import_native, native_state, numa_snapshot,
    read_attribute_index_header, require_local, require_unchanged,
    timed_query, timestamp, verify_index_tail, write_npz,
)
from .figure4a import (
    and_predicates, composite_predicates, ground_truth_prefix, matching_counts,
    read_json, validate_ids,
)


SCHEMA = "ema-static-paper-v1"
REPO = Path(__file__).resolve().parents[1]
SOURCE_PATHS = (
    "exp_benchmark/static_paper.py", "exp_benchmark/figure4a.py",
    "exp_benchmark/aggregate.py", "exp_benchmark/dynamic/runtime.py",
    "exp_benchmark/dynamic/gt.py", "tests/hashann.py",
)
PARAMETERS = {
    "M": 40, "ef_construction": 300, "ft_bits": 128, "edge_level_ft": True,
    "routing_min_deg": 16, "ef_top": 1, "thresholds": [0.0001] * 3,
    "build_threads": 32, "query_threads": 1, "k": 10,
    "tail_backfill": False, "augmentation": False,
}


def source_identity():
    return {name: file_info(REPO / name) for name in SOURCE_PATHS}


def load_builder():
    path = REPO / "tests/hashann.py"
    spec = importlib.util.spec_from_file_location("ema_static_native_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import the repository's native index builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class Attributes:
    kinds: tuple
    records: list
    numeric: np.ndarray
    categorical: np.ndarray

    @classmethod
    def from_records(cls, records, kinds, count):
        kinds = tuple(kinds)
        if kinds not in ((0,), (1,), (0, 1)):
            raise ValueError("Static paper profiles require [0], [1], or [0,1] attributes")
        if not isinstance(records, list) or len(records) != count:
            raise ValueError("Attribute count differs from the declared full dataset")
        numeric = np.zeros(count, dtype=np.int32)
        categorical = np.zeros(count, dtype=np.uint32)
        for row, record in enumerate(records):
            if not isinstance(record, list) or len(record) != len(kinds):
                raise ValueError(f"Attribute layout mismatch at source row {row}")
            for kind, values in zip(kinds, record):
                if not isinstance(values, list) or any(type(x) is not int for x in values):
                    raise ValueError(f"Non-integer attribute record at source row {row}")
                if kind == 0:
                    if len(values) != 1 or not -(2**31) <= values[0] < 2**31:
                        raise ValueError(f"Expected one int32 numeric attribute at source row {row}")
                    numeric[row] = values[0]
                else:
                    if any(x < 0 or x >= 32 for x in values) or len(set(values)) != len(values):
                        raise ValueError(f"Invalid synthetic categorical labels at source row {row}")
                    mask = 0
                    for label in values:
                        mask |= 1 << label
                    categorical[row] = mask
        return cls(kinds, records, numeric, categorical)

    @classmethod
    def load(cls, path, kinds, count):
        return cls.from_records(read_json(path), kinds, count)

    def words(self):
        columns = [self.numeric.view(np.uint32) if kind == 0 else self.categorical
                   for kind in self.kinds]
        return np.column_stack(columns)


@dataclass
class Predicates:
    raw: list
    bounds: np.ndarray
    masks: np.ndarray
    tails: np.ndarray | None
    dnf: bool

    @classmethod
    def parse(cls, raw, kinds, count, dnf=False):
        kinds = tuple(kinds)
        if not isinstance(raw, list) or len(raw) != count or count <= 0:
            raise ValueError("Original predicate count differs from the explicit query count")
        tails = None
        if dnf:
            if kinds != (0, 1):
                raise ValueError("Only the saved mixed two-branch DNF profile is supported")
            bounds, masks, tails = composite_predicates(raw, count)
        elif kinds == (0, 1):
            bounds, masks = and_predicates(raw, count)
        elif kinds == (0,):
            if any(not isinstance(p, list) or len(p) != 1 for p in raw):
                raise ValueError("Expected single-attribute numeric predicates [[low,high]]")
            bounds, _ = and_predicates([[p[0], [0]] for p in raw], count)
            masks = np.zeros(count, dtype=np.uint32)
        elif kinds == (1,):
            if any(not isinstance(p, list) or len(p) != 1 for p in raw):
                raise ValueError("Expected single-attribute categorical predicates [[labels]]")
            bounds, masks = and_predicates(
                [[[-(2**31), 2**31 - 1], p[0]] for p in raw], count)
        else:
            raise ValueError("Unsupported static attribute layout")
        if np.any(bounds < -(2**31)) or np.any(bounds >= 2**31):
            raise ValueError("Numeric predicate bounds exceed native int32")
        return cls(raw, bounds, masks, tails, dnf)

    def matches(self, ids, attributes):
        numeric, categorical = attributes.numeric[ids], attributes.categorical[ids]
        result = ((numeric >= self.bounds[:, :1]) & (numeric <= self.bounds[:, 1:])
                  & ((categorical & self.masks[:, None]) == self.masks[:, None]))
        if self.tails is not None:
            result |= (categorical & self.tails[:, None]) == self.tails[:, None]
        return result

    def counts(self, attributes):
        return matching_counts(attributes.numeric, attributes.categorical,
                               self.bounds, self.masks, self.tails)


def prepare_vectors(path, count, dimension, metric, normalization):
    if metric not in ("l2", "ip") or normalization not in ("none", "l2"):
        raise ValueError("Explicit original L2/IP metric and normalization policy are required")
    if metric == "l2" and normalization != "none":
        raise ValueError("Do not normalize the original SIFT L2 workload")
    raw = read_fvecs(path, count, dimension, integer_sift=metric == "l2")
    vectors = np.ascontiguousarray(raw)
    norms = []
    for first in range(0, count, 65_536):
        block = vectors[first:first + 65_536]
        squared = np.einsum("ij,ij->i", block, block, dtype=np.float64)
        norms.append((float(squared.min()), float(squared.max())))
        if normalization == "l2":
            lengths = np.linalg.norm(block, axis=1, keepdims=True)
            if np.any(lengths == 0) or not np.isfinite(lengths).all():
                raise ValueError("Cannot normalize zero/nonfinite embedding vectors")
            block /= lengths
    return vectors, {
        "normalization": normalization, "rows": count, "dimension": dimension,
        "source_squared_norm_min": min(x[0] for x in norms),
        "source_squared_norm_max": max(x[1] for x in norms),
        "prepared_payload_sha256": hashlib.sha256(memoryview(vectors)).hexdigest(),
    }


def verify_records(path, header, attributes, vectors):
    count, stride = header["count"], header["record_bytes"]
    start = header["records_offset"]
    if (header["attr_type"] != list(attributes.kinds) or header["attr_words"] != len(attributes.kinds)
            or header["label_offset"] - header["vector_offset"] != vectors.shape[1] * 4):
        raise ValueError("Native checkpoint layout differs from the static dataset profile")
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    labels = np.ndarray((count,), dtype="<u8", buffer=raw,
                        offset=start + header["label_offset"], strides=(stride,))
    flags = np.ndarray((count,), dtype=np.uint8, buffer=raw,
                       offset=start + header["offset_level0"] + 2, strides=(stride,))
    words = np.ndarray((count, len(attributes.kinds)), dtype="<u4", buffer=raw,
                       offset=start + header["attr_offset"], strides=(stride, 4))
    expected = attributes.words()
    seen = np.zeros(count, dtype=bool)
    try:
        for first in range(0, count, 100_000):
            stop = min(first + 100_000, count)
            ids = labels[first:stop].astype(np.int64)
            if (np.any(ids < 0) or np.any(ids >= count) or seen[ids].any()
                    or len(np.unique(ids)) != len(ids)):
                raise RuntimeError("Checkpoint labels are not a full source-row permutation")
            seen[ids] = True
            if np.any(flags[first:stop] & 1):
                raise RuntimeError("A static checkpoint contains deleted source rows")
            if not np.array_equal(words[first:stop], expected[ids]):
                raise RuntimeError("Checkpoint attributes differ from the original source records")
        sample = np.unique(np.linspace(0, count - 1, min(count, 1024), dtype=np.int64))
        stored = np.ndarray(vectors.shape, dtype="<f4", buffer=raw,
                            offset=start + header["vector_offset"], strides=(stride, 4))
        if not np.array_equal(stored[sample], vectors[labels[sample].astype(np.int64)]):
            raise RuntimeError("Checkpoint vector/external-label mapping differs from the source")
        if not seen.all():
            raise RuntimeError("Checkpoint omits source rows")
    finally:
        raw._mmap.close()
    return {
        "all_external_labels_verified": True, "all_delete_flags_clear": True,
        "all_attribute_words_verified": True, "vector_mapping_samples": len(sample),
    }


def validate_result(result, truth, vectors, queries, attributes, predicates, metric, k):
    labels, distances = result
    ids = validate_ids(labels, len(vectors), len(queries), k)
    if distances.shape != ids.shape or not np.isfinite(distances).all():
        raise RuntimeError("Invalid native query distance array")
    if not predicates.matches(ids, attributes).all():
        raise RuntimeError("ANN result violates its original predicate")
    selected = vectors[ids].astype(np.float64)
    query = queries[:, None, :].astype(np.float64)
    if metric == "l2":
        exact = np.sum((selected - query) ** 2, axis=2)
        correct = np.array_equal(exact, distances)
    elif metric == "ip":
        products = selected * query
        exact = 1 - np.sum(products, axis=2)
        # Bound float32 accumulation error, including the native final1-dot subtraction.
        eps = np.finfo(np.float32).eps
        gamma = vectors.shape[1] * eps / (1 - vectors.shape[1] * eps)
        tolerance = gamma * np.sum(np.abs(products), axis=2) + 2 * eps * (1 + np.abs(exact))
        correct = np.all(np.abs(exact - distances) <= tolerance)
    else:
        raise ValueError("Unsupported metric")
    if not correct:
        raise RuntimeError("Native distance/vector/source-label mismatch")
    if np.any(np.diff(distances, axis=1) < 0):
        raise RuntimeError("Native results are not ordered by increasing distance")
    recall = float(np.any(ids[:, :, None] == truth[:, None, :], axis=1).mean())
    return {
        "recall": recall, "labels_sha256": array_digest(labels),
        "distances_sha256": array_digest(distances),
    }


def summarize_target(points, medians, target):
    crossing = next((i for i, p in enumerate(points) if p["recall"] >= target), None)
    if crossing is None:
        raise RuntimeError(f"No ef reached recall {target}; no throughput substitution")
    selected = points[max(0, crossing - 1):crossing + 1]
    curve = [(p["ef"], p["recall"], medians[p["ef"]]) for p in selected]
    minimum_above = len(curve) == 1 and curve[0][1] > target
    qps = None if minimum_above else qps_at_recall(curve, target)
    if not minimum_above and (qps is None or not math.isfinite(qps) or qps <= 0):
        raise RuntimeError("No supported recall interpolation bracket")
    return {
        "target_recall": target, "qps_at_target": qps,
        "reported_qps": curve[-1][2] if minimum_above else qps,
        "observed_qps": curve[-1][2], "observed_recall": curve[-1][1],
        "selected_ef": curve[-1][0], "curve": curve,
        "method": ("observed_at_minimum_ef_NO_extrapolation" if minimum_above else
                   "observed_exact_target" if curve[-1][1] == target else
                   "adjacent_integer_ef_linear_interpolation"),
    }


def measure_cell(index, query, validate, query_count, targets, max_ef,
                 directory, environment, limits, progress):
    if not targets or len(set(targets)) != len(targets) or any(t not in (0.90, 0.95) for t in targets):
        raise ValueError("Static paper targets must explicitly be95%, or the plotted90% auxiliary")
    os.sched_setaffinity(0, environment["update_cpus"])
    core, _ = choose_core(environment["query_candidates"])
    os.sched_setaffinity(0, {core})
    points, labels, distances = [], [], []
    for ef in range(PARAMETERS["k"], max_ef + 1):
        index.set_ef(ef)
        result = query()
        point = {"ef": ef, **validate(result)}
        points.append(point)
        labels.append(result[0])
        distances.append(result[1])
        atomic_json(directory / "calibration.json", {"targets": targets, "points": points})
        progress("calibrating", ef=ef, recall=point["recall"])
        if point["recall"] >= max(targets):
            break
    if not points or points[-1]["recall"] < max(targets):
        raise RuntimeError(f"No ef reached recall {max(targets)} by ef={max_ef}")
    arrays = write_npz(directory / "calibration-arrays.npz",
                       labels=np.stack(labels), distances=np.stack(distances))
    needed = set()
    for target in targets:
        crossing = next(i for i, p in enumerate(points) if p["recall"] >= target)
        needed.update(range(max(0, crossing - 1), crossing + 1))
    bracket = [points[i] for i in sorted(needed)]
    attempts, accepted = [], None
    remaining = list(environment["query_candidates"])
    for _ in range(min(limits["max_cores"], len(remaining))):
        os.sched_setaffinity(0, environment["update_cpus"])
        core, loads = choose_core(remaining)
        remaining.remove(core)
        os.sched_setaffinity(0, {core})
        attempt = {"core": core, "selection_loads": loads, "rounds": [], "accepted_rounds": []}
        attempts.append(attempt)
        for repeat in range(limits["max_rounds_per_core"]):
            order = bracket[repeat % len(bracket):] + bracket[:repeat % len(bracket)]
            samples = []
            for point in order:
                index.set_ef(point["ef"])
                query()
                result, sample = timed_query(
                    query, core, environment["siblings"][str(core)], limits)
                actual = validate(result)
                if any(actual[key] != point[key] for key in actual):
                    raise RuntimeError("Frozen query labels/distances/recall changed during timing")
                sample.update(ef=point["ef"], recall=point["recall"],
                              qps=query_count / sample["wall_s"], **{
                                  key: actual[key] for key in ("labels_sha256", "distances_sha256")})
                samples.append(sample)
            attempt["rounds"].append(samples)
            if all(sample["clean"] for sample in samples):
                attempt["accepted_rounds"].append(repeat)
            atomic_json(directory / "timing.json", {"limits": limits, "attempts": attempts})
            progress("timing", core=core, round=repeat, accepted=len(attempt["accepted_rounds"]))
            if len(attempt["accepted_rounds"]) == limits["clean_rounds"]:
                accepted = attempt
                break
        if accepted is not None:
            break
    if accepted is None:
        raise RuntimeError("Insufficient CPU-clean timing rounds; no noisy-sample fallback")
    medians = {
        point["ef"]: statistics.median(
            sample["qps"] for repeat in accepted["accepted_rounds"]
            for sample in accepted["rounds"][repeat] if sample["ef"] == point["ef"])
        for point in bracket
    }
    return {
        "targets": [summarize_target(points, medians, target) for target in targets],
        "selected_core": accepted["core"], "accepted_rounds": accepted["accepted_rounds"],
        "calibration_arrays": arrays,
    }


def check_profile(job):
    build_only = job.get("build_only", False)
    if (job["metric"] not in ("l2", "ip") or job["N"] <= 0 or job["dimension"] <= 0
            or job["query_count"] not in (50, 100, 1000) or job["query_offset"] != 0
            or job["normalization"] not in ("none", "l2")
            or job["attribute_types"] not in ([0], [1], [0, 1])
            or (not job["cells"] and not build_only) or job["max_ef"] < PARAMETERS["k"]
            or (build_only and (job["cells"] or job.get("checkpoint") is not None))):
        raise ValueError("Invalid explicit static paper dataset/query profile")
    if len({cell["name"] for cell in job["cells"]}) != len(job["cells"]):
        raise ValueError("Duplicate query cells must be represented by one shared measurement")
    if not build_only and (any(cell["query_count"] not in (50, 100, 1000)
            or cell["query_count"] > job["query_count"]
            or cell["predicate_rows"] < cell["query_count"] for cell in job["cells"])
            or max(cell["query_count"] for cell in job["cells"]) != job["query_count"]):
        raise ValueError("Every cell must declare its exact original query/predicate prefix")
    if job.get("parameters") != PARAMETERS:
        raise ValueError("Static job does not use the frozen128-bit M40/efc300/routing16 protocol")


def run_job(job, output, native_dir, native_sha, expected_source, *, resume=False):
    check_profile(job)
    def progress(phase, **detail):
        atomic_json(output / "status.json", {
            "status": "running", "phase": phase, "updated_at": timestamp(), **detail})

    sources = source_identity()
    if sources != expected_source:
        raise RuntimeError("Executed static source differs from the frozen suite manifest")
    environment = environment_config(0, 32, requested_cores=[8, 16, 24],
                                     update_cpus=list(range(32)))
    limits = {**DEFAULT_LIMITS, "frequency_khz": [3_267_000, 3_333_000]}
    native, native_info = import_native(native_dir, native_sha)
    progress("verifying_original_inputs")
    inputs = {}
    for name in ("vectors", "queries", "attributes"):
        expected = job["inputs"][name]
        inputs[name] = file_info(expected["path"], expected)
    if inputs["vectors"]["bytes"] != job["N"] * (job["dimension"] + 1) * 4:
        raise ValueError("Base fvecs file is not exactly the declared full dataset")
    if inputs["queries"]["bytes"] != job["query_file_rows"] * (job["dimension"] + 1) * 4:
        raise ValueError("Original query-file dimensions differ from the audited workload")
    vectors, vector_meta = prepare_vectors(inputs["vectors"]["path"], job["N"], job["dimension"],
                                           job["metric"], job["normalization"])
    queries, query_meta = prepare_vectors(inputs["queries"]["path"], job["query_count"],
                                         job["dimension"], job["metric"], job["normalization"])
    attributes = Attributes.load(inputs["attributes"]["path"], job["attribute_types"], job["N"])
    cells = []
    for cell in job["cells"]:
        predicate_info = file_info(cell["predicate"]["path"], cell["predicate"])
        gt_info = file_info(cell["ground_truth"]["path"], cell["ground_truth"])
        raw_predicates = read_json(predicate_info["path"])
        if not isinstance(raw_predicates, list) or len(raw_predicates) != cell["predicate_rows"]:
            raise ValueError("Original predicate dimensions differ from the declared file")
        cell_queries = queries[:cell["query_count"]]
        predicates = Predicates.parse(raw_predicates[:len(cell_queries)], attributes.kinds,
                                      len(cell_queries), dnf=cell["dnf"])
        raw_truth = np.asarray(read_json(gt_info["path"]))
        stored_shape = (cell["ground_truth_rows"], cell["ground_truth_k"])
        if raw_truth.shape != stored_shape or stored_shape[1] < PARAMETERS["k"]:
            raise ValueError("Original GT shape differs from the explicit row/top-k declaration")
        truth = ground_truth_prefix(raw_truth[:, :PARAMETERS["k"]], job["N"], len(cell_queries),
                                    PARAMETERS["k"], stored_shape[0])
        if not predicates.matches(truth, attributes).all():
            raise ValueError(f"Original GT violates its predicate: {cell['name']}")
        counts = predicates.counts(attributes)
        if np.any(counts < PARAMETERS["k"]):
            raise ValueError("Fewer than k eligible source records")
        cell_info = {
            "request": cell, "matching_min": int(counts.min()), "matching_max": int(counts.max()),
            "actual_selectivity": float(counts.mean() / job["N"]),
            "matching_counts_sha256": array_digest(counts),
            "used_gt_sha256": array_digest(truth), "original_gt_predicate_validated": True,
            "query_count": len(cell_queries), "query_payload_sha256": array_digest(cell_queries),
        }
        cells.append((cell, predicates, truth, cell_info))
    preparation = {
        "schema": SCHEMA, "job": job, "source": sources, "native": native_info,
        "environment": environment, "limits": limits, "inputs": inputs,
        "vector_preparation": vector_meta, "query_preparation": query_meta,
        "cells": [item[3] for item in cells],
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if not resume or read_json(manifest_path) != preparation:
            raise RuntimeError("Static job resume manifest mismatch")
    else:
        immutable_json(manifest_path, preparation)
    build_path = output / "build.json"
    completed_cells = list((output / "cells").glob("*/result.json")) if (output / "cells").exists() else []
    if completed_cells and not build_path.exists():
        raise RuntimeError("Completed query cells cannot be reused without their frozen build identity")
    if build_path.exists():
        if not resume:
            raise RuntimeError("Existing build requires explicit resume")
        build = read_json(build_path)
        checkpoint = file_info(build["checkpoint"]["path"], build["checkpoint"])
        progress("loading_current_checkpoint")
        index = native.Index(space=job["metric"], dim=job["dimension"])
        index.set_num_threads(32)
        index.load_index(checkpoint["path"])
    elif job.get("checkpoint") is not None:
        checkpoint = file_info(job["checkpoint"]["path"], job["checkpoint"])
        header = read_attribute_index_header(checkpoint["path"])
        progress("loading_audited_current_checkpoint")
        index = native.Index(space=job["metric"], dim=job["dimension"])
        index.set_num_threads(32)
        index.load_index(checkpoint["path"])
        build = {"reused": True, "checkpoint": checkpoint, "construction_s": None}
    else:
        progress("building_current_index")
        directory = output / "build-attempts" / uuid.uuid4().hex
        directory.mkdir(parents=True)
        target = directory / "checkpoint.index"
        wrapper = load_builder().HashANN()
        params = {
            "M": 40, "ef_construction": 300, "metric": job["metric"],
            "dim": job["dimension"], "N": job["N"], "max_elements": job["N"],
            "ft_bits": 128, "edge_level_ft": True, "threads": 32,
            "clustering_cache_root": str(output.parent / f"{job['dataset']}-level-plan"),
        }
        wrapper.init_params(params)
        started = time.perf_counter()
        index = wrapper.build_index(params, vectors, attributes.records, list(attributes.kinds),
                                    str(target), 32, "HNSW")
        elapsed = time.perf_counter() - started
        checkpoint = file_info(target)
        build = {
            "reused": False, "checkpoint": checkpoint,
            "construction_wall_s": wrapper.construction_duration_s,
            "construction_s": wrapper.construction_duration_s - wrapper.clustering_identity_s,
            "provenance_vector_hash_s": wrapper.clustering_identity_s,
            "save_s": wrapper.save_duration_s, "build_and_save_s": elapsed,
            "clustering": wrapper.clustering_metadata,
            "clustering_cache_reused": wrapper.clustering_cache_reused,
            "clustering_cache": file_info(wrapper.clustering_cache_path),
            "insertion_policy": "one native add_items call over all source rows in original order",
            "native_random_seed": 100, "build_threads": 32,
        }
    progress("verifying_saved_index")
    header = read_attribute_index_header(checkpoint["path"])
    if (header["count"] != job["N"] or header["max_elements"] != job["N"]
            or header["M"] != 40 or header["ef_construction"] != 300 or header["ft_bits"] != 128
            or header["maxM0"] != 80 or header["minM0"] != 16):
        raise ValueError("Saved index differs from the requested full-dataset static configuration")
    records = verify_records(checkpoint["path"], header, attributes, vectors)
    tail = verify_index_tail(checkpoint["path"], header)
    if not build_path.exists():
        build.update(header=header, records=records, tail=tail)
        immutable_json(build_path, build)
    index.generateAttrIndexes()
    index.set_thresholds(*PARAMETERS["thresholds"])
    index.set_num_threads(1)
    index.set_ef_top(1)
    index.set_ft_flag(True)
    index.set_ft_routing_flag(True)
    index.set_ft_routing_min_deg(16)
    index.set_ft_routing_backfill_tail(False)
    native_state(index, job["N"], 0)
    require_local(numa_snapshot(), 0)
    results = []
    reused_cells = 0
    manifest_identity = file_info(manifest_path)
    for cell, predicates, truth, cell_info in cells:
        cell_queries = queries[:cell["query_count"]]
        frozen = {"checkpoint": checkpoint, "native": native_info, "manifest": manifest_identity,
                  "query_payload_sha256": cell_info["query_payload_sha256"]}
        directory = output / "cells" / cell["name"]
        directory.mkdir(parents=True, exist_ok=True)
        result_path = directory / "result.json"
        if result_path.exists():
            if not resume:
                raise RuntimeError("Completed query cell requires explicit resume")
            result = read_json(result_path)
            if result.get("input") != cell_info or result.get("measurement_identity") != frozen:
                raise RuntimeError("Completed static query cell changed")
            for info in result["artifacts"].values():
                file_info(info["path"], info)
            results.append(result)
            reused_cells += 1
            continue
        attempt = directory / uuid.uuid4().hex
        attempt.mkdir()
        method = index.hybrid_knn_query_dnf if predicates.dnf else index.hybrid_knn_query
        query = lambda: method(cell_queries, predicates.raw, k=PARAMETERS["k"], num_threads=1)
        validate = lambda result: validate_result(
            result, truth, vectors, cell_queries, attributes, predicates, job["metric"], PARAMETERS["k"])
        measured = measure_cell(
            index, query, validate, len(cell_queries), cell["targets"], job["max_ef"], attempt,
            environment, limits, lambda phase, **detail: progress(phase, cell=cell["name"], **detail))
        result = {
            "schema": SCHEMA, "input": cell_info, "measurement_identity": frozen,
            "completed_at": timestamp(), **measured,
            "artifacts": {name: file_info(attempt / name) for name in (
                "calibration.json", "calibration-arrays.npz", "timing.json")},
        }
        immutable_json(result_path, result)
        results.append(result)
        atomic_json(output / "results.json", {"status": "running", "cells": results})
    native_state(index, job["N"], 0)
    require_unchanged(checkpoint)
    for info in [native_info, *inputs.values(), *sources.values()]:
        require_unchanged(info)
    for cell in job["cells"]:
        require_unchanged(cell["predicate"])
        require_unchanged(cell["ground_truth"])
    atomic_json(output / "results.json", {
        "schema": SCHEMA, "status": "complete", "completed_at": timestamp(), "cells": results,
        "reused_cells": reused_cells, "newly_measured_cells": len(results) - reused_cells,
        "build_only": job.get("build_only", False)})
    atomic_json(output / "status.json", {"status": "complete", "completed_at": timestamp()})


def suite(request, output, *, resume=False):
    manifest_path = output / "suite-manifest.json"
    if manifest_path.exists():
        if not resume:
            raise RuntimeError("Existing static suite requires explicit resume")
        manifest = read_json(manifest_path)
        if manifest["request"] != request or manifest["source"] != source_identity():
            raise RuntimeError("Static suite request/source changed")
    else:
        if resume:
            raise RuntimeError("Cannot resume a suite without its frozen manifest")
        manifest = {"schema": SCHEMA, "request": request, "source": source_identity(),
                    "started_at": timestamp()}
        immutable_json(manifest_path, manifest)
        snapshot = output / "source"
        for name, info in manifest["source"].items():
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with Path(info["path"]).open("rb") as source, target.open("xb") as destination:
                destination.write(source.read())
            if file_info(target)["sha256"] != info["sha256"]:
                raise RuntimeError("Static executed-source snapshot mismatch")
    for job in request["jobs"]:
        job_output = output / "jobs" / job["name"]
        job_output.mkdir(parents=True, exist_ok=True)
        result_path = job_output / "results.json"
        if result_path.exists() and read_json(result_path).get("status") == "complete":
            if not resume:
                raise RuntimeError("Completed static job unexpectedly predates this suite")
        atomic_json(output / "status.json", {
            "status": "running", "job": job["name"], "updated_at": timestamp()})
        command = [
            sys.executable, "-m", "exp_benchmark.static_paper", "worker",
            "--manifest", str(manifest_path), "--job", job["name"], "--output", str(job_output),
        ]
        if resume:
            command.append("--resume")
        log_path = job_output / f"worker-{uuid.uuid4().hex}.log"
        construction_started = None
        timed_out = False
        with log_path.open("xb") as log:
            process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                status_path = job_output / "status.json"
                phase = read_json(status_path).get("phase") if status_path.exists() else None
                if phase == "building_current_index":
                    if construction_started is None:
                        construction_started = time.monotonic()
                    if time.monotonic() - construction_started > 20 * 3600:
                        timed_out = True
                        process.terminate()
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        break
                time.sleep(5)
            returncode = process.wait()
        if returncode or timed_out:
            atomic_json(output / "status.json", {
                "status": "failed", "job": job["name"], "returncode": returncode,
                "construction_timeout": timed_out, "log": str(log_path), "updated_at": timestamp()})
            raise RuntimeError(f"Static job failed: {job['name']}; see {log_path}")
    atomic_json(output / "status.json", {"status": "complete", "completed_at": timestamp()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    whole = commands.add_parser("suite")
    whole.add_argument("--request", required=True)
    worker = commands.add_parser("worker")
    worker.add_argument("--manifest", required=True)
    worker.add_argument("--job", required=True)
    for command in (whole, worker):
        command.add_argument("--output", required=True)
        command.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=args.resume or args.command == "worker")
    with OutputLock(output):
        try:
            if args.command == "suite":
                suite(read_json(args.request), output, resume=args.resume)
            else:
                manifest = read_json(args.manifest)
                jobs = [job for job in manifest["request"]["jobs"] if job["name"] == args.job]
                if len(jobs) != 1:
                    raise ValueError("Worker job does not identify exactly one frozen request")
                run_job(jobs[0], output, manifest["request"]["native_dir"],
                        manifest["request"]["native_sha256"], manifest["source"], resume=args.resume)
        except (OSError, ValueError, RuntimeError) as error:
            failure = {
                "status": "failed", "error_type": type(error).__name__, "error": str(error),
                "at": timestamp()}
            atomic_json(output / "failure.json", failure)
            atomic_json(output / "status.json", failure)
            raise


if __name__ == "__main__":
    main()
