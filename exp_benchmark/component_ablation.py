"""Paired C1/C2/C3 ablations on the original YouTube mixed-AND selectivity grid."""

import argparse
import csv
import gc
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
    DEFAULT_LIMITS, OutputLock, atomic_json, bootstrap_manifest, choose_core, environment_config,
    file_info, immutable_json, import_native, numa_snapshot, require_local,
    require_unchanged, siblings, timed_query, timestamp, write_npz,
)
from .index_metadata import read_layout
from .static_mixed_inputs import HIGH, LOW, and_path
from .static_paper import (
    Attributes, Predicates, load_builder, read_json, validate_ids,
    validate_result, verify_records,
)


REPO = Path(__file__).resolve().parents[1]
SCHEMA = "ema-youtube-components-v1"
VARIANTS = {
    "baseline": {"cht": False, "marker": False, "recovery": False},
    "c1": {"cht": True, "marker": False, "recovery": False},
    "c1_c2": {"cht": True, "marker": True, "recovery": False},
    "full": {"cht": True, "marker": True, "recovery": True},
    "without_c1": {"cht": False, "marker": True, "recovery": True},
}
CONTRASTS = (
    ("C1_without_Marker", "c1", "baseline"),
    ("C2_without_recovery", "c1_c2", "c1"),
    ("C3_with_Marker", "full", "c1_c2"),
    ("C1_in_full_method", "full", "without_c1"),
    ("full_vs_baseline", "full", "baseline"),
)
PARAMETERS = {
    "M": 40, "ef_construction": 300, "ft_bits": 128, "edge_level_ft": True,
    "routing_min_deg": 16, "ef_top": 1, "k": 10, "target_recall": 0.95,
    "max_ef": 4096, "native_seed": 100, "build_threads": 32,
    "query_threads": 1, "normalization": "none", "metric": "ip",
}
SOURCE_PATHS = (
    "exp_benchmark/component_ablation.py", "exp_benchmark/static_paper.py",
    "exp_benchmark/static_mixed_inputs.py", "exp_benchmark/static_inputs.py",
    "exp_benchmark/figure4a.py", "exp_benchmark/figure4a_work_control.py",
    "exp_benchmark/index_metadata.py", "exp_benchmark/aggregate.py",
    "exp_benchmark/dynamic/runtime.py", "exp_benchmark/dynamic/gt.py",
    "exp_benchmark/dynamic/protocol.py", "tests/hashann.py",
    "exp_benchmark/ablation/build_component_native.sh",
    "exp_benchmark/ablation/component_native.patch",
)
COUNTERS = (
    "base_expanded", "empty_marker_rows", "backfill_edges", "backfill_visits",
    "base_distance_evaluations", "predicate_checked", "ft_false_positives",
    "total_neighbors", "ft_passed_total",
)


class TimingIncomplete(RuntimeError):
    pass


def progress(root, phase, **details):
    value = {"at": timestamp(), "phase": phase, **details}
    atomic_json(root / "status.json", value)
    print(json.dumps(value), flush=True)


def prepare(data_root, output, native_provenance, pilot_size=0):
    root, output = Path(data_root).resolve(strict=True), Path(output).resolve()
    if pilot_size and not 1024 <= pilot_size < 1_000_000:
        raise ValueError("Pilot size must be in [1024, 1000000)")
    output.mkdir(parents=True)
    inputs = {
        key: file_info(root / name) for key, name in (
            ("vectors", "rgb.fvecs"), ("queries", "rgb_query.fvecs"),
            ("attributes", "label/arbi_0_1_random/attr_arbi_0_1_random.json"))
    }
    if inputs["vectors"]["bytes"] != 1_000_000 * 1025 * 4:
        raise ValueError("Expected the complete original YouTube1M 1024-dimensional file")
    cells = []
    labels = root / "label/arbi_0_1_random"
    for nominal, selector in (*LOW, *HIGH[1:]):
        predicate = and_path(labels, selector)
        ground_truth = predicate.with_name(predicate.name.replace("predicate_", "gt_", 1))
        cells.append({
            "name": f"T{nominal}", "nominal_selectivity": nominal / 100,
            "predicate": file_info(predicate), "ground_truth": file_info(ground_truth),
            "selector": selector,
        })
    provenance = file_info(Path(native_provenance).resolve(strict=True))
    native = read_json(provenance["path"])
    if native["schema"] != "ema-component-native-v1":
        raise ValueError("Expected the isolated component-ablation native pair")
    request = {
        "schema": SCHEMA, "created_at": timestamp(), "dataset": "youtube1m",
        "profile": "pilot" if pilot_size else "full", "N": pilot_size or 1_000_000,
        "dimension": 1024, "query_count": 32 if pilot_size else 1000,
        "inputs": inputs, "cells": cells, "parameters": PARAMETERS,
        "variants": VARIANTS, "limits": DEFAULT_LIMITS, "native_provenance": provenance,
        "native": native, "sources": {name: file_info(REPO / name) for name in SOURCE_PATHS},
        "c1_isolation": "Same source, vectors, attributes, Codebook, level plan, insertion IDs and seed; only CHT admission changes",
        "parallel_build_caveat": "One graph per construction policy; parallel interleavings are not deterministic",
        "baseline_semantics": "Same filtered-result traversal and edge-Marker layout, not an upstream post-filter binary",
        "pilot_ground_truth": "Recomputed on the prefix; never used for full-run results" if pilot_size else None,
    }
    snapshot = output / "source"
    for name in SOURCE_PATHS:
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((REPO / name).read_bytes())
    immutable_json(output / "request.json", request)
    print(output / "request.json", flush=True)


def load_request(path):
    path = Path(path).resolve(strict=True)
    request = read_json(path)
    if (request["schema"] != SCHEMA or request["parameters"] != PARAMETERS
            or request["variants"] != VARIANTS or request["limits"] != DEFAULT_LIMITS):
        raise ValueError("This runner does not silently change an existing experiment protocol")
    identities = [*request["inputs"].values(), request["native_provenance"]]
    identities.extend(info for cell in request["cells"]
                      for info in (cell["predicate"], cell["ground_truth"]))
    for info in identities:
        require_unchanged(info)
    for name, expected in request["sources"].items():
        if file_info(REPO / name)["sha256"] != expected["sha256"]:
            raise RuntimeError(f"Source changed since preparation: {name}")
    for info in request["native"]["modules"].values():
        if file_info(info["path"])["sha256"] != info["sha256"]:
            raise RuntimeError("A selected native binary changed")
    return request, path.parent


def select_native(request, diagnostic=False):
    flavor = "diagnostic" if diagnostic else "timing"
    info = request["native"]["modules"][flavor]
    native, _ = import_native(Path(info["path"]).parent, info["sha256"])
    probe = native.Index(space="ip", dim=request["dimension"])
    if bool(probe.get_ablation_stats()["diagnostic_build"]) != diagnostic:
        raise RuntimeError("Timing and diagnostic native flavors were interchanged")
    return native


def exact_prefix_gt(vectors, query, predicate, attributes, k):
    mask = ((attributes.numeric >= predicate[0][0])
            & (attributes.numeric <= predicate[0][1]))
    for label in predicate[1]:
        mask &= (attributes.categorical & (1 << label)) != 0
    ids = np.flatnonzero(mask)
    if len(ids) < k:
        raise ValueError("The pilot predicate has fewer than k eligible points; use a larger pilot")
    scores = np.empty(len(ids), dtype=np.float64)
    for start in range(0, len(ids), 32768):
        stop = min(start + 32768, len(ids))
        scores[start:stop] = np.einsum(
            "ij,j->i", vectors[ids[start:stop]], query, dtype=np.float64)
    best = np.argpartition(-scores, k - 1)[:k]
    best = best[np.lexsort((ids[best], -scores[best]))]
    return ids[best], scores[best[-1]]


def load_data(request, root, audit=False):
    n, dimension, nq = request["N"], request["dimension"], request["query_count"]
    vectors = np.ascontiguousarray(read_fvecs(
        request["inputs"]["vectors"]["path"], n, dimension, integer_sift=False))
    queries = np.ascontiguousarray(read_fvecs(
        request["inputs"]["queries"]["path"], nq, dimension, integer_sift=False))
    if not np.isfinite(vectors).all() or not np.isfinite(queries).all():
        raise ValueError("Nonfinite original embeddings")
    mapping_records = read_json(request["inputs"]["attributes"]["path"])
    if len(mapping_records) != 1_000_000:
        raise ValueError("Expected all original attribute records")
    attributes = Attributes.from_records(mapping_records[:n], [0, 1], n)
    cells, audits = {}, []
    for cell in request["cells"]:
        raw = read_json(cell["predicate"]["path"])[:nq]
        predicates = Predicates.parse(raw, [0, 1], nq)
        counts = predicates.counts(attributes)
        if np.any(counts < PARAMETERS["k"]):
            raise ValueError(f"{cell['name']} has fewer than k eligible points")
        truth = np.asarray(read_json(cell["ground_truth"]["path"]))
        if truth.ndim != 2 or truth.shape[0] < nq or truth.shape[1] < PARAMETERS["k"]:
            raise ValueError("Original GT lacks the requested query prefix/top-k columns")
        if request["profile"] == "pilot":
            truth = np.asarray([exact_prefix_gt(
                vectors, query, predicate, attributes, PARAMETERS["k"])[0]
                for query, predicate in zip(queries, raw)])
        else:
            truth = truth[:nq, :PARAMETERS["k"]]
        truth = validate_ids(truth, n, nq, PARAMETERS["k"])
        if not predicates.matches(truth, attributes).all():
            raise ValueError(f"{cell['name']} ground truth violates the saved predicate")
        checks = []
        if audit and request["profile"] == "full":
            for q in (0, nq // 2, nq - 1):
                exact, cutoff = exact_prefix_gt(
                    vectors, queries[q], raw[q], attributes, PARAMETERS["k"])
                original_scores = np.einsum(
                    "ij,j->i", vectors[truth[q]], queries[q], dtype=np.float64)
                tolerance = 2e-5 * max(1, abs(cutoff))
                if float(original_scores.min()) < cutoff - tolerance:
                    raise RuntimeError(f"{cell['name']} original GT disagrees with raw-IP exact search at query {q}")
                checks.append({"query": q, "exact_ids": exact.tolist(), "cutoff": cutoff,
                               "original_worst_score": float(original_scores.min())})
        cells[cell["name"]] = {"spec": cell, "predicates": predicates, "truth": truth}
        audits.append({
            "cell": cell["name"], "nominal_selectivity": cell["nominal_selectivity"],
            "actual_selectivity": float(counts.mean() / n),
            "actual_selectivity_min": float(counts.min() / n),
            "actual_selectivity_max": float(counts.max() / n),
            "eligible_counts_sha256": array_digest(counts),
            "ground_truth_sha256": array_digest(truth), "exact_gt_spot_checks": checks,
        })
    if audit:
        bootstrap_manifest(root / "input-audit.json", {
            "profile": request["profile"], "cells": audits,
            "query_payload_sha256": array_digest(queries),
            "vector_payload_sha256": array_digest(vectors), "normalization": "none",
        })
    return vectors, queries, attributes, mapping_records, cells


def build_graphs(request, root):
    native = select_native(request)
    progress(root, "input_audit", status="running")
    vectors, queries, attributes, mapping_records, cells = load_data(request, root, audit=True)
    environment = environment_config(0, PARAMETERS["build_threads"], requested_cores=[8, 16, 24])
    builder = load_builder().HashANN()
    builder.init_params({"N": request["N"], "dim": request["dimension"],
                         "metric": "ip", "threads": PARAMETERS["build_threads"]})
    level_root = root / "levels"
    level_root.mkdir(exist_ok=True)
    builder.save_root = str(level_root / "shared")
    progress(root, "shared_level_plan", status="running")
    top_size, levels, _, _ = builder.clustering(vectors)
    level_identity = file_info(builder.clustering_cache_path)
    max_cate = max(value for record in mapping_records for value in record[1]) + 1
    for cht in (False, True):
        name = "cht_on" if cht else "cht_off"
        directory = root / "graphs" / name
        directory.mkdir(parents=True, exist_ok=True)
        record_path = directory / "build.json"
        if record_path.exists():
            record = read_json(record_path)
            if record["level_plan"]["sha256"] != level_identity["sha256"] or record["cht"] != cht:
                raise RuntimeError("Existing graph does not belong to this paired build")
            file_info(record["checkpoint"]["path"], record["checkpoint"])
            continue
        progress(root, "building_graph", status="running", graph=name, threads=PARAMETERS["build_threads"])
        index = native.Index(space="ip", dim=request["dimension"])
        index.init_index(
            max_elements=request["N"], top_elements=top_size, M=PARAMETERS["M"],
            ef_construction=PARAMETERS["ef_construction"], ft_bits=PARAMETERS["ft_bits"],
            attr_type=[0, 1], max_cate_size=max_cate, edge_level_ft=True,
            random_seed=PARAMETERS["native_seed"])
        index.set_ablation_cht(cht)
        index.initAttrMapping(mapping_records)
        started = time.perf_counter()
        index.add_items(vectors, attributes.records, ids=np.arange(request["N"], dtype=np.uint64),
                        levels=levels, num_threads=PARAMETERS["build_threads"])
        add_s = time.perf_counter() - started
        pending = directory / f"index-{uuid.uuid4().hex}.pending"
        started = time.perf_counter()
        index.save_index(str(pending))
        save_s = time.perf_counter() - started
        header = read_layout(pending)
        if header["format"] != 10:
            raise RuntimeError("The complete-Codebook paired graph must use current edge format10")
        verified = verify_records(pending, header, attributes, vectors)
        checkpoint = directory / "index.bin"
        if checkpoint.exists():
            raise RuntimeError("An uncommitted graph exists; preserve it before restarting")
        pending.rename(checkpoint)
        degrees = np.asarray(index.get_degrees())
        immutable_json(record_path, {
            "cht": cht, "add_wall_s": add_s, "save_wall_s": save_s,
            "checkpoint": file_info(checkpoint), "header": header, "verified": verified,
            "level_plan": level_identity, "levels_sha256": array_digest(levels),
            "degree_mean": float(degrees.mean()), "degree_min": int(degrees.min()),
            "degree_max": int(degrees.max()), "environment": environment,
            "native_sha256": request["native"]["modules"]["timing"]["sha256"],
        })
        del index
        gc.collect()
    progress(root, "graphs_complete", status="complete")


def load_graphs(native, request, root):
    indexes = {}
    for cht in (False, True):
        name = "cht_on" if cht else "cht_off"
        record = read_json(root / "graphs" / name / "build.json")
        if record["cht"] != cht or record["native_sha256"] != request["native"]["modules"]["timing"]["sha256"]:
            raise RuntimeError("Paired graph construction/native provenance differs")
        file_info(record["checkpoint"]["path"], record["checkpoint"])
        index = native.Index(space="ip", dim=request["dimension"])
        index.load_index(record["checkpoint"]["path"])
        index.generateAttrIndexes()
        index.set_num_threads(1)
        indexes[cht] = index
    return indexes


def configure(index, variant, ef):
    config = VARIANTS[variant]
    index.set_ef(ef)
    index.set_ef_top(PARAMETERS["ef_top"])
    index.set_ft_flag(config["marker"])
    index.set_ft_routing_flag(config["recovery"])
    index.set_ft_routing_min_deg(PARAMETERS["routing_min_deg"])
    index.set_ft_routing_backfill_tail(False)


def calibration_plan(measure, k=10, maximum=4096, target=0.95):
    points, coarse = {}, []
    def sample(ef):
        if ef not in points:
            points[ef] = {"ef": ef, **measure(ef)}
        return points[ef]
    grid = sorted({k, maximum, *(min(maximum, 20 * 2**i) for i in range(9))})
    crossing = None
    for ef in grid:
        point = sample(ef)
        coarse.append(ef)
        if crossing is not None:
            break
        if point.get("recall", -1) >= target:
            crossing = ef
    if crossing is not None:
        lower = max((ef for ef, p in points.items()
                     if ef < crossing and "recall" in p and p["recall"] < target), default=None)
        upper = crossing
        while lower is not None and upper - lower > 1:
            mid = (lower + upper) // 2
            point = sample(mid)
            if "recall" not in point:
                break
            if point["recall"] >= target:
                upper = mid
            else:
                lower = mid
    valid = sorted((p for p in points.values() if "recall" in p), key=lambda p: p["ef"])
    passing = [p for p in valid if p["recall"] >= target]
    selected = passing[0]["ef"] if passing else None
    preceding = max((p["ef"] for p in valid if selected is not None and p["ef"] < selected
                     and p["recall"] < target), default=None)
    timing = {ef for ef in (selected, preceding) if ef is not None}
    for recall in (0.80, 0.90):
        eligible = [p for p in valid if p["recall"] >= recall]
        if eligible:
            timing.add(eligible[0]["ef"])
    good_coarse = [ef for ef in coarse if "recall" in points[ef]]
    if good_coarse:
        timing.update((good_coarse[0], good_coarse[-1]))
    return {
        "points": sorted(points.values(), key=lambda p: p["ef"]),
        "timing_efs": sorted(timing), "selected_ef": selected, "lower_ef": preceding,
        "status": ("unreached" if selected is None else
                   "minimum_above" if preceding is None and points[selected]["recall"] > target else
                   "reached"),
        "crossing_policy": "First sampled crossing, refined within its bracket; no global monotonicity assumption",
    }


def target_values(plan, qps, target=0.95):
    upper, lower = plan["selected_ef"], plan["lower_ef"]
    if upper is None:
        return {"qps95": None, "observed_qps": None, "observed_recall": None}
    points = {p["ef"]: p for p in plan["points"]}
    recall = points[upper]["recall"]
    bracket = [(ef, points[ef]["recall"], qps[ef]) for ef in (lower, upper) if ef is not None]
    strict = None if lower is None and recall > target else qps_at_recall(bracket, target)
    return {"qps95": strict, "observed_qps": qps[upper], "observed_recall": recall}


def paired_timing(indexes, queries, predicates, plans, directory, limits, core, loads):
    jobs = [(variant, ef) for variant, plan in plans.items() for ef in plan["timing_efs"]]
    attempt = directory / f"timing-{uuid.uuid4().hex}.json"
    records, accepted = [], []
    for round_id in range(limits["max_rounds_per_core"]):
        order = np.random.default_rng(20261101 + round_id).permutation(len(jobs))
        round_records = []
        for position in order:
            variant, ef = jobs[position]
            index = indexes[VARIANTS[variant]["cht"]]
            point = next(p for p in plans[variant]["points"] if p["ef"] == ef)
            configure(index, variant, ef)
            arguments = {"k": PARAMETERS["k"], "num_threads": 1}
            index.hybrid_knn_query(queries[:3], predicates.raw[:3], **arguments)
            repeats = max(1, min(100, math.ceil(0.25 / point["calibration_wall_s"])))
            def call():
                for _ in range(repeats):
                    result = index.hybrid_knn_query(queries, predicates.raw, **arguments)
                return result
            result, sample = timed_query(call, core, siblings(core), limits)
            if (array_digest(result[0]) != point["labels_sha256"]
                    or array_digest(result[1]) != point["distances_sha256"]):
                raise RuntimeError("Read-only timed output differs from calibration")
            round_records.append({
                **sample, "variant": variant, "ef": ef, "round": round_id,
                "batch_repeats": repeats, "qps": len(queries) * repeats / sample["wall_s"],
                "labels_sha256": point["labels_sha256"], "distances_sha256": point["distances_sha256"],
            })
        clean = all(row["clean"] for row in round_records)
        for row in round_records:
            row["accepted_complete_round"] = clean
        records.extend(round_records)
        if clean:
            accepted.append(round_id)
        atomic_json(attempt, {"core": core, "core_selection_loads": loads,
                              "accepted_rounds": accepted, "samples": records})
        print(f"{directory.name}: timing round {round_id}, accepted={clean}, complete={len(accepted)}", flush=True)
        if len(accepted) >= limits["clean_rounds"]:
            break
    return attempt, accepted


def measurement(request, root):
    native = select_native(request)
    vectors, queries, attributes, _, cells = load_data(request, root)
    indexes = load_graphs(native, request, root)
    environment_config(0, PARAMETERS["build_threads"], requested_cores=[8, 16, 24])
    for name, cell in cells.items():
        directory = root / "measurements" / name
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "complete.json").exists():
            continue
        candidates = [8, 16, 24]
        core, loads = choose_core(candidates)
        os.sched_setaffinity(0, {core})
        require_local(numa_snapshot(), 0)
        progress(root, "calibrating", status="running", cell=name, core=core)
        plans = {}
        for variant, config in VARIANTS.items():
            index = indexes[config["cht"]]
            raw = directory / variant
            raw.mkdir(exist_ok=True)
            plan_path = raw / "calibration.json"
            def measure(ef):
                cached_path = raw / f"ef-{ef}.json"
                if cached_path.exists():
                    saved = read_json(cached_path)
                    file_info(saved["raw"]["path"], saved["raw"])
                    return saved
                configure(index, variant, ef)
                start = time.perf_counter()
                try:
                    result = index.hybrid_knn_query(
                        queries, cell["predicates"].raw, k=PARAMETERS["k"], num_threads=1)
                except RuntimeError as error:
                    if "Cannot return the results in a contiguous 2D array" not in str(error):
                        raise
                    return {"error": "fewer_than_k_results", "message": str(error)}
                wall = time.perf_counter() - start
                validated = validate_result(
                    result, cell["truth"], vectors, queries, attributes, cell["predicates"],
                    "ip", PARAMETERS["k"])
                path = raw / f"ef-{ef}-{uuid.uuid4().hex}.npz"
                write_npz(path, labels=result[0], distances=result[1])
                value = {**validated, "calibration_wall_s": wall, "raw": file_info(path)}
                immutable_json(cached_path, value)
                print(f"{name} {variant} ef={ef} recall={validated['recall']:.6f}", flush=True)
                return value
            if plan_path.exists():
                plans[variant] = read_json(plan_path)
                for point in plans[variant]["points"]:
                    if "raw" in point:
                        file_info(point["raw"]["path"], point["raw"])
            else:
                plans[variant] = calibration_plan(measure)
                immutable_json(plan_path, plans[variant])
        for core_attempt in range(min(len(candidates), request["limits"]["max_cores"])):
            progress(root, "paired_timing", status="running", cell=name, core=core)
            attempt, accepted = paired_timing(
                indexes, queries, cell["predicates"], plans, directory,
                request["limits"], core, loads)
            candidates.remove(core)
            if (len(accepted) >= request["limits"]["clean_rounds"] or not candidates
                    or core_attempt + 1 >= request["limits"]["max_cores"]):
                break
            core, loads = choose_core(candidates)
            os.sched_setaffinity(0, {core})
        if len(accepted) < request["limits"]["clean_rounds"]:
            progress(root, "timing_needs_retry", status="needs_timing", cell=name, attempt=str(attempt))
            raise TimingIncomplete("Insufficient complete CPU-clean paired rounds; existing calibration/graphs are retained")
        immutable_json(directory / "complete.json", {
            "cell": name, "plans": plans, "timing": file_info(attempt),
            "accepted_rounds": accepted, "core": core,
        })
        os.sched_setaffinity(0, set(range(32)))
    progress(root, "measurement_complete", status="complete")


def diagnostics(request, root):
    native = select_native(request, diagnostic=True)
    vectors, queries, attributes, _, cells = load_data(request, root)
    indexes = load_graphs(native, request, root)
    environment_config(0, PARAMETERS["build_threads"], requested_cores=[8, 16, 24])
    for name, cell in cells.items():
        directory = root / "measurements" / name
        if (directory / "diagnostics.json").exists():
            continue
        complete = read_json(directory / "complete.json")
        os.sched_setaffinity(0, {complete["core"]})
        rows = []
        for variant, plan in complete["plans"].items():
            index = indexes[VARIANTS[variant]["cht"]]
            for ef in plan["timing_efs"]:
                point = next(p for p in plan["points"] if p["ef"] == ef)
                configure(index, variant, ef)
                index.reset_ft_stats()
                result = index.hybrid_knn_query(
                    queries, cell["predicates"].raw, k=PARAMETERS["k"], num_threads=1)
                if (array_digest(result[0]) != point["labels_sha256"]
                        or array_digest(result[1]) != point["distances_sha256"]):
                    raise RuntimeError("Diagnostic build changed the timed query output")
                stats = index.get_ablation_stats()
                if not stats["diagnostic_build"] or any(stats[key] < 0 for key in COUNTERS):
                    raise RuntimeError("Invalid diagnostic counters")
                rows.append({
                    "variant": variant, "ef": ef, "recall": point["recall"],
                    "query_count": len(queries), "counters": {key: stats[key] for key in COUNTERS},
                    "per_query": {key: stats[key] / len(queries) for key in COUNTERS},
                    "labels_sha256": point["labels_sha256"],
                })
        immutable_json(directory / "diagnostics.json", {
            "native_sha256": request["native"]["modules"]["diagnostic"]["sha256"], "rows": rows})
        os.sched_setaffinity(0, set(range(32)))
    progress(root, "diagnostics_complete", status="complete")


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(request, root):
    audits = {cell["cell"]: cell for cell in read_json(root / "input-audit.json")["cells"]}
    summary, curves, contrasts = [], [], []
    for cell in request["cells"]:
        name = cell["name"]
        directory = root / "measurements" / name
        complete = read_json(directory / "complete.json")
        file_info(complete["timing"]["path"], complete["timing"])
        samples = read_json(complete["timing"]["path"])["samples"]
        accepted = [row for row in samples if row["accepted_complete_round"]]
        diag = read_json(directory / "diagnostics.json")["rows"]
        values_by_variant = {}
        for variant, plan in complete["plans"].items():
            qps = {ef: statistics.median(row["qps"] for row in accepted
                                        if row["variant"] == variant and row["ef"] == ef)
                   for ef in plan["timing_efs"]}
            values = target_values(plan, qps)
            work = next((row["per_query"] for row in diag
                         if row["variant"] == variant and row["ef"] == plan["selected_ef"]), {})
            summary.append({
                "dataset": request["dataset"], "profile": request["profile"],
                "cell": name, "nominal_selectivity": cell["nominal_selectivity"],
                "actual_selectivity": audits[name]["actual_selectivity"], "variant": variant,
                "status": plan["status"], "selected_ef": plan["selected_ef"],
                **values, **{key: work.get(key) for key in COUNTERS},
            })
            rounds = {}
            for round_id in complete["accepted_rounds"]:
                round_qps = {row["ef"]: row["qps"] for row in accepted
                             if row["variant"] == variant and row["round"] == round_id}
                rounds[round_id] = target_values(plan, round_qps)
            values_by_variant[variant] = rounds
            for point in plan["points"]:
                if point["ef"] in qps:
                    curves.append({
                        "cell": name, "nominal_selectivity": cell["nominal_selectivity"],
                        "variant": variant, "ef": point["ef"], "recall": point["recall"],
                        "qps": qps[point["ef"]],
                    })
        for component, numerator, denominator in CONTRASTS:
            ratios, threshold_ratios = [], []
            for round_id in complete["accepted_rounds"]:
                a = values_by_variant[numerator][round_id]
                b = values_by_variant[denominator][round_id]
                if a["qps95"] is not None and b["qps95"] is not None:
                    ratios.append(a["qps95"] / b["qps95"])
                if a["observed_qps"] is not None and b["observed_qps"] is not None:
                    threshold_ratios.append(a["observed_qps"] / b["observed_qps"])
            contrasts.append({
                "cell": name, "nominal_selectivity": cell["nominal_selectivity"],
                "component": component, "numerator": numerator, "denominator": denominator,
                "paired_qps95_speedup": statistics.median(ratios) if ratios else None,
                "paired_qps95_speedup_min": min(ratios) if ratios else None,
                "paired_qps95_speedup_max": max(ratios) if ratios else None,
                "paired_at_least95_speedup": statistics.median(threshold_ratios) if threshold_ratios else None,
                "paired_rounds": len(complete["accepted_rounds"]),
            })
    write_csv(root / "summary.csv", summary)
    write_csv(root / "curves.csv", curves)
    write_csv(root / "contrasts.csv", contrasts)
    atomic_json(root / "results.json", {
        "schema": SCHEMA, "profile": request["profile"], "status": "complete",
        "summary": summary, "contrasts": contrasts,
        "notes": [
            "QPS95 is null when no bracket exists; minimum-above points retain their actual recall.",
            "At-least95 ratios compare the first sampled qualifying ef, not identical achieved recall.",
            "C1 is a paired-build observation, not a multi-seed construction stability claim.",
            "Counters are untimed diagnostic replays with identical outputs, not proof of a hardware bottleneck.",
            "Incremental effects depend on component order and are not additive.",
        ],
    })
    progress(root, "complete", status="complete", output=str(root))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("prepare")
    setup.add_argument("--data-root", required=True)
    setup.add_argument("--output", required=True)
    setup.add_argument("--native-provenance", required=True)
    setup.add_argument("--pilot-size", type=int, default=0)
    for name in ("build", "measure", "diagnose", "report", "run"):
        command = sub.add_parser(name)
        command.add_argument("--request", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.data_root, args.output, args.native_provenance, args.pilot_size)
        return
    request, root = load_request(args.request)
    if args.command == "run":
        for phase in ("build", "measure", "diagnose", "report"):
            with (root / f"{phase}-{uuid.uuid4().hex}.log").open("w") as stream:
                subprocess.run([sys.executable, "-B", "-m", "exp_benchmark.component_ablation",
                                phase, "--request", str(Path(args.request).resolve())],
                               cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, check=True)
        return
    with OutputLock(root):
        try:
            {"build": build_graphs, "measure": measurement,
             "diagnose": diagnostics, "report": report}[args.command](request, root)
        except TimingIncomplete:
            raise
        except Exception as error:
            progress(root, args.command, status="failed", error=str(error))
            raise


if __name__ == "__main__":
    main()
