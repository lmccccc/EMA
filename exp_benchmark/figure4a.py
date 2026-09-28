"""Read-only Figure 4 SIFT10M trials with existing predicates, GT and index."""

import argparse
import json
import os
from pathlib import Path
import statistics

import numpy as np

from .aggregate import qps_at_recall
from .dynamic.gt import array_digest, read_fvecs
from .dynamic.runtime import (
    DEFAULT_LIMITS, atomic_json, choose_core, environment_config, file_info,
    import_native, native_state, read_index_header, require_unchanged,
    timed_query, verify_index_records,
)

CELL_LABELS = {"T10": 7, "T20": 5, "T40": 4, "T60": 2, "T80": 1, "T100": 0}
LOW_CELL_LABELS = {"T1": 9, "T2": 8, "T3": 8, "T5": 7, "T7": 7, "T10": 7}
COMPOSITE_CELL_LABELS = {
    "T1": (19, 18), "T2": (18, 17), "T3": (17, 16),
    "T5": (16, 14), "T7": (14, 12), "T10": (9, 12),
}


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def and_predicates(value, count):
    if not isinstance(value, list) or len(value) != count:
        raise ValueError("Predicate count differs from the query count")
    bounds, masks = [], []
    for predicate in value:
        if (not isinstance(predicate, list) or len(predicate) != 2
                or not isinstance(predicate[0], list) or len(predicate[0]) != 2
                or any(type(x) is not int for x in predicate[0])
                or predicate[0][0] > predicate[0][1]
                or not isinstance(predicate[1], list) or not predicate[1]
                or any(type(x) is not int or x < 0 or x >= 32 for x in predicate[1])):
            raise ValueError("Expected legacy AND [[inclusive low, high], [labels]], not DNF")
        bounds.append(predicate[0])
        mask = 0
        for label in predicate[1]:
            mask |= 1 << label
        masks.append(mask)
    return np.asarray(bounds, dtype=np.int64), np.asarray(masks, dtype=np.uint32)


def composite_predicates(value, count):
    if not isinstance(value, list) or len(value) != count:
        raise ValueError("Composite predicate count differs from the query count")
    first, second = [], []
    for predicate in value:
        if (not isinstance(predicate, list) or len(predicate) != 2
                or not isinstance(predicate[1], list) or len(predicate[1]) != 2
                or predicate[1][0] != []):
            raise ValueError("Expected existing ((range AND labels) OR labels) DNF")
        first.append(predicate[0])
        second.append([[-(2**31), 2**31 - 1], predicate[1][1]])
    bounds, masks = and_predicates(first, count)
    _, tail_masks = and_predicates(second, count)
    return bounds, masks, tail_masks


def validate_ids(ids, n, count, k):
    ids = np.asarray(ids)
    if (ids.shape != (count, k) or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 0) or np.any(ids >= n)
            or np.any(np.diff(np.sort(ids, axis=1), axis=1) == 0)):
        raise ValueError("Invalid, missing, duplicate, or out-of-range result/GT labels")
    return ids.astype(np.int64, copy=False)


def ground_truth_prefix(value, n, count, k, stored_rows):
    truth = np.asarray(value)
    if stored_rows < count or truth.shape != (stored_rows, k):
        raise ValueError("GT dimensions differ from the audited file/query prefix")
    return validate_ids(truth[:count], n, count, k)


def matches(ids, numeric, categorical, bounds, masks, tail_masks=None):
    values = numeric[ids]
    result = ((values >= bounds[:, :1]) & (values <= bounds[:, 1:])
              & ((categorical[ids] & masks[:, None]) == masks[:, None]))
    if tail_masks is not None:
        result |= ((categorical[ids] & tail_masks[:, None]) == tail_masks[:, None])
    return result


def matching_counts(numeric, categorical, bounds, masks, tail_masks=None):
    if not len(masks) or not np.all(masks == masks[0]):
        raise ValueError("Selectivity counting requires fixed per-cell categorical conditions")
    eligible = (categorical & masks[0]) == masks[0]
    tail_count = 0
    if tail_masks is not None:
        if len(tail_masks) != len(masks) or not np.all(tail_masks == tail_masks[0]):
            raise ValueError("Selectivity counting requires a fixed second DNF branch")
        tail = (categorical & tail_masks[0]) == tail_masks[0]
        eligible &= ~tail
        tail_count = int(tail.sum())
    # Count disjoint branches so points satisfying both terms appear only once.
    values = np.sort(numeric[eligible])
    return (np.searchsorted(values, bounds[:, 1], side="right")
            - np.searchsorted(values, bounds[:, 0], side="left") + tail_count)


def record_attributes(path, header):
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    n, start, stride = header["count"], header["records_offset"], header["record_bytes"]
    numeric, categorical = np.empty(n, dtype=np.int32), np.empty(n, dtype=np.uint32)
    labels = np.ndarray((n,), dtype="<u8", buffer=raw,
                        offset=start + header["label_offset"], strides=(stride,))
    num_view = np.ndarray((n,), dtype="<i4", buffer=raw,
                          offset=start + header["attr_offset"], strides=(stride,))
    cat_view = np.ndarray((n,), dtype="<u4", buffer=raw,
                          offset=start + header["attr_offset"] + 4, strides=(stride,))
    try:
        for first in range(0, n, 100_000):
            last = min(first + 100_000, n)
            ids = labels[first:last].astype(np.int64)
            numeric[ids], categorical[ids] = num_view[first:last], cat_view[first:last]
    finally:
        raw._mmap.close()
    return numeric, categorical


def run(config, output):
    def progress(phase, **detail):
        atomic_json(output / "status.json", {"status": "running", "phase": phase, **detail})

    if (config["dataset"] != "sift10m" or config["N"] != 10_000_000
            or config["dimension"] != 128 or config["query_count"] != 1000
            or config["k"] != 10 or config["target_recall"] != 0.95):
        raise ValueError("This entrypoint requires the full SIFT10M Figure4 protocol")
    panel = config.get("figure_panel", "a")
    labels_by_panel = {"a": CELL_LABELS, "b": LOW_CELL_LABELS, "c": COMPOSITE_CELL_LABELS}
    if panel not in labels_by_panel:
        raise ValueError("Figure4 panel must be a, b, or c")
    cell_labels = labels_by_panel[panel]
    if (len(config["cells"]) != 6
            or {cell["name"] for cell in config["cells"]} != set(cell_labels)):
        raise ValueError("All six selectivity cells for the selected panel are required")
    for cell in config["cells"]:
        if cell["nominal_selectivity"] != int(cell["name"][1:]) / 100:
            raise ValueError("Cell name and nominal selectivity disagree")
    query_api = "hybrid_knn_query_dnf" if panel == "c" else "hybrid_knn_query"
    environment = environment_config(0, 32, requested_cores=[8, 16, 24],
                                     update_cpus=list(range(32)))
    limits = {**DEFAULT_LIMITS, "frequency_khz": [3_267_000, 3_333_000]}
    source = {name: file_info(Path(__file__).parent / name) for name in
              ("figure4a.py", "aggregate.py", "dynamic/runtime.py", "dynamic/gt.py")}
    progress("verifying_inputs")
    inputs = {}
    for name in ("vectors", "queries", "attributes"):
        expected = config["inputs"][name]
        info = file_info(expected["path"])
        if info["sha256"] != expected["sha256"]:
            raise ValueError(f"Original {name} payload changed")
        inputs[name] = info
    native, native_info = import_native(config["native_dir"], config["native_sha256"])
    progress("verifying_existing_checkpoint")
    checkpoint = file_info(config["checkpoint"]["path"], config["checkpoint"])
    header = read_index_header(checkpoint["path"])
    if (header["count"] != config["N"] or header["M"] != 40
            or header["ef_construction"] != 300 or header["attr_positions"] != [0, 1]
            or header["attr_words"] != 2):
        raise ValueError("Existing checkpoint does not match the full10M M40/efc300 trial")
    vectors = read_fvecs(inputs["vectors"]["path"], config["N"], 128, integer_sift=True)
    queries = np.ascontiguousarray(read_fvecs(inputs["queries"]["path"], 1000, 128))
    progress("verifying_checkpoint_records")
    records = verify_index_records(checkpoint["path"], header,
                                   np.ones(config["N"], dtype=bool), vectors=vectors)
    numeric, categorical = record_attributes(checkpoint["path"], header)
    cells = []
    for cell in config["cells"]:
        predicate_info, gt_info = file_info(cell["predicate"]), file_info(cell["ground_truth"])
        if (predicate_info["sha256"] != cell["predicate_sha256"]
                or gt_info["sha256"] != cell["ground_truth_sha256"]):
            raise ValueError("Audited predicate/GT files changed before the run")
        predicates = read_json(cell["predicate"])
        tail_masks = None
        if panel == "c":
            bounds, masks, tail_masks = composite_predicates(predicates, len(queries))
            first_label, second_label = cell_labels[cell["name"]]
            if (not np.all(masks == 1 << first_label)
                    or not np.all(tail_masks == 1 << second_label)):
                raise ValueError("Categorical conditions differ from the existing composite cell")
        else:
            bounds, masks = and_predicates(predicates, len(queries))
            if not np.all(masks == 1 << cell_labels[cell["name"]]):
                raise ValueError("Categorical condition differs from the existing AND cell")
        stored_rows = cell.get("ground_truth_rows", len(queries))
        truth = ground_truth_prefix(read_json(cell["ground_truth"]),
                                    config["N"], len(queries), config["k"], stored_rows)
        if not matches(truth, numeric, categorical, bounds, masks, tail_masks).all():
            raise ValueError(f"Existing GT violates its predicate: {cell['name']}")
        cells.append((cell, predicates, bounds, masks, tail_masks, truth))
        counts = matching_counts(numeric, categorical, bounds, masks, tail_masks)
        if np.any(counts < config["k"]):
            raise ValueError("Fewer than k eligible records in a Figure4 query")
        inputs[cell["name"]] = {
            "predicates": predicate_info, "ground_truth": gt_info,
            "ground_truth_rows": stored_rows, "used_ground_truth_rows": len(queries),
            "actual_selectivity_mean": float(counts.mean() / config["N"]),
            "actual_selectivity_min": float(counts.min() / config["N"]),
            "actual_selectivity_max": float(counts.max() / config["N"]),
            "existing_gt_predicate_validated": True,
        }
    atomic_json(output / "manifest.json", {
        "request": config, "native": native_info, "checkpoint": checkpoint,
        "header": header, "record_validation": records, "inputs": inputs,
        "environment": environment, "limits": limits,
        "figure_panel": panel, "query_api": query_api,
        "stats_api_timed": False, "augmentation": False,
        "query_payload_sha256": array_digest(queries), "source": source,
    })
    progress("loading_existing_index")
    index = native.Index(space="l2", dim=128)
    index.set_num_threads(32)
    index.load_index(checkpoint["path"])
    index.generateAttrIndexes()
    index.set_thresholds(0.0001, 0.0001, 0.0001)
    index.set_num_threads(1)
    index.set_ef_top(1)
    index.set_ft_flag(True)
    index.set_ft_routing_flag(True)
    index.set_ft_routing_min_deg(config["routing_min_deg"])
    index.set_ft_routing_backfill_tail(False)
    native_state(index, config["N"], 0)
    summaries = []
    query_method = getattr(index, query_api)
    for cell, predicates, bounds, masks, tail_masks, truth in cells:
        def query():
            return query_method(queries, predicates, k=config["k"], num_threads=1)

        def validate(result):
            labels, distances = result
            ids = validate_ids(labels, config["N"], len(queries), config["k"])
            if not matches(ids, numeric, categorical, bounds, masks, tail_masks).all():
                raise RuntimeError("ANN result violates its predicate")
            if distances.shape != ids.shape or not np.isfinite(distances).all():
                raise RuntimeError("Invalid ANN distances")
            delta = vectors[ids].astype(np.float64) - queries[:, None, :].astype(np.float64)
            exact = np.sum(delta * delta, axis=2)
            if not np.array_equal(exact, distances):
                raise RuntimeError("ANN vector/label/distance mismatch")
            recall = float(np.any(labels[:, :, None] == truth[:, None, :], axis=1).mean())
            return {"recall": recall, "labels_sha256": array_digest(labels),
                    "distances_sha256": array_digest(distances)}

        os.sched_setaffinity(0, environment["update_cpus"])
        core, _ = choose_core(environment["query_candidates"])
        os.sched_setaffinity(0, {core})
        points = []
        for ef in range(config["k"], config["max_ef"] + 1):
            index.set_ef(ef)
            point = {"ef": ef, **validate(query())}
            points.append(point)
            progress("calibrating", cell=cell["name"], ef=ef, recall=point["recall"])
            atomic_json(output / f"{cell['name']}-calibration.json", {"points": points})
            if point["recall"] >= config["target_recall"]:
                break
        if points[-1]["recall"] < config["target_recall"]:
            raise RuntimeError(f"{cell['name']}: no ef reached95%; no timing fallback")
        bracket = points[-2:] if len(points) > 1 else points
        attempts, summary = [], None
        remaining = list(environment["query_candidates"])
        for _ in range(limits["max_cores"]):
            os.sched_setaffinity(0, environment["update_cpus"])
            core, loads = choose_core(remaining)
            remaining.remove(core)
            os.sched_setaffinity(0, {core})
            attempt = {"core": core, "loads": loads, "rounds": [], "accepted": []}
            attempts.append(attempt)
            for repeat in range(limits["max_rounds_per_core"]):
                samples = []
                order = bracket[repeat % len(bracket):] + bracket[:repeat % len(bracket)]
                for point in order:
                    index.set_ef(point["ef"])
                    query()
                    result, sample = timed_query(
                        query, core, environment["siblings"][str(core)], limits)
                    actual = validate(result)
                    if any(actual[key] != point[key] for key in actual):
                        raise RuntimeError("Frozen query results changed during timing")
                    sample.update(ef=point["ef"], recall=point["recall"],
                                  qps=len(queries) / sample["wall_s"])
                    samples.append(sample)
                attempt["rounds"].append(samples)
                if all(sample["clean"] for sample in samples):
                    attempt["accepted"].append(repeat)
                atomic_json(output / f"{cell['name']}-timing.json", {"attempts": attempts})
                progress("timing", cell=cell["name"], core=core, round=repeat,
                         accepted=len(attempt["accepted"]))
                if len(attempt["accepted"]) == limits["clean_rounds"]:
                    break
            if len(attempt["accepted"]) == limits["clean_rounds"]:
                curve = []
                for point in bracket:
                    samples = [sample["qps"] for r in attempt["accepted"]
                               for sample in attempt["rounds"][r] if sample["ef"] == point["ef"]]
                    curve.append((point["ef"], point["recall"], statistics.median(samples)))
                above = len(curve) == 1 and curve[0][1] > config["target_recall"]
                qps95 = None if above else qps_at_recall(curve, config["target_recall"])
                summary = {**cell, "curve": curve, "qps95": qps95,
                           "observed_qps": curve[-1][2], "observed_recall": curve[-1][1],
                           "selected_core": core, "accepted_rounds": attempt["accepted"],
                           "relative_to_plotted_reference": (
                               qps95 / cell["reference_qps"] - 1 if qps95 is not None else None)}
                break
        if summary is None:
            raise RuntimeError(f"{cell['name']}: insufficient CPU-clean rounds")
        summaries.append(summary)
        atomic_json(output / "results.json", {"status": "running", "cells": summaries})
    native_state(index, config["N"], 0)
    require_unchanged(checkpoint)
    for name in ("vectors", "queries", "attributes"):
        require_unchanged(inputs[name])
    require_unchanged(native_info)
    for cell in config["cells"]:
        require_unchanged(inputs[cell["name"]]["predicates"])
        require_unchanged(inputs[cell["name"]]["ground_truth"])
    for info in source.values():
        require_unchanged(info)
    atomic_json(output / "results.json", {
        "status": "complete", "cells": summaries, "comparison_limits": config["comparison_limits"]})
    atomic_json(output / "status.json", {"status": "complete", "completed_cells": len(summaries)})
    print(json.dumps(summaries, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = read_json(args.config)
    atomic_json(output / "request.json", config)
    try:
        run(config, output)
    except (OSError, ValueError, RuntimeError) as error:
        failure = {"status": "failed", "error_type": type(error).__name__, "error": str(error)}
        atomic_json(output / "status.json", failure)
        results_path = output / "results.json"
        if results_path.exists():
            partial = read_json(results_path)
            atomic_json(results_path, {**partial, "status": "partial", "failure": failure})
        raise


if __name__ == "__main__":
    main()
