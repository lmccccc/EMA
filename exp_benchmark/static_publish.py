"""Validate raw static measurements and export paper data, never substitute old EMA curves."""

import argparse
import csv
import io
import math
from pathlib import Path
import statistics

import numpy as np

from .dynamic.gt import array_digest
from .dynamic.runtime import (
    DEFAULT_LIMITS, file_info, immutable_json, noise_reasons, require_unchanged,
)
from .static_paper import PARAMETERS, SCHEMA, read_json, summarize_target


COLUMNS = (
    "figure", "panel", "dataset", "nominal_selectivity", "actual_selectivity",
    "target_recall", "qps_at_target", "reported_qps", "observed_recall", "selected_ef",
    "method", "query_count", "native_sha256", "index_sha256", "query_sha256",
    "predicate_sha256", "ground_truth_sha256", "measurement",
)


def checked_timing(points, timing, limits, count, targets, *, legacy=False):
    if ([p["ef"] for p in points] != list(range(10, 10 + len(points))) or not points
            or any(not 0 <= p["recall"] <= 1 for p in points)
            or any(p["recall"] >= max(targets) for p in points[:-1])
            or points[-1]["recall"] < max(targets)):
        raise ValueError("Invalid ascending first-crossing calibration trace")
    needed = set()
    for target in targets:
        crossing = next(i for i, point in enumerate(points) if point["recall"] >= target)
        needed.update(range(max(0, crossing - 1), crossing + 1))
    bracket = [points[i] for i in sorted(needed)]
    by_ef = {p["ef"]: p for p in bracket}
    expected_limits = {**DEFAULT_LIMITS, "frequency_khz": [3_267_000, 3_333_000]}
    if limits != expected_limits:
        raise ValueError("Static timing admission limits differ from the frozen protocol")
    accepted_name = "accepted" if legacy else "accepted_rounds"
    selected = None
    attempts = timing["attempts"]
    if not attempts or len(attempts) > limits["max_cores"]:
        raise ValueError("Invalid number of static timing attempts")
    cores = [attempt["core"] for attempt in attempts]
    if len(set(cores)) != len(cores) or any(core not in (8, 16, 24) for core in cores):
        raise ValueError("Timing did not use distinct prescribed physical cores")
    for attempt in attempts:
        clean = []
        if selected is not None or len(attempt["rounds"]) > limits["max_rounds_per_core"]:
            raise ValueError("Timing continued after admission or exceeded the round budget")
        for repeat, samples in enumerate(attempt["rounds"]):
            order = bracket[repeat % len(bracket):] + bracket[:repeat % len(bracket)]
            if [sample["ef"] for sample in samples] != [point["ef"] for point in order]:
                raise ValueError("Timing rounds do not cover the prescribed rotated ef bracket")
            for sample in samples:
                reasons = noise_reasons(sample, attempt["core"], limits)
                point = by_ef[sample["ef"]]
                if (sample["clean"] != (not reasons) or sample["rejection_reasons"] != reasons
                        or sample["recall"] != point["recall"]
                        or not math.isfinite(sample["wall_s"]) or sample["wall_s"] <= 0
                        or not math.isclose(sample["qps"], count / sample["wall_s"], rel_tol=1e-12)):
                    raise ValueError("Raw timing/recall/noise evidence is inconsistent")
                if not legacy and any(sample[key] != point[key] for key in
                                      ("labels_sha256", "distances_sha256")):
                    raise ValueError("Timed results differ from the frozen calibration hashes")
            if all(sample["clean"] for sample in samples):
                clean.append(repeat)
            if len(clean) == limits["clean_rounds"] and repeat != len(attempt["rounds"]) - 1:
                raise ValueError("Timing did not stop after the first seven complete clean rounds")
        if attempt[accepted_name] != clean:
            raise ValueError("Recorded clean-round selection differs from raw CPU telemetry")
        if len(clean) == limits["clean_rounds"]:
            selected = attempt
        elif len(attempt["rounds"]) != limits["max_rounds_per_core"]:
            raise ValueError("A timing core was abandoned before its prescribed round budget")
    if selected is None:
        raise ValueError("No CPU-clean attempt supports this paper measurement")
    medians = {
        point["ef"]: statistics.median(
            sample["qps"] for repeat in selected[accepted_name]
            for sample in selected["rounds"][repeat] if sample["ef"] == point["ef"])
        for point in bracket
    }
    return [{**summarize_target(points, medians, target), "selected_core": selected["core"],
             "accepted_rounds": selected[accepted_name]} for target in targets]


def check_summary(recorded, actual):
    for key in ("target_recall", "qps_at_target", "reported_qps", "observed_qps",
                "observed_recall", "selected_ef", "method"):
        if recorded[key] != actual[key]:
            raise ValueError(f"Published static summary differs from raw evidence: {key}")


def fresh_rows(directory):
    directory = Path(directory)
    manifest = read_json(directory / "suite-manifest.json")
    if read_json(directory / "status.json").get("status") != "complete":
        raise ValueError(f"Static suite is not complete: {directory}")
    rows, sources, costs = [], [], []
    for job in manifest["request"]["jobs"]:
        root = directory / "jobs" / job["name"]
        prepared = read_json(root / "manifest.json")
        build = read_json(root / "build.json")
        result = read_json(root / "results.json")
        if (prepared["job"] != job or prepared["source"] != manifest["source"]
                or job["parameters"] != PARAMETERS or result.get("schema") != SCHEMA
                or result.get("status") != "complete" or len(result["cells"]) != len(job["cells"])):
            raise ValueError("Incomplete/mismatched static job")
        require_unchanged(build["checkpoint"])
        require_unchanged(prepared["native"])
        for info in prepared["inputs"].values():
            require_unchanged(info)
        if job.get("build_only"):
            if build["reused"] or build["clustering_cache_reused"]:
                raise ValueError("Cold index-construction costs cannot use a reused graph/level plan")
            costs.append({
                "dataset": job["paper_dataset"], "index_bytes": build["checkpoint"]["bytes"],
                "index_gb": build["checkpoint"]["bytes"] / 1_000_000_000,
                "index_gib": build["checkpoint"]["bytes"] / 1024**3,
                "construction_s": build["construction_s"], "save_s": build["save_s"],
                "checkpoint": build["checkpoint"], "build_record": file_info(root / "build.json"),
            })
        for cell, recorded in zip(job["cells"], result["cells"]):
            path = root / "cells" / cell["name"] / "result.json"
            if read_json(path) != recorded or recorded["input"]["request"] != cell:
                raise ValueError("Static cell/aggregate records disagree")
            identity = recorded["measurement_identity"]
            if (identity["checkpoint"] != build["checkpoint"] or identity["native"] != prepared["native"]
                    or identity["query_payload_sha256"] != recorded["input"]["query_payload_sha256"]):
                raise ValueError("A query cell was measured against another index/native/query prefix")
            file_info(identity["manifest"]["path"], identity["manifest"])
            for info in recorded["artifacts"].values():
                file_info(info["path"], info)
            calibration = read_json(recorded["artifacts"]["calibration.json"]["path"])
            timing = read_json(recorded["artifacts"]["timing.json"]["path"])
            points = calibration["points"]
            if calibration["targets"] != cell["targets"] or timing["limits"] != prepared["limits"]:
                raise ValueError("Static calibration/timing targets changed")
            file_info(cell["ground_truth"]["path"], cell["ground_truth"])
            file_info(cell["predicate"]["path"], cell["predicate"])
            truth = np.asarray(read_json(cell["ground_truth"]["path"]))[:cell["query_count"], :10]
            with np.load(recorded["artifacts"]["calibration-arrays.npz"]["path"],
                         allow_pickle=False) as arrays:
                labels, distances = arrays["labels"], arrays["distances"]
                if labels.shape != (len(points), cell["query_count"], 10) or distances.shape != labels.shape:
                    raise ValueError("Raw calibration array dimensions changed")
                for point, found, distance in zip(points, labels, distances):
                    recall = float(np.any(found[:, :, None] == truth[:, None, :], axis=1).mean())
                    if (recall != point["recall"] or array_digest(found) != point["labels_sha256"]
                            or array_digest(distance) != point["distances_sha256"]):
                        raise ValueError("Raw calibration labels/distances do not support the recorded recall")
            actual = checked_timing(points, timing, prepared["limits"], cell["query_count"], cell["targets"])
            if len(recorded["targets"]) != len(actual):
                raise ValueError("Missing or extra recall-target summaries")
            for old, summary in zip(recorded["targets"], actual):
                check_summary(old, summary)
                if (recorded["selected_core"] != summary["selected_core"]
                        or recorded["accepted_rounds"] != summary["accepted_rounds"]):
                    raise ValueError("Recorded CPU/round selection differs from admitted timing")
                for plot in cell["plots"]:
                    rows.append({
                        **plot, "dataset": job["paper_dataset"],
                        "nominal_selectivity": cell["nominal_selectivity"],
                        "actual_selectivity": recorded["input"]["actual_selectivity"],
                        **{key: summary[key] for key in ("target_recall", "qps_at_target", "reported_qps",
                                                        "observed_recall", "selected_ef", "method")},
                        "query_count": cell["query_count"],
                        "native_sha256": prepared["native"]["sha256"],
                        "index_sha256": build["checkpoint"]["sha256"],
                        "query_sha256": recorded["input"]["query_payload_sha256"],
                        "predicate_sha256": cell["predicate"]["sha256"],
                        "ground_truth_sha256": cell["ground_truth"]["sha256"],
                        "measurement": str(path),
                    })
            sources.append(file_info(path))
    return rows, sources, costs


def sift_rows(directories):
    rows, sources, canonical = [], [], {}
    for panel, directory in zip(("a", "b", "c"), directories):
        root = Path(directory)
        manifest, results = read_json(root / "manifest.json"), read_json(root / "results.json")
        request = manifest["request"]
        if (results.get("status") != "complete" or request["target_recall"] != 0.95
                or request["routing_min_deg"] != 16 or request["query_count"] != 1000
                or request.get("figure_panel", "a") != panel or len(results["cells"]) != 6):
            raise ValueError("Invalid completed SIFT source trial")
        require_unchanged(manifest["checkpoint"])
        require_unchanged(manifest["native"])
        for name in ("vectors", "queries", "attributes"):
            require_unchanged(manifest["inputs"][name])
        for cell in results["cells"]:
            for key in ("predicates", "ground_truth"):
                info = manifest["inputs"][cell["name"]][key]
                file_info(info["path"], info)
            calibration = read_json(root / f"{cell['name']}-calibration.json")
            timing = read_json(root / f"{cell['name']}-timing.json")
            summary, = checked_timing(
                calibration["points"], timing, manifest["limits"], 1000, [0.95], legacy=True)
            if (cell["qps95"] != summary["qps_at_target"]
                    or cell["observed_qps"] != summary["observed_qps"]
                    or cell["observed_recall"] != summary["observed_recall"]
                    or cell["selected_core"] != summary["selected_core"]
                    or cell["accepted_rounds"] != summary["accepted_rounds"]):
                raise ValueError("Original SIFT summary is not supported by its raw timing")
            identity = (manifest["native"]["sha256"], manifest["checkpoint"]["sha256"],
                        manifest["query_payload_sha256"], cell["predicate_sha256"],
                        cell["ground_truth_sha256"])
            measurement = str(root / "results.json") + "#" + cell["name"]
            if identity in canonical:
                first, points, measurement = canonical[identity]
                if points != calibration["points"]:
                    raise ValueError("Duplicate SIFT workload has different calibration results")
                summary = first
            else:
                canonical[identity] = (summary, calibration["points"], measurement)
            rows.append({
                "figure": 5, "panel": panel, "dataset": "SIFT10M",
                "nominal_selectivity": cell["nominal_selectivity"],
                "actual_selectivity": manifest["inputs"][cell["name"]]["actual_selectivity_mean"],
                **{key: summary[key] for key in ("target_recall", "qps_at_target", "reported_qps",
                                                "observed_recall", "selected_ef", "method")},
                "query_count": 1000, "native_sha256": identity[0], "index_sha256": identity[1],
                "query_sha256": identity[2], "predicate_sha256": identity[3],
                "ground_truth_sha256": identity[4], "measurement": measurement,
            })
            sources.extend(file_info(root / f"{cell['name']}-{suffix}.json")
                           for suffix in ("calibration", "timing"))
        sources.extend(file_info(root / name) for name in ("manifest.json", "results.json"))
    return rows, sources


def expected_keys():
    low, high = (1, 2, 3, 5, 7, 10), (10, 20, 40, 60, 80, 100)
    keys = set()
    for dataset in ("YoutubeRGB1M", "Redcaps4M", "SIFT10M", "Wiki15.4M"):
        for panel, values in (("a", high), ("b", low), ("c", low)):
            keys.update((5, panel, dataset, value / 100, 0.95) for value in values)
    for dataset in ("YoutubeRGB1M", "Redcaps4M"):
        for panel in ("a", "b"):
            keys.update((6, panel, dataset, value / 100, 0.95) for value in low)
    keys.update((6, "a", "YoutubeRGB1M", value / 100, 0.90) for value in low)
    keys.update((6, "c", "Wiki15.4M", value / 100, 0.95) for value in (1, 5, 10, 15, 23))
    return keys


def export(initial, mixed, sift, output):
    first, first_sources, costs = fresh_rows(initial)
    second, second_sources, _ = fresh_rows(mixed)
    previous, previous_sources = sift_rows(sift)
    rows = first + second + previous
    key = lambda row: tuple(row[name] for name in
                            ("figure", "panel", "dataset", "nominal_selectivity", "target_recall"))
    if len(rows) != len(expected_keys()) or {key(row) for row in rows} != expected_keys():
        raise ValueError("Paper export does not contain exactly all107 requested EMA plotted points")
    if len({row["native_sha256"] for row in rows}) != 1:
        raise ValueError("The static paper cannot silently mix different native implementations")
    if {item["dataset"] for item in costs} != {"YoutubeRGB1M", "Redcaps4M", "Wiki15.4M", "SIFT10M"}:
        raise ValueError("Fresh static construction costs are incomplete")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(sorted(rows, key=key))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    csv_path = output / "ema_static_v11.csv"
    with csv_path.open("x", newline="") as destination:
        destination.write(stream.getvalue())
    cost_path = output / "ema_static_index_costs.csv"
    with cost_path.open("x", newline="") as destination:
        fields = ("dataset", "index_bytes", "index_gb", "index_gib", "construction_s", "save_s")
        writer = csv.DictWriter(destination, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(costs)
    immutable_json(output / "ema_static_v11_provenance.json", {
        "schema": "ema-static-paper-export-v1", "data": file_info(csv_path),
        "index_cost_data": file_info(cost_path),
        "rows": len(rows), "parameters": PARAMETERS,
        "measurements": first_sources + second_sources + previous_sources,
        "construction": costs,
        "sift_duplicate_policy": "Earliest supplied identical workload/calibration, never maximum QPS",
        "minimum_ef_policy": "Report observed throughput at the attained recall; no95% extrapolation",
        "input_policy": "Preserved original files/prefixes and raw IP; no predicate or GT regeneration",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", required=True)
    parser.add_argument("--mixed", required=True)
    parser.add_argument("--sift-a", required=True)
    parser.add_argument("--sift-b", required=True)
    parser.add_argument("--sift-c", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    export(args.initial, args.mixed, (args.sift_a, args.sift_b, args.sift_c), args.output)


if __name__ == "__main__":
    main()
