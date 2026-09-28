"""Paired regular/with-stats timings on a completed Figure4a trial's frozen graph."""

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
    import_native, native_state, require_unchanged, timed_query,
)
from .figure4a import read_json


def run(reference, output):
    manifest = read_json(reference / "manifest.json")
    results = read_json(reference / "results.json")
    if results["status"] != "complete":
        raise ValueError("API control requires a completed reference trial")
    config = manifest["request"]
    environment = environment_config(0, 32, requested_cores=[8, 16, 24],
                                     update_cpus=list(range(32)))
    limits = {**DEFAULT_LIMITS, "frequency_khz": [3_267_000, 3_333_000]}
    native, native_info = import_native(config["native_dir"], config["native_sha256"])
    require_unchanged(manifest["checkpoint"])
    require_unchanged(manifest["inputs"]["queries"])
    queries = np.ascontiguousarray(read_fvecs(
        manifest["inputs"]["queries"]["path"], config["query_count"], config["dimension"]))
    if array_digest(queries) != manifest["query_payload_sha256"]:
        raise ValueError("Reference query payload changed")
    atomic_json(output / "manifest.json", {
        "reference_manifest": file_info(reference / "manifest.json"),
        "reference_results": file_info(reference / "results.json"),
        "native": native_info, "checkpoint": manifest["checkpoint"],
        "environment": environment, "limits": limits,
        "source": file_info(__file__),
        "comparison": "same graph, predicates, ef, result identities and CPU; only query API varies",
    })
    atomic_json(output / "status.json", {"status": "running", "phase": "loading_index"})
    index = native.Index(space="l2", dim=config["dimension"])
    index.set_num_threads(32)
    index.load_index(manifest["checkpoint"]["path"])
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
    for cell in results["cells"]:
        predicate_info = manifest["inputs"][cell["name"]]["predicates"]
        require_unchanged(predicate_info)
        predicates = read_json(predicate_info["path"])
        calibration = read_json(reference / f"{cell['name']}-calibration.json")["points"]
        points = calibration[-2:] if len(calibration) > 1 else calibration
        if [p["ef"] for p in points] != [p[0] for p in cell["curve"]]:
            raise ValueError("Reference calibration bracket changed")
        methods = {
            "regular": lambda: index.hybrid_knn_query(
                queries, predicates, k=config["k"], num_threads=1),
            "with_stats": lambda: index.hybrid_knn_query_with_stats(
                queries, predicates, k=config["k"]),
        }

        def validate(point, result):
            if (array_digest(result[0]) != point["labels_sha256"]
                    or array_digest(result[1]) != point["distances_sha256"]):
                raise RuntimeError("API control changed frozen labels/distances; not pure timing overhead")

        cases = [(mode, point) for point in points for mode in methods]
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
                order = cases[repeat % len(cases):] + cases[:repeat % len(cases)]
                samples = []
                for mode, point in order:
                    index.set_ef(point["ef"])
                    validate(point, methods[mode]())
                    result, sample = timed_query(
                        methods[mode], core, environment["siblings"][str(core)], limits)
                    validate(point, result)
                    sample.update(mode=mode, ef=point["ef"], recall=point["recall"],
                                  qps=len(queries) / sample["wall_s"])
                    if mode == "with_stats":
                        sample.update(mean_distance_computations=float(np.mean(result[2])),
                                      mean_hops=float(np.mean(result[3])))
                    samples.append(sample)
                attempt["rounds"].append(samples)
                if all(sample["clean"] for sample in samples):
                    attempt["accepted"].append(repeat)
                atomic_json(output / f"{cell['name']}-timing.json", {"attempts": attempts})
                atomic_json(output / "status.json", {
                    "status": "running", "cell": cell["name"], "core": core,
                    "round": repeat, "accepted": len(attempt["accepted"]),
                })
                if len(attempt["accepted"]) == limits["clean_rounds"]:
                    break
            if len(attempt["accepted"]) == limits["clean_rounds"]:
                curves, qps = {}, {}
                for mode in methods:
                    curve = []
                    for point in points:
                        values = [sample["qps"] for r in attempt["accepted"]
                                  for sample in attempt["rounds"][r]
                                  if sample["mode"] == mode and sample["ef"] == point["ef"]]
                        curve.append((point["ef"], point["recall"], statistics.median(values)))
                    if len(curve) < 2 and curve[0][1] != config["target_recall"]:
                        raise ValueError("Reference has no valid target-recall bracket")
                    curves[mode] = curve
                    qps[mode] = qps_at_recall(curve, config["target_recall"])
                summary = {
                    "cell": cell["name"], "nominal_selectivity": cell["nominal_selectivity"],
                    "qps95": qps, "curves": curves,
                    "regular_gain_over_stats": qps["regular"] / qps["with_stats"] - 1,
                    "historical_reference_qps": cell["reference_qps"],
                    "core": core, "accepted_rounds": attempt["accepted"],
                    "identical_frozen_query_results": True,
                }
                break
        if summary is None:
            raise RuntimeError(f"{cell['name']}: insufficient clean paired API rounds")
        summaries.append(summary)
        require_unchanged(predicate_info)
        atomic_json(output / "results.json", {"status": "running", "cells": summaries})
    native_state(index, config["N"], 0)
    require_unchanged(manifest["checkpoint"])
    atomic_json(output / "results.json", {"status": "complete", "cells": summaries})
    atomic_json(output / "status.json", {"status": "complete", "completed_cells": len(summaries)})
    print(json.dumps(summaries, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        run(Path(args.reference).resolve(), output)
    except (OSError, ValueError, RuntimeError) as error:
        atomic_json(output / "status.json", {
            "status": "failed", "error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
