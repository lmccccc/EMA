"""Screen the six original YouTube AND cells without rebuilding or launching a suite."""

import argparse
import contextlib
import csv
import importlib
import os
from pathlib import Path
import re
import shutil
import sys

import numpy as np

from .dynamic.runtime import OutputLock, atomic_json, file_info, immutable_json, require_unchanged, write_npz
from .index_metadata import record_sample
from .replay_existing import (
    REPO, predicate_validator, read_json, summaries, unchanged_index, verify_measurement,
)
from .static_paper import Attributes, Predicates


COARSE_EFS = (10, 20, 40, 80, 120, 160)
SELECTIVITIES = (1, 2, 3, 5, 7, 10)


def original_qps(path):
    text = Path(path).read_text()
    match = re.search(
        r"bfann_youtube_rgb_95_df\s*<-\s*data.frame\(\s*x\s*=\s*c\(([^)]*)\),"
        r"\s*y\s*=\s*c\(([^)]*)\)", text)
    if match is None:
        raise ValueError("Missing original YouTube QPS95 plotting data")
    xs = [float(value) for value in match[1].split(",")]
    ys = [float(value) for value in match[2].split(",")]
    if len(xs) != len(ys) or any(value <= 0 for value in ys):
        raise ValueError("Invalid original QPS95 values")
    values = dict(zip((round(x * 100) for x in xs), ys))
    if sorted(values) != list(SELECTIVITIES):
        raise ValueError("Expected the six original low-selectivity AND points")
    return values


def recall_bracket(measure, target=0.95, grid=COARSE_EFS):
    lower = None
    for upper in grid:
        if measure(upper) >= target:
            break
        lower = upper
    else:
        raise RuntimeError("Recall95 not reached within the bounded screening grid")
    if lower is None:
        return [upper]
    while upper - lower > 1:
        middle = (upper + lower) // 2
        if measure(middle) >= target:
            upper = middle
        else:
            lower = middle
    return [lower, upper]


def comparison(nominal, old_qps, target, reused=False):
    qps = target["reported_qps"]
    if qps is None or target["status"] == "unreached":
        raise RuntimeError("An unreached target cannot be reported as an improvement")
    gain = qps / old_qps - 1
    return {
        "selectivity_pct": nominal, "original_qps95": old_qps,
        "qps_at_target": target["qps_at_target"], "qualified_qps": qps,
        "target_status": target["status"], "observed_recall": target["observed_recall"],
        "gain_pct": gain * 100, "exceeds_5pct": qps > old_qps * 1.05, "reused": reused,
    }


def run(args, output):
    reference = Path(args.reference).resolve()
    request = read_json(reference / "manifest.json")
    if read_json(reference / "status.json")["status"] != "complete":
        raise RuntimeError("The cached1% reference is not complete")
    group = next(g for g in read_json(args.matrix)["groups"] if g["name"] == "youtube1m-mixed")
    cached_group = request["groups"][0]
    if ({k: v for k, v in group.items() if k != "cells"}
            != {k: v for k, v in cached_group.items() if k != "cells"}):
        raise RuntimeError("Screening inputs/index differ from the completed1% reference")
    settings = request["settings"]
    if (settings["ef_top"] != 1 or settings["use_ft"] is not True or settings["repeats"] != 3
            or settings["routing_min_deg"] != 16 or settings["k"] != 10
            or settings["backfill_tail"] is not False
            or "query_patch" not in request["native_provenance"]):
        raise ValueError("Keep the recovered Marker-on, ef_top1, three-repeat configuration")
    for info in request["source"].values():
        file_info(info["path"], info)
    native_info = file_info(request["native"]["path"], request["native"])
    if any(os.environ.get(key) != value for key, value in settings["environment"].items()):
        raise RuntimeError("Use the reference OpenMP/BLAS environment")
    if settings["query_core"] not in os.sched_getaffinity(0):
        raise RuntimeError("Reference query core is outside the launch affinity")
    unchanged_index(group["index"])
    baseline_info = file_info(args.baseline)
    baseline = original_qps(args.baseline)
    immutable_json(output / "manifest.json", {
        "schema": "youtube-low-and-screen-v1", "selectivities_pct": SELECTIVITIES,
        "reference": file_info(reference / "manifest.json"), "matrix": file_info(args.matrix),
        "baseline": baseline_info, "baseline_values": baseline, "native": native_info,
        "source": file_info(__file__), "query_sources": request["source"], "group": group,
        "settings": {key: value for key, value in settings.items() if key != "efs"},
        "reference_settings": settings, "coarse_efs": COARSE_EFS,
        "calibration": "recall only; one batch per sampled ef; bisect to adjacent integer ef",
        "timing": "three full batches per bracket endpoint, median of all three",
        "decision": "strictly more than5% relative to original plotting values; no automatic suite launch",
    })
    shutil.copyfile(__file__, output / "screen_youtube.py")
    cells = {cell["name"]: cell for cell in group["cells"]}
    cached = read_json(reference / group["name"] / "and-T1.json")
    verify_measurement(cached, group, cells["and-T1"], request)
    results = [comparison(1, baseline[1], cached["targets"][0], reused=True)]
    immutable_json(output / "and-T1.json", {"reused": file_info(
        reference / group["name"] / "and-T1.json"), "comparison": results[0]})
    sys.path.insert(0, str(Path(native_info["path"]).parent))
    native = importlib.import_module("hashannlib")
    if Path(native.__file__).resolve() != Path(native_info["path"]).resolve():
        raise RuntimeError("The wrong native was imported")
    sys.path.insert(0, str(REPO / "tests"))
    queries_api = importlib.import_module("hashann_query")
    builder_api = importlib.import_module("hashann")
    for field in ("queries", "attributes"):
        file_info(group[field]["path"], group[field])
    attributes = Attributes.load(group["attributes"]["path"], [0, 1], group["N"])
    samples = record_sample(group["index"]["path"], group["index"]["layout"],
                            np.linspace(0, group["N"] - 1, 256, dtype=np.int64))
    for sample in samples:
        if sample["label"] >= group["N"]:
            raise RuntimeError("Saved label is outside the source corpus")
        record = attributes.records[sample["label"]]
        if sample["attributes"] != [record[0], sorted(record[1])]:
            raise RuntimeError("Saved index attributes differ from the source")
    atomic_json(output / "status.json", {"status": "running", "phase": "loading_one_existing_index"})
    wrapper = builder_api.HashANN()
    params = {"metric": group["metric"], "dim": group["dimension"], "N": group["N"],
              "M": 40, "ef_construction": 300, "threads": 32}
    wrapper.init_params(params)
    index = wrapper.load_index(params, [0, 1], group["index"]["path"], 32)
    if index.get_current_count() != group["N"] or index.get_deleted_ratio() != 0:
        raise RuntimeError("The saved static corpus changed")
    os.sched_setaffinity(0, {settings["query_core"]})
    for nominal in SELECTIVITIES[1:]:
        cell = cells[f"and-T{nominal}"]
        directory = output / cell["name"]
        directory.mkdir()
        for field in ("predicate", "ground_truth"):
            file_info(cell[field]["path"], cell[field])
        q, p, gt = queries_api.load_query_data(
            group["queries"]["path"], cell["predicate"]["path"], cell["ground_truth"]["path"],
            group["N"], cell["query_count"], 10)
        if q.shape[1] != group["dimension"]:
            raise RuntimeError("Query dimensions differ from the saved index")
        validate = predicate_validator(attributes, Predicates.parse(p, [0, 1], len(q)))
        validate(gt)
        calibration = {}

        def measure(ef):
            if ef not in calibration:
                with (directory / f"calibration-ef{ef}.log").open("x") as log, contextlib.redirect_stdout(log):
                    measured = queries_api.query_sweep(
                        index, q, p, gt, count=group["N"], k=10, efs=[ef], ef_top=1,
                        use_ft=True, routing_min_deg=16, repeats=1, validate_labels=validate,
                        on_result=lambda e, labels, distances: write_npz(
                            directory / f"calibration-ef{e}.npz", labels=labels, distances=distances))
                calibration[ef] = measured["points"][0]
                atomic_json(directory / "calibration.json", {"points": list(calibration.values())})
                atomic_json(output / "status.json", {
                    "status": "running", "phase": "recall_calibration", "cell": cell["name"],
                    "ef": ef, "recall": calibration[ef]["recall"]})
            return calibration[ef]["recall"]

        selected = recall_bracket(measure)
        raw_files = {}

        def save_result(ef, labels, distances):
            raw_files[str(ef)] = write_npz(
                directory / f"timed-ef{ef}.npz", labels=labels, distances=distances)

        atomic_json(output / "status.json", {"status": "running", "phase": "timing",
                                             "cell": cell["name"], "efs": selected})
        with (directory / "timing.log").open("x") as log, contextlib.redirect_stdout(log):
            timed = queries_api.query_sweep(
                index, q, p, gt, count=group["N"], k=10, efs=selected, ef_top=1,
                use_ft=True, routing_min_deg=16, repeats=3, validate_labels=validate, on_result=save_result)
        for point in timed["points"]:
            expected = calibration[point["ef"]]
            if (point["recall"] != expected["recall"] or any(
                    sample[key] != expected["samples"][0][key] for sample in point["samples"]
                    for key in ("labels_sha256", "distances_sha256"))):
                raise RuntimeError("Timing no longer matches the recall-only calibration")
        result = {"status": "complete", "cell": cell, "measurement": timed, "raw_files": raw_files,
                  "targets": summaries(timed["rows"], [0.95])}
        for field in ("predicate", "ground_truth"):
            require_unchanged(cell[field])
        verification = {**request, "settings": {**settings, "efs": selected}}
        verify_measurement(result, group, cell, verification)
        compared = comparison(nominal, baseline[nominal], result["targets"][0])
        result["comparison"] = compared
        immutable_json(directory / "results.json", result)
        results.append(compared)
        print(compared, flush=True)
        atomic_json(output / "results.json", {"status": "running", "rows": results})
    unchanged_index(group["index"])
    for info in [native_info, baseline_info, group["queries"], group["attributes"],
                 *request["source"].values()]:
        require_unchanged(info)
    with (output / "summary.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    decision = {
        "status": "complete", "rows": results, "index_unchanged": True,
        "above_5pct": [row["selectivity_pct"] for row in results if row["exceeds_5pct"]],
        "global_suite_started": False,
    }
    atomic_json(output / "results.json", decision)
    atomic_json(output / "status.json", {key: value for key, value in decision.items() if key != "rows"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, help="Original input matrix manifest; no execution")
    parser.add_argument("--reference", required=True, help="Completed corrected-native 1%% fine-ef run")
    parser.add_argument("--baseline", required=True, help="Original range_label_sel_recall_plot.R")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True)
    with OutputLock(output):
        try:
            run(args, output)
        except (ValueError, RuntimeError, OSError) as error:
            atomic_json(output / "status.json", {"status": "failed", "error": str(error)})
            raise


if __name__ == "__main__":
    main()
