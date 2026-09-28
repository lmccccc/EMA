"""Reuse the original HashANN load/query pipeline and collect existing-index results."""

import argparse
import contextlib
import csv
import importlib
import json
import os
from pathlib import Path
import statistics
import struct
import subprocess
import sys

import numpy as np

from .aggregate import qps_at_recall
from .dynamic.gt import array_digest
from .dynamic.runtime import (
    OutputLock, atomic_json, file_info, immutable_json, require_unchanged, timestamp, write_npz,
)
from .index_metadata import read_layout, record_sample
from .static_inputs import DATASETS, LABELS, OCQ, SELECTIVITIES
from .static_mixed_inputs import HIGH, LOW, and_path
from .static_paper import Attributes, Predicates


REPO = Path(__file__).resolve().parents[1]
QUERY_REPEATS = 3
EFS = [10, 12, 15, 18, 20, 30, 40, 50, 80, 100, 120, 150, 180, 200,
       300, 400, 600, 800, 1000, 1200, 1500, 2000, 2500, 3000, 4000]
SOURCE_FILES = (
    "tests/hashann_query.py", "tests/hashann.py", "tests/utils.py",
    "exp_benchmark/replay_existing.py", "exp_benchmark/aggregate.py",
    "exp_benchmark/index_metadata.py", "exp_benchmark/static_inputs.py",
    "exp_benchmark/static_mixed_inputs.py", "exp_benchmark/static_paper.py",
    "exp_benchmark/figure4a.py", "exp_benchmark/figure4a_work_control.py",
    "exp_benchmark/dynamic/runtime.py", "exp_benchmark/dynamic/gt.py",
    "exp_benchmark/dynamic/protocol.py", "exp_benchmark/legacy_query_compat.patch",
    "exp_benchmark/legacy_query_mask_compat.patch",
    "exp_benchmark/build_legacy_query_native.sh",
)
ENVIRONMENT = {
    "OMP_NUM_THREADS": "32", "OMP_PROC_BIND": "false", "OMP_WAIT_POLICY": "PASSIVE",
    "OMP_DYNAMIC": "FALSE", "OPENBLAS_NUM_THREADS": "1",
}


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def index_identity(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    layout = read_layout(path)
    after = path.stat()
    if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise RuntimeError("Index changed while reading its metadata")
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
            "inode": stat.st_ino, "device": stat.st_dev, "layout": layout}


def unchanged_index(info):
    now = Path(info["path"]).stat()
    if (now.st_size, now.st_mtime_ns, now.st_ctime_ns, now.st_ino, now.st_dev) != (
            info["bytes"], info["mtime_ns"], info["ctime_ns"], info["inode"], info["device"]):
        raise RuntimeError("The existing index changed during read-only replay")


def prepare(data_root, native_provenance):
    root = Path(data_root).resolve(strict=True)
    provenance = read_json(native_provenance)
    if (provenance["source_commit"] != "28e07e33cb2dd4e7c173ca1030e6c5bbd7f4f7bb"
            or provenance["index_rebuilt"] or provenance["index_files_modified"]):
        raise ValueError("This replay requires the legacy source with the compatibility patches, not V11")
    if "query_patch" not in provenance:
        raise ValueError("The loader-only native still omits numerical boundary buckets; "
                         "build the corrected query native without rebuilding an index")
    for field in ("native", "patch", "query_patch", "patched_header"):
        actual = file_info(provenance[field]["path"])
        if any(actual[key] != provenance[field][key] for key in ("path", "bytes", "sha256")):
            raise ValueError(f"The compatibility native provenance no longer matches: {field}")
    native = file_info(provenance["native"]["path"])
    cache = {}
    def info(path):
        path = str(path)
        if path not in cache:
            cache[path] = file_info(path)
        return cache[path]
    def cell(name, nominal, predicate, gt, count, plots, targets=(0.95,)):
        p, truth = read_json(predicate), read_json(gt)
        if (not isinstance(p, list) or not isinstance(truth, list) or not truth
                or not isinstance(truth[0], list) or len(truth[0]) < 10):
            raise ValueError("Malformed original predicate/GT files")
        if count not in (50, 100, 1000) or min(len(p), len(truth)) < count:
            raise ValueError("Unexpected original query prefix; do not regenerate or silently truncate")
        return {
            "name": name, "nominal_selectivity": nominal / 100, "plots": plots,
            "predicate": info(predicate), "ground_truth": info(gt),
            "predicate_rows": len(p), "ground_truth_rows": len(truth),
            "ground_truth_k": len(truth[0]), "query_count": count, "targets": list(targets),
        }
    def group(dataset, name, index, kinds, attributes, queries):
        spec = DATASETS[dataset]
        saved = index_identity(index)
        h = saved["layout"]
        if (h["format"] not in (1, 2, 3, 4) or h["count"] != spec["N"]
                or h["dimension"] != spec["dimension"] or h["attr_type"] != kinds
                or h["M"] != 40 or h["ef_construction"] != 300
                or h["ft_bits"] != (128 if len(kinds) == 2 else 256)):
            raise ValueError(f"Existing cache does not match the original query profile: {index}")
        query_info = info(queries)
        with Path(queries).open("rb") as stream:
            dimension = struct.unpack("<i", stream.read(4))[0]
        if dimension != spec["dimension"] or query_info["bytes"] % ((dimension + 1) * 4):
            raise ValueError(f"Original query fvecs geometry differs from the index: {queries}")
        return {
            "name": name, "dataset": spec["paper_dataset"], "N": spec["N"],
            "dimension": spec["dimension"], "metric": spec["metric"],
            "index": saved, "attributes": info(attributes), "queries": query_info,
            "normalization": "none", "query_file_rows": query_info["bytes"] // ((dimension + 1) * 4),
            "attribute_types": kinds, "cells": [],
        }
    groups = []
    for dataset in ("youtube1m", "redcaps4m", "sift10m", "wiki15m"):
        spec = DATASETS[dataset]
        directory = root / spec["directory"]
        labels = directory / "label/arbi_0_1_random"
        query = directory / spec["queries"]
        mixed = group(
            dataset, f"{dataset}-mixed",
            directory / "hashann/index/index_40_300_arbi_0_1_random_128_edgeFT", [0, 1],
            labels / "attr_arbi_0_1_random.json", query)
        cells = {}
        for panel, specifications in (("a", HIGH), ("b", LOW)):
            for nominal, selector in specifications:
                p = and_path(labels, selector)
                name = f"and-T{nominal}"
                if name in cells:
                    cells[name]["plots"].append({"figure": 5, "panel": panel})
                else:
                    cells[name] = cell(name, nominal, p, p.with_name(p.name.replace("predicate_", "gt_", 1)),
                                       1000, [{"figure": 5, "panel": panel}])
        for nominal in SELECTIVITIES:
            cells[f"dnf-T{nominal}"] = cell(
                f"dnf-T{nominal}", nominal, labels / f"predicate_dnf_or_T{nominal}.json",
                labels / f"gt_dnf_or_T{nominal}.json", 100 if dataset == "youtube1m" else 1000,
                [{"figure": 5, "panel": "c"}])
        mixed["cells"] = list(cells.values())
        groups.append(mixed)
        if dataset in ("youtube1m", "redcaps4m"):
            for kind, panel in ((0, "a"), (1, "b")):
                labels = directory / f"label/arbi_{kind}_random"
                single = group(
                    dataset, f"{dataset}-{'range' if kind == 0 else 'label'}",
                    directory / f"hashann/index/index_40_300_arbi_{kind}_random_256",
                    [kind], labels / f"attr_arbi_{kind}_random.json", query)
                selectors = [n / 100 for n in SELECTIVITIES] if kind == 0 else LABELS
                for nominal, selector in zip(SELECTIVITIES, selectors):
                    suffix = f"arbi_{kind}_[{selector}].json"
                    single["cells"].append(cell(
                        f"T{nominal}", nominal, labels / f"predicate_{suffix}", labels / f"gt_{suffix}",
                        100 if dataset == "youtube1m" and kind == 1 else 1000,
                        [{"figure": 6, "panel": panel}],
                        (0.90, 0.95) if dataset == "youtube1m" and kind == 0 else (0.95,)))
                groups.append(single)
        elif dataset == "wiki15m":
            ocq = directory / "neg_correlated"
            single = group(
                dataset, "wiki15m-ocq", ocq / "hashann/index/index_40_300_arbi_0_random_256", [0],
                directory / "fvecs/wiki_15.4M_birthdate.json", ocq / "join_22.93_embedding.fvecs")
            for nominal, suffix in OCQ:
                alias = info(ocq / f"join_{suffix}_embedding.fvecs")
                if alias["sha256"] != single["queries"]["sha256"]:
                    raise ValueError("Original OCQ files no longer contain identical query vectors")
                single["cells"].append(cell(
                    f"T{nominal}", nominal, ocq / f"join_{suffix}_predicate.json",
                    ocq / f"join_{suffix}_gt.json", 50, [{"figure": 6, "panel": "c"}]))
            groups.append(single)
    return {
        "schema": "existing-index-query-replay-v1", "native": native,
        "native_provenance": provenance, "groups": groups,
        "settings": {"k": 10, "efs": EFS, "ef_top": 1, "routing_min_deg": 16,
                     "use_ft": True, "backfill_tail": False, "repeats": QUERY_REPEATS, "query_core": 24,
                     "environment": ENVIRONMENT},
        "source": {name: file_info(REPO / name) for name in SOURCE_FILES},
        "scope": "query only; original graph, levels, Markers and inputs; no V11 graph claim",
    }


def predicate_validator(attributes, predicates):
    def validate(labels):
        if not predicates.matches(labels, attributes).all():
            raise ValueError("Query/GT labels violate the original predicate and attribute source")
    return validate


def summaries(rows, targets):
    curve = [list(row[:3]) for row in rows]
    result = []
    for target in targets:
        value = qps_at_recall(curve, target)
        minimum_above = curve[0][1] > target
        crossing = next((i for i, point in enumerate(curve) if point[1] >= target), None)
        bracket = [] if crossing is None else curve[max(0, crossing - 1):crossing + 1]
        result.append({
            "target_recall": target,
            "status": "unreached" if value is None else "minimum_above" if minimum_above else "reached",
            "qps_at_target": None if minimum_above else value,
            "reported_qps": value,
            "observed_recall": None if crossing is None else curve[crossing][1],
            "bracket": bracket,
        })
    return result


def worker(request, name, output):
    groups = [group for group in request["groups"] if group["name"] == name]
    if len(groups) != 1:
        raise ValueError("The requested group is not unique")
    group, settings = groups[0], request["settings"]
    if any(os.environ.get(key) != value for key, value in settings["environment"].items()):
        raise RuntimeError("Use the documented fixed OpenMP/BLAS replay environment")
    if settings["query_core"] not in os.sched_getaffinity(0):
        raise RuntimeError("The requested physical query core is outside the launch affinity")
    for info in request["source"].values():
        file_info(info["path"], info)
    native_info = file_info(request["native"]["path"], request["native"])
    sys.path.insert(0, str(Path(native_info["path"]).parent))
    native = importlib.import_module("hashannlib")
    if Path(native.__file__).resolve() != Path(native_info["path"]).resolve():
        raise RuntimeError("The wrong native module was imported")
    sys.path.insert(0, str(REPO / "tests"))
    query_module = importlib.import_module("hashann_query")
    builder_module = importlib.import_module("hashann")
    unchanged_index(group["index"])
    for field in ("attributes", "queries"):
        file_info(group[field]["path"], group[field])
    records = read_json(group["attributes"]["path"])
    if len(records) != group["N"]:
        raise ValueError("Attribute source count differs from the saved index")
    samples = record_sample(group["index"]["path"], group["index"]["layout"],
                            np.unique(np.linspace(0, group["N"] - 1, 256, dtype=np.int64)))
    for sample in samples:
        if sample["label"] >= group["N"]:
            raise ValueError("Sampled index label is outside the original corpus")
        record = records[sample["label"]]
        expected = [values if kind == 0 else sorted(set(values))
                    for kind, values in zip(group["attribute_types"], record)]
        if expected != sample["attributes"]:
            raise ValueError("Existing index attributes differ from the supplied experiment inputs")
    attributes = Attributes.from_records(records, group["attribute_types"], group["N"])
    atomic_json(output / "status.json", {"status": "running", "phase": "loading_existing_index"})
    wrapper = builder_module.HashANN()
    params = {"metric": group["metric"], "dim": group["dimension"], "N": group["N"],
              "M": 40, "ef_construction": 300, "threads": 32}
    wrapper.init_params(params)
    index = wrapper.load_index(params, group["attribute_types"], group["index"]["path"], 32)
    if index.get_current_count() != group["N"]:
        raise ValueError("Loaded index count differs from the declared corpus")
    if index.get_deleted_ratio() != 0:
        raise ValueError("Existing static index contains deleted points")
    os.sched_setaffinity(0, {settings["query_core"]})
    results, failed = [], False
    for cell in group["cells"]:
        prefix = output / cell["name"]
        atomic_json(output / "status.json", {"status": "running", "phase": "querying", "cell": cell["name"]})
        try:
            for field in ("predicate", "ground_truth"):
                file_info(cell[field]["path"], cell[field])
            queries, predicates, truth = query_module.load_query_data(
                group["queries"]["path"], cell["predicate"]["path"], cell["ground_truth"]["path"],
                group["N"], cell["query_count"], settings["k"])
            if queries.shape[1] != group["dimension"]:
                raise ValueError("Original query dimensions differ from the loaded index")
            parsed = Predicates.parse(predicates, group["attribute_types"], len(queries),
                                      query_module.is_dnf_predicate(predicates))
            validate = predicate_validator(attributes, parsed)
            validate(truth)
            payloads = output / (cell["name"] + "-arrays")
            payloads.mkdir()
            raw_files = {}
            def save_result(ef, labels, distances):
                raw_files[str(ef)] = write_npz(payloads / f"ef-{ef}.npz", labels=labels, distances=distances)
            with prefix.with_suffix(".log").open("x") as log, contextlib.redirect_stdout(log):
                measured = query_module.query_sweep(
                    index, queries, predicates, truth, count=group["N"], k=settings["k"],
                    efs=settings["efs"], ef_top=settings["ef_top"], use_ft=settings["use_ft"],
                    routing_min_deg=settings["routing_min_deg"], backfill_tail=settings["backfill_tail"],
                    repeats=settings["repeats"], target_recall=max(cell["targets"]),
                    validate_labels=validate, on_result=save_result,
                    on_point=lambda point: atomic_json(output / "status.json", {
                        "status": "running", "phase": "querying", "cell": cell["name"],
                        "ef": point["ef"], "recall": point["recall"]}))
            for field in ("predicate", "ground_truth"):
                require_unchanged(cell[field])
            result = {"status": "complete", "cell": cell, "measurement": measured, "raw_files": raw_files,
                      "targets": summaries(measured["rows"], cell["targets"])}
            if any(target["status"] == "unreached" for target in result["targets"]):
                result["status"], failed = "unreached", True
        except (OSError, ValueError, RuntimeError) as error:
            failed = True
            result = {"status": "failed", "cell": cell, "error_type": type(error).__name__,
                      "error": str(error)}
            print(json.dumps(result), file=sys.stderr, flush=True)
        immutable_json(prefix.with_suffix(".json"), result)
        results.append(result)
        atomic_json(output / "results.json", {"status": "running", "cells": results})
    unchanged_index(group["index"])
    for info in [native_info, *request["source"].values(), group["attributes"], group["queries"]]:
        require_unchanged(info)
    atomic_json(output / "results.json", {
        "status": "failed" if failed else "complete", "group": group, "cells": results,
        "native": native_info, "index_unchanged": True, "index_rebuilt": False,
        "sampled_index_attribute_rows": len(samples), "finished_at": timestamp()})
    atomic_json(output / "status.json", {"status": "failed" if failed else "complete"})
    return 1 if failed else 0


def verify_measurement(result, group, cell, request):
    if result["cell"] != cell:
        raise ValueError("Collected cell differs from the frozen workload")
    measured, settings = result["measurement"], request["settings"]
    truth = np.asarray(read_json(cell["ground_truth"]["path"]))[:cell["query_count"], :settings["k"]]
    if array_digest(truth) != measured["ground_truth_sha256"]:
        raise ValueError("Collected query/GT prefix differs from the original ground truth")
    points, rows = measured["points"], measured["rows"]
    if (len(points) != len(rows) or not points or measured["query_count"] != cell["query_count"]
            or [point["ef"] for point in points] != settings["efs"][:len(points)]):
        raise ValueError("Collected points differ from the predefined query grid")
    for point, row in zip(points, rows):
        raw = result["raw_files"][str(point["ef"])]
        file_info(raw["path"], raw)
        with np.load(raw["path"], allow_pickle=False) as arrays:
            labels, distances = arrays["labels"], arrays["distances"]
        if (labels.shape != truth.shape or not np.issubdtype(labels.dtype, np.integer)
                or np.any(labels < 0) or np.any(labels >= group["N"])
                or np.any(np.diff(np.sort(labels, axis=1), axis=1) == 0)
                or distances.shape != labels.shape or not np.isfinite(distances).all()):
            raise ValueError("Invalid collected native output")
        recall = float(np.any(labels[:, :, None] == truth[:, None, :], axis=1).mean())
        if len(point["samples"]) != settings["repeats"] or point["recall"] != recall:
            raise ValueError("Collected recall or repeat count differs from raw evidence")
        for sample in point["samples"]:
            if (sample["wall_s"] <= 0 or sample["qps"] != cell["query_count"] / sample["wall_s"]
                    or sample["recall"] != recall or sample["labels_sha256"] != array_digest(labels)
                    or sample["distances_sha256"] != array_digest(distances)):
                raise ValueError("Collected timing or labels differ from raw evidence")
        qps = statistics.median(sample["qps"] for sample in point["samples"])
        comparisons = statistics.median(sample["cmps"] for sample in point["samples"])
        if row != [point["ef"], recall, qps, comparisons] or point["qps"] != qps:
            raise ValueError("Collected row differs from median timings")
    if result["targets"] != summaries(rows, cell["targets"]):
        raise ValueError("Collected target throughput differs from the original interpolation rule")


def export(request, output):
    rows = []
    for group in request["groups"]:
        path = output / group["name"] / "results.json"
        saved = read_json(path) if path.is_file() else {"status": "failed", "cells": []}
        accepted = (saved.get("index_unchanged") is True and saved.get("group") == group
                    and saved.get("native") == request["native"])
        completed = {cell["cell"]["name"]: cell for cell in saved["cells"]} if accepted else {}
        for cell in group["cells"]:
            result = completed.get(cell["name"])
            if result and result["status"] in ("complete", "unreached"):
                for field in ("predicate", "ground_truth"):
                    file_info(cell[field]["path"], cell[field])
                verify_measurement(result, group, cell, request)
            targets = result["targets"] if result and result["status"] in ("complete", "unreached") else [
                {"target_recall": target, "status": "failed", "qps_at_target": None,
                 "reported_qps": None, "observed_recall": None} for target in cell["targets"]]
            for target in targets:
                for plot in cell["plots"]:
                    rows.append({
                        "figure": plot["figure"], "panel": plot["panel"], "dataset": group["dataset"],
                        "cell": cell["name"], "nominal_selectivity": cell["nominal_selectivity"],
                        "query_count": cell["query_count"], "target_recall": target["target_recall"],
                        "status": target["status"], "qps_at_target": target["qps_at_target"],
                        "reported_qps": target["reported_qps"], "observed_recall": target["observed_recall"],
                        "index_format": group["index"]["layout"]["format"],
                        "ft_bits": group["index"]["layout"]["ft_bits"], "index": group["index"]["path"],
                        "native_sha256": request["native"]["sha256"],
                    })
    with (output / "summary.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    immutable_json(output / "summary.json", {"rows": rows, "plotted_rows": len(rows),
                                             "scope": request["scope"]})
    return rows


def collect(request_path, output):
    request = read_json(request_path)
    immutable_json(output / "manifest.json", request)
    archive = output / "source"
    for name, info in request["source"].items():
        file_info(info["path"], info)
        destination = archive / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Path(info["path"]).open("rb") as source, destination.open("xb") as target:
            target.write(source.read())
    failed = []
    for group in request["groups"]:
        directory = output / group["name"]
        directory.mkdir()
        atomic_json(output / "status.json", {"status": "running", "group": group["name"]})
        with (directory / "worker.log").open("x") as log:
            result = subprocess.run([
                sys.executable, "-m", "exp_benchmark.replay_existing", "worker",
                "--request", str(output / "manifest.json"), "--group", group["name"],
                "--output", str(directory)], cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            failed.append(group["name"])
            atomic_json(directory / "status.json", {"status": "failed", "worker_exit": result.returncode})
            completed = directory / "results.json"
            if not completed.is_file() or not read_json(completed).get("index_unchanged"):
                # A setup/worker crash is not a low-recall cell; stop before loading more indexes.
                break
    rows = export(request, output)
    atomic_json(output / "status.json", {
        "status": "failed" if failed else "complete", "failed_groups": failed,
        "finished_at": timestamp(), "index_rebuilt": False, "plotted_rows": len(rows)})
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("prepare")
    audit.add_argument("--data-root", required=True)
    audit.add_argument("--native-provenance", required=True)
    audit.add_argument("--request", required=True)
    run = commands.add_parser("collect")
    run.add_argument("--request", required=True)
    run.add_argument("--output", required=True)
    child = commands.add_parser("worker")
    child.add_argument("--request", required=True)
    child.add_argument("--group", required=True)
    child.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        immutable_json(args.request, prepare(args.data_root, args.native_provenance))
        return 0
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=args.command == "worker")
    with OutputLock(output):
        if args.command == "collect":
            return collect(args.request, output)
        return worker(read_json(args.request), args.group, output)


if __name__ == "__main__":
    sys.exit(main())
