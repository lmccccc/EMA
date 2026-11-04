"""Compare strict neighbor filtering, Markers and recovery on one frozen graph."""

import argparse
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import uuid

import numpy as np

from . import component_ablation as components
from .dynamic.gt import array_digest
from .dynamic.runtime import (
    DEFAULT_LIMITS, OutputLock, atomic_json, bootstrap_manifest, choose_core,
    environment_config, file_info, immutable_json, import_native, numa_snapshot,
    require_local, require_unchanged, siblings, timed_query, timestamp, write_npz,
)
from .static_paper import Predicates, read_json, validate_result


REPO = Path(__file__).resolve().parents[1]
SCHEMA = "ema-strict-filter-comparison-v1"
POLICIES = ("strict", "c1_c2", "full")
CONTRASTS = (
    ("Marker_vs_strict", "c1_c2", "strict"),
    ("full_vs_strict", "full", "strict"),
    ("recovery_vs_Marker", "full", "c1_c2"),
)
SOURCES = (
    "exp_benchmark/strict_filter_ablation.py",
    "exp_benchmark/ablation/build_strict_native.py",
    "exp_benchmark/ablation/strict_native.patch",
)


def prepare(parent_path, native_path, output, query_count=1000):
    parent, parent_root = components.load_request(parent_path)
    if parent["profile"] != "full" or not 1 <= query_count <= parent["query_count"]:
        raise ValueError("Select the full original study and a valid original-query prefix")
    native_path = Path(native_path).resolve(strict=True)
    native = read_json(native_path)
    if (native["schema"] != "ema-strict-filter-native-v1"
            or native["base_provenance"] != parent["native_provenance"]
            or native["component_patch"]["sha256"] != parent["native"]["patch"]["sha256"]):
        raise ValueError("The strict extension must derive from this graph's component-native source")
    for key, name in (("builder", SOURCES[1]), ("strict_patch", SOURCES[2])):
        if native[key]["sha256"] != file_info(REPO / name)["sha256"]:
            raise RuntimeError(f"The native build does not match the current {key}")
    record_path = parent_root / "graphs/cht_on/build.json"
    graph = read_json(record_path)
    if not graph["cht"] or graph["native_sha256"] != parent["native"]["modules"]["timing"]["sha256"]:
        raise RuntimeError("Expected the unchanged CHT-on graph from the original study")
    file_info(graph["checkpoint"]["path"], graph["checkpoint"])
    root = Path(output).resolve()
    root.mkdir(parents=True)
    sources = {**parent["sources"], **{name: file_info(REPO / name) for name in SOURCES}}
    for name in sources:
        destination = root / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((REPO / name).read_bytes())
    immutable_json(root / "request.json", {
        "schema": SCHEMA, "created_at": timestamp(),
        "profile": "full" if query_count == parent["query_count"] else "pilot",
        "parent_request": file_info(Path(parent_path).resolve()),
        "native_provenance": file_info(native_path), "native": native,
        "graph_record": file_info(record_path), "checkpoint": graph["checkpoint"],
        "parent_audit": file_info(parent_root / "input-audit.json"),
        "query_count": query_count, "policies": list(POLICIES),
        "parameters": components.PARAMETERS, "limits": DEFAULT_LIMITS,
        "timing_order_seed": 20261102, "sources": sources,
        "control_plans": {
            cell["name"]: {
                policy: file_info(parent_root / "measurements" / cell["name"] / policy / "calibration.json")
                for policy in ("c1_c2", "full")}
            for cell in parent["cells"]},
        "entry_policy": native["entry_policy"],
        "strict_policy": native["strict_policy"],
        "comparison_policy": "All three policies are freshly paired on the same saved CHT-on graph; old QPS is not reused",
        "pilot_policy": "Original full graph and GT; only query/predicate/GT rows are prefixed",
    })
    print(root / "request.json", flush=True)


def load_request(path):
    path = Path(path).resolve(strict=True)
    request = read_json(path)
    if (request["schema"] != SCHEMA or request["policies"] != list(POLICIES)
            or request["parameters"] != components.PARAMETERS or request["limits"] != DEFAULT_LIMITS):
        raise ValueError("The frozen strict-filter protocol differs")
    for key in ("parent_request", "native_provenance", "graph_record", "checkpoint", "parent_audit"):
        require_unchanged(request[key])
    parent, _ = components.load_request(request["parent_request"]["path"])
    for name, info in request["sources"].items():
        if file_info(REPO / name)["sha256"] != info["sha256"]:
            raise RuntimeError(f"Frozen comparison source changed: {name}")
    for info in request["native"]["modules"].values():
        file_info(info["path"], info)
    for plans in request["control_plans"].values():
        for info in plans.values():
            file_info(info["path"], info)
    return request, parent, path.parent


def load_inputs(request, parent, root):
    data = components.load_data({**parent, "query_count": request["query_count"]}, root)
    vectors, queries, attributes, _, cells = data
    audits = []
    original = {cell["cell"]: cell for cell in read_json(request["parent_audit"]["path"])["cells"]}
    for name, cell in cells.items():
        if request["profile"] == "full":
            audits.append(original[name])
        else:
            counts = cell["predicates"].counts(attributes)
            audits.append({
                "cell": name, "actual_selectivity": float(counts.mean() / parent["N"]),
                "ground_truth_sha256": array_digest(cell["truth"]),
            })
    bootstrap_manifest(root / "input-audit.json", {
        "profile": request["profile"], "N": parent["N"], "query_count": len(queries),
        "cells": audits, "original_audit_sha256": request["parent_audit"]["sha256"],
        "ground_truth_policy": "Unmodified original GT query prefix on the full original base",
    })
    return vectors, queries, attributes, cells


def load_index(request, parent, diagnostic=False):
    info = request["native"]["modules"]["diagnostic" if diagnostic else "timing"]
    native, _ = import_native(Path(info["path"]).parent, info["sha256"])
    index = native.Index(space="ip", dim=parent["dimension"])
    if bool(index.get_ablation_stats()["diagnostic_build"]) != diagnostic:
        raise RuntimeError("Timing and diagnostic binaries were interchanged")
    file_info(request["checkpoint"]["path"], request["checkpoint"])
    index.load_index(request["checkpoint"]["path"])
    index.generateAttrIndexes()
    index.set_num_threads(1)
    index.set_ablation_strict_filter(False)
    return index


def configure(index, policy, ef):
    components.configure(index, "c1" if policy == "strict" else policy, ef)
    index.set_ablation_strict_filter(policy == "strict")


def execute(index, policy, ef, queries, cell, vectors, attributes):
    configure(index, policy, ef)
    started = time.perf_counter()
    try:
        result = index.hybrid_knn_query(
            queries, cell["predicates"].raw, k=components.PARAMETERS["k"], num_threads=1)
    except RuntimeError as error:
        if "Cannot return the results in a contiguous 2D array" not in str(error):
            raise
        return None, {"error": "fewer_than_k_results", "message": str(error)}
    wall = time.perf_counter() - started
    validated = validate_result(result, cell["truth"], vectors, queries, attributes,
                                cell["predicates"], "ip", components.PARAMETERS["k"])
    return result, {**validated, "calibration_wall_s": wall}


def verify_control(point, result, reference, query_count, original_query_count):
    if reference is None:
        return "no_original_sample_at_this_ef"
    if "error" in reference:
        if query_count == original_query_count and "error" not in point:
            raise RuntimeError("A formerly incomplete control batch changed under the new native")
        return "matched_incomplete_batch" if "error" in point else "prefix_of_incomplete_batch"
    if result is None:
        raise RuntimeError("A formerly complete control batch now fails")
    file_info(reference["raw"]["path"], reference["raw"])
    with np.load(reference["raw"]["path"], allow_pickle=False) as saved:
        for position, key in enumerate(("labels", "distances")):
            if array_digest(result[position]) != array_digest(saved[key][:query_count]):
                raise RuntimeError("Default-off strict patch changed an original control result")
    return "bitwise_original_labels_and_distances"


def calibrated_point(index, policy, ef, queries, cell, vectors, attributes, raw, references, original_nq):
    cached = raw / f"ef-{ef}.json"
    if cached.exists():
        point = read_json(cached)
        if "raw" in point:
            file_info(point["raw"]["path"], point["raw"])
        return point
    result, point = execute(index, policy, ef, queries, cell, vectors, attributes)
    if policy != "strict":
        point["original_control_check"] = verify_control(
            point, result, references.get(ef), len(queries), original_nq)
    if result is not None:
        path = raw / f"ef-{ef}-{uuid.uuid4().hex}.npz"
        point["raw"] = write_npz(path, labels=result[0], distances=result[1])
    immutable_json(cached, point)
    print(f"{raw.parent.name} {policy} ef={ef}: {point.get('recall', point.get('error'))}", flush=True)
    return point


def paired_timing(index, queries, predicates, plans, directory, request, core, loads):
    jobs = [(policy, ef) for policy, plan in plans.items() for ef in plan["timing_efs"]]
    attempt = directory / f"timing-{uuid.uuid4().hex}.json"
    rows, accepted = [], []
    for round_id in range(request["limits"]["max_rounds_per_core"]):
        current = []
        order = np.random.default_rng(request["timing_order_seed"] + round_id).permutation(len(jobs))
        for position in order:
            policy, ef = jobs[position]
            point = next(p for p in plans[policy]["points"] if p["ef"] == ef)
            configure(index, policy, ef)
            arguments = {"k": components.PARAMETERS["k"], "num_threads": 1}
            index.hybrid_knn_query(queries[:3], predicates.raw[:3], **arguments)
            repeats = max(1, min(100, math.ceil(.25 / point["calibration_wall_s"])))
            def call():
                for _ in range(repeats):
                    result = index.hybrid_knn_query(queries, predicates.raw, **arguments)
                return result
            result, sample = timed_query(call, core, siblings(core), request["limits"])
            if (array_digest(result[0]) != point["labels_sha256"]
                    or array_digest(result[1]) != point["distances_sha256"]):
                raise RuntimeError("Timed results changed from the frozen calibration")
            current.append({
                **sample, "variant": policy, "ef": ef, "round": round_id,
                "batch_repeats": repeats, "qps": len(queries) * repeats / sample["wall_s"],
                "labels_sha256": point["labels_sha256"], "distances_sha256": point["distances_sha256"],
            })
        clean = all(row["clean"] for row in current)
        for row in current:
            row["accepted_complete_round"] = clean
        rows.extend(current)
        if clean:
            accepted.append(round_id)
        atomic_json(attempt, {"core": core, "core_selection_loads": loads,
                              "accepted_rounds": accepted, "samples": rows})
        print(f"{directory.name}: round {round_id}, clean={clean}, accepted={len(accepted)}", flush=True)
        if len(accepted) >= request["limits"]["clean_rounds"]:
            break
    return attempt, accepted


def measure(request, parent, root):
    index = load_index(request, parent)
    vectors, queries, attributes, cells = load_inputs(request, parent, root)
    environment_config(0, components.PARAMETERS["build_threads"], requested_cores=[8, 16, 24])
    for name, cell in cells.items():
        directory = root / "measurements" / name
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "complete.json").exists():
            continue
        candidates = [8, 16, 24]
        core, loads = choose_core(candidates)
        os.sched_setaffinity(0, {core})
        require_local(numa_snapshot(), 0)
        components.progress(root, "calibrating", status="running", cell=name, core=core)
        plans = {}
        for policy in POLICIES:
            raw = directory / policy
            raw.mkdir(exist_ok=True)
            plan_path = raw / "calibration.json"
            references = {} if policy == "strict" else {
                p["ef"]: p for p in read_json(request["control_plans"][name][policy]["path"])["points"]}
            if plan_path.exists():
                plans[policy] = read_json(plan_path)
                for point in plans[policy]["points"]:
                    if "raw" in point:
                        file_info(point["raw"]["path"], point["raw"])
            else:
                plans[policy] = components.calibration_plan(
                    lambda ef: calibrated_point(index, policy, ef, queries, cell, vectors, attributes,
                                                raw, references, parent["query_count"]))
                immutable_json(plan_path, plans[policy])
        for core_attempt in range(min(len(candidates), request["limits"]["max_cores"])):
            components.progress(root, "paired_timing", status="running", cell=name, core=core)
            attempt, accepted = paired_timing(
                index, queries, cell["predicates"], plans, directory, request, core, loads)
            candidates.remove(core)
            if (len(accepted) >= request["limits"]["clean_rounds"] or not candidates
                    or core_attempt + 1 >= request["limits"]["max_cores"]):
                break
            core, loads = choose_core(candidates)
            os.sched_setaffinity(0, {core})
        if len(accepted) < request["limits"]["clean_rounds"]:
            components.progress(root, "timing_needs_retry", status="needs_timing",
                                cell=name, attempt=str(attempt))
            raise components.TimingIncomplete("Complete clean paired rounds were insufficient")
        immutable_json(directory / "complete.json", {
            "cell": name, "plans": plans, "timing": file_info(attempt),
            "accepted_rounds": accepted, "core": core,
        })
        os.sched_setaffinity(0, set(range(32)))
    components.progress(root, "measurement_complete", status="complete")


def counters(index, policy):
    stats = index.get_ablation_stats()
    if (not stats["diagnostic_build"] or stats["strict_in_filtering"] != (policy == "strict")
            or any(stats[key] < 0 for key in components.COUNTERS)):
        raise RuntimeError("Diagnostic flavor, policy or counters differ")
    return {key: stats[key] for key in components.COUNTERS}


def diagnose(request, parent, root):
    index = load_index(request, parent, diagnostic=True)
    vectors, queries, attributes, cells = load_inputs(request, parent, root)
    environment_config(0, components.PARAMETERS["build_threads"], requested_cores=[8, 16, 24])
    for name, cell in cells.items():
        directory = root / "measurements" / name
        if (directory / "diagnostics.json").exists():
            continue
        complete = read_json(directory / "complete.json")
        os.sched_setaffinity(0, {complete["core"]})
        rows, failures = [], []
        for policy, plan in complete["plans"].items():
            for ef in plan["timing_efs"]:
                point = next(p for p in plan["points"] if p["ef"] == ef)
                index.reset_ft_stats()
                _, validated = execute(index, policy, ef, queries, cell, vectors, attributes)
                if any(validated.get(key) != point[key] for key in ("labels_sha256", "distances_sha256")):
                    raise RuntimeError("Diagnostic output differs from timed output")
                work = counters(index, policy)
                rows.append({
                    "variant": policy, "ef": ef, "recall": point["recall"],
                    "counters": work, "per_query": {key: value / len(queries) for key, value in work.items()},
                })
            failed_efs = [p["ef"] for p in plan["points"] if "error" in p]
            if plan["status"] == "unreached" and failed_efs:
                ef = max(failed_efs)
                for q in range(len(queries)):
                    one = {"truth": cell["truth"][q:q + 1],
                           "predicates": Predicates.parse(cell["predicates"].raw[q:q + 1], [0, 1], 1)}
                    index.reset_ft_stats()
                    _, observation = execute(index, policy, ef, queries[q:q + 1], one, vectors, attributes)
                    if "error" in observation:
                        failures.append({
                            "variant": policy, "ef": ef, "first_failed_query": q,
                            "examined_queries": q + 1, "counters": counters(index, policy),
                            "error": observation["error"],
                            "note": "First-failure witness only; not a failure-frequency or mean-recall estimate",
                        })
                        break
                else:
                    raise RuntimeError("An incomplete batch did not reproduce query by query")
        immutable_json(directory / "diagnostics.json", {
            "native_sha256": request["native"]["modules"]["diagnostic"]["sha256"],
            "rows": rows, "failures": failures,
        })
        os.sched_setaffinity(0, set(range(32)))
    components.progress(root, "diagnostics_complete", status="complete")


def report(request, parent, root):
    audits = {c["cell"]: c for c in read_json(root / "input-audit.json")["cells"]}
    summary, curves, contrasts = [], [], []
    for cell in parent["cells"]:
        name = cell["name"]
        directory = root / "measurements" / name
        complete = read_json(directory / "complete.json")
        file_info(complete["timing"]["path"], complete["timing"])
        samples = read_json(complete["timing"]["path"])["samples"]
        accepted = [r for r in samples if r["accepted_complete_round"]]
        diag = read_json(directory / "diagnostics.json")["rows"]
        per_round = {}
        for policy, plan in complete["plans"].items():
            qps = {ef: statistics.median(r["qps"] for r in accepted
                                        if r["variant"] == policy and r["ef"] == ef)
                   for ef in plan["timing_efs"]}
            values = components.target_values(plan, qps)
            work_ef = plan["selected_ef"] or max(plan["timing_efs"], default=None)
            work = next((r for r in diag if r["variant"] == policy and r["ef"] == work_ef), {})
            valid = [p for p in plan["points"] if "recall" in p]
            summary.append({
                "dataset": parent["dataset"], "profile": request["profile"], "cell": name,
                "nominal_selectivity": cell["nominal_selectivity"],
                "actual_selectivity": audits[name]["actual_selectivity"], "variant": policy,
                "status": plan["status"], "selected_ef": plan["selected_ef"], **values,
                "max_sampled_recall": max((p["recall"] for p in valid), default=None),
                "complete_batch_observed": bool(valid),
                "diagnostic_ef": work_ef, "diagnostic_recall": work.get("recall"),
                **{key: work.get("per_query", {}).get(key) for key in components.COUNTERS},
            })
            per_round[policy] = {
                round_id: components.target_values(plan, {
                    r["ef"]: r["qps"] for r in accepted
                    if r["variant"] == policy and r["round"] == round_id})
                for round_id in complete["accepted_rounds"]
            }
            for point in plan["points"]:
                if point["ef"] in qps:
                    curves.append({
                        "cell": name, "nominal_selectivity": cell["nominal_selectivity"],
                        "variant": policy, "ef": point["ef"], "recall": point["recall"],
                        "qps": qps[point["ef"]],
                    })
        for component, numerator, denominator in CONTRASTS:
            ratios, threshold = [], []
            for round_id in complete["accepted_rounds"]:
                a, b = per_round[numerator][round_id], per_round[denominator][round_id]
                if a["qps95"] is not None and b["qps95"] is not None:
                    ratios.append(a["qps95"] / b["qps95"])
                if a["observed_qps"] is not None and b["observed_qps"] is not None:
                    threshold.append(a["observed_qps"] / b["observed_qps"])
            contrasts.append({
                "cell": name, "nominal_selectivity": cell["nominal_selectivity"],
                "component": component, "numerator": numerator, "denominator": denominator,
                "paired_qps95_speedup": statistics.median(ratios) if ratios else None,
                "paired_qps95_speedup_min": min(ratios) if ratios else None,
                "paired_qps95_speedup_max": max(ratios) if ratios else None,
                "paired_at_least95_speedup": statistics.median(threshold) if threshold else None,
                "paired_rounds": len(complete["accepted_rounds"]),
            })
    for name, rows in (("summary", summary), ("curves", curves), ("contrasts", contrasts)):
        components.write_csv(root / f"{name}.csv", rows)
    atomic_json(root / "results.json", {
        "schema": SCHEMA, "profile": request["profile"], "status": "complete",
        "summary": summary, "contrasts": contrasts, "entry_policy": request["entry_policy"],
        "notes": [
            "All policies use the same original graph and are freshly timed together.",
            "The common initial seed is a bootstrap exception; newly reached neighbors must satisfy the exact predicate in strict mode.",
            "Incomplete batches have no fabricated recall or QPS; diagnostic failure witnesses do not estimate failure rates.",
            "Per-round ranges are not confidence intervals. Work counters do not identify a hardware bottleneck.",
            "Diagnostic work at an unreached target is explicitly labeled with its actual ef and recall.",
        ],
    })
    components.progress(root, "complete", status="complete", output=str(root))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("prepare")
    setup.add_argument("--parent-request", required=True)
    setup.add_argument("--native-provenance", required=True)
    setup.add_argument("--output", required=True)
    setup.add_argument("--query-count", type=int, default=1000)
    for name in ("measure", "diagnose", "report", "run"):
        command = sub.add_parser(name)
        command.add_argument("--request", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.parent_request, args.native_provenance, args.output, args.query_count)
        return
    request, parent, root = load_request(args.request)
    if args.command == "run":
        for phase in ("measure", "diagnose", "report"):
            with (root / f"{phase}-{uuid.uuid4().hex}.log").open("x") as stream:
                subprocess.run([sys.executable, "-B", "-m", "exp_benchmark.strict_filter_ablation",
                                phase, "--request", str(Path(args.request).resolve())],
                               cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, check=True)
        return
    with OutputLock(root):
        try:
            {"measure": measure, "diagnose": diagnose, "report": report}[args.command](request, parent, root)
        except components.TimingIncomplete:
            raise
        except Exception as error:
            components.progress(root, args.command, status="failed", error=str(error))
            raise


if __name__ == "__main__":
    main()
