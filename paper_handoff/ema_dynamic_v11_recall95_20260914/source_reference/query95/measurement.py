"""Recall-only ef calibration and CPU-clean paired timing; never fastest-sample selection."""

import copy
import math
import os
import statistics

import numpy as np

from exp_benchmark.aggregate import qps_at_recall
from .gt import array_digest
from .protocol import TARGET_RECALL
from .runtime import atomic_json, choose_core, numa_snapshot, require_local, timed_query


def bracket(state):
    return state["points"][-2:] if len(state["points"]) > 1 else state["points"]


def validate_calibration(state, k=10):
    points = state["points"]
    if (state.get("target_recall") != TARGET_RECALL
            or not points or [p["ef"] for p in points] != list(range(k, k + len(points)))
            or any(not 0 <= p["recall"] <= 1 for p in points)
            or any(p["recall"] >= TARGET_RECALL for p in points[:-1])
            or points[-1]["recall"] < TARGET_RECALL):
        raise RuntimeError("Calibration is not an ascending first-crossing recall95 integer-ef trace")


def summarize_curve(curve, target=TARGET_RECALL):
    if target != TARGET_RECALL:
        raise ValueError("Canonical dynamic summaries require target_recall=0.95; never relabel recall90")
    if not curve or any(not 0 <= point[1] <= 1 for point in curve) or curve[-1][1] < target:
        raise ValueError("No qualifying observed recall")
    if (any(point[1] >= target for point in curve[:-1])
            or any(right[0] != left[0] + 1 for left, right in zip(curve, curve[1:]))
            or any(not math.isfinite(point[2]) or point[2] <= 0 for point in curve)):
        raise ValueError("No supported first-crossing recall bracket")
    above = len(curve) == 1 and curve[0][1] > target
    value = qps_at_recall(curve, target)
    if value is None:
        raise ValueError("No supported recall bracket")
    return {
        "target_recall": TARGET_RECALL, "qps95": None if above else float(value),
        "observed_qps": float(curve[-1][2]), "selected_ef": int(curve[-1][0]),
        "selected_recall": float(curve[-1][1]), "actual_curve": curve,
        "recall_status": ">=95%; minimum legal ef already above target" if above else "target95",
        "method": ("observed_at_minimum_ef_NO_extrapolation" if above
                   else "observed_exact_target" if len(curve) == 1 else "adjacent_integer_ef_linear_interpolation"),
    }


def calibrate(index, query, validate, k, max_ef, directory, name, reference=None):
    if reference is not None:
        validate_calibration(reference, k)
    points, all_labels, all_distances = [], [], []
    efs = [p["ef"] for p in reference["points"]] if reference is not None else range(k, max_ef + 1)
    for ef in efs:
        index.set_ef(ef)
        labels, distances = query()
        point = {"ef": ef, "recall": validate(labels, distances),
                 "labels_sha256": array_digest(labels), "distances_sha256": array_digest(distances)}
        if reference is not None and any(point[key] != reference["points"][len(points)][key] for key in point):
            raise RuntimeError("Query labels/distances/recall differ from the frozen same-graph reference")
        points.append(point)
        all_labels.append(labels)
        all_distances.append(distances)
        atomic_json(directory / f"{name}-calibration-progress.json", {
            "target_recall": TARGET_RECALL, "points": points,
        })
        if reference is None and point["recall"] >= TARGET_RECALL:
            break
    if not points or points[-1]["recall"] < TARGET_RECALL:
        raise RuntimeError("No ef reached95% before --max-query-ef; no timing fallback")
    state = {"name": name, "points": points, "target_recall": TARGET_RECALL,
             "selection": "first qualifying integer ef, ascending from k, recall ONLY",
             "initial_static_dynamic_identical": reference is not None}
    validate_calibration(state, k)
    return state, np.stack(all_labels), np.stack(all_distances)


class FrozenQuery:
    def __init__(self, runner, truths):
        self.runner, self.truths = runner, truths
        self.queries, self.predicates, self.k = runner.data.queries, runner.data.predicates, runner.data.k
        self.query_sha = array_digest(self.queries)
        self.gt_shas = {name: array_digest(truth) for name, truth in truths.items()}

    def validate(self, owner, point, result):
        labels, distances = result
        recall = self.runner.validate_result(owner, labels, distances, self.truths[owner])
        if (recall != point["recall"] or array_digest(labels) != point["labels_sha256"]
                or array_digest(distances) != point["distances_sha256"]):
            raise RuntimeError(f"Frozen results changed: {owner} ef={point['ef']}")

    def __call__(self, owner, point):
        result = self.runner.query(owner)
        self.validate(owner, point, result)
        return result


def paired_timing(indexes, states, query, environment, limits, progress, *,
                  sampler=timed_query, picker=choose_core, locality=True, pilot=False):
    for owner in ("initial", "current"):
        validate_calibration(states[owner], query.k)
    cases = {f"{owner}_ef{p['ef']}": (owner, p) for owner in ("initial", "current")
             for p in bracket(states[owner])}
    names = list(cases)
    report = {"status": "timing", "attempts": [], "accepted_attempt": None,
              "limits": copy.deepcopy(limits), "numa_before": numa_snapshot(), "summary": None,
              "official_timing": not pilot, "target_recall": TARGET_RECALL,
              "qps95_change_fraction": None}
    if locality:
        require_local(report["numa_before"], environment["node"])
    remaining = list(environment["query_candidates"])
    try:
        for attempt_number in range(1 if pilot else min(limits["max_cores"], len(remaining))):
            os.sched_setaffinity(0, environment["update_cpus"])
            core, loads = (remaining[0], {}) if pilot else picker(remaining)
            remaining.remove(core)
            os.sched_setaffinity(0, {core})
            attempt = {"number": attempt_number, "core": core, "selection_pair_loads": loads,
                       "samples": {name: [] for name in names}, "accepted_rounds": []}
            report["attempts"].append(attempt)
            progress(report)
            for owner, point in cases.values():
                indexes[owner].set_ef(point["ef"])
                for _ in range(2):
                    query(owner, point)
            for repeat in range(1 if pilot else limits["max_rounds_per_core"]):
                order = names[repeat % len(names):] + names[:repeat % len(names)]
                clean = True
                for case in order:
                    owner, point = cases[case]
                    index = indexes[owner]
                    index.set_ef(point["ef"])
                    query(owner, point)
                    result, sample = sampler(
                        lambda: index.hybrid_knn_query_dnf(
                            query.queries, query.predicates, k=query.k, num_threads=1),
                        core, environment["siblings"][str(core)], limits,
                    )
                    query.validate(owner, point, result)
                    sample.update(
                        repeat=repeat, order=order, case=case, qps=len(query.queries) / sample["wall_s"],
                        labels_sha256=array_digest(result[0]), distances_sha256=array_digest(result[1]),
                        ef=point["ef"], recall=point["recall"],
                        query_payload_sha256=query.query_sha, ground_truth_sha256=query.gt_shas[owner],
                    )
                    attempt["samples"][case].append(sample)
                    clean = clean and sample["clean"]
                    progress(report)
                if clean and not pilot:
                    attempt["accepted_rounds"].append(repeat)
                progress(report)
                if len(attempt["accepted_rounds"]) == limits["clean_rounds"]:
                    break
            attempt["clean"] = len(attempt["accepted_rounds"]) == limits["clean_rounds"]
            if attempt["clean"] or pilot:
                accepted = [0] if pilot else attempt["accepted_rounds"]
                medians = {name: statistics.median(attempt["samples"][name][r]["qps"] for r in accepted)
                           for name in names}
                attempt["median_qps"] = medians
                report["summary"] = {
                    owner: summarize_curve([(p["ef"], p["recall"], medians[f"{owner}_ef{p['ef']}"])
                                            for p in bracket(states[owner])])
                    for owner in ("initial", "current")
                }
                if pilot:
                    for summary in report["summary"].values():
                        summary.update(qps95=None, method="NONOFFICIAL_pilot_observation_NO_QPS95")
                else:
                    report["accepted_attempt"] = attempt_number
                initial, current = (report["summary"][owner]["qps95"] for owner in ("initial", "current"))
                report["qps95_change_fraction"] = current / initial - 1 if initial and current else None
                progress(report)
                break
            progress(report)
    finally:
        os.sched_setaffinity(0, environment["update_cpus"])
    report["numa_after"] = numa_snapshot()
    if locality:
        require_local(report["numa_after"], environment["node"])
    report["status"] = "complete" if pilot or report["accepted_attempt"] is not None else "noise_contaminated"
    progress(report)
    return report
