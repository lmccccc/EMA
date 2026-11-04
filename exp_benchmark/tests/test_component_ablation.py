import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from exp_benchmark import component_ablation as study
from exp_benchmark.ablation.diagnose_failures import probe
from exp_benchmark.component_ablation import calibration_plan, exact_prefix_gt, target_values
from exp_benchmark.static_paper import Attributes, Predicates


class ComponentProtocolTests(unittest.TestCase):
    def test_calibration_refines_recall_without_using_throughput(self):
        calls = []
        def measure(ef):
            calls.append(ef)
            return {"recall": min(1, .90 + ef * .001), "qps": 1 / ef}
        plan = calibration_plan(measure, maximum=160)
        self.assertEqual(plan["selected_ef"], 50)
        self.assertEqual(plan["lower_ef"], 49)
        self.assertEqual(len(calls), len(set(calls)))
        values = target_values(plan, {ef: 1000 / ef for ef in plan["timing_efs"]})
        self.assertAlmostEqual(values["qps95"], 20)

    def test_minimum_above_target_is_not_labeled_qps95(self):
        plan = calibration_plan(lambda ef: {"recall": .98}, maximum=40)
        self.assertEqual(plan["status"], "minimum_above")
        values = target_values(plan, {ef: 500 for ef in plan["timing_efs"]})
        self.assertIsNone(values["qps95"])
        self.assertEqual(values["observed_recall"], .98)

    def test_unreachable_target_and_incomplete_batches_remain_explicit(self):
        for observation in ({"recall": .8}, {"error": "fewer_than_k_results"}):
            with self.subTest(observation=observation):
                plan = calibration_plan(lambda ef: observation, maximum=80)
                self.assertEqual(plan["status"], "unreached")
                self.assertIsNone(plan["selected_ef"])
                self.assertEqual(plan["points"][-1]["ef"], 80)
                self.assertIsNone(target_values(plan, {})["qps95"])

    def test_pilot_gt_keeps_raw_ip_and_inclusive_predicates(self):
        vectors = np.array([[10, 0], [1, 1], [2, 2], [100, 0]], dtype=np.float32)
        attrs = Attributes.from_records([[[i], [1]] for i in range(4)], [0, 1], 4)
        ids, cutoff = exact_prefix_gt(vectors, np.array([1, 1]), [[0, 2], [1]], attrs, 2)
        np.testing.assert_array_equal(ids, [0, 2])
        self.assertEqual(cutoff, 4)

    def test_report_keeps_incremental_losses_and_leave_one_out_separate(self):
        throughputs = {"baseline": 100, "c1": 200, "c1_c2": 150,
                       "full": 300, "without_c1": 250}
        plan = calibration_plan(lambda ef: {"recall": .95}, maximum=10)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "measurements/T10"
            directory.mkdir(parents=True)
            study.atomic_json(root / "input-audit.json", {
                "cells": [{"cell": "T10", "actual_selectivity": .099}]})
            rows = [
                {"variant": variant, "ef": 10, "round": round_id, "qps": qps * scale,
                 "accepted_complete_round": True}
                for round_id, scale in enumerate((1, 1.1, .9))
                for variant, qps in throughputs.items()
            ]
            timing = directory / "timing.json"
            study.atomic_json(timing, {"samples": rows})
            study.atomic_json(directory / "complete.json", {
                "plans": {variant: plan for variant in throughputs},
                "timing": study.file_info(timing), "accepted_rounds": [0, 1, 2]})
            study.atomic_json(directory / "diagnostics.json", {"rows": []})
            study.report({"dataset": "youtube1m", "profile": "pilot",
                          "cells": [{"name": "T10", "nominal_selectivity": .1}]}, root)
            values = {row["component"]: row["paired_qps95_speedup"]
                      for row in study.read_json(root / "results.json")["contrasts"]}
            self.assertAlmostEqual(values["C1_without_Marker"], 2)
            self.assertAlmostEqual(values["C2_without_recovery"], .75)
            self.assertAlmostEqual(values["C3_with_Marker"], 2)
            self.assertAlmostEqual(values["C1_in_full_method"], 1.2)
            self.assertAlmostEqual(values["full_vs_baseline"], 3)


class ComponentResumeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.request = {"limits": {**study.DEFAULT_LIMITS, "clean_rounds": 1,
                                  "max_rounds_per_core": 1}}
        self.result = (np.array([[0]], dtype=np.uint64), np.array([[.5]], dtype=np.float32))
        self.index = Mock()
        self.index.hybrid_knn_query.return_value = self.result
        self.cell = {"predicates": SimpleNamespace(raw=[[[0, 1], [1]]]),
                     "truth": self.result[0]}
        self.mock_function("VARIANTS", {"baseline": study.VARIANTS["baseline"]})
        self.mock_function("select_native")
        self.mock_function("load_data", return_value=(
            np.array([[.5]]), np.array([[1.]]), None, [], {"T1": self.cell}))
        self.mock_function("load_graphs", return_value={False: self.index})
        self.mock_function("environment_config")
        self.mock_function("require_local")
        self.mock_function("numa_snapshot")
        affinity = patch.object(study.os, "sched_setaffinity")
        affinity.start()
        self.addCleanup(affinity.stop)
        self.mock_function("choose_core", side_effect=lambda cpus: (cpus[0], {c: 0 for c in cpus}))
        self.mock_function("siblings", side_effect=lambda core: [core + 64])
        self.timer = self.mock_function("timed_query", side_effect=lambda call, *args: (
            call(), {"wall_s": .25, "clean": True}))
        self.validator = self.mock_function("validate_result", return_value={
            "recall": .95, "labels_sha256": study.array_digest(self.result[0]),
            "distances_sha256": study.array_digest(self.result[1])})
        self.calibrator = self.mock_function(
            "calibration_plan", side_effect=lambda measure: calibration_plan(measure, maximum=10))

    def mock_function(self, name, *args, **kwargs):
        replacement = patch.object(study, name, *args, **kwargs)
        result = replacement.start()
        self.addCleanup(replacement.stop)
        return result

    def interrupt_calibration(self):
        def interrupt(measure):
            measure(10)
            raise RuntimeError("interrupted before final calibration manifest")
        self.calibrator.side_effect = interrupt
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            study.measurement(self.request, self.root)
        self.calibrator.side_effect = lambda measure: calibration_plan(measure, maximum=10)
        return study.read_json(self.root / "measurements/T1/baseline/ef-10.json")

    def test_partial_resume_reuses_frozen_point_and_preserves_orphan(self):
        raw = self.root / "measurements/T1/baseline"
        raw.mkdir(parents=True)
        orphan = raw / "ef-10.npz"
        orphan.write_bytes(b"unfinished artifact")
        point = self.interrupt_calibration()
        study.measurement(self.request, self.root)
        self.validator.assert_called_once()
        study.file_info(point["raw"]["path"], point["raw"])
        self.assertEqual(orphan.read_bytes(), b"unfinished artifact")
        self.assertEqual(len(list(raw.glob("*.npz"))), 2)
        self.assertTrue((raw.parent / "complete.json").exists())

    def test_resume_rejects_modified_raw_calibration(self):
        point = self.interrupt_calibration()
        Path(point["raw"]["path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "mismatch"):
            study.measurement(self.request, self.root)
        self.validator.assert_called_once()

    def test_native_incomplete_batch_is_not_replaced_by_another_search(self):
        self.index.hybrid_knn_query.side_effect = RuntimeError(
            "Cannot return the results in a contiguous 2D array. Probably ef or M is too small")
        study.measurement(self.request, self.root)
        plan = study.read_json(self.root / "measurements/T1/baseline/calibration.json")
        self.assertEqual(plan["status"], "unreached")
        self.assertEqual(plan["points"][0]["error"], "fewer_than_k_results")
        self.assertNotIn("recall", plan["points"][0])
        self.assertEqual(plan["timing_efs"], [])
        self.validator.assert_not_called()

    def test_noise_retries_a_different_core_without_selecting_by_qps(self):
        self.timer.side_effect = lambda call, core, *args: (
            call(), {"wall_s": .25 if core == 8 else 1.0, "clean": core != 8})
        study.measurement(self.request, self.root)
        complete = study.read_json(self.root / "measurements/T1/complete.json")
        self.assertEqual(complete["core"], 16)
        attempts = list((self.root / "measurements/T1").glob("timing-*.json"))
        self.assertEqual(len(attempts), 2)
        self.assertEqual({study.read_json(path)["core"] for path in attempts}, {8, 16})

    def test_exhausted_cpu_budget_keeps_needs_timing_status(self):
        self.timer.side_effect = lambda call, *args: (
            call(), {"wall_s": .25, "clean": False})
        self.mock_function("load_request", return_value=(self.request, self.root))
        with patch.object(sys, "argv", ["component_ablation", "measure", "--request", "unused"]):
            with self.assertRaises(study.TimingIncomplete):
                study.main()
        status = study.read_json(self.root / "status.json")
        self.assertEqual(status["status"], "needs_timing")
        self.assertEqual(status["cell"], "T1")
        self.assertFalse((self.root / "measurements/T1/complete.json").exists())


class FailureWitnessTests(unittest.TestCase):
    def setUp(self):
        self.index = Mock()
        self.index.get_ablation_stats.return_value = {
            "diagnostic_build": True, **{key: 0 for key in study.COUNTERS},
            "base_expanded": 1, "empty_marker_rows": 1, "total_neighbors": 19}
        self.vectors = np.eye(16, dtype=np.float32)
        self.attributes = Attributes.from_records([[[0], [0]]] * 16, [0, 1], 16)
        self.cell = {
            "predicates": Predicates.parse([[[0, 0], [0]]], [0, 1], 1),
            "truth": np.arange(10, dtype=np.uint64).reshape(1, 10)}

    def observe(self):
        return probe(self.index, "c1_c2", 4096, 0, self.cell,
                     self.vectors, self.vectors[:1], self.attributes)

    def test_incomplete_witness_preserves_work_without_inventing_recall(self):
        self.index.hybrid_knn_query.side_effect = RuntimeError(
            "Cannot return the results in a contiguous 2D array")
        observation = self.observe()
        self.assertFalse(observation["complete"])
        self.assertNotIn("recall", observation)
        self.assertNotIn("labels", observation)
        self.assertEqual(observation["counters"]["base_expanded"], 1)
        self.index.reset_ft_stats.assert_called_once()

    def test_successful_witness_validates_original_ip_results(self):
        self.index.hybrid_knn_query.return_value = (
            self.cell["truth"], np.array([[0, *([1] * 9)]], dtype=np.float32))
        observation = self.observe()
        self.assertTrue(observation["complete"])
        self.assertEqual(observation["recall"], 1)
        self.assertEqual(observation["labels"], [list(range(10))])

    def test_unrelated_native_error_is_not_a_reachability_failure(self):
        self.index.hybrid_knn_query.side_effect = RuntimeError("unexpected native failure")
        with self.assertRaisesRegex(RuntimeError, "unexpected native failure"):
            self.observe()


NATIVE_FIXTURE = r"""
import hashlib, json
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
import hashannlib
import numpy as np
root = Path(sys.argv[2])
policy = sys.argv[3]
vectors = np.eye(128, dtype=np.float32)
records = [[[0], [row % 4]] for row in range(128)]
levels = np.zeros(128, dtype=np.int32)
levels[:4] = 1
index = hashannlib.Index(space="ip", dim=128)
index.init_index(max_elements=128, top_elements=4, M=12, ef_construction=64,
                 attr_type=[0, 1], ft_bits=128, max_cate_size=4,
                 edge_level_ft=True, random_seed=100)
if policy != "default":
    index.set_ablation_cht(policy == "on")
index.initAttrMapping(records)
index.add_items(vectors, records, levels=levels, num_threads=1)
if hasattr(index, "set_ablation_cht"):
    try:
        index.set_ablation_cht(False)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Construction policy changed after insertion")
index.save_index(str(root / "graph.bin"))
index.set_num_threads(1)
index.set_ef_top(1)
index.set_ef(128)
index.set_ft_routing_min_deg(16)
index.set_ft_routing_backfill_tail(False)
queries = vectors[[20, 54, 77]]
raw = [[[0, 0], [3]]] * len(queries)
outputs, counters = {}, {}
for mode, marker, recovery in (("off", False, False), ("marker", True, False),
                               ("full", True, True)):
    index.set_ft_flag(marker)
    index.set_ft_routing_flag(recovery)
    index.reset_ft_stats()
    result = index.hybrid_knn_query(queries, raw, k=5, num_threads=1)
    outputs[mode] = [value.tolist() for value in result]
    if hasattr(index, "get_ablation_stats"):
        counters[mode] = dict(index.get_ablation_stats())
payload = {"outputs": outputs, "degrees": index.get_degrees().tolist(),
           "neighbors": [index.get_neighbors(row).tolist() for row in range(128)],
           "graph_sha256": hashlib.sha256((root / "graph.bin").read_bytes()).hexdigest(),
           "counters": counters}
(root / "result.json").write_text(json.dumps(payload))
"""


@unittest.skipUnless(os.environ.get("EMA_COMPONENT_TIMING_NATIVE")
                     and os.environ.get("EMA_COMPONENT_DIAGNOSTIC_NATIVE"),
                     "Select the isolated component-ablation native pair")
class ComponentNativeTests(unittest.TestCase):
    def fixture(self, directory, policy):
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [sys.executable, "-B", "-c", NATIVE_FIXTURE, directory, temporary, policy],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr[-4000:])
            return json.loads((Path(temporary) / "result.json").read_text())

    def test_only_construction_policy_changes_the_paired_graph(self):
        directory = os.environ["EMA_COMPONENT_TIMING_NATIVE"]
        default = self.fixture(directory, "default")
        enabled = self.fixture(directory, "on")
        disabled = self.fixture(directory, "off")
        self.assertEqual(default["graph_sha256"], enabled["graph_sha256"])
        self.assertNotEqual(enabled["graph_sha256"], disabled["graph_sha256"])
        self.assertNotEqual(enabled["neighbors"], disabled["neighbors"])
        reference = os.environ.get("EMA_COMPONENT_REFERENCE_NATIVE")
        if reference:
            original = self.fixture(reference, "default")
            self.assertEqual(original["graph_sha256"], enabled["graph_sha256"])
            self.assertEqual(original["outputs"], enabled["outputs"])

    def test_diagnostics_preserve_outputs_and_separate_backfill_work(self):
        for policy in ("on", "off"):
            with self.subTest(policy=policy):
                timing = self.fixture(os.environ["EMA_COMPONENT_TIMING_NATIVE"], policy)
                diagnostic = self.fixture(os.environ["EMA_COMPONENT_DIAGNOSTIC_NATIVE"], policy)
                self.assertEqual(timing["graph_sha256"], diagnostic["graph_sha256"])
                self.assertEqual(timing["outputs"], diagnostic["outputs"])
                self.assertFalse(timing["counters"]["full"]["diagnostic_build"])
                self.assertEqual(diagnostic["counters"]["off"]["backfill_edges"], 0)
                self.assertEqual(diagnostic["counters"]["marker"]["backfill_edges"], 0)
                self.assertGreater(diagnostic["counters"]["full"]["backfill_edges"], 0)
                stats = diagnostic["counters"]["full"]
                self.assertGreater(stats["base_expanded"], 0)
                self.assertGreaterEqual(stats["backfill_edges"], stats["backfill_visits"])
                self.assertGreaterEqual(stats["base_distance_evaluations"], stats["backfill_visits"])


if __name__ == "__main__":
    unittest.main()
