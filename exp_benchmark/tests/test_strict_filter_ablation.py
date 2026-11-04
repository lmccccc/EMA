import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from exp_benchmark import strict_filter_ablation as study


class StrictProtocolTests(unittest.TestCase):
    def test_every_policy_switch_resets_strict_gate(self):
        index = Mock()
        for policy in ("strict", "full", "strict", "c1_c2"):
            study.configure(index, policy, 64)
        self.assertEqual([call.args[0] for call in index.set_ablation_strict_filter.call_args_list],
                         [True, False, True, False])

    def test_incomplete_calibration_is_cached_without_fabricated_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary)
            error = {"error": "fewer_than_k_results", "message": "native incomplete batch"}
            with patch.object(study, "execute", return_value=(None, error)) as execute:
                first = study.calibrated_point(None, "strict", 10, [None], None, None, None,
                                               raw, {}, 1)
                second = study.calibrated_point(None, "strict", 10, [None], None, None, None,
                                                raw, {}, 1)
            self.assertEqual(first, second)
            self.assertNotIn("recall", second)
            self.assertNotIn("raw", second)
            self.assertEqual(list(raw.glob("*.npz")), [])
            execute.assert_called_once()

    def test_control_equivalence_checks_original_prefix_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            labels = np.array([[0, 1], [1, 0]], dtype=np.uint64)
            distances = np.array([[0, 1], [0, 1]], dtype=np.float32)
            identity = study.write_npz(Path(temporary) / "original.npz",
                                       labels=labels, distances=distances)
            reference = {"raw": identity}
            result = (labels[:1], distances[:1])
            self.assertEqual(study.verify_control({}, result, reference, 1, 2),
                             "bitwise_original_labels_and_distances")
            with self.assertRaisesRegex(RuntimeError, "changed an original control"):
                study.verify_control({}, (labels[1:], distances[:1]), reference, 1, 2)
            with self.assertRaisesRegex(RuntimeError, "now fails"):
                study.verify_control({"error": "incomplete"}, None, reference, 1, 2)
            incomplete = {"error": "incomplete"}
            self.assertEqual(study.verify_control({}, result, incomplete, 1, 2),
                             "prefix_of_incomplete_batch")
            with self.assertRaisesRegex(RuntimeError, "formerly incomplete"):
                study.verify_control({}, (labels, distances), incomplete, 2, 2)

    def test_resume_rejects_modified_raw_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary)
            result = (np.array([[0]], dtype=np.uint64), np.array([[0]], dtype=np.float32))
            point = {"recall": .95, "calibration_wall_s": .1}
            with patch.object(study, "execute", return_value=(result, point)):
                saved = study.calibrated_point(None, "strict", 10, [None], None, None, None,
                                               raw, {}, 1)
            Path(saved["raw"]["path"]).write_bytes(b"modified")
            with self.assertRaisesRegex(RuntimeError, "mismatch"):
                study.calibrated_point(None, "strict", 10, [None], None, None, None, raw, {}, 1)

    def test_report_does_not_compare_an_incomplete_policy_at_95_percent(self):
        valid = study.components.calibration_plan(lambda ef: {"recall": .95}, maximum=10)
        failed = study.components.calibration_plan(
            lambda ef: {"error": "fewer_than_k_results"}, maximum=10)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "measurements/T20"
            directory.mkdir(parents=True)
            study.atomic_json(root / "input-audit.json", {
                "cells": [{"cell": "T20", "actual_selectivity": .20}]})
            timing = directory / "timing.json"
            study.atomic_json(timing, {"samples": [
                {"variant": policy, "ef": 10, "round": round_id, "qps": qps,
                 "accepted_complete_round": True}
                for round_id in (0, 1) for policy, qps in (("c1_c2", 100), ("full", 95))]})
            study.atomic_json(directory / "complete.json", {
                "plans": {"strict": failed, "c1_c2": valid, "full": valid},
                "timing": study.file_info(timing), "accepted_rounds": [0, 1]})
            study.atomic_json(directory / "diagnostics.json", {"rows": []})
            study.report({"profile": "pilot", "entry_policy": "common seed"}, {
                "dataset": "youtube1m", "cells": [{"name": "T20", "nominal_selectivity": .2}]}, root)
            result = study.read_json(root / "results.json")
            strict = next(row for row in result["summary"] if row["variant"] == "strict")
            self.assertIsNone(strict["qps95"])
            self.assertIsNone(strict["max_sampled_recall"])
            self.assertFalse(strict["complete_batch_observed"])
            contrasts = {row["component"]: row["paired_qps95_speedup"] for row in result["contrasts"]}
            self.assertIsNone(contrasts["Marker_vs_strict"])
            self.assertIsNone(contrasts["full_vs_strict"])
            self.assertAlmostEqual(contrasts["recovery_vs_Marker"], .95)


NATIVE_FIXTURE = r"""
import hashlib, json
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
import hashannlib
import numpy as np
root = Path(sys.argv[2])
vectors = np.arange(32, dtype=np.float32).reshape(-1, 1)
records = [[[0], [0]]] * 31 + [[[0], [0, 1]]]
levels = np.zeros(32, dtype=np.int32)
levels[0] = 1
index = hashannlib.Index(space="l2", dim=1)
index.init_index(max_elements=32, top_elements=1, M=2, ef_construction=64,
                 attr_type=[0, 1], ft_bits=128, max_cate_size=2,
                 edge_level_ft=True, random_seed=100)
index.initAttrMapping(records)
index.add_items(vectors, records, levels=levels, num_threads=1)
index.save_index(str(root / "graph.bin"))
neighbors = [index.get_neighbors(row).tolist() for row in range(32)]
assert 31 not in neighbors[0], "Fixture must require a nonmatching bridge"
index.set_num_threads(1)
index.set_ef(64)
index.set_ef_top(1)
index.set_ft_routing_min_deg(16)
index.set_ft_routing_backfill_tail(False)
query = np.array([[31.125]], dtype=np.float32)
strict_available = hasattr(index, "set_ablation_strict_filter")

def run(policy, dnf, label):
    if strict_available:
        index.set_ablation_strict_filter(policy == "strict")
    index.set_ft_flag(policy in ("marker", "full"))
    index.set_ft_routing_flag(policy == "full")
    index.reset_ft_stats()
    raw = [[[[0, 0], [label]], [[], [label]]]] if dnf else [[[0, 0], [label]]]
    method = index.hybrid_knn_query_dnf if dnf else index.hybrid_knn_query
    try:
        result = method(query, raw, k=1, num_threads=1)
        observation = {"complete": True, "outputs": [value.tolist() for value in result]}
    except RuntimeError as error:
        if "Cannot return the results in a contiguous 2D array" not in str(error):
            raise
        observation = {"complete": False}
    return observation, dict(index.get_ablation_stats())

controls, strict = {}, {}
for deleted in (False, True):
    if deleted:
        index.mark_deleted(16)
    for dnf in (False, True):
        prefix = f"{deleted}-{dnf}"
        for policy in ("ordinary", "marker", "full"):
            for label in (0, 1):
                observation, _ = run(policy, dnf, label)
                controls[f"{prefix}-{policy}-{label}"] = observation
        assert controls[f"{prefix}-ordinary-1"]["outputs"][0] == [[31]]
        if strict_available:
            matching, stats = run("strict", dnf, 0)
            assert matching == controls[f"{prefix}-ordinary-0"]
            rejected, stats = run("strict", dnf, 1)
            assert not rejected["complete"]
            if stats["diagnostic_build"]:
                assert stats["base_expanded"] == 1, stats
                assert stats["base_distance_evaluations"] == 0, stats
                assert stats["predicate_checked"] == len(neighbors[0]), stats
            restored, _ = run("ordinary", dnf, 1)
            assert restored == controls[f"{prefix}-ordinary-1"]
            strict[prefix] = {"matching": matching, "rejected": rejected, "counters": stats}
(root / "result.json").write_text(json.dumps({
    "controls": controls, "strict": strict,
    "graph_sha256": hashlib.sha256((root / "graph.bin").read_bytes()).hexdigest()}))
"""


@unittest.skipUnless(os.environ.get("EMA_STRICT_TIMING_NATIVE")
                     and os.environ.get("EMA_STRICT_DIAGNOSTIC_NATIVE")
                     and os.environ.get("EMA_COMPONENT_TIMING_NATIVE"),
                     "Select the isolated strict pair and the original component timing extension")
class StrictNativeTests(unittest.TestCase):
    def fixture(self, directory):
        with tempfile.TemporaryDirectory() as temporary:
            completed = subprocess.run(
                [sys.executable, "-B", "-c", NATIVE_FIXTURE, directory, temporary],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stderr[-4000:])
            return json.loads((Path(temporary) / "result.json").read_text())

    def test_native_gate_and_default_off_equivalence(self):
        reference = self.fixture(os.environ["EMA_COMPONENT_TIMING_NATIVE"])
        timing = self.fixture(os.environ["EMA_STRICT_TIMING_NATIVE"])
        diagnostic = self.fixture(os.environ["EMA_STRICT_DIAGNOSTIC_NATIVE"])
        for current in (timing, diagnostic):
            self.assertEqual(current["graph_sha256"], reference["graph_sha256"])
            self.assertEqual(current["controls"], reference["controls"])
        for mode in timing["strict"]:
            self.assertEqual(timing["strict"][mode]["matching"], diagnostic["strict"][mode]["matching"])
            self.assertEqual(timing["strict"][mode]["rejected"], diagnostic["strict"][mode]["rejected"])


if __name__ == "__main__":
    unittest.main()
