import copy
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from exp_benchmark.dynamic import remeasure
from exp_benchmark.dynamic.protocol import OPERATIONS, QUERY_CONFIG
from exp_benchmark.dynamic.runtime import file_info, mutation_timing


def source_config():
    return {
        "protocol": remeasure.SOURCE_PROTOCOL, "mode": "official",
        "capacity": 10_000_000, "initial": 5_000_000, "step": 1_000_000,
        "stage_progress": list(range(0, 5_000_001, 1_000_000)),
        "query": {**QUERY_CONFIG, "target_recall": .90}, "index_format": 10,
        "numeric_marker_version": 2, "marker_owner_version": 2,
        "distance_order_version": 1, "pickle_state_version": 4,
        "candidate_order": "vector-distance-only-v1",
        "operations": list(OPERATIONS), "slot_reuse": False,
    }


def point(ef, recall):
    return {"ef": ef, "recall": recall,
            "labels_sha256": f"labels-{ef}", "distances_sha256": f"distances-{ef}"}


class RecallRevisionContractTests(unittest.TestCase):
    def test_source_requires_unchanged_nonrecall_parameters(self):
        config = source_config()
        remeasure.require_source_config(config)
        for key, value in (("ft_routing_min_deg", 16), ("target_recall", .95),
                           ("ft_bits_per_attribute", 256), ("k", 20)):
            changed = copy.deepcopy(config)
            changed["query"][key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                remeasure.require_source_config(changed)

    def test_source_rejects_legacy_marker_graphs_and_nonofficial_traces(self):
        for key, value in (("index_format", 8), ("marker_owner_version", 1),
                           ("distance_order_version", 0), ("mode", "pilot"),
                           ("capacity", 1000), ("slot_reuse", True)):
            config = source_config()
            config[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                remeasure.require_source_config(config)

    def test_calibration_keeps_original_query_prefix(self):
        source = {"points": [point(10, .89), point(11, .91)]}
        current = {"points": [point(10, .89), point(11, .91), point(12, .96)]}
        source["points"][-1]["untimed_ft_work_counters"] = {"neighbors": 123}
        remeasure.verify_prefix(current, source)
        for key in ("ef", "recall", "labels_sha256", "distances_sha256"):
            changed = copy.deepcopy(current)
            changed["points"][0][key] = "changed"
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                remeasure.verify_prefix(changed, source)
        with self.assertRaises(RuntimeError):
            remeasure.verify_prefix({"points": []}, source)

    def test_source_copy_preserves_old_code_after_live_source_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "source.py"
            copied = Path(directory) / "snapshot.py"
            original.write_text("original")
            copied.write_text("original")
            before, snapshot = file_info(original), file_info(copied)
            inventory = {str(original): {"original": before, "copy": snapshot}}
            original.write_text("revised")
            self.assertEqual(remeasure.verify_source_copy(before, inventory), snapshot)
            with self.assertRaises(RuntimeError):
                remeasure.verify_source_copy(before, None)
            with self.assertRaises(RuntimeError):
                remeasure.verify_source_copy(before, {})
            copied.write_text("changed snapshot")
            with self.assertRaises(RuntimeError):
                remeasure.verify_source_copy(before, inventory)

    def test_source_copy_must_match_executed_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "source.py"
            copied = Path(directory) / "snapshot.py"
            original.write_text("original")
            copied.write_text("different")
            before, snapshot = file_info(original), file_info(copied)
            inventory = {str(original): {"original": before, "copy": snapshot}}
            with self.assertRaises(RuntimeError):
                remeasure.verify_source_copy(before, inventory)

    def test_revision_preserves_mutation_cost_and_marks_query_only_reuse(self):
        stage = {"stage": 1, "stage_file": {"sha256": "stage"},
                 "files": {"results": {"sha256": "original-result"}}}
        original = {"mutation": {
            "wall_s": 60., "records": 1_000_000,
            "phases": {"delete": {"wall_s": 40.}, "add": {"wall_s": 20.}},
        }}
        timing = {"accepted_attempt": 0, "attempts": [
            {"core": 24, "accepted_rounds": list(range(7))}]}
        result = {"path": "/new/result.json"}
        with patch.object(remeasure, "paper_row", return_value={
                **mutation_timing(original["mutation"]), "qps95": 1000.}) as publisher:
            row = remeasure.revision_row(
                "point_update", stage, original, {}, result, timing, "native")
        self.assertEqual(row["maintenance_seconds"], 60.)
        self.assertEqual(row["maintenance_minutes"], 1.)
        self.assertEqual(row["delete_seconds"], 40.)
        self.assertEqual(row["add_seconds"], 20.)
        self.assertIsNone(row["delete_scrub_seconds"])
        self.assertIsNone(row["delete_non_scrub_seconds"])
        self.assertIsNone(row["delete_scrub_fraction"])
        self.assertIsNone(row["maintenance_scrub_fraction"])
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=remeasure.CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(row)
        exported = next(csv.DictReader(io.StringIO(stream.getvalue())))
        self.assertEqual(exported["delete_scrub_seconds"], "")
        self.assertEqual(exported["delete_non_scrub_seconds"], "")
        self.assertEqual(exported["delete_scrub_fraction"], "")
        self.assertEqual(row["operation_results"], stage["files"]["results"])
        self.assertEqual(row["results"], result)
        self.assertTrue(row["queries_retimed"])
        self.assertTrue(row["maintenance_reused"])
        self.assertFalse(row["mutations_executed"])
        self.assertNotIn("qps90", row)
        publisher.assert_called_once()

    def test_revision_preserves_recorded_scrub_cost_without_new_mutations(self):
        stage = {"stage": 1, "stage_file": {"sha256": "stage"},
                 "files": {"results": {"sha256": "original-result"}}}
        original = {"mutation": {
            "wall_s": 60., "records": 1_000_000,
            "phases": {"delete": {"wall_s": 40., "scrub_wall_s": 10.},
                       "add": {"wall_s": 20.}},
        }}
        before = copy.deepcopy(original)
        timing = {"accepted_attempt": 0, "attempts": [{"core": 24, "accepted_rounds": list(range(7))}]}
        with patch.object(remeasure, "paper_row", return_value=mutation_timing(original["mutation"])):
            row = remeasure.revision_row(
                "point_update", stage, original, {}, {"path": "/new/result.json"}, timing, "native")
        self.assertEqual(row["delete_scrub_seconds"], 10.)
        self.assertEqual(row["delete_non_scrub_seconds"], 30.)
        self.assertEqual(row["delete_scrub_fraction"], .25)
        self.assertAlmostEqual(row["maintenance_scrub_fraction"], 1 / 6)
        self.assertEqual(row["maintenance_seconds"], 60.)
        self.assertFalse(row["mutations_executed"])
        self.assertEqual(original, before)
        for name in ("delete_scrub_seconds", "delete_non_scrub_seconds",
                     "delete_scrub_fraction", "maintenance_scrub_fraction"):
            self.assertIn(name, remeasure.CSV_FIELDS)

    def test_mutation_timing_cannot_disappear_or_change(self):
        stage = {"stage": 1, "stage_file": {}, "files": {"results": {}}}
        timing = {"accepted_attempt": 0, "attempts": [{"core": 24, "accepted_rounds": list(range(7))}]}
        for mutation in (
            None,
            {"records": 1_000_000, "phases": {"add": {"wall_s": 10.}}},
            {"records": 2_000_000, "phases": {"add": {"wall_s": 60.}}},
            {"records": 1_000_000, "phases": {"delete": {"wall_s": 60.}}},
        ):
            with self.subTest(mutation=mutation):
                with patch.object(remeasure, "paper_row", return_value={"maintenance_wall_s": 60.}):
                    with self.assertRaises(RuntimeError):
                        remeasure.revision_row(
                            "insert", stage, {"mutation": mutation}, {}, {"path": "/new"}, timing, "native")

    def test_completed_stages_must_form_unique_canonical_prefix(self):
        for completed in (
            [{"operation": "delete", "stage": 0}],
            [{"operation": "insert", "stage": 1}],
            [{"operation": "insert", "stage": 0}, {"operation": "insert", "stage": 0}],
        ):
            with self.subTest(completed=completed), self.assertRaises(RuntimeError):
                remeasure.validate_completed(
                    {"schema": remeasure.SCHEMA, "completed": completed}, {}, {}, 0)

    def test_publish_rejects_incomplete_or_relabelled90_percent_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(RuntimeError):
                remeasure.publish(root, {"completed": []})
            entries = [{"operation": operation, "stage": number,
                        "row": {"recall": .9, "qps95": 1234.}}
                       for operation in OPERATIONS for number in range(6)]
            with self.assertRaises(RuntimeError):
                remeasure.publish(root, {"completed": entries})
            self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
