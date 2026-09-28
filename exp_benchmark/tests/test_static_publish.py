import copy
import unittest
from unittest import mock

from exp_benchmark.dynamic.runtime import DEFAULT_LIMITS
from exp_benchmark.static_publish import checked_timing, check_summary, expected_keys


class StaticPublicationTests(unittest.TestCase):
    def evidence(self):
        points = [
            {"ef": 10, "recall": 0.94, "labels_sha256": "labels10", "distances_sha256": "distances10"},
            {"ef": 11, "recall": 0.96, "labels_sha256": "labels11", "distances_sha256": "distances11"},
        ]
        rounds = []
        for repeat in range(7):
            order = points[repeat % 2:] + points[:repeat % 2]
            samples = []
            for point in order:
                qps = (100 if point["ef"] == 10 else 80) + repeat
                samples.append({
                    **point, "qps": qps, "wall_s": 1000 / qps,
                    "clean": True, "rejection_reasons": [],
                })
            rounds.append(samples)
        limits = {**DEFAULT_LIMITS, "frequency_khz": [3_267_000, 3_333_000]}
        timing = {"attempts": [{"core": 24, "rounds": rounds, "accepted_rounds": list(range(7))}]}
        return points, timing, limits

    def validate(self, points, timing, limits, targets=(0.95,)):
        with mock.patch("exp_benchmark.static_publish.noise_reasons",
                        side_effect=lambda sample, *_: sample["rejection_reasons"]):
            return checked_timing(points, timing, limits, 1000, targets)

    def test_raw_round_medians_and_distinct_recall_targets(self):
        points, timing, limits = self.evidence()
        auxiliary, primary = self.validate(points, timing, limits, (0.90, 0.95))
        self.assertIsNone(auxiliary["qps_at_target"])
        self.assertEqual(auxiliary["reported_qps"], 103)
        self.assertAlmostEqual(primary["qps_at_target"], 93)
        self.assertEqual(primary["selected_core"], 24)
        self.assertEqual(primary["accepted_rounds"], list(range(7)))
        changed = copy.deepcopy(primary)
        changed["reported_qps"] += 1
        with self.assertRaises(ValueError):
            check_summary(changed, primary)

    def test_modified_qps_or_result_hash_is_rejected(self):
        for field, value in (("qps", 999), ("labels_sha256", "different")):
            points, timing, limits = self.evidence()
            timing["attempts"][0]["rounds"][0][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(points, timing, limits)

    def test_extra_rounds_or_early_core_abandonment_are_rejected(self):
        points, timing, limits = self.evidence()
        timing["attempts"][0]["rounds"].append(copy.deepcopy(timing["attempts"][0]["rounds"][1]))
        timing["attempts"][0]["accepted_rounds"].append(7)
        with self.assertRaisesRegex(ValueError, "first seven"):
            self.validate(points, timing, limits)
        points, timing, limits = self.evidence()
        early = copy.deepcopy(timing["attempts"][0])
        early["core"], early["accepted_rounds"] = 8, []
        early["rounds"] = early["rounds"][:1]
        for sample in early["rounds"][0]:
            sample["clean"], sample["rejection_reasons"] = False, ["busy"]
        timing["attempts"].insert(0, early)
        with self.assertRaisesRegex(ValueError, "round budget"):
            self.validate(points, timing, limits)

    def test_expected_grid_includes_all_primary_and_auxiliary_points(self):
        keys = expected_keys()
        self.assertEqual(len(keys), 107)
        self.assertEqual(sum(key[0] == 5 for key in keys), 72)
        self.assertEqual(sum(key[-1] == 0.90 for key in keys), 6)
        self.assertIn((6, "c", "Wiki15.4M", 0.23, 0.95), keys)


if __name__ == "__main__":
    unittest.main()
