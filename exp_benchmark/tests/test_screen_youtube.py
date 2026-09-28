import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from exp_benchmark.screen_youtube import comparison, main, original_qps, recall_bracket


class ScreeningTests(unittest.TestCase):
    def test_help_prints_literal_percentage_without_starting_a_run(self):
        with mock.patch("sys.argv", ["screen_youtube", "--help"]), \
                mock.patch("sys.stdout", new_callable=io.StringIO) as output, \
                mock.patch("exp_benchmark.screen_youtube.run") as run:
            with self.assertRaises(SystemExit) as exit:
                main()
            self.assertEqual(exit.exception.code, 0)
            self.assertIn("corrected-native 1% fine-ef run", output.getvalue())
            run.assert_not_called()

    def test_recall_only_refinement_returns_adjacent_integer_bracket(self):
        measure = mock.Mock(side_effect=lambda ef: 0.94 + ef * 0.0003)
        self.assertEqual(recall_bracket(measure), [33, 34])
        self.assertEqual(len(measure.call_args_list), 8)

    def test_unreached_calibration_stops_at_the_frozen_ceiling(self):
        measure = mock.Mock(return_value=0.90)
        with self.assertRaisesRegex(RuntimeError, "bounded screening grid"):
            recall_bracket(measure)
        self.assertEqual(measure.call_args_list[-1].args, (160,))

    def test_minimum_ef_above_target_is_observed(self):
        measure = mock.Mock(return_value=0.97)
        self.assertEqual(recall_bracket(measure), [10])
        measure.assert_called_once_with(10)

    def test_five_percent_is_not_enough_to_trigger_more_work(self):
        for qps, expected in ((100, False), (105, False), (105.01, True)):
            target = {"reported_qps": qps, "qps_at_target": qps,
                      "status": "reached", "observed_recall": 0.9501}
            self.assertEqual(comparison(2, 100, target)["exceeds_5pct"], expected)

    def test_reference_reads_the_original_low_and_curve(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "reference.R"
            path.write_text(
                "bfann_youtube_rgb_95_df <- data.frame(\n"
                "x = c(0.10, 0.07, 0.05, 0.03, 0.02, 0.01),\n"
                "y = c(349.89, 321.46, 293.15, 251, 170, 89)\n)\n")
            values = original_qps(path)
        self.assertEqual(values[1], 89)
        self.assertEqual(values[10], 349.89)


if __name__ == "__main__":
    unittest.main()
