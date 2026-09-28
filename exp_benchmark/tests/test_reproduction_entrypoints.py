import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


class ShellEntrypointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scripts = self.root / "exp_benchmark"
        (self.scripts / "ablation").mkdir(parents=True)
        for name in (
            "env.sh", "conf.sh", "query.sh", "main_experiment.sh",
            "ground_truth_generator.sh", "ablation/M_sweep.sh",
            "ablation/min_deg_sweep.sh", "ablation/ft_bits_sweep.sh",
        ):
            shutil.copy2(ROOT / "exp_benchmark" / name, self.scripts / name)
        self.env = {
            key: value for key, value in os.environ.items()
            if key not in {
                "EMA_CONDA_ENV", "EMA_DATASETS", "M_VALS", "MD_VALS", "FT_VALS",
                "threads", "dataset", "M", "ft_bits", "attr_type", "dnf_name",
                "dnf_spec", "GT_BACKEND", "FT_ROUTING_MIN_DEG",
            }
        }
        self.env.update(DATA_ROOT=str(self.root / "data"), HOME=str(self.root),
                        CAPTURE_FILE=str(self.root / "arguments"))
        binaries = self.root / "bin"
        binaries.mkdir()
        python = binaries / "python"
        python.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURE_FILE"\n')
        python.chmod(0o755)
        self.env["PATH"] = f"{binaries}:{os.environ['PATH']}"

    def run_script(self, script, **overrides):
        return subprocess.run(
            ["bash", script], cwd=self.scripts, env={**self.env, **overrides},
            text=True, capture_output=True, timeout=20,
        )

    def prepare_query_files(self):
        subprocess.run(
            ["bash", "-ec", 'source ./conf.sh; touch "$hashann_index_file" '
             '"$query_predicate_file" "$ground_truth_file"'],
            cwd=self.scripts, env=self.env, check=True, capture_output=True,
        )

    def test_query_default_precedes_construction_thread_default(self):
        self.prepare_query_files()
        for overrides, expected in (({}, "1"), ({"threads": "7"}, "7")):
            with self.subTest(overrides=overrides):
                result = self.run_script("query.sh", **overrides)
                self.assertEqual(result.returncode, 0, result.stderr)
                args = (self.root / "arguments").read_text().splitlines()
                self.assertEqual(args[args.index("--threads") + 1], expected)

    def test_environment_keeps_caller_python_unless_explicitly_selected(self):
        code = 'conda() { echo unexpected-activation >&2; return 17; }; source ./env.sh'
        result = subprocess.run(["bash", "-ec", code], cwd=self.scripts,
                                env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("unexpected-activation", result.stderr)
        result = subprocess.run(
            ["bash", "-ec", code], cwd=self.scripts,
            env={**self.env, "EMA_CONDA_ENV": "missing"},
            capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("failed to activate", result.stderr)

    def test_space_separated_dataset_and_sweep_overrides(self):
        (self.scripts / "ablation/selectivity_specs.sh").write_text(
            "EMA_SEL6=('10|test|[]')\n")
        for name in ("build_index.sh", "predicate_generator.sh",
                     "ground_truth_generator.sh"):
            path = self.scripts / name
            path.write_text("#!/bin/sh\nexit 0\n")
            path.chmod(0o755)
        query = self.scripts / "query.sh"
        query.write_text(
            '#!/bin/sh\n'
            'printf "%s %s %s %s\\n" "$dataset" "${M:-40}" '
            '"${FT_ROUTING_MIN_DEG:-16}" "${ft_bits:-128}" >> "$CAPTURE_FILE"\n'
            'printf "Final results\\n[10, 0.95, 1.0, -1]\\n"\n')
        for script, key, values, column in (
            ("main_experiment.sh", "EMA_DATASETS", "sift10m youtube_rgb", 0),
            ("ablation/M_sweep.sh", "M_VALS", "16 32", 1),
            ("ablation/min_deg_sweep.sh", "MD_VALS", "0 8", 2),
            ("ablation/ft_bits_sweep.sh", "FT_VALS", "32 64", 3),
        ):
            with self.subTest(script=script):
                capture = self.root / "arguments"
                capture.write_text("")
                result = self.run_script(script, **{key: values})
                self.assertEqual(result.returncode, 0, result.stderr)
                actual = [line.split()[column] for line in capture.read_text().splitlines()]
                self.assertEqual(actual, values.split())

    def test_ground_truth_backend_is_explicit_not_a_silent_fallback(self):
        result = self.run_script("ground_truth_generator.sh", GT_BACKEND="numpy")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = (self.root / "arguments").read_text().splitlines()
        self.assertEqual(args[0], "../tests/groundtruth_bruteforce.py")
        self.assertEqual(args[args.index("--attr_type_list") + 1], "[0,1]")
        for backend in ("cpp", "invalid"):
            with self.subTest(backend=backend):
                result = self.run_script(
                    "ground_truth_generator.sh", GT_BACKEND=backend,
                    FAISS_ROOT=str(self.root / "missing-faiss"),
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("[gt]", result.stderr)


class FalsePositiveInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        labels = self.root / "redcaps4m/label/arbi_0_random"
        labels.mkdir(parents=True)
        vectors = np.array([[0, 1], [1, 0], [2, 1]], dtype=np.float32)
        encoded = np.empty((3, 3), dtype=np.float32)
        encoded[:, 0] = np.array([2], dtype=np.int32).view(np.float32)[0]
        encoded[:, 1:] = vectors
        encoded.tofile(self.root / "queries.fvecs")
        (labels / "predicate_arbi_0_[0,9].json").write_text(
            json.dumps([[[0, 9]]] * 3))
        self.gt = labels / "gt_arbi_0_[0,9].json"
        self.gt.write_text(json.dumps([[0, 2, 3], [1, 3, 0], [2, 0, 1]]))
        (self.root / "index").touch()

    def run_fpr(self):
        code = """
import runpy, sys, types
import numpy as np
class Index:
    def set_num_threads(self, value):
        assert value == 1
    def set_ft_routing_flag(self, value): pass
    def set_ft_routing_min_deg(self, value): pass
    def set_ef(self, value): pass
    def set_ft_flag(self, value): pass
    def reset_ft_stats(self): pass
    def hybrid_knn_query(self, queries, predicates, k, num_threads):
        assert len(queries) == len(predicates) == 2
        assert k == num_threads == 1
        return np.array([[2], [1]], dtype=np.uint64), np.zeros((2, 1))
    def get_ft_stats(self):
        return dict(ft_passed_total=4, ft_false_positives=2,
                    ft_true_positives=2, ft_fp_rate=.5)
class Wrapper:
    def init_params(self, params): pass
    def load_index(self, *args): return Index()
sys.modules["hashann"] = types.SimpleNamespace(HashANN=Wrapper)
runpy.run_module("exp_benchmark.ft_fpr", run_name="__main__")
"""
        return subprocess.run(
            [sys.executable, "-c", code, "--data_root", str(self.root),
             "--index_path", str(self.root / "index"),
             "--query_file", str(self.root / "queries.fvecs"),
             "--label_subdir", "label/arbi_0_random", "--attr_type", "0",
             "--N", "4", "--dim", "2", "--K", "1", "--n_query", "2",
             "--ef_search_list", "10", "--selectivities", "[0,9]:all",
             "--out_json", str(self.root / "results.json")],
            cwd=ROOT, capture_output=True, text=True, timeout=20,
        )

    def test_recall_uses_the_requested_prefix_and_top_k_columns(self):
        result = self.run_fpr()
        self.assertEqual(result.returncode, 0, result.stderr)
        row = json.loads((self.root / "results.json").read_text())["results"][0]
        self.assertEqual(row["queries"], 2)
        self.assertEqual(row["recall"], .5)

    def test_short_ground_truth_fails_instead_of_changing_the_query_count(self):
        self.gt.write_text("[[0, 2, 3]]")
        result = self.run_fpr()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requested prefix", result.stderr)
        self.assertFalse((self.root / "results.json").exists())


if __name__ == "__main__":
    unittest.main()
