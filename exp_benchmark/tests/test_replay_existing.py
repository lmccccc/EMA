import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from exp_benchmark import replay_existing as replay
from exp_benchmark.dynamic.runtime import atomic_json, file_info
from exp_benchmark.static_paper import Attributes, Predicates


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("query_pipeline_under_test", ROOT / "tests/hashann_query.py")
query = importlib.util.module_from_spec(spec)
with mock.patch.dict("sys.modules", {
        "hashann": types.SimpleNamespace(HashANN=mock.Mock()),
        "utils": types.SimpleNamespace(arg_init=mock.Mock(), fvecs_read=mock.Mock(), read_multy_attr=mock.Mock())}):
    spec.loader.exec_module(query)


class FakeIndex:
    def __init__(self):
        self.ef = 2
        self.calls = 0
        self.failure = None

    def __getattr__(self, name):
        if name.startswith("set_"):
            return lambda value: setattr(self, name[4:], value)
        raise AttributeError(name)

    def hybrid_knn_query_with_stats(self, queries, predicates, k):
        self.calls += 1
        if self.failure:
            raise self.failure
        labels = np.array([[0, 4], [2, 5]], dtype=np.uint64)
        if self.ef >= 4:
            labels[0, 1] = 1
        if self.ef >= 8:
            labels[1, 1] = 3
        labels = labels[:len(queries)]
        return labels, np.ones(labels.shape, dtype=np.float32), np.full(len(queries), 12)

    def hybrid_knn_query_dnf(self, queries, predicates, k, num_threads):
        if num_threads != 1:
            raise ValueError("The replay requires one query thread")
        return self.hybrid_knn_query_with_stats(queries, predicates, k)[:2]

    def get_current_count(self):
        return 6

    def get_deleted_ratio(self):
        return 0


class ExistingQueryTests(unittest.TestCase):
    def setUp(self):
        self.queries = np.zeros((2, 2), dtype=np.float32)
        self.predicates = [[[0, 10]], [[0, 10]]]
        self.truth = np.array([[0, 1], [2, 3]])

    def measure(self, index=None, **kwargs):
        parameters = dict(count=6, k=2, efs=[2, 4, 8, 16], repeats=1, target_recall=0.95)
        parameters.update(kwargs)
        with contextlib.redirect_stdout(io.StringIO()):
            return query.query_sweep(index or FakeIndex(), self.queries, self.predicates,
                                     self.truth, **parameters)

    def test_gt_keeps_rows_and_selects_top_k_columns(self):
        truth = np.arange(200).reshape(2, 100).tolist()
        with mock.patch.object(query, "fvecs_read", return_value=self.queries), \
                mock.patch.object(query, "read_multy_attr", side_effect=[self.predicates, truth]):
            _, _, selected = query.load_query_data("q.fvecs", "p.json", "gt.json", 200, 2, 2)
        np.testing.assert_array_equal(selected, [[0, 1], [100, 101]])

    def test_incomplete_prefix_is_rejected(self):
        with mock.patch.object(query, "fvecs_read", return_value=self.queries[:1]), \
                mock.patch.object(query, "read_multy_attr", side_effect=[self.predicates, self.truth]):
            with self.assertRaisesRegex(ValueError, "requested prefix"):
                query.load_query_data("q.fvecs", "p.json", "gt.json", 6, 2, 2)

    def test_warmup_failure_is_not_swallowed_or_retried(self):
        index = FakeIndex()
        index.failure = RuntimeError("native failure")
        with self.assertRaisesRegex(RuntimeError, "native failure"):
            self.measure(index)
        self.assertEqual(index.calls, 1)

    def test_median_uses_every_repeat_not_fastest(self):
        with mock.patch.object(query.time, "perf_counter", side_effect=[0, 2, 10, 11, 20, 24]):
            result = self.measure(efs=[8], repeats=3)
        self.assertEqual([sample["qps"] for sample in result["points"][0]["samples"]], [1, 2, 0.5])
        self.assertEqual(result["rows"][0][2], 1)

    def test_default_query_and_collection_repeat_count_is_three(self):
        self.assertEqual(replay.QUERY_REPEATS, 3)
        with contextlib.redirect_stdout(io.StringIO()):
            result = query.query_sweep(
                FakeIndex(), self.queries, self.predicates, self.truth, count=6, k=2, efs=[8])
        self.assertEqual(len(result["points"][0]["samples"]), 3)

    def test_recall_stopping_keeps_the_predefined_grid(self):
        result = self.measure()
        self.assertEqual([point["ef"] for point in result["points"]], [2, 4, 8])
        self.assertEqual([point["recall"] for point in result["points"]], [0.5, 0.75, 1])
        with self.assertRaisesRegex(ValueError, "increasing ef"):
            self.measure(efs=[4, 2, 8])

    def test_missing_duplicate_sentinel_and_noninteger_labels_fail(self):
        for labels in (np.array([[0, 0], [2, 3]]), np.array([[-1, 1], [2, 3]]),
                       np.array([[0, 6], [2, 3]]), np.array([[0, 1]]),
                       np.array([[0.0, 1.0], [2.0, 3.0]])):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                query.validate_ids(labels, 6, 2, 2)

    def test_timed_batch_failure_has_no_per_query_fallback(self):
        index = FakeIndex()
        method = index.hybrid_knn_query_with_stats
        with mock.patch.object(index, "hybrid_knn_query_with_stats",
                               side_effect=[method(self.queries, self.predicates, 2),
                                            RuntimeError("batch failure")]) as called:
            with self.assertRaisesRegex(RuntimeError, "batch failure"):
                self.measure(index)
        self.assertEqual(called.call_count, 2)

    def test_result_changes_between_repeats_fail(self):
        index = FakeIndex()
        first = index.hybrid_knn_query_with_stats(self.queries, self.predicates, 2)
        second = (first[0][:, ::-1], *first[1:])
        with mock.patch.object(index, "hybrid_knn_query_with_stats", side_effect=[first, first, second]):
            with self.assertRaisesRegex(RuntimeError, "changed between"):
                self.measure(index, repeats=2)

    def test_dnf_uses_dnf_api_and_keeps_raw_outputs(self):
        self.predicates = [[[[0, 10]], [[0, 5]]]] * 2
        saved = mock.Mock()
        result = self.measure(efs=[8], repeats=3, on_result=saved)
        self.assertEqual(result["query_api"], "hybrid_knn_query_dnf")
        self.assertEqual(result["rows"][0][3], -1)
        saved.assert_called_once()


class ReplayCollectionTests(unittest.TestCase):
    def test_loader_only_native_cannot_restart_the_known_bad_sweep(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "provenance.json"
            atomic_json(path, {
                "source_commit": "28e07e33cb2dd4e7c173ca1030e6c5bbd7f4f7bb",
                "index_rebuilt": False, "index_files_modified": False,
            })
            with self.assertRaisesRegex(ValueError, "numerical boundary buckets"):
                replay.prepare(temporary, path)

    def test_multilabel_conditions_require_complete_containment(self):
        attributes = Attributes.from_records([[[1]], [[1, 2]], [[2]]], [1], 3)
        predicate = Predicates.parse([[[1, 2]]], [1], 1)
        check = replay.predicate_validator(attributes, predicate)
        check(np.array([[1]]))
        with self.assertRaises(ValueError):
            check(np.array([[0]]))

    def test_auxiliary_and_primary_use_their_own_first_brackets(self):
        rows = [[10, 0.89, 100, 0], [20, 0.91, 80, 0],
                [40, 0.94, 70, 0], [80, 0.96, 50, 0]]
        auxiliary, primary = replay.summaries(rows, [0.90, 0.95])
        self.assertAlmostEqual(auxiliary["qps_at_target"], 90)
        self.assertAlmostEqual(primary["qps_at_target"], 60)
        self.assertEqual(auxiliary["observed_recall"], 0.91)
        self.assertEqual([point[0] for point in primary["bracket"]], [40, 80])

    def test_minimum_above_and_unreached_are_not_exact_target_qps(self):
        minimum = replay.summaries([[10, 0.96, 100, 0]], [0.95])[0]
        self.assertEqual(minimum["status"], "minimum_above")
        self.assertIsNone(minimum["qps_at_target"])
        self.assertEqual(minimum["reported_qps"], 100)
        unreachable = replay.summaries([[10, 0.5, 100, 0]], [0.95])[0]
        self.assertEqual(unreachable["status"], "unreached")
        self.assertIsNone(unreachable["reported_qps"])

    def test_worker_only_loads_one_existing_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            atomic_json(root / "attributes.json", [[[i]] for i in range(6)])
            atomic_json(root / "queries.json", {})
            atomic_json(root / "predicates.json", [[[0, 10]], [[0, 10]]])
            atomic_json(root / "truth.json", [[0, 1], [2, 3]])
            atomic_json(root / "native.json", {})
            cell = {
                "name": "T10", "nominal_selectivity": 0.1, "query_count": 2, "targets": [0.95],
                "predicate": file_info(root / "predicates.json"),
                "ground_truth": file_info(root / "truth.json"), "plots": [{"figure": 5, "panel": "a"}],
            }
            group = {
                "name": "fixture", "dataset": "fixture", "N": 6, "dimension": 2, "metric": "l2",
                "attribute_types": [0], "attributes": file_info(root / "attributes.json"),
                "queries": file_info(root / "queries.json"), "cells": [cell],
                "index": {"path": str(root / "existing.index"), "layout": {"format": 4, "ft_bits": 128}},
            }
            request = {
                "groups": [group], "source": {}, "native": file_info(root / "native.json"),
                "settings": {"k": 2, "efs": [8], "ef_top": 1, "routing_min_deg": 16,
                             "use_ft": True, "backfill_tail": False, "repeats": 3,
                             "query_core": 24, "environment": {}},
            }
            builder = mock.Mock()
            builder.load_index.return_value = FakeIndex()
            builder.build_index.side_effect = AssertionError("Index construction is forbidden")
            builder.clustering.side_effect = AssertionError("Clustering is forbidden")
            modules = {
                "hashannlib": types.SimpleNamespace(__file__=str(root / "native.json")),
                "hashann": types.SimpleNamespace(HashANN=lambda: builder),
                "hashann_query": query,
            }
            with mock.patch.object(replay.importlib, "import_module", side_effect=modules.__getitem__), \
                    mock.patch.object(replay, "unchanged_index") as unchanged, \
                    mock.patch.object(replay, "record_sample", return_value=[
                        {"label": 0, "attributes": [[0]]}, {"label": 5, "attributes": [[5]]}]), \
                    mock.patch.object(replay.os, "sched_getaffinity", return_value={24}), \
                    mock.patch.object(replay.os, "sched_setaffinity"), \
                    mock.patch.object(query, "load_query_data", return_value=(
                        np.zeros((2, 2), dtype=np.float32), [[[0, 10]], [[0, 10]]],
                        np.array([[0, 1], [2, 3]]))):
                self.assertEqual(replay.worker(request, "fixture", output), 0)
            builder.load_index.assert_called_once()
            builder.build_index.assert_not_called()
            builder.clustering.assert_not_called()
            self.assertEqual(unchanged.call_count, 2)
            result = replay.read_json(output / "results.json")
            replay.verify_measurement(result["cells"][0], group, cell, request)
            result["cells"][0]["measurement"]["rows"][0][2] *= 2
            with self.assertRaisesRegex(ValueError, "median"):
                replay.verify_measurement(result["cells"][0], group, cell, request)


if __name__ == "__main__":
    unittest.main()
