import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from exp_benchmark.dynamic.runtime import (
    atomic_json, import_native, read_attribute_index_header, read_index_header, verify_index_tail,
)
from exp_benchmark.static_paper import (
    PARAMETERS, Attributes, Predicates, check_profile, load_builder, suite,
    summarize_target, validate_result, verify_records,
)


class StaticPredicateTests(unittest.TestCase):
    def test_numeric_single_attribute(self):
        attributes = Attributes.from_records([[[1]], [[2]], [[3]], [[4]]], [0], 4)
        predicates = Predicates.parse([[[2, 3]], [[1, 4]]], [0], 2)
        np.testing.assert_array_equal(predicates.counts(attributes), [2, 4])
        self.assertTrue(predicates.matches(np.array([[1, 2], [0, 3]]), attributes).all())
        self.assertFalse(predicates.matches(np.array([[0], [3]]), attributes).all())

    def test_multilabel_subset_is_not_any_label(self):
        attributes = Attributes.from_records(
            [[[1, 2]], [[1]], [[2]], [[1, 2, 3]]], [1], 4)
        predicates = Predicates.parse([[[1, 2]], [[1, 2]]], [1], 2)
        np.testing.assert_array_equal(predicates.counts(attributes), [2, 2])
        self.assertTrue(predicates.matches(np.array([[0, 3], [3, 0]]), attributes).all())
        self.assertFalse(predicates.matches(np.array([[1], [2]]), attributes).all())

    def test_dnf_counts_union_once(self):
        attributes = Attributes.from_records(
            [[[1], [1]], [[2], [1, 2]], [[3], [2]], [[4], [1]]], [0, 1], 4)
        raw = [[[[1, 2], [1]], [[], [2]]]] * 2
        predicates = Predicates.parse(raw, [0, 1], 2, dnf=True)
        np.testing.assert_array_equal(predicates.counts(attributes), [3, 3])
        self.assertTrue(predicates.matches(np.array([[0, 1, 2], [2, 1, 0]]), attributes).all())

    def test_malformed_records_and_bounds_rejected(self):
        for records, kinds in [([[[1, 2]]], [0]), ([[[32]]], [1]),
                               ([[[1, 1]]], [1]), ([[[True]]], [0])]:
            with self.subTest(records=records):
                with self.assertRaises(ValueError):
                    Attributes.from_records(records, kinds, 1)
        with self.assertRaises(ValueError):
            Predicates.parse([[[0, 2**31]]], [0], 1)
        with self.assertRaises(ValueError):
            Predicates.parse([[[1]]], [0], 1)

    def test_ip_distance_is_one_minus_dot(self):
        vectors = np.asarray([[1, 0], [0.8, 0.6], [0, 1]], dtype=np.float32)
        queries = np.asarray([[1, 0]], dtype=np.float32)
        attributes = Attributes.from_records([[[1]]] * 3, [1], 3)
        predicates = Predicates.parse([[[1]]], [1], 1)
        labels = np.asarray([[0, 1]], dtype=np.uint64)
        distances = np.asarray([[0, 0.2]], dtype=np.float32)
        result = validate_result((labels, distances), labels, vectors, queries,
                                 attributes, predicates, "ip", 2)
        self.assertEqual(result["recall"], 1)
        with self.assertRaisesRegex(RuntimeError, "distance/vector"):
            validate_result((labels, distances - 1), labels, vectors, queries,
                            attributes, predicates, "ip", 2)

    def test_minimum_ef_is_observed_not_extrapolated(self):
        summary = summarize_target([{"ef": 10, "recall": 0.9504}], {10: 648.1}, 0.95)
        self.assertIsNone(summary["qps_at_target"])
        self.assertEqual(summary["reported_qps"], 648.1)
        self.assertEqual(summary["observed_recall"], 0.9504)
        self.assertEqual(summary["method"], "observed_at_minimum_ef_NO_extrapolation")

    def test_primary_and_auxiliary_use_their_first_crossings(self):
        points = [{"ef": 10, "recall": 0.89}, {"ef": 11, "recall": 0.91},
                  {"ef": 12, "recall": 0.945}, {"ef": 13, "recall": 0.955}]
        medians = {10: 100, 11: 90, 12: 80, 13: 70}
        auxiliary = summarize_target(points, medians, 0.90)
        primary = summarize_target(points, medians, 0.95)
        self.assertAlmostEqual(auxiliary["qps_at_target"], 95)
        self.assertAlmostEqual(primary["qps_at_target"], 75)
        self.assertEqual(primary["selected_ef"], 13)
        with self.assertRaises(RuntimeError):
            summarize_target(points, medians, 0.99)

    def test_original_100_query_prefix_and_build_only_are_explicit(self):
        job = {
            "metric": "ip", "N": 1_000_000, "dimension": 1024, "query_count": 100,
            "query_offset": 0, "normalization": "none", "attribute_types": [1],
            "cells": [{"name": "T1", "query_count": 100, "predicate_rows": 100}],
            "max_ef": 1000, "parameters": PARAMETERS,
        }
        check_profile(job)
        job["cells"][0]["query_count"] = 1000
        with self.assertRaises(ValueError):
            check_profile(job)
        job["cells"], job["build_only"] = [], True
        check_profile(job)
        job["checkpoint"] = {"path": "/not-opened"}
        with self.assertRaises(ValueError):
            check_profile(job)

    def test_completed_suite_jobs_still_enter_worker_resume_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            request = {"jobs": [{"name": "completed"}]}
            atomic_json(output / "suite-manifest.json", {"request": request, "source": {}})
            directory = output / "jobs/completed"
            directory.mkdir(parents=True)
            atomic_json(directory / "results.json", {"status": "complete"})
            process = mock.Mock()
            process.poll.return_value = 0
            process.wait.return_value = 0
            with mock.patch("exp_benchmark.static_paper.source_identity", return_value={}), \
                    mock.patch("exp_benchmark.static_paper.subprocess.Popen", return_value=process) as spawn:
                suite(request, output, resume=True)
            spawn.assert_called_once()
            self.assertIn("--resume", spawn.call_args.args[0])


@unittest.skipUnless(os.environ.get("HASHANN_STATIC_NATIVE_DIR")
                     and os.environ.get("HASHANN_STATIC_NATIVE_SHA256"),
                     "Set the explicit static native directory and SHA for native integration")
class StaticNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native, _ = import_native(
            os.environ["HASHANN_STATIC_NATIVE_DIR"], os.environ["HASHANN_STATIC_NATIVE_SHA256"])
        cls.builder = load_builder()

    def test_ip_representatives_maximize_similarity_and_ties_use_first_id(self):
        distances = np.array([0.9, 0.2, 0.7, 0.1], dtype=np.float32)
        groups = np.array([0, 0, 1, 1], dtype=np.int64)
        closest, levels = self.builder.closest_cluster_representatives(distances, groups, 3, "ip")
        np.testing.assert_array_equal(closest, [0, 2, -1])
        np.testing.assert_array_equal(levels, [1, 0, 1, 0])
        closest, _ = self.builder.closest_cluster_representatives(distances, groups, 3, "l2")
        np.testing.assert_array_equal(closest, [1, 3, -1])
        closest, _ = self.builder.closest_cluster_representatives(
            np.ones(4), groups, 2, "ip")
        np.testing.assert_array_equal(closest, [0, 2])

    def test_level_cache_pins_source_metric_and_exact_count(self):
        vectors = np.random.default_rng(9).normal(size=(64, 8)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        with tempfile.TemporaryDirectory() as temporary:
            builder = self.builder.HashANN()
            builder.init_params({"N": 64, "dim": 8, "metric": "ip", "threads": 1})
            builder.save_root = str(Path(temporary) / "levels")
            fresh = builder.clustering(vectors)
            self.assertFalse(builder.clustering_cache_reused)
            with mock.patch.object(self.builder.faiss, "Clustering", side_effect=AssertionError):
                cached = builder.clustering(vectors)
            self.assertTrue(builder.clustering_cache_reused)
            for left, right in zip(fresh, cached):
                np.testing.assert_array_equal(left, right)
            changed = vectors.copy()
            changed[0, 0] += 0.1
            with self.assertRaisesRegex(RuntimeError, "source/metric/policy"):
                builder.clustering(changed)
            builder.metric = "l2"
            with self.assertRaisesRegex(RuntimeError, "source/metric/policy"):
                builder.clustering(vectors)
            builder.metric, builder.N = "ip", 65
            with self.assertRaisesRegex(RuntimeError, "source/metric/policy"):
                builder.clustering(np.vstack([vectors, vectors[:1]]))

    def test_current_single_and_mixed_indexes_match_source_and_native_ip(self):
        rng = np.random.default_rng(19)
        vectors = rng.normal(size=(128, 16)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        queries = vectors[[25, 54, 77]].copy()
        for kinds in ([0], [1], [0, 1]):
            with self.subTest(kinds=kinds), tempfile.TemporaryDirectory() as temporary:
                records = [
                    [[row] if kind == 0 else [row % 4] for kind in kinds]
                    for row in range(len(vectors))
                ]
                attributes = Attributes.from_records(records, kinds, len(vectors))
                index = self.native.Index(space="ip", dim=16)
                index.init_index(max_elements=128, top_elements=16, M=40, ef_construction=300,
                                 ft_bits=128, attr_type=kinds, max_cate_size=4, edge_level_ft=True)
                index.initAttrMapping(records)
                levels = np.zeros(128, dtype=np.int32)
                levels[:16] = 1
                index.add_items(vectors, records, levels=levels, num_threads=1)
                path = Path(temporary) / "index"
                index.save_index(str(path))
                header = read_attribute_index_header(path)
                self.assertEqual(header["attr_type"], kinds)
                self.assertEqual(header["ft_bytes_per_record"], 16 * len(kinds))
                verify_records(path, header, attributes, vectors)
                verify_index_tail(path, header)
                if kinds == [0, 1]:
                    read_index_header(path)
                    raw = [[[[20, 90], [1]], [[], [2]]]] * len(queries)
                    dnf = True
                else:
                    with self.assertRaises(ValueError):
                        read_index_header(path)
                    raw = ([[[20, 90]]] if kinds == [0] else [[[1]]]) * len(queries)
                    dnf = False
                predicates = Predicates.parse(raw, kinds, len(queries), dnf=dnf)
                ids = np.broadcast_to(np.arange(128), (len(queries), 128))
                allowed = predicates.matches(ids, attributes)
                exact = 1 - queries.astype(np.float64) @ vectors.astype(np.float64).T
                exact[~allowed] = np.inf
                truth = np.argsort(exact, axis=1)[:, :10]
                index.generateAttrIndexes()
                index.set_thresholds(0.0001, 0.0001, 0.0001)
                index.set_ft_flag(True)
                index.set_ft_routing_flag(True)
                index.set_ft_routing_min_deg(16)
                index.set_ft_routing_backfill_tail(False)
                index.set_ef_top(1)
                index.set_ef(128)
                method = index.hybrid_knn_query_dnf if dnf else index.hybrid_knn_query
                result = method(queries, raw, k=10, num_threads=1)
                validated = validate_result(
                    result, truth, vectors, queries, attributes, predicates, "ip", 10)
                self.assertGreaterEqual(validated["recall"], 0.95)


if __name__ == "__main__":
    unittest.main()
