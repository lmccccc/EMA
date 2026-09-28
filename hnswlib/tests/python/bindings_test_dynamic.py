import struct
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

import hashannlib


class DynamicApiTestCase(unittest.TestCase):
    preparation_counters = {
        "parallel_preparations", "reused_preparations", "recomputed_preparations",
    }

    def assert_same_deletion_work(self, serial, parallel):
        excluded = self.preparation_counters | {"scrub_wall_s"}
        self.assertEqual(
            {key: value for key, value in serial.items() if key not in excluded},
            {key: value for key, value in parallel.items() if key not in excluded},
        )

    def edge_markers(self, index):
        state = index.__getstate__()[0]
        rows = state["data_level0"].view(np.uint8).reshape(
            state["cur_element_count"], state["size_data_per_element"]
        )
        max_degree = state["max_M0"]
        links_offset = state["offset_level0"]
        markers_offset = links_offset + 4 + 4 * max_degree
        markers_end = state["offset_data"]
        result = {}
        for source, row in enumerate(rows):
            degree = int(row[links_offset:links_offset + 2].copy().view(np.uint16)[0])
            neighbors = row[
                links_offset + 4:links_offset + 4 + 4 * degree
            ].copy().view(np.uint32)
            markers = row[markers_offset:markers_end].reshape(max_degree, -1)
            for edge, target in enumerate(neighbors):
                result[source, int(target)] = markers[edge].copy()
        return result

    def own_masks(self, index, attrs, attr_type):
        predicates = [
            [
                [values[0], values[0]] if kind == 0 else values
                for kind, values in zip(attr_type, record)
            ]
            for record in attrs
        ]
        return np.asarray([
            [ord(char) for char in mask] for mask in index.predicateToFT(predicates)
        ], dtype=np.uint8)

    def assert_live_graph(self, index, deleted, masks=None):
        deleted = set(map(int, deleted))
        state = index.__getstate__()[0]
        count = state["cur_element_count"]
        rows = state["data_level0"].view(np.uint8).reshape(
            count, state["size_data_per_element"],
        )
        marked = set(np.flatnonzero((rows[:, state["offset_level0"] + 2] & 1) != 0))
        self.assertEqual(marked, deleted)
        self.assertEqual(index.get_dirty_count(), 0)
        self.assertEqual(index.repair_candidates_size(), 0)
        for source in range(count):
            neighbors = list(map(int, index.get_neighbors(source)))
            self.assertLessEqual(len(neighbors), state["max_M0"])
            self.assertEqual(len(neighbors), len(set(neighbors)))
            self.assertNotIn(source, neighbors)
            self.assertTrue(all(0 <= target < count and target not in deleted for target in neighbors))
            if source in deleted:
                self.assertEqual(neighbors, [])
        if masks is not None:
            for (source, target), marker in self.edge_markers(index).items():
                self.assertNotIn(source, deleted)
                np.testing.assert_array_equal(marker & masks[target], masks[target])
        upper = state["link_lists"].view(np.uint8)
        levels = state["element_levels"][:count]
        stride, offset = state["size_links_per_element"], 0
        for source, level_count in enumerate(levels):
            for level in range(1, int(level_count) + 1):
                row = upper[offset:offset + stride]
                degree = int(row[:2].copy().view(np.uint16)[0])
                neighbors = row[4:4 + degree * 4].copy().view(np.uint32)
                self.assertLessEqual(degree, state["max_M"])
                self.assertEqual(len(neighbors), len(set(map(int, neighbors))))
                self.assertFalse(np.isin(neighbors, list(deleted)).any())
                self.assertFalse(np.any(neighbors == source))
                self.assertTrue(np.all(neighbors < count))
                self.assertTrue(np.all(levels[neighbors] >= level))
                if source in deleted:
                    self.assertEqual(degree, 0)
                offset += stride
        self.assertEqual(offset, upper.size)
        if len(deleted) < count:
            self.assertNotIn(state["enterpoint_node"], deleted)

    def assert_valid_query(self, index, queries, result, eligible, space="l2"):
        labels, distances = result
        self.assertTrue(np.isin(labels, eligible).all())
        self.assertTrue(np.isfinite(distances).all())
        self.assertTrue(np.all(np.diff(distances, axis=1) >= 0))
        for row in labels:
            self.assertEqual(len(row), len(set(map(int, row))))
        stored = index.get_items(labels.reshape(-1)).reshape(
            labels.shape + (queries.shape[1],),
        )
        if space == "cosine":
            queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        expected = (
            ((queries[:, None] - stored) ** 2).sum(axis=2)
            if space == "l2" else 1 - np.einsum("qd,qkd->qk", queries, stored)
        )
        np.testing.assert_allclose(distances, expected, rtol=5e-5, atol=1e-6)

    def make_index(self):
        data = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            dtype=np.float32,
        )
        attrs = [[[0]], [[1]], [[2]], [[3]]]
        index = hashannlib.Index(space="l2", dim=2)
        index.init_index(
            max_elements=len(data),
            top_elements=1,
            M=2,
            ef_construction=20,
            ft_bits=8,
            attr_type=[1],
            max_cate_size=4,
            edge_level_ft=True,
        )
        index.initAttrMapping(attrs)
        index.add_items(data, attrs, ids=np.arange(len(data)))
        index.set_ef(20)
        return index, data

    def make_tracking_index(self, count=70):
        data = np.column_stack((np.arange(count), np.zeros(count))).astype(np.float32)
        attrs = [[[0]] for _ in range(count)]
        levels = np.zeros(count, dtype=np.int32)
        levels[:3] = 1
        index = hashannlib.Index(space="l2", dim=2)
        index.init_index(
            max_elements=count, top_elements=3, M=80, ef_construction=160,
            ft_bits=8, attr_type=[1], max_cate_size=4, edge_level_ft=True,
        )
        index.initAttrMapping(attrs)
        index.add_items(
            data, attrs, ids=np.arange(count), levels=levels, num_threads=1,
        )
        return index

    def make_inplace_index(self, M=6, unique_attributes=False, attribute_count=1):
        rng = np.random.default_rng(941)
        data = rng.random((96, 4), dtype=np.float32)
        categories = len(data) if unique_attributes else 8
        attrs = [[[i % categories] for _ in range(attribute_count)] for i in range(len(data))]
        levels = np.zeros(len(data), dtype=np.int32)
        levels[:8] = 1
        index = hashannlib.Index(space="l2", dim=4)
        index.init_index(
            max_elements=len(data), top_elements=8, M=M,
            ef_construction=128, ft_bits=128, attr_type=[1] * attribute_count,
            max_cate_size=categories, edge_level_ft=True,
        )
        index.initAttrMapping(attrs)
        index.add_items(
            data, attrs, ids=np.arange(len(data)), levels=levels, num_threads=1,
        )
        index.set_ef(192)
        return index, data, attrs

    def test_loaded_index_can_append_and_replace_at_fresh_labels(self):
        data = np.random.default_rng(712).random((96, 4), dtype=np.float32)
        attrs = [[[i % 13], [i % 7, (i + 2) % 7]] for i in range(len(data))]
        labels = np.arange(len(data), dtype=np.uint64) * 17 + 1000
        levels = np.zeros(len(data), dtype=np.int32)
        levels[::16] = 1
        index = hashannlib.Index(space="l2", dim=4)
        index.init_index(
            max_elements=len(data), top_elements=1, M=8,
            ef_construction=64, ft_bits=128, attr_type=[0, 1],
            max_cate_size=8, edge_level_ft=True,
        )
        index.initAttrMapping(attrs)
        index.add_items(
            data[:48], attrs[:48], ids=labels[:48],
            levels=levels[:48], num_threads=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            initial = str(Path(directory) / "initial.index")
            index.save_index(initial)
            for dynamic in (False, True):
                with self.subTest(dynamic=dynamic):
                    loaded = hashannlib.Index(space="l2", dim=4)
                    loaded.load_index(initial, max_elements=96, top_elements=1, dynamic=dynamic)
                    loaded.add_items(
                        data[48:64], attrs[48:64], ids=labels[48:64],
                        levels=levels[48:64], num_threads=1,
                    )
                    self.assertEqual(loaded.get_current_count(), 64)
                    np.testing.assert_array_equal(loaded.get_items(labels[48:64]), data[48:64])
                    loaded.set_ef(128)
                    found, _ = loaded.knn_query(data[48:64], k=1, num_threads=1)
                    np.testing.assert_array_equal(found[:, 0], labels[48:64])
                    if not dynamic:
                        continue
                    self.assertEqual(loaded.delete_items(labels[4:12], num_threads=2)["marked"], 8)
                    loaded.add_items(
                        data[64:80], attrs[64:80], ids=labels[64:80],
                        levels=levels[64:80], num_threads=1, replace_deleted=False,
                    )
                    checkpoint = str(Path(directory) / "updated.index")
                    loaded.save_index(checkpoint)
                    resumed = hashannlib.Index(space="l2", dim=4)
                    resumed.load_index(checkpoint, max_elements=112, top_elements=1, dynamic=True)
                    resumed.add_items(
                        data[80:], attrs[80:], ids=labels[80:],
                        levels=levels[80:], num_threads=1, replace_deleted=False,
                    )
                    self.assertEqual(resumed.get_current_count(), 96)
                    self.assertEqual(resumed.get_max_elements(), 112)
                    self.assertAlmostEqual(resumed.get_deleted_ratio(), 8 / 96)
                    live = np.concatenate((np.arange(4), np.arange(12, 96)))
                    np.testing.assert_array_equal(resumed.get_items(labels[live]), data[live])

    def test_categorical_indexes_require_current_marker_and_order_semantics(self):
        index, data = self.make_index()
        state = index.__getstate__()[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "categorical.index"
            index.save_index(str(path))
            payload = bytearray(path.read_bytes())
            version_offset = (
                len(payload) - state["data_level0"].nbytes - state["link_lists"].nbytes
                - 4 * state["cur_element_count"] - 4
            )
            self.assertEqual(struct.unpack_from("<i", payload, version_offset)[0], 10)
            restored = hashannlib.Index(space="l2", dim=2)
            restored.load_index(str(path), max_elements=6, top_elements=1, dynamic=True)
            np.testing.assert_array_equal(restored.get_items(np.arange(4)), data)
            restored.add_items(
                [[4.0, 0.0]], [[[0]]], ids=[99], levels=[0], num_threads=1,
            )
            np.testing.assert_array_equal(restored.get_items([99]), [[4.0, 0.0]])
            for version in (2, 3, 4, 5, 6):
                struct.pack_into("<i", payload, version_offset, version)
                legacy = Path(directory) / f"legacy-{version}.index"
                legacy.write_bytes(payload)
                rejected = hashannlib.Index(space="l2", dim=2)
                with self.assertRaisesRegex(RuntimeError, "Legacy Marker ownership.*rebuild"):
                    rejected.load_index(str(legacy), max_elements=6, top_elements=1, dynamic=True)
            for version in (7, 8):
                struct.pack_into("<i", payload, version_offset, version)
                legacy = Path(directory) / f"mixed-order-{version}.index"
                legacy.write_bytes(payload)
                rejected = hashannlib.Index(space="l2", dim=2)
                with self.assertRaisesRegex(RuntimeError, "Legacy candidate ordering.*rebuild"):
                    rejected.load_index(str(legacy), max_elements=6, top_elements=1, dynamic=True)

    def test_m2_disjoint_categories_keep_a_navigation_backbone(self):
        count = 12
        data = np.column_stack((np.arange(count), np.zeros(count))).astype(np.float32)
        attrs = [[[i]] for i in range(count)]
        index = hashannlib.Index(space="l2", dim=2)
        index.init_index(
            max_elements=count, top_elements=1, M=2, ef_construction=20,
            ft_bits=16, attr_type=[1], max_cate_size=count, edge_level_ft=True,
        )
        index.initAttrMapping(attrs)
        index.add_items(data, attrs, ids=np.arange(count), levels=np.zeros(count, dtype=np.int32),
                        num_threads=1)
        for source in range(count):
            self.assertGreater(index.get_neighbors(source).size, 0)

    def test_attribute_weighted_candidate_order_cannot_be_enabled(self):
        index, _ = self.make_index()
        self.assertEqual(index.get_attr_sort_alpha(), 0.0)
        index.set_attr_sort_alpha(0.0)
        for alpha in (-1.0, 0.5, 1.0, float("inf"), float("-inf"), float("nan")):
            with self.subTest(alpha=alpha):
                with self.assertRaisesRegex(RuntimeError, "Attribute-weighted candidate ordering.*not supported"):
                    index.set_attr_sort_alpha(alpha)
        self.assertEqual(index.get_attr_sort_alpha(), 0.0)

    def test_rng_ranks_by_vector_distance_without_attribute_similarity(self):
        data = np.zeros((82, 128), dtype=np.float32)
        data[0, 0], data[1, 0] = 1.0, 1.1
        for point in range(2, 81):
            data[point, point - 1] = 4 + 2 * point
        for near_value, far_value in ((100, 0), (0, 100)):
            with self.subTest(near_value=near_value, far_value=far_value):
                attrs = [[[0], [0]] for _ in range(82)]
                attrs[0], attrs[1] = [[near_value], [0]], [[far_value], [0]]
                index = hashannlib.Index(space="l2", dim=128)
                index.init_index(
                    max_elements=82, top_elements=1, M=40, ef_construction=300,
                    ft_bits=128, attr_type=[0, 1], max_cate_size=1, edge_level_ft=True,
                )
                index.initAttrMapping(attrs)
                index.add_items(
                    data[:81], attrs[:81], ids=np.arange(81),
                    levels=np.zeros(81, dtype=np.int32), num_threads=1,
                )
                index.add_items(data[81:], attrs[81:], ids=[81], levels=[0], num_threads=1)
                targets = set(map(int, index.get_neighbors(81)))
                self.assertIn(0, targets)
                self.assertNotIn(1, targets)

    def test_pruning_witnesses_follow_neighbor_identity_not_heap_position(self):
        data = np.zeros((82, 128), dtype=np.float32)
        for point in range(80):
            data[point, point] = 2 + 2 * point
        data[80, 0] = 3
        labels = 17 + 7 * np.arange(82, dtype=np.uint64)
        for edge_level in (False, True):
            for different_numeric_buckets in (False, True):
                with self.subTest(edge_level=edge_level, different_numeric_buckets=different_numeric_buckets):
                    attrs = [
                        [[100 if different_numeric_buckets and point == 0 else 0], [0, point % 19 + 1]]
                        for point in range(80)
                    ] + [[[100 if different_numeric_buckets else 0], [0, 20]], [[0], list(range(21))]]
                    index = hashannlib.Index(space="l2", dim=128)
                    index.init_index(
                        max_elements=82, top_elements=1, M=40, ef_construction=300,
                        ft_bits=128, attr_type=[0, 1], max_cate_size=21,
                        edge_level_ft=edge_level,
                    )
                    index.initAttrMapping(attrs)
                    index.add_items(
                        data[:81], attrs[:81], ids=labels[:81],
                        levels=np.zeros(81, dtype=np.int32), num_threads=1,
                    )
                    index.add_items(data[81:], attrs[81:], ids=labels[81:], levels=[0],
                                    num_threads=1)
                    value = attrs[80][0][0]
                    point_mask = np.array([
                        ord(c) for c in index.predicateToFT([[[value, value], [0, 20]]])[0]
                    ], dtype=np.uint8)
                    if edge_level:
                        markers = self.edge_markers(index)
                        owners = [
                            target for (source, target), marker in markers.items()
                            if source == 81 and np.all((marker & point_mask) == point_mask)
                        ]
                        self.assertEqual(owners, [0])
                        np.testing.assert_array_equal(markers[0, 80], point_mask)
                        index.update_point(int(labels[81]), data[81])
                        np.testing.assert_array_equal(self.edge_markers(index)[0, 80], point_mask)
                    else:
                        def node_markers():
                            state = index.__getstate__()[0]
                            rows = state["data_level0"].view(np.uint8).reshape(
                                82, state["size_data_per_element"],
                            )
                            offset = state["offset_level0"] + 4 + 4 * state["max_M0"]
                            return rows[:, offset:state["offset_data"]].copy()

                        markers = node_markers()
                        self.assertTrue(np.all((markers[0] & point_mask) == point_mask))
                        self.assertFalse(np.all((markers[79] & point_mask) == point_mask))
                        index.batch_update_attr([int(labels[0])], [attrs[0]], num_threads=1)
                        np.testing.assert_array_equal(node_markers()[0], markers[0])
                        index.update_attr(int(labels[0]), attrs[0])
                        np.testing.assert_array_equal(node_markers()[0], markers[0])

    def test_delete_items_reports_scrub_wall_time_per_batch(self):
        for threads in (1, 4):
            with self.subTest(threads=threads):
                index, _, _ = self.make_inplace_index()
                for start in (0, 8):
                    deleted = np.arange(start, start + 8, dtype=np.uint64)
                    started = time.perf_counter()
                    stats = index.delete_items(deleted, num_threads=threads)
                    elapsed = time.perf_counter() - started
                    self.assertIs(type(stats["scrub_wall_s"]), float)
                    self.assertGreater(stats["scrub_wall_s"], 0.0)
                    self.assertLessEqual(stats["scrub_wall_s"], elapsed)
                    self.assertEqual(stats["marked"], len(deleted))
                    self.assertTrue(all(type(value) is int for key, value in stats.items()
                                        if key != "scrub_wall_s"))
                    self.assert_live_graph(index, range(start + 8))
                self.assertEqual(index.delete_items([], num_threads=threads)["scrub_wall_s"], 0.0)

    def test_inplace_delete_removes_retired_navigation_before_return(self):
        index, data, attrs = self.make_inplace_index()
        deleted = np.arange(12, dtype=np.uint64)
        stats = index.delete_items(deleted, num_threads=4)
        self.assertEqual(stats["marked"], len(deleted))
        self.assertEqual(index.get_current_count(), len(data))
        self.assertEqual(index.get_dirty_count(), 0)
        self.assertEqual(index.repair_candidates_size(), 0)
        masks = np.asarray([
            [ord(char) for char in mask] for mask in index.predicateToFT(attrs)
        ], dtype=np.uint8)
        deleted_set = set(deleted)
        for (source, target), marker in self.edge_markers(index).items():
            self.assertNotIn(source, deleted_set)
            self.assertNotIn(target, deleted_set)
            self.assertNotEqual(source, target)
            np.testing.assert_array_equal(marker & masks[target], masks[target])
        for source in range(len(data)):
            neighbors = index.get_neighbors(source)
            self.assertEqual(len(neighbors), len(set(neighbors)))
            self.assertLessEqual(len(neighbors), 12)
        state = index.__getstate__()[0]
        upper = state["link_lists"].view(np.uint8)
        stride = state["size_links_per_element"]
        offset = 0
        for source, level_count in enumerate(state["element_levels"][:len(data)]):
            for _ in range(level_count):
                row = upper[offset:offset + stride]
                degree = int(row[:2].copy().view(np.uint16)[0])
                neighbors = row[4:4 + degree * 4].copy().view(np.uint32)
                if source in deleted_set:
                    self.assertEqual(degree, 0)
                self.assertFalse(np.isin(neighbors, deleted).any())
                offset += stride
        self.assertEqual(offset, upper.size)

        queries = data[20:28]
        labels, returned_distances = index.hybrid_knn_query(
            queries, [[[0]]] * len(queries), k=5, num_threads=1,
        )
        self.assertTrue(np.all(labels < len(data)))
        self.assertFalse(np.isin(labels, deleted).any())
        self.assertTrue(np.all(labels % 8 == 0))
        for row in labels:
            self.assertEqual(len(set(row)), 5)
        np.testing.assert_allclose(
            returned_distances, ((queries[:, None] - data[labels]) ** 2).sum(axis=2),
            rtol=1e-5,
        )
        self.assertTrue(np.all(np.diff(returned_distances, axis=1) >= 0))

    def test_inplace_delete_preserves_exact_results_on_a_complete_graph(self):
        # Sparse ANN graphs need not be fully reachable; exact recall needs a known topology.
        index, data, _ = self.make_inplace_index(M=48)
        for source in range(len(data)):
            self.assertEqual(len(index.get_neighbors(source)), len(data) - 1)
        deleted = np.arange(12, dtype=np.uint64)
        index.delete_items(deleted, num_threads=4)
        live = set(range(12, len(data)))
        for source in live:
            self.assertEqual(set(map(int, index.get_neighbors(source))), live - {source})
        queries = data[20:28]
        labels, _ = index.hybrid_knn_query(
            queries, [[[0]]] * len(queries), k=5, num_threads=1,
        )
        eligible = np.arange(16, len(data), 8)
        distances = ((queries[:, None] - data[eligible]) ** 2).sum(axis=2)
        expected = eligible[np.argsort(distances, axis=1)[:, :5]]
        np.testing.assert_array_equal(labels, expected)

    def test_empty_marker_cleanup_rows_preserve_target_attributes(self):
        for operation, count in (("cleanup", 70), ("delete", 70), ("cleanup", 160), ("delete", 160)):
            with self.subTest(operation=operation, count=count):
                index = self.make_tracking_index(count=count)
                if operation == "cleanup":
                    before = self.edge_markers(index)
                    index.batch_mark_deleted([0, 1], num_threads=2)
                    stats = index.cleanup_deleted_marker_bits([0, 1], num_threads=2)
                    after = self.edge_markers(index)
                    self.assertEqual(before.keys(), after.keys())
                    for edge in before:
                        np.testing.assert_array_equal(before[edge], after[edge])
                else:
                    stats = index.delete_items([0, 1], num_threads=4)
                    for source, target in self.edge_markers(index):
                        self.assertNotIn(source, (0, 1))
                        self.assertNotIn(target, (0, 1))
                self.assertGreater(stats["empty_marker_rows"], 0)
                self.assertEqual(stats["empty_marker_rows"], stats["candidate_sources"])
                self.assertEqual(stats["matched_edges"], 0)
                self.assertEqual(stats["cleared_bits"], 0)
                self.assertEqual(stats["support_checks"], 0)

    def test_inplace_delete_repairs_both_edge_directions(self):
        index, data, attrs = self.make_inplace_index()
        original = {
            source: set(index.get_neighbors(source)) for source in range(len(data))
        }
        def reachable_from(entry):
            reachable = {entry}
            pending = [entry]
            while pending:
                for target in original[pending.pop()]:
                    if target not in reachable:
                        reachable.add(target)
                        pending.append(target)
            return reachable

        state = index.__getstate__()[0]
        reachable = reachable_from(state["enterpoint_node"])
        for entry in np.flatnonzero(state["element_levels"][:len(data)]):
            self.assertEqual(reachable_from(int(entry)), reachable)
        reachable_ids = np.asarray(sorted(reachable))
        chosen = None
        for deleted in range(8, len(data)):
            distances = ((data[reachable_ids] - data[deleted]) ** 2).sum(axis=1)
            candidates = reachable_ids[np.argsort(distances)[:50]]
            incoming = {
                source for source in reachable if deleted in original[source]
            }
            plans = {}

            def add(source, target):
                plans.setdefault(source, original[source] - {deleted}).add(target)

            def replacements(point):
                eligible = candidates[(candidates != point) & (candidates != deleted)]
                distances = ((data[eligible] - data[point]) ** 2).sum(axis=1)
                return eligible[np.argsort(distances)[:3]]

            for source in incoming:
                plans[source] = original[source] - {deleted}
                for target in replacements(source):
                    add(source, int(target))
            outgoing_only = []
            for target in original[deleted]:
                for source in replacements(target):
                    source = int(source)
                    add(source, target)
                    if source not in incoming and target not in original[source]:
                        outgoing_only.append((source, target))
            outgoing_only = [
                edge for edge in outgoing_only if len(plans[edge[0]]) <= 12
            ]
            if outgoing_only:
                chosen = deleted, incoming, plans, outgoing_only
                break
        self.assertIsNotNone(chosen)
        deleted, incoming, plans, outgoing_only = chosen
        before = self.edge_markers(index)
        stats = index.delete_items([deleted], num_threads=4)
        self.assertEqual(stats["incoming_edges_repaired"], len(incoming))
        self.assertGreater(stats["outgoing_edges_added"], 0)
        after = self.edge_markers(index)
        for source, neighbors in plans.items():
            if len(neighbors) <= 12:
                self.assertEqual(set(index.get_neighbors(source)), neighbors)
        for edge in outgoing_only:
            self.assertIn(edge, after)
        self.assertEqual(index.get_neighbors(deleted).tolist(), [])

        masks = np.asarray([
            [ord(char) for char in mask] for mask in index.predicateToFT(attrs)
        ], dtype=np.uint8)
        alive = np.arange(len(data)) != deleted
        for (source, target), marker in after.items():
            old = before.get((source, target), np.zeros_like(marker))
            added = marker & ~(old | masks[target])
            geometry = ((data - data[target]) ** 2).sum(axis=1) < (
                (data - data[source]) ** 2
            ).sum(axis=1)
            supporters = alive & geometry
            supporters[source] = False
            supporters[target] = False
            supported = np.bitwise_or.reduce(
                masks[supporters], axis=0, initial=np.uint8(0),
            )
            np.testing.assert_array_equal(added & supported, added)

    def test_inplace_delete_handles_last_upper_entry_and_empty_index(self):
        index, data = self.make_index()
        index.delete_items([0], num_threads=2)
        labels, _ = index.hybrid_knn_query(data[1:2], [[[1]]], k=1, num_threads=1)
        self.assertEqual(labels[0, 0], 1)
        labels, _ = index.knn_query(data[1:2], k=1, num_threads=1)
        self.assertEqual(labels[0, 0], 1)
        labels, _ = index.hybrid_knn_query_dnf(
            data[1:2], [[[[]]]], k=1, num_threads=1,
        )
        self.assertEqual(labels[0, 0], 1)
        index.delete_items([1, 2, 3], num_threads=2)
        for source in range(4):
            self.assertEqual(index.get_neighbors(source).tolist(), [])
        labels, _, _, hops = index.hybrid_knn_query_with_stats(data[:1], [[[0]]], k=1)
        self.assertEqual(labels[0, 0], np.iinfo(np.uint64).max)
        self.assertEqual(hops[0], 0)
        index.resize_index(5)
        index.add_items(
            data[:1], [[[0]]], ids=[4], levels=np.asarray([1], dtype=np.int32),
            num_threads=1,
        )
        labels, _ = index.hybrid_knn_query(data[:1], [[[0]]], k=1, num_threads=1)
        self.assertEqual(labels[0, 0], 4)

    def test_inplace_delete_parallel_points_preserve_live_navigation(self):
        serial, data, attrs = self.make_inplace_index()
        parallel, _, _ = self.make_inplace_index()
        labels = np.arange(24, dtype=np.uint64)
        serial_stats = serial.delete_items(labels, num_threads=1)
        parallel_stats = parallel.delete_items(labels, num_threads=4)
        for stats in (serial_stats, parallel_stats):
            self.assertEqual(stats["requested"], len(labels))
            self.assertEqual(stats["marked"], len(labels))
        self.assertEqual(serial_stats["parallel_preparations"], 0)
        self.assertEqual(parallel_stats["parallel_preparations"], 0)
        self.assertEqual(parallel_stats["recomputed_preparations"], 0)
        self.assertEqual(parallel_stats["parallel_deletions"], len(labels))
        self.assertGreaterEqual(parallel_stats["peak_parallel_deletions"], 2)
        for index in (serial, parallel):
            self.assert_live_graph(index, labels, self.own_masks(index, attrs, [1]))
        queries = data[32:48]
        predicates = attrs[32:48]
        serial_labels, _ = serial.hybrid_knn_query(queries, predicates, k=1, num_threads=1)
        parallel_labels, _ = parallel.hybrid_knn_query(queries, predicates, k=1, num_threads=1)
        np.testing.assert_array_equal(serial_labels, parallel_labels)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "index.bin")
            parallel.save_index(path)
            restored = hashannlib.Index(space="l2", dim=4)
            restored.load_index(path, max_elements=len(data), top_elements=8, dynamic=True)
            restored.delete_items([25, 26], num_threads=2)
            for source, target in self.edge_markers(restored):
                self.assertNotIn(source, set(labels) | {25, 26})
                self.assertNotIn(target, set(labels) | {25, 26})

    def test_parallel_point_deletion_preserves_separated_regions(self):
        rng = np.random.default_rng(415)
        count = 512
        data = rng.integers(0, 64, (count, 8)).astype(np.float32)
        data[count // 2:] += 128
        attrs = [[[i % 31], [i % 8]] for i in range(count)]
        ids = rng.permutation(count).astype(np.uint64)
        levels = np.zeros(count, dtype=np.int32)
        levels[[0, count // 2]] = 1

        def build():
            index = hashannlib.Index(space="l2", dim=8)
            index.init_index(
                max_elements=count, top_elements=2, M=8, ef_construction=48,
                ft_bits=128, attr_type=[0, 1], max_cate_size=8, edge_level_ft=True,
            )
            index.initAttrMapping(attrs)
            index.add_items(data, attrs, ids=ids, levels=levels, num_threads=1)
            index.set_ef(64)
            return index

        serial, parallel = build(), build()
        deleted = ids[[40, 296, 75, 330, 100, 390, 160, 450]]
        serial_stats = serial.delete_items(deleted, num_threads=1)
        parallel_stats = parallel.delete_items(deleted, num_threads=2)
        for stats in (serial_stats, parallel_stats):
            self.assertEqual(stats["marked"], len(deleted))
        self.assertEqual(parallel_stats["parallel_deletions"], len(deleted))
        self.assertEqual(parallel_stats["recomputed_preparations"], 0)
        deleted_internal = np.flatnonzero(np.isin(ids, deleted))
        for index in (serial, parallel):
            self.assert_live_graph(index, deleted_internal, self.own_masks(index, attrs, [0, 1]))
        predicates = [[[[0, 30], [3]], [[], [4]]]] * 8
        serial_result = serial.hybrid_knn_query_dnf(data[200:208], predicates, k=2, num_threads=1)
        parallel_result = parallel.hybrid_knn_query_dnf(data[200:208], predicates, k=2, num_threads=1)
        eligible = ids[
            ~np.isin(ids, deleted) & np.isin(np.arange(count) % 8, (3, 4))
        ]
        self.assert_valid_query(serial, data[200:208], serial_result, eligible)
        self.assert_valid_query(parallel, data[200:208], parallel_result, eligible)

    def test_shared_deletion_matching_preserves_semantics_across_layouts_and_batches(self):
        count = 193
        rng = np.random.default_rng(762)
        data = rng.random((count, 8), dtype=np.float32)
        ids = np.arange(count, dtype=np.uint64) + 1000
        levels = np.zeros(count, dtype=np.int32)
        levels[:4] = 1
        deleted = ids[rng.permutation(count)[:14]]
        layouts = ([0, 1], [1, 0], [1, 1], [0, 1, 0])
        for ft_bits in (8, 72, 128):
            for attr_type in layouts:
                for space in ("l2", "ip", "cosine"):
                    with self.subTest(ft_bits=ft_bits, attr_type=attr_type, space=space):
                        attrs = [
                            [
                                [(point * (column + 1)) % 43] if kind == 0
                                else [point % 11, (3 * point + 5 + column) % 11]
                                for column, kind in enumerate(attr_type)
                            ]
                            for point in range(count)
                        ]
                        initial = hashannlib.Index(space=space, dim=8)
                        initial.init_index(
                            max_elements=count, top_elements=4, M=6, ef_construction=139,
                            ft_bits=ft_bits, attr_type=attr_type, max_cate_size=11,
                            edge_level_ft=True,
                        )
                        initial.initAttrMapping(attrs)
                        initial.add_items(data, attrs, ids=ids, levels=levels, num_threads=1)
                        with tempfile.TemporaryDirectory() as directory:
                            path = Path(directory) / "initial.index"
                            initial.save_index(str(path))
                            graphs, queries, work = [], [], []
                            for threads in (1, 1, 4):
                                index = hashannlib.Index(space=space, dim=8)
                                index.load_index(
                                    str(path), max_elements=count, top_elements=4, dynamic=True,
                                )
                                first = index.delete_items(deleted[:7], num_threads=threads)
                                self.assertGreater(first["candidate_sources"], 64)
                                changed = next(i for i in range(count) if ids[i] not in deleted)
                                index.batch_update_attr(
                                    [int(ids[changed])], [attrs[(changed + 17) % count]],
                                    num_threads=threads,
                                )
                                second = index.delete_items(deleted[7:], num_threads=threads)
                                current_attrs = list(attrs)
                                current_attrs[changed] = attrs[(changed + 17) % count]
                                self.assert_live_graph(
                                    index, deleted - 1000,
                                    self.own_masks(index, current_attrs, attr_type),
                                )
                                work.append((first, second))
                                state = index.__getstate__()[0]
                                graphs.append((state["data_level0"], state["link_lists"]))
                                index.generateAttrIndexes()
                                index.set_ef(192)
                                predicate = [
                                    [0] if column == attr_type.index(1) else []
                                    for column in range(len(attr_type))
                                ]
                                queries.append(index.hybrid_knn_query_dnf(
                                    data[32:37], [[predicate]] * 5, k=3, num_threads=1,
                                ))
                                categorical = attr_type.index(1)
                                eligible = ids[
                                    np.asarray([0 in record[categorical] for record in current_attrs])
                                    & ~np.isin(ids, deleted)
                                ]
                                self.assert_valid_query(
                                    index, data[32:37], queries[-1], eligible, space,
                                )
                            # Point-parallel interleavings need not reproduce a serial graph.
                            # Serial cache reuse must remain deterministic.
                            for reference, repeated in zip(work[0], work[1]):
                                self.assert_same_deletion_work(reference, repeated)
                            for reference, repeated in zip(graphs[0], graphs[1]):
                                np.testing.assert_array_equal(reference, repeated)
                            for reference, repeated in zip(queries[0], queries[1]):
                                np.testing.assert_array_equal(reference, repeated)

    def test_parallel_deletion_of_dense_shared_rows_preserves_every_live_edge(self):
        count = 96
        rng = np.random.default_rng(1261)
        data = rng.random((count, 6), dtype=np.float32)
        attrs = [[[i % 13], [i % 5, (i + 2) % 5]] for i in range(count)]
        ids = 1000 + np.arange(count, dtype=np.uint64) * 17
        levels = np.zeros(count, dtype=np.int32)
        levels[:8] = 1
        initial = hashannlib.Index(space="l2", dim=6)
        initial.init_index(
            max_elements=count, top_elements=8, M=64, ef_construction=160,
            ft_bits=128, attr_type=[0, 1], max_cate_size=5, edge_level_ft=True,
        )
        initial.initAttrMapping(attrs)
        initial.add_items(data, attrs, ids=ids, levels=levels, num_threads=1)
        for source in range(count):
            self.assertEqual(len(initial.get_neighbors(source)), count - 1)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "complete.index")
            initial.save_index(path)
            for repetition in range(3):
                with self.subTest(repetition=repetition):
                    index = hashannlib.Index(space="l2", dim=6)
                    index.load_index(path, max_elements=count, top_elements=8, dynamic=True)
                    deleted = np.concatenate((np.arange(8), rng.permutation(np.arange(8, count))[:56]))
                    stats = index.delete_items(ids[deleted], num_threads=32)
                    self.assertEqual(stats["marked"], len(deleted))
                    masks = self.own_masks(index, attrs, [0, 1])
                    self.assert_live_graph(index, deleted, masks)
                    live = np.flatnonzero(~np.isin(np.arange(count), deleted))
                    for source in live:
                        self.assertEqual(set(map(int, index.get_neighbors(int(source)))), set(live) - {source})
                    for (_, target), marker in self.edge_markers(index).items():
                        np.testing.assert_array_equal(marker, masks[target])
                    index.generateAttrIndexes()
                    index.set_ef(count * 2)
                    index.set_thresholds(0.0001, 0.0001, 0.0001)
                    query = data[live[:6]]
                    eligible = live[[0 in attrs[i][1] for i in live]]
                    found = index.hybrid_knn_query_dnf(
                        query, [[[[], [0]]]] * len(query), k=3, num_threads=1,
                    )
                    distances = ((query[:, None] - data[eligible]) ** 2).sum(axis=2)
                    expected = ids[eligible[np.argsort(distances, axis=1)[:, :3]]]
                    np.testing.assert_array_equal(found[0], expected)
                    self.assert_valid_query(index, query, found, ids[eligible])

    def test_parallel_deletion_can_retire_all_upper_entries_and_all_points(self):
        count = 70
        for edge_level in (False, True):
            for survivors in (0, 1):
                with self.subTest(edge_level=edge_level, survivors=survivors):
                    data = np.column_stack((np.arange(count), np.zeros(count))).astype(np.float32)
                    attrs = [[[0]] for _ in range(count)]
                    levels = np.ones(count, dtype=np.int32)
                    ids = np.arange(count, dtype=np.uint64) + 200
                    index = hashannlib.Index(space="l2", dim=2)
                    index.init_index(
                        max_elements=count + 1, top_elements=count, M=80, ef_construction=160,
                        ft_bits=72, attr_type=[1], max_cate_size=4, edge_level_ft=edge_level,
                    )
                    index.initAttrMapping(attrs + [attrs[-1]])
                    index.add_items(data, attrs, ids=ids, levels=levels, num_threads=1)
                    deleted = np.arange(count - survivors, dtype=np.uint64)
                    stats = index.delete_items(ids[deleted], num_threads=32)
                    self.assertEqual(stats["marked"], len(deleted))
                    self.assert_live_graph(
                        index, deleted, self.own_masks(index, attrs, [1]) if edge_level else None,
                    )
                    if survivors:
                        index.set_ef(count * 2)
                        labels, _ = index.knn_query(data[-1:], k=1, num_threads=1)
                        self.assertEqual(labels[0, 0], ids[-1])
                        index.delete_items(ids[-1:], num_threads=4)
                    self.assert_live_graph(index, np.arange(count))
                    index.add_items(
                        data[-1:], attrs[-1:], ids=[9999], levels=[1],
                        num_threads=1, replace_deleted=False,
                    )
                    index.set_ef(count * 2)
                    labels, _ = index.knn_query(data[-1:], k=1, num_threads=1)
                    self.assertEqual(labels[0, 0], 9999)

    def test_parallel_deletion_merges_disjoint_repairs_to_the_same_source(self):
        count = 129
        rng = np.random.default_rng(64201)
        data = np.zeros((count, 2), dtype=np.float32)
        data[1:65, 0] = np.sort(rng.uniform(-12, -8, 64)).astype(np.float32)
        data[65:, 0] = np.sort(rng.uniform(8, 12, 64)).astype(np.float32)
        attrs = [[[0]] for _ in range(count)]
        initial = hashannlib.Index(space="l2", dim=2)
        initial.init_index(
            max_elements=count, top_elements=1, M=8, ef_construction=96,
            ft_bits=8, attr_type=[1], max_cate_size=1, edge_level_ft=True,
        )
        initial.initAttrMapping(attrs)
        initial.add_items(
            data, attrs, ids=np.arange(count), levels=np.zeros(count, dtype=np.int32),
            num_threads=1,
        )
        state = initial.__getstate__()[0]
        raw = state["data_level0"].copy()
        rows = raw.view(np.uint8).reshape(count, state["size_data_per_element"])
        offset = state["offset_level0"]
        marker_offset = offset + 4 + 4 * state["max_M0"]
        masks = self.own_masks(initial, attrs, [1])
        adjacency = {0: [32, 97]}
        for start in (1, 65):
            for point in range(start, start + 64):
                adjacency[point] = [
                    start + ((point - start + delta) % 64) for delta in (-2, -1, 1, 2)
                ]
            adjacency[start].append(0)
        for source, neighbors in adjacency.items():
            row = rows[source]
            row[offset:offset + 4] = 0
            row[offset:offset + 2] = np.asarray([len(neighbors)], dtype=np.uint16).view(np.uint8)
            row[offset + 4:marker_offset] = 0
            row[offset + 4:offset + 4 + 4 * len(neighbors)] = np.asarray(
                neighbors, dtype=np.uint32,
            ).view(np.uint8)
            marker_rows = row[marker_offset:state["offset_data"]].reshape(state["max_M0"], -1)
            marker_rows[:] = 0
            marker_rows[:len(neighbors)] = masks[neighbors]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "two-regions.index"
            initial.save_index(str(path))
            payload = bytearray(path.read_bytes())
            data_offset = (
                len(payload) - state["data_level0"].nbytes
                - state["link_lists"].nbytes - 4 * count
            )
            self.assertEqual(struct.unpack_from("<i", payload, data_offset - 4)[0], 10)
            self.assertEqual(
                payload[data_offset:data_offset + raw.nbytes],
                state["data_level0"].tobytes(),
            )
            payload[data_offset:data_offset + raw.nbytes] = raw.tobytes()
            path.write_bytes(payload)
            deleted = np.asarray([32, 97], dtype=np.uint64)
            required = set()
            for point in deleted:
                nearest = np.argsort(((data - data[point]) ** 2).sum(axis=1), kind="stable")[:50]
                candidates = nearest[~np.isin(nearest, deleted) & (nearest != 0)]
                chosen = candidates[np.argsort((data[candidates] ** 2).sum(axis=1), kind="stable")[:3]]
                required.update(map(int, chosen))
            self.assertEqual(len(required), 6)
            for repetition in range(12):
                with self.subTest(repetition=repetition):
                    index = hashannlib.Index(space="l2", dim=2)
                    index.load_index(str(path), max_elements=count, top_elements=1, dynamic=True)
                    index.delete_items(deleted, num_threads=2)
                    self.assert_live_graph(index, deleted, masks)
                    self.assertTrue(required.issubset(set(map(int, index.get_neighbors(0)))))

    def test_inplace_delete_overflow_markers_have_live_geometric_support(self):
        index, data, attrs = self.make_inplace_index(M=2)
        before = self.edge_markers(index)
        deleted = np.arange(8, 16, dtype=np.uint64)
        stats = index.delete_items(deleted, num_threads=4)
        self.assertGreater(stats["pruned_edges"], 0)
        masks = np.asarray([
            [ord(char) for char in mask] for mask in index.predicateToFT(attrs)
        ], dtype=np.uint8)
        alive = ~np.isin(np.arange(len(data)), deleted)
        for (source, target), marker in self.edge_markers(index).items():
            self.assertTrue(alive[source] and alive[target])
            self.assertLessEqual(index.get_neighbors(source).size, 4)
            np.testing.assert_array_equal(marker & masks[target], masks[target])
            old = before.get((source, target), np.zeros_like(marker))
            added = marker & ~(old | masks[target])
            geometry = ((data - data[target]) ** 2).sum(axis=1) < (
                (data - data[source]) ** 2
            ).sum(axis=1)
            supporters = alive & geometry
            supporters[source] = False
            supporters[target] = False
            supported = np.bitwise_or.reduce(
                masks[supporters], axis=0, initial=np.uint8(0),
            )
            np.testing.assert_array_equal(added & supported, added)

    def test_inplace_delete_does_not_duplicate_known_witness_routes(self):
        for attribute_count in (1, 2):
            with self.subTest(attribute_count=attribute_count):
                index, data, attrs = self.make_inplace_index(
                    unique_attributes=True, attribute_count=attribute_count,
                )
                masks = np.asarray([
                    [ord(char) for char in mask] for mask in index.predicateToFT(attrs)
                ], dtype=np.uint8)
                before = self.edge_markers(index)
                index.delete_items([8], num_threads=4)
                after = self.edge_markers(index)
                additions = 0
                for source in range(len(data)):
                    targets = {target for owner, target in after if owner == source}
                    for witness in range(len(data)):
                        if witness in targets or witness in (source, 8):
                            continue
                        new_owners, retained_owners = [], []
                        for target in targets:
                            marker = after[source, target]
                            if not np.all((marker & masks[witness]) == masks[witness]):
                                continue
                            old = before.get((source, target), np.zeros_like(marker))
                            if not np.all((old & masks[witness]) == masks[witness]):
                                new_owners.append(target)
                            elif np.sum((data[target] - data[witness]) ** 2) < np.sum(
                                (data[source] - data[witness]) ** 2
                            ):
                                retained_owners.append(target)
                        additions += len(new_owners)
                        self.assertLessEqual(
                            len(new_owners), 1, (source, witness, new_owners),
                        )
                        if retained_owners:
                            self.assertEqual(
                                new_owners, [], (source, witness, retained_owners),
                            )
                self.assertGreater(additions, 0)

    def assert_deletion_tracking(self, index, seen_dirty):
        state = index.__getstate__()[0]
        count = state["cur_element_count"]
        rows = state["data_level0"].view(np.uint8).reshape(
            count, state["size_data_per_element"]
        )
        deleted = (rows[:, state["offset_level0"] + 2] & 1) != 0
        degrees = np.zeros(count, dtype=np.int64)
        dead_degrees = np.zeros(count, dtype=np.int64)
        for source, target in self.edge_markers(index):
            degrees[source] += 1
            dead_degrees[source] += deleted[target]
        expected = {
            source: float(dead_degrees[source]) / degrees[source]
            for source in range(count) if dead_degrees[source]
        }
        index.set_repair_threshold(0)
        index.clear_repair_candidates()
        index.set_ef(count * 2)
        labels, _ = index.hybrid_knn_query(
            np.zeros((1, 2), dtype=np.float32), [[[0]]], k=1, num_threads=1,
        )
        label_offset = state["label_offset"]
        stored_labels = rows[:, label_offset:label_offset + 8].copy().view(np.uint64)
        self.assertIn(labels[0, 0], stored_labels[~deleted])
        self.assertAlmostEqual(index.get_deleted_ratio(), float(deleted.mean()))
        actual = index.get_repair_candidates()
        self.assertEqual(actual.keys(), expected.keys())
        for source, ratio in expected.items():
            self.assertAlmostEqual(actual[source], ratio, places=6)
        seen_dirty.update(expected)
        self.assertEqual(index.get_dirty_count(), len(seen_dirty))

    def test_deletion_tracking_survives_updates_resize_and_rebuild(self):
        index = self.make_tracking_index(count=6)
        seen_dirty = set()
        index.mark_deleted(4)
        self.assert_deletion_tracking(index, seen_dirty)
        index.batch_mark_deleted([1, 5], num_threads=2)
        self.assert_deletion_tracking(index, seen_dirty)
        index.unmark_deleted(4)
        self.assert_deletion_tracking(index, seen_dirty)

        index.resize_index(130)
        self.assert_deletion_tracking(index, seen_dirty)
        ids = np.arange(6, 70)
        data = np.column_stack((ids, np.zeros(len(ids)))).astype(np.float32)
        index.add_items(
            data, [[[0]]] * len(ids), ids=ids,
            levels=np.zeros(len(ids), dtype=np.int32), num_threads=1,
        )
        index.mark_deleted(65)
        self.assert_deletion_tracking(index, seen_dirty)
        index.unmark_deleted(65)
        self.assert_deletion_tracking(index, seen_dirty)
        index.resize_index(70)
        self.assert_deletion_tracking(index, seen_dirty)

        index.delete_items([68], num_threads=2)
        seen_dirty.clear()
        self.assert_deletion_tracking(index, seen_dirty)
        index.rebuild_graph()
        self.assertEqual(index.get_current_count(), 67)
        seen_dirty.clear()
        self.assert_deletion_tracking(index, seen_dirty)
        index.mark_deleted(69)
        self.assert_deletion_tracking(index, seen_dirty)

    def test_deletion_tracking_restored_from_inline_flags(self):
        index = self.make_tracking_index()
        index.batch_mark_deleted([1, 63, 64, 65], num_threads=4)
        self.assert_deletion_tracking(index, set())
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "index.bin")
            index.save_index(path)
            for dynamic in (False, True):
                with self.subTest(dynamic=dynamic):
                    restored = hashannlib.Index(space="l2", dim=2)
                    restored.load_index(
                        path, max_elements=130, top_elements=3, dynamic=dynamic,
                    )
                    seen_dirty = set()
                    self.assert_deletion_tracking(restored, seen_dirty)
                    if dynamic:
                        restored.resize_index(130)
                        restored.unmark_deleted(64)
                        restored.batch_mark_deleted([62, 66], num_threads=2)
                        self.assert_deletion_tracking(restored, seen_dirty)

    def test_deletion_tracking_clears_reactivated_slots(self):
        index = self.make_tracking_index()
        index.batch_mark_deleted([62, 63, 64, 65, 66], num_threads=4)
        seen_dirty = set()
        self.assert_deletion_tracking(index, seen_dirty)
        ids = np.arange(62, 66)
        data = np.column_stack((ids + 0.25, np.zeros(len(ids)))).astype(np.float32)
        index.add_items(
            data, [[[0]]] * len(ids), ids=ids, num_threads=1,
        )
        self.assertEqual(index.get_current_count(), 70)
        self.assertAlmostEqual(index.get_deleted_ratio(), 1 / 70)
        self.assert_deletion_tracking(index, seen_dirty)

    def test_parallel_deletion_tracking_preserves_shared_bitmap_words(self):
        index = self.make_tracking_index()
        ids = np.arange(1, 65, dtype=np.uint64)
        seen_dirty = set()
        for _ in range(4):
            self.assertEqual(index.batch_mark_deleted(ids, num_threads=16), len(ids))
            self.assert_deletion_tracking(index, seen_dirty)
            for label in ids:
                index.unmark_deleted(int(label))
            index.mark_deleted(69)
            self.assert_deletion_tracking(index, seen_dirty)
            index.unmark_deleted(69)

    def make_registration_index(self, capacity=8, edge_level=True, M=2):
        data = np.asarray([[0, 0], [10, 0]], dtype=np.float32)
        attrs = [[[0], [0]], [[10], [0]]]
        index = hashannlib.Index(space="l2", dim=2)
        index.init_index(
            max_elements=capacity, top_elements=1, M=M, ef_construction=max(40, 2 * M),
            ft_bits=8, attr_type=[0, 1], max_cate_size=12, edge_level_ft=edge_level,
        )
        index.initAttrMapping(attrs)
        index.add_items(data, attrs, ids=[0, 1], num_threads=1)
        index.set_ef(200)
        index.set_ft_flag(True)
        index.set_ft_routing_flag(True)
        index.set_ft_routing_min_deg(16)
        return index, data

    def registration_mask(self, index, label):
        return self.own_masks(index, [[[0], [label]]], [0, 1])[0, 1:]

    def test_unseen_label_registration_is_once_only_and_does_not_change_graph(self):
        index, data = self.make_registration_index()
        original_mask = self.registration_mask(index, 0)
        original_rows = index.__getstate__()[0]["data_level0"].copy()
        self.assertFalse(self.registration_mask(index, 7).any())
        self.assertEqual(index.register_attr_values([[[100], [7, 8, 7]]]), 2)
        self.assertTrue(self.registration_mask(index, 7).any())
        self.assertFalse(np.array_equal(
            self.registration_mask(index, 7), self.registration_mask(index, 8)))
        self.assertEqual(index.register_attr_values([[[100], [7, 8]]]), 0)
        np.testing.assert_array_equal(self.registration_mask(index, 0), original_mask)
        np.testing.assert_array_equal(index.__getstate__()[0]["data_level0"], original_rows)
        np.testing.assert_array_equal(index.get_items([0, 1]), data)
        with self.assertRaisesRegex(RuntimeError, "Cannot remap a populated index"):
            index.initAttrMapping([[[100], [7]]])
        with self.assertRaisesRegex(RuntimeError, r"outside \[0, 12\)"):
            index.register_attr_values([[[1], [9]], [[1], [12]]])
        self.assertFalse(self.registration_mask(index, 9).any())

    def test_parallel_insertion_registers_labels_before_marker_construction(self):
        # Keep the fixture below its degree budget so ANN pruning is not the variable under test.
        index, _ = self.make_registration_index(capacity=70, M=40)
        old_mask = self.registration_mask(index, 0)
        values = np.arange(100, 164)
        data = np.column_stack((values, np.zeros(len(values)))).astype(np.float32)
        attrs = [[[int(value)], [7 + i % 2]] for i, value in enumerate(values)]
        ids = np.arange(2, 66, dtype=np.uint64)
        index.add_items(data, attrs, ids=ids, num_threads=4)
        self.assertEqual(index.register_attr_values(attrs), 0)
        np.testing.assert_array_equal(self.registration_mask(index, 0), old_mask)
        predicates = [
            [[[int(values[i]), int(values[i])], attrs[i][1]]]
            for i in (0, len(values) - 1)
        ]
        labels, _ = index.hybrid_knn_query_dnf(
            data[[0, -1]], predicates, k=1, num_threads=1)
        np.testing.assert_array_equal(labels[:, 0], ids[[0, -1]])

    def test_all_attribute_update_entries_register_unseen_labels(self):
        for method in ("update_attr", "batch_update_attr", "update_point_attr",
                       "batch_update_point_attr", "add_items"):
            with self.subTest(method=method):
                index, data = self.make_registration_index()
                old_mask = self.registration_mask(index, 0)
                attrs = [[-100], [7]]
                vector = np.asarray([-100, 0], dtype=np.float32)
                if method == "update_attr":
                    index.update_attr(0, attrs)
                elif method == "batch_update_attr":
                    self.assertEqual(index.batch_update_attr([0], [attrs], num_threads=2), 1)
                elif method == "update_point_attr":
                    index.update_point_attr(0, vector, attrs)
                elif method == "batch_update_point_attr":
                    self.assertEqual(index.batch_update_point_attr(
                        [0], vector[None, :], [attrs], num_threads=2), 1)
                else:
                    index.add_items(vector[None, :], [attrs], ids=[0], num_threads=1)
                self.assertEqual(index.register_attr_values([attrs]), 0)
                self.assertTrue(self.registration_mask(index, 7).any())
                np.testing.assert_array_equal(self.registration_mask(index, 0), old_mask)
                result = index.hybrid_knn_query_dnf(
                    data[:1], [[[[-100, -100], [7]]]], k=1, num_threads=1)
                np.testing.assert_array_equal(result[0], [[0]])

    def test_unseen_query_labels_do_not_allocate_or_corrupt_marker_masks(self):
        index, data = self.make_registration_index()
        predicates = [[[[0, 10], [0, 7]]]]
        labels, _ = index.hybrid_knn_query_dnf(
            data[1:2], predicates, check_modes=[[[0, 0]]], k=1, num_threads=1)
        np.testing.assert_array_equal(labels, [[1]])
        with self.assertRaisesRegex(RuntimeError, "Cannot return"):
            index.hybrid_knn_query_dnf(data[:1], predicates, k=1, num_threads=1)
        self.assertFalse(self.registration_mask(index, 7).any())
        self.assertEqual(index.register_attr_values([[[0], [7]]]), 1)
        with self.assertRaisesRegex(RuntimeError, r"outside \[0, 12\)"):
            index.predicateToFT([[[0, 10], [12]]])

    def test_incremental_assignments_survive_native_save_load(self):
        from exp_benchmark.index_metadata import read_layout
        from exp_benchmark.dynamic.runtime import read_attribute_index_header

        for edge_level in (False, True):
            with self.subTest(edge_level=edge_level), tempfile.TemporaryDirectory() as directory:
                index, _ = self.make_registration_index(edge_level=edge_level)
                index.register_attr_values([[[100], [7]]])
                masks = {label: self.registration_mask(index, label) for label in (0, 7, 8)}
                path = str(Path(directory) / "incremental.bin")
                index.save_index(path)
                header = read_layout(path, include_mappings=True)
                self.assertEqual(header["format"], 12 if edge_level else 11)
                self.assertEqual(header["counting_hash_table_mapping"][1][8], -1)
                if edge_level:
                    self.assertEqual(read_attribute_index_header(path)["format"], 12)
                restored = hashannlib.Index(space="l2", dim=2)
                restored.load_index(path, max_elements=8, top_elements=1, dynamic=True)
                for label, expected in masks.items():
                    np.testing.assert_array_equal(self.registration_mask(restored, label), expected)
                self.assertEqual(restored.register_attr_values([[[100], [7]]]), 0)
                self.assertEqual(restored.register_attr_values([[[100], [8]]]), 1)
                np.testing.assert_array_equal(self.registration_mask(restored, 7), masks[7])
                restored.add_items([[100, 0]], [[[100], [7]]], ids=[2], num_threads=1)
                restored.set_ef(40)
                restored.set_ft_flag(True)
                restored.set_ft_routing_flag(True)
                restored.set_ft_routing_min_deg(16)
                found, _ = restored.hybrid_knn_query_dnf(
                    [[100, 0]], [[[[100, 100], [7]]]], k=1, num_threads=1)
                np.testing.assert_array_equal(found, [[2]])
                restored.register_attr_values([[[100], list(range(12))]])
                full_path = str(Path(directory) / "complete.bin")
                restored.save_index(full_path)
                self.assertEqual(read_layout(full_path)["format"], 10 if edge_level else 9)

    def test_full_codebook_assigns_new_labels_without_remapping_old_labels(self):
        index, _ = self.make_registration_index()
        index.register_attr_values([[[0], list(range(8))]])
        old = [self.registration_mask(index, label) for label in range(8)]
        self.assertEqual(index.register_attr_values([[[0], [8, 9]]]), 2)
        np.testing.assert_array_equal(self.registration_mask(index, 8), old[0])
        np.testing.assert_array_equal(self.registration_mask(index, 9), old[1])
        for label, expected in enumerate(old):
            np.testing.assert_array_equal(self.registration_mask(index, label), expected)

    def test_invalid_attribute_batches_fail_before_registration_or_mutation(self):
        index, data = self.make_registration_index()
        rows = index.__getstate__()[0]["data_level0"].copy()
        with self.assertRaisesRegex(RuntimeError, "label not found"):
            index.batch_update_point_attr(
                [0, 999], data + 100, [[[0], [7]], [[10], [8]]], num_threads=2)
        self.assertFalse(self.registration_mask(index, 7).any())
        np.testing.assert_array_equal(index.__getstate__()[0]["data_level0"], rows)
        with self.assertRaisesRegex(RuntimeError, "attribute rows"):
            index.add_items(data, [[[0], [7]]], ids=[2, 3], num_threads=2)
        with self.assertRaisesRegex(RuntimeError, r"outside \[0, 12\)"):
            index.add_items(data, [[[0], [7]], [[10], [12]]], ids=[2, 3], num_threads=2)
        self.assertEqual(index.get_current_count(), 2)
        self.assertFalse(self.registration_mask(index, 7).any())

    def test_attribute_update_requires_complete_valid_record(self):
        index, _ = self.make_index()

        index.update_attr(0, [[2, 3]])
        with self.assertRaisesRegex(RuntimeError, "expected 1 attributes"):
            index.update_attr(0, [])
        with self.assertRaisesRegex(RuntimeError, r"outside \[0, 4\)"):
            index.update_attr(0, [[4]])

    def test_batch_attribute_update_reports_failure(self):
        index, _ = self.make_index()

        updated = index.batch_update_attr(
            np.asarray([0, 1], dtype=np.int64),
            [[[1]], [[2, 3]]],
            num_threads=2,
        )
        self.assertEqual(updated, 2)
        with self.assertRaisesRegex(RuntimeError, "expected 1 attributes"):
            index.batch_update_attr(
                np.asarray([0, 1], dtype=np.int64),
                [[[1]], []],
                num_threads=2,
            )

    def test_maintenance_does_not_repeat_without_new_deletes(self):
        index, _ = self.make_index()
        index.set_maintenance_thresholds(
            patch_ratio=0.1,
            rebuild_ratio=1.0,
            patch_min_new=1,
        )
        index.mark_deleted(3)

        self.assertEqual(index.maintain_deletes(num_threads=1), "patch")
        self.assertEqual(index.maintain_deletes(num_threads=1), "none")

    def test_delete_items_does_not_trigger_legacy_patch_or_rebuild(self):
        index, _ = self.make_index()
        index.set_maintenance_thresholds(
            patch_ratio=0.1,
            rebuild_ratio=0.1,
            patch_min_new=1,
        )
        before = self.edge_markers(index)
        stats = index.delete_items(
            np.asarray([3], dtype=np.int64),
            num_threads=1,
        )

        self.assertEqual(stats["marked"], 1)
        self.assertIn("candidate_sources", stats)
        self.assertIn("matched_edges", stats)
        self.assertIn("cleared_bits", stats)
        self.assertEqual(index.get_current_count(), 4)
        self.assertEqual(index.get_deleted_ratio(), 0.25)
        self.assertTrue(self.edge_markers(index).keys() < before.keys())
        self.assertEqual(index.get_neighbors(3).size, 0)
        self.assertGreater(stats["incoming_edges_repaired"], 0)

    def test_unmark_deleted_rejects_cleaned_marker_contributions(self):
        index, _ = self.make_index()
        index.delete_items([3], num_threads=1)

        with self.assertRaisesRegex(
            RuntimeError, "Cannot restore a deleted element after Marker cleanup"
        ):
            index.unmark_deleted(3)

    def test_tombstone_only_edge_deletion_remains_reversible(self):
        index, _ = self.make_index()
        before = self.edge_markers(index)
        index.mark_deleted(3)
        index.unmark_deleted(3)
        self.assertEqual(index.batch_mark_deleted([1, 2], num_threads=2), 2)
        index.unmark_deleted(1)
        index.unmark_deleted(2)

        self.assertEqual(index.get_deleted_ratio(), 0)
        for edge, marker in self.edge_markers(index).items():
            np.testing.assert_array_equal(marker, before[edge])

    def test_cleanup_retirement_guard_survives_save_load(self):
        index, _ = self.make_index()
        index.delete_items([3], num_threads=1)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "index.bin")
            index.save_index(path)
            restored = hashannlib.Index(space="l2", dim=2)
            restored.load_index(path, max_elements=4, top_elements=1, dynamic=True)
            with self.assertRaisesRegex(RuntimeError, "after Marker cleanup"):
                restored.unmark_deleted(3)
            restored.mark_deleted(2)
            with self.assertRaisesRegex(RuntimeError, "after Marker cleanup"):
                restored.unmark_deleted(2)

    def test_cleanup_also_prevents_revival_of_prior_tombstones(self):
        index, _ = self.make_index()
        index.mark_deleted(3)
        index.delete_items([2], num_threads=1)
        with self.assertRaisesRegex(RuntimeError, "after Marker cleanup"):
            index.unmark_deleted(3)

    def test_invalid_delete_inputs_do_not_mutate_the_index(self):
        invalid_calls = [
            lambda idx: idx.mark_deleted(99),
            lambda idx: idx.batch_mark_deleted([0, 99], num_threads=2),
            lambda idx: idx.delete_items([0, 99], num_threads=2),
            lambda idx: idx.batch_mark_deleted([0, 0], num_threads=2),
            lambda idx: idx.delete_items([0, 0], num_threads=2),
            lambda idx: idx.delete_items(np.asarray([[0, 1]])),
        ]
        for call in invalid_calls:
            with self.subTest(call=call):
                index, _ = self.make_index()
                before = self.edge_markers(index)
                with self.assertRaises(RuntimeError):
                    call(index)
                self.assertEqual(index.get_deleted_ratio(), 0)
                for edge, marker in self.edge_markers(index).items():
                    np.testing.assert_array_equal(marker, before[edge])

    def test_invalid_cleanup_inputs_do_not_mutate_markers_or_guard(self):
        for labels in ([3, 0], [3, 99], [3, 3]):
            with self.subTest(labels=labels):
                index, _ = self.make_index()
                index.mark_deleted(3)
                before = self.edge_markers(index)
                with self.assertRaises(RuntimeError):
                    index.cleanup_deleted_marker_bits(labels)
                for edge, marker in self.edge_markers(index).items():
                    np.testing.assert_array_equal(marker, before[edge])
                index.unmark_deleted(3)

    def test_delete_rejects_already_deleted_labels_before_marking_others(self):
        index, _ = self.make_index()
        index.mark_deleted(3)
        with self.assertRaisesRegex(RuntimeError, "already deleted"):
            index.delete_items([0, 3], num_threads=2)
        with self.assertRaisesRegex(RuntimeError, "already deleted"):
            index.batch_mark_deleted([0, 3], num_threads=2)
        self.assertEqual(index.get_deleted_ratio(), 0.25)
        index.unmark_deleted(3)

    def test_empty_and_all_deleted_batches(self):
        index, _ = self.make_index()
        empty = index.delete_items([])
        self.assertEqual(empty["marked"], 0)
        self.assertEqual(empty["scrub_wall_s"], 0.0)
        self.assertEqual(index.batch_mark_deleted([]), 0)
        self.assertEqual(
            index.cleanup_deleted_marker_bits([])["deleted_points"], 0
        )
        started = time.perf_counter()
        stats = index.delete_items(np.arange(4), num_threads=2)
        elapsed = time.perf_counter() - started
        self.assertEqual(stats["marked"], 4)
        self.assertEqual(stats["scrubbed_edges"], 0)
        self.assertGreater(stats["scrub_wall_s"], 0.0)
        self.assertLessEqual(stats["scrub_wall_s"], elapsed)
        self.assertEqual(self.edge_markers(index), {})
        self.assertEqual(index.get_deleted_ratio(), 1)

    def test_batch_mark_and_cleanup_accept_strided_label_arrays(self):
        index, _ = self.make_index()
        labels = np.arange(4, dtype=np.uint64)[::2]
        self.assertEqual(index.batch_mark_deleted(labels, num_threads=2), 2)
        stats = index.cleanup_deleted_marker_bits(labels, num_threads=2)
        self.assertEqual(stats["deleted_points"], 2)
        for label in (0, 2):
            with self.assertRaisesRegex(RuntimeError, "after Marker cleanup"):
                index.unmark_deleted(label)
        with self.assertRaisesRegex(RuntimeError, "already deleted"):
            index.mark_deleted(0)
        index.mark_deleted(1)
        self.assertEqual(index.get_deleted_ratio(), 0.75)

    def test_unmark_deleted_remains_available_without_edge_markers(self):
        data = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        attrs = [[[0]], [[1]]]
        index = hashannlib.Index(space="l2", dim=2)
        index.init_index(
            max_elements=len(data),
            top_elements=1,
            M=2,
            ef_construction=20,
            ft_bits=8,
            attr_type=[1],
            max_cate_size=2,
            edge_level_ft=False,
        )
        index.initAttrMapping(attrs)
        index.add_items(data, attrs, ids=np.arange(len(data)))

        index.mark_deleted(1)
        index.unmark_deleted(1)
        index.mark_deleted(1)

    def test_delete_items_accepts_strided_label_arrays(self):
        data = np.arange(12, dtype=np.float32).reshape(6, 2)
        attrs = [[[i]] for i in range(6)]
        index = hashannlib.Index(space="l2", dim=2)
        index.init_index(
            max_elements=len(data),
            top_elements=1,
            M=2,
            ef_construction=20,
            ft_bits=8,
            attr_type=[1],
            max_cate_size=6,
            edge_level_ft=False,
        )
        index.initAttrMapping(attrs)
        index.add_items(data, attrs, ids=np.arange(len(data)))

        stats = index.delete_items(
            np.arange(len(data), dtype=np.uint64)[::2],
            num_threads=1,
        )

        self.assertEqual(stats["marked"], 3)
        for label in (0, 2, 4):
            with self.assertRaisesRegex(RuntimeError, "in-place deletion"):
                index.unmark_deleted(label)
            self.assertEqual(index.get_neighbors(label).size, 0)
        with self.assertRaisesRegex(RuntimeError, "not deleted"):
            index.unmark_deleted(1)

    def test_add_items_cannot_restore_deleted_edge_marker_point(self):
        index, data = self.make_index()
        index.delete_items([3], num_threads=1)

        with self.assertRaisesRegex(
            RuntimeError, "Cannot restore a deleted element after Marker cleanup"
        ):
            index.add_items(
                data[3:4],
                [[[3]]],
                ids=np.asarray([3], dtype=np.uint64),
            )

    def test_add_items_cannot_replace_deleted_edge_marker_slot(self):
        data = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        attrs = [[[0]], [[1]]]
        index = hashannlib.Index(space="l2", dim=2)
        index.init_index(
            max_elements=len(data),
            top_elements=1,
            M=2,
            ef_construction=20,
            ft_bits=8,
            attr_type=[1],
            max_cate_size=2,
            allow_replace_deleted=True,
            edge_level_ft=True,
        )
        index.initAttrMapping(attrs)
        index.add_items(data, attrs, ids=np.arange(len(data)))
        index.delete_items([1], num_threads=1)

        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot replace a deleted element after Marker cleanup",
        ):
            index.add_items(
                np.asarray([[2.0, 0.0]], dtype=np.float32),
                [[[0]]],
                ids=np.asarray([2], dtype=np.uint64),
                replace_deleted=True,
            )

    def test_batch_update_refreshes_asymmetric_incoming_edges(self):
        rng = np.random.default_rng(13)
        count = 2000
        data = rng.random((count, 8), dtype=np.float32)
        attrs = [[[i % 8]] for i in range(count)]
        levels = np.zeros(count, dtype=np.int32)
        levels[:160] = 1

        index = hashannlib.Index(space="l2", dim=8)
        index.init_index(
            max_elements=count,
            top_elements=160,
            M=12,
            ef_construction=80,
            ft_bits=128,
            attr_type=[1],
            max_cate_size=8,
            edge_level_ft=True,
        )
        index.initAttrMapping(attrs)
        index.add_items(
            data,
            attrs,
            ids=np.arange(count),
            levels=levels,
            num_threads=4,
        )
        index.set_ef(800)

        update_ids = np.arange(8, 808, 8, dtype=np.int64)
        before, _ = index.hybrid_knn_query(
            data[update_ids],
            [[[0]]] * len(update_ids),
            k=1,
            num_threads=1,
        )
        self.assertTrue(np.array_equal(before[:, 0], update_ids))

        updated = index.batch_update_attr(
            update_ids,
            [[[7]]] * len(update_ids),
            num_threads=4,
        )
        self.assertEqual(updated, len(update_ids))
        after, _ = index.hybrid_knn_query(
            data[update_ids],
            [[[7]]] * len(update_ids),
            k=1,
            num_threads=1,
        )
        self.assertTrue(np.array_equal(after[:, 0], update_ids))

    def test_cleanup_uses_construction_ef_independently_of_query_ef_after_load(self):
        rng = np.random.default_rng(101)
        count = 128
        data = rng.random((count, 4), dtype=np.float32)
        attrs = [[[i % 4]] for i in range(count)]
        levels = np.zeros(count, dtype=np.int32)
        levels[:16] = 1
        index = hashannlib.Index(space="l2", dim=4)
        index.init_index(
            max_elements=count, top_elements=16, M=8, ef_construction=24,
            ft_bits=128, attr_type=[1], max_cate_size=4, edge_level_ft=True,
        )
        index.initAttrMapping(attrs)
        index.add_items(data, attrs, ids=np.arange(count), levels=levels, num_threads=1)
        index.set_ef(1)
        stats = index.delete_items([3], num_threads=1)
        self.assertEqual(stats["search_candidates"], 24)
        self.assertEqual(index.ef, 1)

        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "index.bin")
            index.save_index(path)
            restored = hashannlib.Index(space="l2", dim=4)
            restored.load_index(path, max_elements=count, top_elements=16, dynamic=True)
            self.assertEqual(restored.ef_construction, 24)
            restored.set_ef(96)
            stats = restored.cleanup_deleted_marker_bits([3], num_threads=1)
            self.assertEqual(stats["candidate_sources"], 24)
            stats = restored.delete_items([4], num_threads=1)
            self.assertEqual(stats["search_candidates"], 24)
            self.assertEqual(restored.ef, 96)

    def test_cleanup_preserves_colliding_and_multilabel_live_support(self):
        for multi_label, ft_bits, categories in ((False, 8, 16), (True, 128, 8)):
            for threads in (1, 2):
                with self.subTest(multi_label=multi_label, threads=threads):
                    rng = np.random.default_rng(91)
                    count = 256
                    data = rng.random((count, 4), dtype=np.float32)
                    attrs = [
                        [[i % 4, i % 4 + 4]] if multi_label else [[i % categories]]
                        for i in range(count)
                    ]
                    index = hashannlib.Index(space="l2", dim=4)
                    index.init_index(
                        max_elements=count, top_elements=16, M=8,
                        ef_construction=80, ft_bits=ft_bits, attr_type=[1],
                        max_cate_size=categories, edge_level_ft=True,
                    )
                    index.initAttrMapping(attrs)
                    point_masks = np.asarray([
                        [ord(char) for char in mask]
                        for mask in index.predicateToFT(attrs)
                    ], dtype=np.uint8)
                    deleted = np.arange(8, dtype=np.uint64)
                    used = set(deleted)
                    for label in deleted:
                        twins = [
                            i for i in range(64, count)
                            if i not in used
                            and (multi_label or attrs[i] != attrs[label])
                            and np.array_equal(point_masks[i], point_masks[label])
                        ]
                        self.assertTrue(twins)
                        twin = twins[0]
                        data[twin] = data[label]
                        used.add(twin)
                    levels = np.zeros(count, dtype=np.int32)
                    levels[:16] = 1
                    index.add_items(
                        data, attrs, ids=np.arange(count), levels=levels,
                        num_threads=1,
                    )
                    before = self.edge_markers(index)
                    index.set_ef(37)
                    index.batch_mark_deleted(deleted, num_threads=threads)
                    stats = index.cleanup_deleted_marker_bits(
                        deleted, num_threads=threads,
                    )
                    after = self.edge_markers(index)
                    self.assertEqual(index.ef, 37)
                    self.assertGreater(stats["candidate_sources"], 0)
                    self.assertLessEqual(
                        stats["candidate_sources"], len(deleted) * index.ef_construction
                    )
                    self.assertGreater(stats["support_checks"], 0)
                    self.assertEqual(stats["cleared_bits"], 0)
                    self.assertEqual(before.keys(), after.keys())
                    for (source, target), marker in after.items():
                        np.testing.assert_array_equal(marker, before[source, target])
                        np.testing.assert_array_equal(
                            marker & point_masks[target], point_masks[target]
                        )

    def test_cleanup_preserves_mixed_attribute_targets_in_all_spaces(self):
        for space in ("l2", "ip", "cosine"):
            with self.subTest(space=space):
                rng = np.random.default_rng(102)
                count = 128
                data = rng.normal(size=(count, 8)).astype(np.float32)
                attrs = [[[7], [i % 8]] for i in range(count)]
                levels = np.zeros(count, dtype=np.int32)
                levels[:16] = 1
                index = hashannlib.Index(space=space, dim=8)
                index.init_index(
                    max_elements=count, top_elements=16, M=8,
                    ef_construction=80, ft_bits=128, attr_type=[0, 1],
                    max_cate_size=8, edge_level_ft=True,
                )
                index.initAttrMapping(attrs)
                index.add_items(
                    data, attrs, ids=np.arange(count), levels=levels,
                    num_threads=1,
                )
                categorical_masks = np.asarray([
                    [ord(char) for char in mask[16:]]
                    for mask in index.predicateToFT([[[], attr[1]] for attr in attrs])
                ], dtype=np.uint8)
                before = self.edge_markers(index)
                index.batch_mark_deleted(np.arange(16), num_threads=2)
                stats = index.cleanup_deleted_marker_bits(np.arange(16), num_threads=2)
                self.assertGreater(stats["candidate_sources"], 0)
                after = self.edge_markers(index)
                self.assertEqual(before.keys(), after.keys())
                for (source, target), marker in after.items():
                    np.testing.assert_array_equal(
                        marker[:16], before[source, target][:16]
                    )
                    np.testing.assert_array_equal(
                        marker[16:] & categorical_masks[target],
                        categorical_masks[target],
                    )

    def test_cleanup_clears_unsupported_rare_bits(self):
        rng = np.random.default_rng(23)
        count = 3000
        data = rng.random((count, 8), dtype=np.float32)
        attrs = [[[127]] for _ in range(count)]
        for label in range(128):
            attrs[label] = [[label]]
        levels = np.zeros(count, dtype=np.int32)
        levels[:240] = 1

        index = hashannlib.Index(space="l2", dim=8)
        index.init_index(
            max_elements=count,
            top_elements=240,
            M=16,
            ef_construction=120,
            ft_bits=128,
            attr_type=[1],
            max_cate_size=128,
            edge_level_ft=True,
        )
        index.initAttrMapping(attrs)
        index.add_items(
            data,
            attrs,
            ids=np.arange(count),
            levels=levels,
            num_threads=1,
        )
        index.batch_mark_deleted(np.arange(64, dtype=np.int64), num_threads=1)
        index.set_ef(800)
        probes = data[500:540] + np.float32(0.001)
        query_attrs = [[[127]]] * len(probes)
        before, _ = index.hybrid_knn_query(
            probes, query_attrs, k=10, num_threads=1,
        )
        bits_before = int(index.get_edge_ft_bit_stats()[0][0])

        stats = index.cleanup_deleted_marker_bits(
            np.arange(64, dtype=np.int64),
            num_threads=4,
        )
        bits_after = int(index.get_edge_ft_bit_stats()[0][0])

        self.assertEqual(stats["deleted_points"], 64)
        self.assertGreater(stats["matched_edges"], 0)
        self.assertGreater(stats["cleared_bits"], 0)
        self.assertLess(bits_after, bits_before)
        self.assertEqual(bits_before - bits_after, stats["cleared_bits"])
        repeated = index.cleanup_deleted_marker_bits(
            np.arange(64, dtype=np.int64), num_threads=4,
        )
        self.assertEqual(repeated["cleared_bits"], 0)
        after, _ = index.hybrid_knn_query(
            probes, query_attrs, k=10, num_threads=1,
        )
        self.assertFalse(np.isin(after, np.arange(64)).any())
        np.testing.assert_array_equal(after, before)


if __name__ == "__main__":
    unittest.main()
