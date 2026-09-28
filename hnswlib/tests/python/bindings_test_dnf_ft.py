import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

import hashannlib


class DnfMarkerTestCase(unittest.TestCase):
    def make_index(self, bits, edge_level, uniform_category=False):
        count = 96
        attributes = [
            [[101 * i + 1], [0] if uniform_category else [i % 7, (i + 1) % 7]]
            for i in range(count)
        ]
        index = hashannlib.Index(space="l2", dim=4)
        index.init_index(
            max_elements=count, top_elements=1, M=count, ef_construction=count + 8,
            ft_bits=bits, attr_type=[0, 1], max_cate_size=7,
            edge_level_ft=edge_level,
        )
        index.initAttrMapping(attributes)
        levels = np.zeros(count, dtype=np.int32)
        levels[0] = 1
        index.add_items(
            np.zeros((count, 4), dtype=np.float32), attributes,
            ids=np.arange(count, dtype=np.uint64), levels=levels, num_threads=1,
        )
        index.set_ef(count + 8)
        index.set_ef_top(1)
        index.set_ft_flag(True)
        index.set_ft_routing_flag(False)
        return index, attributes

    def read_markers(self, index, width, edge_level):
        state = index.__getstate__()[0]
        rows = state["data_level0"].view(np.uint8).reshape(
            state["cur_element_count"], state["size_data_per_element"]
        )
        links_offset = state["offset_level0"]
        marker_offset = links_offset + 4 + 4 * state["max_M0"]
        adjacency, markers = {}, {}
        for source, row in enumerate(rows):
            count = int(row[links_offset:links_offset + 2].copy().view(np.uint16)[0])
            adjacency[source] = row[
                links_offset + 4:links_offset + 4 + 4 * count
            ].copy().view(np.uint32).tolist()
            slots = state["max_M0"] if edge_level else 1
            masks = row[marker_offset:state["offset_data"]].reshape(slots, 2 * width)
            for slot in range(count if edge_level else 1):
                key = (source, adjacency[source][slot]) if edge_level else source
                markers[key] = tuple(
                    int.from_bytes(masks[slot, a * width:(a + 1) * width].tobytes(), "little")
                    for a in range(2)
                )
        return adjacency, markers

    def term_masks(self, index, term, width):
        ranges, labels = term
        variants = [[ranges[i:i + 2], labels] for i in range(0, len(ranges), 2)] if ranges else [term]
        masks = [0, 0]
        for variant in variants:
            encoded = bytes(ord(c) for c in index.predicateToFT([variant])[0])
            self.assertEqual(len(encoded), 2 * width)
            for attr in range(2):
                masks[attr] |= int.from_bytes(encoded[attr * width:(attr + 1) * width], "little")
        return masks

    @staticmethod
    def marker_passes(marker, terms, masks, modes):
        for number, term in enumerate(terms):
            passed = True
            for attr, values in enumerate(term):
                if not values:
                    continue
                any_mode = attr == 0 or (modes is not None and modes[number][attr] == 0)
                overlap = marker[attr] & masks[number][attr]
                matches = bool(overlap) if any_mode else overlap == masks[number][attr]
                if not matches:
                    passed = False
                    break
            if passed:
                return True
        return False

    @staticmethod
    def matches(attributes, terms, modes):
        for number, (ranges, labels) in enumerate(terms):
            numeric = not ranges or any(
                low <= attributes[0][0] <= high for low, high in zip(ranges[::2], ranges[1::2])
            )
            if not labels:
                categorical = True
            elif modes is not None and modes[number][1] == 0:
                categorical = bool(set(labels) & set(attributes[1]))
            else:
                categorical = set(labels).issubset(attributes[1])
            if numeric and categorical:
                return True
        return False

    def test_inclusive_numeric_marker_ranges_keep_all_matching_points(self):
        for bits in (8, 72, 128):
            for edge_level in (False, True):
                index, attributes = self.make_index(bits, edge_level, uniform_category=True)
                values = [record[0][0] for record in attributes]
                ranges_to_check = (
                    [values[63], values[63]],
                    [values[0], values[63]],
                    [values[63], values[64]],
                    [values[-1], values[-1]],
                    [values[63] - 1, values[63] + 1],
                    [-100, values[-1] + 100],
                    [values[0], values[0], values[63], values[63], values[-1], values[-1]],
                )
                for ranges in ranges_to_check:
                    with self.subTest(bits=bits, edge_level=edge_level, ranges=ranges):
                        terms = [[ranges, [0]]]
                        eligible = {
                            i for i, record in enumerate(attributes)
                            if self.matches(record, terms, None)
                        }
                        labels, distances = index.hybrid_knn_query_dnf(
                            np.zeros((1, 4), dtype=np.float32), [terms],
                            k=len(eligible), num_threads=1,
                        )
                        self.assertEqual(set(map(int, labels[0])), eligible)
                        self.assertTrue(np.all(distances == 0))
                        if len(ranges) == 2:
                            labels, distances = index.hybrid_knn_query(
                                np.zeros((1, 4), dtype=np.float32), [terms[0]],
                                k=len(eligible), num_threads=1,
                            )
                            self.assertEqual(set(map(int, labels[0])), eligible)
                            self.assertTrue(np.all(distances == 0))

    def test_numeric_bucket_stays_inside_its_marker_column(self):
        count = 32
        data = np.arange(1, count * 4 + 1, dtype=np.float32).reshape(count, 4)
        ids = np.arange(count, dtype=np.uint64)
        levels = np.zeros(count, dtype=np.int32)
        levels[0] = 1
        for edge_level in (False, True):
            for numerical_column in (0, 1):
                with self.subTest(edge_level=edge_level, numerical_column=numerical_column):
                    attr_types = [1, 1]
                    attr_types[numerical_column] = 0
                    attributes = [[[i], [0]] if numerical_column == 0 else [[0], [i]]
                                  for i in range(count)]
                    index = hashannlib.Index(space="l2", dim=4)
                    index.init_index(
                        max_elements=count, top_elements=1, M=count,
                        ef_construction=count + 8, ft_bits=8, attr_type=attr_types,
                        max_cate_size=1, edge_level_ft=edge_level,
                    )
                    index.initAttrMapping(attributes)
                    index.add_items(data, attributes, ids=ids, levels=levels, num_threads=1)
                    if not edge_level:
                        index.batch_update_attr(ids, attributes, num_threads=1)
                    np.testing.assert_array_equal(index.get_items(ids), data)
                    adjacency, masks = self.read_markers(index, 1, edge_level)
                    for source in range(count):
                        targets = adjacency[source] if edge_level else [source]
                        for target in targets:
                            key = (source, target) if edge_level else source
                            self.assertEqual(masks[key][numerical_column], 1 << (target // 4))
                            self.assertEqual(masks[key][1 - numerical_column], 1)

                    index.set_ef(count + 8)
                    index.set_ft_flag(True)
                    index.set_ft_routing_flag(False)
                    for value in (np.iinfo(np.int32).max, np.iinfo(np.int32).min):
                        replacement = [[int(value)], [0]] if numerical_column == 0 else [[0], [int(value)]]
                        index.batch_update_attr([count - 1], [replacement], num_threads=1)
                        np.testing.assert_array_equal(index.get_items(ids), data)
                        predicate = [[int(value), int(value)], [0]] if numerical_column == 0 else [[0], [int(value), int(value)]]
                        found, _ = index.hybrid_knn_query_dnf(
                            data[-1:], [[predicate]], k=1, num_threads=1,
                        )
                        self.assertEqual(int(found[0, 0]), count - 1)

    def test_constant_maximum_integer_numeric_mapping(self):
        value = int(np.iinfo(np.int32).max)
        attributes = [[[value], [0]] for _ in range(4)]
        data = np.zeros((4, 4), dtype=np.float32)
        index = hashannlib.Index(space="l2", dim=4)
        index.init_index(
            max_elements=4, top_elements=1, M=4, ef_construction=16,
            ft_bits=8, attr_type=[0, 1], max_cate_size=1, edge_level_ft=True,
        )
        index.initAttrMapping(attributes)
        index.add_items(data, attributes, ids=np.arange(4), levels=[1, 0, 0, 0], num_threads=1)
        index.set_ef(16)
        index.set_ft_flag(True)
        index.set_ft_routing_flag(False)
        labels, _ = index.hybrid_knn_query_dnf(
            data[:1], [[[[value, value], [0]]]], k=4, num_threads=1,
        )
        self.assertEqual(set(map(int, labels[0])), set(range(4)))

    def test_legacy_numeric_marker_files_and_pickle_states_require_rebuild(self):
        for edge_level in (False, True):
            index, _ = self.make_index(72, edge_level, uniform_category=True)
            state = index.__getstate__()[0]
            self.assertEqual(state["numeric_marker_version"], 2)
            self.assertEqual(state["marker_owner_version"], 2)
            self.assertEqual(state["distance_order_version"], 1)
            self.assertEqual(state["ser_version"], 4)
            for version in (1, 2, 3):
                legacy_state = dict(state, ser_version=version)
                with self.assertRaisesRegex(RuntimeError, "Legacy Marker pickle.*rebuild"):
                    hashannlib.Index(legacy_state)
            for field, wrong in (
                ("numeric_marker_version", 1), ("marker_owner_version", 1),
                ("distance_order_version", 0),
            ):
                legacy_state = dict(state, **{field: wrong})
                with self.assertRaisesRegex(RuntimeError, "Incompatible Marker pickle.*rebuild"):
                    hashannlib.Index(legacy_state)
                legacy_state = dict(state)
                del legacy_state[field]
                with self.assertRaisesRegex(RuntimeError, "Incompatible Marker pickle.*rebuild"):
                    hashannlib.Index(legacy_state)
            with tempfile.TemporaryDirectory() as directory:
                current = Path(directory) / "current.index"
                index.save_index(str(current))
                payload = bytearray(current.read_bytes())
                version_offset = (
                    len(payload) - state["data_level0"].nbytes - state["link_lists"].nbytes
                    - 4 * state["cur_element_count"] - 4
                )
                self.assertEqual(struct.unpack_from("<i", payload, version_offset)[0],
                                 12 if edge_level else 11)
                restored = hashannlib.Index(space="l2", dim=4)
                restored.load_index(str(current), max_elements=96, top_elements=1, dynamic=True)
                np.testing.assert_array_equal(
                    restored.__getstate__()[0]["data_level0"], state["data_level0"],
                )
                # Relabel the fixture to exercise the legacy-version guard before graph loading.
                struct.pack_into("<i", payload, version_offset, 4 if edge_level else 3)
                legacy = Path(directory) / "legacy.index"
                legacy.write_bytes(payload)
                rejected = hashannlib.Index(space="l2", dim=4)
                with self.assertRaisesRegex(RuntimeError, "Legacy numerical Marker.*rebuild"):
                    rejected.load_index(str(legacy), max_elements=96, top_elements=1, dynamic=True)
                struct.pack_into("<i", payload, version_offset, 6 if edge_level else 5)
                legacy.write_bytes(payload)
                with self.assertRaisesRegex(RuntimeError, "Legacy Marker ownership.*rebuild"):
                    rejected.load_index(str(legacy), max_elements=96, top_elements=1, dynamic=True)
                struct.pack_into("<i", payload, version_offset, 8 if edge_level else 7)
                legacy.write_bytes(payload)
                with self.assertRaisesRegex(RuntimeError, "Legacy candidate ordering.*rebuild"):
                    rejected.load_index(str(legacy), max_elements=96, top_elements=1, dynamic=True)

    def test_node_and_edge_dnf_masks_match_scalar_oracle(self):
        predicates = [
            ([[[0, 4000], [3]]], None),
            ([[[], [6]]], None),
            ([[[], [2, 3]]], None),
            ([[[], [5, 0]]], [[-1, 0]]),
            ([[[0, 2000], [4, 5]], [[8000, 10000], [6]]], None),
            ([[[], []]], None),
            ([[[2000, 3000, 8000, 9000], []]], None),
            ([[[], [6, 0]]], None),
            ([[[], [0, 3]]], None),
        ]
        for bits in (8, 16, 32, 64, 72, 128, 136, 256):
            for edge_level in (False, True):
                index, attributes = self.make_index(bits, edge_level)
                adjacency, markers = self.read_markers(index, bits // 8, edge_level)
                for source, neighbors in adjacency.items():
                    self.assertEqual(set(neighbors), set(adjacency) - {source})
                for terms, modes in predicates:
                    with self.subTest(bits=bits, edge_level=edge_level, terms=terms, modes=modes):
                        masks = [self.term_masks(index, term, bits // 8) for term in terms]
                        passing = {
                            source: [
                                target for target in neighbors
                                if self.marker_passes(
                                    markers[(source, target) if edge_level else target],
                                    terms, masks, modes,
                                )
                            ]
                            for source, neighbors in adjacency.items()
                        }
                        # Equal vectors and ef>N remove distance stopping from the scalar routing oracle.
                        reachable, pending = {0}, [0]
                        while pending:
                            for target in passing[pending.pop()]:
                                if target not in reachable:
                                    reachable.add(target)
                                    pending.append(target)
                        expected_passes = sum(len(passing[source]) for source in reachable)
                        expected_visits = sum(len(adjacency[source]) for source in reachable)
                        eligible = {
                            i for i, record in enumerate(attributes) if self.matches(record, terms, modes)
                        }
                        kwargs = {"check_modes": [modes]} if modes is not None else {}
                        index.reset_ft_stats()
                        if eligible:
                            labels, distances = index.hybrid_knn_query_dnf(
                                np.zeros((1, 4), dtype=np.float32), [terms], k=1, num_threads=1, **kwargs
                            )
                            self.assertIn(int(labels[0, 0]), eligible)
                            self.assertEqual(float(distances[0, 0]), 0.0)
                        else:
                            with self.assertRaises(RuntimeError):
                                index.hybrid_knn_query_dnf(
                                    np.zeros((1, 4), dtype=np.float32), [terms], k=1, num_threads=1, **kwargs
                                )
                        stats = index.get_ft_stats()
                        self.assertEqual(stats["ft_passed_total"], expected_passes)
                        self.assertEqual(stats["total_neighbors"], expected_visits)


if __name__ == "__main__":
    unittest.main()
