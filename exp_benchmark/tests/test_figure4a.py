import unittest

import numpy as np

from exp_benchmark.figure4a import (
    and_predicates, composite_predicates, ground_truth_prefix,
    matches, matching_counts, validate_ids,
)


class Figure4aInputTests(unittest.TestCase):
    def test_legacy_and_with_inclusive_bounds(self):
        bounds, masks = and_predicates([[[1, 3], [0, 2]]], 1)
        numeric = np.array([1, 3, 2, 4])
        categorical = np.array([5, 5, 1, 5], dtype=np.uint32)
        np.testing.assert_array_equal(
            matches(np.array([[0, 1, 2, 3]]), numeric, categorical, bounds, masks),
            [[True, True, False, False]])

    def test_rejects_dynamic_or_predicate(self):
        with self.assertRaises(ValueError):
            and_predicates([[[[0, 10], [9]], [[], [12]]]], 1)

    def test_rejects_wrong_count_and_reversed_range(self):
        for value in ([], [[[3, 1], [0]]]):
            with self.assertRaises(ValueError):
                and_predicates(value, 1)

    def test_rejects_missing_duplicate_and_out_of_range_ids(self):
        for labels in ([[0]], [[0, 0]], [[0, 4]], [[-1, 1]]):
            with self.assertRaises(ValueError):
                validate_ids(np.array(labels), 4, 1, 2)

    def test_valid_external_ids(self):
        np.testing.assert_array_equal(validate_ids(np.array([[3, 1]]), 4, 1, 2), [[3, 1]])

    def test_composite_or_and_inclusive_bounds(self):
        bounds, masks, tail_masks = composite_predicates([[[[1, 10], [1]], [[], [2]]]], 1)
        numeric = np.array([0, 1, 10, 11, 20])
        categorical = np.array([4, 2, 6, 2, 0], dtype=np.uint32)
        np.testing.assert_array_equal(
            matches(np.array([[0, 1, 2, 3, 4]]), numeric, categorical,
                    bounds, masks, tail_masks),
            [[True, True, True, False, False]])
        np.testing.assert_array_equal(
            matching_counts(numeric, categorical, bounds, masks, tail_masks), [3])

    def test_composite_multilabel_terms_require_all_labels(self):
        bounds, masks, tail_masks = composite_predicates(
            [[[[1, 2], [0, 1]], [[], [2, 3]]]], 1)
        numeric = np.array([1, 2, 0, 3, 1])
        categorical = np.array([3, 1, 12, 4, 15], dtype=np.uint32)
        np.testing.assert_array_equal(
            matches(np.array([[0, 1, 2, 3, 4]]), numeric, categorical,
                    bounds, masks, tail_masks),
            [[True, False, True, False, True]])
        np.testing.assert_array_equal(
            matching_counts(numeric, categorical, bounds, masks, tail_masks), [3])

    def test_composite_rejects_different_shape_and_conditions(self):
        for value in (
            [], [[[1, 2], [1]]], [[[[1, 2], [1]]]],
            [[[[1, 2], [1]], [[1, 3], [2]]]],
            [[[[2, 1], [1]], [[], [2]]]],
            [[[[1, 2], [1]], [[], []]]],
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                composite_predicates(value, 1)

    def test_and_counts_match_predicate_evaluation(self):
        bounds, masks = and_predicates([[[1, 3], [0]], [[3, 5], [0]]], 2)
        numeric = np.array([1, 2, 3, 4, 5])
        categorical = np.array([1, 0, 1, 1, 0], dtype=np.uint32)
        np.testing.assert_array_equal(
            matching_counts(numeric, categorical, bounds, masks), [2, 2])

    def test_counts_reject_varying_categorical_conditions(self):
        numeric = np.array([1, 2])
        categorical = np.array([1, 2], dtype=np.uint32)
        bounds = np.array([[1, 2], [1, 2]])
        with self.assertRaises(ValueError):
            matching_counts(numeric, categorical, bounds, np.array([1, 2]))
        with self.assertRaises(ValueError):
            matching_counts(numeric, categorical, bounds, np.array([1, 1]),
                            np.array([2, 3]))

    def test_explicit_ground_truth_prefix(self):
        np.testing.assert_array_equal(
            ground_truth_prefix([[0, 1], [2, 3], [3, 2]], 4, 2, 2, 3),
            [[0, 1], [2, 3]])

    def test_ground_truth_prefix_rejects_undeclared_or_short_input(self):
        for stored_rows in (1, 2, 4):
            with self.subTest(stored_rows=stored_rows), self.assertRaises(ValueError):
                ground_truth_prefix([[0, 1], [2, 3], [3, 2]], 4, 2, 2, stored_rows)


if __name__ == "__main__":
    unittest.main()
