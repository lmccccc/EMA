import unittest

import numpy as np

import hashannlib


class DynamicApiTestCase(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
