import contextlib
import importlib
import io
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

from exp_benchmark.index_metadata import read_layout, record_sample


@unittest.skipUnless(os.environ.get("HASHANN_LEGACY_QUERY_NATIVE"),
                     "Set HASHANN_LEGACY_QUERY_NATIVE to the isolated loader-only native directory")
class LegacyQueryNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = Path(os.environ["HASHANN_LEGACY_QUERY_NATIVE"]).resolve()
        sys.path.insert(0, str(directory))
        cls.native = importlib.import_module("hashannlib")
        if Path(cls.native.__file__).resolve().parent != directory:
            raise RuntimeError("The native integration test imported the wrong extension")
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
        cls.query = importlib.import_module("hashann_query")

    def test_inclusive_legacy_numeric_query_masks(self):
        index = self.native.Index(space="l2", dim=4)
        index.init_index(
            max_elements=96, top_elements=1, M=4, ef_construction=16,
            ft_bits=16, attr_type=[0], max_cate_size=1, edge_level_ft=True)
        index.initAttrMapping([[[row]] for row in range(96)])
        for bounds, expected in (([7, 8], [4, 0]), ([7, 13], [12, 0]), ([12, 12], [4, 0])):
            with self.subTest(bounds=bounds):
                mask = index.predicateToFT([[bounds]])[0]
                self.assertEqual([ord(value) if isinstance(value, str) else value for value in mask], expected)

    def test_unversioned_edge_layouts_preserve_queries_and_attributes(self):
        vectors = np.random.default_rng(4).normal(size=(96, 4)).astype(np.float32)
        levels = np.zeros(len(vectors), dtype=np.int32)
        levels[0] = 1
        for kinds in ([0], [1], [0, 1]):
            with self.subTest(kinds=kinds), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                records = [[[100 * row + 1] if kind == 0 else [row % 7, (row + 1) % 7]
                            for kind in kinds] for row in range(len(vectors))]
                index = self.native.Index(space="l2", dim=4)
                index.init_index(
                    max_elements=len(vectors), top_elements=1, M=96, ef_construction=104,
                    ft_bits=256, attr_type=kinds, max_cate_size=7, edge_level_ft=True)
                index.initAttrMapping(records)
                index.add_items(vectors, records, ids=np.arange(len(vectors), dtype=np.uint64),
                                levels=levels, num_threads=1)
                saved = directory / "version4.index"
                index.save_index(str(saved))
                header, source = read_layout(saved), saved.read_bytes()
                detailed = read_layout(saved, include_mappings=True)
                mappings = detailed.pop("counting_hash_table_mapping")
                self.assertEqual(detailed, header)
                self.assertEqual(len(mappings), len(kinds))
                self.assertTrue(all(len(mapping) == 256 for mapping in mappings))
                self.assertEqual(header["format"], 4)
                start, size = header["records_offset"], header["record_bytes"]
                self.assertGreater(struct.unpack_from("<I", source, start)[0], 10)
                paths = [saved]
                packed = directory / "unversioned-packed.index"
                packed.write_bytes(source[:start - 4] + source[start:])
                paths.append(packed)

                # Convert only this tiny generated fixture, never a benchmark cache.
                prefix = bytearray(source[:start - 4])
                data_offset = header["ft_offset"]
                label_offset = data_offset + 4 * vectors.shape[1]
                ft_offset = label_offset + 8
                struct.pack_into("<Q", prefix, 6 * 8, label_offset)
                struct.pack_into("<Q", prefix, 7 * 8, data_offset)
                struct.pack_into("<Q", prefix, start - 28, ft_offset)
                struct.pack_into("<i", prefix, start - 12, ft_offset)
                rows = []
                for row in range(len(vectors)):
                    record = source[start + row * size:start + (row + 1) * size]
                    rows.append(record[:header["ft_offset"]]
                                + record[header["vector_offset"]:header["attr_offset"]]
                                + record[header["ft_offset"]:header["vector_offset"]]
                                + record[header["attr_offset"]:])
                tail = directory / "unversioned-tail.index"
                tail.write_bytes(prefix + b"".join(rows) + source[start + len(vectors) * size:])
                paths.append(tail)
                expected = None
                predicates = [[[1001, 7501] if kind == 0 else [2] for kind in kinds]] * 3
                for path in paths:
                    before = path.stat()
                    layout = read_layout(path)
                    self.assertEqual([sample["attributes"] for sample in record_sample(path, layout, [0, 31, 95])],
                                     [records[row] if kinds == [0] else
                                      [values if kind == 0 else sorted(values)
                                       for kind, values in zip(kinds, records[row])]
                                      for row in (0, 31, 95)])
                    loaded = self.native.Index(space="l2", dim=4)
                    loaded.load_index(str(path))
                    loaded.generateAttrIndexes()
                    loaded.set_num_threads(1)
                    loaded.set_ef(96)
                    loaded.set_ef_top(1)
                    loaded.set_ft_flag(True)
                    loaded.set_ft_routing_flag(True)
                    loaded.set_ft_routing_min_deg(16)
                    loaded.set_ft_routing_backfill_tail(False)
                    result = loaded.hybrid_knn_query(vectors[[20, 50, 70]], predicates, k=5, num_threads=1)
                    if expected is None:
                        expected = result
                    else:
                        np.testing.assert_array_equal(result[0], expected[0])
                        np.testing.assert_array_equal(result[1], expected[1])
                    workloads = [predicates]
                    if kinds == [0, 1]:
                        workloads.append([[predicate, predicate] for predicate in predicates])
                    for workload in workloads:
                        with contextlib.redirect_stdout(io.StringIO()):
                            measurement = self.query.query_sweep(
                                loaded, vectors[[20, 50, 70]], workload, expected[0],
                                count=len(vectors), k=5, efs=[96], repeats=2)
                        self.assertEqual(measurement["rows"][0][1], 1)
                    self.assertEqual((path.stat().st_size, path.stat().st_mtime_ns),
                                     (before.st_size, before.st_mtime_ns))


if __name__ == "__main__":
    unittest.main()
