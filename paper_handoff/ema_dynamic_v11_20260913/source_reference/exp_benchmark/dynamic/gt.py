"""Exact live-set GT for the canonical ((range AND label9) OR label12) workload.

The two FAISS Flat indexes contain DISJOINT branches. Branch A is sorted by
numeric value, allowing an inclusive numeric predicate to become a contiguous
IDSelectorRange without constructing a new index for every query. External
labels are mapped explicitly; they are never mistaken for index positions.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def predicate_bounds(predicates, query_count=None):
    if not isinstance(predicates, list) or not predicates:
        raise ValueError("Expected a nonempty per-query DNF list")
    if query_count is not None and len(predicates) != query_count:
        raise ValueError("DNF count does not match the unchanged query prefix")
    bounds = []
    for predicate in predicates:
        if (not isinstance(predicate, list) or len(predicate) != 2
                or not isinstance(predicate[0], list) or len(predicate[0]) != 2
                or predicate[0][1] != [9] or predicate[1] != [[], [12]]
                or not isinstance(predicate[0][0], list) or len(predicate[0][0]) != 2):
            raise ValueError("Only the canonical mixed ((range AND [9]) OR [12]) DNF is allowed")
        low, high = predicate[0][0]
        if (type(low) is not int or type(high) is not int or low > high
                or low < -(2**31) or high >= 2**31):
            raise ValueError("Numerical bounds must be ordered inclusive int32 values")
        bounds.append((low, high))
    return np.asarray(bounds, dtype=np.int64)


@dataclass
class MixedAttributes:
    numeric: np.ndarray
    label9: np.ndarray
    label12: np.ndarray

    def __post_init__(self):
        self.numeric = np.asarray(self.numeric)
        self.label9 = np.asarray(self.label9, dtype=bool)
        self.label12 = np.asarray(self.label12, dtype=bool)
        if (self.numeric.ndim != 1 or self.label9.shape != self.numeric.shape
                or self.label12.shape != self.numeric.shape
                or not np.issubdtype(self.numeric.dtype, np.integer)):
            raise ValueError("Invalid mixed attribute arrays")

    @classmethod
    def from_records(cls, records, expected_count):
        if not isinstance(records, list) or len(records) != expected_count:
            raise ValueError("Attribute record count mismatch")
        numeric = np.empty(expected_count, dtype=np.int32)
        nine = np.empty(expected_count, dtype=bool)
        twelve = np.empty(expected_count, dtype=bool)
        for i, record in enumerate(records):
            if (not isinstance(record, list) or len(record) != 2
                    or not isinstance(record[0], list) or len(record[0]) != 1
                    or type(record[0][0]) is not int
                    or not -(2**31) <= record[0][0] < 2**31
                    or not isinstance(record[1], list)
                    or any(type(label) is not int or label < 0 for label in record[1])):
                raise ValueError(f"Malformed [[numeric], [categorical labels]] at row {i}")
            numeric[i] = record[0][0]
            nine[i] = 9 in record[1]
            twelve[i] = 12 in record[1]
        return cls(numeric, nine, twelve)

    @classmethod
    def load(cls, path, expected_count):
        with Path(path).open() as stream:
            return cls.from_records(json.load(stream), expected_count)

    def matches(self, rows, low, high):
        return (self.label12[rows]
                | (self.label9[rows] & (self.numeric[rows] >= low)
                   & (self.numeric[rows] <= high)))


def read_fvecs(path, count, dimension, *, integer_sift=False):
    path = Path(path)
    stride = (dimension + 1) * 4
    if path.stat().st_size < count * stride or path.stat().st_size % stride:
        raise ValueError("Incomplete or unexpectedly sized fvecs file")
    records = np.memmap(path, dtype="<i4", mode="r", shape=(count, dimension + 1))
    vectors = records.view("<f4")[:, 1:]
    for start in range(0, count, 65_536):
        end = min(start + 65_536, count)
        data = vectors[start:end]
        if not np.all(records[start:end, 0] == dimension) or not np.isfinite(data).all():
            raise ValueError("Invalid fvecs dimension or nonfinite payload")
        if integer_sift and (np.any(data < 0) or np.any(data > 255)
                             or not np.equal(data, np.rint(data)).all()):
            raise ValueError("Exact canonical tie proof requires integer SIFT coordinates in [0,255]")
    return vectors


@contextmanager
def faiss_threads(faiss, threads):
    previous = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(threads)
    try:
        yield
    finally:
        faiss.omp_set_num_threads(previous)


def validate_range_api(faiss):
    index = faiss.IndexFlatL2(2)
    index.add(np.asarray([[0, 0], [2, 0], [3, 0], [1, 0]], dtype=np.float32))
    params = faiss.SearchParameters()
    params.sel = faiss.IDSelectorRange(1, 3)
    distances, positions = index.search(np.zeros((1, 2), dtype=np.float32), 3, params=params)
    if (positions.tolist() != [[1, 2, -1]]
            or distances[0, :2].tolist() != [4.0, 9.0]):
        raise RuntimeError("Installed FAISS does not honor exact Flat IDSelectorRange")


def selection_summary(counts, live_count, occupied_count):
    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim != 1 or not len(counts) or np.any(counts < 0) or np.any(counts > live_count):
        raise ValueError("Invalid per-query matching live counts")
    denominator = live_count if live_count else 1
    return {
        "live": int(live_count), "occupied": int(occupied_count),
        "matching_live_min": int(counts.min()),
        "matching_live_mean": float(counts.mean()),
        "matching_live_max": int(counts.max()),
        "matching_live_sum": int(counts.sum()),
        "selectivity_denominator": "live IDs, NOT original occupied count",
        "selectivity_live_min": float(counts.min() / denominator),
        "selectivity_live_mean": float(counts.mean() / denominator),
        "selectivity_live_max": float(counts.max() / denominator),
        "matching_over_occupied_mean": float(counts.mean() / occupied_count),
        "matching_counts_sha256": array_digest(counts),
    }


@dataclass
class ExactTruth:
    labels: np.ndarray
    distances: np.ndarray
    matching_counts: np.ndarray
    metadata: dict


class ExactMixedDNF:
    def __init__(self, vectors, attributes, *, external_ids=None, threads=32):
        import faiss

        self.faiss = faiss
        self.vectors = np.asarray(vectors)
        self.attributes = attributes
        self.threads = threads
        n = len(self.vectors)
        self.dense_external_ids = external_ids is None
        self.label_order = None
        self.external_ids = (np.arange(n, dtype=np.int64) if external_ids is None
                             else np.asarray(external_ids, dtype=np.int64))
        if (self.vectors.ndim != 2 or self.vectors.dtype != np.float32
                or len(attributes.numeric) != n or self.external_ids.shape != (n,)
                or np.any(self.external_ids < 0) or len(np.unique(self.external_ids)) != n
                or threads < 1):
            raise ValueError("Invalid exact-GT inputs or external label mapping")
        with faiss_threads(faiss, threads):
            validate_range_api(faiss)

    def _build_branch(self, rows):
        index = self.faiss.IndexFlatL2(self.vectors.shape[1])
        for start in range(0, len(rows), 65_536):
            index.add(np.ascontiguousarray(self.vectors[rows[start:start + 65_536]]))
        return index

    def compute(self, queries, predicates, live, k=10, progress=None, occupied_count=None):
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        bounds = predicate_bounds(predicates, len(queries))
        live = np.asarray(live, dtype=bool)
        occupied_count = len(live) if occupied_count is None else occupied_count
        if (queries.ndim != 2 or queries.shape[1] != self.vectors.shape[1]
                or live.shape != (len(self.vectors),) or k < 1 or not np.isfinite(queries).all()
                or not int(live.sum()) <= occupied_count <= len(live)):
            raise ValueError("Invalid query/live-mask shape")
        started = time.perf_counter()
        attr = self.attributes
        branch_b = np.flatnonzero(live & attr.label12)
        branch_a = np.flatnonzero(live & attr.label9 & ~attr.label12)
        order = np.lexsort((self.external_ids[branch_a], attr.numeric[branch_a]))
        branch_a = branch_a[order]
        numeric_a = attr.numeric[branch_a]
        left = np.searchsorted(numeric_a, bounds[:, 0], side="left")
        right = np.searchsorted(numeric_a, bounds[:, 1], side="right")
        counts = (len(branch_b) + right - left).astype(np.int64)
        labels = np.full((len(queries), k), -1, dtype=np.int64)
        distances = np.full((len(queries), k), np.inf, dtype=np.float32)
        with faiss_threads(self.faiss, self.threads):
            a_index, b_index = self._build_branch(branch_a), self._build_branch(branch_b)
            built = time.perf_counter()
            db, ib = b_index.search(queries, k)
            for row, (lo, hi) in enumerate(zip(left, right)):
                params = self.faiss.SearchParameters()
                params.sel = self.faiss.IDSelectorRange(int(lo), int(hi))
                da, ia = a_index.search(queries[row:row + 1], k, params=params)
                keep_a, keep_b = ia[0] >= 0, ib[row] >= 0
                ids = np.concatenate((self.external_ids[branch_a[ia[0, keep_a]]],
                                      self.external_ids[branch_b[ib[row, keep_b]]]))
                ds = np.concatenate((da[0, keep_a], db[row, keep_b]))
                chosen = np.lexsort((ids, ds))[:k]
                labels[row, :len(chosen)] = ids[chosen]
                distances[row, :len(chosen)] = ds[chosen]
                if len(chosen) != min(k, counts[row]):
                    raise RuntimeError("Flat exact search returned an inconsistent number of matches")
                if progress and (row + 1) % 100 == 0:
                    progress(row + 1, len(queries))
        metadata = {
            "method": "disjoint live label12 UNION (live label9 EXCLUDING label12 in inclusive range)",
            "backend": "FAISS IndexFlatL2 + verified SearchParameters/IDSelectorRange",
            "faiss_version": self.faiss.__version__, "threads": self.threads,
            "branch_b_live": len(branch_b), "branch_a_live_before_range": len(branch_a),
            "range_boundary": "searchsorted(low,left), searchsorted(high,right)",
            "tie_policy": "valid exact top-k; candidate ties ordered by external ID; no claim of global ID tie-break",
            "insufficient_results": "labels=-1, distances=+inf; matching_counts records the actual cardinality",
            "build_wall_s": built - started, "total_wall_s": time.perf_counter() - started,
            "labels_sha256": array_digest(labels), "distances_sha256": array_digest(distances),
            "source_rows": len(live),
            **selection_summary(counts, int(live.sum()), occupied_count),
        }
        return ExactTruth(labels, distances, counts, metadata)

    def rows_for_labels(self, labels):
        labels = np.asarray(labels)
        if np.any(labels < 0):
            raise ValueError("Unknown external ground-truth ID")
        if self.dense_external_ids:
            if np.any(labels >= len(self.external_ids)):
                raise ValueError("Unknown external ground-truth ID")
            return labels.astype(np.int64, copy=False)
        if self.label_order is None:
            self.label_order = np.argsort(self.external_ids)
        order = self.label_order
        positions = np.searchsorted(self.external_ids[order], labels)
        if np.any(positions >= len(order)):
            raise ValueError("Unknown external ground-truth ID")
        rows = order[positions]
        if not np.array_equal(self.external_ids[rows], labels):
            raise ValueError("Unknown external ground-truth ID")
        return rows

    def validate_canonical(self, canonical, exact, queries, predicates, live):
        """Require equality of exact distances; different IDs need explicit boundary-tie proofs."""
        canonical = np.asarray(canonical)
        if (canonical.shape != exact.labels.shape or not np.issubdtype(canonical.dtype, np.integer)
                or np.any(canonical < 0)):
            raise ValueError("Invalid canonical GT shape or IDs")
        rows = self.rows_for_labels(canonical)
        bounds = predicate_bounds(predicates, len(queries))
        distances = np.empty(canonical.shape, dtype=np.float64)
        tie_rows, reordered_rows = [], []
        for q, (low, high) in enumerate(bounds):
            if (len(set(canonical[q].tolist())) != canonical.shape[1]
                    or not live[rows[q]].all()
                    or not self.attributes.matches(rows[q], low, high).all()):
                raise ValueError(f"Canonical GT has duplicate/deleted/nonmatching IDs at query {q}")
            delta = self.vectors[rows[q]].astype(np.float64) - queries[q].astype(np.float64)
            distances[q] = np.sum(delta * delta, axis=1)
            if not np.array_equal(distances[q], exact.distances[q].astype(np.float64)):
                raise ValueError(f"Canonical GT is not exact, or distance arithmetic differs, at query {q}")
            canonical_set, exact_set = set(canonical[q].tolist()), set(exact.labels[q].tolist())
            if canonical_set != exact_set:
                old, new = sorted(canonical_set - exact_set), sorted(exact_set - canonical_set)
                all_rows = self.rows_for_labels(np.asarray(old + new, dtype=np.int64))
                delta = self.vectors[all_rows].astype(np.float64) - queries[q].astype(np.float64)
                tie_distances = np.sum(delta * delta, axis=1)
                boundary = float(distances[q, -1])
                if not np.all(tie_distances == boundary):
                    raise ValueError(f"Canonical mismatch is NOT an equal-distance boundary tie at query {q}")
                tie_rows.append({"query": q, "canonical_only": old, "exact_only": new,
                                 "squared_l2_float64": boundary,
                                 "proof": "ALL symmetric-difference IDs equal the exact kth distance"})
            elif not np.array_equal(canonical[q], exact.labels[q]):
                reordered_rows.append(q)
        return {
            "validated_all_queries": len(queries), "canonical_retained": True,
            "distance_comparison": "exact equality, float64 direct squared L2 versus exhaustive Flat distances",
            "boundary_tie_rows": tie_rows, "order_only_rows": reordered_rows,
            "canonical_labels_sha256": array_digest(canonical.astype(np.int64)),
            "independent_exact_labels_sha256": array_digest(exact.labels),
        }
