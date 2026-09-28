"""Canonical inputs, complete-record traces, and production-style level plans."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .gt import ExactMixedDNF, MixedAttributes, array_digest, faiss_threads, predicate_bounds, read_fvecs
from .runtime import (
    NUMERIC_EDGE_FORMAT, NUMERIC_MARKER_SEMANTICS, file_info, object_digest, require_unchanged,
)


OPERATIONS = ("insert", "delete", "point_update")
PROTOCOL = "sift10m-mixed-dnf-dynamic-rebuild-v5"
TARGET_RECALL = 0.95
QUERY_IMPLEMENTATION = NUMERIC_MARKER_SEMANTICS
REBUILD_POLICY = "fix_rebuild_rerun_all"
INCOMPATIBLE_GRAPH_NATIVES = frozenset({
    "4f82614654509bc5987988fa473fa23cc0fa3a08c15d64f8db6a78d9cf0e7fc7",
    "40d52c4a3b57e3a49517fd23e2619946134432b93f02021336e9305a48171a3b",
    "4e6efcbe5c88e60f035c9dfdb146c5d93c8ba53143cfbb71cbdfcdd13d85e6d4",
    "9901ba84cf52f127cbaa5a867697434388e56bdcbd39044b47be2b913a84b0a1",
    "7d53d7ddb4cd3633d56adf5e4eb03f2ea953bf536bf8944c27b1ecbabb573459",
    "6fa57c172261bdf6d56e5f5b3381b37fb2594075768b68acc69cdddaf844f0c3",
    "4c775ff6a7d66508e9cd315a5e9d6e8828ab605e309d662eada31639902ecb58",
    "79c15e5d81f6dfb6a238a76b4c750aab90234c6bee4f556da2a2199bc5692836",
})
TRACE_VERSIONS = {
    "insert": "source-prefix-5M-to-10M-v1",
    "delete": "default_rng-12345-permutation-10M-v1",
    "point_update": "complete-record-consecutive-fresh-label-v1",
}
LEVEL_POLICY = {
    "version": "production-centroid-representatives-whole-source-v1",
    "source": "all source rows; use identical frozen slices for initial prefix and later adds",
    "clusters": "floor(sqrt(source_count))*4",
    "training_limit": 2_560_000, "iterations": 25,
    "sampling": "numpy.default_rng(level_seed).choice(N, min(N,2560000), replace=False)",
    "representative": "nearest assigned source row per centroid; equal distances choose lowest source row",
    "native_random_seed": 100,
}
QUERY_SHA = "5fc3ea9c506f7ca2b250879f15c1674a8e70b1e7afb7c2cea60fa4396c61cb75"
ORDER_SHA = "ad71165820148868ea655be3fe91efea9d9e9184542373fdfc8d5c280b0e3baa"
INPUT_SHA = {
    "base_vectors": "d3a64e449eac7b1abe20830970223d9f30f804da908ceeaebc3accac8e42b76f",
    "queries": "4e686785082f56916a6422b120ad4c54b4110701195c17fa582391156bcb32cc",
    "attributes": "84f6dc200ebd1458d98e604c727d94b0ff0ee2c630dcd37d66262b5a2ad72542",
    "predicates": "84638044eaccd90f23cb94b954eda91e0cff7bd6e3a4de0b2ed24d32f7aa4ebc",
    "canonical_gt0": "94043b7543f447e45a6e84fa4e7213e4656ddb9ea31402b402017e066f91e18c",
}
QUERY_CONFIG = {
    "space": "l2", "dimension": 128, "query_count": 1000, "k": 10,
    "attr_type": [0, 1], "M": 40, "ef_construction": 300,
    "ft_bits_per_attribute": 128, "edge_ft": True,
    "thresholds": [0.0001, 0.0001, 0.0001],
    "ef_top": 1, "ft_flag": True, "ft_routing_flag": True,
    "ft_routing_min_deg": 8, "ft_routing_backfill_tail": False,
    "threads_query": 1, "target_recall": TARGET_RECALL,
    "api": "hybrid_knn_query_dnf",
    "predicate": "PER QUERY: ((inclusive numeric range AND label9) OR label12)",
}


def require_current_query_native(sha):
    if sha in INCOMPATIBLE_GRAPH_NATIVES:
        raise RuntimeError("Superseded graph native lacks current Marker/ranking semantics; rebuild with "
                           "format10 edge FT, numeric2/owner2/distance-order1/state4")


def input_paths(data_root):
    root = Path(data_root).resolve(strict=True)
    label = root / "label/arbi_0_1_random"
    return {
        "base_vectors": root / "sift10m.fvecs", "queries": root / "sift10m_query.fvecs",
        "attributes": label / "attr_arbi_0_1_random.json",
        "predicates": label / "predicate_dnf_or_T10.json",
        "canonical_gt0": label / "gt_dnf_or_T10.json",
    }


def prefix_info(path, byte_count):
    path = Path(path)
    before = path.stat()
    if path.is_symlink():
        raise RuntimeError("Symlinked input")
    with path.open("rb") as stream:
        data = stream.read(byte_count)
    if len(data) != byte_count:
        raise ValueError("Truncated pilot input")
    info = {"path": str(path), "bytes": before.st_size, "mtime_ns": before.st_mtime_ns,
            "sha256": hashlib.sha256(data).hexdigest(), "hash_scope": "pilot_prefix",
            "hashed_bytes": byte_count}
    require_unchanged(info)
    return info


def json_record_prefix(path, count):
    """Bounded pilot reads, not json.load of the entire 10M-record attribute file."""
    decoder = json.JSONDecoder()
    buffer, position = "", 0
    records = []
    with Path(path).open() as stream:
        def more():
            nonlocal buffer, position
            text = stream.read(65_536)
            if not text:
                raise ValueError("Truncated JSON attribute prefix")
            buffer, position = buffer[position:] + text, 0

        more()
        while position < len(buffer) and buffer[position].isspace():
            position += 1
        if position == len(buffer) or buffer[position] != "[":
            raise ValueError("Expected a JSON array")
        position += 1
        while len(records) < count:
            while position >= len(buffer) or buffer[position].isspace():
                if position >= len(buffer):
                    more()
                else:
                    position += 1
            if records:
                if buffer[position] != ",":
                    raise ValueError("Insufficient/malformed attribute records")
                position += 1
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                try:
                    value, end = decoder.raw_decode(buffer, position)
                    break
                except json.JSONDecodeError:
                    more()
            records.append(value)
            position = end
    return records


@dataclass
class Trace:
    operation: str
    capacity: int
    initial: int
    step: int
    seed: int = 12345

    def __post_init__(self):
        if (self.operation not in OPERATIONS or self.capacity != 2 * self.initial
                or self.step < 1 or self.initial % self.step):
            raise ValueError("Invalid canonical dynamic trace")
        self.order = np.random.default_rng(self.seed).permutation(self.capacity).astype(np.uint64)
        self.order.setflags(write=False)
        self.order_sha = array_digest(self.order)
        self._states = {}

    @property
    def progress(self):
        return list(range(0, self.initial + 1, self.step))

    def state(self, progress):
        if progress not in self.progress:
            raise ValueError("Progress is not a complete stage boundary")
        if progress in self._states:
            return self._states[progress]
        active = np.zeros(self.capacity, dtype=bool)
        if self.operation == "delete":
            occupied, deleted = self.capacity, progress
            active[:] = True
            active[self.order[:progress]] = False
        elif self.operation == "insert":
            occupied, deleted = self.initial + progress, 0
            active[:occupied] = True
        else:
            occupied, deleted = self.initial + progress, progress
            active[progress:occupied] = True
        active.setflags(write=False)
        self._states[progress] = {"occupied": occupied, "deleted": deleted, "live": int(active.sum()),
                                 "active": active}
        return self._states[progress]

    def mutation(self, previous, progress):
        if previous not in self.progress or progress not in self.progress or progress != previous + self.step:
            raise ValueError("Mutation must advance exactly one ordered stage")
        empty = np.empty(0, dtype=np.uint64)
        deleted = (self.order[previous:progress] if self.operation == "delete"
                   else np.arange(previous, progress, dtype=np.uint64) if self.operation == "point_update" else empty)
        added = (np.arange(self.initial + previous, self.initial + progress, dtype=np.uint64)
                 if self.operation != "delete" else empty)
        return deleted, added

    def logical_labels(self, external):
        external = np.asarray(external)
        return np.where(external >= self.initial, external - self.initial, external) if (
            self.operation == "point_update") else external

    def arrays(self, progress):
        state = self.state(progress)
        if self.operation == "point_update":
            logical_to_external = np.arange(self.initial, dtype=np.int64)
            logical_to_external[:progress] += self.initial
        else:
            logical_to_external = np.flatnonzero(state["active"]).astype(np.int64)
        return {"active_source_rows": state["active"], "logical_to_external": logical_to_external}

    def identity(self, progress):
        state = self.state(progress)
        arrays = self.arrays(progress)
        return {
            "operation": self.operation, "version": TRACE_VERSIONS[self.operation],
            "progress": progress, "occupied": state["occupied"], "deleted": state["deleted"], "live": state["live"],
            "source_row_equals_external_label": True,
            "replacement": "complete original vector AND original mixed attributes of row initial+logical_slot",
            "id_space_for_recall": "external source-row labels; logical recall additionally checked for point_update",
            "permutation_sha256": self.order_sha,
            **{name + "_sha256": array_digest(value) for name, value in arrays.items()},
        }


class Dataset:
    def __init__(self, paths, config, progress=lambda *args, **kwargs: None):
        self.paths, self.config = paths, config
        self.n, self.dim, self.k = config["capacity"], config["query"]["dimension"], config["query"]["k"]
        self.inputs = {}
        pilot = config["mode"] == "pilot"
        for name, path in paths.items():
            if name not in INPUT_SHA:
                raise RuntimeError("Only canonical raw data/query/GT inputs are accepted; never a legacy base index")
            progress("hashing_input", input=name)
            if pilot and name == "base_vectors":
                self.inputs[name] = prefix_info(path, self.n * (self.dim + 1) * 4)
            elif pilot and name == "attributes":
                self.inputs[name] = prefix_info(path, min(Path(path).stat().st_size, 65_536))
            else:
                self.inputs[name] = file_info(path)
                if config["mode"] == "official" and self.inputs[name]["sha256"] != INPUT_SHA[name]:
                    raise RuntimeError(f"Not the canonical SIFT10M {name} input")
        self.predicates = json.loads(Path(paths["predicates"]).read_text())
        self.bounds = predicate_bounds(self.predicates, config["query"]["query_count"])
        self.queries = np.ascontiguousarray(read_fvecs(
            paths["queries"], len(self.predicates), self.dim, integer_sift=True))
        if self.config["mode"] == "official" and array_digest(self.queries) != QUERY_SHA:
            raise RuntimeError("Canonical query tensor/order mismatch")
        self.base = read_fvecs(paths["base_vectors"], self.n, self.dim, integer_sift=True)
        if pilot:
            self.records = json_record_prefix(paths["attributes"], self.n)
            self.inputs["attributes"]["record_prefix_sha256"] = object_digest(self.records)
        else:
            with Path(paths["attributes"]).open() as stream:
                self.records = json.load(stream)
        self.attributes = MixedAttributes.from_records(self.records, self.n)
        canonical_rows = json.loads(Path(paths["canonical_gt0"]).read_text())
        self.canonical = np.asarray(canonical_rows[:len(self.queries)], dtype=np.int64)
        if self.config["mode"] == "official" and (
                len(canonical_rows) != 10000 or self.canonical.shape != (1000, 10)
                or len({json.dumps(p) for p in self.predicates}) != 996
                or int(self.attributes.label9.sum()) != 999261
                or int(self.attributes.label12.sum()) != 700596):
            raise RuntimeError("Canonical mixed-DNF cardinality/schema mismatch")
        self.oracle = ExactMixedDNF(self.base, self.attributes, threads=config["threads_gt"])
        self.identity = {
            "query_payload_sha256": array_digest(self.queries),
            "predicates_payload_sha256": object_digest(self.predicates),
            "numeric_payload_sha256": array_digest(self.attributes.numeric),
            "label9_payload_sha256": array_digest(self.attributes.label9),
            "label12_payload_sha256": array_digest(self.attributes.label12),
            "complete_records_sha256": self.inputs["attributes"].get(
                "record_prefix_sha256", self.inputs["attributes"]["sha256"]),
            "input_sha256": {name: info["sha256"] for name, info in self.inputs.items()},
            "integer_distance_domain": "all base/query coordinates integral in [0,255]; dim<=128",
        }
        self.verify()

    def verify(self):
        for info in self.inputs.values():
            require_unchanged(info)


def representative_levels(vectors, seed, threads):
    """Same FAISS clustering/representative rule as tests/hashann.py, with explicit RNG."""
    import faiss

    n, dim = vectors.shape
    clusters = min(n, int(math.sqrt(n)) * 4)
    if not n or not clusters:
        raise ValueError("Cannot generate levels for an empty source")
    rng = np.random.default_rng(seed)
    train_ids = (rng.choice(n, size=LEVEL_POLICY["training_limit"], replace=False)
                 if n > LEVEL_POLICY["training_limit"] else np.arange(n, dtype=np.int64))
    with faiss_threads(faiss, threads):
        clustering = faiss.Clustering(dim, clusters)
        clustering.seed, clustering.niter = seed, LEVEL_POLICY["iterations"]
        assignment = faiss.IndexFlatL2(dim)
        clustering.train(np.ascontiguousarray(vectors[train_ids]), assignment)
        centroids = faiss.vector_to_array(clustering.centroids).reshape(clusters, dim)
        index = faiss.IndexFlatL2(dim)
        index.add(centroids)
        best = np.full(clusters, np.inf)
        closest = np.full(clusters, -1, dtype=np.int64)
        for start in range(0, n, 65_536):
            distances, groups = index.search(np.ascontiguousarray(vectors[start:start + 65_536]), 1)
            for offset, (distance, group) in enumerate(zip(distances[:, 0], groups[:, 0])):
                if distance < best[group]:
                    best[group], closest[group] = distance, start + offset
    if np.any(closest < 0):
        raise RuntimeError("Empty centroid assignments; do not silently mark source row -1 as an entry point")
    levels = np.zeros(n, dtype=np.int32)
    levels[closest] = 1
    return levels, {
        **LEVEL_POLICY, "seed": seed, "clusters": clusters,
        "training_ids_sha256": array_digest(train_ids.astype(np.int64)),
        "centroids_sha256": array_digest(centroids), "representatives_sha256": array_digest(closest),
        "levels_sha256": array_digest(levels), "faiss_version": faiss.__version__,
    }
