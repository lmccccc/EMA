"""Isolated lifecycle gate with separately reported ANN quality and FT diagnostics."""

import argparse
import copy
import json
from pathlib import Path
import resource

import numpy as np

from .gt import ExactMixedDNF, MixedAttributes
from .protocol import require_current_query_native
from .runner import configure_index
from .runtime import (
    CANDIDATE_ORDER, DISTANCE_ORDER_VERSION, MARKER_OWNER_VERSION,
    NUMERIC_EDGE_FORMAT, NUMERIC_MARKER_VERSION, PICKLE_STATE_VERSION,
    immutable_json, import_native, native_state,
    read_index_header, validate_counters,
    verify_index_records, verify_index_tail,
)


SCHEMA = "canonical-native-lifecycle-canary-v4"
LIFECYCLE_CHECKS = (
    "state_counts", "complete_records", "live_predicate_labels",
    "returned_squared_l2", "save_load_results_identical",
)


def eligible_ids(records, predicate, active):
    return np.asarray([
        row for row, record in enumerate(records) if active[row] and any(
            (not numeric or numeric[0] <= record[0][0] <= numeric[1])
            and all(label in record[1] for label in categorical)
            for numeric, categorical in predicate
        )
    ], dtype=np.int64)


def query_evidence(index, vectors, records, queries, predicates, active, oracle, *, ft_enabled=True):
    labels, distances = index.hybrid_knn_query_dnf(queries, predicates, k=10, num_threads=1)
    labels, distances = np.asarray(labels), np.asarray(distances)
    if (labels.shape != (len(queries), 10) or distances.shape != labels.shape
            or not np.issubdtype(labels.dtype, np.integer)
            or np.any(labels < 0) or np.any(labels >= len(vectors)) or not active[labels].all()):
        raise RuntimeError("Canary returned malformed/inactive external labels")
    exact = oracle.compute(queries, predicates, active, k=10, occupied_count=index.get_current_count())
    enumerated, per_query = [], []
    hits = 0
    for q, predicate in enumerate(predicates):
        eligible = eligible_ids(records, predicate, active)
        if len(np.unique(labels[q])) != 10 or not np.isin(labels[q], eligible).all():
            raise RuntimeError("Canary returned duplicate or predicate-invalid labels")
        delta = vectors[labels[q]].astype(np.float64) - queries[q].astype(np.float64)
        if not np.array_equal(distances[q], np.sum(delta * delta, axis=1)):
            raise RuntimeError("Canary distances differ from direct squared L2 for returned labels")
        if np.any(np.diff(distances[q]) < 0):
            raise RuntimeError("Canary returned unsorted neighbor distances")
        delta = vectors[eligible].astype(np.float64) - queries[q].astype(np.float64)
        nearest = sorted(zip(np.sum(delta * delta, axis=1).tolist(), eligible.tolist()))[:10]
        if len(nearest) != 10 or not np.array_equal([distance for distance, _ in nearest], exact.distances[q]):
            raise RuntimeError("Independent enumeration differs from FAISS mixed-DNF GT")
        enumerated.append([label for _, label in nearest])
        truth_ids, returned_ids = set(exact.labels[q].tolist()), set(labels[q].tolist())
        count = len(truth_ids & returned_ids)
        hits += count
        per_query.append({
            "query": q, "hits": count, "recall": count / 10,
            "missing": [{"label": int(label), "distance": float(distance)}
                        for label, distance in zip(exact.labels[q], exact.distances[q]) if label not in returned_ids],
            "extra": [{"label": int(label), "distance": float(distance)}
                      for label, distance in zip(labels[q], distances[q]) if label not in truth_ids],
        })
    proof = oracle.validate_canonical(np.asarray(enumerated), exact, queries, predicates, active)
    return {
        "k": 10, "ef": 300, "ft_enabled": ft_enabled, "hits": hits, "total": int(labels.size),
        "recall": hits / labels.size, "labels": labels.tolist(), "distances": distances.tolist(),
        "exact_labels": exact.labels.tolist(), "exact_distances": exact.distances.tolist(),
        "per_query": per_query, "independent_exact_gt_verified": True, "enumeration_tie_proof": proof,
    }


def verify_complete_records(path, active, attributes, records, vectors):
    header = read_index_header(path)
    checks = verify_index_records(path, header, active, attributes, vectors)
    checks.update(verify_index_tail(path, header))
    if checks["vector_mapping_sample_count"] != header["count"]:
        raise RuntimeError("Tiny canary must verify every stored vector, including retired records")
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    try:
        n, stride, start = header["count"], header["record_bytes"], header["records_offset"]
        labels = np.ndarray((n,), dtype="<u8", buffer=raw, offset=start + header["label_offset"], strides=(stride,))
        categorical = np.ndarray(
            (n, header["cate_int_words"]), dtype="<u4", buffer=raw,
            offset=start + header["attr_offset"] + 4 * header["attr_positions"][1], strides=(stride, 4),
        )
        expected = np.zeros_like(categorical)
        for row, label in enumerate(labels):
            for category in records[int(label)][1]:
                expected[row, category // 32] |= np.uint32(1 << (category % 32))
        if not np.array_equal(categorical, expected):
            raise RuntimeError("Canary lost or changed a complete categorical record")
    finally:
        raw._mmap.close()
    return {**checks, "complete_categorical_bitsets_verified": True,
            "index_format": header["format"], "numeric_marker_semantics": header["numeric_marker_semantics"],
            "numeric_marker_version": header["numeric_marker_version"],
            "marker_owner_version": header["marker_owner_version"],
            "distance_order_version": header["distance_order_version"],
            "candidate_order": header["candidate_order"]}


def boundary_diagnostic(index, formal, vectors, records, queries, predicates, active, oracle):
    """Fixture-only probes; neither variant replaces the formal query/recall."""
    widened = copy.deepcopy(predicates)
    widened[3][0][0][1] = 64
    eligible = eligible_ids(records, predicates[3], active)
    if not np.array_equal(eligible, eligible_ids(records, widened[3], active)):
        raise RuntimeError("Boundary diagnostic unexpectedly changed the eligible set")
    upper = query_evidence(index, vectors, records, queries, widened, active, oracle)
    index.set_ft_flag(False)
    try:
        without_ft = query_evidence(index, vectors, records, queries, predicates, active, oracle, ft_enabled=False)
    finally:
        index.set_ft_flag(True)
    restored = query_evidence(index, vectors, records, queries, predicates, active, oracle)
    if (restored["labels"], restored["distances"]) != (formal["labels"], formal["distances"]):
        raise RuntimeError("Diagnostic probes changed the restored formal query")
    present = {
        "formal": 63 in formal["labels"][3], "upper_bound_64": 63 in upper["labels"][3],
        "ft_disabled": 63 in without_ft["labels"][3],
    }
    return {
        "query": 3, "label": 63, "formal_numeric_bounds": predicates[3][0][0],
        "diagnostic_numeric_bounds": widened[3][0][0], "eligible_ids": eligible.tolist(),
        "eligible_set_unchanged": True, "formal_query_restored": True, "label_returned": present,
        "sensitivity_observed": not present["formal"] and (present["upper_bound_64"] or present["ft_disabled"]),
        "upper_bound_64": upper, "ft_disabled": without_ft,
        "scope": "tiny fixture diagnostic only; formal predicates/FT settings and reported ANN recall are unchanged",
    }


def exercise(native, output):
    rng = np.random.default_rng(4096)
    vectors = rng.integers(0, 128, (128, 8)).astype(np.float32)
    records = [[[i % 101], [0] + ([9] if i % 2 else []) + ([12] if i % 3 else [])]
               for i in range(128)]
    levels = np.zeros(128, dtype=np.int32)
    levels[[0, 8, 16, 64, 96]] = 1
    queries = vectors[5:9]
    predicates = [[[[i, 60 + i], [9]], [[], [12]]] for i in range(len(queries))]
    attributes = MixedAttributes.from_records(records, 128)
    oracle = ExactMixedDNF(vectors, attributes, threads=1)
    active = np.zeros(128, dtype=bool)
    stages, diagnostics = [], {}

    def round_trip(index, phase, occupied, retired, diagnose=False):
        configure_index(index)
        index.set_ef(300)
        state = native_state(index, occupied, retired)
        before = query_evidence(index, vectors, records, queries, predicates, active, oracle)
        if diagnose:
            diagnostics[phase] = boundary_diagnostic(index, before, vectors, records, queries, predicates, active, oracle)
        path = output / f"{phase}.index"
        index.save_index(str(path))
        checks = verify_complete_records(path, active, attributes, records, vectors)
        loaded = native.Index(space="l2", dim=8)
        loaded.load_index(str(path), max_elements=128, top_elements=1, dynamic=True, allow_replace_deleted=False)
        configure_index(loaded)
        loaded.set_ef(300)
        after = query_evidence(loaded, vectors, records, queries, predicates, active, oracle)
        if (before["labels"], before["distances"]) != (after["labels"], after["distances"]):
            raise RuntimeError(f"Canary save/load changed query results at {phase}")
        native_state(loaded, occupied, retired)
        saved_state = loaded.__getstate__()[0]
        versions = {name: saved_state.get(name) for name in (
            "numeric_marker_version", "marker_owner_version", "distance_order_version", "ser_version")}
        del saved_state
        if checks["index_format"] != NUMERIC_EDGE_FORMAT or versions != {
                "numeric_marker_version": NUMERIC_MARKER_VERSION, "marker_owner_version": MARKER_OWNER_VERSION,
                "distance_order_version": DISTANCE_ORDER_VERSION, "ser_version": PICKLE_STATE_VERSION}:
            raise RuntimeError("Format10 and pickle numeric/owner Marker/ranking/serialization versions disagree")
        alpha = loaded.get_attr_sort_alpha()
        if alpha != 0:
            raise RuntimeError("Vector-only ranking requires attr_sort_alpha=0")
        stages.append({"phase": phase, "state": state, "records": checks, "ann": before,
                       "save_load_results_identical": True, "pickle_state_versions": versions,
                       "attr_sort_alpha": alpha})
        return loaded

    index = native.Index(space="l2", dim=8)
    index.init_index(max_elements=128, top_elements=3, M=40, ef_construction=300,
                     ft_bits=128, attr_type=[0, 1], max_cate_size=20, edge_level_ft=True,
                     allow_replace_deleted=False)
    index.initAttrMapping(records)
    index.add_items(vectors[:64], records[:64], np.arange(64, dtype=np.uint64),
                    num_threads=1, levels=levels[:64], replace_deleted=False)
    active[:64] = True
    index = round_trip(index, "prefix", 64, 0, diagnose=True)
    index.add_items(vectors[64:96], records[64:96], np.arange(64, 96, dtype=np.uint64),
                    num_threads=1, levels=levels[64:96], replace_deleted=False)
    active[64:96] = True
    index = round_trip(index, "inserted", 96, 0)
    counters = index.delete_items(np.arange(3, dtype=np.uint64), num_threads=1)
    validate_counters(counters, 3)
    native_state(index, 96, 3)
    active[:3] = False
    index.add_items(vectors[96:99], records[96:99], np.arange(96, 99, dtype=np.uint64),
                    num_threads=1, levels=levels[96:99], replace_deleted=False)
    active[96:99] = True
    index = round_trip(index, "replaced", 99, 3, diagnose=True)
    native_state(index, 99, 3)
    warnings = []
    if any(stage["ann"]["recall"] < 1 for stage in stages):
        warnings.append("ANN misses observed; lifecycle integrity is not an exact-neighbor guarantee")
    if any(probe["sensitivity_observed"] for probe in diagnostics.values()):
        warnings.append("Numeric Marker endpoint diagnostic triggered: pre-fix builds have a confirmed "
                        "inclusive-upper-slot false-negative, not merely an approximate-recall difference")
    return {
        "schema": SCHEMA, "status": "complete", "prefix": 64, "after_loaded_add": 96,
        "after_replacement": {"occupied": 99, "live": 96, "retired": 3},
        "lifecycle": dict.fromkeys(LIFECYCLE_CHECKS, True), "ann": stages[-1]["ann"],
        "stages": stages, "diagnostics": diagnostics, "warnings": warnings, "replace_deleted": False,
        "marker_fidelity_certified": False,
        "candidate_completeness_certified": False,
        "index_format": stages[-1]["records"]["index_format"],
        "numeric_marker_semantics": stages[-1]["records"]["numeric_marker_semantics"],
        "numeric_marker_version": stages[-1]["pickle_state_versions"]["numeric_marker_version"],
        "marker_owner_version": stages[-1]["pickle_state_versions"]["marker_owner_version"],
        "distance_order_version": stages[-1]["pickle_state_versions"]["distance_order_version"],
        "candidate_order": CANDIDATE_ORDER, "attr_sort_alpha": stages[-1]["attr_sort_alpha"],
        "pickle_state_version": stages[-1]["pickle_state_versions"]["ser_version"],
        "marker_coverage_limit": "Metadata and the padded128-slot lifecycle fixture do not certify quantile max-bin "
                                 "mapping, witness-to-owner alignment, or complete Marker fidelity",
        "purpose": "tiny lifecycle gate; ANN recall is measured separately, not required to equal100%",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_current_query_native(args.expected_sha256)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    args.output.mkdir(parents=True, exist_ok=False)
    native, info = import_native(args.native_dir, args.expected_sha256)
    report = {**exercise(native, args.output), "native": info}
    immutable_json(args.output / "report.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
