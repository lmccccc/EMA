"""Query an existing HashANN index; never build an index or regenerate inputs."""

import ast
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np

from hashann import HashANN
from utils import arg_init, fvecs_read, read_multy_attr


def is_dnf_predicate(raw_predicate):
    if not raw_predicate or not raw_predicate[0]:
        return False
    first = raw_predicate[0][0]
    return isinstance(first, list) and bool(first) and isinstance(first[0], list)


def validate_ids(ids, count, query_count, k):
    ids = np.asarray(ids)
    if (ids.shape != (query_count, k) or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 0) or np.any(ids >= count)
            or np.any(np.diff(np.sort(ids, axis=1), axis=1) == 0)):
        raise ValueError("Missing, duplicate, sentinel or out-of-range query/GT labels")
    return ids


def load_query_data(query_file, qrange_file, gt_file, N, Nq, k):
    if N <= 0 or Nq <= 0 or k <= 0:
        raise ValueError("Positive corpus size, query count and k are required")
    if Path(query_file).suffix != ".fvecs":
        raise ValueError("Expected the original fvecs query file")
    if Path(qrange_file).suffix != ".json" or Path(gt_file).suffix != ".json":
        raise ValueError("Expected the original predicate and GT JSON files")
    queries = fvecs_read(query_file)
    predicates = read_multy_attr(qrange_file)
    truth = np.asarray(read_multy_attr(gt_file))
    if (len(queries) < Nq or not np.isfinite(queries[:Nq]).all()
            or len(predicates) < Nq or truth.ndim != 2
            or truth.shape[0] < Nq or truth.shape[1] < k):
        raise ValueError("Original query/predicate/GT files do not contain the requested prefix")
    # Select columns, not reshape: a stored top100 row is still one query.
    truth = validate_ids(truth[:Nq, :k], N, Nq, k)
    return np.ascontiguousarray(queries[:Nq]), predicates[:Nq], truth


def array_hash(value):
    return hashlib.sha256(memoryview(np.ascontiguousarray(value))).hexdigest()


def query_sweep(index, queries, predicates, truth, *, count, k, efs, ef_top=1,
                use_ft=True, routing_min_deg=16, backfill_tail=False,
                repeats=3, target_recall=None, on_point=None, validate_labels=None,
                on_result=None):
    if (len(queries) == 0 or k <= 0 or not efs
            or any(type(ef) is not int or ef < k for ef in efs)
            or type(repeats) is not int or repeats <= 0 or ef_top <= 0
            or target_recall is not None and not 0 < target_recall <= 1):
        raise ValueError("Invalid query sweep parameters")
    if target_recall is not None and any(a >= b for a, b in zip(efs, efs[1:])):
        raise ValueError("Recall-based early stopping requires an increasing ef grid")
    validate_ids(truth, count, len(queries), k)
    if len(predicates) != len(queries):
        raise ValueError("Predicate and query counts differ")
    dnf = is_dnf_predicate(predicates)
    if dnf:
        api = "hybrid_knn_query_dnf"
    elif hasattr(index, "hybrid_knn_query_with_stats"):
        api = "hybrid_knn_query_with_stats"
    else:
        api = "hybrid_knn_query"
    method = getattr(index, api)
    # The statistics binding is sequential and does not accept num_threads.
    query_arguments = {"k": k} if api.endswith("_with_stats") else {"k": k, "num_threads": 1}
    index.set_num_threads(1)
    index.set_ef_top(ef_top)
    index.set_ft_flag(use_ft)
    index.set_ft_routing_flag(True)
    index.set_ft_routing_min_deg(routing_min_deg)
    index.set_ft_routing_backfill_tail(backfill_tail)
    print(f"Query API: {api}; queries={len(queries)}; threads=1; repeats={repeats}", flush=True)
    rows, points = [], []
    for ef in efs:
        index.set_ef(ef)
        warmup = min(3, len(queries))
        method(queries[:warmup], predicates[:warmup], **query_arguments)
        samples, identity = [], None
        for repeat in range(repeats):
            thread_start = time.thread_time()
            start = time.perf_counter()
            result = method(queries, predicates, **query_arguments)
            elapsed = time.perf_counter() - start
            thread_s = time.thread_time() - thread_start
            if not np.isfinite(elapsed) or elapsed <= 0:
                raise RuntimeError("Invalid query timing duration")
            labels, distances = result[:2]
            validate_ids(labels, count, len(queries), k)
            if distances.shape != labels.shape or not np.isfinite(distances).all():
                raise ValueError("Invalid native query distances")
            if repeat == 0 and validate_labels is not None:
                validate_labels(labels)
            if repeat == 0 and on_result is not None:
                on_result(ef, labels, distances)
            current = (array_hash(labels), array_hash(distances))
            if identity is not None and current != identity:
                raise RuntimeError("Read-only query results changed between timing repetitions")
            identity = current
            recall = float(np.any(labels[:, :, None] == truth[:, None, :], axis=1).mean())
            comparisons = float(np.mean(result[2])) if api.endswith("_with_stats") else -1.0
            samples.append({
                "repeat": repeat, "wall_s": elapsed, "thread_cpu_s": thread_s,
                "thread_cpu_wall_ratio": thread_s / elapsed,
                "qps": len(queries) / elapsed, "recall": recall, "cmps": comparisons,
                "labels_sha256": current[0], "distances_sha256": current[1],
            })
        row = [ef, recall, statistics.median(sample["qps"] for sample in samples),
               statistics.median(sample["cmps"] for sample in samples)]
        point = {"ef": ef, "recall": recall, "qps": row[2], "cmps": row[3], "samples": samples}
        rows.append(row)
        points.append(point)
        print(f"ef search: {ef}, recall: {recall:.4f}, QPS: {row[2]:.6f}", flush=True)
        if on_point is not None:
            on_point(point)
        if target_recall is not None and recall >= target_recall:
            break
    print("Final results (ef_search, recall, QPS, cmps):", flush=True)
    for row in rows:
        print(row, flush=True)
    return {
        "rows": rows, "points": points, "query_api": api, "query_count": len(queries),
        "repeats": repeats, "aggregation": "median of all repetitions",
        "target_recall": target_recall,
        "target_reached": target_recall is None or rows[-1][1] >= target_recall,
        "queries_sha256": array_hash(queries), "ground_truth_sha256": array_hash(truth),
    }


def main():
    args = arg_init()
    queries, predicates, truth = load_query_data(
        args.query_path, args.qrange_path, args.gt_path, args.N, args.n_query_to_use, args.K)
    if queries.shape[1] != args.dim:
        raise ValueError("Query dimension differs from the index configuration")
    wrapper = HashANN()
    params = {"M": args.M, "ef_construction": args.efConstruction, "metric": args.metric.lower(),
              "dim": args.dim, "N": args.N, "threads": args.threads}
    kinds = ast.literal_eval(args.attr_type_list)
    wrapper.init_params(params)
    index = wrapper.load_index(params, kinds, args.index_cache_path, args.threads, args.name)
    if index.get_current_count() != args.N:
        raise ValueError("Loaded index count differs from the requested corpus")
    if args.augment_edges.lower() == "true":
        print(index.augment_ft_neighbors(min_same=args.augment_threshold, max_hops=2))
    if args.augment_cht.lower() == "true":
        index.augment_edges_cht(args.augment_cht_efc, args.augment_cht_threads)
    efs = ast.literal_eval(args.ef_search)
    if type(efs) is int:
        efs = [efs]
    result = query_sweep(
        index, queries, predicates, truth, count=args.N, k=args.K, efs=efs,
        ef_top=args.ef_top, use_ft=args.use_ft.lower() == "true",
        routing_min_deg=args.ft_routing_min_deg,
        backfill_tail=args.ft_routing_backfill_tail.lower() == "true",
        repeats=args.query_repeats, target_recall=args.target_recall)
    if args.result_json:
        with Path(args.result_json).open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")


if __name__ == "__main__":
    main()
