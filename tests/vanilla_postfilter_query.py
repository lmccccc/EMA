"""Vanilla HNSW post-filter query benchmark.
Usage: called from exp3/vanilla_query.sh with the same conf.sh variables.
"""
import argparse
import time
import json
import ast
import numpy as np
import sys
from utils import fvecs_read, read_attr, read_multy_attr


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--index_path", required=True)
    p.add_argument("--query_path", required=True)
    p.add_argument("--attr_path", required=True)
    p.add_argument("--qrange_path", required=True)
    p.add_argument("--gt_path", required=True)
    p.add_argument("--ef_search", required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--n_query", type=int, default=1000)
    p.add_argument("--dim", type=int, required=True)
    p.add_argument("--metric", default="IP")
    p.add_argument("--attr_type_list", default="[0,1]")
    return p.parse_args()


def build_qualifier(attrs, predicate_sample, attr_type_list):
    """Build a set of qualifying item IDs from attrs and a sample predicate."""
    # attr_type_list: [0,1] means [numerical, categorical]
    # predicate_sample: [[low, high], [label]] for [0,1]
    n = len(attrs)
    qualifying = set()
    for i in range(n):
        match = True
        for j, atype in enumerate(attr_type_list):
            if atype == 0:  # numerical range
                low, high = predicate_sample[j]
                val = attrs[i][j][0]
                if val < low or val > high:
                    match = False
                    break
            elif atype == 1:  # categorical membership
                required = set(predicate_sample[j])
                item_labels = set(attrs[i][j])
                if not required.issubset(item_labels):
                    match = False
                    break
        if match:
            qualifying.add(i)
    return qualifying


def main():
    args = parse_args()
    import hashannlib as hnswlib

    efs_list = ast.literal_eval(args.ef_search)
    if not isinstance(efs_list, list):
        efs_list = [efs_list]
    attr_type_list = ast.literal_eval(args.attr_type_list)

    # Load queries
    queries = fvecs_read(args.query_path)
    nq = min(args.n_query, queries.shape[0])
    queries = queries[:nq]

    # Load GT
    gt = read_attr(args.gt_path).reshape(-1, args.k)[:nq]

    # Load predicates
    preds = read_multy_attr(args.qrange_path)[:nq]

    # Load attrs and build qualifying set per query
    print("Loading attrs for post-filter...", flush=True)
    with open(args.attr_path) as f:
        attrs = json.load(f)
    print(f"Attrs loaded: {len(attrs)} items", flush=True)

    # Check if all queries share the same predicate (common case)
    all_same = all(preds[i] == preds[0] for i in range(1, len(preds)))
    if all_same:
        qualifying = build_qualifier(attrs, preds[0], attr_type_list)
        print(f"All queries share same predicate. Qualifying items: {len(qualifying)} ({len(qualifying)/len(attrs)*100:.2f}%)")
        per_query_qualifying = None
    else:
        print("Queries have different predicates, building per-query qualifying sets...")
        per_query_qualifying = []
        for i in range(nq):
            per_query_qualifying.append(build_qualifier(attrs, preds[i], attr_type_list))
        qualifying = None

    del attrs  # free memory

    # Load index
    space = 'ip' if args.metric.upper() == 'IP' else 'l2'
    p = hnswlib.Index(space=space, dim=args.dim)
    p.load_index(args.index_path, max_elements=0)
    p.set_num_threads(1)
    print(f"Index loaded: {p.get_current_count()} elements", flush=True)

    print("\nstart query")
    result = []
    for ef in efs_list:
        p.set_ef(ef)

        # warm up
        p.knn_query_with_stats(queries[:min(3, nq)], k=min(ef, 100))

        # query: k=ef to get enough candidates for post-filtering
        start = time.time()
        labels_all, dists_all, cmps, hops = p.knn_query_with_stats(queries, k=ef)
        elapsed = time.time() - start
        qps = nq / elapsed

        # post-filter
        correct_sum = 0
        for i in range(nq):
            q_set = qualifying if qualifying is not None else per_query_qualifying[i]
            filtered = [int(l) for l in labels_all[i] if int(l) in q_set][:args.k]
            gt_set = set(int(x) for x in gt[i])
            correct_sum += len(set(filtered) & gt_set)

        recall = correct_sum / (nq * args.k)
        avg_cmps = float(np.mean(cmps))
        avg_hops = float(np.mean(hops))

        print(f"Query time: {elapsed} seconds, QPS:{qps}, cmps:{avg_cmps}")
        print(f"ef search: {ef}, recall: {recall:.4f}")
        result.append([ef, recall, qps, avg_cmps])

    print("Final results (ef_search, recall, QPS, cmps):")
    for res in result:
        print(res)


if __name__ == "__main__":
    main()
