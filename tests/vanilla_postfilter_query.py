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


def precompute_attr_indices(attrs, attr_type_list):
    """Vectorize attrs once for fast per-query qualifier building.
    Returns (num_arrays_per_attr, label_bitmaps_per_attr).
    For numerical attr j: num_arrays_per_attr[j] = np.array of N values.
    For categorical attr j: label_bitmaps_per_attr[j] = dict {label: np.bool_ array of N}.
    """
    n = len(attrs)
    num_arrays = [None] * len(attr_type_list)
    label_bitmaps = [None] * len(attr_type_list)
    for j, atype in enumerate(attr_type_list):
        if atype == 0:
            arr = np.empty(n, dtype=np.float64)
            for i in range(n):
                arr[i] = attrs[i][j][0]
            num_arrays[j] = arr
        else:
            bmaps = {}
            for i in range(n):
                for lab in attrs[i][j]:
                    bm = bmaps.get(lab)
                    if bm is None:
                        bm = np.zeros(n, dtype=bool)
                        bmaps[lab] = bm
                    bm[i] = True
            label_bitmaps[j] = bmaps
    return num_arrays, label_bitmaps


def build_qualifier_fast(num_arrays, label_bitmaps, predicate, attr_type_list, n):
    mask = np.ones(n, dtype=bool)
    for j, atype in enumerate(attr_type_list):
        if atype == 0:
            low, high = predicate[j]
            mask &= (num_arrays[j] >= low) & (num_arrays[j] <= high)
        else:
            for lab in predicate[j]:
                bm = label_bitmaps[j].get(lab)
                if bm is None:
                    return set()  # no item has this label
                mask &= bm
    return set(int(x) for x in np.flatnonzero(mask))


def main():
    args = parse_args()
    # Load real vanilla hnswlib (separate .so from hashannlib)
    import sys as _sys
    _sys.path.insert(0, "/home/mocheng/hnswlib")
    import hnswlib
    _sys.path.pop(0)

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
        print(f"All queries share same predicate. Qualifying items: {len(qualifying)} ({len(qualifying)/len(attrs)*100:.2f}%)", flush=True)
        per_query_qualifying = None
    else:
        print("Queries have different predicates, precomputing attr indices...", flush=True)
        t0 = time.time()
        num_arrays, label_bitmaps = precompute_attr_indices(attrs, attr_type_list)
        print(f"  precompute done in {time.time()-t0:.1f}s", flush=True)
        n_items = len(attrs)
        per_query_qualifying = []
        t0 = time.time()
        for i in range(nq):
            per_query_qualifying.append(
                build_qualifier_fast(num_arrays, label_bitmaps, preds[i], attr_type_list, n_items))
        print(f"  built {nq} per-query qualifying sets in {time.time()-t0:.1f}s", flush=True)
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
        p.knn_query(queries[:min(3, nq)], k=min(ef, 100))

        # query: k=ef to get enough candidates for post-filtering
        start = time.time()
        labels_all, dists_all = p.knn_query(queries, k=ef)
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

        print(f"Query time: {elapsed} seconds, QPS:{qps}")
        print(f"ef search: {ef}, recall: {recall:.4f}")
        result.append([ef, recall, qps])

    print("Final results (ef_search, recall, QPS):")
    for res in result:
        print(res)


if __name__ == "__main__":
    main()
