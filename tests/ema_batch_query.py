#!/usr/bin/env python3
"""
ema_batch_query.py — Batch HashANN query for multiple selectivities on one dataset.

Loads the index and builds augmented edges ONCE, then iterates over selectivities.
Results written to --log_file in the same format as extract_results.py.
"""
import argparse
import ast
import fcntl
import sys
import time

import numpy as np

sys.path.insert(0, sys.path[0])  # ensure tests/ is on path
from hashann import HashANN
from utils import fvecs_read, read_multy_attr, read_attr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--index_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--query_path", required=True)
    parser.add_argument("--attr_path", required=True)
    parser.add_argument("--label_root", required=True)
    parser.add_argument("--attr_type", required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--ef_construction", type=int, required=True)
    parser.add_argument("--ef_list", required=True)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--query_size", type=int, default=1000)
    parser.add_argument("--use_ft", default="true")
    parser.add_argument("--augment_edges", default="true")
    parser.add_argument("--augment_threshold", type=int, default=8)
    parser.add_argument("--sel_names", required=True, help="Comma-separated: 10%,20%,...")
    parser.add_argument("--sel_vals", required=True, help="Comma-separated: [0.333,7],[0.4,5],...")
    parser.add_argument("--log_file", required=True)
    args = parser.parse_args()

    attr_type_list = ast.literal_eval(args.attr_type)
    efs_list = ast.literal_eval(args.ef_list)
    if not isinstance(efs_list, list):
        efs_list = [efs_list]
    use_ft = args.use_ft.lower() == "true"

    # Parse selectivities
    sel_names = args.sel_names.split(",")
    # sel_vals are like [0.333,7],[0.4,5],... — need careful parsing
    raw_sel = args.sel_vals
    sel_vals = []
    depth = 0
    current = ""
    for ch in raw_sel:
        if ch == "[":
            depth += 1
            current += ch
        elif ch == "]":
            depth -= 1
            current += ch
            if depth == 0:
                sel_vals.append(current)
                current = ""
        elif ch == "," and depth == 0:
            continue  # skip comma between brackets
        else:
            current += ch

    assert len(sel_names) == len(sel_vals), \
        f"sel_names ({len(sel_names)}) != sel_vals ({len(sel_vals)})"

    # Load queries once
    queries_all = fvecs_read(args.query_path)
    nq = min(args.query_size, queries_all.shape[0])
    queries = queries_all[:nq]
    print(f"Queries loaded: {queries.shape}")

    # Load index once
    hash_ann = HashANN()
    params = {
        "M": args.M, "ef_construction": args.ef_construction,
        "metric": args.metric.lower(), "dim": args.dim,
        "N": args.N, "ef_search": 10, "ef_top": 1,
    }
    hash_ann.init_params(params)
    index = hash_ann.load_index(params, attr_type_list, args.index_path, 1, "HNSW")

    # Build augmented edges once
    if args.augment_edges.lower() == "true":
        print(f"Building augmented edges (threshold={args.augment_threshold})...")
        t0 = time.time()
        aug_result = index.augment_ft_neighbors(
            min_same=args.augment_threshold, max_hops=2
        )
        print(f"Augment done in {time.time()-t0:.1f}s: {aug_result}")

    index.set_num_threads(1)
    has_stats_api = hasattr(index, "hybrid_knn_query_with_stats")

    # Iterate over selectivities
    for sel_name, sel_val in zip(sel_names, sel_vals):
        print(f"\n{'='*60}")
        print(f"Dataset: {args.dataset} | Selectivity: {sel_name} ({sel_val})")
        print(f"{'='*60}")

        # Resolve predicate/gt paths
        pred_path = f"{args.label_root}/predicate_arbi_0_1_{sel_val}.json"
        gt_path = f"{args.label_root}/gt_arbi_0_1_{sel_val}.json"

        try:
            raw_pred = read_multy_attr(pred_path)[:nq]
            query_gt = np.array(read_attr(gt_path)).reshape(-1, args.K)[:nq]
        except Exception as e:
            print(f"  ERROR loading predicate/gt: {e}")
            continue

        result_rows = []
        for ef in efs_list:
            index.set_ef(ef)
            index.set_ef_top(1)
            index.set_ft_flag(use_ft)
            index.set_ft_routing_flag(True)
            index.set_ft_routing_min_deg(16)

            # Warmup
            if has_stats_api:
                index.hybrid_knn_query_with_stats(queries[:3], raw_pred[:3], k=args.K)
            else:
                index.hybrid_knn_query(queries[:3], raw_pred[:3], k=args.K)

            # Timed run
            start = time.time()
            if has_stats_api:
                ids, dists, dcounts, hops = index.hybrid_knn_query_with_stats(
                    queries, raw_pred, k=args.K
                )
                cmps = float(np.mean(dcounts))
            else:
                ids, dists = index.hybrid_knn_query(queries, raw_pred, k=args.K)
                cmps = -1.0
            elapsed = time.time() - start
            qps = nq / elapsed

            # Recall
            correct = 0
            for i in range(nq):
                res = ids[i]
                res_valid = res[res >= 0] if hasattr(res, "__len__") else res
                correct += np.isin(query_gt[i], res_valid).sum()
            recall = correct / (nq * args.K)

            print(f"  ef={ef:4d}  recall={recall:.4f}  QPS={qps:.0f}  cmps={cmps:.0f}")
            result_rows.append((ef, recall, qps, cmps))

        # Write results to log file
        with open(args.log_file, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(f"dataset: {args.dataset}\n")
                f.write(f"attr: {args.attr_type}\n")
                f.write(f"sel: {sel_val}\n")
                f.write(f"M: {args.M}\n")
                f.write(f"K: {args.K}\n")
                f.write(f"algo: hashann_ema\n")
                for ef, recall, qps, cmps in result_rows:
                    f.write(f"[{ef}, {recall:.6f}, {qps:.2f}, {cmps:.2f}]\n")
                f.write("\n")
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

        print(f"  Results appended to {args.log_file}")

    # Print final summary
    print("\n" + "=" * 60)
    print("Final results (ef, recall, QPS, cmps):")
    print("=" * 60)


if __name__ == "__main__":
    main()
