"""
Brute-force groundtruth generator for DNF predicates.
No Milvus dependency — vectorized NumPy computation.

Usage:
  python groundtruth_bruteforce.py \
    --dataset_file sift_base.fvecs \
    --query_file sift_query.fvecs \
    --attr_file attr.json \
    --predicate_file predicate_dnf_*.json \
    --attr_type_list "[0,1]" \
    --N 1000000 --d 128 --query_size 10000 --K 10 \
    --metric L2 \
    --gt_file gt_dnf_*.json
"""

import json
import numpy as np
import argparse
import ast
import time
import sys
sys.path.insert(0, '.')
from utils import fvecs_read, read_multy_attr


def arg_init():
    parser = argparse.ArgumentParser(description="Brute-force GT for DNF predicates")
    parser.add_argument("--dataset_file", type=str, required=True)
    parser.add_argument("--query_file", type=str, required=True)
    parser.add_argument("--attr_file", type=str, required=True)
    parser.add_argument("--predicate_file", type=str, required=True)
    parser.add_argument("--attr_type_list", type=str, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--query_size", type=int, required=True)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--metric", type=str, default="L2", help="L2 or IP")
    parser.add_argument("--gt_file", type=str, required=True)
    return parser.parse_args()


def is_dnf(predicate):
    if not predicate or not predicate[0]:
        return False
    first = predicate[0]
    return isinstance(first[0], list) and len(first[0]) > 0 and isinstance(first[0][0], list)


def preprocess_attrs(attrs, attr_type_list, N):
    """Extract per-attribute arrays for vectorized filtering."""
    processed = []
    for a, atype in enumerate(attr_type_list):
        if atype == 0:
            # numerical: extract scalar array
            arr = np.array([attrs[i][a][0] for i in range(N)], dtype=np.int64)
            processed.append(arr)
        else:
            # categorical: keep as list of sets for fast lookup
            sets = [set(attrs[i][a]) for i in range(N)]
            processed.append(sets)
    return processed


def compute_term_mask(proc_attrs, term, attr_type_list, N):
    """Compute boolean mask for one AND term (vectorized for numerical)."""
    mask = np.ones(N, dtype=bool)
    for a, atype in enumerate(attr_type_list):
        vals = term[a]
        if not vals:
            continue
        if atype == 0:
            arr = proc_attrs[a]
            mask &= (arr >= vals[0]) & (arr <= vals[1])
        else:
            # categorical: must contain all labels
            label_set = set(vals)
            cat_sets = proc_attrs[a]
            # vectorize with numpy fromiter
            cat_mask = np.array([label_set.issubset(cat_sets[i]) for i in range(N)], dtype=bool)
            mask &= cat_mask
    return mask


def compute_query_mask(proc_attrs, pred, attr_type_list, N, dnf_mode):
    """Compute boolean mask for one query predicate."""
    if dnf_mode:
        mask = np.zeros(N, dtype=bool)
        for term in pred:
            mask |= compute_term_mask(proc_attrs, term, attr_type_list, N)
        return mask
    else:
        return compute_term_mask(proc_attrs, pred, attr_type_list, N)


if __name__ == "__main__":
    args = arg_init()
    attr_type_list = ast.literal_eval(args.attr_type_list)
    N = args.N
    K = args.K
    query_size = args.query_size

    print(f"Loading dataset ({N} points)...")
    dataset = fvecs_read(args.dataset_file)
    assert dataset.shape[0] >= N
    dataset = dataset[:N].astype(np.float32)

    print(f"Loading queries ({query_size})...")
    queries = fvecs_read(args.query_file)
    assert queries.shape[0] >= query_size
    queries = queries[:query_size].astype(np.float32)

    print(f"Loading attributes...")
    attrs = read_multy_attr(args.attr_file)
    assert len(attrs) >= N

    print(f"Loading predicates...")
    predicates = read_multy_attr(args.predicate_file)
    assert len(predicates) >= query_size
    predicates = predicates[:query_size]

    dnf_mode = is_dnf(predicates)
    print(f"DNF mode: {dnf_mode}, metric: {args.metric}")

    print(f"Preprocessing attributes...")
    proc_attrs = preprocess_attrs(attrs, attr_type_list, N)

    # Precompute dataset norms for L2
    if args.metric.upper() == "L2":
        dataset_sq = np.sum(dataset ** 2, axis=1)  # (N,)

    print(f"Computing groundtruth...")
    t0 = time.time()
    all_gt = []

    for qi in range(query_size):
        if qi % 200 == 0:
            elapsed = time.time() - t0
            speed = qi / max(elapsed, 0.001)
            eta = (query_size - qi) / max(speed, 0.001)
            print(f"  query {qi}/{query_size}  {speed:.1f} q/s  ETA={eta:.0f}s")

        # Step 1: filter by predicate
        mask = compute_query_mask(proc_attrs, predicates[qi], attr_type_list, N, dnf_mode)
        cand_ids = np.where(mask)[0]

        if len(cand_ids) == 0:
            print(f"  WARNING: query {qi} has 0 matching candidates!")
            all_gt.append([])
            continue

        # Step 2: compute distances (vectorized)
        q = queries[qi]
        if args.metric.upper() == "L2":
            # ||x - q||^2 = ||x||^2 - 2*x·q + ||q||^2
            # We only need relative ordering, so skip ||q||^2
            cand_vecs = dataset[cand_ids]
            dists = dataset_sq[cand_ids] - 2.0 * cand_vecs.dot(q)
            topk_local = np.argpartition(dists, min(K, len(dists)-1))[:K]
            topk_local = topk_local[np.argsort(dists[topk_local])]
        else:  # IP
            cand_vecs = dataset[cand_ids]
            scores = cand_vecs.dot(q)
            topk_local = np.argpartition(-scores, min(K, len(scores)-1))[:K]
            topk_local = topk_local[np.argsort(-scores[topk_local])]

        gt_ids = cand_ids[topk_local].tolist()
        all_gt.append(gt_ids)

    t1 = time.time()
    print(f"\nGroundtruth computed in {t1-t0:.1f}s ({query_size/(t1-t0):.1f} q/s)")

    with open(args.gt_file, 'w') as f:
        json.dump(all_gt, f)
    print(f"Saved to {args.gt_file}")
