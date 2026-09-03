import argparse
import os
import time
from typing import List

import numpy as np
import pandas as pd
import bisect
import sys
import json
from utils import *
import ast


from hashann import HashANN

def is_dnf_predicate(raw_predicate):
    """Detect DNF format: predicate[query][term][attr] = values (3 levels of nesting)
    vs legacy format: predicate[query][attr] = values (2 levels)."""
    if not raw_predicate or not raw_predicate[0]:
        return False
    first_query = raw_predicate[0]
    # DNF: first_query[0] is a term (list of per-attr values)
    # Legacy: first_query[0] is attr values (list of ints)
    # Distinguish: in DNF, first_query[0][0] is a list; in legacy, first_query[0][0] is an int
    if isinstance(first_query[0], list) and len(first_query[0]) > 0 and isinstance(first_query[0][0], list):
        return True
    return False


def load_query_data(query_file, qrange_file, gt_file, N, Nq, k):# fvecs, fvecs, json, json, json

    if(".fvecs" in query_file):
        queries = fvecs_read(query_file)
        print(f"query shape: {queries.shape}")
        assert queries.shape[0] >= Nq
        if queries.shape[0] > Nq:
            queries = queries[:Nq]
    else:
        print("error: query file format not supported")
        sys.exit(-1)
    if(".json" in qrange_file):
        query_filter_ranges = read_multy_attr(qrange_file)
        #convert array into turple list
        # query_filter_ranges = [(query_filter_ranges[i], query_filter_ranges[i+1]) for i in range(0, len(query_filter_ranges), 2)]
        assert len(query_filter_ranges) >= Nq
        if len(query_filter_ranges) > Nq:
            query_filter_ranges = query_filter_ranges[:Nq]
    else:    
        print("error: query range file format not supported")
        sys.exit(-1)
    if(".json" in gt_file):
        query_gt = read_attr(gt_file)
        query_gt = query_gt.reshape(-1, k)
        assert len(query_gt) >= Nq
        if len(query_gt) > Nq:
            query_gt = query_gt[:Nq]

    else:
        print("error: groundtruth file format not supported")
        sys.exit(-1)
    print("sorting for label")

    return queries, query_filter_ranges, query_gt

if __name__ == "__main__":
    args = arg_init()
    print("K:", args.K)
    hash_ann = HashANN()

    # data_path query_path attr_path qrange_path gt_path N n_query_to_use k
    nq = args.n_query_to_use
    attr_type_list = ast.literal_eval(args.attr_type_list)
    params = {"M": args.M, "ef_construction": args.efConstruction, "metric": args.metric.lower(), "dim": args.dim, "N": args.N, "ef_search": args.ef_search, "ef_top": args.ef_top}
    queries, raw_predicate, query_gt = load_query_data(args.query_path, args.qrange_path, args.gt_path, args.N, nq, args.K)
    dnf_mode = is_dnf_predicate(raw_predicate)
    if dnf_mode:
        print("DNF predicate detected (OR support enabled)")
    else:
        print("Legacy AND-only predicate detected")
    hash_ann.init_params(params)

    
    index = hash_ann.load_index(
        params, attr_type_list, args.index_cache_path, args.threads, args.name
    )

    # Adaptive augmented edges
    if args.augment_edges.lower() == 'true':
        print(f"Building augmented edges (threshold={args.augment_threshold})...")
        aug_result = index.augment_ft_neighbors(
            min_same=args.augment_threshold, max_hops=2
        )
        print(f"Augment result: {aug_result}")

    # CHT-based edge augmentation
    if args.augment_cht.lower() == 'true':
        print(f"Building CHT augmented edges (efc={args.augment_cht_efc}, threads={args.augment_cht_threads})...")
        t0 = time.time()
        index.augment_edges_cht(args.augment_cht_efc, args.augment_cht_threads)
        print(f"CHT augment done in {time.time()-t0:.1f}s")

    # index.save_index(args.index_cache_path)
    print("start query")
    # query
    efs_list = ast.literal_eval(args.ef_search)
    if not isinstance(efs_list, List):
        efs_list = [efs_list]
    index.set_num_threads(1)
    has_stats_api = hasattr(index, "hybrid_knn_query_with_stats")

    result = []
    # efs_list = [efs_list[-1]]
    for efs in efs_list:
        index.set_ef(efs)   # Set the first ef for the query
        index.set_ef_top(args.ef_top)
        index.set_ft_flag(args.use_ft.lower() == 'true')
        print("set marker flag:", args.use_ft.lower() == 'true')
        index.set_ft_routing_flag(True)
        index.set_ft_routing_min_deg(args.ft_routing_min_deg)
        index.set_ft_routing_backfill_tail(args.ft_routing_backfill_tail.lower() == 'true')
        print("index ef_search:", efs, " ef_top:", args.ef_top, " ft_routing_min_deg:", args.ft_routing_min_deg,
              " backfill_tail:", args.ft_routing_backfill_tail.lower() == 'true')
        # flatten_predicate = index.predicateTranslate(raw_predicate)
        # print("predicate translated")
        # ft_predicate = index.predicateToFT(raw_predicate)
        # print("ft predicate translated")


        # for test
        # for i in range(queries.shape[0]):
        # target_id = [i for i in range(queries.shape[0])]
        # target_id = [i for i in range(min(100, queries.shape[0]))]
        target_id = [i for i in range(queries.shape[0])]
        # target_id = [1]
        print("query size:", len(target_id))
        # print(f"Query: {target_id}")
        _queries = queries[target_id]
        # _flatten_predicate = [flatten_predicate[i] for i in target_id]
        _raw_predicate = [raw_predicate[i] for i in target_id]
        # _ft_predicate = [ft_predicate[i] for i in target_id]
        _query_gt = [query_gt[i] for i in target_id]
        # print("query predicate:", raw_predicate[target_id])

        # warm up (best-effort; failure is OK, we'll handle in main query loop)
        try:
            if dnf_mode:
                _, _ = index.hybrid_knn_query_dnf(
                    _queries[:min(3, len(_queries))],
                    _raw_predicate[:min(3, len(_raw_predicate))], k=args.K)
            elif has_stats_api:
                index.hybrid_knn_query_with_stats(
                    _queries[:min(3, len(_queries))],
                    _raw_predicate[:min(3, len(_raw_predicate))], k=args.K)
            else:
                _, _ = index.hybrid_knn_query(
                    _queries[:min(3, len(_queries))],
                    _raw_predicate[:min(3, len(_raw_predicate))], k=args.K)
        except RuntimeError as e:
            print(f"[warn] warmup at ef={efs} failed ({e}); continuing to per-query fallback")

        SENTINEL = np.iinfo(np.uint64).max
        def _per_query_dnf(qs, ps):
            out_ids = np.full((len(qs), args.K), SENTINEL, dtype=np.uint64)
            out_d = np.zeros((len(qs), args.K), dtype=np.float32)
            for i in range(len(qs)):
                try:
                    sub_ids, sub_d = index.hybrid_knn_query_dnf(qs[i:i+1], ps[i:i+1], k=args.K)
                    out_ids[i, :sub_ids.shape[1]] = sub_ids[0]
                    out_d[i, :sub_d.shape[1]] = sub_d[0]
                except RuntimeError:
                    pass  # leave row as sentinel
            return out_ids, out_d
        def _per_query_legacy(qs, ps):
            out_ids = np.full((len(qs), args.K), SENTINEL, dtype=np.uint64)
            out_d = np.zeros((len(qs), args.K), dtype=np.float32)
            for i in range(len(qs)):
                try:
                    sub_ids, sub_d = index.hybrid_knn_query(qs[i:i+1], ps[i:i+1], k=args.K)
                    out_ids[i, :sub_ids.shape[1]] = sub_ids[0]
                    out_d[i, :sub_d.shape[1]] = sub_d[0]
                except RuntimeError:
                    pass
            return out_ids, out_d

        start = time.time()
        try:
            if dnf_mode:
                ids, distances = index.hybrid_knn_query_dnf(
                    _queries, _raw_predicate, k=args.K)
                cmps = -1.0
            elif has_stats_api:
                ids, distances, distance_counts, hops = index.hybrid_knn_query_with_stats(
                    _queries, _raw_predicate, k=args.K
                )
                cmps = float(np.mean(distance_counts))
            else:
                ids, distances = index.hybrid_knn_query(_queries, _raw_predicate, k=args.K)
                cmps = -1.0
        except RuntimeError as e:
            print(f"[warn] batch ef={efs} threw ({e}); falling back to per-query mode")
            if dnf_mode:
                ids, distances = _per_query_dnf(_queries, _raw_predicate)
            else:
                ids, distances = _per_query_legacy(_queries, _raw_predicate)
            cmps = -1.0
        end = time.time()
        qps = len(target_id)/(end-start)
        if cmps >= 0:
            print(f"Query time: {end - start} seconds, QPS:{qps}, cmps:{cmps}")
        else:
            print(f"Query time: {end - start} seconds, QPS:{qps}, cmps:N/A")
        
        # recall
        correct_sum = 0
        recall_list = []
        for i in range(len(target_id)):
            gt = _query_gt[i]
            res = ids[i]
            # Filter sentinel padding (UINT64_MAX from per-query fallback) and ids out of range
            if hasattr(res, '__len__'):
                res_valid = res[(res != np.iinfo(np.uint64).max) & (res < args.N)]
            else:
                res_valid = res
            correct = np.isin(gt, res_valid)
            correct_sum += np.sum(correct)
            recall_list.append(np.sum(correct)/len(gt))


            # if (np.sum(correct)/len(gt) < 0.5):
            #     print("query id:", target_id[i], "recall:", np.sum(correct)/len(gt))
            #     attr = read_multy_attr(args.attr_path)
            #     print(f"Query {i} not full recall:")
            #     print("predicate:", _raw_predicate[i])
            #     print("gt id:", gt)
            #     print("gt attr:", [attr[gt_id] for gt_id in gt])
            #     print("result id:", res)
            #     print("result attr:", [attr[res_id] for res_id in res])

            #     exit()
            # print("query id:", target_id[i], "recall:", np.sum(correct)/len(gt))

            # print(f"Query {i}:")
            # print("recall:", np.sum(correct)/len(gt))
            # # print example
            # print("predicate:", _raw_predicate[i])
            # print("gt id:", gt)
            # gt_attr = []
            # for gt_id in gt:
            #     gt_attr.append(attr[gt_id])
            # print("gt attr:", gt_attr)
            # print("result id:", res)
            # result_attr = []
            # for res_id in res:
            #     result_attr.append(attr[res_id])
            # print("result attr:", result_attr)
            # print("result distance:", distances[0])

        # print("recall list:", recall_list)  

        recall = correct_sum / (len(target_id) * args.K)
        print(f"ef search: {efs}, recall: {recall:.4f}")
        result.append([int(efs), float(recall), float(qps), float(cmps)])
        # if recall >= 0.97:
        #     break
    
    print("Final results (ef_search, recall, QPS, cmps):")
    for res in result:
        print(res)
    exit()

    # results = bench_hybrid_query(
    #     index,
    #     args.plan,
    #     args.optimizer_conf_dir,
    #     args.K,
    #     args.ef_list,
    #     args.al_list,
    #     args.low_range,
    #     args.high_range,
    #     queries[:nq],
    #     query_filter_ranges[:nq],
    #     query_gt[:nq],
    # )
    # print(results)
