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
import faiss
import ast


from hashann import HashANN
from hashann_query import is_dnf_predicate

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
    hash_ann.init_params(params)

    
    index = hash_ann.load_index(
        params, attr_type_list, args.index_cache_path, args.threads, args.name
    )
    # index.save_index(args.index_cache_path)
    print("start query")
    # query
    efs_list = ast.literal_eval(args.ef_search)
    if not isinstance(efs_list, List):
        efs_list = [efs_list]
    index.set_num_threads(1)

    result = []
    # efs_list = [efs_list[-1]]
    for efs in efs_list:
        index.set_ef(efs)   # Set the first ef for the query
        index.set_ef_top(args.ef_top)
        index.set_ft_flag(args.use_ft.lower() == 'true')
        print("set marker flag:", args.use_ft.lower() == 'true')
        index.set_ft_routing_flag(False)
        index.set_ft_routing_min_deg(16)
        print("index ef_search:", efs, " ef_top:", args.ef_top)
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

        # warm up
        if dnf_mode:
            _, _ = index.hybrid_knn_query_dnf(
                _queries[:min(3, len(_queries))],
                _raw_predicate[:min(3, len(_raw_predicate))], k=args.K)
        else:
            _, _ = index.hybrid_knn_query(
                _queries[:min(3, len(_queries))],
                _raw_predicate[:min(3, len(_raw_predicate))], k=args.K)

        start = time.time()
        if dnf_mode:
            ids, distances = index.hybrid_knn_query_dnf(
                _queries, _raw_predicate, k=args.K)
        else:
            ids, distances = index.hybrid_knn_query(_queries, _raw_predicate, k=args.K)
        end = time.time()
        qps = len(target_id)/(end-start)
        print(f"Query time: {end - start} seconds, QPS:{qps}")
        
        # recall
        correct_sum = 0
        recall_list = []
        for i in range(len(target_id)):
            # print("predicate:", raw_predicate[i])
            # print("result:", ids[i])
            # print("ground truth:", _query_gt[i])
            gt = _query_gt[i]
            res = ids[i]
            if len(gt) != len(res):
                print(f"Error: ground truth and label length mismatch at query {i}, gt: {len(gt)}, label: {len(res)}")
                exit(1)
            correct = np.isin(gt, res)
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
        result.append([efs, recall, qps])
        # if recall >= 0.97:
        #     break
    
    print("Final results (ef_search, recall, QPS):")
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
