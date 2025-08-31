import argparse
import os
import time
from typing import List

import numpy as np
import pandas as pd
import hashannlib
import bisect
import sys
import json
from utils import *
import faiss
import ast


from hashann import HashANN

def load_query_data(query_file, qrange_file, gt_file, N, Nq, k):# fvecs, fvecs, json, json, json

    if(".fvecs" in query_file):
        queries = fvecs_read(query_file)
        print(f"query shape: {queries.shape}")
        assert queries.shape[0] == Nq
    else:
        print("error: query file format not supported")
        sys.exit(-1)
    if(".json" in qrange_file):
        query_filter_ranges = read_attr(qrange_file)
        #convert array into turple list
        query_filter_ranges = [(query_filter_ranges[i], query_filter_ranges[i+1]) for i in range(0, len(query_filter_ranges), 2)]
        assert len(query_filter_ranges) == Nq
    else:    
        print("error: query range file format not supported")
        sys.exit(-1)
    if(".json" in gt_file):
        query_gt = read_attr(gt_file)
        query_gt = query_gt.reshape(-1, k)
        assert len(query_gt) == Nq
    else:
        print("error: groundtruth file format not supported")
        sys.exit(-1)
    print("sorting for label")

    return queries, query_filter_ranges, query_gt

if __name__ == "__main__":
    args = arg_init()

    hash_ann = HashANN()

    # data_path query_path attr_path qrange_path gt_path N n_query_to_use k
    nq = args.n_query_to_use
    attr_type_list = ast.literal_eval(args.attr_type_list)
    params = {"M": args.M, "ef_construction": args.efConstruction, "metric": args.metric, "dim": args.dim, "N": args.N, "ef_search_list": args.ef_list}
    queries, query_filter_ranges, query_gt = load_query_data(args.query_path, args.qrange_path, args.gt_path, args.N, nq, args.k)
    hash_ann.init_params(params)
    
    index = hash_ann.load_index(
        params, attr_type_list, args.index_cache_path, args.threads, args.name
    )

    # query
    index.set_num_threads(1)
    index.set_ef(args.ef_list[0])  # Set the first ef for the query
    start = time.time()
    labels, distances = index.knn_query(queries, k=args.K)
    end = time.time()
    print(f"Query time: {end - start} seconds, QPS:{queries.shape[0]/(end-start)}")

    # recall
    correct_sum = 0
    for i in range(queries.shape[0]):
        gt = query_gt[i]
        label = labels[i]
        if len(gt) != len(label):
            print(f"Error: ground truth and label length mismatch at query {i}")
            continue
        correct = np.isin(gt, label)
        correct_sum += np.sum(correct)
    recall = correct_sum / (nq * args.K)
    print(f"ef search: {args.ef_list[0]}, recall: {recall:.4f}")
    exit()

    # results = bench_hybrid_query(
    #     index,
    #     args.plan,
    #     args.optimizer_conf_dir,
    #     args.k,
    #     args.ef_list,
    #     args.al_list,
    #     args.low_range,
    #     args.high_range,
    #     queries[:nq],
    #     query_filter_ranges[:nq],
    #     query_gt[:nq],
    # )
    # print(results)
