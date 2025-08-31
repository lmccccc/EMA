import argparse
import os
import time
from typing import List

import numpy as np
import pandas as pd
import hnswlib
import bisect
import sys
import json
from utils import *


def build_or_load_index(name: str, params, base_scalars, attr, index_save_path, threads: int):
    index = None

    if name == "HNSW":
        print("building HNSW index")
        index = hnswlib.Index(space=params["metric"], dim=params["dim"])
    else:
        print("building NSW index")
        index = hnswlib.NSWIndex(space=params["metric"], dim=params["dim"])
    # else:
    #     index = FaissIvfPq(metric="euclidean", method_param=params)
    index.init_index(max_elements=params["N"], ef_construction=params["ef_construction"], M=params["M"])
    index.set_num_threads(threads)
    if os.path.exists(index_save_path):
    # if False:
        print(f"Reading index from {index_save_path} ...")
        start = time.time()
        index.load_index(index_save_path)
        end = time.time()
        print(f"Index loaded: {name}, duration: {end-start}.")
    else:
        print(f"Building index: {name}...")
        start = time.time()
        index.add_items(base_scalars)
        end = time.time()
        index.save_index(index_save_path)

        print(f"Index built: {name}, duration: {end-start}.")

    return index, end - start



if __name__ == "__main__":
    args = arg_init()

    # data_path query_path attr_path qrange_path gt_path N n_query_to_use k
    nq = args.n_query_to_use
    params = {"M": args.M, "ef_construction": args.efConstruction, "metric": args.metric, "dim": args.dim, "N": args.N, "ef_search_list": args.ef_list}
    data, queries, attr, query_filter_ranges, query_gt = load_data(args.data_path, args.query_path, args.attr_path, args.qrange_path, args.gt_path, args.N, nq, args.k)
    index, _ = build_or_load_index(
        args.name, params, data, attr, args.index_cache_path, args.threads
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

