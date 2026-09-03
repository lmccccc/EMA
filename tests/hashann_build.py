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
import ast


from hashann import HashANN



if __name__ == "__main__":
    args = arg_init()

    hash_ann = HashANN()

    # data_path query_path attr_path qrange_path gt_path N n_query_to_use k
    nq = args.n_query_to_use
    attr_type_list = ast.literal_eval(args.attr_type_list)
    params = {"M": args.M, "ef_construction": args.efConstruction, "metric": args.metric.lower(), "dim": args.dim, "N": args.N, "max_elements": args.max_elements if args.max_elements else args.N, "ef_search_list": args.ef_search, "ft_bits": args.ft_bits, "edge_level_ft": args.edge_level_ft}
    data, queries, attr, query_filter_ranges, query_gt = load_data(args.data_path, args.query_path, args.attr_path, args.qrange_path, args.gt_path, args.N, nq, args.K)
    hash_ann.init_params(params)
    # print("thread: ", args.threads)
    index = hash_ann.build_index(
        params, data, attr, attr_type_list, args.index_cache_path, args.threads, args.name
    )

    print("build done")
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
