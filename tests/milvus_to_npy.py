
import sys
from pymilvus import DataType, MilvusClient, connections, utility
import sys
import numpy as np
import math
import time
import argparse
import json
import ast
from utils import *
import os

def read_data(data_file, attr_file, N, d, query_file, predicate_file, Nq):
    dataset = fvecs_read(data_file)
    assert(dataset.shape[0] == N)
    assert(dataset.shape[1] == d)

    attr = read_multy_attr(attr_file)
    assert(len(attr) == N)

    query = fvecs_read(query_file)
    # assert(query.shape[0] == Nq)
    assert(query.shape[0] >= Nq)
    if query.shape[0] > Nq:
        query = query[:Nq]
    assert(query.shape[1] == d)

    predicate = read_multy_attr(predicate_file)
    assert(len(predicate) >= Nq)
    if len(predicate) > Nq:
        predicate = predicate[:Nq]

    return dataset, attr, query, predicate

def arg_init():
    parser = argparse.ArgumentParser(description="Index parameters")
    parser.add_argument("--dataset_file", type=str, help="Dataset file path", required=True)
    parser.add_argument("--d", type=int, help="Dimension of data", required=True)
    parser.add_argument("--attr_file", type=str, help="Output attribute file path", required=True)
    parser.add_argument("--attr_type_list", type=str, required=True, help="List of attribute types, 0 for numerical, 1 for categorical")
    parser.add_argument("--N", type=int, required=True, help="Number of data points")
    parser.add_argument("--query_size", type=int, default=100, help="Number of queries", required=True)
    parser.add_argument("--predicate_file", type=str, help="Output predicate file path", required=True)
    parser.add_argument("--query_file", type=str, help="Query file path", required=True)
    parser.add_argument("--c_name", type=str, help="Collection name", required=False)
    parser.add_argument("--mode", type=str, default="query", help="Dimension of data")
    parser.add_argument("--max_cate_val", type=int, default=5, help="Max cardinality for categorical attributes")
    parser.add_argument("--K", type=int, default=10, help="Top K")
    parser.add_argument("--gt_file", type=str, help="Output groundtruth file", required=True)
    parser.add_argument("--metric", type=str, default="L2", help="Distance metric, L2 or IP")
    args = parser.parse_args()
    return args


# read args
if __name__ == "__main__":
    args = arg_init()

    attr_type_list = ast.literal_eval(args.attr_type_list)

    fields = ["id", "vector"] + [f"attr_{i}" for i in range(len(attr_type_list))]
    print("mode:", args.mode, " collection name:", args.c_name)

    # attr check, must not 1
    for t in attr_type_list:
        assert t == 0
    
    file_partition = 5
    partition_size = args.N // file_partition
    partition_start = [i * partition_size for i in range(file_partition)]

    dataset_root = os.path.dirname(args.dataset_file)
    dataset_root = os.path.join(dataset_root, "bulk")

    all_files = [[] for _ in range(file_partition)] # 5 partitions
    # make dir
    if not os.path.exists(dataset_root):
        os.makedirs(dataset_root)
    for i in range(file_partition):
        partition_dir = os.path.join(dataset_root, f"part_{i}")
        if not os.path.exists(partition_dir):
            os.makedirs(partition_dir)
        all_files[i].append(os.path.join(partition_dir, f"vector.npy"))
        all_files[i].append(os.path.join(partition_dir, f"id.npy"))
        for j in range(len(attr_type_list)):
            all_files[i].append(os.path.join(partition_dir, f"attr_{j}.npy"))

    attr_root = os.path.dirname(args.attr_file)
    attr_paths = []
    for i in range(len(attr_type_list)):
        assert attr_type_list[i] == 0
        bulk_attr_path = os.path.join(attr_root, f"attr_{i}.npy")
        attr_paths.append(bulk_attr_path)

    size_per_partition = args.N // file_partition
    begin = [i * size_per_partition for i in range(file_partition)]
    _end = begin[1:] + [args.N]

    dataset = fvecs_read(args.dataset_file)
    ids = np.arange(args.N).astype(np.int64)
    attr = read_multy_attr(args.attr_file)
    split_attr = []
    for i in range(len(attr_type_list)):
            assert len(attr) == args.N
            attr_i = np.array([attr[j][i][0] for j in range(len(attr))]).astype(np.int64)
            split_attr.append(attr_i)

    for i in range(file_partition):
        start = begin[i]
        end = _end[i]
        for file in all_files[i]:
            if "vector" in file:
                print(f"generating bulk vector file {file}")
                data_part = dataset[start:end]
                np.save(file, data_part)
            elif "id" in file:
                print(f"generating bulk id file {file}")
                id_part = ids[start:end]
                np.save(file, id_part)
            elif "attr_0" in file:
                print (f"generating bulk attr_0 file {file}")
                attr_part = split_attr[0][start:end]
                np.save(file, attr_part)
            elif "attr_1" in file:
                print (f"generating bulk attr_1 file {file}")
                attr_part = split_attr[1][start:end]
                np.save(file, attr_part)
            else:
                print("unknown file type:", file)
        