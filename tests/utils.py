import numpy as np
import sys
import json
import argparse

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    if sys.byteorder == 'big':
        a.byteswap(inplace=True)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def read_attr(fname):
    with open(fname, 'r') as file:
        data = json.load(file)
    assert(isinstance(data, list))
    return np.array(data, dtype="int64")

def read_multy_attr(fname):
    with open(fname, 'r') as file:
        data = json.load(file)
    assert(isinstance(data, list))
    return data

def load_data(dataset_file, query_file, attr_file, qrange_file, gt_file, N, Nq, k):# fvecs, fvecs, json, json, json
    if(".fvecs" in dataset_file):
        data = fvecs_read(dataset_file)
        print(f"data shape: {data.shape}")
        assert data.shape[0] == N
    else:
        print("error: dataset file format not supported")
        sys.exit(-1)
    if(".fvecs" in query_file):
        queries = fvecs_read(query_file)
        print(f"query shape: {queries.shape}")
        assert queries.shape[0] >= Nq
        if queries.shape[0] > Nq:
            queries = queries[:Nq]
    else:
        print("error: query file format not supported")
        sys.exit(-1)
    if(".json" in attr_file):
        attr = read_multy_attr(attr_file)
        assert len(attr) == N
    else:
        print("error: attribution file format not supported")
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

    return data, queries, attr, query_filter_ranges, query_gt


def arg_init():
    parser = argparse.ArgumentParser(description="Index parameters")
    parser.add_argument("--k", type=int, default=10, help="For kNN search")
    parser.add_argument(
        "--name", type=str, required=True, help="HNSW or NSW"
    )
    parser.add_argument(
        "--n_query_to_use", type=int, default=1000, help="Number of queries to use"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the fvecs data file"
    )
    parser.add_argument(
        "--query_path", type=str, required=True, help="Path to the fvecs query data file"
    )
    parser.add_argument(
        "--attr_path", type=str, required=True, help="Path to the json data attribute file"
    )
    parser.add_argument(
        "--qrange_path", type=str, required=True, help="Path to the json query attribute range file"
    )
    parser.add_argument(
        "--gt_path", type=str, required=True, help="Path to the json ground truth file"
    )
    parser.add_argument("--M", type=int, default=16, help="Number of graph connections")
    parser.add_argument("--ft_bits", type=int, default=128, help="Number of bits per filter table/ number of bytes per countering hash table")
    parser.add_argument(
        "--efConstruction", type=int, default=500, help="Parameter for HNSW index"
    )
    parser.add_argument("--attr_type_list", type=str, required=True, help="List of attribute types, 0 for numerical, 1 for categorical")
    parser.add_argument(
        "--index_cache_path",
        type=str,
        required=True,
        help="Directory to load and save the index",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="l2",
        help="l2 or cosine",
    )
    # parser.add_argument(
    #     "--ef_list",
    #     type=int,
    #     nargs="+",
    #     default=list(range(10, 2000, 20)),
    #     help="List of EF values",
    # )
    parser.add_argument(
        "--N",
        type=int,
        default=0,
        help="Data size",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        required=True,
        help="Data size",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=1,
        required=True,
        help="Dimension",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=10,
        help="return knn",
    )
    parser.add_argument(
        "--attr_type",
        type=list,
        default=[0],
        help="attribute type list, 0 for numerical, 1 for categorical",
    )
    parser.add_argument(
        "--ef_search",
        type=str,
        default="[100]",
        help="ef search list",
    )
    parser.add_argument(
        "--ef_top",
        type=int,
        default=10,
        help="ef top",
    )
    parser.add_argument(
        "--use_ft", type=str, required=False, default="true", help="Use filter table"
    )


    args = parser.parse_args()
    return args