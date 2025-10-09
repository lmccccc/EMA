import json
import os
import numpy as np
import random
import argparse
import ast
from utils import *


def arg_init():
    parser = argparse.ArgumentParser(description="Index parameters")
    parser.add_argument("--attr_file", type=str, help="Output attribute file path", required=True)
    parser.add_argument("--attr_type_list", type=str, required=True, help="List of attribute types, 0 for numerical, 1 for categorical")
    parser.add_argument("--N", type=int, required=True, help="Number of data points")
    parser.add_argument("--numerical_max_attr", type=int, default=100000, help="Max value for numerical attributes")
    parser.add_argument("--categorical_attr_max_cardinality", type=int, default=5, help="Max cardinality for each categorical attribute")
    parser.add_argument("--query_sel", type=str, default="[0.5,0.5]", help="selectivity for each attribute. e.g., [0.1,[1,2]], 0.1 for numerical sel, [1,2] for categorical label(s)", required=True)
    parser.add_argument("--query_size", type=int, default=100, help="Number of queries", required=True)
    parser.add_argument("--predicate_file", type=str, help="Output predicate file path", required=True)
    args = parser.parse_args()
    return args

def zipfProb(N=5, s=1.0):
    # 广义调和级数 H_{N,s}
    H_Ns = np.sum([1.0 / (k**s) for k in range(1, N+1)])
    # 每个类别的概率
    probs = [(1.0 / (k**s)) / H_Ns for k in range(1, N+1)]
    print(f"Zipf probabilities: {probs}")
    return np.array(probs)


def assign_labels(probs, num_items):
    """给 num_items 个 item 赋予标签，严格保持 Zipf 概率，允许空标签"""
    all_labels = []
    for _ in range(num_items):
        labels = []
        # while labels == []:
        #     labels = [i for i, p in enumerate(probs, start=1) if np.random.rand() < p]
        labels = [i for i, p in enumerate(probs, start=1) if np.random.rand() < p]
        all_labels.append(labels)  # 可能为空
    return all_labels

def generate_query_selectivity(attr_type_list, query_sel_list):
    sel_list = []
    for idx, attr_type in enumerate(attr_type_list):
        if attr_type == 0:
            sel_list.append(query_sel_list[idx])
        else:
            if isinstance(query_sel_list[idx], list):
                sel_list.append(query_sel_list[idx])
            else:
                assert isinstance(query_sel_list[idx], int)
                sel_list.append([query_sel_list[idx]])
    return sel_list

if __name__ == "__main__":
    args = arg_init()

    attr_type_list = ast.literal_eval(args.attr_type_list)
    query_sel_list = ast.literal_eval(args.query_sel)
    num_max_val = args.numerical_max_attr
    num_min_val = 0
    N = args.N
    zipf_prob = zipfProb(N=args.categorical_attr_max_cardinality, s=1.5)
    sel_list = generate_query_selectivity(attr_type_list, query_sel_list)
    _attr = read_multy_attr(args.attr_file)
    # print("ori attr:", _attr[:5])

    attr = [[] for _ in range(len(attr_type_list))]
    assert(len(_attr) == N)
    for i in range(len(_attr)):
        assert(len(_attr[i]) == len(attr_type_list))
        for idx, attr_type in enumerate(attr_type_list):
            if attr_type == 0:
                val = _attr[i][idx][0]
            else:
                val = _attr[i][idx]
            attr[idx].append(val)



    print("attr type list:", attr_type_list)
    print("query selectivity list:", sel_list)
    print("num data points:", N)

    query_predicate = [[] for _ in range(args.query_size)]
    
    for idx, attr_type in enumerate(attr_type_list):
        if attr_type == 0:
            # numerical attribute
            num_attr = np.array(attr[idx])
            sorted_index = np.argsort(num_attr)
            # generate random query range
            query_nb = int(N * sel_list[idx])
            q_ordered_range = np.random.randint(0, N-query_nb, (args.query_size, 2), dtype='int32')
            q_ordered_range[:, 1] = q_ordered_range[:, 0] + query_nb # [x, x+query_attr_size]
            # convert query index range to attr range
            q_idx = sorted_index[q_ordered_range]
            # print("q_idx:", q_idx[:5])
            query_attr = num_attr[q_idx]

            for i in range(args.query_size):
                query_predicate[i].append([int(query_attr[i][0]), int(query_attr[i][1])])
            # print example
            print(f"Example numerical attribute values for attribute {idx}: {query_predicate[:5]}")

        else:
            # categorical attribute, just keep input attr in sel_list[idx]
            for i in range(args.query_size):
                query_predicate[i].append(sel_list[idx])
            


    # write to json file
    with open(args.predicate_file, 'w') as f:
        json.dump(query_predicate, f)
    print(f"Attributes saved to {args.predicate_file}")
    print(query_predicate[:5])


