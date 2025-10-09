
import sys
import sys
import numpy as np
import math
import time
import argparse
import json
import ast
from utils import *

def read_data(attr_file, N, d, predicate_file, Nq):

    attr = read_multy_attr(attr_file)
    assert(len(attr) == N)

    predicate = read_multy_attr(predicate_file)
    assert(len(predicate) == Nq)

    return attr, predicate

def arg_init():
    parser = argparse.ArgumentParser(description="Index parameters")
    parser.add_argument("--d", type=int, help="Dimension of data", required=True)
    parser.add_argument("--attr_file", type=str, help="Output attribute file path", required=True)
    parser.add_argument("--attr_type_list", type=str, required=True, help="List of attribute types, 0 for numerical, 1 for categorical")
    parser.add_argument("--N", type=int, required=True, help="Number of data points")
    parser.add_argument("--query_size", type=int, default=100, help="Number of queries", required=True)
    parser.add_argument("--predicate_file", type=str, help="Output predicate file path", required=True)
    args = parser.parse_args()
    return args


# read args
if __name__ == "__main__":
    args = arg_init()

    attr_type_list = ast.literal_eval(args.attr_type_list)


    attr, predicates = read_data(args.attr_file, args.N, args.d, args.predicate_file, args.query_size)


    ids = []
    q_t = 0
    hist = np.array([0 for _ in range(11)], dtype='float')
    selectivity = np.zeros((args.query_size,), dtype='float')
    for i in range(args.query_size):
        matched_cnt = 0
        predicate = predicates[i]
        for j in range(args.N):
            match = True
            for attr_idx in range(len(attr_type_list)):
                if attr_type_list[attr_idx] == 0:
                    if len(predicate[attr_idx]) == 0:
                        continue
                    if not len(predicate[attr_idx]) == 2:
                        print("error: numerical predicate length not 2")
                        sys.exit(-1)
                    if attr[j][attr_idx][0] < predicate[attr_idx][0] or attr[j][attr_idx][0] > predicate[attr_idx][1]:
                        match = False
                        break
                else:
                    for val in predicate[attr_idx]:
                        if val not in attr[j][attr_idx]:
                            match = False
                            break
            if match:
                matched_cnt += 1
        selectivity[i] = matched_cnt / args.N

    overall_sel = np.sum(selectivity) / args.query_size        
    print(f"overall selectivity: {overall_sel}")
    print("selectivity dist:")
    hist, _ = np.histogram(selectivity, bins=10, range=(0.0, 1.0))

    hist[-1] += np.sum(np.isclose(selectivity, 1.0))

    for i in range(10):
        print(f"[{i/10}, {(i+1)/10}]: {hist[i]}, {hist[i]/args.query_size}")



        