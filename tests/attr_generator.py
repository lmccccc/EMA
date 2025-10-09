import json
import os
import numpy as np
import random
import argparse
import ast

def arg_init():
    parser = argparse.ArgumentParser(description="Index parameters")
    parser.add_argument("--output_file", type=str, help="Output attribute file path", required=True)
    parser.add_argument("--attr_type_list", type=str, required=True, help="List of attribute types, 0 for numerical, 1 for categorical")
    parser.add_argument("--N", type=int, required=True, help="Number of data points")
    parser.add_argument("--numerical_max_attr", type=int, default=100000, help="Max value for numerical attributes")
    parser.add_argument("--categorical_attr_max_cardinality", type=int, default=5, help="Max cardinality for each categorical attribute")
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

if __name__ == "__main__":
    args = arg_init()

    attr_type_list = ast.literal_eval(args.attr_type_list)
    num_max_val = args.numerical_max_attr
    num_min_val = 0
    N = args.N
    output_file = args.output_file
    zipf_prob = zipfProb(N=args.categorical_attr_max_cardinality, s=1.5)

    print("attr type list:", attr_type_list)
    attrs = []
    for t in attr_type_list:
        if t not in [0, 1]:
            raise ValueError("Attribute types must be 0 (numerical) or 1 (categorical)")
        if t == 0:
            # generate N numerical attributes
            num_attrs = np.random.randint(num_min_val, num_max_val, size=(N, 1))
            attrs.append(num_attrs.tolist())
        else:
            # generate 
            cat_attrs = assign_labels(zipf_prob, N)
            attrs.append(cat_attrs)

    final_attrs = []
    for i in range(N):
        data_point_attrs = []
        for j in range(len(attr_type_list)):
            if attr_type_list[j] == 0:
                data_point_attrs.append(attrs[j][i])
            else:
                data_point_attrs.append(attrs[j][i])
        final_attrs.append(data_point_attrs)
    
    # write to json file
    with open(output_file, 'w') as f:
        json.dump(final_attrs, f)
    print(f"Attributes saved to {output_file}")
    print(final_attrs[:5])
            


# numerical_attr_cardinality = 2
# numerical_max_attr = 100000
# categorical_attr_max_cardinality = 1000
# # N = 1000000
# N = 10
# a = 1.5           # Zipf distribution parameter

# attr = {}
# ranks = np.arange(1, categorical_attr_max_cardinality + 1)
# weights = 1 / np.power(ranks, a)
# probabilities = weights / weights.sum()
# print(f"probabilities: {probabilities}")
# for i in range(N):
#     # generate numerical attributes
#     num_attrs = []
#     for j in range(numerical_attr_cardinality):
#         rand_attr = random.randint(0, numerical_max_attr)
#         num_attrs.append(rand_attr)
    
#     # generate categorical attributes, zipf distribution

#     # Sample from the distribution
#     samples = np.random.choice(ranks, size=categorical_attr_max_cardinality, p=probabilities)
#     print(f"i:{i}, num attr:{num_attrs}, cat attr:{samples}")
    