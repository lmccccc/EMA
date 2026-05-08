



file = "logs.txt"

target_dataset = "youtube_rgb" # wiki_15_4M sift10m Redcaps_4M youtube_rgb wiki_negcorr_1_01
# "[0,1]", "[0,1] high", "[0,0]"", "[0,0] biased", "[1]", "[0]"
target_attr = "[0,1]"
# target_sel_name = "[0,1]"
target_algo = ["bfann", "navix", "acorn", "milvus", "msvbase", "bfann_256", "irange_multi", "diskann", "hnsw", "hnsw_navix"]
target_recall = {
    0.9: 0.9 - 0.001,
    0.95: 0.95 - 0.001,
    0.99: 0.99 - 0.001,
    0.83: 0.83 - 0.001,
    0.80: 0.80 - 0.001
}
target_M = 40
global_target_r = 0.95
target_k = 10


if target_attr == "[0]":
    # [0]
    sel_mapping = {
        "[0.01]": 0.01,
        "[0.02]": 0.02,
        "[0.03]": 0.03,
        "[0.05]": 0.05,
        "[0.07]": 0.07,
        "[0.1]": 0.1
    }
elif target_attr == "[0,1]":
    # [0,1]
    sel_mapping = {
        "[0.1,9]": 0.01,
        "[0.1,8]": 0.02,
        "[0.15,8]": 0.03,
        "[0.177,7]": 0.05,
        "[0.233,7]": 0.07,
        "[0.333,7]": 0.1
    }
elif target_attr == "[0,1] high":
    # [0,1] high
    sel_mapping = {
        "[0.333,7]": 0.1,
        "[0.4,5]": 0.2,
        "[0.667,4]": 0.4,
        "[0.75,2]": 0.6,
        "[0.9,1]": 0.8,
        "[1.0,0]": 1.0
    }
# elif target_attr == "[0,0]":
#     # [0,0]
#     sel_mapping = {
#         "[0.1,0.1]": 0.01,
#         "[0.141,0.141]": 0.02,
#         "[0.173,0.173]": 0.03,
#         "[0.2236,0.2236]": 0.05,
#         "[0.2646,0.2646]": 0.07,
#         "[0.316,0.316]": 0.1
#     }
elif target_attr == "[0,0]":
    # [0,0] biased
    sel_mapping = {
        "[0.55,0.01818]": 0.01,
        "[0.60,0.03333]": 0.02,
        "[0.65,0.04615]": 0.03,
        "[0.70,0.07143]": 0.05,
        "[0.75,0.09333]": 0.07,
        "[0.80,0.125]": 0.1
    }
elif target_attr == "[1]":
    # [1]
    sel_mapping = {
        "[18]": 0.01,
        "[17]": 0.02,
        "[16]": 0.03,
        "[14]": 0.05,
        "[12]": 0.07,
        "[9]": 0.1
    }
elif target_attr == "wiki uncorr":
    # wiki uncorr
    sel_mapping = {
        "[0.0101]": 0.0101,
        "[0.0510]": 0.0510,
        "[0.0996]": 0.0996,
        "[0.1502]": 0.1502,
        "[0.2293]": 0.2293
    }











res = {}

def add_to_res(res, matched_algo, sel_value, target_recall, qps):
    if matched_algo not in res:
        res[matched_algo] = {}
    if sel_value not in res[matched_algo]:
        res[matched_algo][sel_value] = {}
    if target_recall not in res[matched_algo][sel_value]:
        res[matched_algo][sel_value][target_recall] = qps
    else:
        # take max qps
        if qps > res[matched_algo][sel_value][target_recall]:
            res[matched_algo][sel_value][target_recall] = qps

# read by line
with open(file, 'r') as f:
    lines = f.readlines()

algo_start = []
algo_end = []
line_idx = 0
while line_idx < len(lines):
    if "dataset:" in lines[line_idx]:
        algo_start.append(line_idx)
    line_idx += 1
for idx in range(len(algo_start)-1):
    algo_end.append(algo_start[idx+1])
algo_end.append(len(lines))



for i in range(len(algo_start)):
# while line_idx < len(lines):
    line_idx = algo_start[i]
    sel_value = None
    M = 180
    K = 10
    matched_algo = None
    while line_idx < algo_end[i]:
        if "dataset:" in lines[line_idx]:
            if target_dataset not in lines[line_idx]:
                break
        
        if "attr:" in lines[line_idx]:
            if target_attr not in lines[line_idx]:
                break
        
        if "sel:" in lines[line_idx]:
            sel_str = lines[line_idx].strip().split("sel: ")[1]
            if sel_str not in sel_mapping.keys():
                break
            sel_value = sel_mapping[sel_str]

        if f"M:" in lines[line_idx]:
            M = int(lines[line_idx].strip().split("M: ")[1])

        if f"K:" in lines[line_idx]:
            K = int(lines[line_idx].strip().split("K: ")[1])

        if "algo:" in lines[line_idx]:
            for algo in target_algo:
                # "algo: xxxx"
                if f"algo: {algo}\n" == lines[line_idx]:
                    matched_algo = algo
            if matched_algo is None:
                break

        # start check recall
        # type 2: recall: 0.7019, qps: 301.90
        if lines[line_idx].startswith("recall:"):
            if M != target_M:
                break
            if K != target_k:
                break
            parts = lines[line_idx].strip().split(',')
            recall = float(parts[0].split('recall:')[1].strip())
            qps = float(parts[1].split('qps:')[1].strip())
            for target_r in target_recall.keys():
                if recall >= target_recall[target_r]:
                    add_to_res(res, matched_algo, sel_value, target_r, qps)
            line_idx += 1
            continue                    
        # type 1: [400, 0.980, 55.502], efs, recall, qps
        if lines[line_idx][0] == '[':
            if M != target_M:
                break
            if K != target_k:
                break
            parts = lines[line_idx].strip().strip('[]').split(',')
            efs = int(parts[0].strip())
            recall = float(parts[1].strip())
            qps = float(parts[2].strip())

            if matched_algo == "diskann":
                recall /= 100.0
            for target_r in target_recall.keys():
                if recall >= target_recall[target_r]:
                    add_to_res(res, matched_algo, sel_value, target_r, qps)
        line_idx += 1

# print result in R data.frame format
for method_name in res.keys():
    print(f"# {method_name} results")
    method = res[method_name]
    # for target_r in target_recall.keys():
    print(f"# Target recall: {global_target_r}")
    print(f"{method_name}_{target_dataset}_{int(global_target_r*100)}_df <- data.frame(")
    xs = []
    ys = []
    for sel_value in sorted(method.keys(), reverse=True):
        if global_target_r in method[sel_value].keys():
            xs.append(sel_value)
            ys.append(method[sel_value][global_target_r])
    # .2f format
    print(f"  x = c({', '.join(map(lambda x: format(x, '.2f'), xs))}),")
    print(f"  y = c({', '.join(map(lambda y: format(y, '.2f'), ys))})")
    print(")")

    print("\n")


# compute optimization (hashann/sota)

# val_list = []
# for sel_value in sorted(sel_mapping.values(), reverse=True):
#     print(f"# sel_value: {sel_value}")
#     bfann = None
#     navix = None
#     for method_name in res.keys():
#         method = res[method_name]
#         if method_name == "bfann":
#             if sel_value in method.keys():
#                 if global_target_r in method[sel_value].keys():
#                     bfann = method[sel_value][global_target_r]
#                     print(f"bfann qps: {bfann}")
#         elif method_name == "navix":
#             if sel_value in method.keys():
#                 if global_target_r in method[sel_value].keys():
#                     navix = method[sel_value][global_target_r]
#                     print(f"navix qps: {navix}")
#     if bfann is not None and navix is not None:
#         val = bfann / navix
#         val_list.append(val)

# # best and average speedup
# print("Speedup over hashann:")
# if len(val_list) > 0:
#     best_speedup = max(val_list)
#     avg_speedup = sum(val_list) / len(val_list)
#     print(f"Best speedup over hashann: {best_speedup:.2f}x")
#     print(f"Average speedup over hashann: {avg_speedup:.2f}x")
# else:
#     print("No valid comparisons found.")