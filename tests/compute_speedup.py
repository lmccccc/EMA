str1 = '''
# bfann results
# Target recall: 0.95
bfann_youtube_rgb_95_df <- data.frame(
  x = c(0.10, 0.07, 0.05, 0.03, 0.02, 0.01),
  y = c(359.89, 335.58, 311.77, 200.91, 121.94, 52.74)
)

# new
# irange_multi results
# Target recall: 0.95
irange_multi_youtube_rgb_95_df <- data.frame(
  x = c(0.10, 0.07, 0.05, 0.03, 0.02, 0.01),
  y = c(213.01, 159.25, 135.54, 110.01, 93.04, 54.04)
)


# navix results
# Target recall: 0.95
navix_youtube_rgb_95_df <- data.frame(
  x = c(0.10, 0.07, 0.05, 0.03, 0.02, 0.01),
  y = c(29.27, 41.23, 25.20, 25.17, 20.86, 8.61)
)

# milvus results
# Target recall: 0.95
milvus_youtube_rgb_95_df <- data.frame(
  x = c(0.10, 0.07, 0.05, 0.03, 0.02, 0.01),
  y = c(124, 116.43, 85.51, 99.81, 130.51, 128.28)
)
'''


def get_my_data(line_str):
    # extract str to lines
    lines = line_str.strip().split('\n')
    starts = []
    ends = []
    res = {}
    for i, line in enumerate(lines):
        if "_95_df <- data.frame" in line:
            starts.append(i)
        if line == ")":
            ends.append(i)

    res = {}
    for idx, start in enumerate(starts):
        end = ends[idx]
        method = None
        sel_values = None
        qps_values = None
        while start < end:
            if "bfann" in lines[start]:
                method = "bfann"
            if "x = " in lines[start]:
                sel = lines[start]
                sel = sel.strip().replace("x = c(", "").replace("),", "")
                sel_values = [float(x.strip()) for x in sel.split(",")]

            if "y = " in lines[start]:
                qps = lines[start]
                qps = qps.strip().replace("y = c(", "").replace(")", "")
                qps_values = [float(x.strip()) for x in qps.split(",")]
            start += 1
        if method is None:
            continue
        else:
            if len(sel_values) != len(qps_values):
                print(f"Error: length mismatch for {method}, sel_values: {len(sel_values)}, qps_values: {len(qps_values)}")
                continue
            for i in range(len(sel_values)):
                res[sel_values[i]] = qps_values[i]
            return res
    print("Error: bfann data not found")
    exit(1)

def get_sota_data(line_str):
    # extract str to lines
    lines = line_str.strip().split('\n')
    starts = []
    ends = []
    res = {}
    for i, line in enumerate(lines):
        if "_95_df <- data.frame" in line:
            starts.append(i)
        if line == ")":
            ends.append(i)
    print("starts:", starts)
    for idx, start in enumerate(starts):
        end = ends[idx]
        method = None
        sel_values = None
        qps_values = None
        while start < end:
            if "bfann" in lines[start]:
                break
            if "x = " in lines[start]:
                sel = lines[start]
                sel = sel.strip().replace("x = c(", "").replace("),", "")
                sel_values = [float(x.strip()) for x in sel.split(",")]

            if "y = " in lines[start]:
                qps = lines[start]
                qps = qps.strip().replace("y = c(", "").replace(")", "")
                qps_values = [float(x.strip()) for x in qps.split(",")]
            start += 1
        if sel_values is None or qps_values is None:
            continue
        if len(sel_values) != len(qps_values):
            print(f"Error: length mismatch for {method}, sel_values: {len(sel_values)}, qps_values: {len(qps_values)}")
            continue
        for i in range(len(sel_values)):
            if  sel_values[i] in res:
                if qps_values[i] < res[sel_values[i]]:
                    print("skip ", qps_values[i])
                    continue
                else:
                    print("update ", qps_values[i])
            else:
                print("add sel:", sel_values[i], " qps:", qps_values[i])
            res[sel_values[i]] = qps_values[i]
    return res
    
my_data = get_my_data(str1)
sota_data = get_sota_data(str1)

# compute speedup 
speedup_list = []
for sel_value in my_data.keys():
    if sel_value in sota_data:
        my_qps = my_data[sel_value]
        sota_qps = sota_data[sel_value]
        speedup = my_qps / sota_qps
        speedup_list.append(speedup)

avg_speedup = sum(speedup_list) / len(speedup_list)
best_speedup = max(speedup_list)
print(f"Average speedup: {avg_speedup:.2f}x")
print(f"Best speedup: {best_speedup:.2f}x")