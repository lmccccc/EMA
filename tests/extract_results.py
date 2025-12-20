#!/usr/bin/env python3
import sys

if len(sys.argv) != 2:
    print("Usage: python extract_final.py <log_file>")
    sys.exit(1)

log_path = sys.argv[1]
output_path = "logs.txt"
targets = ["Final results", 
           "efs, recall, qps: ",
           "recall:"]

with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

# 从末尾向上找最后一次出现
start_idx = None
for i in range(len(lines) - 1, -1, -1):
    if any(target in lines[i] for target in targets):
        start_idx = i
        break

if start_idx is None:
    print(f'Error: targets not found in {log_path}')
    sys.exit(2)

# 追加写入 logs.txt
with open(output_path, "a", encoding="utf-8") as f:
    f.writelines(lines[start_idx:])
    f.write("\n")   # 结束后追加一个空行

print(f'Appended results from "{log_path}" to {output_path}')
