#!/usr/bin/env python3
import sys

if len(sys.argv) < 7 or len(sys.argv) > 9:
    print("Usage: python extract_results.py <log_file> <dataset> <attr_type> <query_sel> <M> <algo> [K] [predicate_file]")
    sys.exit(1)

K = int(sys.argv[7]) if len(sys.argv) >= 8 else 10
predicate_file = sys.argv[8] if len(sys.argv) >= 9 else ""

log_path = sys.argv[1]
output_path = "logs.txt"
targets = ["Final results", 
           "efs, recall, qps: ",
           "efs, recall, qps, cmps:",
           "recall:",
           "cmps:",
            "================================================================================="]

with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

# 从末尾向上找最后一次出现
start_idx = None
for i in range(len(lines) - 1, -1, -1):
    if any(target in lines[i] for target in targets):
        start_idx = i
        break

if sys.argv[6] == "diskann" or sys.argv[6] == "diskann_stitched":
    # 向上找到 "Final results" 行
    for i in range(len(lines) - 1, -1, -1):
        if "Done searching." in lines[i]:
            end_idx = i
            break

if start_idx is None:
    print(f'Error: targets not found in {log_path}')
    sys.exit(2)


def _extract_result_rows(section_lines):
    """Extract rows like [efs, recall, qps] or [efs, recall, qps, cmps]."""
    rows = []
    for line in section_lines:
        s = line.strip()
        if not (s.startswith("[") and s.endswith("]")):
            continue

        inner = s[1:-1]
        parts = [p.strip() for p in inner.split(",")]
        # old format: [efs, recall, qps]
        # new format: [efs, recall, qps, cmps]
        if len(parts) in (3, 4):
            rows.append(s)
    return rows

# 追加写入 logs.txt
import fcntl
with open(output_path, "a", encoding="utf-8") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    try:
# echo "dataset: $dataset" >> logs.txt
# echo "attr: $attr_type" >> logs.txt
# echo "sel: $query_sel" >> logs.txt
# echo "algo: bfann" >> logs.txt
        f.write("dataset: " + sys.argv[2] + "\n")
        f.write("attr: " + sys.argv[3] + "\n")
        f.write("sel: " + sys.argv[4] + "\n")
        f.write("M: " + sys.argv[5] + "\n")
        f.write("K: " + str(K) + "\n")
        f.write("algo: " + sys.argv[6] + "\n")
        if predicate_file:
            f.write("predicate: " + predicate_file + "\n")
        if sys.argv[6] != "diskann" and sys.argv[6] != "diskann_stitched":
            # Prefer normalized summary rows if present, fallback to raw tail for legacy logs.
            summary_rows = _extract_result_rows(lines[start_idx:])
            if len(summary_rows) > 0:
                for row in summary_rows:
                    f.write(row + "\n")
            else:
                f.writelines(lines[start_idx:])
        else:
            #   10     3685.61            415.74              271.15         615.82       32.15
            for line in lines[start_idx+1:end_idx]:
                parts = line
                parts = parts.split()
                # print("parts:", parts)
                efs = parts[0]
                recall = parts[5]
                qps = parts[1]
                if efs == "Found":
                    continue
                f.write(f"[{efs}, {recall}, {qps}]\n")
        # f.writelines(lines[start_idx:])
        f.write("\n")   # 结束后追加一个空行
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)

print(f'Appended results from "{log_path}" to {output_path}')
