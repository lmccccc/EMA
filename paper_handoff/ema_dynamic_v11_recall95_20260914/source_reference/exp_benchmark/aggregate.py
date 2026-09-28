#!/usr/bin/env python3
"""aggregate.py — collect EMA log directories into a recall=0.95 QPS table.

Usage:
    python aggregate.py <log_dir> [--target_recall 0.95]

Parses every `*.log` produced by query.sh.  Each log is expected to contain a
block of the form:

    Final results (ef_search, recall, QPS, cmps):
    [10, 0.71, 1434.18, -1.0]
    [15, 0.82,  921.78, -1.0]
    ...

For each log we interpolate the QPS at the requested recall and print a TSV.
Filename convention is `<group>__<cell>.log`, e.g. `Redcaps_4M__T5__or_T5.log`.
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from collections import defaultdict


FINAL_RE = re.compile(r"Final results.*?\n((?:\[.*?\]\s*\n)+)", re.S)


def parse_log(path: str) -> list[tuple[int, float, float]] | None:
    try:
        txt = open(path).read()
    except OSError:
        return None
    m = FINAL_RE.search(txt)
    if not m:
        return None
    rows = []
    for line in m.group(1).strip().splitlines():
        try:
            ef, recall, qps, _ = ast.literal_eval(line.strip())
        except (ValueError, SyntaxError):
            continue
        rows.append((int(ef), float(recall), float(qps)))
    rows.sort(key=lambda r: r[0])
    return rows


def qps_at_recall(rows: list[tuple[int, float, float]], target: float) -> float | None:
    if not rows:
        return None
    if rows[0][1] >= target:
        return rows[0][2]
    for (ef0, r0, q0), (ef1, r1, q1) in zip(rows, rows[1:]):
        if r0 <= target <= r1 and r1 > r0:
            return q0 + (q1 - q0) * (target - r0) / (r1 - r0)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log_dir")
    ap.add_argument("--target_recall", type=float, default=0.95)
    args = ap.parse_args()

    if not os.path.isdir(args.log_dir):
        print(f"[aggregate] not a directory: {args.log_dir}", file=sys.stderr)
        return 1

    table: dict[str, dict[str, float | None]] = defaultdict(dict)
    cells: set[str] = set()

    for fn in sorted(os.listdir(args.log_dir)):
        if not fn.endswith(".log"):
            continue
        stem = fn[:-4]
        parts = stem.split("__", 1)
        if len(parts) == 2:
            group, cell = parts
        else:
            group, cell = stem, ""
        rows = parse_log(os.path.join(args.log_dir, fn))
        if rows is None:
            print(f"[warn] no Final results in {fn}", file=sys.stderr)
            continue
        q = qps_at_recall(rows, args.target_recall)
        table[group][cell] = q
        cells.add(cell)

    if not table:
        print("[aggregate] no usable logs", file=sys.stderr)
        return 1

    cells_sorted = sorted(cells)
    print("group\t" + "\t".join(cells_sorted))
    for group in sorted(table):
        row = [group]
        for c in cells_sorted:
            v = table[group].get(c)
            row.append("" if v is None else f"{v:.1f}")
        print("\t".join(row))

    return 0


if __name__ == "__main__":
    sys.exit(main())
