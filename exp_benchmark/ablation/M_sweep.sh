#!/usr/bin/env bash
# M_sweep.sh — sweep HNSW max-degree M across 6 selectivity points.
# Defaults run on Redcaps_4M; override via `dataset=...`.
set -euo pipefail
cd "$(dirname "$0")/.."

source ./env.sh
source ./ablation/selectivity_specs.sh

dataset="${dataset:-Redcaps_4M}"
OUTDIR="logs/M_sweep"
mkdir -p "$OUTDIR"

read -r -a M_VALUES <<< "${M_VALS:-16 32 40 64}"

for Mv in "${M_VALUES[@]}"; do
    echo "==== M=$Mv ===="
    dataset="$dataset" M="$Mv" ./build_index.sh

    for entry in "${EMA_SEL6[@]}"; do
        IFS='|' read -r target name spec <<< "$entry"
        LOG="$OUTDIR/${dataset}__M${Mv}__T${target}__${name}.log"
        if [ -f "$LOG" ] && grep -q "Final results" "$LOG" 2>/dev/null; then
            echo "[skip] $LOG"
            continue
        fi
        echo "[cell] $dataset M=$Mv T${target}"
        dataset="$dataset" M="$Mv" dnf_spec="$spec" dnf_name="$name" \
            ./predicate_generator.sh
        dataset="$dataset" M="$Mv" dnf_spec="$spec" dnf_name="$name" \
            ./ground_truth_generator.sh
        dataset="$dataset" M="$Mv" dnf_spec="$spec" dnf_name="$name" \
            ./query.sh > "$LOG" 2>&1
    done
done

echo "[M-sweep] DONE → $OUTDIR/"
echo "Aggregate: python aggregate.py $OUTDIR"
