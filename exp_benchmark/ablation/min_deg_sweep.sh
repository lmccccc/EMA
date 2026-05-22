#!/usr/bin/env bash
# min_deg_sweep.sh — sweep FT routing min_deg.  Single index (M=40).
set -euo pipefail
cd "$(dirname "$0")/.."

source ./env.sh
source ./ablation/selectivity_specs.sh

dataset="${dataset:-Redcaps_4M}"
OUTDIR="logs/min_deg_sweep"
mkdir -p "$OUTDIR"

MD_VALS=("${MD_VALS:-0 5 8 15 20 30 40 80}")
if [ -z "${MD_VALS:-}" ]; then
    MD_VALS=(0 5 8 15 20 30 40 80)
fi

# Build once.
dataset="$dataset" ./build_index.sh

for md in "${MD_VALS[@]}"; do
    echo "==== min_deg=$md ===="
    for entry in "${EMA_SEL6[@]}"; do
        IFS='|' read -r target name spec <<< "$entry"
        LOG="$OUTDIR/${dataset}__md${md}__T${target}__${name}.log"
        if [ -f "$LOG" ] && grep -q "Final results" "$LOG" 2>/dev/null; then
            echo "[skip] $LOG"
            continue
        fi
        echo "[cell] $dataset md=$md T${target}"
        dataset="$dataset" dnf_spec="$spec" dnf_name="$name" \
            FT_ROUTING_MIN_DEG="$md" \
            ./query.sh > "$LOG" 2>&1
    done
done

echo "[min_deg-sweep] DONE → $OUTDIR/"
echo "Aggregate: python aggregate.py $OUTDIR"
