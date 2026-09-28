#!/usr/bin/env bash
# main_experiment.sh — EMA results on 4 datasets × 6 selectivities.
set -euo pipefail
cd "$(dirname "$0")"

source ./env.sh
source ./ablation/selectivity_specs.sh

read -r -a DATASETS <<< "${EMA_DATASETS:-sift10m youtube_rgb wiki_15_4M Redcaps_4M}"

OUTDIR=logs/main
mkdir -p "$OUTDIR"

for ds in "${DATASETS[@]}"; do
    echo "===================================================="
    echo "[main] dataset=$ds"
    echo "===================================================="

    # Build index once per dataset
    dataset="$ds" ./build_index.sh

    for entry in "${EMA_SEL6[@]}"; do
        IFS='|' read -r target name spec <<< "$entry"
        LOG="$OUTDIR/${ds}__T${target}__${name}.log"
        if [ -f "$LOG" ] && grep -q "Final results" "$LOG" 2>/dev/null; then
            echo "[skip] $LOG"
            continue
        fi
        echo "[cell] $ds T${target}"
        dataset="$ds" dnf_spec="$spec" dnf_name="$name" \
            ./predicate_generator.sh
        dataset="$ds" dnf_spec="$spec" dnf_name="$name" \
            ./ground_truth_generator.sh
        dataset="$ds" dnf_spec="$spec" dnf_name="$name" \
            ./query.sh > "$LOG" 2>&1
        grep -E '^ef search:' "$LOG" | tail -3 || true
    done
done

echo "[main] DONE. Logs under $OUTDIR/"
echo "Run: python aggregate.py $OUTDIR"
