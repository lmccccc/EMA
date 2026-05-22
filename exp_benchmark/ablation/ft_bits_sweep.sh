#!/usr/bin/env bash
# ft_bits_sweep.sh — sweep bloom width ft_bits ∈ {32, 64, 128, 256}.
set -euo pipefail
cd "$(dirname "$0")/.."

source ./env.sh
source ./ablation/selectivity_specs.sh

dataset="${dataset:-Redcaps_4M}"
OUTDIR="logs/ft_bits_sweep"
mkdir -p "$OUTDIR"

FT_VALS=("${FT_VALS:-32 64 128 256}")
if [ -z "${FT_VALS:-}" ]; then
    FT_VALS=(32 64 128 256)
fi

for ftv in "${FT_VALS[@]}"; do
    echo "==== ft_bits=$ftv ===="
    dataset="$dataset" ft_bits="$ftv" ./build_index.sh

    for entry in "${EMA_SEL6[@]}"; do
        IFS='|' read -r target name spec <<< "$entry"
        LOG="$OUTDIR/${dataset}__ft${ftv}__T${target}__${name}.log"
        if [ -f "$LOG" ] && grep -q "Final results" "$LOG" 2>/dev/null; then
            echo "[skip] $LOG"
            continue
        fi
        echo "[cell] $dataset ft=$ftv T${target}"
        dataset="$dataset" ft_bits="$ftv" dnf_spec="$spec" dnf_name="$name" \
            ./predicate_generator.sh
        dataset="$dataset" ft_bits="$ftv" dnf_spec="$spec" dnf_name="$name" \
            ./ground_truth_generator.sh
        dataset="$dataset" ft_bits="$ftv" dnf_spec="$spec" dnf_name="$name" \
            ./query.sh > "$LOG" 2>&1
    done
done

echo "[ft_bits-sweep] DONE → $OUTDIR/"
echo "Aggregate: python aggregate.py $OUTDIR"
