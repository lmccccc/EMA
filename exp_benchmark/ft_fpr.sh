#!/usr/bin/env bash
# ft_fpr.sh — measure EMA / FT false-positive rate at multiple selectivities.
#
# Defaults to Redcaps_4M with the 128-bit node-FT index
# (`index_40_300_arbi_0_1_random_128_nb2_mo2_do1`). Override via env vars below.
#
# Usage:
#   bash exp_benchmark/ft_fpr.sh
#   dataset=Redcaps_4M M=40 ft_bits=128 bash exp_benchmark/ft_fpr.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# shellcheck disable=SC1091
source exp_benchmark/env.sh

dataset="${dataset:-Redcaps_4M}"
M="${M:-40}"
ef_construction="${ef_construction:-300}"
ft_bits="${ft_bits:-128}"
attr_type="${attr_type:-0,1}"
ef_search_list="${ef_search_list:-10,50,200}"
n_query="${n_query:-1000}"
selectivities="${selectivities:-[0.1,9]:1%;[0.177,7]:5%;[0.75,2]:60%}"

ts="$(date +%Y%m%d_%H%M%S)"
out_dir="exp_benchmark/logs/ft_fpr_${dataset}_${ts}"
mkdir -p "$out_dir"

echo "[ft_fpr] dataset=$dataset M=$M efc=$ef_construction ftb=$ft_bits"
echo "[ft_fpr] out_dir=$out_dir"

python exp_benchmark/ft_fpr.py \
    --dataset "$dataset" \
    --M "$M" \
    --ef_construction "$ef_construction" \
    --ft_bits "$ft_bits" \
    --attr_type "$attr_type" \
    --ef_search_list "$ef_search_list" \
    --n_query "$n_query" \
    --selectivities "$selectivities" \
    --out_json "$out_dir/results.json" \
    2>&1 | tee "$out_dir/run.log"

echo "[ft_fpr] done -> $out_dir"
