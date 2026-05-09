#!/bin/bash
# Full selectivity sweep for DiverseTail T16/t4 colored 128bit edge-FT
# Selectivities: 10%, 20%, 40%, 60%, 80%, 100% (target ~95% recall)
set -e
cd "$(dirname "$0")"

IDX="/mnt/data/mocheng/dataset/redcaps4m/hashann/index/index_40_300_arbi_0_1_random_128_edgeFT_diverseTailT16t4"
LABEL_ROOT="/mnt/data/mocheng/dataset/redcaps4m/label/arbi_0_1_random/"
mkdir -p logs
LOG="logs.txt"
: > "$LOG"

run_one() {
    local sel="$1" tag="$2"
    local pred="${LABEL_ROOT}predicate_arbi_0_1_${sel}.json"
    local gt="${LABEL_ROOT}gt_arbi_0_1_${sel}.json"
    {
      echo ""
      echo "===================================================================="
      echo "===== sel=$tag  pred=$sel ====="
      echo "===================================================================="
    } | tee -a "$LOG"
    python -u ../tests/hashann_query.py \
        --data_path "/mnt/data/mocheng/dataset/redcaps4m/image_embeddings.fvecs" \
        --index_cache_path "$IDX" \
        --k 10 --N 4000000 --M 40 --dim 512 --metric IP \
        --efConstruction 300 --ef_search "[10,20,50,100,200]" --ef_top 1 --name HNSW \
        --query_path "/mnt/data/mocheng/dataset/redcaps4m/query.fvecs" \
        --attr_path "/mnt/data/mocheng/dataset/redcaps4m/label/arbi_0_1_random/attr_arbi_0_1_random.json" \
        --qrange_path "$pred" --gt_path "$gt" \
        --n_query_to_use 1000 --attr_type_list "[0,1]" \
        --ft_bits 128 --threads 1 --use_ft true --ft_routing_min_deg 18 \
        2>&1 | tee -a "$LOG" > "logs/diverseTail_full_${tag}.log"
}

run_one "[0.333,7]" "10pct"
run_one "[0.4,5]"   "20pct"
run_one "[0.667,4]" "40pct"
run_one "[0.75,2]"  "60pct"
run_one "[0.9,1]"   "80pct"
run_one "[1.0,0]"   "100pct"

{
  echo ""
  echo "===================================================================="
  echo "===== Summary (recall / QPS per ef) ====="
  echo "===================================================================="
  for tag in 10pct 20pct 40pct 60pct 80pct 100pct; do
      echo ""
      echo "---- $tag ----"
      grep -E "ef search|recall|QPS:" "logs/diverseTail_full_${tag}.log" | tail -30
  done
} | tee -a "$LOG"
