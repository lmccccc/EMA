#!/bin/bash
# Sweep ef_top to test if more entry points recover recall under hard edge-FT
# Uses Redcaps_4M dataset with edge-level FT (32-bit), ~1% selectivity
set -euo pipefail

# --- Config (standalone) ---
dataset="Redcaps_4M"
N=4000000
dim=512
K=10
M=40
ef_construction=300
metric="IP"
algo="HNSW"
query_size=1000
use_ft="true"

index_root="/mnt/data/mocheng/dataset/redcaps4m"
dataset_file="${index_root}/image_embeddings.fvecs"
query_file="${index_root}/query.fvecs"
dataset_attr_file="${index_root}/label/arbi_0_1_random/attr_arbi_0_1_random.json"
query_predicate_file="${index_root}/label/arbi_0_1_random/predicate_arbi_0_1_[0.1,9].json"
ground_truth_file="${index_root}/label/arbi_0_1_random/gt_arbi_0_1_[0.1,9].json"

attr_type="[0,1]"
hashann_index_file="${index_root}/hashann/index/index_40_300_arbi_0_1_random_edgeft32"

# --- Validate ---
if [ ! -f "$hashann_index_file" ]; then
    echo "ERROR: Index not found: $hashann_index_file"
    exit 1
fi

mkdir -p logs
LOG_FILE="logs/sweep_ef_top_edgeft_${dataset}.log"
echo "=== ef_top sweep (edge-FT): ${dataset}, selectivity ~1% ===" | tee "$LOG_FILE"
echo "index: $hashann_index_file" | tee -a "$LOG_FILE"
echo "ef_search: [10,50,100,200]" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

for ef_top_val in 10 50 100 200 500; do
    echo "========== ef_top=${ef_top_val} ==========" | tee -a "$LOG_FILE"
    python -u tests/hashann_query.py \
        --data_path "$dataset_file" \
        --index_cache_path "$hashann_index_file" \
        --k $K \
        --N $N \
        --M $M \
        --dim $dim \
        --metric "${metric}" \
        --efConstruction $ef_construction \
        --ef_search "[10,50,100,200]" \
        --ef_top $ef_top_val \
        --name $algo \
        --query_path "$query_file" \
        --attr_path "$dataset_attr_file" \
        --qrange_path "$query_predicate_file" \
        --gt_path "$ground_truth_file" \
        --n_query_to_use $query_size \
        --attr_type_list "$attr_type" \
        --threads 1 \
        --use_ft "$use_ft" \
        2>&1 | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
done

echo "=== Done ===" | tee -a "$LOG_FILE"
echo "Results in: $LOG_FILE"
