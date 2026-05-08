#!/bin/bash
# Edge-FT bit coloring with K-coverage (multiple disjoint repair paths).
# Tests K=2 and K=4.
#
# Usage: bash color_ft_K_experiment.sh [K]

set -e
cd "$(dirname "$0")"

K=${1:-2}
KK=${K}  # save K-coverage; conf.sh sets K=10 (kNN) below
export query_sel="[0.1,9]"
source ./conf.sh
KCOV=${KK}  # K-coverage (kept separate from $K which is kNN=10 from conf.sh)

ft_bits=32
EDGE_FT_INDEX="${hashann_index_root}index_${M}_${ef_construction}_${attr_index_type}_edgeft32"
COLORED_INDEX="${EDGE_FT_INDEX}_colored_K${KCOV}"

mkdir -p logs
echo "========================================"
echo "  Source index:  ${EDGE_FT_INDEX}"
echo "  Colored index: ${COLORED_INDEX}"
echo "  K (coverage):  ${KCOV}"
echo "========================================"

if [ ! -f "$COLORED_INDEX" ]; then
    echo ""
    echo "=== Step 1: Color all FT bits with K=${KCOV} and save ==="
    python -u -c "
import sys
sys.path.insert(0, '../tests')
from hashann import HashANN
import time

attr_type_list = [0, 1]
params = {'M': ${M}, 'ef_construction': ${ef_construction}, 'metric': '${metric}'.lower(),
          'dim': ${dim}, 'N': ${N}, 'ef_search': 10, 'ef_top': 1, 'ft_bits': 32}

ha = HashANN()
ha.init_params(params)
index = ha.load_index(params, attr_type_list, '${EDGE_FT_INDEX}', 1, '${algo}')

print('Coloring all FT bits with K=${KCOV}...')
t0 = time.time()
total_flipped = index.color_all_ft_bits(${KCOV}, True)
print(f'Done in {time.time()-t0:.1f}s, total bits flipped = {total_flipped}')

print('Saving colored index to ${COLORED_INDEX}...')
index.save_index('${COLORED_INDEX}')
print('Saved.')
" 2>&1 | tee logs/color_ft_K${KCOV}_${dataset}_step1.log
else
    echo "Colored index already exists: ${COLORED_INDEX}"
fi

# Step 2: Query at 1% selectivity
LOG_FILE="logs/color_ft_K${KCOV}_${dataset}_1pct.log"
EF_LIST="[1,5,10,20,50,100,200,500]"

echo ""
echo "=== Step 2: Query 1% selectivity ==="

python -u ../tests/hashann_query.py \
    --data_path "$dataset_file" \
    --index_cache_path "$COLORED_INDEX" \
    --k $K \
    --N $N \
    --M $M \
    --dim $dim \
    --metric ${metric} \
    --efConstruction $ef_construction \
    --ef_search "$EF_LIST" \
    --ef_top $ef_top \
    --name $algo \
    --query_path "$query_file" \
    --attr_path "$dataset_attr_file" \
    --qrange_path "$query_predicate_file" \
    --gt_path "$ground_truth_file" \
    --n_query_to_use $query_size \
    --attr_type_list "$attr_type" \
    --ft_bits ${ft_bits} \
    --threads 1 \
    --use_ft true \
    --ft_routing_min_deg 0 \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=== Done. Log: ${LOG_FILE} ==="
