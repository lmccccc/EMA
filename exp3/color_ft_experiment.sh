#!/bin/bash
# Edge-FT bit coloring experiment.
# Step 1: Load edgeft32 index, color all FT bits (BFS connectivity repair), save.
# Step 2: Query at 1% selectivity to evaluate recall/QPS.
#
# Usage: bash color_ft_experiment.sh

set -e
cd "$(dirname "$0")"

# Force 1% selectivity for predicates
export query_sel="[0.1,9]"

source ./conf.sh

# Override ft_bits to 32 for edge FT (conf.sh uses 128 by default)
ft_bits=32

EDGE_FT_INDEX="${hashann_index_root}index_${M}_${ef_construction}_${attr_index_type}_edgeft32"
COLORED_INDEX="${EDGE_FT_INDEX}_colored"

mkdir -p logs

echo "========================================"
echo "  Source index:  ${EDGE_FT_INDEX}"
echo "  Colored index: ${COLORED_INDEX}"
echo "  Predicate:     ${query_predicate_file}"
echo "  GT:            ${ground_truth_file}"
echo "========================================"

if [ ! -f "$COLORED_INDEX" ]; then
    if [ ! -f "$EDGE_FT_INDEX" ]; then
        echo "ERROR: source edge-ft index not found: $EDGE_FT_INDEX"
        exit 1
    fi
    echo ""
    echo "=== Step 1: Color all FT bits and save ==="
    python -u -c "
import sys
sys.path.insert(0, '../tests')
from hashann import HashANN
import ast, time

attr_type_list = ast.literal_eval('${attr_type}')
params = {'M': ${M}, 'ef_construction': ${ef_construction}, 'metric': '${metric}'.lower(),
          'dim': ${dim}, 'N': ${N}, 'ef_search': 10, 'ef_top': 1, 'ft_bits': 32}

hash_ann = HashANN()
hash_ann.init_params(params)
index = hash_ann.load_index(params, attr_type_list, '${EDGE_FT_INDEX}', 1, '${algo}')

print('Coloring all FT bits...')
t0 = time.time()
total_flipped = index.color_all_ft_bits(True)
print(f'Coloring done in {time.time()-t0:.1f}s, total bits flipped = {total_flipped}')

print('Saving colored index to ${COLORED_INDEX}...')
index.save_index('${COLORED_INDEX}')
print('Saved.')
" 2>&1 | tee logs/color_ft_${dataset}_step1.log
    echo "=== Coloring done ==="
else
    echo "Colored index already exists: ${COLORED_INDEX}"
fi

# Step 2: Query at 1% selectivity
LOG_FILE="logs/color_ft_${dataset}_1pct.log"
EF_LIST="[1,5,10,20,50,100,200,500]"

echo ""
echo "=== Step 2: Query 1% selectivity ==="
echo "ef_search list: $EF_LIST"

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
