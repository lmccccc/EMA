#!/bin/bash
# Compare new edge-level FT 32-bit index: baseline (no coloring) vs colored.
# Both at 1% selectivity ([0.1,9]) on Redcaps_4M.
set -e
cd "$(dirname "$0")"

export query_sel="[0.1,9]"
export ft_bits=32
export edge_level_ft=true
source ./conf.sh

BASE="${hashann_index_root}index_${M}_${ef_construction}_${attr_index_type}_${ft_bits}_edgeFT"
COLORED="${BASE}_colored"
EF_LIST="[1,5,10,20,50,100,200,500]"

echo "Base:    $BASE"
echo "Colored: $COLORED"
echo "Query:   $query_predicate_file"
echo "GT:      $ground_truth_file"
mkdir -p logs

run_query() {
    local idx="$1" log="$2"
    python -u ../tests/hashann_query.py \
        --data_path "$dataset_file" --index_cache_path "$idx" \
        --k $K --N $N --M $M --dim $dim --metric ${metric} \
        --efConstruction $ef_construction \
        --ef_search "$EF_LIST" --ef_top $ef_top --name $algo \
        --query_path "$query_file" --attr_path "$dataset_attr_file" \
        --qrange_path "$query_predicate_file" --gt_path "$ground_truth_file" \
        --n_query_to_use $query_size --attr_type_list "$attr_type" \
        --ft_bits ${ft_bits} --threads 1 --use_ft true --ft_routing_min_deg 0 \
        2>&1 | tee "$log"
}

echo ""
echo "===== STAGE 1: BASELINE (uncolored edge-FT 32) ====="
run_query "$BASE" "logs/edgeFT32_baseline_1pct.log"

if [ ! -f "$COLORED" ]; then
    echo ""
    echo "===== STAGE 2: COLORING ====="
    python -u -c "
import sys, ast, time
sys.path.insert(0, '../tests')
from hashann import HashANN
attr_type_list = ast.literal_eval('${attr_type}')
params = {'M': ${M}, 'ef_construction': ${ef_construction}, 'metric': '${metric}'.lower(),
          'dim': ${dim}, 'N': ${N}, 'ef_search': 10, 'ef_top': 1, 'ft_bits': ${ft_bits}}
ha = HashANN(); ha.init_params(params)
idx = ha.load_index(params, attr_type_list, '${BASE}', 1, '${algo}')
print('Coloring K=1 ...')
t0 = time.time()
flipped = idx.color_all_ft_bits(1, True)
print(f'Coloring done {time.time()-t0:.1f}s, total flipped={flipped}')
idx.save_index('${COLORED}')
print('Saved.')
" 2>&1 | tee logs/edgeFT32_color_step.log
fi

echo ""
echo "===== STAGE 3: COLORED QUERY ====="
run_query "$COLORED" "logs/edgeFT32_colored_1pct.log"

echo ""
echo "===== Summary ====="
echo "Baseline:"
grep -E "Recall|QPS:" logs/edgeFT32_baseline_1pct.log | tail -20
echo "----"
echo "Colored:"
grep -E "Recall|QPS:" logs/edgeFT32_colored_1pct.log | tail -20
