#!/bin/bash
# Color edge-level FT 32-bit with K=2 and K=4.
# Measure FT bit density (honest per-edge) before/after, then run 1% query.
set -e
cd "$(dirname "$0")"

export query_sel="[0.1,9]"
export ft_bits=32
export edge_level_ft=true
source ./conf.sh

BASE="${hashann_index_root}index_${M}_${ef_construction}_${attr_index_type}_${ft_bits}_edgeFT"
EF_LIST="[1,5,10,20,50,100,200,500]"
mkdir -p logs

measure() {
    local idx="$1" tag="$2"
    python -u -c "
import hashannlib, time
idx = hashannlib.Index(space='${metric,,}', dim=${dim})
idx.load_index('$idx')
bits, edges, cap = idx.get_edge_ft_bit_stats()
print(f'[$tag] edges={edges} cap_per_edge={cap}')
for a, b in enumerate(bits):
    print(f'[$tag] attr[{a}]: total_bits={b}  mean={b/edges:.4f}  density={100.0*b/(edges*cap):.4f}%')
" 2>&1 | grep -E "\[$tag\]"
}

color_to() {
    local src="$1" dst="$2" K="$3"
    python -u -c "
import sys, ast, time
sys.path.insert(0, '../tests')
from hashann import HashANN
attr_type_list = ast.literal_eval('${attr_type}')
params = {'M': ${M}, 'ef_construction': ${ef_construction}, 'metric': '${metric}'.lower(),
          'dim': ${dim}, 'N': ${N}, 'ef_search': 10, 'ef_top': 1, 'ft_bits': ${ft_bits}}
ha = HashANN(); ha.init_params(params)
idx = ha.load_index(params, attr_type_list, '$src', 1, '${algo}')
print(f'Coloring K=$K ...')
t0 = time.time()
flipped = idx.color_all_ft_bits($K, True)
print(f'K=$K coloring done {time.time()-t0:.1f}s flipped={flipped}')
idx.save_index('$dst')
print('Saved.')
"
}

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

K2_IDX="${BASE}_coloredK2"
K4_IDX="${BASE}_coloredK4"

echo ""
echo "===== Baseline density ====="
measure "$BASE" "BASE"

if [ ! -f "$K2_IDX" ]; then
    echo ""
    echo "===== Color K=2 ====="
    color_to "$BASE" "$K2_IDX" 2 2>&1 | tee logs/edgeFT32_colorK2_step.log
fi
echo ""
echo "===== K=2 density ====="
measure "$K2_IDX" "K=2"

echo ""
echo "===== K=2 query ====="
run_query "$K2_IDX" "logs/edgeFT32_K2_1pct.log"

if [ ! -f "$K4_IDX" ]; then
    echo ""
    echo "===== Color K=4 ====="
    color_to "$BASE" "$K4_IDX" 4 2>&1 | tee logs/edgeFT32_colorK4_step.log
fi
echo ""
echo "===== K=4 density ====="
measure "$K4_IDX" "K=4"

echo ""
echo "===== K=4 query ====="
run_query "$K4_IDX" "logs/edgeFT32_K4_1pct.log"

echo ""
echo "===== Summary ====="
for tag in baseline colored K2 K4; do
    log="logs/edgeFT32_${tag}_1pct.log"
    [ -f "$log" ] || continue
    echo "---- $tag ----"
    grep -E "Recall|QPS:" "$log" | tail -20
done
