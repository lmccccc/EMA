#!/bin/bash
# Diverse-tail coloring on 128bit edge-FT.
# K_top=16 RNG-prune, then color the K_tail=4 farthest survivors per source.
set -e
cd "$(dirname "$0")"

export query_sel="[0.1,9]"
export ft_bits=128
export edge_level_ft=true
source ./conf.sh

BASE="${hashann_index_root}index_${M}_${ef_construction}_${attr_index_type}_${ft_bits}_edgeFT"
EF_LIST="[10,20,50,100,200,500]"
mkdir -p logs

K_TOP="${K_TOP:-16}"
K_TAIL="${K_TAIL:-4}"
TAG="diverseTailT${K_TOP}t${K_TAIL}"
OUT="${BASE}_${TAG}"

color() {
    python -u -c "
import sys, ast, time
sys.path.insert(0, '../tests')
from hashann import HashANN
attr_type_list = ast.literal_eval('${attr_type}')
params = {'M': ${M}, 'ef_construction': ${ef_construction}, 'metric': 'ip',
          'dim': ${dim}, 'N': ${N}, 'ef_search': 10, 'ef_top': 1, 'ft_bits': ${ft_bits}}
ha = HashANN(); ha.init_params(params)
idx = ha.load_index(params, attr_type_list, '$BASE', 1, '${algo}')
print('diverse-tail coloring K_top=${K_TOP} K_tail=${K_TAIL} ...')
t0 = time.time()
flipped = idx.color_all_ft_bits_diverse_tail(${K_TOP}, ${K_TAIL}, True)
print(f'done {time.time()-t0:.1f}s flipped={flipped}')
idx.save_index('$OUT')
print('Saved.')
"
}

run_query() {
    local idx="$1" log="$2" mindeg="$3"
    python -u ../tests/hashann_query.py \
        --data_path "$dataset_file" --index_cache_path "$idx" \
        --k 10 --N $N --M $M --dim $dim --metric ${metric} \
        --efConstruction $ef_construction \
        --ef_search "$EF_LIST" --ef_top $ef_top --name $algo \
        --query_path "$query_file" --attr_path "$dataset_attr_file" \
        --qrange_path "$query_predicate_file" --gt_path "$ground_truth_file" \
        --n_query_to_use $query_size --attr_type_list "$attr_type" \
        --ft_bits ${ft_bits} --threads 1 --use_ft true --ft_routing_min_deg ${mindeg} \
        2>&1 | tee "$log"
}

if [ ! -f "$OUT" ]; then
    echo "===== Coloring (K_top=${K_TOP} K_tail=${K_TAIL}) ====="
    color 2>&1 | tee "logs/edgeFT128_${TAG}_step.log"
fi

for MD in 0 18; do
    echo ""
    echo "===== Query md=${MD} ====="
    run_query "$OUT" "logs/edgeFT128_${TAG}_md${MD}.log" "$MD"
done

echo ""
echo "===== Summary ====="
for MD in 0 18; do
    log="logs/edgeFT128_${TAG}_md${MD}.log"
    echo "---- md=${MD} ----"
    grep -E "ef search|recall|QPS:" "$log" | tail -20
done
