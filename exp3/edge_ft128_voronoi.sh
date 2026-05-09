#!/bin/bash
# Voronoi K-NN bridge coloring on 128bit edge-FT index.
set -e
cd "$(dirname "$0")"

export query_sel="[0.1,9]"
export ft_bits=128
export edge_level_ft=true
source ./conf.sh

BASE="${hashann_index_root}index_${M}_${ef_construction}_${attr_index_type}_${ft_bits}_edgeFT"
EF_LIST="[10,12,15,20,30,50,100]"
mkdir -p logs

K_LIST="${K_LIST:-4}"
MIN_DEG_LIST="${MIN_DEG_LIST:-18}"

color_voronoi() {
    local src="$1" dst="$2" K="$3"
    python -u -c "
import sys, ast, time
sys.path.insert(0, '../tests')
from hashann import HashANN
attr_type_list = ast.literal_eval('${attr_type}')
params = {'M': ${M}, 'ef_construction': ${ef_construction}, 'metric': 'ip',
          'dim': ${dim}, 'N': ${N}, 'ef_search': 10, 'ef_top': 1, 'ft_bits': ${ft_bits}}
ha = HashANN(); ha.init_params(params)
idx = ha.load_index(params, attr_type_list, '$src', 1, '${algo}')
print(f'Voronoi coloring K_neighbors=$K ...')
t0 = time.time()
flipped = idx.color_all_ft_bits_voronoi($K, True)
print(f'K_neighbors=$K coloring done {time.time()-t0:.1f}s flipped={flipped}')
idx.save_index('$dst')
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

for K in $K_LIST; do
    OUT="${BASE}_voronoiK${K}"
    if [ ! -f "$OUT" ]; then
        echo "===== Voronoi color K_neighbors=$K (128bit) ====="
        color_voronoi "$BASE" "$OUT" "$K" 2>&1 | tee "logs/edgeFT128_voronoiK${K}_step.log"
    else
        echo "===== Existing $OUT, skip coloring ====="
    fi

    for MD in $MIN_DEG_LIST; do
        echo ""
        echo "===== K=$K min_deg=$MD query ====="
        run_query "$OUT" "logs/edgeFT128_voronoiK${K}_md${MD}_1pct.log" "$MD"
    done
done

echo ""
echo "===== Summary ====="
for K in $K_LIST; do
    for MD in $MIN_DEG_LIST; do
        log="logs/edgeFT128_voronoiK${K}_md${MD}_1pct.log"
        [ -f "$log" ] || continue
        echo "---- voronoi K=$K min_deg=$MD ----"
        grep -E "ef search|recall|QPS:" "$log" | tail -30
    done
done
