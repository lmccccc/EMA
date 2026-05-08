#!/bin/bash
# Step 1: Augment the index with CHT-based edges and save it.
# Step 2: Query across all selectivities using the saved augmented index.
#
# Usage: bash augment_cht_experiment.sh
# Results: logs/augment_cht_<dataset>.log

set -e
cd "$(dirname "$0")"

source ./conf.sh

AUGMENT_CHT_EFC=${AUGMENT_CHT_EFC:-500}
AUGMENT_CHT_THREADS=${AUGMENT_CHT_THREADS:-32}

# Augmented index path (separate from original)
augmented_index="${hashann_index_file}_augcht"

echo "========================================"
echo "  Dataset: ${dataset}"
echo "  Original index: ${hashann_index_file}"
echo "  Augmented index: ${augmented_index}"
echo "  augment_cht_efc: ${AUGMENT_CHT_EFC}"
echo "========================================"

# --- Step 1: Build augmented index if not yet saved ---
if [ ! -f "$augmented_index" ]; then
    echo ""
    echo "=== Step 1: Augment index and save ==="
    python -u -c "
import sys
sys.path.insert(0, '../tests')
from hashann import HashANN
import ast, time, numpy as np

attr_type_list = ast.literal_eval('${attr_type}')
params = {'M': ${M}, 'ef_construction': ${ef_construction}, 'metric': '${metric}'.lower(), 'dim': ${dim}, 'N': ${N}, 'ef_search': 10, 'ef_top': 1}

hash_ann = HashANN()
hash_ann.init_params(params)
index = hash_ann.load_index(params, attr_type_list, '${hashann_index_file}', 1, '${algo}')

d = np.array(index.get_degrees())
print(f'Before augment: mean_deg={d.mean():.1f}')

print(f'Augmenting with efc=${AUGMENT_CHT_EFC}, threads=${AUGMENT_CHT_THREADS}...')
t0 = time.time()
index.augment_edges_cht(${AUGMENT_CHT_EFC}, ${AUGMENT_CHT_THREADS})
print(f'Augment done in {time.time()-t0:.1f}s')

d2 = np.array(index.get_degrees())
print(f'After augment: mean_deg={d2.mean():.1f}')

print(f'Saving augmented index to ${augmented_index}...')
index.save_index('${augmented_index}')
print('Saved.')
" 2>&1
    echo "=== Augmented index saved ==="
else
    echo "Augmented index already exists: ${augmented_index}"
fi

# --- Step 2: Query all selectivities ---
# Selectivity configs for attr_type=[0,1]
declare -a SEL_NAMES=("10%" "20%" "40%" "60%" "80%" "100%")
declare -a SEL_VALS=("[0.333,7]" "[0.4,5]" "[0.667,4]" "[0.75,2]" "[0.9,1]" "[1.0,0]")

LOG_FILE="logs/augment_cht_${dataset}.log"
mkdir -p logs
echo "=== augment_cht experiments: ${dataset} ===" > "$LOG_FILE"
echo "date: $(date)" >> "$LOG_FILE"
echo "augment_cht_efc: ${AUGMENT_CHT_EFC}" >> "$LOG_FILE"
echo "index: ${augmented_index}" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

for i in "${!SEL_NAMES[@]}"; do
    sel_name="${SEL_NAMES[$i]}"
    sel_val="${SEL_VALS[$i]}"

    echo ""
    echo "========================================"
    echo "  ${dataset} | ${sel_name} (${sel_val})"
    echo "========================================"

    # Derive predicate/gt paths
    query_file_prefix="arbi_0_1_${sel_val}"
    qp_file="${label_root}predicate_${query_file_prefix}.json"
    gt_file="${label_root}gt_${query_file_prefix}.json"

    if [ ! -f "$qp_file" ]; then
        echo "  SKIP: predicate not found: $qp_file"
        echo "SKIP ${sel_name}: predicate not found" >> "$LOG_FILE"
        continue
    fi
    if [ ! -f "$gt_file" ]; then
        echo "  SKIP: gt not found: $gt_file"
        echo "SKIP ${sel_name}: gt not found" >> "$LOG_FILE"
        continue
    fi

    echo "" >> "$LOG_FILE"
    echo "=== ${sel_name} (${sel_val}) ===" >> "$LOG_FILE"

    # Query using augmented index (no augment_cht needed — already baked in)
    python -u ../tests/hashann_query.py --data_path "$dataset_file" \
        --index_cache_path "$augmented_index" \
        --k $K \
        --N $N \
        --M $M \
        --dim $dim \
        --metric "${metric}" \
        --efConstruction $ef_construction \
        --ef_search "$ef_search_list" \
        --ef_top $ef_top \
        --name $algo \
        --query_path "$query_file" \
        --attr_path "$dataset_attr_file" \
        --qrange_path "$qp_file" \
        --gt_path "$gt_file" \
        --n_query_to_use $query_size \
        --attr_type_list "$attr_type" \
        --threads 1 \
        --use_ft "$use_ft" \
        2>&1 | tee -a "$LOG_FILE"

    echo "  Done: ${sel_name}"
done

echo ""
echo "All done. Results in: $LOG_FILE"
