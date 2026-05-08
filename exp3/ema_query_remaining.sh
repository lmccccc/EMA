#!/bin/bash
#
# ema_query.sh — Batch HashANN query experiment with adaptive augmented edges
#
# Runs [0,1] attr_type (label AND range) queries across:
#   Datasets:      Redcaps_4M, sift10m, youtube_rgb, wiki_15_4M
#   Selectivities: 10%, 20%, 40%, 60%, 80%, 100%
#
# Results appended to logs/ema_log.txt
# Uses exp3/conf.sh for dataset paths; restores conf.sh after each run.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate conda environment with hashannlib
source /home/mocheng/anaconda3/etc/profile.d/conda.sh
conda activate py310

LOG_FILE="logs/ema_log.txt"
mkdir -p logs

# Save original conf.sh
cp conf.sh conf.sh.ema_backup

# Common settings
ATTR_TYPE="[0,1]"
EF_LIST="[1,2,3,5,8,10,12,15,20,25,30]"
USE_FT="true"
AUGMENT="true"
AUGMENT_THRESHOLD=8

datasets=(
    "sift10m"
    "youtube_rgb"
    "wiki_15_4M"
)

# sel conf for [0,1]:
#   10%: [0.333,7]   20%: [0.4,5]   40%: [0.667,4]
#   60%: [0.75,2]    80%: [0.9,1]  100%: [1.0,0]
sel_names=("10%" "20%" "40%" "60%" "80%" "100%")
sel_vals=("[0.333,7]" "[0.4,5]" "[0.667,4]" "[0.75,2]" "[0.9,1]" "[1.0,0]")

echo "========================================" >> "$LOG_FILE"
echo "EMA Batch Experiment: $(date)" >> "$LOG_FILE"
echo "Augmented edges: $AUGMENT (threshold=$AUGMENT_THRESHOLD)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

for dataset in "${datasets[@]}"; do
    echo ""
    echo "============================================="
    echo "Dataset: $dataset"
    echo "============================================="

    # Set dataset and attr_type in conf.sh to resolve paths
    sed -i "s|^dataset=.*|dataset=\"$dataset\"|" conf.sh
    sed -i "s|^attr_type=.*|attr_type=\"$ATTR_TYPE\"|" conf.sh
    sed -i "s|^ef_search_list=.*|ef_search_list=\"$EF_LIST\"|" conf.sh
    sed -i "s|^use_ft=.*|use_ft=$USE_FT|" conf.sh
    # Use first selectivity to resolve base paths
    sed -i "s|^query_sel=.*|query_sel=\"${sel_vals[0]}\"|" conf.sh

    source ./conf.sh

    if [ ! -f "$hashann_index_file" ]; then
        echo "  SKIP: index not found: $hashann_index_file"
        continue
    fi

    # Build selectivity args for the batch Python script
    SEL_NAMES_STR=""
    SEL_VALS_STR=""
    for i in "${!sel_names[@]}"; do
        sname="${sel_names[$i]}"
        sval="${sel_vals[$i]}"

        # Resolve predicate/gt paths for this selectivity
        sed -i "s|^query_sel=.*|query_sel=\"$sval\"|" conf.sh
        source ./conf.sh

        if [ ! -f "$query_predicate_file" ] || [ ! -f "$ground_truth_file" ]; then
            echo "  SKIP: missing predicate/gt for $dataset $sname"
            continue
        fi

        SEL_NAMES_STR="${SEL_NAMES_STR}${sname},"
        SEL_VALS_STR="${SEL_VALS_STR}${sval},"
    done

    # Remove trailing commas
    SEL_NAMES_STR="${SEL_NAMES_STR%,}"
    SEL_VALS_STR="${SEL_VALS_STR%,}"

    echo "  Running all selectivities for $dataset (augment_edges built once)..."

    # Run the batch query script (loads index + builds augmented edges ONCE per dataset)
    python -u ../tests/ema_batch_query.py \
        --dataset "$dataset" \
        --index_path "$hashann_index_file" \
        --data_path "$dataset_file" \
        --query_path "$query_file" \
        --attr_path "$dataset_attr_file" \
        --label_root "$label_root" \
        --attr_type "$ATTR_TYPE" \
        --N "$N" \
        --dim "$dim" \
        --M "$M" \
        --metric "$metric" \
        --ef_construction "$ef_construction" \
        --ef_list "$EF_LIST" \
        --K "$K" \
        --query_size "$query_size" \
        --use_ft "$USE_FT" \
        --augment_edges "$AUGMENT" \
        --augment_threshold "$AUGMENT_THRESHOLD" \
        --sel_names "$SEL_NAMES_STR" \
        --sel_vals "$SEL_VALS_STR" \
        --log_file "$LOG_FILE" \
        2>&1 | tee "logs/ema_${dataset}.log"

    status=$?
    if [ $status -ne 0 ]; then
        echo "  FAILED: $dataset"
    else
        echo "  OK: $dataset"
    fi
done

# Restore original conf.sh
cp conf.sh.ema_backup conf.sh
rm -f conf.sh.ema_backup

echo ""
echo "========================================" >> "$LOG_FILE"
echo "EMA Batch Experiment DONE: $(date)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"
echo ""
echo "All done. Results in: $LOG_FILE"
