#!/usr/bin/env bash
# predicate_generator.sh — produce a DNF predicate JSON for the current cell.
# Reads conf.sh; expects dnf_spec / dnf_name set.
set -euo pipefail
cd "$(dirname "$0")"

source ./env.sh
source ./conf.sh

if [ -z "${dnf_spec:-}" ] || [ -z "${dnf_name:-}" ]; then
    echo "[predicate_generator] dnf_spec / dnf_name must be set" >&2
    exit 1
fi

if [ -f "$query_predicate_file" ]; then
    echo "[predicate_generator] $query_predicate_file exists, skip"
    exit 0
fi

python -u ../tests/predicate_generator_dnf.py \
    --attr_file       "$dataset_attr_file" \
    --N               "$N" \
    --query_size      "$query_size" \
    --attr_type_list  "$attr_type" \
    --dnf_spec        "$dnf_spec" \
    --predicate_file  "$query_predicate_file"
