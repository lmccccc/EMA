#!/usr/bin/env bash
# conf.sh — resolve dataset paths and index file for EMA experiments.
#
# Inputs (env vars; sensible defaults provided):
#   dataset                 sift10m | youtube_rgb | wiki_15_4M | Redcaps_4M
#   DATA_ROOT               base directory containing all datasets
#   M, ef_construction, K, threads, ef_search_list
#   ft_bits, edge_level_ft, use_ft, ft_routing_min_deg
#   dnf_spec, dnf_name      DNF predicate spec + short name (see selectivity_specs.sh)
#
# Outputs (set via `source ./conf.sh`):
#   N, dim, metric, dataset_file, query_file, dataset_attr_file
#   query_predicate_file, ground_truth_file
#   hashann_index_file, hashann_index_root, label_root

DATA_ROOT="${DATA_ROOT:-/mnt/data/mocheng/dataset}"

# ---- algorithm defaults --------------------------------------------------
dataset="${dataset:-Redcaps_4M}"
attr_type="${attr_type:-[0,1]}"
distribution_type="${distribution_type:-random}"
M="${M:-40}"
ef_construction="${ef_construction:-300}"
ef_search_list="${ef_search_list:-[10,20,40,80,150,300]}"
K="${K:-10}"
threads="${threads:-32}"
ft_bits="${ft_bits:-128}"
edge_level_ft="${edge_level_ft:-true}"
use_ft="${use_ft:-true}"
ft_routing_min_deg="${ft_routing_min_deg:-16}"

# Attribute generation defaults (used by attr_generator.sh).
categorical_attr_max_cardinality="${categorical_attr_max_cardinality:-21}"
numerical_max_attr="${numerical_max_attr:-100000}"

# DNF predicate (paper uses OR-of-AND label predicates).
# Defaults to the 10% selectivity spec on attr_type=[0,1]; override per-cell.
# Note: assigned in two steps because bash's ${var:-default} ends at the
# first '}' it sees, which would mangle a JSON literal default.
: "${dnf_spec:=}"
if [ -z "$dnf_spec" ]; then
    dnf_spec='[{"0":0.3,"1":[9]},{"1":[12]}]'
fi
dnf_name="${dnf_name:-or_T10}"

# ---- attribute layout ----------------------------------------------------
case "$attr_type" in
    "[0]")   attr_index_type="arbi_0_${distribution_type}" ;;
    "[1]")   attr_index_type="arbi_1_${distribution_type}" ;;
    "[0,1]") attr_index_type="arbi_0_1_${distribution_type}" ;;
    "[0,0]") attr_index_type="arbi_0_0_${distribution_type}" ;;
    *)       attr_index_type="arbi_${distribution_type}" ;;
esac

# ---- per-dataset paths ---------------------------------------------------
case "$dataset" in
    sift10m)
        dim=128
        N=10000000
        query_size=1000
        metric="L2"
        index_root="${DATA_ROOT}/sift10m"
        dataset_file="${index_root}/sift10m.fvecs"
        query_file="${index_root}/sift10m_query.fvecs"
        ;;
    youtube_rgb)
        dim=1024
        N=1000000
        query_size=100
        metric="IP"
        index_root="${DATA_ROOT}/youtube1m"
        dataset_file="${index_root}/rgb.fvecs"
        query_file="${index_root}/rgb_query.fvecs"
        ;;
    wiki_15_4M)
        dim=1024
        N=15435516
        query_size=1000
        metric="IP"
        index_root="${DATA_ROOT}/navix_dataset/wiki_15.4M"
        dataset_file="${index_root}/fvecs/wiki_15.4M.fvecs"
        query_file="${index_root}/fvecs/wiki_15.4M_query.fvecs"
        ;;
    Redcaps_4M)
        dim=512
        N=4000000
        query_size=1000
        metric="IP"
        index_root="${DATA_ROOT}/redcaps4m"
        dataset_file="${index_root}/image_embeddings.fvecs"
        query_file="${index_root}/query.fvecs"
        ;;
    *)
        echo "[conf.sh] Unknown dataset: $dataset" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac

# ---- predicate / GT paths -----------------------------------------------
label_root="${index_root}/label/${attr_index_type}/"
dataset_attr_file="${label_root}attr_${attr_index_type}.json"
query_predicate_file="${label_root}predicate_dnf_${dnf_name}.json"
ground_truth_file="${label_root}gt_dnf_${dnf_name}.json"
if [ "$K" -ne 10 ]; then
    ground_truth_file="${ground_truth_file%.json}_k${K}.json"
fi

# ---- index file ----------------------------------------------------------
hashann_root="${index_root}/hashann/"
hashann_index_root="${hashann_root}index/"
edge_ft_suffix=""
[ "$edge_level_ft" = "true" ] && edge_ft_suffix="_edgeFT"
numeric_marker_suffix=""
case "$attr_type" in
    *0*) numeric_marker_suffix="_nb2" ;;
esac
hashann_index_file="${hashann_index_root}index_${M}_${ef_construction}_${attr_index_type}_${ft_bits}${edge_ft_suffix}${numeric_marker_suffix}_mo2_do1"

mkdir -p "$hashann_index_root" "$label_root"
