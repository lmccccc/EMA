source ./conf.sh

if [ -z "$dnf_spec" ] || [ -z "$dnf_name" ]; then
    echo "Error: dnf_spec and dnf_name must be set in conf.sh for DNF ground truth generation."
    exit 1
fi

if [ -f ${ground_truth_file} ]; then
    echo "${ground_truth_file} exists, skip DNF ground truth generation."
    exit 0
fi

# C++ brute-force GT generator (fast, no Milvus required)
FAISS_ROOT="/home/mocheng/code/faiss"
GT_BIN="${FAISS_ROOT}/build/demos/generate_groundtruth_arbi"
if [ ! -f "${GT_BIN}" ]; then
    echo "Error: GT binary not found at ${GT_BIN}"
    echo "Build it: cd ${FAISS_ROOT} && make -C build generate_groundtruth_arbi"
    exit 1
fi

# Convert attr_type from JSON array "[0,1]" to comma-separated "0,1"
attr_type_csv=$(echo "${attr_type}" | tr -d '[] ')

${GT_BIN} \
    ${N} \
    ${dataset_file} \
    ${dataset_attr_file} \
    ${query_file} \
    ${query_predicate_file} \
    ${ground_truth_file} \
    ${K} \
    ${dim} \
    ${threads} \
    "${attr_type_csv}"
