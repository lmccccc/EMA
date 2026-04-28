source ./conf.sh

if [ -z "$dnf_spec" ] || [ -z "$dnf_name" ]; then
    echo "Error: dnf_spec and dnf_name must be set in conf.sh for DNF predicate generation."
    exit 1
fi

# if file exist, skip generation
if [ -f ${query_predicate_file} ]; then
    echo "${query_predicate_file} exists, skip DNF predicate generation."
    exit 0
fi

python ../tests/predicate_generator_dnf.py --attr_file ${dataset_attr_file} \
                              --N ${N} \
                              --query_size ${query_size} \
                              --attr_type_list ${attr_type} \
                              --dnf_spec "${dnf_spec}" \
                              --predicate_file ${query_predicate_file}
