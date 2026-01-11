source ./conf.sh

# if file exist, skip generation
if [ -f ${query_predicate_file} ]; then
    echo "${query_predicate_file} exists, skip predicate generation."
    exit 0
fi

python ../tests/predicate_generator.py --attr_file ${dataset_attr_file} \
                              --N ${N} \
                              --query_size ${query_size} \
                              --attr_type_list ${attr_type} \
                              --query_sel ${query_sel} \
                              --categorical_attr_max_cardinality ${categorical_attr_max_cardinality} \
                              --numerical_max_attr ${numerical_max_attr} \
                              --predicate_file ${query_predicate_file}

