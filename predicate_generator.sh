source ./conf.sh

python tests/predicate_generator.py --attr_file ${dataset_attr_file} \
                              --N ${N} \
                              --query_size ${query_size} \
                              --attr_type_list ${attr_type} \
                              --num_query_sel ${num_query_sel} \
                              --cate_query_sel ${cate_query_sel} \
                              --categorical_attr_max_cardinality ${categorical_attr_max_cardinality} \
                              --numerical_max_attr ${numerical_max_attr} \
                              --predicate_file ${query_predicate_file}

