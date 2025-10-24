source ./conf.sh


# python attr_generator.py --output_file ${dataset_attr_file} --N ${N} --attr_type_list ${attr_type} --categorical_attr_max_cardinality 5 --numerical_max_attr 100000

# python predicate_generator.py --attr_file ${dataset_attr_file} \
#                               --N ${N} \
#                               --query_size ${query_size} \
#                               --attr_type_list ${attr_type} \
#                               --query_sel ${query_sel} \
#                               --categorical_attr_max_cardinality 5 \
#                               --numerical_max_attr 100000 \
#                               --predicate_file ${query_predicate_file}

# python groundtruth_generator.py --dataset_file ${dataset_file} \
#                              --d ${dim} \
#                              --attr_file ${dataset_attr_file} \
#                              --query_file ${query_file} \
#                              --attr_type_list ${attr_type} \
#                              --N ${N} \
#                              --query_size ${query_size} \
#                              --predicate_file ${query_predicate_file} \
#                              --c_name ${dataset}_arbi_0_1_random \
#                              --mode construction \
#                              --max_cate_val 5 \
#                              --K 10 \
#                              --gt_file ${ground_truth_file}



python tests/groundtruth_generator.py --dataset_file ${dataset_file} \
                             --d ${dim} \
                             --attr_file ${dataset_attr_file} \
                             --query_file ${query_file} \
                             --attr_type_list ${attr_type} \
                             --N ${N} \
                             --query_size ${query_size} \
                             --predicate_file ${query_predicate_file} \
                             --c_name ${ground_truth_collection_name} \
                             --mode query \
                             --max_cate_val ${categorical_attr_max_cardinality} \
                             --K ${K} \
                             --gt_file ${ground_truth_file}

# python selectivity.py --d ${dim} \
#                              --attr_file ${dataset_attr_file} \
#                              --attr_type_list ${attr_type} \
#                              --N ${N} \
#                              --query_size ${query_size} \
#                              --predicate_file ${query_predicate_file} \