source ./conf.sh



python tests/groundtruth_generator.py --dataset_file ${dataset_file} \
                             --d ${dim} \
                             --attr_file ${dataset_attr_file} \
                             --query_file ${query_file} \
                             --attr_type_list ${attr_type} \
                             --N ${N} \
                             --query_size ${query_size} \
                             --predicate_file ${query_predicate_file} \
                             --c_name ${ground_truth_collection_name} \
                             --mode construction \
                             --max_cate_val ${categorical_attr_max_cardinality} \
                             --K ${K} \
                             --gt_file ${ground_truth_file}
