source ./conf.sh
source ./msvbase_conf.sh


echo "====================================================" >> logs/msvbase_construction.log
echo "dataset: $dataset" >> logs/msvbase_construction.log
echo "attr: $attr_type" >> logs/msvbase_construction.log
echo "algo: msvbase" >> logs/msvbase_construction.log

python -u ../tests/msvbase.py --dataset_file ${dataset_file} \
                             --d ${dim} \
                             --attr_file ${dataset_attr_file} \
                             --query_file ${query_file} \
                             --attr_type_list ${attr_type} \
                             --N ${N} \
                             --query_size ${query_size} \
                             --predicate_file ${query_predicate_file} \
                             --schema_name ${vbase_schema_name} \
                             --table_name ${vbase_table_name} \
                             --table_file ${vbase_dataset_file} \
                             --mode "construction" \
                             --max_cate_val ${categorical_attr_max_cardinality} \
                             --K ${K} \
                             --gt_file ${ground_truth_file} \
                             --metric ${metric} \
                            2>&1 | tee -a logs/msvbase_construction.log

echo "\n" >> logs/msvbase_construction.log