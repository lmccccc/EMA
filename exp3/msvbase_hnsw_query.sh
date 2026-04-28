source ./conf.sh
source ./msvbase_conf.sh

# echo "dataset: $dataset" >> logs.txt
# echo "attr: $attr_type" >> logs.txt
# echo "sel: $query_sel" >> logs.txt
# echo "algo: msvbase" >> logs.txt

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
                             --mode "query" \
                             --max_cate_val ${categorical_attr_max_cardinality} \
                             --K ${K} \
                             --gt_file ${ground_truth_file} \
                             --metric ${metric} \
                            2>&1 | tee logs/msvbase.log


python ../tests/extract_results.py "logs/msvbase.log" $dataset $attr_type $query_sel $M "msvbase" $K $query_predicate_file