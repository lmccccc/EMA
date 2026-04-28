
source ./conf.sh
source ./irange_conf.sh

echo "dataset bin file: $dataset_bin_file"
echo "query bin file: $query_bin_file"
echo "attr bin file: $attr_bin_file"
echo "qrange bin file: $predicate_bin_file"
echo "ground truth bin file: $ground_truth_bin_file"
echo "index file: $irange_index_file"
echo "result file: $irange_result_file"
echo "M: $M"
echo "id2od file: $irange_id2od_file"
echo "N: $N"
echo "Nq: $query_size"
echo "K: $K"
echo "ef_search: $ef_search_list"
echo "metric: $metric"


# echo "dataset: $dataset" >> logs.txt
# echo "attr: $attr_type" >> logs.txt
# echo "sel: $query_sel" >> logs.txt
# echo "algo: irange" >> logs.txt

../../code/iRangeGraph/build/tests/search --data_path $dataset_bin_file\
                                --query_path $query_bin_file \
                                --attr_file $attr_bin_file \
                                --range_saveprefix $predicate_bin_file \
                                --groundtruth_saveprefix $ground_truth_bin_file \
                                --index_file $irange_index_file \
                                --result_saveprefix $irange_result_file \
                                --M $M \
                                --id2od_file $irange_id2od_file \
                                --N $N \
                                --Nq $query_size \
                                --K $K \
                                --ef_search $ef_search_list \
                                --metric $metric  \
                                2>&1 | tee logs/irange.log

python ../tests/extract_results.py "logs/irange.log" $dataset $attr_type $query_sel $M "irange" $K $query_predicate_file