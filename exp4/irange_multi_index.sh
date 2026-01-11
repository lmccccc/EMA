
source ./conf.sh
source ./irange_conf.sh


echo "construct index"
echo "N: $N"
echo "data_path: $dataset_bin_file"
echo "index_file: $irange_index_file"
echo "attr_file: $attr_bin_file"
echo "id2od_file: $irange_id2od_file"
echo "M: $M"
echo "ef_construction: $ef_construction"
echo "threads: $threads"
if [ -e $irange_index_file ]; then
    echo "irange index already exist"
    exit 0
fi
./../code/iRangeGraph/build/tests/buildindex --N $N \
                                        --data_path $dataset_bin_file \
                                        --index_file $irange_index_file \
                                        --attr_file $attr_bin_file \
                                        --id2od_file $irange_id2od_file  \
                                        --M $M \
                                        --ef_construction $ef_construction \
                                        --threads $threads \
                                        --metric $metric  \
                                        2>&1 | tee -a logs/irange_construction.log  


