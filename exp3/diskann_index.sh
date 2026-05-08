#! /bin/bash

source ./conf.sh
source ./diskann_conf.sh


        
echo "dataset file: $dataset_bin_file" 
echo "index path prefix: $diskann_index_prefix"
echo "label file: $label_file"

../../code/DiskANN/build/apps/build_memory_index  --data_type float \
                                                --dist_fn $metric \
                                                --data_path $dataset_bin_file \
                                                --index_path_prefix $diskann_index_prefix \
                                                -R $M \
                                                --alpha $alpha \
                                                --Lbuild $ef_construction \
                                                --label_file $label_file \
                                                -T $threads \
                                                2>&1 | tee -a logs/diskann_construction.log