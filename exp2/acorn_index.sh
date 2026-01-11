# export debugSearchFlag=0
#! /bin/bash



# cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -B build

# make -C build -j faiss
# make -C build utils
# make -C build test_acorn

source ./conf.sh
source ./acorn_conf.sh

# check acorn_index_file exists

if [ -f "$acorn_index_file" ]; then
    echo "Acorn index file $acorn_index_file exists. Skipping construction."
    exit 0
fi


echo "====================================================" >> logs/acorn_construction.log
echo "dataset: $dataset" >> logs/acorn_construction.log
echo "attr: $attr_type" >> logs/acorn_construction.log
echo "algo: acorn" >> logs/acorn_construction.log



../../code/ACORN/build/demos/acorn_build $dataset \
                            $N \
                            $gamma \
                            $M \
                            $M_beta \
                            $K \
                            $threads \
                            $dataset_file \
                            $dataset_attr_file \
                            $acorn_index_file \
                            $dim \
                            $metric \
                            2>&1 | tee -a logs/acorn_construction.log

echo "\n" >> logs/acorn_construction.log