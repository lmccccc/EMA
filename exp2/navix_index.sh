# export debugSearchFlag=0
#! /bin/bash



# cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -B build

# make -C build -j faiss
# make -C build utils
# make -C build test_acorn

source ./conf.sh
source ./navix_conf.sh

# check navix_index_file exists
if [ -f "$navix_index_file" ]; then
    echo "Navix index file $navix_index_file exists. Skipping construction."
    exit 0
fi

echo "====================================================" >> logs/navix_construction.log
echo "dataset: $dataset" >> logs/navix_construction.log
echo "attr: $attr_type" >> logs/navix_construction.log
echo "algo: navix" >> logs/navix_construction.log

../../code/faiss-navix/build/demos/navix_build $dataset \
                            $N \
                            $M \
                            $K \
                            $threads \
                            $dataset_file \
                            $dataset_attr_file \
                            $navix_index_file \
                            $dim \
                            $metric  \
                            2>&1 | tee -a logs/navix_construction.log   

echo "\n" >> logs/navix_construction.log