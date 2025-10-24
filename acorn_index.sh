# export debugSearchFlag=0
#! /bin/bash



# cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -B build

# make -C build -j faiss
# make -C build utils
# make -C build test_acorn

source ./conf.sh
source ./acorn_conf.sh

../code/ACORN/build/demos/acorn_build $dataset \
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