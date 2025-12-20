# export debugSearchFlag=0
#! /bin/bash



# cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -B build

# make -C build -j faiss
# make -C build utils
# make -C build test_acorn

source ./conf.sh
source ./navix_conf.sh

../code/faiss-navix/build/demos/navix_build $dataset \
                            $N \
                            $M \
                            $K \
                            $threads \
                            $dataset_file \
                            $dataset_attr_file \
                            $navix_index_file \
                            $dim \
                            $metric 