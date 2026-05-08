#used to convert json attributions into typical format for DiskANN

import json
import numpy as np
import sys
from utils import check_file, check_dir
import struct


if __name__ == "__main__":
    if not len(sys.argv) == 4 :
        print("error wrong argument")
        sys.exit(-1)
    else:
        attr_file = sys.argv[1]
        print("gt file:", attr_file)
        check_file(attr_file)

        output_attr_new_file = sys.argv[2]
        print("output file:", output_attr_new_file)
        check_dir(output_attr_new_file)
        
        topk = int(sys.argv[3])
        print("top k:", topk)

    #json: [1,2,3]
    #txt:  1\n2\n3

    #read
    with open(attr_file, 'r') as file:
        data = json.load(file)
    assert(isinstance(data, list))
    assert(len(data) > 0)
    assert(len(data[0]) == topk)

    new_data = []
    print("data size: ", len(data), ", topk: ", topk, " data[0] size: ", len(data[0]))
    for item in data:
        new_data.extend(item)
    print("total number of attr entries: ", len(new_data))
    #write
    with open(output_attr_new_file, mode='wb') as file:
        N = len(data)
        len_pack = struct.pack('i', int(N))
        file.write(len_pack)
        size_pack = struct.pack('i', topk)
        file.write(size_pack)
        format_str = f'{len(new_data)}i'
        list_pack = struct.pack(format_str, *new_data)
        file.write(list_pack)

    print("succeed to transport ", attr_file, " into ", output_attr_new_file, " for DiskANN groundtruth file")
