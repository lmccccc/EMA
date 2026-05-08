#used to convert json attributions into typical format for DiskANN

import numpy as np
import sys
from utils import check_file, check_dir, ivecs_read
import struct
import json


if __name__ == "__main__":
    if not len(sys.argv) == 3 :
        print("error wrong argument size:", len(sys.argv))
        sys.exit(-1)
    else:
        qrange_file = sys.argv[1]
        print("input file:", qrange_file)
        check_file(qrange_file)

        output_qrange_file = sys.argv[2]
        print("output file:", output_qrange_file)
        check_dir(output_qrange_file)
        

    #json: [1,2,3]
    #txt:  1\n2\n3

    #read
    with open(qrange_file, 'r') as file:
        data = json.load(file)
    assert(isinstance(data, list))
    assert(len(data) > 0)

    # data = [[[val],[val]], [[val],[val]], ...]

    res = []
    for item in data:
        assert(isinstance(item, list))
        assert(len(item) == 2)
        a = int(item[0][0])
        b = int(item[1][0])
        res.append(a)
        res.append(b)

    assert(len(res) == 2 * len(data))

    #write
    with open(output_qrange_file, mode='wb') as file:
        format_str = f'{len(res)}i'
        list_pack = struct.pack(format_str, *res)
        file.write(list_pack)

    print("succeed to transport ", qrange_file, " into ", output_qrange_file, " for iRangeGraph query range file")
