
import argparse
import os
import sys
from utils import read_multy_attr
import struct

def sort_attr_get_order(attr):
    """
    attr: List[int]

    return:
        id2order: List[int]，原始下标 -> 排序后位置
    """
    N = len(attr)

    # (attr_value, original_id)
    p = [(attr[i], i) for i in range(N)]

    # 稳定排序（按 attr 值）
    p.sort(key=lambda x: x[0])

    id2order = [0] * N
    for order, (_, pid) in enumerate(p):
        id2order[pid] = order

    return id2order


# main

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Convert id2od2 json to bin')
    parser.add_argument('input_json', type=str, help='Input id2od2 json file')
    parser.add_argument('output_bin', type=str, help='Output id2od2 bin file')

    args = parser.parse_args()

    input_json = args.input_json
    output_bin = args.output_bin

    print(f"Reading id2od2 from {input_json}")
    data = read_multy_attr(input_json)

    # extract attr 2
    data2 = [item[1][0] for item in data]

    # id2od
    id2ord = sort_attr_get_order(data2)

    print(f"Writing id2od2 to {output_bin}")
    # write bin format 
    # size, data....
    N = len(id2ord)
    with open(output_bin, 'wb') as f:
        len_pack = struct.pack('i', int(N))
        f.write(len_pack)
        format_str = f'{N}i'
        list_pack = struct.pack(format_str, *id2ord)
        f.write(list_pack)

    print("Conversion completed successfully.")