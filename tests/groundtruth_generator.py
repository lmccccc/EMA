
import sys
from pymilvus import DataType, MilvusClient
import sys
import numpy as np
import math
import time
import argparse
import json
import ast
from utils import *

def read_data(data_file, attr_file, N, d, query_file, predicate_file, Nq):
    dataset = fvecs_read(data_file)
    assert(dataset.shape[0] == N)
    assert(dataset.shape[1] == d)

    attr = read_multy_attr(attr_file)
    assert(len(attr) == N)

    query = fvecs_read(query_file)
    assert(query.shape[0] == Nq)
    assert(query.shape[1] == d)

    predicate = read_multy_attr(predicate_file)
    assert(len(predicate) == Nq)

    return dataset, attr, query, predicate

def arg_init():
    parser = argparse.ArgumentParser(description="Index parameters")
    parser.add_argument("--dataset_file", type=str, help="Dataset file path", required=True)
    parser.add_argument("--d", type=int, help="Dimension of data", required=True)
    parser.add_argument("--attr_file", type=str, help="Output attribute file path", required=True)
    parser.add_argument("--attr_type_list", type=str, required=True, help="List of attribute types, 0 for numerical, 1 for categorical")
    parser.add_argument("--N", type=int, required=True, help="Number of data points")
    parser.add_argument("--query_size", type=int, default=100, help="Number of queries", required=True)
    parser.add_argument("--predicate_file", type=str, help="Output predicate file path", required=True)
    parser.add_argument("--query_file", type=str, help="Query file path", required=True)
    parser.add_argument("--c_name", type=str, help="Collection name", required=False)
    parser.add_argument("--mode", type=str, default="query", help="Dimension of data")
    parser.add_argument("--max_cate_val", type=int, default=5, help="Max cardinality for categorical attributes")
    parser.add_argument("--K", type=int, default=10, help="Top K")
    parser.add_argument("--gt_file", type=str, help="Output groundtruth file", required=True)
    args = parser.parse_args()
    return args


# read args
if __name__ == "__main__":
    args = arg_init()

    attr_type_list = ast.literal_eval(args.attr_type_list)

    client = MilvusClient(
        uri="http://localhost:19530"
    )
    fields = ["id", "vector"] + [f"attr_{i}" for i in range(len(attr_type_list))]

    # create collection
    if client.has_collection(args.c_name):
        print("collection ", args.c_name, " exists")

        client.load_collection(collection_name=args.c_name, 
                               replica_number=1,
                               load_fields=fields)
        res = client.query(
            collection_name=args.c_name,
            output_fields=["count(*)"]
        )

        print("collection size:", res)


    elif ((not client.has_collection(args.c_name)) or args.mode == "construction"):
        if client.has_collection(args.c_name):
            client.drop_collection(args.c_name)

        # load data
        # dataset = read_file(args.dataset_file)
        # N, _d = dataset.shape
        # assert(_d == args.d)
        # print("get dataset size:", N, " d:", _d)
        # print("dim:", args.d)

        # attr = read_attr(args.attr_file)
        # _N = len(attr)
        # print("_N:", _N, " N:", N)
        # assert(_N == N)
        dataset, attr, query, predicate = read_data(args.dataset_file, args.attr_file, args.N, args.d, args.query_file, args.predicate_file, args.query_size)


        # create collection
        schema = MilvusClient.create_schema(
            auto_id=False
        )
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=args.d)
        # add multi_attr
        for i, attr_type in enumerate(attr_type_list):
            if attr_type == 0:
                schema.add_field(field_name=f"attr_{i}", datatype=DataType.INT64)
            else:
                schema.add_field(field_name=f"attr_{i}",
                                datatype=DataType.ARRAY,
                                element_type=DataType.INT64,  # array data type
                                max_capacity=args.max_cate_val+1)  # array max length [0, max_cate_val]
        client.create_collection(collection_name=args.c_name, 
                                    schema=schema)
        

        data = [
            {"id": i, 
             "vector": dataset[i].tolist(), 
             **{f"attr_{j}": attr[i][j][0] if attr_type_list[j] == 0 else attr[i][j] for j in range(len(attr[i]))}}
            for i in range(args.N)
        ]

        # insert data
        print("start insertion")
        # set start time
        t0 = time.time()
        max_message_size = 67108864
        max_batch_size = max_message_size // (args.d * 4)
        batch_size = max_batch_size // 2  # Divide by 2 to be safe
        print("batch size:", batch_size, " total batch:", round(args.N/batch_size))
        for i in range(0, args.N, batch_size):
            client.insert(collection_name=args.c_name, data=data[i:min(i+batch_size, args.N)])
            print("insert batch:", i/batch_size, " of ", round(args.N/batch_size), " from ", i, " to ", min(i+batch_size, args.N))
        # client.insert(collection_name=c_name, data=data)
        # print("insert res:", res)
        
        client.flush(collection_name=args.c_name)

        t1 = time.time()
        print("insertion time:", t1-t0)
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="vector", 
            index_type="FLAT",# IVF_FLAT IVF_PQ IVF_SQ8 HNSW SCANN
            metric_type="L2",
            index_name="FLAT",
            params={}, # see https://milvus.io/docs/configure_querynode.md#queryNodesegcoreinterimIndexnlist
            sync=True
        )
        # print("creating index")
        client.create_index(
            collection_name=args.c_name,
            index_params=index_params,
            sync=True # Whether to wait for index creation to complete before returning. Defaults to True.
        )
        t2 = time.time()


        client.load_collection(collection_name=args.c_name, 
                               replica_number=1,
                               load_fields=fields)
        res = client.query(
            collection_name=args.c_name,
            output_fields=["count(*)"]
        )

        print("collection size:", res)
        # print("create index suc, time cost:", t2-t1)
        print("insert suc, time cost:", t2-t1)
        if args.mode == "construction":
            print("construction done")
            exit()

    dataset, attr, query, predicate = read_data(args.dataset_file, args.attr_file, args.N, args.d, args.query_file, args.predicate_file, args.query_size)


    ids = []
    q_t = 0
    hist = np.array([0 for _ in range(11)], dtype='float')
    total_size = args.N * args.K
    positive_size = 0
    for i in range(args.query_size):
        # if(i % 100 == 0):
        #     print("batch ", i)
        q_cnt = 0
        exp = ""
        for j in range(len(attr_type_list)):
            if attr_type_list[j] == 0:
                if j != 0:
                    exp += " and "
                exp += f"{predicate[i][j][0]} <= attr_{j} <= {predicate[i][j][1]}"
            else:
                for k in range(len(predicate[i][j])):
                    if j != 0 or k != 0:
                        exp += " and "
                    exp += f"array_contains(attr_{j}, {predicate[i][j][k]})"
        # qr = [i for i in range(qrange[i][0], qrange[i][1] + 1)]
        # exp = "label in " + str(qr)
        t0 = time.time()
        res = client.search(
                            collection_name=args.c_name,
                            data=[query[i].tolist()], 
                            filter=exp,
                            limit=args.K,
                            group_strict_size=True,
                            )
        t1 = time.time()
        q_t += t1 - t0
        res_id = [x['id'] for x in res[0]]
        ids.append(res_id)

    total_size = args.query_size * args.K

    # write to json file
    with open(args.gt_file, 'w') as file:
        json.dump(ids, file)
    print("groundtruth file saved to ", args.gt_file)


        