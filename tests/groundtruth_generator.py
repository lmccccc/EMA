
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
    # assert(query.shape[0] == Nq)
    assert(query.shape[0] >= Nq)
    if query.shape[0] > Nq:
        query = query[:Nq]
    assert(query.shape[1] == d)

    predicate = read_multy_attr(predicate_file)
    assert(len(predicate) >= Nq)
    if len(predicate) > Nq:
        predicate = predicate[:Nq]

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
    parser.add_argument("--metric", type=str, default="L2", help="Distance metric, L2 or IP")
    args = parser.parse_args()
    return args


# read args
if __name__ == "__main__":
    args = arg_init()

    attr_type_list = ast.literal_eval(args.attr_type_list)

    client = MilvusClient(uri="http://localhost:19530")
    fields = ["id", "vector"] + [f"attr_{i}" for i in range(len(attr_type_list))]
    print("mode:", args.mode, " collection name:", args.c_name)
    # create collection
    if client.has_collection(args.c_name):
        try:
            print("collection ", args.c_name, " exists")
            # client.drop_collection(args.c_name)
            # print("drop existing collection ", args.c_name)
            # exit()
            # client.release_collection(args.c_name)
            # print("release collection ", args.c_name)
            # exit()
            # client.drop_index(args.c_name, "FLAT")
            # print("drop existing index FLAT")
            # exit()
            client.load_collection(collection_name=args.c_name, 
                                replica_number=1,
                                load_fields=fields)

            res = client.query(
                collection_name=args.c_name,
                output_fields=["count(*)"]
            )

            print("collection size:", res)
            if args.mode == "construction":
                print("collection already exists, exit")

                # # drop 
                # client.drop_collection(args.c_name)
                # print("drop existing collection ", args.c_name)
                # exit()
        except Exception as e:
            print("error loading collection:", e)
            # client.drop_collection(args.c_name)
            # print("drop existing collection ", args.c_name)
            client.close()
            exit()


    elif ((not client.has_collection(args.c_name)) or args.mode == "construction"):
        # if client.has_collection(args.c_name):
        #     client.drop_collection(args.c_name)
        json_data_file = args.predicate_file.replace(".json", "json_data.json")
        if not os.path.isfile(json_data_file):
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
        

        if not os.path.isfile(json_data_file):
            data = [
                {"id": i, 
                "vector": dataset[i].tolist(), 
                **{f"attr_{j}": attr[i][j][0] if attr_type_list[j] == 0 else attr[i][j] for j in range(len(attr[i]))}}
                for i in range(args.N)
            ]
            # write json data
            with open(json_data_file, 'w') as file:
                json.dump(data, file)
            print("json data file saved to ", json_data_file)
        else:
            # load json data file
            with open(json_data_file, 'r') as file:
                data = json.load(file)
            print("json data file loaded from ", json_data_file)


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
            metric_type=args.metric,
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
            client.close()
            exit()

    dataset, attr, query, predicate = read_data(args.dataset_file, args.attr_file, args.N, args.d, args.query_file, args.predicate_file, args.query_size)

    # Detect DNF format: predicate[query][term][attr] vs predicate[query][attr]
    dnf_mode = False
    if predicate and predicate[0] and isinstance(predicate[0][0], list) \
       and len(predicate[0][0]) > 0 and isinstance(predicate[0][0][0], list):
        dnf_mode = True
        print("DNF predicate format detected")

    def build_term_expr(term, attr_type_list):
        """Build Milvus filter expression for a single AND term."""
        parts = []
        for j in range(len(attr_type_list)):
            vals = term[j]
            if not vals:
                continue
            if attr_type_list[j] == 0:
                parts.append(f"{vals[0]} <= attr_{j} <= {vals[1]}")
            else:
                for label in vals:
                    parts.append(f"array_contains(attr_{j}, {label})")
        return " and ".join(parts) if parts else ""

    def build_query_expr(pred_i, attr_type_list, is_dnf):
        """Build Milvus filter expression for one query."""
        if is_dnf:
            term_exprs = []
            for term in pred_i:
                t_expr = build_term_expr(term, attr_type_list)
                if t_expr:
                    term_exprs.append(f"({t_expr})")
            return " or ".join(term_exprs) if term_exprs else ""
        else:
            return build_term_expr(pred_i, attr_type_list)

    ids = []
    q_t = 0
    hist = np.array([0 for _ in range(11)], dtype='float')
    total_size = args.N * args.K
    positive_size = 0
    for i in range(args.query_size):
        q_cnt = 0
        exp = build_query_expr(predicate[i], attr_type_list, dnf_mode)
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
    client.close()


        