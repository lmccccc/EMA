
import sys
from pymilvus import DataType, MilvusClient, connections, utility
import sys
import numpy as np
import math
import time
import argparse
import json
import ast
from utils import *

def load_query_data(query_file, qrange_file, gt_file, N, Nq, k):# fvecs, fvecs, json, json, json

    if(".fvecs" in query_file):
        queries = fvecs_read(query_file)
        print(f"ori query shape: {queries.shape}")
        assert queries.shape[0] >= Nq
        if queries.shape[0] > Nq:
            queries = queries[:Nq]
        print(f"used query shape: {queries.shape}")
    else:
        print("error: query file format not supported")
        sys.exit(-1)
    if(".json" in qrange_file):
        query_filter_ranges = read_multy_attr(qrange_file)
        #convert array into turple list
        # query_filter_ranges = [(query_filter_ranges[i], query_filter_ranges[i+1]) for i in range(0, len(query_filter_ranges), 2)]
        assert len(query_filter_ranges) >= Nq
        if len(query_filter_ranges) > Nq:
            query_filter_ranges = query_filter_ranges[:Nq]
    else:    
        print("error: query range file format not supported")
        sys.exit(-1)
    if(".json" in gt_file):
        query_gt = read_attr(gt_file)
        query_gt = query_gt.reshape(-1, k)
        assert len(query_gt) >= Nq
        if len(query_gt) > Nq:
            query_gt = query_gt[:Nq]

    else:
        print("error: groundtruth file format not supported")
        sys.exit(-1)
    # print("sorting for label")

    return queries, query_filter_ranges, query_gt

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
    parser.add_argument("--ef_construction", type=int, required=True, help="ef construction parameter for HNSW")
    parser.add_argument("--M", type=int, required=True, help="M parameter for HNSW")
    parser.add_argument("--ef_search", type=str, required=True, help="ef search list parameter for HNSW")
    args = parser.parse_args()
    return args


def drop_index(client, col_name, index_name="HNSW"):
    client.release_collection(
        collection_name=col_name
    )
    client.drop_index(
        collection_name=col_name,   # Name of the collection
        index_name=index_name  # Name of the index to drop
    )

def check_size(client, col_name, expected_size):
    res = client.query(
        collection_name=col_name,
        output_fields=["count(*)"]
    )
    actual_size = res[0]['count(*)']
    if actual_size != expected_size:
        print(f"Error: collection size mismatch, expected {expected_size}, got {actual_size}")
    else:
        print(f"Collection size check passed: {actual_size} entries")
    return actual_size == expected_size

# read args
if __name__ == "__main__":
    args = arg_init()

    attr_type_list = ast.literal_eval(args.attr_type_list)

    client = MilvusClient(
        uri="http://localhost:19530"
    )
    fields = ["id", "vector"] + [f"attr_{i}" for i in range(len(attr_type_list))]
    print("mode:", args.mode, " collection name:", args.c_name)

    # list collection
    # collections = client.list_collections()
    # print("collections: ", collections)
    # exit()

    # create collection
    if args.mode == "query" and not client.has_collection(args.c_name):
        print("error: collection ", args.c_name, " does not exist")
        exit()
    if client.has_collection(args.c_name):
        try:
            print("collection ", args.c_name, " exists. Loading collection...")

            # exit()

            # client.drop_collection(args.c_name) 
            # print("dropped existing collection ", args.c_name)
            # exit()


            # res = client.get_collection_stats(collection_name=args.c_name)
            # print("collection stats:", res)
            # client.release_collection(collection_name=args.c_name)
            # print("collection released")
            # describe index
            client.load_collection(collection_name=args.c_name, 
                                replica_number=1,
                                load_fields=fields)
            print("collection loaded for querying")

            collection_info = client.get_collection_stats(collection_name=args.c_name)
            print("collection stats:", collection_info)

            res = client.query(
                collection_name=args.c_name,
                output_fields=["count(*)"]
            )

            print("collection size:", res)
        
            res = client.describe_index(
                collection_name=args.c_name,
                index_name="HNSW"
            )
            time1 = time.time()
            print("index info:", res)
            # indexed_rows1 = res['indexed_rows']

            # time.sleep(100)
            # res = client.describe_index(
            #     collection_name=args.c_name,
            #     index_name="HNSW"
            # )

            # time2 = time.time()
            # print("index info:", res)
            # indexed_rows2 = res['indexed_rows']

            # all_row_cnt = 15435516
            # #compute expected index time
            # expected_index_time = (time2 - time1) * all_row_cnt / (indexed_rows2 - indexed_rows1)
            # print(f"Expected index time for full dataset ({all_row_cnt} rows): {expected_index_time:.2f} seconds")

            # if args.mode == "construction":
            #     print("collection exists, skip construction")
            #     exit()
        except Exception as e:
            print("error loading collection:", e)
            # client.drop_collection(args.c_name)
            # print("drop existing collection ", args.c_name)
            client.close()
            exit()


    if (args.mode == "construction"):
        # if client.has_collection(args.c_name):
        #     client.drop_collection(args.c_name)
        print("start construction")

        # if has collection check it
        if (client.has_collection(args.c_name)):
            if(check_size(client, args.c_name, args.N)):
                print("collection exists, skip add items")
        
        else:
            # create collection
            # json_data_file = args.predicate_file.replace(".json", "json_data.json")
            # if not os.path.isfile(json_data_file):
            print("reading data from files")
            dataset, attr, query, predicate = read_data(args.dataset_file, args.attr_file, args.N, args.d, args.query_file, args.predicate_file, args.query_size)

            # label_set = set(attr)
            # label_cnt = len(label_set)
            # partition_size = min(64, label_cnt)

            # partition_attr_type = attr_type_list[-1]
            # if partition_attr_type == 0:
            #     partition_size = 64
            # elif partition_attr_type == 1:
            #     partition_size = min(64, args.max_cate_val+1)
            # else:
            #     print("error: unsupported partition attribute type ", partition_attr_type)
            #     exit()
            partition_size = -1
            partition_idx = -1
            for i in range(len(attr_type_list)):
                partition_attr_type = attr_type_list[i]
                if partition_attr_type == 0:
                    partition_size = min(64, args.max_cate_val+1)
                    partition_idx = i
                    break
                elif partition_attr_type == 1:
                    continue
                else:
                    print("error: unsupported partition attribute type ", partition_attr_type)
                    exit()
            print("partition size:", partition_size)
            if (partition_size == -1):
                print("categorical multi-attribute, do not support partitioning")
            # create collection
            if (partition_size > 0):
                partition_attr = f"attr_{partition_idx}"
                schema = MilvusClient.create_schema(
                    auto_id=False,
                    partition_key_field=partition_attr,      # default partition=64
                    num_partitions=partition_size
                )
            else:
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
            print("converting data to json format for insertion")
            data = [
                {"id": i, 
                "vector": dataset[i].tolist(), 
                **{f"attr_{j}": attr[i][j][0] if attr_type_list[j] == 0 else attr[i][j] for j in range(len(attr[i]))}}
                for i in range(args.N)
            ]

            # if not os.path.isfile(json_data_file):
            #     data = [
            #         {"id": i, 
            #         "vector": dataset[i].tolist(), 
            #         **{f"attr_{j}": attr[i][j][0] if attr_type_list[j] == 0 else attr[i][j] for j in range(len(attr[i]))}}
            #         for i in range(args.N)
            #     ]
            #     # write json data
            #     with open(json_data_file, 'w') as file:
            #         json.dump(data, file)
            #     print("json data file saved to ", json_data_file)
            # else:
            #     # load json data file
            #     with open(json_data_file, 'r') as file:
            #         data = json.load(file)
            #     print("json data file loaded from ", json_data_file)

        # check index information
        collection_info = client.get_collection_stats(collection_name=args.c_name)
        print("collection stats:", collection_info)
        if (collection_info["row_count"] == args.N):
            print("collection size matches expected ", args.N)
            print("skip insertion")
        if (collection_info["row_count"] == 0):
            print("collection is empty, proceed to insertion")
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
            
        else:
            print("collection size:", collection_info["row_count"], " expected:", args.N)
            # client.drop_collection(args.c_name)
            # print("drop existing collection ", args.c_name)
            exit()


        if ("index_type" in collection_info):
            if (collection_info["row_count"] != args.N):
                print("collection size mismatch, expected ", args.N, " got ", collection_info["row_count"])
                # client.drop_collection(args.c_name)
                client.close()
                exit()
            if (collection_info["pending_index_rows"] != 0 or collection_info["indexed_rows"] != args.N):
                print("index not ready, pending_index_rows:", collection_info["pending_index_rows"], " indexed_rows:", collection_info["indexed_rows"])
                # drop_index(client, args.c_name)
                client.close()
                exit()
            if (collection_info["state"] != "Finished"):
                print("index state not finished, state:", collection_info["state"])
                # drop_index(client, args.c_name)
                client.close()
                exit()
        else:
            print("no index info found, proceed to construction")
            t1 = time.time()
            index_params = client.prepare_index_params()
            index_params.add_index(
                field_name="vector", 
                index_type="HNSW",# IVF_FLAT IVF_PQ IVF_SQ8 HNSW SCANN
                metric_type=args.metric,
                index_name="HNSW",
                params={ "M": args.M, "efConstruction": args.ef_construction }, # see https://milvus.io/docs/configure_querynode.md#queryNodesegcoreinterimIndexnlist
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
            print("construction suc, time cost:", t2-t1)
            print("construction done")
            client.close()
            exit()

    # query -------------------------
    queries, raw_predicate, query_gt = load_query_data(args.query_file, args.predicate_file, args.gt_file, args.N, args.query_size, args.K)
    

    efs_list = ast.literal_eval(args.ef_search)
    result = []
    for efs in efs_list:
        ids = []
        q_t = 0
        hist = np.array([0 for _ in range(11)], dtype='float')
        total_size = args.N * args.K
        positive_size = 0
        time_one_batch = 0
        for i in range(args.query_size):
            # if(i % 100 == 0):
            #     print("batch ", i)
            q_cnt = 0
            exp = ""
            for j in range(len(attr_type_list)):
                if attr_type_list[j] == 0:
                    if j != 0:
                        exp += " and "
                    exp += f"{raw_predicate[i][j][0]} <= attr_{j} <= {raw_predicate[i][j][1]}"
                else:
                    for k in range(len(raw_predicate[i][j])):
                        if j != 0 or k != 0:
                            exp += " and "
                        exp += f"array_contains(attr_{j}, {raw_predicate[i][j][k]})"
            # qr = [i for i in range(qrange[i][0], qrange[i][1] + 1)]
            # exp = "label in " + str(qr)
            s_params = {"metric_type": args.metric, "params": {"ef": efs}}
            start = time.time()
            res = client.search(
                                collection_name=args.c_name,
                                data=[queries[i].tolist()], 
                                filter=exp,
                                limit=args.K,
                                search_params=s_params, 
                                group_strict_size=True,
                                )
            end = time.time()
            time_one_batch += end - start
            res_id = [x['id'] for x in res[0]]
            ids.append(res_id)
        qps = args.query_size/time_one_batch
        print(f"Query time: {time_one_batch} seconds, QPS:{qps}")

        # recall
        correct_sum = 0
        recall_list = []
        for i in range(args.query_size):
            # print("predicate:", raw_predicate[i])
            # print("result:", ids[i])
            # print("ground truth:", _query_gt[i])
            gt = query_gt[i]
            res = ids[i]
            if len(gt) != len(res):
                print(f"Error: ground truth and label length mismatch at query {i}, gt: {len(gt)}, label: {len(res)}")
                continue
            correct = np.isin(gt, res)
            correct_sum += np.sum(correct)
            recall_list.append(np.sum(correct)/len(gt))
        recall = correct_sum / (args.query_size * args.K)
        print(f"ef search: {efs}, recall: {recall:.4f}")
        result.append([efs, recall, qps])
        if recall >= 0.99:
            break

    # total_size = args.query_size * args.K
    print("Final results (ef_search, recall, QPS):")
    for res in result:
        print(res)
    client.close()
    exit()



        