import psycopg2
from psycopg2 import sql
import sys
import sys
import numpy as np
import math
import time
import argparse
import json
import ast
from utils import *
import os

def load_query_data(query_file, qrange_file, gt_file, N, Nq, k):# fvecs, fvecs, json, json, json

    if(".fvecs" in query_file):
        queries = fvecs_read(query_file)
        print(f"ori query file shape: {queries.shape}")
        assert queries.shape[0] >= Nq
        if queries.shape[0] > Nq:
            queries = queries[:Nq]
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
    parser.add_argument("--schema_name", type=str, help="Schema name", required=True)
    parser.add_argument("--table_name", type=str, help="Table name", required=True)
    parser.add_argument("--table_file", type=str, help="Table file path", required=True)
    parser.add_argument("--mode", type=str, default="query", help="Dimension of data")
    parser.add_argument("--max_cate_val", type=int, default=5, help="Max cardinality for categorical attributes")
    parser.add_argument("--K", type=int, default=10, help="Top K")
    parser.add_argument("--gt_file", type=str, help="Output groundtruth file", required=True)
    parser.add_argument("--metric", type=str, default="L2", help="Distance metric, L2 or IP")
    args = parser.parse_args()
    return args

def create_schema_if_not_exists(cur, conn, schema_name):
    # Check if schema exists    
    query = sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
        sql.Identifier(schema_name)
    )
    cur.execute(query)
    conn.commit()

def create_table_file(vector_data, attr_list, attr_type_list, table_file):
    start_id = 0
    float_fmt = ".7f"
    with open(table_file, "w", encoding="utf-8") as f:
        for idx, (vec, attr) in enumerate(zip(vector_data, attr_list)):
            row_id = start_id + idx

            # vector -> {0.1234567,0.6543210}
            vec_str = "{" + ",".join(format(x, float_fmt) for x in vec) + "}"

            line = f"{row_id}\t"

            for i, attr_type in enumerate(attr_type_list):
                if attr_type == 0:
                    assert(len(attr[i]) == 1)
                    line += f"{attr[i][0]}\t"
                elif attr_type == 1:
                    if len(attr[i]) == 0:
                        cate_attr_str = "{}"
                    else:
                        cate_attr_str = "{" + ",".join(str(x) for x in attr[i]) + "}"
                    line += f"{cate_attr_str}\t"
                else:
                    print("error: unsupported attribute type ", attr_type)
                    exit()
            line = line + f"{vec_str}\n"
            f.write(line)
            # progress print
            if (idx + 1) % 100000 == 0:
                print(f"Written {idx + 1} / {len(vector_data)} rows", end="\r")

def table_exists(cur, schema_name: str, table_name: str) -> bool:
    """
    检查 schema.table 是否存在
    """
    cur.execute(
        sql.SQL("SELECT to_regclass(%s);"),
        (f"{table_name}",)
    )
    exist = cur.fetchone()[0] is not None
    if exist:
        print(f"Table {table_name} exists.")
    else:
        print(f"Table {table_name} does not exist.")
    return exist

def index_exists(cur, schema_name: str, table_name: str) -> bool:
    """
    检查index是否存在
    """
    cur.execute(
        "SELECT to_regclass(%s);",
        (f"index_{table_name}",)
    )
    res = cur.fetchone()[0]
    exist = res is not None
    if exist:
        print(f"Index index_{table_name} on table index_{table_name} exists.")
    else:
        print(f"Index index_{table_name} on table index_{table_name} does not exist.")
    return exist

def create_table_if_not_exists(cur, conn,schema_name, table_name, d, attr_type_list):
    raw_sql = "CREATE TABLE IF NOT EXISTS " + table_name + " (id INT, "
    for i, attr_type in enumerate(attr_type_list):
        if attr_type == 0:
            raw_sql += f"attr_{i} int, "
        elif attr_type == 1:
            raw_sql += f"attr_{i} int[], "
        else:
            print("error: unsupported attribute type ", attr_type)
            exit()
    raw_sql = raw_sql + f"vector_0 FLOAT8[{d}]);"
    create_table_sql = sql.SQL(raw_sql)
    cur.execute(create_table_sql)
    conn.commit()

def delete_table_if_exists(cur, conn, schema_name, table_name):
    raw_sql = "DROP TABLE IF EXISTS " + table_name
    create_table_sql = sql.SQL(raw_sql)
    cur.execute(create_table_sql)
    conn.commit()
    print("drop suc")

def insert_data(cur, conn, schema_name, table_name, dataset, attr, attr_type_list, table_file):
    if not os.path.isfile(table_file):
        print(f"Creating table file {table_file} ...")
        create_table_file(dataset, attr, attr_type_list, table_file)
        print(f"Table file {table_file} created.")
    else:
        print(f"Table file {table_file} already exists.")
    
    sql = f"""
        COPY {table_name}
        FROM STDIN
        DELIMITER E'\\t'
        CSV
        QUOTE E'\\x01'
        """
    # Use COPY command to bulk insert data from file
    with open(table_file, "r", encoding="utf-8") as f:
        cur.copy_expert(sql, f)
    conn.commit()

def delete_index_if_exists(cur, conn, schema_name, table_name):
    raw_sql = "DROP INDEX IF EXISTS " + "index_" + table_name
    drop_index_sql = sql.SQL(raw_sql)
    cur.execute(drop_index_sql)
    conn.commit()
    print("drop index suc")
    
def construct_index(cur, conn, schema_name, table_name, metric, d):
    # Create index on vector column
    cur.execute(sql.SQL("SET max_parallel_maintenance_workers = 64;"))
    if metric == "IP":
        distmethod = "inner_product" # inner_product
    elif metric == "L2":
        distmethod = "l2_distance"
    else:
        print("error: unsupported metric ", metric)
        exit()
    if metric == "IP":
        print("creating ip hnsw index...")
        raw_sql = f"CREATE INDEX IF NOT EXISTS index_{table_name} ON {table_name} USING hnsw (vector_0 hnsw_vector_inner_product_ops) WITH (dimension={d},distmethod={distmethod})"
        # raw_sql = f"CREATE INDEX index_{table_name} ON {table_name} USING sptag (vector_0 vector_inner_product_ops) WITH (distmethod={distmethod})"
    else:
        print("creating l2 hnsw index...")
        raw_sql = f"CREATE INDEX IF NOT EXISTS index_{table_name} ON {table_name} USING hnsw (vector_0) WITH (dimension={d},distmethod={distmethod})"
    create_index_sql = sql.SQL(raw_sql)
    cur.execute(create_index_sql)
    conn.commit()

def create_extension(cur, conn):
    # Create the vector extension if it doesn't exist
    cur.execute(sql.SQL("create extension if not exists vectordb;"))
    conn.commit()

def get_count(cur, schema_name, table_name):
    raw_sql = f"SELECT COUNT(*) FROM {table_name}"
    count_sql = sql.SQL(raw_sql)
    cur.execute(count_sql)
    count = cur.fetchone()[0]
    print(f"Table {table_name} has {count} rows.")
    return count

def generate_sql(schema_name, table_name, query_vector, raw_predicate, K, attr_type_list, metric):
    # example:
    # select id from t_table where price > 15 order by vector_1 <-> '{5,9,8,6,2,1,1,0,4,3}' limit 10;
    # attr_type_list = []
    # demo: explain
    # raw_sql = f"EXPLAIN (ANALYZE, BUFFERS, VERBOSE) SELECT id FROM {table_name} "
    raw_sql = f"SELECT id FROM {table_name} "
    if len(attr_type_list) > 0:
        raw_sql += "WHERE "
    for idx, attr_type in enumerate(attr_type_list):
        if attr_type == 0:
            raw_sql += f"attr_{idx} >= {raw_predicate[idx][0]} AND attr_{idx} <= {raw_predicate[idx][1]} "
            if idx != len(attr_type_list) - 1:
                raw_sql += " AND "
        elif attr_type == 1:
            if len(raw_predicate[idx]) == 0:
                continue
            for j, val in enumerate(raw_predicate[idx]):
                raw_sql += f"attr_{idx} @> "
                raw_sql += "'{" + ",".join(map(str, raw_predicate[idx])) + "}' "
                if idx < len(attr_type_list) - 1 or j < len(raw_predicate[idx]) - 1:
                    raw_sql += " AND "
    raw_sql += f"ORDER BY vector_0{'<->' if metric == 'L2' else '<*>'}"
    raw_sql += "'{" + ",".join(format(x, ".7f") for x in query_vector) + "}' "
    raw_sql += f"LIMIT {K};"
    return raw_sql

def execute_query(cur, raw_sql):
    # raw_sql = sql.SQL(raw_sql_str)
    cur.execute(raw_sql)
    results = cur.fetchall()
    # print("query results:", results)
    ids = [row[0] for row in results]
    return ids

if __name__ == "__main__":
    args = arg_init()

    attr_type_list = ast.literal_eval(args.attr_type_list)
    conn = psycopg2.connect(
        host="127.0.0.1",
        port=15432,
        database="vectordb",
        user="vectordb",
        password="vectordb",
    )
    conn.autocommit = True
    fields = ["id", "vector_0"] + [f"attr_{i}" for i in range(len(attr_type_list))]
    print("mode:", args.mode, " schema name:", args.schema_name, " table name:", args.table_name)

    cur = conn.cursor()

    # command_list = [
    #     "ALTER SYSTEM SET effective_cache_size = '1500GB';",
    #     "ALTER SYSTEM SET work_mem = '256MB';",
    #     "ALTER SYSTEM SET maintenance_work_mem = '8GB';",
    # ]

    # for command in command_list:
    #     cur.execute(sql.SQL(command))
    #     conn.commit()

    cur.execute(sql.SQL("show effective_cache_size;"))
    res = cur.fetchone()
    print("effective_cache_size:", res)
    cur.execute(sql.SQL("show work_mem;"))
    res = cur.fetchone()
    print("work_mem:", res)
    cur.execute(sql.SQL("show maintenance_work_mem;"))
    res = cur.fetchone()
    print("maintenance_work_mem:", res)
    cur.execute(sql.SQL("show shared_buffers;"))
    res = cur.fetchone()
    print("shared_buffers:", res)
    # exit()

    # index_name = f"index_{args.table_name}"
    # # get statistics
    # sql_str = f"SELECT pg_relation_size('{args.table_name}');"
    # cur.execute(sql.SQL(sql_str))
    # res = cur.fetchone()
    # print("res:", res)
    # conn.close()
    # exit()

    
    if args.mode == "construction":
        # create schema
        # create_schema_if_not_exists(cur, conn, args.schema_name)
        create_extension(cur, conn)
        # create table
        # delete_table_if_exists(cur, conn, args.schema_name, args.table_name)
        create_table_if_not_exists(cur, conn, args.schema_name, args.table_name, args.d, attr_type_list)
        # table_exists(cur, args.schema_name, args.table_name)
        # insert data
        line_cnt = get_count(cur, args.schema_name, args.table_name)
        if line_cnt != args.N:
            dataset, attr, query, predicate = read_data(args.dataset_file, args.attr_file, args.N, args.d, args.query_file, args.predicate_file, args.query_size)
            print("creating data csv file...")
            t0 = time.time()
            insert_data(cur, conn, args.schema_name, args.table_name, dataset, attr, attr_type_list, args.table_file)
            t1 = time.time()
            print(f"Data insertion time: {t1 - t0} seconds")
        else:
            print("data already inserted, size:", line_cnt)

        # construct index
        print("constructing index...")
        # delete_index_if_exists(cur, conn, args.schema_name, args.table_name)
        t2 = time.time()
        construct_index(cur, conn, args.schema_name, args.table_name, args.metric, args.d)
        t3 = time.time()
        print(f"Index construction time: {t3 - t2} seconds")
        # print("overall index construction time: ", t3 - t0, " seconds")

        cur.close()
        conn.close()
        exit()



    if args.mode == "query":
        # check if table exists
        if not table_exists(cur, args.schema_name, args.table_name):
            print("error: table ", args.table_name, " does not exist")
            cur.close()
            conn.close()
            exit()
        # check if index exists
        if not index_exists(cur, args.schema_name, args.table_name):
            print("error: index on table ", args.table_name, " does not exist")
            cur.close()
            conn.close()
            exit()
        
        # load query data
        queries, query_filter_ranges, query_gt = load_query_data(args.query_file, args.predicate_file, args.gt_file, args.N, args.query_size, args.K)
        print("query shape:", queries.shape)
        print("query filter ranges length:", len(query_filter_ranges))
        print("query gt shape:", query_gt.shape)
        
        result = []

        sql_str = sql.SQL("set enable_seqscan = false;")
        cur.execute(sql_str)
        conn.commit()

        # sql_str = sql.SQL("ALTER SYSTEM SET shared_buffers = '256GB';")
        # cur.execute(sql_str)
        # conn.commit()

        # generate raw sql list
        raw_sql_list = []
        for i in range(args.query_size):
            raw_sql = generate_sql(args.schema_name, args.table_name, queries[i], query_filter_ranges[i], args.K, attr_type_list, args.metric)
            raw_sql_list.append(sql.SQL(raw_sql))
        
        # print sql example
        # print("example sql:\n", raw_sql_list[0])
        
        ids = []
        q_t = 0
        hist = np.array([0 for _ in range(11)], dtype='float')
        total_size = args.N * args.K
        positive_size = 0
        time_one_batch = 0
        # cur.execute("SET enable_seqscan = off;") 
        # cur.execute("SET enable_indexscan = on;")
        # cur.execute("SET hnsw.ef_search = 1000;")
        # cur.execute("SET msvbase.ef_search = 1000;")
        # warm up

        print("warming up ...")
        for i in range(2):
            res_id = execute_query(cur, raw_sql_list[i])

        print("start querying ...")

        for i in range(args.query_size):
            q_cnt = 0
            start = time.time()
            res_id = execute_query(cur, raw_sql_list[i])
            end = time.time()
            # print(f"Query {i} time: {end - start} seconds, return res: {res_id}")
            time_one_batch += end - start
            # print(f"Query {i} time: {end - start} seconds")
            ids.append(res_id)

            # cur.close()
            # conn.close()
            # exit()
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
            # print(f"query {i} recall: {np.sum(correct)}/{len(gt)}")
            # print("gt:", gt)
            # print("res:", res)
            recall_list.append(np.sum(correct)/len(gt))
        recall = correct_sum / (args.query_size * args.K)
        print(f"recall: {recall:.4f}, qps: {qps:.2f}")

    cur.close()
    conn.close()

