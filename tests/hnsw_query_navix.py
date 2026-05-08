import ast
import time
from typing import List

import hnswlib
import numpy as np

from utils import arg_init, read_attr, read_multy_attr, fvecs_read


def _normalize_numeric(value):
    if isinstance(value, list):
        if len(value) == 0:
            return None
        return value[0]
    return value


def _normalize_categorical(value):
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _match_predicate(attr_row, predicate_row, attr_type_list):
    for idx, attr_type in enumerate(attr_type_list):
        cond = predicate_row[idx]
        if attr_type == 0:
            if len(cond) == 0:
                continue
            val = _normalize_numeric(attr_row[idx])
            if val is None:
                return False
            if val < cond[0] or val > cond[1]:
                return False
        else:
            required_labels = cond
            if len(required_labels) == 0:
                continue
            labels = _normalize_categorical(attr_row[idx])
            for target in required_labels:
                if target not in labels:
                    return False
    return True


def _load_query_data(query_file, qrange_file, gt_file, nq, k):
    if ".fvecs" not in query_file:
        raise ValueError(f"query file format not supported: {query_file}")
    if ".json" not in qrange_file:
        raise ValueError(f"predicate file format not supported: {qrange_file}")
    if ".json" not in gt_file:
        raise ValueError(f"ground truth file format not supported: {gt_file}")

    queries = fvecs_read(query_file)
    predicates = read_multy_attr(qrange_file)
    gt = read_attr(gt_file).reshape(-1, k)

    if queries.shape[0] < nq:
        raise ValueError(f"query rows {queries.shape[0]} < requested {nq}")
    if len(predicates) < nq:
        raise ValueError(f"predicate rows {len(predicates)} < requested {nq}")
    if len(gt) < nq:
        raise ValueError(f"gt rows {len(gt)} < requested {nq}")

    return queries[:nq], predicates[:nq], gt[:nq]


def main():
    args = arg_init()
    attr_type_list = ast.literal_eval(args.attr_type_list)
    nq = args.n_query_to_use

    attrs = read_multy_attr(args.attr_path)
    if len(attrs) < args.N:
        raise ValueError(f"attr rows {len(attrs)} < N={args.N}")

    queries, predicates, query_gt = _load_query_data(
        args.query_path, args.qrange_path, args.gt_path, nq, args.K
    )

    metric = args.metric.lower()
    index = hnswlib.Index(space=metric, dim=args.dim)
    index.load_index(args.index_cache_path)

    # Python callback filters remain slow under multi-threaded hnswlib execution.
    index.set_num_threads(1)

    efs_list = ast.literal_eval(args.ef_search)
    if not isinstance(efs_list, List):
        efs_list = [efs_list]

    print(f"query shape: {queries.shape}")
    print(f"attr rows: {len(attrs)}")
    print(f"predicate rows: {len(predicates)}")
    print(f"metric: {metric}")
    print(f"ef list: {efs_list}")
    print("query mode: navix")

    results = []
    for efs in efs_list:
        index.set_ef(efs)
        print(f"start navix query with ef_search={efs}")

        labels_all = []
        start = time.time()
        for i in range(nq):
            predicate = predicates[i]

            def _filter(label_id, _pred=predicate):
                lid = int(label_id)
                if lid < 0 or lid >= len(attrs):
                    return False
                return _match_predicate(attrs[lid], _pred, attr_type_list)

            labels, _ = index.knn_query_navix(
                queries[i : i + 1],
                k=args.K,
                num_threads=1,
                filter=_filter,
            )
            labels_all.append(labels[0])
        end = time.time()

        correct_sum = 0
        for i in range(nq):
            gt = query_gt[i]
            pred = np.asarray(labels_all[i])
            pred = pred[pred < len(attrs)]
            if len(pred) == 0:
                continue
            correct = np.isin(gt, pred)
            correct_sum += int(np.sum(correct))

        recall = correct_sum / (nq * args.K)
        qps = nq / max(end - start, 1e-12)
        print(f"ef search: {efs}, recall: {recall:.4f}, qps: {qps:.2f}")
        results.append([efs, recall, qps])
        if recall >= 0.99:
            break

    print("Final navix results (ef_search, recall, QPS):")
    for row in results:
        print(row)


if __name__ == "__main__":
    main()