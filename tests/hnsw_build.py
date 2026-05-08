import argparse
import os
import time

import hnswlib

from utils import fvecs_read


def arg_init():
    parser = argparse.ArgumentParser(description="Build an HNSW index with hnswlib")
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the fvecs base data file"
    )
    parser.add_argument(
        "--index_cache_path",
        type=str,
        required=True,
        help="Path to save the built index",
    )
    parser.add_argument("--M", type=int, default=16, help="Number of graph connections")
    parser.add_argument(
        "--efConstruction", type=int, default=500, help="Construction ef parameter"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="l2",
        choices=["l2", "ip", "cosine", "L2", "IP", "COSINE"],
        help="Distance metric used by hnswlib",
    )
    parser.add_argument("--N", type=int, required=True, help="Number of base vectors")
    parser.add_argument("--threads", type=int, default=1, help="Build thread count")
    parser.add_argument("--dim", type=int, required=True, help="Vector dimension")
    parser.add_argument(
        "--name",
        type=str,
        default="HNSW",
        help="Index name kept for compatibility with existing scripts",
    )
    return parser.parse_args()


def load_base_data(data_path: str, expected_count: int, expected_dim: int):
    if ".fvecs" not in data_path:
        raise ValueError(f"Unsupported dataset format: {data_path}")

    data = fvecs_read(data_path)
    print(f"data shape: {data.shape}")

    if data.shape[0] != expected_count:
        raise ValueError(f"Expected N={expected_count}, got {data.shape[0]}")
    if data.shape[1] != expected_dim:
        raise ValueError(f"Expected dim={expected_dim}, got {data.shape[1]}")

    return data


def build_index(args):
    if args.name.upper() not in {"HNSW", "HNSWLIB"}:
        raise ValueError(f"Unsupported index name for hnsw_build.py: {args.name}")

    metric = args.metric.lower()
    data = load_base_data(args.data_path, args.N, args.dim)

    index_dir = os.path.dirname(args.index_cache_path)
    if index_dir:
        os.makedirs(index_dir, exist_ok=True)

    index = hnswlib.Index(space=metric, dim=args.dim)
    index.init_index(
        max_elements=args.N,
        ef_construction=args.efConstruction,
        M=args.M,
    )
    index.set_num_threads(args.threads)

    print(
        f"Building HNSW index: M={args.M}, efConstruction={args.efConstruction}, metric={metric}"
    )
    start = time.time()
    index.add_items(data)
    end = time.time()

    index.save_index(args.index_cache_path)
    print(f"Index built and saved to {args.index_cache_path}")
    print(f"Build time: {end - start} seconds")


if __name__ == "__main__":
    build_index(arg_init())
