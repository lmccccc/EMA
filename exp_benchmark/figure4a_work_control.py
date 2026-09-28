"""Hardware-independent recall/search-work replay using each native's own index format."""

import argparse
import json
import os
from pathlib import Path
import struct

import numpy as np

from .dynamic.gt import array_digest, read_fvecs
from .dynamic.runtime import atomic_json, file_info, import_native, require_unchanged
from .figure4a import read_json


def header_prefix(path):
    fields = (
        "offset_level0", "max_elements", "top_elements", "count", "record_bytes",
        "neighbor_bytes", "label_offset", "vector_offset", "max_level", "entrypoint",
        "maxM", "maxM0", "M0_mul", "minM0", "M", "mult", "ef_construction", "ft_bits",
    )
    layout = "<8QiI5QdQQ"
    with Path(path).open("rb") as stream:
        values = struct.unpack(layout, stream.read(struct.calcsize(layout)))
    return dict(zip(fields, values))


def run(args, output):
    reference = Path(args.reference)
    manifest = read_json(reference / "manifest.json")
    config = manifest["request"]
    selected = [cell for cell in config["cells"] if cell["name"] in args.cells.split(",")]
    if len(selected) != len(args.cells.split(",")):
        raise ValueError("Unknown or duplicate workload cell")
    atomic_json(output / "status.json", {"status": "running", "phase": "verifying_index"})
    native, native_info = import_native(args.native_dir, args.native_sha)
    index_info = file_info(args.index)
    header = header_prefix(args.index)
    if (header["count"] != config["N"] or header["M"] != 40
            or header["ef_construction"] != 300 or header["ft_bits"] != 128):
        raise ValueError("The work comparison requires full10M, M40, efc300,128-bit indexes")
    query_info = manifest["inputs"]["queries"]
    require_unchanged(query_info)
    queries = np.ascontiguousarray(read_fvecs(query_info["path"], 1000, 128))
    if array_digest(queries) != manifest["query_payload_sha256"]:
        raise ValueError("Query payload changed")
    atomic_json(output / "manifest.json", {
        "reference": file_info(reference / "manifest.json"),
        "native": native_info, "index": index_info, "header": header,
        "source": file_info(__file__), "native_header_source": file_info(args.source_header),
        "cells": [cell["name"] for cell in selected],
        "query_api": "hybrid_knn_query_with_stats", "target_recall": 0.95,
        "routing_min_degree": 16, "ef_top": 1, "query_threads": 1,
        "measurement": "integer ef, recall, distance-computation counts and expansions; no QPS attribution",
        "old_index_format_policy": "Only the explicitly selected native loads its own supported format; no retagging",
    })
    atomic_json(output / "status.json", {"status": "running", "phase": "loading_index"})
    index = native.Index(space="l2", dim=128)
    index.set_num_threads(32)
    index.load_index(args.index)
    if index.get_current_count() != config["N"] or index.get_deleted_ratio() != 0:
        raise ValueError("Comparison index is not the full10M zero-deletion corpus")
    index.generateAttrIndexes()
    index.set_thresholds(0.0001, 0.0001, 0.0001)
    index.set_num_threads(1)
    index.set_ef_top(1)
    index.set_ft_flag(True)
    index.set_ft_routing_flag(True)
    index.set_ft_routing_min_deg(16)
    index.set_ft_routing_backfill_tail(False)
    os.sched_setaffinity(0, {24})
    summaries = []
    for cell in selected:
        inputs = manifest["inputs"][cell["name"]]
        require_unchanged(inputs["predicates"])
        require_unchanged(inputs["ground_truth"])
        predicates = read_json(inputs["predicates"]["path"])
        truth = np.asarray(read_json(inputs["ground_truth"]["path"]), dtype=np.int64)
        if len(predicates) != 1000 or truth.shape != (1000, 10):
            raise ValueError("Wrong predicate/GT dimensions")
        points = []
        for ef in range(10, config["max_ef"] + 1):
            index.set_ef(ef)
            labels, distances, counts, hops = index.hybrid_knn_query_with_stats(
                queries, predicates, k=10)
            valid = (labels >= 0) & (labels < config["N"])
            recall = float(np.any(
                (labels[:, :, None] == truth[:, None, :]) & valid[:, :, None], axis=1).mean())
            complete = bool(valid.all())
            point = {
                "ef": ef, "recall": recall, "complete_results": complete,
                "mean_distance_computations": float(np.mean(counts)),
                "mean_expanded_nodes": float(np.mean(hops)),
                "labels_sha256": array_digest(labels),
                "distances_sha256": array_digest(distances),
            }
            points.append(point)
            atomic_json(output / f"{cell['name']}-calibration.json", {"points": points})
            atomic_json(output / "status.json", {
                "status": "running", "cell": cell["name"], "ef": ef, "recall": recall})
            if complete and recall >= 0.95:
                break
        if not points[-1]["complete_results"] or points[-1]["recall"] < 0.95:
            raise RuntimeError("No complete95%-recall result before the frozen ef ceiling")
        # Repeat the selected point to rule out stateful query/work-count variation.
        repeated = index.hybrid_knn_query_with_stats(queries, predicates, k=10)
        if (array_digest(repeated[0]) != points[-1]["labels_sha256"]
                or array_digest(repeated[1]) != points[-1]["distances_sha256"]
                or float(np.mean(repeated[2])) != points[-1]["mean_distance_computations"]
                or float(np.mean(repeated[3])) != points[-1]["mean_expanded_nodes"]):
            raise RuntimeError("Query output or search work changed on an unchanged index")
        summaries.append({"cell": cell["name"], "selected": points[-1],
                          "bracket": points[-2:] if len(points) > 1 else points})
        atomic_json(output / "results.json", {"status": "running", "cells": summaries})
    require_unchanged(index_info)
    require_unchanged(native_info)
    atomic_json(output / "results.json", {
        "status": "complete", "cells": summaries,
        "limitation": "This identifies the selected native/index pair, not the unrecorded historical Figure4 binary.",
    })
    atomic_json(output / "status.json", {"status": "complete", "cells": len(summaries)})
    print(json.dumps(summaries, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--native-dir", required=True)
    parser.add_argument("--native-sha", required=True)
    parser.add_argument("--source-header", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cells", default="T10,T100")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        run(args, output)
    except (OSError, ValueError, RuntimeError) as error:
        atomic_json(output / "status.json", {
            "status": "failed", "error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
