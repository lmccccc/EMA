"""Untimed first-failure witnesses for incomplete component-ablation batches."""

import argparse
import os
from pathlib import Path

from ..component_ablation import (
    COUNTERS, PARAMETERS, configure, load_data, load_graphs, load_request,
    read_json, select_native,
)
from ..dynamic.runtime import (
    OutputLock, bootstrap_manifest, environment_config, file_info, immutable_json,
)
from ..static_paper import Predicates, validate_result


def probe(index, variant, ef, query_id, cell, vectors, queries, attributes):
    configure(index, variant, ef)
    index.reset_ft_stats()
    query = queries[query_id:query_id + 1]
    raw = cell["predicates"].raw[query_id:query_id + 1]
    observation = {"variant": variant, "ef": ef, "query_id": query_id}
    try:
        result = index.hybrid_knn_query(query, raw, k=PARAMETERS["k"], num_threads=1)
    except RuntimeError as error:
        if "Cannot return the results in a contiguous 2D array" not in str(error):
            raise
        observation.update(complete=False, error="fewer_than_k_results")
    else:
        verified = validate_result(
            result, cell["truth"][query_id:query_id + 1], vectors, query, attributes,
            Predicates.parse(raw, [0, 1], 1), "ip", PARAMETERS["k"])
        observation.update(
            complete=True, **verified, labels=result[0].tolist(), distances=result[1].tolist())
    stats = index.get_ablation_stats()
    if not stats["diagnostic_build"] or any(stats[key] < 0 for key in COUNTERS):
        raise RuntimeError("Failure witnesses require valid untimed diagnostic counters")
    observation["counters"] = {key: stats[key] for key in COUNTERS}
    return observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request, root = load_request(args.request)
    output = root / "failure_diagnostics"
    output.mkdir(exist_ok=True)
    with OutputLock(output):
        bootstrap_manifest(output / "manifest.json", {
            "request": file_info(Path(args.request).resolve()),
            "source": file_info(Path(__file__).resolve()),
            "native": request["native"]["modules"]["diagnostic"],
            "policy": "First failing original query at the largest failed ef; not a failure-rate or QPS estimate",
        })
        snapshot = output / "source.py"
        if not snapshot.exists():
            snapshot.write_bytes(Path(__file__).read_bytes())
        native = select_native(request, diagnostic=True)
        vectors, queries, attributes, _, cells = load_data(request, root)
        index = load_graphs(native, request, root)[True]
        environment_config(0, PARAMETERS["build_threads"], requested_cores=[8, 16, 24])
        for name, cell in cells.items():
            complete = read_json(root / "measurements" / name / "complete.json")
            plan = complete["plans"]["c1_c2"]
            failed_efs = [p["ef"] for p in plan["points"]
                          if p.get("error") == "fewer_than_k_results"]
            if not failed_efs or (output / f"{name}.json").exists():
                continue
            os.sched_setaffinity(0, {complete["core"]})
            maximum = max(failed_efs)
            scan = []
            for query_id in range(len(queries)):
                observation = probe(
                    index, "c1_c2", maximum, query_id, cell, vectors, queries, attributes)
                scan.append(observation)
                if not observation["complete"]:
                    break
            else:
                raise RuntimeError(f"{name}: the recorded incomplete batch did not reproduce")
            probes = [observation]
            controls = [("c1_c2", min(failed_efs)), ("c1", maximum), ("full", maximum)]
            selected = complete["plans"]["full"]["selected_ef"]
            if selected is not None:
                controls.append(("full", selected))
            for variant, ef in controls:
                probes.append(probe(
                    index, variant, ef, query_id, cell, vectors, queries, attributes))
            immutable_json(output / f"{name}.json", {
                "cell": name, "examined_queries": len(scan), "scan": scan, "probes": probes,
                "note": "The first failure demonstrates incomplete reachability; these selected queries do not estimate mean recall or failure frequency.",
            })
            print(f"{name}: first incomplete original query {query_id}, ef={maximum}", flush=True)
            os.sched_setaffinity(0, set(range(32)))


if __name__ == "__main__":
    main()
