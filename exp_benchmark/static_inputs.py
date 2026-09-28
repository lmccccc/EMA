"""Audit original static-paper inputs without regenerating predicates or ground truth."""

import argparse
import json
from pathlib import Path
import struct

from .dynamic.runtime import atomic_json, file_info, immutable_json, require_unchanged
from .static_paper import PARAMETERS, read_json


DATASETS = {
    "youtube1m": {
        "directory": "youtube1m", "N": 1_000_000, "dimension": 1024, "metric": "ip",
        "vectors": "rgb.fvecs", "queries": "rgb_query.fvecs", "paper_dataset": "YoutubeRGB1M",
    },
    "redcaps4m": {
        "directory": "redcaps4m", "N": 4_000_000, "dimension": 512, "metric": "ip",
        "vectors": "image_embeddings.fvecs", "queries": "query.fvecs", "paper_dataset": "Redcaps4M",
    },
    "wiki15m": {
        "directory": "navix_dataset/wiki_15.4M", "N": 15_435_516, "dimension": 1024, "metric": "ip",
        "vectors": "fvecs/wiki_15.4M.fvecs", "queries": "fvecs/wiki_15.4M_query.fvecs",
        "paper_dataset": "Wiki15.4M",
    },
    "sift10m": {
        "directory": "sift10m", "N": 10_000_000, "dimension": 128, "metric": "l2",
        "vectors": "sift10m.fvecs", "queries": "sift10m_query.fvecs", "paper_dataset": "SIFT10M",
    },
}
SELECTIVITIES = (1, 2, 3, 5, 7, 10)
LABELS = (18, 17, 16, 14, 12, 9)
OCQ = ((1, "1.01"), (5, "5.10"), (10, "9.96"), (15, "15.02"), (23, "22.93"))


class Audit:
    def __init__(self, root, output):
        self.root, self.output = Path(root).resolve(strict=True), Path(output)
        self.path = self.output / "input-fingerprints.json"
        self.files = read_json(self.path) if self.path.exists() else {}

    def fingerprint(self, path):
        path = Path(path)
        key = str(path)
        if key not in self.files:
            print(f"Hashing original input: {path}", flush=True)
            self.files[key] = file_info(path)
            atomic_json(self.path, self.files)
        else:
            require_unchanged(self.files[key])
        return self.files[key]

    def fvecs(self, path, dimension):
        info = self.fingerprint(path)
        with Path(path).open("rb") as stream:
            encoded = stream.read(4)
        if (len(encoded) != 4 or struct.unpack("<i", encoded)[0] != dimension
                or info["bytes"] % ((dimension + 1) * 4)):
            raise ValueError(f"Invalid original fvecs geometry: {path}")
        return info, info["bytes"] // ((dimension + 1) * 4)

    def job(self, dataset, name, kinds, attribute, *, queries=None, count=1000):
        spec = DATASETS[dataset]
        root = self.root / spec["directory"]
        vectors, rows = self.fvecs(root / spec["vectors"], spec["dimension"])
        if rows != spec["N"]:
            raise ValueError("The original base file is not the full declared dataset")
        query, query_rows = self.fvecs(queries or root / spec["queries"], spec["dimension"])
        if query_rows < count:
            raise ValueError("The original query file lacks the requested prefix")
        return {
            "name": name, "dataset": dataset, "paper_dataset": spec["paper_dataset"],
            "N": spec["N"], "dimension": spec["dimension"], "metric": spec["metric"],
            "normalization": "none", "attribute_types": kinds,
            "query_count": count, "query_file_rows": query_rows, "query_offset": 0,
            "inputs": {"vectors": vectors, "queries": query, "attributes": self.fingerprint(attribute)},
            "parameters": PARAMETERS, "max_ef": 4096, "cells": [],
        }

    def cell(self, name, nominal, predicate, ground_truth, count, *, dnf, targets, plots):
        pred_info, gt_info = self.fingerprint(predicate), self.fingerprint(ground_truth)
        predicates, truth = read_json(predicate), read_json(ground_truth)
        if (not isinstance(predicates, list) or len(predicates) < count
                or not isinstance(truth, list) or len(truth) < count
                or not truth or not isinstance(truth[0], list) or len(truth[0]) < 10
                or any(not isinstance(row, list) or len(row) != len(truth[0]) for row in truth)):
            raise ValueError(f"Original predicate/GT dimensions cannot support {name}: {predicate}")
        return {
            "name": name, "nominal_selectivity": nominal / 100,
            "predicate": pred_info, "ground_truth": gt_info,
            "query_count": count, "predicate_rows": len(predicates),
            "ground_truth_rows": len(truth), "ground_truth_k": len(truth[0]),
            "dnf": dnf, "targets": targets, "plots": plots,
            "first_original_predicate": predicates[0],
        }

    def initial_jobs(self):
        jobs = []
        for dataset in ("youtube1m", "redcaps4m", "wiki15m", "sift10m"):
            root = self.root / DATASETS[dataset]["directory"]
            mixed = self.job(
                dataset, f"build-{dataset}-mixed", [0, 1],
                root / "label/arbi_0_1_random/attr_arbi_0_1_random.json")
            mixed["build_only"] = True
            jobs.append(mixed)
            if dataset in ("youtube1m", "redcaps4m"):
                for kind, panel in ((0, "a"), (1, "b")):
                    attribute_root = root / f"label/arbi_{kind}_random"
                    count = 100 if dataset == "youtube1m" and kind == 1 else 1000
                    job = self.job(
                        dataset, f"figure6-{dataset}-{'range' if kind == 0 else 'label'}", [kind],
                        attribute_root / f"attr_arbi_{kind}_random.json", count=count)
                    selectors = [value / 100 for value in SELECTIVITIES] if kind == 0 else LABELS
                    for nominal, selector in zip(SELECTIVITIES, selectors):
                        suffix = f"arbi_{kind}_[{selector}].json"
                        job["cells"].append(self.cell(
                            f"T{nominal}", nominal, attribute_root / f"predicate_{suffix}",
                            attribute_root / f"gt_{suffix}", count, dnf=False,
                            targets=[0.90, 0.95] if dataset == "youtube1m" and kind == 0 else [0.95],
                            plots=[{"figure": 6, "panel": panel}]))
                    jobs.append(job)
            elif dataset == "wiki15m":
                directory = root / "neg_correlated"
                aliases = [self.fingerprint(directory / f"join_{suffix}_embedding.fvecs")
                           for _, suffix in OCQ]
                if len({item["sha256"] for item in aliases}) != 1:
                    raise ValueError("OCQ selectivity files have different query orders/payloads")
                job = self.job(
                    dataset, "figure6-wiki15m-ocq", [0],
                    root / "fvecs/wiki_15.4M_birthdate.json",
                    queries=directory / "join_22.93_embedding.fvecs", count=50)
                job["equivalent_original_query_files"] = aliases
                for nominal, suffix in OCQ:
                    job["cells"].append(self.cell(
                        f"T{nominal}", nominal, directory / f"join_{suffix}_predicate.json",
                        directory / f"join_{suffix}_gt.json", 50, dnf=False, targets=[0.95],
                        plots=[{"figure": 6, "panel": "c"}]))
                jobs.append(job)
        return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--native-dir", required=True)
    parser.add_argument("--native-sha256", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    audit = Audit(args.data_root, output)
    request = {
        "native_dir": str(Path(args.native_dir).resolve(strict=True)),
        "native_sha256": args.native_sha256, "jobs": audit.initial_jobs(),
        "purpose": "Current mixed-index costs and original Figure6 range/label/OCQ queries",
        "input_policy": "Original saved vectors, predicates and GT; no regeneration or normalization",
        "audit_source": file_info(__file__),
    }
    immutable_json(output / "initial-request.json", request)
    print(json.dumps({"jobs": len(request["jobs"]),
                      "query_cells": sum(len(job["cells"]) for job in request["jobs"])}), flush=True)


if __name__ == "__main__":
    main()
