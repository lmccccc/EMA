"""Audit remaining Figure5 cells against freshly completed current mixed indexes."""

import argparse
import ast
from pathlib import Path

import numpy as np

from .dynamic.runtime import file_info, immutable_json
from .figure4a import and_predicates, composite_predicates
from .static_inputs import Audit, DATASETS
from .static_paper import read_json


HIGH = ((10, [0.333, 7]), (20, [0.4, 5]), (40, [0.667, 4]),
        (60, [0.75, 2]), (80, [0.9, 1]), (100, [1.0, 0]))
LOW = ((1, [0.1, 9]), (2, [0.1, 8]), (3, [0.15, 8]),
       (5, [0.177, 7]), (7, [0.233, 7]), (10, [0.333, 7]))
COMPOSITE = ((1, 19, 18), (2, 18, 17), (3, 17, 16),
             (5, 16, 14), (7, 14, 12), (10, 9, 12))


def and_path(directory, selector):
    matches = []
    prefix = "predicate_arbi_0_1_"
    for path in directory.glob(prefix + "*.json"):
        suffix = path.name[len(prefix):-len(".json")]
        try:
            value = ast.literal_eval(suffix)
        except (SyntaxError, ValueError):
            continue
        if value == selector:
            matches.append(path)
    if len(matches) != 1:
        raise ValueError(f"Expected one original AND predicate for {selector}: {matches}")
    return matches[0]


def original_prefix(predicate, ground_truth):
    predicates, truth = read_json(predicate), read_json(ground_truth)
    if not isinstance(predicates, list) or not isinstance(truth, list):
        raise ValueError("Original predicate/GT files must contain query rows")
    count = min(1000, len(predicates), len(truth))
    if count not in (100, 1000):
        raise ValueError("Mixed paper inputs must support an explicit100- or1000-query prefix")
    return predicates[:count], count


def build_request(audit, construction):
    construction = Path(construction)
    if read_json(construction / "status.json").get("status") != "complete":
        raise RuntimeError("Finish the construction/Figure6 queue before starting mixed query timing")
    original = read_json(construction / "suite-manifest.json")["request"]
    jobs = []
    for dataset in ("youtube1m", "redcaps4m", "wiki15m"):
        root = audit.root / DATASETS[dataset]["directory"]
        labels = root / "label/arbi_0_1_random"
        build_root = construction / "jobs" / f"build-{dataset}-mixed"
        build = read_json(build_root / "build.json")
        prepared = read_json(build_root / "manifest.json")
        if read_json(build_root / "results.json").get("status") != "complete":
            raise ValueError("Mixed index construction is not complete")
        job = audit.job(dataset, f"figure5-{dataset}-mixed", [0, 1],
                        labels / "attr_arbi_0_1_random.json")
        for name in ("vectors", "queries", "attributes"):
            if job["inputs"][name] != prepared["inputs"][name]:
                raise ValueError("Mixed query source differs from its fresh build source")
        job["checkpoint"] = file_info(build["checkpoint"]["path"], build["checkpoint"])
        job["construction_record"] = file_info(build_root / "build.json")
        cells = {}
        for panel, specifications in (("a", HIGH), ("b", LOW)):
            for nominal, selector in specifications:
                predicate = and_path(labels, selector)
                gt = predicate.with_name(predicate.name.replace("predicate_", "gt_", 1))
                raw, count = original_prefix(predicate, gt)
                _, masks = and_predicates(raw, count)
                if not np.all(masks == 1 << selector[1]):
                    raise ValueError(f"Saved AND categorical condition differs from its cell: {predicate}")
                name = f"and-T{nominal}"
                cell = audit.cell(name, nominal, predicate, gt, count, dnf=False,
                                  targets=[0.95], plots=[{"figure": 5, "panel": panel}])
                if name in cells:
                    prior = cells[name]
                    if any(prior[key] != cell[key] for key in
                           ("predicate", "ground_truth", "query_count", "nominal_selectivity")):
                        raise ValueError("Duplicated10% AND panels do not use the same workload")
                    prior["plots"].extend(cell["plots"])
                else:
                    cells[name] = cell
        for nominal, first, second in COMPOSITE:
            predicate = labels / f"predicate_dnf_or_T{nominal}.json"
            gt = labels / f"gt_dnf_or_T{nominal}.json"
            raw, count = original_prefix(predicate, gt)
            _, masks, tails = composite_predicates(raw, count)
            if not np.all(masks == 1 << first) or not np.all(tails == 1 << second):
                raise ValueError(
                    f"Saved composite does not match the preserved paper family: {predicate}; "
                    "do not regenerate it from current defaults")
            name = f"dnf-T{nominal}"
            cells[name] = audit.cell(name, nominal, predicate, gt, count, dnf=True,
                                     targets=[0.95], plots=[{"figure": 5, "panel": "c"}])
        job["cells"] = list(cells.values())
        job["query_count"] = max(cell["query_count"] for cell in job["cells"])
        jobs.append(job)
    return {
        "native_dir": original["native_dir"], "native_sha256": original["native_sha256"],
        "jobs": jobs, "purpose": "Remaining Figure5 EMA queries on the fresh mixed indexes",
        "query_prefix_policy": "Use the available original prefix up to1000, explicitly recorded per cell",
        "construction_suite": file_info(construction / "suite-manifest.json"),
        "audit_source": file_info(__file__),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--construction-run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    request = build_request(Audit(args.data_root, output), args.construction_run)
    immutable_json(output / "mixed-request.json", request)
    print(f"Audited {sum(len(job['cells']) for job in request['jobs'])} unique mixed query cells",
          flush=True)


if __name__ == "__main__":
    main()
