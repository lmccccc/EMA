"""Remeasure compatible archived dynamic checkpoints at95% recall, without mutations."""

import argparse
import copy
import csv
import gc
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import uuid

import numpy as np

from .aggregate import paper_row, validate_calibration, validate_timing
from .gt import MixedAttributes, array_digest, predicate_bounds, read_fvecs
from .protocol import INPUT_SHA, OPERATIONS, PROTOCOL, QUERY_CONFIG, Trace, require_current_query_native
from .runner import OperationRunner
from .runtime import (
    OutputLock, atomic_json, environment_config, file_info, immutable_json,
    import_native, object_digest, read_index_header, require_unchanged,
    timestamp, verify_index_records, write_npz,
)


SCHEMA = "canonical-dynamic-recall-revision-v1"
SOURCE_PROTOCOL = "sift10m-mixed-dnf-dynamic-rebuild-v4"
CSV_FIELDS = (
    "operation", "stage", "progress", "occupied", "deleted", "live",
    "maintenance_records", "maintenance_seconds", "maintenance_minutes",
    "delete_seconds", "delete_scrub_seconds", "delete_non_scrub_seconds",
    "delete_scrub_fraction", "maintenance_scrub_fraction", "add_seconds",
    "qps95", "observed_qps", "recall", "ef",
    "recall_status", "paired_initial_qps95", "same_recall_qps95_change",
    "selectivity_live_mean", "selected_query_core", "accepted_rounds", "raw_results",
)


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def require_source_config(config):
    if QUERY_CONFIG["target_recall"] != .95:
        raise RuntimeError("Remeasurement requires the canonical95%-recall implementation")
    expected_query = {**QUERY_CONFIG, "target_recall": .90}
    expected = {
        "protocol": SOURCE_PROTOCOL, "mode": "official", "capacity": 10_000_000,
        "initial": 5_000_000, "step": 1_000_000,
        "stage_progress": list(range(0, 5_000_001, 1_000_000)),
        "query": expected_query, "index_format": 10, "numeric_marker_version": 2,
        "marker_owner_version": 2, "distance_order_version": 1,
        "pickle_state_version": 4, "candidate_order": "vector-distance-only-v1",
        "operations": list(OPERATIONS), "slot_reuse": False,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Source is not a complete compatible V11-style90% protocol")


def verify_source_copy(info, inventory):
    if inventory is None:
        return file_info(info["path"], info)
    entry = inventory.get(info["path"])
    if entry is None or entry["original"] != info:
        raise RuntimeError("Missing or mismatched original-source snapshot identity")
    copied = file_info(entry["copy"]["path"], entry["copy"])
    if copied["sha256"] != info["sha256"]:
        raise RuntimeError("Frozen source copy does not match the executed source")
    return copied


def verify_prefix(current, original):
    keys = ("ef", "recall", "labels_sha256", "distances_sha256")
    if len(current["points"]) < len(original["points"]):
        raise RuntimeError("The95% calibration does not contain the source90% prefix")
    for actual, expected in zip(current["points"], original["points"]):
        if any(actual[key] != expected[key] for key in keys):
            raise RuntimeError("Archived graph/query/GT replay differs from its original calibration")


def read_source(root, snapshot):
    pipeline = read_json(root / "pipeline-manifest.json")
    require_source_config(pipeline["config"])
    paper = read_json(root / "paper.json")
    if (paper["schema"] != "canonical-dynamic-paper-current-v5"
            or paper["status"] != "complete" or len(paper["rows"]) != 18):
        raise RuntimeError("Source must have complete original three-operation paper evidence")
    if snapshot is not None and snapshot["source_run"] != str(root):
        raise RuntimeError("Source snapshot belongs to another run")
    source_infos = {info["path"]: info for info in pipeline["source"].values()}
    documents = {
        "pipeline_manifest": file_info(root / "pipeline-manifest.json"),
        "paper": file_info(root / "paper.json"),
    }
    operations = {}
    for operation in OPERATIONS:
        manifest = read_json(root / operation / "manifest.json")
        journal = read_json(root / operation / "journal.json")
        if (manifest["config"] != {**pipeline["config"], "operation": operation}
                or manifest["inputs"] != pipeline["inputs"]
                or journal["schema"] != "canonical-dynamic-operation-v3"
                or journal["status"] != "complete" or journal["pending"] is not None
                or len(journal["stages"]) != 6):
            raise RuntimeError("Incomplete or inconsistent source operation")
        if (manifest["operation_native"] != manifest["native"]
                or manifest["query_native"] != manifest["native"]):
            raise RuntimeError("Source mutations and queries used different natives")
        documents[f"{operation}_manifest"] = file_info(root / operation / "manifest.json")
        documents[f"{operation}_journal"] = file_info(root / operation / "journal.json")
        trace = Trace(operation, 10_000_000, 5_000_000, 1_000_000)
        for number, stage in enumerate(journal["stages"]):
            expected = {key: value for key, value in trace.state(number * 1_000_000).items()
                        if key != "active"}
            if (stage["stage"] != number or stage["progress"] != number * 1_000_000
                    or stage["operation"] != operation or stage["status"] != "complete"
                    or stage["state"] != expected):
                raise RuntimeError("Source stage does not match its canonical trace")
            for name in ("results", "truth", "truth_metadata", "trace", "calibration"):
                info = stage["files"][name]
                file_info(info["path"], info)
            require_unchanged(stage["files"]["checkpoint"])
            result = read_json(stage["files"]["results"]["path"])
            if (result["native"] != manifest["native"]
                    or result["operation_native"] != manifest["native"]
                    or result["query_native"] != manifest["native"]
                    or result["trace"] != trace.identity(stage["progress"])
                    or result["checkpoint"] != stage["files"]["checkpoint"]):
                raise RuntimeError("Source result/native/graph/trace identity is inconsistent")
        for info in manifest["source"].values():
            if info["path"] in source_infos and source_infos[info["path"]] != info:
                raise RuntimeError("Source operations used different versions of the same file")
            source_infos[info["path"]] = info
        operations[operation] = {"manifest": manifest, "journal": journal}
    native = operations["insert"]["manifest"]["native"]
    require_current_query_native(native["sha256"])
    for operation in operations.values():
        if (operation["manifest"]["native"] != native
                or operation["manifest"]["dataset"] != operations["insert"]["manifest"]["dataset"]):
            raise RuntimeError("Source operations do not share one native and workload")
    insert = operations["insert"]["journal"]["stages"]
    delete = operations["delete"]["journal"]["stages"]
    update = operations["point_update"]["journal"]["stages"]
    if (delete[0]["files"]["checkpoint"] != insert[5]["files"]["checkpoint"]
            or update[0]["files"]["checkpoint"] != insert[0]["files"]["checkpoint"]):
        raise RuntimeError("Archived operations lost the required shared graph lineage")
    verified = {
        path: verify_source_copy(info, snapshot["source_inventory"] if snapshot else None)
        for path, info in source_infos.items()
    }
    return pipeline, operations, documents, source_infos, verified


class QueryData:
    def __init__(self, inputs, identity):
        self.inputs, self.identity = inputs, identity
        self.n, self.dim, self.k = 10_000_000, 128, 10
        if {name: info["sha256"] for name, info in inputs.items()} != INPUT_SHA:
            raise RuntimeError("Source input identities differ from canonical SIFT10M")
        for info in inputs.values():
            file_info(info["path"], info)
        self.base = read_fvecs(inputs["base_vectors"]["path"], self.n, self.dim, integer_sift=True)
        self.queries = np.ascontiguousarray(read_fvecs(inputs["queries"]["path"], 1000, self.dim))
        self.predicates = read_json(inputs["predicates"]["path"])
        self.bounds = predicate_bounds(self.predicates, 1000)
        self.attributes = MixedAttributes.load(inputs["attributes"]["path"], self.n)
        actual = {
            "query_payload_sha256": array_digest(self.queries),
            "predicates_payload_sha256": object_digest(self.predicates),
            "numeric_payload_sha256": array_digest(self.attributes.numeric),
            "label9_payload_sha256": array_digest(self.attributes.label9),
            "label12_payload_sha256": array_digest(self.attributes.label12),
        }
        if any(identity[key] != value for key, value in actual.items()):
            raise RuntimeError("Parsed query or attribute payload differs from the source run")
        self.queries.setflags(write=False)
        for values in (self.attributes.numeric, self.attributes.label9, self.attributes.label12):
            values.setflags(write=False)

    def verify(self):
        for info in self.inputs.values():
            require_unchanged(info)
        if (array_digest(self.queries) != self.identity["query_payload_sha256"]
                or object_digest(self.predicates) != self.identity["predicates_payload_sha256"]):
            raise RuntimeError("Read-only query payload changed")


def current_sources():
    directory = Path(__file__).parent
    return {name: file_info(directory / name) for name in (
        "remeasure.py", "runner.py", "protocol.py", "measurement.py",
        "aggregate.py", "gt.py", "runtime.py",
    )}


def copy_sources(output, current, originals, verified):
    copied = {"current": {}, "legacy": {}}
    for name, info in current.items():
        destination = output / "source_reference" / "current" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(info["path"], destination)
        copied["current"][name] = file_info(destination)
    for number, (path, info) in enumerate(sorted(originals.items())):
        destination = output / "source_reference" / "legacy" / f"{number:03d}-{Path(path).name}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(verified[path]["path"], destination)
        saved = file_info(destination)
        if saved["sha256"] != info["sha256"]:
            raise RuntimeError("Legacy source snapshot copy changed")
        copied["legacy"][path] = {"original": info, "copy": saved}
    return copied


def revision_row(operation, stage, original, summaries, result_info, timing, native_sha):
    row = paper_row(operation, stage, original, summaries, native_sha)
    mutation = original.get("mutation")
    if bool(stage["stage"]) != (mutation is not None):
        raise RuntimeError("Missing mutation timing or a mutation attached to stage0")
    phases = mutation["phases"] if mutation else {}
    row.update(
        reused=True, queries_retimed=True, maintenance_reused=True,
        mutations_executed=False, results=result_info,
        operation_results=stage["files"]["results"],
        source_stage=stage["stage_file"],
        maintenance_seconds=row["maintenance_wall_s"],
        maintenance_minutes=row["maintenance_wall_s"] / 60 if mutation else None,
        delete_seconds=phases["delete"]["wall_s"] if "delete" in phases else None,
        delete_scrub_seconds=row.get("delete_scrub_wall_s"),
        delete_non_scrub_seconds=row.get("delete_non_scrub_wall_s"),
        add_seconds=phases["add"]["wall_s"] if "add" in phases else None,
        selected_query_core=timing["attempts"][timing["accepted_attempt"]]["core"],
        accepted_rounds=";".join(str(r) for r in
                                timing["attempts"][timing["accepted_attempt"]]["accepted_rounds"]),
        raw_results=result_info["path"],
    )
    if mutation:
        expected_phases = {"insert": {"add"}, "delete": {"delete"},
                           "point_update": {"add", "delete"}}[operation]
        if (set(phases) != expected_phases or mutation["records"] != 1_000_000
                or not np.isfinite(row["maintenance_seconds"]) or row["maintenance_seconds"] <= 0
                or not np.isclose(sum(value["wall_s"] for value in phases.values()),
                                  row["maintenance_seconds"], rtol=1e-12, atol=1e-12)):
            raise RuntimeError("Original mutation timing is inconsistent")
    return row


def validate_completed(state, operations, limits, node):
    expected_order = [(operation, number) for operation in OPERATIONS for number in range(6)]
    observed = [(entry["operation"], entry["stage"]) for entry in state["completed"]]
    if state["schema"] != SCHEMA or observed != expected_order[:len(observed)]:
        raise RuntimeError("Completed recall-revision stages are not a unique canonical prefix")
    for entry in state["completed"]:
        source = operations[entry["operation"]]
        stage = source["journal"]["stages"][entry["stage"]]
        for name in ("results", "calibration"):
            file_info(entry[name]["path"], entry[name])
        result = read_json(entry["results"]["path"])
        if (result["schema"] != SCHEMA or result["query"] != QUERY_CONFIG
                or result["native"] != source["manifest"]["native"]
                or result["checkpoint"] != stage["files"]["checkpoint"]
                or result["ground_truth"] != stage["files"]["truth"]
                or result["source_results"] != stage["files"]["results"]
                or result["mutations_executed"] is not False):
            raise RuntimeError("Completed query revision changed its source or target identity")
        for calibration in result["calibrations"].values():
            validate_calibration(calibration)
        verify_prefix(result["calibrations"]["current"], stage["calibrations"]["current"])
        summaries = validate_timing(result["timing"], result["calibrations"], limits, node)
        row = revision_row(
            entry["operation"], stage, read_json(stage["files"]["results"]["path"]),
            summaries, entry["results"], result["timing"], source["manifest"]["native"]["sha256"],
        )
        if entry["row"] != row:
            raise RuntimeError("Published row differs from its raw95% timing or original mutation evidence")


def run(args, output):
    source_root = Path(args.source_run).resolve(strict=True)
    saved = read_json(output / "manifest.json") if args.resume else None
    snapshot = ({"source_run": saved["request"]["source_run"],
                 "source_inventory": saved["source_copies"]["legacy"]} if saved else
                read_json(args.source_snapshot) if args.source_snapshot else None)
    pipeline, operations, documents, legacy_sources, verified_sources = read_source(source_root, snapshot)
    config = pipeline["config"]
    environment = environment_config(
        config["environment"]["node"], config["threads_update"],
        requested_cores=config["environment"]["query_candidates"],
        update_cpus=config["environment"]["update_cpus"],
    )
    native_info = operations["insert"]["manifest"]["native"]
    native, loaded = import_native(str(Path(native_info["path"]).parent), native_info["sha256"])
    if loaded != native_info:
        raise RuntimeError("Remeasurement loaded a different native")
    sources = current_sources()
    request = {
        "schema": SCHEMA, "source_run": str(source_root), "source_documents": documents,
        "source_protocol": config["protocol"], "target_protocol": PROTOCOL,
        "query": QUERY_CONFIG, "native": native_info, "source": sources,
        "legacy_source": legacy_sources, "limits": config["limits"],
        "environment": environment, "mutations_executed": False,
        "maintenance_policy": "Reuse original native mutation timings for the exact archived graph states",
    }
    manifest_path = output / "manifest.json"
    journal_path = output / "journal.json"
    if args.resume:
        # Process-local telemetry can differ on resume, but CPU placement must not.
        comparable = {key: value for key, value in saved["request"].items() if key != "environment"}
        if comparable != {key: value for key, value in request.items() if key != "environment"}:
            raise RuntimeError("Remeasurement resume changed its immutable source/protocol identity")
        for key in ("node", "update_cpus", "query_candidates", "siblings"):
            if saved["request"]["environment"][key] != environment[key]:
                raise RuntimeError("Remeasurement resume changed CPU placement")
        state = read_json(journal_path)
        for name, info in saved["source_copies"]["current"].items():
            if file_info(info["path"], info)["sha256"] != sources[name]["sha256"]:
                raise RuntimeError("Current-source snapshot changed")
        validate_completed(state, operations, config["limits"], environment["node"])
    else:
        copied = copy_sources(output, sources, legacy_sources, verified_sources)
        immutable_json(manifest_path, {"request": request, "source_copies": copied})
        state = {"schema": SCHEMA, "created_at": timestamp(), "status": "running",
                 "completed": [], "attempts": [], "pending": None}
        immutable_json(journal_path, state)

    def progress(phase, **details):
        atomic_json(output / "status.json", {
            "status": "running", "phase": phase, "completed_stages": len(state["completed"]), **details})

    def save_state():
        state["updated_at"] = timestamp()
        atomic_json(journal_path, state)

    if len(state["completed"]) == 18:
        publish(output, state)
        return
    state["status"] = "running"
    save_state()
    progress("loading_source_data")
    data = QueryData(pipeline["inputs"], operations["insert"]["manifest"]["dataset"])
    verified_checkpoints = {}

    def verify_checkpoint(stage, trace):
        info = stage["files"]["checkpoint"]
        if info["path"] not in verified_checkpoints:
            progress("verifying_checkpoint", operation=trace.operation, stage=stage["stage"])
            file_info(info["path"], info)
            header = read_index_header(info["path"])
            if (header["count"] != stage["state"]["occupied"]
                    or header["max_elements"] != data.n or header["M"] != 40
                    or header["ef_construction"] != 300 or header["format"] != 10):
                raise RuntimeError("Archived checkpoint has incompatible graph semantics")
            verify_index_records(info["path"], header, trace.state(stage["progress"])["active"],
                                 data.attributes, data.base)
            verified_checkpoints[info["path"]] = info
        elif verified_checkpoints[info["path"]] != info:
            raise RuntimeError("Conflicting identities for one archived checkpoint")
        require_unchanged(info)

    completed = {(entry["operation"], entry["stage"]): entry for entry in state["completed"]}
    for operation in OPERATIONS:
        source = operations[operation]
        if all((operation, number) in completed for number in range(6)):
            continue
        op_dir = output / operation
        op_dir.mkdir(exist_ok=True)
        target_config = {**source["manifest"]["config"], "protocol": PROTOCOL,
                         "query": copy.deepcopy(QUERY_CONFIG), "environment": environment}
        view = SimpleNamespace(
            manifest={"config": target_config, "native": native_info, "source": sources},
            action=lambda phase, **details: progress(phase, operation=operation, **details),
        )
        runner = OperationRunner(view, data, native, levels=None)
        stages = source["journal"]["stages"]
        verify_checkpoint(stages[0], runner.trace)
        runner.load("initial", stages[0]["files"]["checkpoint"])
        initial_truth = runner.load_truth(stages[0])
        initial_dir = op_dir / f"initial-calibration-{uuid.uuid4().hex}"
        initial_dir.mkdir()
        progress("calibrating_initial", operation=operation)
        initial_state, initial_labels, initial_distances = runner.calibrate(
            "initial", initial_truth, initial_dir)
        verify_prefix(initial_state, stages[0]["calibrations"]["initial"])
        validate_calibration(initial_state)
        for stage in stages:
            number = stage["stage"]
            if (operation, number) in completed:
                continue
            directory = op_dir / f"stage-{number}-{uuid.uuid4().hex}"
            directory.mkdir()
            state["pending"] = {"operation": operation, "stage": number, "directory": str(directory)}
            state["attempts"].append(copy.deepcopy(state["pending"]))
            save_state()
            runner.progress = stage["progress"]
            verify_checkpoint(stage, runner.trace)
            runner.load("current", stage["files"]["checkpoint"])
            truth = runner.load_truth(stage)
            original = read_json(stage["files"]["results"]["path"])
            if array_digest(truth) != original["ground_truth_details"]["used_labels_sha256"]:
                raise RuntimeError("Source GT payload identity changed")
            reference = initial_state if number == 0 else None
            progress("calibrating_current", operation=operation, stage=number)
            current, labels, distances = runner.calibrate("current", truth, directory, reference)
            verify_prefix(current, stage["calibrations"]["current"])
            validate_calibration(current)
            if operation == "delete" and number == 0:
                insertion = completed.get(("insert", 5))
                if insertion is None:
                    raise RuntimeError("Insertion-final95% evidence must precede deletion")
                prior = read_json(insertion["results"]["path"])
                verify_prefix(current, prior["calibrations"]["current"])
                if len(current["points"]) != len(prior["calibrations"]["current"]["points"]):
                    raise RuntimeError("Shared insert/delete graph changed its95% calibration")
            states = {"initial": copy.deepcopy(initial_state), "current": current}
            calibration = write_npz(
                directory / "calibration.npz", initial_labels=initial_labels,
                initial_distances=initial_distances, current_labels=labels, current_distances=distances,
            )
            progress("timing", operation=operation, stage=number)
            timing = runner.measure(states, truth, initial_truth, directory)
            result = {
                "schema": SCHEMA, "operation": operation, "stage": number, "progress": stage["progress"],
                "state": stage["state"], "query": QUERY_CONFIG, "native": native_info,
                "checkpoint": stage["files"]["checkpoint"], "ground_truth": stage["files"]["truth"],
                "source_results": stage["files"]["results"], "source_stage": stage["stage_file"],
                "source_query_prefix_identical": True, "mutations_executed": False,
                "maintenance_reused": True, "mutation": original.get("mutation"),
                "ground_truth_details": original["ground_truth_details"],
                "calibrations": states, "timing": timing, "query_payload_sha256": array_digest(data.queries),
            }
            result_info = immutable_json(directory / "results.json", result)
            if timing["status"] != "complete":
                state["status"] = "needs_timing"
                state["pending"]["results"] = result_info
                save_state()
                raise RuntimeError(f"{operation}/{number}: no CPU-clean95% timing; preserved for --resume")
            summaries = validate_timing(timing, states, config["limits"], environment["node"])
            row = revision_row(operation, stage, original, summaries, result_info, timing, native_info["sha256"])
            entry = {"operation": operation, "stage": number, "results": result_info,
                     "calibration": calibration, "row": row}
            completed[(operation, number)] = entry
            state["completed"].append(entry)
            state["pending"] = None
            save_state()
            progress("stage_complete", operation=operation, stage=number)
            print(json.dumps({"operation": operation, "stage": number, "qps95": row["qps95"],
                              "recall": row["recall"], "ef": row["ef"]}), flush=True)
            del runner.indexes["current"]
            gc.collect()
        runner.indexes.clear()
        del runner
        gc.collect()
    data.verify()
    for info in (*verified_checkpoints.values(), *documents.values(), native_info, *sources.values()):
        require_unchanged(info)
    validate_completed(state, operations, config["limits"], environment["node"])
    publish(output, state)


def publish(output, state):
    expected = {(operation, number) for operation in OPERATIONS for number in range(6)}
    if {(entry["operation"], entry["stage"]) for entry in state["completed"]} != expected:
        raise RuntimeError("Cannot publish an incomplete95%-recall revision")
    rows = [entry["row"] for entry in state["completed"]]
    if len(rows) != 18 or any(row["recall"] < .95 for row in rows):
        raise RuntimeError("Missing complete95%-recall stage rows")
    pending = output / f".stage_metrics.{uuid.uuid4().hex}.pending"
    with pending.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, output / "stage_metrics.csv")
    atomic_json(output / "paper.json", {
        "schema": SCHEMA, "status": "complete", "protocol": PROTOCOL,
        "query": QUERY_CONFIG, "queries_retimed": True, "maintenance_reused": True,
        "mutations_executed": False, "rows": rows,
        "stage_metrics": file_info(output / "stage_metrics.csv"),
        "provenance": file_info(output / "manifest.json"),
    })
    state["status"], state["pending"], state["updated_at"] = "complete", None, timestamp()
    atomic_json(output / "journal.json", state)
    atomic_json(output / "status.json", {"status": "complete", "completed_stages": 18})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--source-snapshot",
                        help="Verified original-source inventory when repository sources have since changed")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    source_root = Path(args.source_run).resolve(strict=True)
    if output == source_root or source_root in output.parents or output in source_root.parents:
        raise ValueError("Recall-revision output must be separate from the frozen source tree")
    if args.resume:
        if not output.is_dir():
            raise RuntimeError("No existing recall-revision output to resume")
    else:
        output.mkdir(parents=True, exist_ok=False)
    with OutputLock(output):
        try:
            run(args, output)
        except (OSError, ValueError, RuntimeError) as error:
            atomic_json(output / "status.json", {
                "status": "failed", "error_type": type(error).__name__, "error": str(error)})
            raise


if __name__ == "__main__":
    main()
