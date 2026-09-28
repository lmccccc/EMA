#!/usr/bin/env python3
"""Canonical dynamic reproduction: insert, delete, point_update; never attribute-only.

Run from the repository root with:
  python -m exp_benchmark.dynamic.pipeline run --help

Use explicit NUMA binding, a SHA-pinned production native, and a fresh data-disk
output. --pilot-size scales the source to a real prefix and remains nonofficial.
Only a complete compatible three-operation run can atomically publish a current
paper manifest. Deletion starts from this run's completed insertion checkpoint,
never a legacy numeric index. This command never deletes any artifact.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import traceback
import uuid

import numpy as np

from .aggregate import publish_paper, validate_operation
from .protocol import (
    Dataset, LEVEL_POLICY, OPERATIONS, PROTOCOL, QUERY_CONFIG, QUERY_IMPLEMENTATION,
    REBUILD_POLICY, TRACE_VERSIONS, input_paths, require_current_query_native,
)
from .runner import (
    OperationRunner, completed_insert_checkpoint, full_checkpoint, prefix_checkpoint, prepare_levels,
)
from .runtime import (
    ATTRIBUTE_NODE_FORMAT, CANDIDATE_ORDER, DEFAULT_LIMITS, DISTANCE_ORDER_VERSION, LOCK_NAME,
    MARKER_OWNER_VERSION, NUMERIC_EDGE_FORMAT, NUMERIC_MARKER_SEMANTICS,
    NUMERIC_MARKER_VERSION, PICKLE_STATE_VERSION,
    OutputLock, StageStore, atomic_json, bootstrap_manifest, cpu_list,
    environment_config, file_info, immutable_json, import_native, object_digest,
    require_unchanged, sync_directory, timestamp,
)


REPOSITORY = Path(__file__).resolve().parents[2]


def positive(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def nonnegative(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return value


def sha(text):
    value = text.lower()
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise argparse.ArgumentTypeError("expected SHA256")
    return value


def frequency_range(text):
    try:
        low, high = (int(value) for value in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("use MIN_KHZ,MAX_KHZ") from error
    if not 0 < low <= high:
        raise argparse.ArgumentTypeError("frequency interval must be positive and ordered")
    return [low, high]


def operation_list(text):
    values = text.split(",")
    if len(set(values)) != len(values) or not values or not set(values) <= set(OPERATIONS):
        raise argparse.ArgumentTypeError("only insert,delete,point_update; no duplicates")
    return [name for name in OPERATIONS if name in values]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="build/share a prefix and execute canonical operations")
    run.add_argument("--data-root", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True, help="fresh versioned directory on /mnt/data")
    run.add_argument("--native-dir", type=Path, required=True)
    run.add_argument("--expected-sha256", type=sha, required=True)
    run.add_argument("--operations", type=operation_list, default=list(OPERATIONS))
    run.add_argument("--initial-prefix", type=Path, help="explicit compatible prefix cache; never silently rebuilt")
    run.add_argument("--initial-full", type=Path,
                     help="explicit compatible format10 full-cache directory for delete without insert; otherwise build fresh")
    run.add_argument("--numa-node", type=nonnegative, required=True)
    run.add_argument("--update-cpus", type=cpu_list)
    run.add_argument("--query-cores", type=cpu_list)
    run.add_argument("--frequency-khz", type=frequency_range, help="required for official timing; no host clock default")
    run.add_argument("--threads-update", type=positive, default=32)
    run.add_argument("--threads-build", type=positive, default=32)
    run.add_argument("--threads-gt", type=positive, default=32)
    run.add_argument("--minimum-thread-cpu-wall", type=float, default=.98)
    run.add_argument("--maximum-sibling-busy", type=float, default=.05)
    run.add_argument("--clean-rounds", type=positive, default=7)
    run.add_argument("--max-rounds-per-core", type=positive, default=21)
    run.add_argument("--max-cores", type=positive, default=3)
    run.add_argument("--max-query-ef", type=positive, default=3000)
    run.add_argument("--level-seed", type=nonnegative, default=1234)
    run.add_argument("--add-chunk", type=positive, default=100_000)
    run.add_argument("--pilot-size", type=positive, help="nonofficial real source prefix, >=1000 and divisible by10")
    run.add_argument("--through-stage", type=int, choices=range(6))
    run.add_argument("--resume", action="store_true")
    run.add_argument("--publish-current", type=Path, help="atomic pointer updated ONLY after all3 operations succeed")
    run.add_argument("--latex", action="store_true")
    aggregate = sub.add_parser("aggregate", help="revalidate and publish an already complete pipeline")
    aggregate.add_argument("--run", type=Path, required=True)
    aggregate.add_argument("--publish-current", type=Path, required=True)
    aggregate.add_argument("--latex", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "run":
        if args.pilot_size and (args.pilot_size < 1000 or args.pilot_size >= 10_000_000 or args.pilot_size % 10):
            parser.error("--pilot-size must be >=1000, <10M and divisible by10")
        if args.initial_full and ("delete" not in args.operations or "insert" in args.operations):
            parser.error("--initial-full is only for delete without insert; combined runs use insert stage5")
        if args.initial_prefix and not any(name in args.operations for name in ("insert", "point_update")):
            parser.error("--initial-prefix requires insert or point_update")
        if not args.pilot_size and args.frequency_khz is None:
            parser.error("official timing requires explicit --frequency-khz MIN,MAX")
        if (not 0 < args.minimum_thread_cpu_wall <= 1 or not 0 <= args.maximum_sibling_busy <= 1
                or args.clean_rounds > args.max_rounds_per_core or args.max_query_ef < 10):
            parser.error("Invalid noise/calibration controls")
    return args


def source_inventory():
    files = sorted(Path(__file__).parent.glob("*.py"))
    files += [REPOSITORY / "exp_benchmark/aggregate.py", REPOSITORY / "tests/hashann.py"]
    return {str(path.relative_to(REPOSITORY)): file_info(path) for path in files}


def native_source_inventory(directory, native_info):
    path = Path(directory).resolve() / "provenance.json"
    if not path.exists():
        return {}
    info = file_info(path)
    saved = json.loads(path.read_text())
    edge_formats = [saved[key] for key in (
        "numeric_edge_file_format", "categorical_edge_file_format", "edge_file_format", "edge_format_version")
        if key in saved]
    node_formats = [saved[key] for key in (
        "numeric_node_file_format", "categorical_node_file_format", "node_file_format", "node_format_version")
        if key in saved]
    state_versions = [saved[key] for key in ("pickle_state_version", "serialization_version") if key in saved]
    if (saved.get("native_sha256") != native_info["sha256"]
            or saved.get("numeric_marker_version") != NUMERIC_MARKER_VERSION
            or saved.get("marker_owner_version") != MARKER_OWNER_VERSION
            or saved.get("distance_order_version") != DISTANCE_ORDER_VERSION
            or saved.get("candidate_order") != CANDIDATE_ORDER
            or not edge_formats or any(value != NUMERIC_EDGE_FORMAT for value in edge_formats)
            or not node_formats or any(value != ATTRIBUTE_NODE_FORMAT for value in node_formats)
            or not state_versions or any(value != PICKLE_STATE_VERSION for value in state_versions)):
        raise RuntimeError("Frozen native build provenance/version differs from the selected extension")
    sources = {"native/provenance.json": info}
    for keys, relative in (
            (("hnswalg_sha256", "header_sha256"), "hnswlib/hnswalg.h"),
            (("bindings_sha256",), "python_bindings/bindings.cpp"),
            (("setup_sha256",), "setup.py")):
        source = file_info(path.parent / relative)
        hashes = [saved[key] for key in keys if key in saved]
        if not hashes or any(value != source["sha256"] for value in hashes):
            raise RuntimeError("Frozen native source differs from its build provenance")
        sources[f"native/{relative}"] = source
    auxiliary = saved.get("auxiliary_source_sha256", {})
    if not isinstance(auxiliary, dict):
        raise RuntimeError("Frozen auxiliary source provenance must be a relative-path/SHA256 mapping")
    for relative, expected_sha in auxiliary.items():
        if (not isinstance(relative, str) or not relative or "\\" in relative or "\0" in relative
                or Path(relative).is_absolute() or ".." in Path(relative).parts or str(Path(relative)) != relative
                or not isinstance(expected_sha, str) or len(expected_sha) != 64
                or any(character not in "0123456789abcdef" for character in expected_sha)):
            raise RuntimeError("Invalid frozen auxiliary source path/SHA256")
        source_path = path.parent / relative
        if source_path.resolve() != source_path:
            raise RuntimeError("Frozen auxiliary sources must not use symlinks or escape the native directory")
        source = file_info(source_path)
        if source["sha256"] != expected_sha:
            raise RuntimeError("Frozen auxiliary source differs from its build provenance")
        sources[f"native/{relative}"] = source
    return sources


def configuration(args, environment):
    capacity = args.pilot_size or 10_000_000
    limits = {**DEFAULT_LIMITS, "frequency_khz": args.frequency_khz,
              "thread_cpu_wall_min": args.minimum_thread_cpu_wall, "sibling_busy_max": args.maximum_sibling_busy,
              "clean_rounds": args.clean_rounds, "max_rounds_per_core": args.max_rounds_per_core,
              "max_cores": args.max_cores}
    operations = args.operations
    return {
        "protocol": PROTOCOL, "mode": "pilot" if args.pilot_size else "official",
        "capacity": capacity, "initial": capacity // 2, "step": capacity // 10,
        "stage_progress": list(range(0, capacity // 2 + 1, capacity // 10)),
        "query": copy.deepcopy(QUERY_CONFIG), "operations": operations,
        "query_implementation": QUERY_IMPLEMENTATION, "rebuild_policy": REBUILD_POLICY,
        "index_format": NUMERIC_EDGE_FORMAT, "numeric_marker_semantics": NUMERIC_MARKER_SEMANTICS,
        "numeric_marker_version": NUMERIC_MARKER_VERSION,
        "marker_owner_version": MARKER_OWNER_VERSION, "pickle_state_version": PICKLE_STATE_VERSION,
        "distance_order_version": DISTANCE_ORDER_VERSION, "candidate_order": CANDIDATE_ORDER,
        "initial_prefix": str(args.initial_prefix.resolve()) if args.initial_prefix else None,
        "initial_full": str(args.initial_full.resolve()) if args.initial_full else None,
        "deletion_initial": "completed_insert_stage5" if "insert" in operations else "fresh_full_cache",
        "environment": environment, "limits": limits,
        "threads_update": args.threads_update, "threads_build": args.threads_build, "threads_gt": args.threads_gt,
        "max_query_ef": args.max_query_ef, "level_seed": args.level_seed, "level_policy": LEVEL_POLICY,
        "add_chunk": args.add_chunk, "trace_versions": TRACE_VERSIONS,
        "deletion": "synchronous delete_items structural repair + complete-record Markers + distance-free scrub",
        "point_update": "complete original source record; delete old label, then fresh add(replace_deleted=False)",
        "slot_reuse": False, "numpy_version": np.__version__,
    }


def prepare_output(path, resume):
    path = path.expanduser().absolute()
    if path.resolve() != path or Path("/mnt/data") not in path.parents:
        raise RuntimeError("Use a nonsymlinked output directory on the data disk, not /home or temporary storage")
    if resume:
        if not (path / "pipeline-manifest.json").is_file():
            raise RuntimeError("No canonical pipeline to resume")
        if (path / "pipeline.json").is_symlink():
            raise RuntimeError("Symlinked pipeline state")
    elif path.exists() and any(p.name != LOCK_NAME for p in path.iterdir()):
        raise RuntimeError("Use a fresh versioned output; never overwrite prior active results")
    path.mkdir(parents=True, exist_ok=True)
    sync_directory(path.parent)
    return path


def bootstrap_pipeline(output, manifest, *, resume=False, fault=None):
    manifest_path, state_path = output / "pipeline-manifest.json", output / "pipeline.json"
    if state_path.is_symlink():
        raise RuntimeError("Symlinked pipeline state")
    if state_path.exists():
        if not resume:
            raise RuntimeError("Pipeline state already exists; use resume")
        state = json.loads(state_path.read_text())
        file_info(manifest_path, state["manifest"])
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError("Resume binary/source/input/trace/config mismatch")
        return state
    info, recovered = bootstrap_manifest(manifest_path, manifest, resume=resume)
    if fault:
        fault("after_pipeline_manifest")
    # Existing prefix caches and operation journals are validated/reconciled by
    # run(), not overwritten or declared complete by an empty top-level state.
    state = {"status": "running", "manifest": info, "operations": {}, "invocations": [],
             "bootstrap_recovered": recovered}
    atomic_json(state_path, state)
    return state


def native_canary(root, args):
    from .canary import LIFECYCLE_CHECKS, SCHEMA as CANARY_SCHEMA

    require_current_query_native(args.expected_sha256)
    directory = root / f"native-canary-{uuid.uuid4().hex}"
    command = [
        sys.executable, "-B", "-m", "exp_benchmark.dynamic.canary",
        "--native-dir", str(args.native_dir.resolve()), "--expected-sha256", args.expected_sha256,
        "--output", str(directory),
    ]
    log = root / f"{directory.name}.log"
    with log.open("x") as stream:
        result = subprocess.run(command, cwd=REPOSITORY, stdout=stream, stderr=subprocess.STDOUT,
                                timeout=120, check=False)
        stream.flush()
        os.fsync(stream.fileno())
    if result.returncode:
        raise RuntimeError(f"Isolated load/add canary failed ({result.returncode}); fix native, no rebuild fallback: {log}")
    info = file_info(directory / "report.json")
    report = json.loads(Path(info["path"]).read_text())
    if (report.get("schema") != CANARY_SCHEMA or report["status"] != "complete"
            or report["native"]["sha256"] != args.expected_sha256
            or any(report.get("lifecycle", {}).get(check) is not True for check in LIFECYCLE_CHECKS)
            or not 0 <= report.get("ann", {}).get("recall", -1) <= 1):
        raise RuntimeError("Native canary identity/result mismatch")
    if report["warnings"]:
        print(json.dumps({"native_canary_lifecycle": "passed", "ann_recall": report["ann"]["recall"],
                          "warnings": report["warnings"], "evidence": info["path"]}), flush=True)
    if (
            report.get("index_format") != NUMERIC_EDGE_FORMAT
            or report.get("numeric_marker_semantics") != NUMERIC_MARKER_SEMANTICS
            or report.get("numeric_marker_version") != NUMERIC_MARKER_VERSION
            or report.get("marker_owner_version") != MARKER_OWNER_VERSION
            or report.get("pickle_state_version") != PICKLE_STATE_VERSION
            or report.get("distance_order_version") != DISTANCE_ORDER_VERSION
            or report.get("candidate_order") != CANDIDATE_ORDER
            or report.get("attr_sort_alpha") != 0):
        raise RuntimeError("Native lifecycle passed but format10, numeric2/owner2/distance-order1/state4 "
                           "and vector-only ranking are required; legacy graphs/native builds cannot be reused")
    return info


def read_operation(root, operation):
    return StageStore(root / operation, resume=True)


def publish_existing(root, current_path, latex=False):
    manifest = json.loads((root / "pipeline-manifest.json").read_text())
    state = json.loads((root / "pipeline.json").read_text())
    if (state["status"] != "complete" or manifest["config"]["mode"] != "official"
            or manifest.get("schema") != PROTOCOL):
        raise RuntimeError("Cannot publish a partial pipeline or a pilot")
    file_info(root / "pipeline-manifest.json", state["manifest"])
    operations = {}
    config, inputs = manifest["config"], manifest["inputs"]
    for name in config["operations"]:
        operations[name] = validate_operation(read_operation(root, name), config, inputs)
    return publish_paper(root, operations, expected_native=manifest["native"]["sha256"],
                         current_path=current_path, latex=latex)


def run(args, *, fault=None):
    require_current_query_native(args.expected_sha256)
    output = prepare_output(args.output, args.resume)
    accepted, active_store = False, None
    invocation_path = output / f"invocation-{uuid.uuid4().hex}.json"
    invocation = {"at": timestamp(), "status": "preflight", "resume": args.resume,
                  "through_stage": args.through_stage}

    def progress(action, **details):
        invocation["action"] = {"name": action, **details}
        atomic_json(invocation_path, invocation)

    with OutputLock(output):
        try:
            environment = environment_config(args.numa_node, args.threads_update, args.query_cores, args.update_cpus)
            if max(args.threads_gt, args.threads_build) > len(environment["update_cpus"]):
                raise RuntimeError("Build/GT thread counts exceed the chosen local CPU set")
            os.sched_setaffinity(0, environment["update_cpus"])
            config = configuration(args, environment)
            native, module = import_native(args.native_dir, args.expected_sha256)
            sources = source_inventory()
            sources.update(native_source_inventory(args.native_dir, module))
            progress("isolated_native_format10_distance_order_v1_lifecycle_gate")
            canary = native_canary(output, args)
            paths = input_paths(args.data_root)
            data = Dataset(paths, config, progress)
            manifest = {
                "schema": PROTOCOL, "config": config, "config_sha256": object_digest(config),
                "dataset": data.identity, "inputs": data.inputs, "native": module, "source": sources,
            }
            state_path = output / "pipeline.json"
            state = bootstrap_pipeline(output, manifest, resume=args.resume, fault=fault)
            accepted = True
            state["invocations"].append({"path": str(invocation_path), "native_canary": canary,
                                         "resume": args.resume, "through_stage": args.through_stage})
            state["status"] = "running"
            atomic_json(state_path, state)
            progress("deterministic_production_level_plan")
            levels, plan = prepare_levels(output, data, config)
            prefix = None
            if any(name != "delete" for name in config["operations"]):
                progress("build_or_verify_shared_first_prefix")
                prefix = prefix_checkpoint(output, data, config, native, module, levels, plan, sources,
                                           explicit=args.initial_prefix)
            for operation in config["operations"]:
                directory = output / operation
                directory.mkdir(exist_ok=True)
                full = None
                if operation == "delete":
                    if "insert" in config["operations"]:
                        full = completed_insert_checkpoint(output, data, config, module, sources)
                    else:
                        full = {"kind": "fresh_full_cache", **full_checkpoint(
                            output, data, config, native, module, levels, plan, sources, explicit=args.initial_full)}
                    initial = full["checkpoint"]
                else:
                    initial = prefix["checkpoint"]
                operation_config = {**config, "operation": operation}
                operation_manifest = {
                    "config": operation_config, "inputs": data.inputs, "dataset": data.identity,
                    "native": module, "source": sources, "initial_checkpoint": initial,
                    "operation_native": module, "query_native": module,
                    "operation_source": sources, "query_source": sources,
                    "initial_prefix_provenance": prefix if operation != "delete" else None,
                    "initial_full_provenance": full,
                }
                operation_fault = (lambda event: fault(f"operation_{operation}:{event}")) if fault else None
                if (directory / "journal.json").exists():
                    active_store = StageStore(directory, manifest=operation_manifest, resume=True, fault=operation_fault)
                    active_store.verify_committed()
                    if active_store.doc["status"] == "complete":
                        state["operations"][operation] = {"status": "complete", "reused": False,
                                                         "journal": file_info(active_store.path)}
                        atomic_json(state_path, state)
                        active_store = None
                        continue
                    if args.through_stage is not None and args.through_stage < len(active_store.doc["stages"]) - 1:
                        raise RuntimeError("--through-stage is behind the committed operation")
                    active_store.recover_pending()
                else:
                    active_store = StageStore(directory, manifest=operation_manifest, fault=operation_fault)
                worker = OperationRunner(active_store, data, native, levels)
                complete = worker.run(args.through_stage)
                state["operations"][operation] = {"status": active_store.doc["status"], "reused": False,
                                                 "journal": file_info(active_store.path)}
                atomic_json(state_path, state)
                if not complete:
                    state["status"] = active_store.doc["status"]
                    atomic_json(state_path, state)
                    invocation["status"] = state["status"]
                    atomic_json(invocation_path, invocation)
                    return 2 if state["status"] == "paused_noise" else 0
                del worker
                active_store = None
            all_three = set(state["operations"]) == set(OPERATIONS)
            state["status"] = "complete" if all_three else "scope_complete_no_paper"
            atomic_json(state_path, state)
            if all_three and config["mode"] == "official":
                publish_existing(output, args.publish_current or output / "current.json", args.latex)
            elif config["mode"] == "pilot":
                atomic_json(output / "pilot-summary.json", {
                    "status": state["status"], "official": False, "capacity": data.n,
                    "operations": state["operations"], "paper_publication": "forbidden for pilots",
                })
            invocation["status"] = state["status"]
            atomic_json(invocation_path, invocation)
            return 0
        except BaseException as error:
            invocation.update(status="failed", error=str(error), traceback="".join(traceback.format_exception(error)))
            atomic_json(invocation_path, invocation)
            if accepted:
                state["status"] = "failed"
                atomic_json(output / "pipeline.json", state)
            if active_store is not None:
                active_store.record_error(error)
            raise


def main(argv=None):
    args = parse_args(argv)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    def interrupted(signum, frame):
        raise InterruptedError(f"Signal{signum}; uncommitted mutations replay from the prior checkpoint")

    signal.signal(signal.SIGTERM, interrupted)
    if args.command == "run":
        return run(args)
    if args.command == "aggregate":
        root = args.run.resolve(strict=True)
        if (not (root / "pipeline-manifest.json").is_file() or not (root / "pipeline.json").is_file()
                or json.loads((root / "pipeline-manifest.json").read_text()).get("schema") != PROTOCOL):
            raise RuntimeError("Aggregate requires a canonical pipeline, not an arbitrary/legacy output directory")
        with OutputLock(args.run):
            result = publish_existing(root, args.publish_current, args.latex)
        print(json.dumps(result, indent=2))
        return 0
    raise RuntimeError("Unsupported reproduction command")


if __name__ == "__main__":
    sys.exit(main())
