"""Validate complete operation evidence and atomically publish the three paper series."""

import csv
import io
import json
import math
import os
from pathlib import Path
import statistics
import uuid

import numpy as np

from .gt import array_digest
from .measurement import bracket, summarize_curve, validate_calibration
from .protocol import (
    INPUT_SHA, OPERATIONS, PROTOCOL, QUERY_CONFIG, QUERY_IMPLEMENTATION,
    QUERY_SHA, REBUILD_POLICY, TARGET_RECALL, TRACE_VERSIONS, Trace, require_current_query_native,
)
from .runtime import (
    CANDIDATE_ORDER, DISTANCE_ORDER_VERSION, MARKER_OWNER_VERSION,
    NUMERIC_EDGE_FORMAT, NUMERIC_MARKER_SEMANTICS, NUMERIC_MARKER_VERSION,
    PICKLE_STATE_VERSION, SCHEMA, atomic_json, file_info,
    immutable_json, noise_reasons, read_index_header, require_local, sync_directory,
)


DELETE_NATIVE_SHA = "4e6efcbe5c88e60f035c9dfdb146c5d93c8ba53143cfbb71cbdfcdd13d85e6d4"
CURRENT_SCHEMA = "canonical-dynamic-paper-current-v6"


def close(actual, expected):
    if actual is None or expected is None:
        return actual is expected
    return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)


def verify_info(info):
    file_info(info["path"], info)


def validate_timing(timing, states, limits, node):
    if (timing.get("status") != "complete" or timing.get("accepted_attempt") is None
            or timing.get("official_timing") is not True or timing.get("target_recall") != TARGET_RECALL):
        raise RuntimeError("Missing CPU-clean complete official recall95 timing")
    for state in states.values():
        validate_calibration(state)
    if timing["limits"] != limits or not 0 <= timing["accepted_attempt"] < limits["max_cores"]:
        raise RuntimeError("Incompatible noise controls or accepted attempt")
    require_local(timing["numa_before"], node)
    require_local(timing["numa_after"], node)
    expected_cases = {f"{name}_ef{p['ef']}": p for name in ("initial", "current") for p in bracket(states[name])}
    accepted_number = timing["accepted_attempt"]
    if len(timing["attempts"]) != accepted_number + 1:
        raise RuntimeError("Timed attempt selection did not stop at first successful core")
    medians = None
    seen_cores = set()
    for i, attempt in enumerate(timing["attempts"]):
        if attempt["core"] in seen_cores or set(attempt["samples"]) != set(expected_cases):
            raise RuntimeError("Cases or cores changed during paired timing")
        seen_cores.add(attempt["core"])
        lengths = {len(values) for values in attempt["samples"].values()}
        if len(lengths) != 1:
            raise RuntimeError("Incomplete paired round")
        rounds = lengths.pop()
        if not 1 <= rounds <= limits["max_rounds_per_core"]:
            raise RuntimeError("Invalid paired-round count")
        clean_rounds = []
        for repeat in range(rounds):
            clean = True
            for name, point in expected_cases.items():
                sample = attempt["samples"][name][repeat]
                sample_clean = not noise_reasons(sample, attempt["core"], limits)
                if (sample["repeat"] != repeat or sample["clean"] != sample_clean
                        or sample["ef"] != point["ef"] or sample["recall"] != point["recall"]
                        or sample["labels_sha256"] != point["labels_sha256"]
                        or sample["distances_sha256"] != point["distances_sha256"]
                        or sample["query_payload_sha256"] != QUERY_SHA
                        or sample["wall_s"] <= 0 or not close(sample["qps"], 1000 / sample["wall_s"])):
                    raise RuntimeError("Raw timing/evidence was changed or contains a different query")
                clean = clean and sample_clean
            if clean:
                clean_rounds.append(repeat)
        expected = clean_rounds[:limits["clean_rounds"]]
        if attempt["accepted_rounds"] != expected:
            raise RuntimeError("Accepted rounds are not the FIRST CPU-clean complete pairs")
        qualified = len(expected) == limits["clean_rounds"]
        if qualified != (i == accepted_number) or attempt["clean"] != qualified:
            raise RuntimeError("Cross-core pooling or skipped successful attempt")
        if qualified:
            if expected[-1] != rounds - 1:
                raise RuntimeError("Timing continued after first required clean rounds")
            medians = {name: statistics.median(attempt["samples"][name][r]["qps"] for r in expected)
                       for name in expected_cases}
            if any(not close(attempt["median_qps"][name], value) for name, value in medians.items()):
                raise RuntimeError("QPS summary does not match accepted raw samples")
    summaries = {}
    for name in ("initial", "current"):
        curve = [(p["ef"], p["recall"], medians[f"{name}_ef{p['ef']}"]) for p in bracket(states[name])]
        expected = summarize_curve(curve)
        actual = timing["summary"][name]
        if (actual.get("target_recall") != TARGET_RECALL or "qps95" not in actual or "qps90" in actual
                or not close(actual["qps95"], expected["qps95"])
                or not close(actual.get("observed_qps"), expected["observed_qps"])
                or actual["selected_ef"] != expected["selected_ef"]
                or actual["selected_recall"] != expected["selected_recall"]
                or actual.get("recall_status") != expected["recall_status"]
                or actual.get("method") != expected["method"]):
            raise RuntimeError("Invalid QPS95/observed-recall summary (including minimum-ef extrapolation)")
        summaries[name] = expected
    initial, current = (summaries[name]["qps95"] for name in ("initial", "current"))
    change = current / initial - 1 if initial and current else None
    if ("qps95_change_fraction" not in timing or "qps90_change_fraction" in timing
            or not close(timing["qps95_change_fraction"], change)):
        raise RuntimeError("Invalid paired QPS95 change")
    return summaries


def verify_gt(stage, results, *, trace=None):
    info = stage["files"]["truth"]
    with np.load(info["path"], allow_pickle=False) as arrays:
        labels, counts = arrays["labels"], arrays["matching_counts"]
        if (labels.shape != (1000, 10) or counts.shape != (1000,)
                or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0)
                or np.any(labels >= 10_000_000) or np.any(counts < 10)):
            raise RuntimeError("Invalid complete mixed GT artifact")
        details = results["ground_truth_details"]
        live = stage["state"]["live"]
        if (details["used_labels_sha256"] != array_digest(labels)
                or details["matching_counts_sha256"] != array_digest(counts)
                or np.any(counts > live) or not close(details["selectivity_live_mean"], float(counts.mean()) / live)
                or details["live"] != live):
            raise RuntimeError("GT/live-cardinality provenance mismatch")
        if trace is not None:
            progress = stage["progress"]
            if not trace.state(progress)["active"][labels].all():
                raise RuntimeError("Ground truth contains retired or not-yet-inserted source IDs")
            if not np.array_equal(arrays["logical_labels"], trace.logical_labels(labels)):
                raise RuntimeError("GT logical/fresh-external label spaces disagree")
            if details["trace"] != trace.identity(progress):
                raise RuntimeError("Ground truth used an incompatible activation/source-row trace")
    return details


def paper_row(operation, stage, results, summaries, native_sha):
    progress, counts = stage["progress"], stage["state"]
    mutation = results.get("mutation")
    wall = mutation["wall_s"] if mutation else None
    current, initial = summaries["current"], summaries["initial"]
    gt = results["ground_truth_details"]
    prefix = results.get("initial_prefix_provenance")
    return {
        "operation": operation, "stage": stage["stage"], "progress": progress, **counts,
        "maintenance_wall_s": wall, "maintenance_records": 1_000_000 if mutation else 0,
        "maintenance_records_s": 1_000_000 / wall if wall else None,
        "initial_build_wall_s": prefix["build_wall_s"] if prefix and operation == "insert" and not stage["stage"] else None,
        "initial_prefix_shared": operation == "point_update",
        "selectivity_live_min": gt["selectivity_live_min"], "selectivity_live_mean": gt["selectivity_live_mean"],
        "selectivity_live_max": gt["selectivity_live_max"], "matching_live_mean": gt["matching_live_mean"],
        "target_recall": TARGET_RECALL, "qps95": current["qps95"], "observed_qps": current["observed_qps"],
        "recall": current["selected_recall"], "ef": current["selected_ef"],
        "recall_status": current["recall_status"], "qps_method": current["method"],
        "paired_initial_qps95": initial["qps95"], "paired_initial_observed_qps": initial["observed_qps"],
        "paired_initial_recall": initial["selected_recall"], "paired_initial_ef": initial["selected_ef"],
        "same_recall_qps95_change": (current["qps95"] / initial["qps95"] - 1
                                     if current["qps95"] is not None and initial["qps95"] else None),
        "native_sha256": native_sha, "reused": False, "trace_version": TRACE_VERSIONS[operation],
        "operation_native_sha256": native_sha, "query_native_sha256": native_sha,
        "query_implementation": QUERY_IMPLEMENTATION, "index_format": NUMERIC_EDGE_FORMAT,
        "numeric_marker_version": NUMERIC_MARKER_VERSION,
        "marker_owner_version": MARKER_OWNER_VERSION, "pickle_state_version": PICKLE_STATE_VERSION,
        "distance_order_version": DISTANCE_ORDER_VERSION, "candidate_order": CANDIDATE_ORDER,
        "queries_retimed": False,
        "results": stage["files"]["results"], "ground_truth": stage["files"]["truth"],
        "operation_results": stage["files"].get("operation_results", stage["files"]["results"]),
        "checkpoint": stage["files"]["checkpoint"],
        "mutation_work_counts": mutation.get("counters") if mutation else None,
    }


def validate_operation(store, config, expected_inputs):
    manifest = store.manifest
    operation = manifest["config"]["operation"]
    if operation not in OPERATIONS or store.doc["schema"] != SCHEMA:
        raise RuntimeError("Unsupported paper operation")
    if (manifest["config"]["mode"] != "official" or store.doc["status"] != "complete"
            or store.doc["pending"] is not None or len(store.doc["stages"]) != 6
            or manifest["config"].get("protocol") != PROTOCOL
            or manifest["config"]["query"] != QUERY_CONFIG
            or manifest["config"]["limits"] != config["limits"]
            or manifest["config"]["environment"]["node"] != config["environment"]["node"]):
        raise RuntimeError("Only complete compatible official operations can enter paper output")
    require_current_query_native(manifest["native"]["sha256"])
    if (config.get("query_implementation") != QUERY_IMPLEMENTATION or config.get("rebuild_policy") != REBUILD_POLICY
            or config.get("index_format") != NUMERIC_EDGE_FORMAT
            or config.get("numeric_marker_version") != NUMERIC_MARKER_VERSION
            or config.get("marker_owner_version") != MARKER_OWNER_VERSION
            or config.get("pickle_state_version") != PICKLE_STATE_VERSION
            or config.get("distance_order_version") != DISTANCE_ORDER_VERSION
            or config.get("candidate_order") != CANDIDATE_ORDER
            or config.get("numeric_marker_semantics") != NUMERIC_MARKER_SEMANTICS):
        raise RuntimeError("Evidence predates the fresh vector-only ranking rebuild")
    if manifest["config"] != {**config, "operation": operation}:
        raise RuntimeError("Operation construction/mutation/trace settings differ from the pipeline")
    if {k: v["sha256"] for k, v in manifest["inputs"].items()} != {
            k: v["sha256"] for k, v in expected_inputs.items()}:
        raise RuntimeError("Operation used incompatible inputs")
    store.verify_committed()
    verify_info(manifest["native"])
    for info in manifest["source"].values():
        verify_info(info)
    origin = manifest.get("initial_full_provenance")
    if operation == "delete":
        if not origin or origin["checkpoint"] != manifest["initial_checkpoint"]:
            raise RuntimeError("Deletion is missing its fresh full-index provenance")
        if origin["kind"] == "completed_insert_stage5":
            for key in ("insert_manifest", "insert_journal", "insert_stage", "insert_truth", "insert_calibration"):
                verify_info(origin[key])
        elif origin["kind"] != "fresh_full_cache":
            raise RuntimeError("Legacy deletion initial indexes are forbidden")
    rows = []
    trace = Trace(operation, 10_000_000, 5_000_000, 1_000_000)
    for i, stage in enumerate(store.doc["stages"]):
        progress = i * 1_000_000
        occupied = 10_000_000 if operation == "delete" else 5_000_000 + progress
        deleted = progress if operation != "insert" else 0
        if (stage["stage"] != i or stage["progress"] != progress or stage["status"] != "complete"
                or stage["operation"] != operation or stage["state"] != {
                    "occupied": occupied, "deleted": deleted, "live": occupied - deleted}):
            raise RuntimeError("Noncanonical operation count/trace")
        results = json.loads(Path(stage["files"]["results"]["path"]).read_text())
        if (results["mode"] != "official" or results["query_payload_sha256"] != QUERY_SHA
                or results["native"] != manifest["native"]
                or results["operation_native"] != manifest["native"] or results["query_native"] != manifest["native"]
                or results["query_implementation"] != QUERY_IMPLEMENTATION
                or results["index_format"] != NUMERIC_EDGE_FORMAT
                or results["numeric_marker_version"] != NUMERIC_MARKER_VERSION
                or results["marker_owner_version"] != MARKER_OWNER_VERSION
                or results["pickle_state_version"] != PICKLE_STATE_VERSION
                or results["distance_order_version"] != DISTANCE_ORDER_VERSION
                or results["candidate_order"] != CANDIDATE_ORDER
                or results["numeric_marker_semantics"] != NUMERIC_MARKER_SEMANTICS
                or results["initial_full_provenance"] != origin
                or (operation == "delete" and origin["kind"] == "completed_insert_stage5"
                    and results["initial_matches_completed_insert"] is not True)
                or results["trace"]["version"] != TRACE_VERSIONS[operation]):
            raise RuntimeError("Stale result/native/trace provenance")
        header = read_index_header(stage["files"]["checkpoint"]["path"])
        if header["format"] != NUMERIC_EDGE_FORMAT or header["count"] != occupied or header["max_elements"] != 10_000_000:
            raise RuntimeError("Paper checkpoint is not a fresh compatible format10 vector-only graph")
        for state in stage["calibrations"].values():
            validate_calibration(state)
        summaries = validate_timing(results["timing"], stage["calibrations"], config["limits"],
                                    config["environment"]["node"])
        verify_gt(stage, results, trace=trace)
        with np.load(stage["files"]["trace"]["path"], allow_pickle=False) as arrays:
            for name, expected in trace.arrays(progress).items():
                if not np.array_equal(arrays[name], expected):
                    raise RuntimeError("Stored activation/logical mapping trace is stale")
        mutation = results.get("mutation")
        if bool(mutation) != bool(i):
            raise RuntimeError("Missing or spurious committed mutation")
        if mutation:
            phases = {"delete"} if operation == "delete" else {"add"} if operation == "insert" else {"delete", "add"}
            if (mutation["records"] != 1_000_000 or set(mutation["phases"]) != phases
                    or mutation["wall_s"] <= 0 or mutation["replace_deleted"] is not False
                    or not close(mutation["wall_s"], sum(p["wall_s"] for p in mutation["phases"].values()))):
                raise RuntimeError("Wrong mutation semantics, volume or wall-time accounting")
        rows.append(paper_row(operation, stage, results, summaries, manifest["native"]["sha256"]))
    return {"operation": operation, "status": "complete", "fully_verified": True, "reused": False,
            "native": manifest["native"], "source": manifest["source"], "protocol": PROTOCOL,
            "target_recall": TARGET_RECALL,
            "operation_native": manifest["native"], "query_native": manifest["native"],
            "operation_source": manifest["source"], "query_source": manifest["source"],
            "query_implementation": QUERY_IMPLEMENTATION, "queries_retimed": False,
            "rebuild_policy": REBUILD_POLICY, "index_format": NUMERIC_EDGE_FORMAT,
            "numeric_marker_version": NUMERIC_MARKER_VERSION,
            "marker_owner_version": MARKER_OWNER_VERSION, "pickle_state_version": PICKLE_STATE_VERSION,
            "distance_order_version": DISTANCE_ORDER_VERSION, "candidate_order": CANDIDATE_ORDER,
            "initial_checkpoint": manifest["initial_checkpoint"], "initial_full_provenance": origin,
            "initial_calibration": store.doc["stages"][0]["calibrations"]["initial"],
            "final_calibration": store.doc["stages"][-1]["calibrations"]["current"],
            "final_stage_file": store.doc["stages"][-1]["stage_file"],
            "input_sha256": {k: v["sha256"] for k, v in manifest["inputs"].items()},
            "original_manifest": file_info(store.output / "manifest.json"),
            "original_journal": file_info(store.path), "rows": rows}


def verify_shared_initial_graphs(operations):
    inserted, deleted, updated = (operations[name] for name in OPERATIONS)
    origin = deleted.get("initial_full_provenance") or {}
    if (origin.get("kind") != "completed_insert_stage5"
            or origin.get("checkpoint") != inserted["rows"][5]["checkpoint"]
            or deleted["initial_checkpoint"] != inserted["rows"][5]["checkpoint"]
            or deleted["rows"][0]["checkpoint"] != inserted["rows"][5]["checkpoint"]
            or origin.get("insert_stage") != inserted["final_stage_file"]
            or origin.get("insert_manifest") != inserted["original_manifest"]
            or origin.get("insert_journal") != inserted["original_journal"]
            or updated["initial_checkpoint"] != inserted["initial_checkpoint"]):
        raise RuntimeError("Paper operations did not share the fresh insert-final/full and original prefix checkpoints")
    points = inserted["final_calibration"]["points"]
    for reference in (origin.get("initial_query_reference", {}).get("points", []),
                      deleted["initial_calibration"]["points"]):
        if not points or len(points) != len(reference) or any(
                a[key] != b[key] for a, b in zip(points, reference)
                for key in ("ef", "recall", "labels_sha256", "distances_sha256")):
            raise RuntimeError("Delete initial query differs from insert final at identical parameters")


def publish_paper(output, operations, *, expected_native, current_path, latex=False):
    require_current_query_native(expected_native)
    if set(operations) != set(OPERATIONS):
        raise RuntimeError("Paper output requires exactly insert, delete, point_update (no attribute-only series)")
    identities = []
    for name, operation in operations.items():
        if (operation["operation"] != name or operation["status"] != "complete"
                or not operation["fully_verified"] or operation["protocol"] != PROTOCOL
                or operation.get("target_recall") != TARGET_RECALL
                or len(operation["rows"]) != 6):
            raise RuntimeError("Partial/unverified/noncanonical paper component")
        if (operation.get("query_implementation") != QUERY_IMPLEMENTATION
                or operation.get("query_native", {}).get("sha256") != expected_native
                or operation.get("operation_native") != operation["native"]):
            raise RuntimeError("Stale query curves or inconsistent operation/query native identities")
        if (operation["reused"] or operation.get("queries_retimed")
                or operation["native"]["sha256"] != expected_native
                or operation.get("rebuild_policy") != REBUILD_POLICY
                or operation.get("numeric_marker_version") != NUMERIC_MARKER_VERSION
                or operation.get("marker_owner_version") != MARKER_OWNER_VERSION
                or operation.get("pickle_state_version") != PICKLE_STATE_VERSION
                or operation.get("distance_order_version") != DISTANCE_ORDER_VERSION
                or operation.get("candidate_order") != CANDIDATE_ORDER
                or operation.get("index_format") != NUMERIC_EDGE_FORMAT):
            raise RuntimeError("Only freshly rebuilt/rerun format10 vector-only operations on one native may be published")
        identities.append(operation["input_sha256"])
        for i, row in enumerate(operation["rows"]):
            if row["operation"] != name or row["stage"] != i:
                raise RuntimeError("Mislabeled or missing operation rows")
            if (row.get("operation_native_sha256") != operation["operation_native"]["sha256"]
                    or row.get("query_native_sha256") != expected_native
                    or row.get("query_implementation") != QUERY_IMPLEMENTATION
                    or row.get("numeric_marker_version") != NUMERIC_MARKER_VERSION
                    or row.get("marker_owner_version") != MARKER_OWNER_VERSION
                    or row.get("pickle_state_version") != PICKLE_STATE_VERSION
                    or row.get("distance_order_version") != DISTANCE_ORDER_VERSION
                    or row.get("candidate_order") != CANDIDATE_ORDER
                    or row.get("queries_retimed") is not False or row.get("index_format") != NUMERIC_EDGE_FORMAT):
                raise RuntimeError("Row mixed up original mutation provenance and refreshed query evidence")
            if row.get("target_recall") != TARGET_RECALL or any("qps90" in key for key in row):
                raise RuntimeError("Paper rows require explicit recall95 evidence, never relabeled QPS90")
            for prefix in ("", "paired_initial_"):
                value = row.get(prefix + "qps95")
                ef, recall = row.get(prefix + "ef"), row.get(prefix + "recall")
                if (prefix + "qps95" not in row or ef is None or ef < QUERY_CONFIG["k"]
                        or recall is None or not TARGET_RECALL <= recall <= 1):
                    raise RuntimeError("Missing or nonqualifying QPS95 row evidence")
                above = ef == QUERY_CONFIG["k"] and recall > TARGET_RECALL
                if above and value is not None:
                    raise RuntimeError("Cannot publish extrapolated minimum-ef QPS95")
                if not above and (value is None or not math.isfinite(value) or value <= 0):
                    raise RuntimeError("Missing supported QPS95 value")
    if any(identity != identities[0] for identity in identities) or identities[0] != INPUT_SHA:
        raise RuntimeError("Canonical input identities disagree")
    verify_shared_initial_graphs(operations)
    rows = [row for name in OPERATIONS for row in operations[name]["rows"]]
    document = {
        "schema": CURRENT_SCHEMA, "protocol": PROTOCOL, "status": "complete", "target_recall": TARGET_RECALL,
        "operations": operations, "native_by_operation": {k: v["native"] for k, v in operations.items()},
        "operation_native_by_operation": {k: v["operation_native"] for k, v in operations.items()},
        "query_native_by_operation": {k: v["query_native"] for k, v in operations.items()},
        "query_implementation": QUERY_IMPLEMENTATION,
        "rebuild_policy": REBUILD_POLICY, "index_format": NUMERIC_EDGE_FORMAT,
        "numeric_marker_semantics": NUMERIC_MARKER_SEMANTICS,
        "numeric_marker_version": NUMERIC_MARKER_VERSION,
        "marker_owner_version": MARKER_OWNER_VERSION, "pickle_state_version": PICKLE_STATE_VERSION,
        "distance_order_version": DISTANCE_ORDER_VERSION, "candidate_order": CANDIDATE_ORDER,
        "mixed_native_policy": "forbidden; every mutation and query uses the same fixed native",
        "rows": rows, "input_sha256": identities[0],
    }
    output = Path(output)
    if (output / "paper.json").exists():
        existing = json.loads((output / "paper.json").read_text())
        if existing != document:
            raise RuntimeError("Refusing to overwrite a published run")
    else:
        immutable_json(output / "paper.json", document)
    fields = [key for key in rows[0] if key not in (
        "results", "operation_results", "ground_truth", "checkpoint", "mutation_work_counts")]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: "NA" if row[key] is None else row[key] for key in fields})
    write_text_once(output / "paper.tsv", stream.getvalue())
    if latex:
        lines = [r"\begin{tabular}{lrrrrrr}", r"Operation & Stage & Live & Maint. s & Recall & ef & QPS95 \\", r"\hline"]
        for row in rows:
            values = [row["operation"].replace("_", r"\_"), str(row["stage"]), str(row["live"]),
                      "--" if row["maintenance_wall_s"] is None else f"{row['maintenance_wall_s']:.3f}",
                      f"{row['recall']:.4f}", str(row["ef"]), "--" if row["qps95"] is None else f"{row['qps95']:.3f}"]
            lines.append(" & ".join(values) + r" \\")
        lines.append(r"\end{tabular}")
        write_text_once(output / "paper.tex", "\n".join(lines) + "\n")
    pointer = {"schema": CURRENT_SCHEMA, "status": "complete", "protocol": PROTOCOL,
               "target_recall": TARGET_RECALL,
               "paper_json": file_info(output / "paper.json"), "paper_tsv": file_info(output / "paper.tsv"),
               "operations": list(OPERATIONS), "operation_native_sha256": {
                   k: v["operation_native"]["sha256"] for k, v in operations.items()},
               "query_native_sha256": {k: v["query_native"]["sha256"] for k, v in operations.items()},
               "query_implementation": QUERY_IMPLEMENTATION, "rebuild_policy": REBUILD_POLICY,
               "index_format": NUMERIC_EDGE_FORMAT, "numeric_marker_version": NUMERIC_MARKER_VERSION,
               "marker_owner_version": MARKER_OWNER_VERSION, "pickle_state_version": PICKLE_STATE_VERSION,
               "distance_order_version": DISTANCE_ORDER_VERSION, "candidate_order": CANDIDATE_ORDER}
    current_path = Path(current_path)
    if current_path.is_symlink():
        raise RuntimeError("Refusing a symlinked current manifest")
    if current_path.exists():
        previous = json.loads(current_path.read_text())
        if (previous.get("schema") not in (CURRENT_SCHEMA, "canonical-dynamic-paper-current-v1",
                                         "canonical-dynamic-paper-current-v2", "canonical-dynamic-paper-current-v3",
                                         "canonical-dynamic-paper-current-v4", "canonical-dynamic-paper-current-v5")
                or previous.get("status") != "complete"
                or not {"paper_json", "paper_tsv"} <= previous.keys()
                or set(previous.get("operations", [])) != set(OPERATIONS)):
            raise RuntimeError("Refusing to overwrite anything except a canonical current-manifest pointer")
    current_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(current_path, pointer)
    return pointer


def write_text_once(path, text):
    if path.exists():
        if path.read_text() != text:
            raise RuntimeError("Refusing to overwrite immutable paper output")
        return
    pending = path.parent / f".{path.name}.{uuid.uuid4().hex}.pending"
    with pending.open("x") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.rename(pending, path)
    sync_directory(path.parent)
