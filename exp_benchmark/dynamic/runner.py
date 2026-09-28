"""One stage engine for insertion, structural deletion and full-record replacement."""

import copy
import json
import os
from pathlib import Path
import resource
import shutil
import time
import uuid

import numpy as np

from .gt import array_digest
from .measurement import FrozenQuery, bracket, calibrate, paired_timing
from .protocol import LEVEL_POLICY, ORDER_SHA, Trace, representative_levels
from .runtime import (
    CANDIDATE_ORDER, DISTANCE_ORDER_VERSION, MARKER_OWNER_VERSION,
    NUMERIC_EDGE_FORMAT, NUMERIC_MARKER_SEMANTICS, NUMERIC_MARKER_VERSION,
    PICKLE_STATE_VERSION, StageStore, atomic_json, file_info,
    immutable_json, mutation_timing, native_state, numa_snapshot,
    object_digest, read_index_header, require_local, require_unchanged, sync_directory,
    split_deletion_stats, timestamp, usage_delta, verify_index_records, verify_index_tail, write_npz,
)


COUNTER_NOTE = "Native work counters, NOT unique edges/nodes. Deletion width is loaded ef_construction=300 only."


def configure_index(index):
    index.generateAttrIndexes()
    index.set_thresholds(0.0001, 0.0001, 0.0001)
    index.set_ef_top(1)
    index.set_ft_flag(True)
    index.set_ft_routing_flag(True)
    index.set_ft_routing_min_deg(8)
    index.set_ft_routing_backfill_tail(False)
    index.set_num_threads(1)


def timed_mutation(call):
    process, thread = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_THREAD)
    pcpu, tcpu, started = time.process_time(), time.thread_time(), time.perf_counter()
    result = call()
    wall, thread_s, process_s = time.perf_counter() - started, time.thread_time() - tcpu, time.process_time() - pcpu
    return result, {
        "wall_s": wall, "thread_cpu_s": thread_s, "process_cpu_s": process_s,
        "process_rusage": usage_delta(process, resource.getrusage(resource.RUSAGE_SELF)),
        "thread_rusage": usage_delta(thread, resource.getrusage(resource.RUSAGE_THREAD)),
    }


def check_space(output, index):
    needed = int(index.index_file_size()) + min(2 * 1024**3, int(index.index_file_size()))
    free = shutil.disk_usage(output).free
    if free < needed:
        raise RuntimeError(f"Need {needed} DATA bytes; have {free}; old checkpoints are never removed")
    return {"free_bytes": free, "needed_bytes": needed}


def save_verified_index(index, path, state, data):
    if path.exists() or path.with_suffix(".pending").exists():
        raise RuntimeError("Refusing to overwrite an immutable checkpoint")
    check_space(path.parent, index)
    pending = path.with_suffix(".pending")
    index.save_index(str(pending))
    with pending.open("rb") as stream:
        os.fsync(stream.fileno())
    header = read_index_header(pending)
    if (header["count"] != state["occupied"] or header["max_elements"] != len(state["active"])
            or header["M"] != 40 or header["ef_construction"] != 300 or header["format"] != NUMERIC_EDGE_FORMAT):
        raise RuntimeError("Saved checkpoint changed count/capacity/M/construction ef")
    records = verify_index_records(pending, header, state["active"], data.attributes, data.base)
    tail = verify_index_tail(pending, header)
    os.rename(pending, path)
    sync_directory(path.parent)
    return file_info(path), {"header": header, "records": records, "tail": tail}


def add_source_rows(index, data, levels, start, end, threads, chunk):
    for first in range(start, end, chunk):
        last = min(first + chunk, end)
        index.add_items(
            np.ascontiguousarray(data.base[first:last]), data.records[first:last],
            ids=np.arange(first, last, dtype=np.uint64), num_threads=threads,
            levels=np.ascontiguousarray(levels[first:last], dtype=np.int32), replace_deleted=False,
        )


def prepare_levels(root, data, config):
    manifest_path = root / "levels.json"
    fingerprint = {
        "dataset": data.identity, "policy": LEVEL_POLICY, "seed": config["level_seed"],
        "threads": config["threads_build"],
    }
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved["fingerprint"] != fingerprint:
            raise RuntimeError("Incompatible level-plan cache")
        info = saved["artifact"]
        file_info(info["path"], info)
        with np.load(info["path"], allow_pickle=False) as arrays:
            levels = arrays["levels"].copy()
        if levels.shape != (data.n,) or array_digest(levels) != saved["details"]["levels_sha256"]:
            raise RuntimeError("Corrupt level-plan payload")
        return levels, saved
    levels, details = representative_levels(data.base, config["level_seed"], config["threads_build"])
    directory = root / f"level-plan-{uuid.uuid4().hex}"
    directory.mkdir()
    saved = {"fingerprint": fingerprint, "details": details,
             "artifact": write_npz(directory / "levels.npz", levels=levels)}
    immutable_json(manifest_path, saved)
    return levels, saved


def prefix_checkpoint(root, data, config, native, native_info, levels, plan, sources, explicit=None):
    return initial_cache(root, data, config, native, native_info, levels, plan, sources,
                         kind="prefix", count=config["initial"], explicit=explicit)


def full_checkpoint(root, data, config, native, native_info, levels, plan, sources, explicit=None):
    return initial_cache(root, data, config, native, native_info, levels, plan, sources,
                         kind="full", count=data.n, explicit=explicit)


def initial_cache(root, data, config, native, native_info, levels, plan, sources, *, kind, count, explicit):
    directory = Path(explicit).resolve(strict=True) if explicit else root / f"initial-{kind}"
    manifest_path = directory / f"{kind}-cache.json"
    active = np.zeros(data.n, dtype=bool)
    active[:count] = True
    state = {"occupied": count, "deleted": 0, "live": count, "active": active}
    fingerprint = {
        "dataset": data.identity, "native_sha256": native_info["sha256"],
        "query": config["query"], "capacity": config["capacity"], "initial": count, "kind": kind,
        "index_format": NUMERIC_EDGE_FORMAT, "numeric_marker_semantics": NUMERIC_MARKER_SEMANTICS,
        "numeric_marker_version": NUMERIC_MARKER_VERSION,
        "marker_owner_version": MARKER_OWNER_VERSION, "pickle_state_version": PICKLE_STATE_VERSION,
        "distance_order_version": DISTANCE_ORDER_VERSION, "candidate_order": CANDIDATE_ORDER,
        "level_plan": plan["details"], "threads_build": config["threads_build"],
        "add_chunk": config["add_chunk"], "source_sha256": {k: v["sha256"] for k, v in sources.items()},
        "codebook": "all source records, including every future addition; no padding",
    }
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved.get("status") != "complete" or saved["fingerprint"] != fingerprint:
            raise RuntimeError(f"Incompatible/noncomplete initial-{kind} cache; no rebuild fallback")
        info = saved["checkpoint"]
        if directory not in Path(info["path"]).parents:
            raise RuntimeError("Initial checkpoint escaped its cache directory")
        file_info(info["path"], info)
        header = read_index_header(info["path"])
        if header["count"] != count or header["max_elements"] != data.n or header["format"] != NUMERIC_EDGE_FORMAT:
            raise RuntimeError("Initial cache has incompatible source count/capacity/numeric Marker format")
        verify_index_records(info["path"], header, state["active"], data.attributes, data.base)
        return saved
    if explicit:
        raise RuntimeError(f"Explicit {kind} cache is missing; no undocumented in-process rebuild")
    directory.mkdir(exist_ok=True)
    attempt = directory / f"build-{uuid.uuid4().hex}"
    attempt.mkdir()
    atomic_json(directory / "build-state.json", {
        "status": "building", "attempt": str(attempt), "fingerprint": fingerprint,
        "restart_policy": "an interrupted UNCOMMITTED build starts fresh; old attempts are preserved",
    })
    started = time.perf_counter()
    index = native.Index(space="l2", dim=data.dim)
    top_count = int(levels[:count].sum())
    if top_count < 1:
        raise RuntimeError("Frozen level plan gives the initial graph no representative entry points")
    max_cate = max(label for record in data.records for label in record[1]) + 1
    index.init_index(
        max_elements=data.n, top_elements=top_count, M=40, ef_construction=300,
        random_seed=100, ft_bits=128, attr_type=[0, 1], max_cate_size=max_cate,
        allow_replace_deleted=False, edge_level_ft=True,
    )
    index.initAttrMapping(data.records)
    initialization_s = time.perf_counter() - started
    _, initial_insert = timed_mutation(lambda: add_source_rows(
        index, data, levels, 0, count, config["threads_build"], config["add_chunk"]))
    native_state(index, state["occupied"], 0)
    io_started = time.perf_counter()
    checkpoint, verification = save_verified_index(index, attempt / f"{kind}.index", state, data)
    saved = {
        "status": "complete", "fingerprint": fingerprint, "checkpoint": checkpoint,
        "verification": verification,
        "initialization_and_codebook_wall_s": initialization_s,
        "initial_insert": initial_insert,
        "build_wall_s": initialization_s + initial_insert["wall_s"],
        "checkpoint_io_wall_s": time.perf_counter() - io_started,
        "total_preparation_wall_s": time.perf_counter() - started,
        "first_source_row": 0, "past_last_source_row": count,
    }
    immutable_json(manifest_path, saved)
    atomic_json(directory / "build-state.json", {"status": "complete", "manifest": str(manifest_path)})
    return saved


def completed_insert_checkpoint(root, data, config, native_info, sources):
    store = StageStore(root / "insert", resume=True)
    if (store.doc["status"] != "complete" or store.doc["pending"]
            or len(store.doc["stages"]) != len(config["stage_progress"])
            or store.manifest["config"] != {**config, "operation": "insert"}
            or store.manifest["native"] != native_info or store.manifest["source"] != sources
            or store.manifest["inputs"] != data.inputs or store.manifest["dataset"] != data.identity):
        raise RuntimeError("Deletion requires the complete compatible fresh insertion run")
    store.verify_committed()
    final = store.doc["stages"][-1]
    if (final["stage"] != 5 or final["progress"] != config["initial"]
            or final["state"] != {"occupied": data.n, "deleted": 0, "live": data.n}):
        raise RuntimeError("Insertion stage5 is not the complete fresh source")
    header = read_index_header(final["files"]["checkpoint"]["path"])
    if header["format"] != NUMERIC_EDGE_FORMAT or header["count"] != data.n or header["max_elements"] != data.n:
        raise RuntimeError("Insertion final checkpoint has an incompatible numeric Marker format/capacity")
    return {
        "kind": "completed_insert_stage5", "checkpoint": final["files"]["checkpoint"],
        "native": native_info, "index_format": NUMERIC_EDGE_FORMAT,
        "numeric_marker_semantics": NUMERIC_MARKER_SEMANTICS,
        "numeric_marker_version": NUMERIC_MARKER_VERSION,
        "marker_owner_version": MARKER_OWNER_VERSION, "pickle_state_version": PICKLE_STATE_VERSION,
        "distance_order_version": DISTANCE_ORDER_VERSION, "candidate_order": CANDIDATE_ORDER,
        "insert_manifest": file_info(store.output / "manifest.json"), "insert_journal": file_info(store.path),
        "insert_stage": final["stage_file"], "insert_truth": final["files"]["truth"],
        "insert_calibration": final["files"]["calibration"],
        "initial_query_reference": copy.deepcopy(final["calibrations"]["current"]),
    }


class OperationRunner:
    def __init__(self, store, data, native, levels, *, fault=None):
        self.store, self.data, self.native, self.levels = store, data, native, levels
        self.config = store.manifest["config"]
        self.environment, self.operation = self.config["environment"], self.config["operation"]
        self.trace = Trace(self.operation, data.n, self.config["initial"], self.config["step"])
        if self.config["mode"] == "official" and array_digest(self.trace.order) != ORDER_SHA:
            raise RuntimeError("Canonical permutation changed")
        self.progress, self.indexes = 0, {}
        self.fault = fault or (lambda phase: None)

    @property
    def state(self):
        return self.trace.state(self.progress)

    def pin_update(self):
        os.sched_setaffinity(0, self.environment["update_cpus"])

    def verify_runtime(self):
        self.data.verify()
        for info in [self.store.manifest["native"], *self.store.manifest["source"].values()]:
            file_info(info["path"], info)

    def index_state(self, owner):
        state = self.trace.state(0 if owner == "initial" else self.progress)
        result = native_state(self.indexes[owner], state["occupied"], state["deleted"])
        if int(self.indexes[owner].get_max_elements()) != self.data.n:
            raise RuntimeError("Native capacity changed; retired slots must never be reused")
        return result

    def load(self, owner, info):
        self.pin_update()
        self.store.action("load_saved_graph", owner=owner, path=info["path"], progress=self.progress)
        require_unchanged(info)
        header = read_index_header(info["path"])
        expected = self.trace.state(0 if owner == "initial" else self.progress)
        if (header["format"] != NUMERIC_EDGE_FORMAT or header["count"] != expected["occupied"]
                or header["max_elements"] != self.data.n):
            raise RuntimeError("Cannot load a legacy/incompatible numeric checkpoint")
        index = self.native.Index(space="l2", dim=self.data.dim)
        if owner == "initial":
            index.load_index(info["path"])
        else:
            index.load_index(info["path"], max_elements=self.data.n, top_elements=1,
                             allow_replace_deleted=False, dynamic=True)
        configure_index(index)
        require_unchanged(info)
        self.indexes[owner] = index
        self.index_state(owner)

    def query(self, owner):
        return self.indexes[owner].hybrid_knn_query_dnf(
            self.data.queries, self.data.predicates, k=self.data.k, num_threads=1)

    def validate_result(self, owner, labels, distances, truth):
        shape = (len(self.data.queries), self.data.k)
        state = self.trace.state(0 if owner == "initial" else self.progress)
        if (labels.shape != shape or distances.shape != shape
                or not np.issubdtype(labels.dtype, np.integer)
                or np.any(labels < 0) or np.any(labels >= self.data.n)
                or not np.isfinite(distances).all() or np.any(distances < 0)
                or np.any(np.diff(distances, axis=1) < 0)
                or np.any(np.diff(np.sort(labels, axis=1), axis=1) == 0)):
            raise RuntimeError("Invalid native result shape, labels, distances or duplicate IDs")
        ids = labels.astype(np.int64)
        if not state["active"][ids].all():
            raise RuntimeError("Query returned a retired or not-yet-inserted source label")
        if not self.data.attributes.matches(
                ids, self.data.bounds[:, 0, None], self.data.bounds[:, 1, None]).all():
            raise RuntimeError("Query violated its per-query mixed DNF")
        delta = self.data.base[ids].astype(np.float64) - self.data.queries[:, None, :].astype(np.float64)
        if not np.array_equal(np.sum(delta * delta, axis=2), distances.astype(np.float64)):
            raise RuntimeError("Result distances do not belong to the external/source-row labels")
        recall = float(np.any(ids[:, :, None] == truth[:, None, :], axis=2).sum() / ids.size)
        if self.operation == "point_update":
            mapped, logical_gt = self.trace.logical_labels(ids), self.trace.logical_labels(truth)
            logical_recall = float(np.any(mapped[:, :, None] == logical_gt[:, None, :], axis=2).sum() / ids.size)
            if logical_recall != recall or np.any(np.diff(np.sort(mapped, axis=1), axis=1) == 0):
                raise RuntimeError("Logical-slot and fresh external-label recall disagree")
        self.index_state(owner)
        return recall

    def calibrate(self, owner, truth, directory, reference=None):
        os.sched_setaffinity(0, {self.environment["query_candidates"][0]})
        return calibrate(
            self.indexes[owner], lambda: self.query(owner),
            lambda labels, distances: self.validate_result(owner, labels, distances, truth),
            self.data.k, self.config["max_query_ef"], directory, owner, reference,
        )

    def ground_truth(self, directory):
        self.pin_update()
        state = self.state
        self.store.action("independent_exact_mixed_live_GT", progress=self.progress)
        exact = self.data.oracle.compute(
            self.data.queries, self.data.predicates, state["active"], self.data.k,
            occupied_count=state["occupied"],
            progress=lambda done, total: self.store.action("GT_progress", done=done, total=total),
        )
        if np.any(exact.matching_counts < self.data.k):
            raise RuntimeError("Native k=10 DNF API needs at least ten matches in every query")
        labels, proof = exact.labels, None
        if self.config["mode"] == "official" and state["active"].all():
            proof = self.data.oracle.validate_canonical(
                self.data.canonical, exact, self.data.queries, self.data.predicates, state["active"])
            labels = self.data.canonical
            if (exact.matching_counts.min() != 978433 or exact.matching_counts.max() != 980599
                    or exact.matching_counts.sum() != 979325947):
                raise RuntimeError("Full-source selectivity is not the canonical mixed workload")
        arrays = self.trace.arrays(self.progress)
        trace_file = write_npz(directory / "trace.npz", **arrays)
        gt_file = write_npz(
            directory / "ground-truth.npz", labels=labels, distances=exact.distances,
            exact_labels=exact.labels, matching_counts=exact.matching_counts,
            logical_labels=self.trace.logical_labels(labels),
        )
        details = {**exact.metadata, "canonical_validation": proof,
                   "used_labels_sha256": array_digest(labels), "trace": self.trace.identity(self.progress)}
        return labels, {
            "truth": gt_file, "truth_metadata": immutable_json(directory / "ground-truth.json", details),
            "trace": trace_file,
        }, details

    def mutate(self, previous, progress, directory):
        self.pin_update()
        self.verify_runtime()
        self.index_state("current")
        check_space(directory, self.indexes["current"])
        deleted, added = self.trace.mutation(previous, progress)
        phases, counters = {}, None
        self.store.action("synchronous_mutation", operation=self.operation, previous=previous, progress=progress)
        before_numa, started = numa_snapshot(), time.perf_counter()
        if len(deleted):
            stats, phases["delete"] = timed_mutation(
                lambda: self.indexes["current"].delete_items(deleted, num_threads=self.config["threads_update"]))
            counters, scrub_wall_s = split_deletion_stats(stats, len(deleted))
            phases["delete"]["scrub_wall_s"] = scrub_wall_s
            previous_state = self.trace.state(previous)
            native_state(self.indexes["current"], previous_state["occupied"],
                         previous_state["deleted"] + len(deleted))
            atomic_json(directory / "mutation-progress.json", {"phase": "deleted_before_optional_add",
                                                               "counters": counters, "phases": phases})
            self.fault("after_delete")
        if len(added):
            _, phases["add"] = timed_mutation(lambda: add_source_rows(
                self.indexes["current"], self.data, self.levels, int(added[0]), int(added[-1]) + 1,
                self.config["threads_update"], self.config["add_chunk"]))
            self.fault("after_add")
        wall = time.perf_counter() - started
        self.progress = progress
        state = self.index_state("current")
        mutation = {
            "wall_s": sum(p["wall_s"] for p in phases.values()), "transaction_wall_s": wall,
            "records": progress - previous, "phases": phases, "counters": counters, "state": state,
            "throughput_records_s": (progress - previous) / sum(p["wall_s"] for p in phases.values()),
            "throughput_unit": "logical full-point replacements/s" if self.operation == "point_update" else "records/s",
            "deleted_labels_sha256": array_digest(deleted), "added_source_rows_sha256": array_digest(added),
            "replace_deleted": False, "counter_note": COUNTER_NOTE,
            "numa_before": before_numa, "numa_after": numa_snapshot(),
        }
        mutation["timing_breakdown"] = mutation_timing(mutation)
        atomic_json(directory / "mutation.json", mutation)
        return mutation

    def load_truth(self, stage):
        info = stage["files"]["truth"]
        require_unchanged(info)
        with np.load(info["path"], allow_pickle=False) as arrays:
            labels = arrays["labels"].copy()
            logical = arrays["logical_labels"].copy()
        require_unchanged(info)
        state = self.trace.state(stage["progress"])
        if (labels.shape != (len(self.data.queries), self.data.k)
                or np.any(labels < 0) or np.any(labels >= self.data.n)
                or not state["active"][labels].all()
                or not np.array_equal(logical, self.trace.logical_labels(labels))):
            raise RuntimeError("Committed GT/label mapping is inconsistent")
        trace_info = stage["files"]["trace"]
        require_unchanged(trace_info)
        with np.load(trace_info["path"], allow_pickle=False) as arrays:
            for name, expected in self.trace.arrays(stage["progress"]).items():
                if not np.array_equal(arrays[name], expected):
                    raise RuntimeError("Committed trace changed")
        require_unchanged(trace_info)
        return labels

    def verify_reload(self, stage):
        truth = self.load_truth(stage)
        initial_truth = self.load_truth(self.store.doc["stages"][0])
        states = copy.deepcopy(stage["calibrations"])
        query = FrozenQuery(self, {"initial": initial_truth, "current": truth})
        os.sched_setaffinity(0, {self.environment["query_candidates"][0]})
        for owner in ("initial", "current"):
            for point in bracket(states[owner]):
                self.indexes[owner].set_ef(point["ef"])
                query(owner, point)
        self.store.action("saved_graph_and_GT_reload_verified", stage=stage["stage"])
        return states, truth, initial_truth

    def measure(self, states, truth, initial_truth, directory):
        self.verify_runtime()
        query = FrozenQuery(self, {"initial": initial_truth, "current": truth})
        for owner in ("initial", "current"):
            for point in bracket(states[owner]):
                self.indexes[owner].set_ef(point["ef"])
                self.indexes[owner].reset_ft_stats()
                query(owner, point)
                point["untimed_ft_work_counters"] = dict(self.indexes[owner].get_ft_stats())
        return paired_timing(
            self.indexes, states, query, self.environment, self.config["limits"],
            lambda report: atomic_json(directory / "timing-progress.json", report),
            pilot=self.config["mode"] == "pilot", locality=self.config["mode"] != "fixture",
        )

    def finish(self, stage, directory, files, states, truth, initial_truth, details):
        timing = self.measure(states, truth, initial_truth, directory)
        self.verify_runtime()
        for owner in self.indexes:
            self.index_state(owner)
        results = {
            "operation": self.operation, "mode": self.config["mode"], "stage": stage,
            "progress": self.progress, "trace": self.trace.identity(self.progress),
            "calibrations": states, "timing": timing, "query_payload_sha256": array_digest(self.data.queries),
            "ground_truth": files["truth"], "checkpoint": files["checkpoint"],
            "native": self.store.manifest["native"], **details,
            "operation_native": self.store.manifest["native"], "query_native": self.store.manifest["native"],
            "query_implementation": self.config.get("query_implementation", "nonofficial-fixture"),
        }
        results["initial_prefix_provenance"] = self.store.manifest.get("initial_prefix_provenance")
        results["initial_full_provenance"] = self.store.manifest.get("initial_full_provenance")
        results["initial_matches_completed_insert"] = bool(
            (self.store.manifest.get("initial_full_provenance") or {}).get("initial_query_reference"))
        results["index_format"] = NUMERIC_EDGE_FORMAT
        results["numeric_marker_semantics"] = NUMERIC_MARKER_SEMANTICS
        results["numeric_marker_version"] = NUMERIC_MARKER_VERSION
        results["marker_owner_version"] = MARKER_OWNER_VERSION
        results["pickle_state_version"] = PICKLE_STATE_VERSION
        results["distance_order_version"] = DISTANCE_ORDER_VERSION
        results["candidate_order"] = CANDIDATE_ORDER
        files["results"] = immutable_json(directory / "results.json", results)
        self.fault("before_stage_commit")
        payload = {
            "stage": stage, "progress": self.progress, "operation": self.operation,
            "attempt_id": self.store.doc["pending"]["attempt_id"],
            "status": "complete" if timing["status"] == "complete" else "needs_timing",
            "state": {k: v for k, v in self.state.items() if k != "active"},
            "files": files, "calibrations": states, "details": details,
        }
        self.store.commit(payload)
        print(json.dumps({"operation": self.operation, "stage": stage, "status": payload["status"],
                          "state": payload["state"], "timing": timing["summary"],
                          "mutation_timing": mutation_timing(details.get("mutation"))}), flush=True)
        return payload["status"] == "complete"

    def run(self, through_stage=None):
        latest = self.store.doc["stages"][-1] if self.store.doc["stages"] else None
        self.progress = latest["progress"] if latest else 0
        self.load("initial", self.store.manifest["initial_checkpoint"])
        self.load("current", latest["files"]["checkpoint"] if latest else self.store.manifest["initial_checkpoint"])
        if latest:
            states, truth, initial_truth = self.verify_reload(latest)
            if latest["status"] == "needs_timing":
                directory = self.store.begin(latest["stage"], latest["progress"], kind="retime")
                if not self.finish(latest["stage"], directory, copy.deepcopy(latest["files"]),
                                   states, truth, initial_truth, copy.deepcopy(latest["details"])):
                    return False
        limit = len(self.trace.progress) - 1 if through_stage is None else through_stage
        for stage in range(len(self.store.doc["stages"]), min(limit + 1, len(self.trace.progress))):
            progress = self.trace.progress[stage]
            directory = self.store.begin(stage, progress)
            mutation = self.mutate(self.progress, progress, directory) if stage else None
            truth, files, gt_details = self.ground_truth(directory)
            if stage == 0:
                initial_truth = truth
                origin = self.store.manifest.get("initial_full_provenance") or {}
                reference = origin.get("initial_query_reference")
                if reference is not None:
                    info = origin["insert_truth"]
                    file_info(info["path"], info)
                    with np.load(info["path"], allow_pickle=False) as arrays:
                        if not np.array_equal(arrays["labels"], initial_truth):
                            raise RuntimeError("Fresh deletion initial GT differs from completed insertion GT")
                baseline, bl, bd = self.calibrate("initial", initial_truth, directory, reference)
                current, cl, cd = self.calibrate("current", truth, directory, baseline)
            else:
                baseline = copy.deepcopy(self.store.doc["stages"][0]["calibrations"]["initial"])
                current, cl, cd = self.calibrate("current", truth, directory)
                bl = bd = np.empty(0, dtype=np.float32)
            files["calibration"] = write_npz(
                directory / "calibration.npz", baseline_labels=bl, baseline_distances=bd,
                current_labels=cl, current_distances=cd,
                current_efs=np.asarray([p["ef"] for p in current["points"]], dtype=np.int64))
            self.pin_update()
            if stage == 0:
                files["checkpoint"] = self.store.manifest["initial_checkpoint"]
                header = read_index_header(files["checkpoint"]["path"])
                verification = {"header": header, "records": verify_index_records(
                    files["checkpoint"]["path"], header, self.state["active"], self.data.attributes, self.data.base)}
            else:
                files["checkpoint"], verification = save_verified_index(
                    self.indexes["current"], directory / "checkpoint.index", self.state, self.data)
            if not self.finish(stage, directory, files, {"initial": baseline, "current": current},
                               truth, initial_truth, {"mutation": mutation, "ground_truth_details": gt_details,
                                                      "checkpoint_validation": verification}):
                return False
        complete = len(self.store.doc["stages"]) == len(self.trace.progress)
        self.store.doc["status"] = "complete" if complete else "paused"
        self.store.publish()
        return complete
