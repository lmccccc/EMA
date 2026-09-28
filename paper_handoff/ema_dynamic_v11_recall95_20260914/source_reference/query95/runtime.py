"""Durable operation journals, SHA-pinned native loading and local CPU telemetry.

Ported from the validated mixed-DNF deletion harness, including the strict
default-MPOL_BIND/local-FAISS-arena residency checks. No session-private imports.
"""

import copy
import ctypes
import datetime
import fcntl
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import resource
import stat
import struct
import sys
import time
import traceback
import uuid

import numpy as np


SCHEMA = "canonical-dynamic-operation-v4"
ATTRIBUTE_NODE_FORMAT = 9
NUMERIC_EDGE_FORMAT = 10
NUMERIC_MARKER_VERSION = 2
MARKER_OWNER_VERSION = 2
DISTANCE_ORDER_VERSION = 1
PICKLE_STATE_VERSION = 4
CANDIDATE_ORDER = "vector-distance-only-v1"
NUMERIC_MARKER_SEMANTICS = "bounded-floor-buckets-inclusive-ranges-v2"
LOCK_NAME = ".canonical-dynamic.lock"
COUNTERS = {
    "requested", "marked", "search_candidates", "search_expanded",
    "incoming_edges_repaired", "outgoing_edges_added", "rewired_nodes",
    "pruned_edges", "scrubbed_edges", "candidate_sources", "matched_edges",
    "cleared_bits", "support_checks",
}
DEFAULT_LIMITS = {
    "sibling_busy_max": 0.05, "thread_cpu_wall_min": 0.98,
    "frequency_khz": None, "major_faults_max": 0,
    "clean_rounds": 7, "max_rounds_per_core": 21, "max_cores": 3,
}


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def object_digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def file_info(path, expected=None):
    path = Path(path)
    if path.is_symlink():
        raise RuntimeError(f"Symlinked provenance/artifact is not allowed: {path}")
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"Expected a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if identity(before) != identity(after):
        raise RuntimeError(f"File changed during hashing: {path}")
    info = {"path": str(path), "bytes": after.st_size, "mtime_ns": after.st_mtime_ns,
            "sha256": digest.hexdigest()}
    if expected is not None and info != expected:
        raise RuntimeError(f"Frozen file/config/checkpoint mismatch: {path}; no fallback")
    return info


def require_unchanged(info):
    path = Path(info["path"])
    now = path.stat()
    if path.is_symlink() or (now.st_size, now.st_mtime_ns) != (info["bytes"], info["mtime_ns"]):
        raise RuntimeError(f"Input/artifact changed after hashing: {path}")


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, value):
    path = Path(path)
    pending = path.parent / f".{path.name}.{uuid.uuid4().hex}.pending"
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)
    sync_directory(path.parent)


def write_npz(path, **arrays):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"Refusing to overwrite an immutable artifact: {path}")
    pending = path.parent / f".{path.name}.{uuid.uuid4().hex}.pending"
    with pending.open("xb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.rename(pending, path)
    sync_directory(path.parent)
    return file_info(path)


def immutable_json(path, value):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"Refusing to replace an immutable result: {path}")
    atomic_json(path, value)
    return file_info(path)


def bootstrap_manifest(path, requested, *, resume=False):
    """Validate a manifest-only bootstrap without replacing its immutable bytes."""
    path = Path(path)
    if requested is None:
        raise RuntimeError("Bootstrap recovery requires the requested immutable manifest")
    if path.exists() or path.is_symlink():
        info = file_info(path)
        saved = json.loads(path.read_text())
        require_unchanged(info)
        if saved != requested:
            raise RuntimeError("Bootstrap manifest differs from requested immutable identity")
        return info, True
    if resume:
        raise RuntimeError("No immutable manifest exists for bootstrap recovery")
    return immutable_json(path, requested), False


class OutputLock:
    def __init__(self, output):
        self.path = Path(output) / LOCK_NAME

    def __enter__(self):
        self.fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.ftruncate(self.fd, 0)
            os.write(self.fd, json.dumps({"pid": os.getpid(), "host": platform.node(),
                                         "started_at": timestamp()}).encode())
            os.fsync(self.fd)
        except BaseException:
            os.close(self.fd)
            raise
        return self

    def __exit__(self, *_):
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        os.close(self.fd)
        # Keep the inode: unlinking a lock file allows a second independent lock.


class StageStore:
    """Artifacts precede the journal commit; interrupted attempts are never reused."""

    def __init__(self, output, *, manifest=None, resume=False, fault=None):
        self.output = Path(output)
        self.path = self.output / "journal.json"
        self.fault = fault or (lambda event: None)
        manifest_path = self.output / "manifest.json"
        if self.path.is_symlink():
            raise RuntimeError("Symlinked journal")
        if self.path.exists():
            if not resume:
                raise RuntimeError("Output already has a journal; use resume")
            self.doc = json.loads(self.path.read_text())
            if self.doc.get("schema") != SCHEMA:
                raise RuntimeError("This output is not a canonical recall95 dynamic operation; "
                                   "old recall protocols cannot be resumed as recall95")
            file_info(self.output / "manifest.json", self.doc["manifest_file"])
            self.manifest = json.loads((self.output / "manifest.json").read_text())
            if manifest is not None and self.manifest != manifest:
                raise RuntimeError("Resume manifest differs from binary/config/input hashes")
            self.validate_structure()
        else:
            attempts = self.output / "attempts"
            if attempts.is_symlink() or (attempts.exists() and any(attempts.iterdir())):
                raise RuntimeError("Missing operation journal with existing attempts; cannot infer a committed checkpoint")
            manifest_file, recovered = bootstrap_manifest(manifest_path, manifest, resume=resume)
            self.manifest = manifest
            self.fault("after_manifest")
            self.doc = {
                "schema": SCHEMA, "run_id": uuid.uuid4().hex, "created_at": timestamp(),
                "manifest_file": manifest_file, "status": "running", "stages": [],
                "pending": None, "superseded_attempts": [], "errors": [],
                "invocations": [], "bootstrap_recovered": recovered,
            }
            self.publish()

    def publish(self):
        self.doc["updated_at"] = timestamp()
        atomic_json(self.path, self.doc)

    def action(self, name, **details):
        self.doc["action"] = {"name": name, "at": timestamp(), **details}
        self.publish()

    def validate_structure(self):
        plan = self.manifest["config"]["stage_progress"]
        for i, stage in enumerate(self.doc["stages"]):
            if (i >= len(plan) or stage["stage"] != i or stage["progress"] != plan[i]
                    or stage["status"] not in ("complete", "needs_timing")
                    or (i < len(self.doc["stages"]) - 1 and stage["status"] != "complete")):
                raise RuntimeError("Inconsistent committed stage sequence")
        pending = self.doc["pending"]
        if pending:
            expected = len(self.doc["stages"]) - (pending["kind"] == "retime")
            if (pending["kind"] not in ("update", "retime") or pending["stage"] != expected
                    or expected < 0 or expected >= len(plan) or pending["progress"] != plan[expected]):
                raise RuntimeError("Pending transaction does not follow the committed checkpoint")
            self.owned_path(pending["directory"])
        if self.doc["status"] == "complete" and (
                len(self.doc["stages"]) != len(plan) or pending
                or self.doc["stages"][-1]["status"] != "complete"):
            raise RuntimeError("False completion in journal")

    def owned_path(self, path):
        path = Path(path)
        if self.output not in path.parents or path.resolve() != path:
            raise RuntimeError(f"Artifact escaped this run or used a symlink: {path}")
        return path

    def artifact_path(self, info):
        if info != self.manifest["initial_checkpoint"]:
            self.owned_path(info["path"])
        return Path(info["path"])

    def verify_committed(self, verified_inputs=None):
        checked = dict(verified_inputs or {})
        historical = [item["previous_stage"] for item in self.doc["superseded_attempts"]
                      if "previous_stage" in item]
        for stage in [*self.doc["stages"], *historical]:
            info = stage["stage_file"]
            self.owned_path(info["path"])
            file_info(info["path"], info)
            payload = json.loads(Path(info["path"]).read_text())
            if any(payload[key] != stage[key] for key in ("stage", "progress", "status", "attempt_id")):
                raise RuntimeError("Journal and immutable stage result disagree")
            if payload != {key: value for key, value in stage.items() if key != "stage_file"}:
                raise RuntimeError("Stage metadata differs from its durable commit")
            for artifact in stage["files"].values():
                self.artifact_path(artifact)
                if artifact["path"] in checked:
                    if checked[artifact["path"]] != artifact:
                        raise RuntimeError("Conflicting artifact identities")
                else:
                    file_info(artifact["path"], artifact)
                    checked[artifact["path"]] = artifact
        return checked

    def recover_pending(self):
        if self.doc["pending"] is not None:
            old = copy.deepcopy(self.doc["pending"])
            old.update(status="superseded", at=timestamp(),
                       reason="Never resume a partially mutated graph; replay from the last committed checkpoint")
            self.doc["superseded_attempts"].append(old)
            self.doc["pending"] = None
        self.doc["status"] = "running"
        self.publish()

    def begin(self, stage, progress, kind="update"):
        if self.doc["pending"] is not None:
            raise RuntimeError("An attempt is already in flight")
        expected = len(self.doc["stages"]) - (kind == "retime")
        plan = self.manifest["config"]["stage_progress"]
        if (kind not in ("update", "retime") or stage != expected or not 0 <= stage < len(plan)
                or progress != plan[stage]
                or (kind == "retime" and self.doc["stages"][-1]["status"] != "needs_timing")):
            raise RuntimeError("Invalid next-stage/retiming transition")
        attempt_id = uuid.uuid4().hex
        root = self.output / "attempts"
        if root.is_symlink():
            raise RuntimeError("Symlinked attempts directory")
        root.mkdir(exist_ok=True)
        directory = root / f"stage-{stage}-{attempt_id}"
        directory.mkdir()
        sync_directory(root)
        self.doc["pending"] = {
            "stage": stage, "progress": progress, "kind": kind, "attempt_id": attempt_id,
            "directory": str(directory), "started_at": timestamp(),
        }
        self.doc["status"] = "running"
        self.publish()
        self.fault("after_begin")
        return directory

    def commit(self, payload):
        pending = self.doc["pending"]
        if (pending is None or any(payload[k] != pending[k] for k in ("stage", "progress", "attempt_id"))
                or payload["status"] not in ("complete", "needs_timing")
                or not {"checkpoint", "truth", "calibration", "results"} <= set(payload["files"])):
            raise RuntimeError("Invalid stage commit")
        for info in payload["files"].values():
            self.artifact_path(info)
            file_info(info["path"], info)
        self.fault("after_artifacts")
        stage_file = immutable_json(Path(pending["directory"]) / "stage.json", payload)
        self.fault("before_journal_commit")
        next_doc = copy.deepcopy(self.doc)
        stage = {**payload, "stage_file": stage_file}
        if pending["kind"] == "retime":
            next_doc["superseded_attempts"].append({
                "reason": "Completed timing revision; old checkpoint/GT/results all retained",
                "previous_stage": next_doc["stages"][-1],
            })
            next_doc["stages"][-1] = stage
        else:
            next_doc["stages"].append(stage)
        next_doc["pending"] = None
        next_doc["status"] = "paused_noise" if payload["status"] == "needs_timing" else "running"
        atomic_json(self.path, next_doc)
        self.doc = next_doc
        self.fault("after_journal_commit")

    def record_error(self, error):
        self.doc["errors"].append({
            "at": timestamp(), "type": type(error).__name__, "message": str(error),
            "traceback": "".join(traceback.format_exception(error)),
            "action": self.doc.get("action"),
        })
        self.doc["status"] = "interrupted" if isinstance(error, (KeyboardInterrupt, InterruptedError)) else "failed"
        self.publish()


def import_native(directory, expected_sha):
    directory = Path(directory).expanduser().resolve(strict=True)
    if "profile" in str(directory).lower():
        raise RuntimeError("Profiling builds are forbidden for deletion updates")
    if "hashannlib" in sys.modules:
        raise RuntimeError("Native validation requires a fresh process")
    spec = importlib.machinery.PathFinder.find_spec("hashannlib", [str(directory)])
    if spec is None or not isinstance(spec.loader, importlib.machinery.ExtensionFileLoader):
        raise RuntimeError("No directly importable native extension in --native-dir")
    path = Path(spec.origin)
    if path.parent != directory:
        raise RuntimeError("Native module escaped --native-dir")
    info = file_info(path)
    if info["sha256"] != expected_sha:
        raise RuntimeError(f"Native SHA mismatch BEFORE import: {info['sha256']}")
    sys.path.insert(0, str(directory))
    importlib.invalidate_caches()
    native = importlib.import_module("hashannlib")
    if Path(native.__file__).resolve() != path or file_info(path) != info:
        raise RuntimeError("Imported native origin/content changed")
    if hasattr(native.Index, "set_profile_dnf_ft_simd"):
        raise RuntimeError("Profiling-toggle binary is not the production deletion implementation")
    return native, info


def import_aggregate(path):
    info = file_info(path)
    spec = importlib.util.spec_from_file_location("mixed_dnf_existing_aggregate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the repository qps_at_recall helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if file_info(path) != info:
        raise RuntimeError("Interpolation helper changed while importing")
    return module, info


def read_index_header(path):
    """Decode format10 mixed edge FT; old graphs lack the required ranking-policy identity."""
    size = Path(path).stat().st_size
    with Path(path).open("rb") as stream:
        def read(fmt):
            length = struct.calcsize("<" + fmt)
            data = stream.read(length)
            if len(data) != length:
                raise ValueError("Truncated native index header")
            return struct.unpack("<" + fmt, data)[0]

        def skip(length):
            if length < 0 or stream.tell() + length > size:
                raise ValueError("Invalid native index array length")
            stream.seek(length, 1)

        def vector(fmt, retain=False):
            count = read("Q")
            unit = struct.calcsize("<" + fmt)
            if count > size // unit or (retain and count > 32):
                raise ValueError("Invalid native index vector length")
            if retain:
                return [read(fmt) for _ in range(count)]
            skip(count * unit)
            return count

        header = {}
        for name in ("offset_level0", "max_elements", "top_elements", "count",
                     "record_bytes", "neighbor_bytes", "label_offset", "vector_offset"):
            header[name] = read("Q")
        header["max_level"], header["entrypoint"] = read("i"), read("I")
        for name in ("maxM", "maxM0", "M0_mul", "minM0", "M"):
            header[name] = read("Q")
        header["mult"] = read("d")
        header["ef_construction"], header["ft_bits"] = read("Q"), read("Q")
        header["cate_int_words"], header["max_cate_size"] = read("i"), read("i")
        header["attr_type"], header["attr_positions"] = vector("i", True), vector("i", True)
        header["attr_words"] = read("i")
        vector("I")
        buckets = read("i")
        for _ in range(3):
            vector("I")
        table_size = read("i")
        skip(4 * table_size * buckets * len(header["attr_type"]))
        mappings = read("Q")
        if mappings > 32:
            raise ValueError("Invalid counting-table mappings")
        for _ in range(mappings):
            vector("i")
        header["predicate_words"] = read("i")
        header["predicate_offsets"] = vector("i", True)
        header["ft_offset"], header["attr_offset"] = read("Q"), read("Q")
        header["neighbor_ft_offset"], header["ft_bytes_per_record"] = read("i"), read("i")
        header["format"] = read("i")
        header["records_offset"] = stream.tell()
    if (header["format"] != NUMERIC_EDGE_FORMAT or header["attr_type"] != [0, 1]
            or header["ft_bits"] != 128 or header["ft_bytes_per_record"] != 32
            or header["count"] <= 0 or header["count"] > header["max_elements"]
            or header["records_offset"] + header["count"] * header["record_bytes"] > size):
        raise ValueError("Expected a rebuilt mixed [0,1], 128-bits/PER-ATTRIBUTE, edge-FT format10 index; "
                         "attribute-bearing formats1-8 cannot be retagged or reused")
    header["numeric_marker_semantics"] = NUMERIC_MARKER_SEMANTICS
    header["numeric_marker_version"] = NUMERIC_MARKER_VERSION
    header["marker_owner_version"] = MARKER_OWNER_VERSION
    header["distance_order_version"] = DISTANCE_ORDER_VERSION
    header["candidate_order"] = CANDIDATE_ORDER
    return header


def verify_index_records(path, header, active, attributes=None, vectors=None):
    n, stride, start = header["count"], header["record_bytes"], header["records_offset"]
    if len(active) != header["max_elements"] or n > len(active) or np.any(active[n:]):
        raise ValueError("Active source rows disagree with occupied/capacity prefix")
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    labels = np.ndarray((n,), dtype="<u8", buffer=raw, offset=start + header["label_offset"],
                        strides=(stride,))
    flags = np.ndarray((n,), dtype=np.uint8, buffer=raw,
                       offset=start + header["offset_level0"] + 2, strides=(stride,))
    numeric = np.ndarray((n,), dtype="<i4", buffer=raw,
                         offset=start + header["attr_offset"] + 4 * header["attr_positions"][0],
                         strides=(stride,))
    categorical = np.ndarray((n,), dtype="<u4", buffer=raw,
                             offset=start + header["attr_offset"] + 4 * header["attr_positions"][1],
                             strides=(stride,))
    seen = np.zeros(n, dtype=bool)
    marked = 0
    try:
        for first in range(0, n, 100_000):
            end = min(first + 100_000, n)
            ids = labels[first:end].astype(np.int64)
            deleted = (flags[first:end] & 1) != 0
            if (np.any(ids < 0) or np.any(ids >= n) or seen[ids].any()
                    or len(np.unique(ids)) != len(ids)):
                raise RuntimeError("Checkpoint external labels are not a permutation of 0..N-1")
            seen[ids] = True
            if not np.array_equal(deleted, ~active[ids]):
                raise RuntimeError("Inline DELETE_MARKs disagree with the active external-ID trace")
            marked += int(deleted.sum())
            if attributes is not None and (
                    not np.array_equal(numeric[first:end], attributes.numeric[ids])
                    or not np.array_equal((categorical[first:end] & (1 << 9)) != 0, attributes.label9[ids])
                    or not np.array_equal((categorical[first:end] & (1 << 12)) != 0, attributes.label12[ids])):
                raise RuntimeError("Index records do not contain the canonical mixed attributes")
        samples = np.unique(np.linspace(0, n - 1, min(256, n), dtype=np.int64))
        if vectors is not None:
            stored = np.ndarray((n, vectors.shape[1]), dtype="<f4", buffer=raw,
                                offset=start + header["vector_offset"], strides=(stride, 4))
            if not np.array_equal(stored[samples], vectors[labels[samples].astype(np.int64)]):
                raise RuntimeError("Index vector/external-label mapping differs from the base-vector file")
        if marked != n - int(active.sum()) or not seen.all():
            raise RuntimeError("Saved marked/occupied counts are incorrect")
    finally:
        raw._mmap.close()
    return {
        "occupied": n, "marked_count": marked, "live": n - marked,
        "inline_flags_all_records_verified": True, "external_label_permutation_verified": True,
        "canonical_mixed_attributes_all_records_verified": attributes is not None,
        "vector_mapping_sample_count": len(samples) if vectors is not None else 0,
    }


def verify_index_tail(path, header):
    """Reject a short native save even if the complete level0 block was written."""
    size = Path(path).stat().st_size
    position = header["records_offset"] + header["count"] * header["record_bytes"]
    if size - position < 4 * header["count"]:
        raise RuntimeError("Truncated upper-layer size records in checkpoint")
    with Path(path).open("rb") as stream:
        stream.seek(position)
        for _ in range(header["count"]):
            data = stream.read(4)
            if len(data) != 4:
                raise RuntimeError("Truncated upper-layer checkpoint record")
            length = struct.unpack("<I", data)[0]
            if stream.tell() + length > size:
                raise RuntimeError("Truncated upper-layer checkpoint links")
            if length:
                stream.seek(length, 1)
        if stream.tell() != size:
            raise RuntimeError("Unexpected trailing native checkpoint payload")
    return {"upper_layer_records": header["count"], "exact_file_boundary_verified": True}


def native_state(index, occupied, deleted, construction_ef=300, m=40):
    ratio = float(index.get_deleted_ratio())
    state = {
        "occupied": int(index.get_current_count()), "live": occupied - deleted,
        "deleted_ratio": ratio, "marked_count_from_native_ratio": int(round(ratio * occupied)),
        "dirty_count": int(index.get_dirty_count()),
        "repair_candidates": int(index.repair_candidates_size()),
        "ef_construction": int(index.ef_construction), "M": int(index.M),
    }
    if (state["occupied"] != occupied or state["marked_count_from_native_ratio"] != deleted
            or not math.isclose(ratio, deleted / occupied, rel_tol=0, abs_tol=2e-8)
            or state["dirty_count"] or state["repair_candidates"]
            or state["ef_construction"] != construction_ef or state["M"] != m):
        raise RuntimeError(f"Native state invariant violation: {state}")
    return state


def validate_counters(counters, requested):
    if not isinstance(counters, dict) or not COUNTERS <= counters.keys():
        raise RuntimeError("delete_items did not expose the expected structural/Marker/scrub work counters")
    if any(type(value) is not int or value < 0 for value in counters.values()):
        raise RuntimeError("Invalid native deletion work counter")
    if counters["requested"] != requested or counters["marked"] != requested:
        raise RuntimeError("Native deletion did not mark exactly the requested new external labels")
    return counters


def cpu_list(text):
    result = set()
    for part in text.strip().split(","):
        bounds = part.split("-")
        if len(bounds) not in (1, 2):
            raise ValueError("Invalid CPU list")
        first, last = int(bounds[0]), int(bounds[-1])
        if first < 0 or last < first:
            raise ValueError("Invalid CPU range")
        result.update(range(first, last + 1))
    return sorted(result)


def siblings(core):
    values = cpu_list(Path(
        f"/sys/devices/system/cpu/cpu{core}/topology/thread_siblings_list",
    ).read_text())
    return [cpu for cpu in values if cpu != core]


def cpu_ticks():
    result = {}
    for line in Path("/proc/stat").read_text().splitlines():
        parts = line.split()
        if parts[0].startswith("cpu") and parts[0][3:].isdigit():
            values = list(map(int, parts[1:9]))
            result[int(parts[0][3:])] = (sum(values), values[3] + values[4])
    return result


def frequency(core):
    path = Path(f"/sys/devices/system/cpu/cpu{core}/cpufreq/scaling_cur_freq")
    return int(path.read_text()) if path.exists() else None


def busy(before, after, cpu):
    if cpu not in before or cpu not in after:
        return None
    total = after[cpu][0] - before[cpu][0]
    idle = after[cpu][1] - before[cpu][1]
    return (total - idle) / total if total > 0 and 0 <= idle <= total else None


def memory_policy():
    api = ctypes.CDLL("libnuma.so.1", use_errno=True)
    api.get_mempolicy.argtypes = [
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_ulong),
        ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong,
    ]
    api.get_mempolicy.restype = ctypes.c_int
    mode, mask = ctypes.c_int(), ctypes.c_ulong()
    bits = ctypes.sizeof(mask) * 8
    if api.get_mempolicy(ctypes.byref(mode), ctypes.byref(mask), bits, None, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return {"mode": mode.value, "nodes": [node for node in range(bits) if mask.value & (1 << node)]}


def numa_snapshot():
    anonymous, anonymous_policies, files, policies = {}, {}, {}, {}
    for line in Path("/proc/self/numa_maps").read_text().splitlines():
        parts = line.split()
        page_kib = next((int(p.split("=")[1]) for p in parts if p.startswith("kernelpagesize_kB=")), 4)
        file_backed = any(p.startswith("file=") for p in parts)
        for part in parts:
            if part.startswith("N") and "=" in part and part[1:part.index("=")].isdigit():
                node, pages = part.split("=")
                kib = int(pages) * page_kib
                policies[parts[1]] = policies.get(parts[1], 0) + kib
                if not file_backed:
                    anonymous[node] = anonymous.get(node, 0) + kib
                    anonymous_policies[parts[1]] = anonymous_policies.get(parts[1], 0) + kib
                else:
                    files[node] = files.get(node, 0) + kib
    affinities, exited = {}, []
    for task in Path("/proc/self/task").iterdir():
        try:
            affinities[task.name] = sorted(os.sched_getaffinity(int(task.name)))
        except ProcessLookupError:
            exited.append(task.name)
    return {"anonymous_mapping_kib": anonymous, "anonymous_policies_kib": anonymous_policies,
            "file_mapping_kib": files, "policies_kib": policies,
            "default_memory_policy": memory_policy(), "thread_affinities": affinities,
            "threads_exited_during_snapshot": exited}


def require_local(snapshot, node):
    # FAISS also creates local-policy anonymous arenas. They remain local when
    # every worker is node-bound; verify actual residency and default binding too.
    node_cpus = set(cpu_list(Path(f"/sys/devices/system/node/node{node}/cpulist").read_text()))
    affinities = snapshot["thread_affinities"].values()
    if (set(snapshot["anonymous_mapping_kib"]) != {f"N{node}"}
            or not snapshot["anonymous_policies_kib"]
            or not set(snapshot["anonymous_policies_kib"]) <= {f"bind:{node}", "local"}
            or snapshot["default_memory_policy"] != {"mode": 2, "nodes": [node]}
            or not affinities or any(not cpus or not set(cpus) <= node_cpus for cpus in affinities)):
        raise RuntimeError(f"NUMA locality/policy violation for node {node}: {snapshot}")


def environment_config(node, update_threads, requested_cores=None, update_cpus=None):
    allowed = sorted(os.sched_getaffinity(0))
    node_cpus = cpu_list(Path(f"/sys/devices/system/node/node{node}/cpulist").read_text())
    if not allowed or not set(allowed) <= set(node_cpus) or update_threads > len(allowed):
        raise RuntimeError(f"Launch with numactl --cpunodebind={node} --membind={node}")
    expected = {"OMP_NUM_THREADS": str(update_threads), "OMP_PROC_BIND": "false",
                "OMP_WAIT_POLICY": "passive", "OPENBLAS_NUM_THREADS": "1"}
    actual = {key: os.environ.get(key, "").lower() for key in expected}
    if actual != expected:
        raise RuntimeError(f"Required environment {expected}, found {actual}")
    candidates = [cpu for cpu in allowed if cpu == min([cpu, *siblings(cpu)])]
    if requested_cores:
        if len(set(requested_cores)) != len(requested_cores) or not set(requested_cores) <= set(candidates):
            raise RuntimeError("Query cores must be distinct allowed local physical cores")
        candidates = requested_cores
    if update_cpus is not None:
        if not set(update_cpus) <= set(allowed) or len(update_cpus) < update_threads:
            raise RuntimeError("--update-cpus must be allowed, local and sufficient for mutation threads")
    else:
        update_cpus = allowed
    if not candidates:
        raise RuntimeError("No allowed physical query core")
    require_local(numa_snapshot(), node)
    return {
        "node": node, "update_cpus": update_cpus, "allowed_cpus": allowed, "query_candidates": candidates,
        "siblings": {str(c): siblings(c) for c in candidates}, "environment": actual,
        "other_environment": {k: os.environ.get(k) for k in (
            "OMP_PLACES", "OMP_DYNAMIC", "OMP_THREAD_LIMIT", "MKL_NUM_THREADS",
            "GOMP_CPU_AFFINITY", "KMP_AFFINITY", "LD_PRELOAD",
        )},
        "host": platform.node(), "kernel": platform.release(),
        "machine": platform.machine(), "python": sys.version,
    }


def choose_core(candidates, sample_seconds=2):
    before = cpu_ticks()
    time.sleep(sample_seconds)
    after = cpu_ticks()
    loads = {}
    for core in candidates:
        values = [busy(before, after, cpu) for cpu in [core, *siblings(core)]]
        if all(value is not None for value in values):
            loads[core] = sum(values)
    if not loads:
        raise RuntimeError("No usable CPU utilization interval for core selection")
    return min(loads, key=lambda core: (loads[core], core)), loads


def usage_delta(before, after):
    return {name: getattr(after, attr) - getattr(before, attr) for name, attr in (
        ("user_cpu_s", "ru_utime"), ("system_cpu_s", "ru_stime"),
        ("minor_faults", "ru_minflt"), ("major_faults", "ru_majflt"),
        ("voluntary_switches", "ru_nvcsw"), ("involuntary_switches", "ru_nivcsw"),
    )}


def timed_query(call, core, sibling_cpus, limits):
    cores = [core, *sibling_cpus]
    before_ticks = cpu_ticks()
    before_frequency = {str(c): frequency(c) for c in cores}
    before_thread = resource.getrusage(resource.RUSAGE_THREAD)
    before_process = resource.getrusage(resource.RUSAGE_SELF)
    actual_before = ctypes.CDLL(None).sched_getcpu()
    process_start, thread_start, wall_start = time.process_time(), time.thread_time(), time.perf_counter()
    result = call()
    wall = time.perf_counter() - wall_start
    thread_cpu, process_cpu = time.thread_time() - thread_start, time.process_time() - process_start
    actual_after = ctypes.CDLL(None).sched_getcpu()
    after_thread, after_process = resource.getrusage(resource.RUSAGE_THREAD), resource.getrusage(resource.RUSAGE_SELF)
    after_frequency = {str(c): frequency(c) for c in cores}
    after_ticks = cpu_ticks()
    sample = {
        "wall_s": wall, "thread_cpu_s": thread_cpu, "process_cpu_s": process_cpu,
        "thread_cpu_wall_ratio": thread_cpu / wall,
        "thread_rusage": usage_delta(before_thread, after_thread),
        "process_rusage": usage_delta(before_process, after_process),
        "process_peak_rss_kib": after_process.ru_maxrss,
        "actual_cpu_before": actual_before, "actual_cpu_after": actual_after,
        "affinity_after": sorted(os.sched_getaffinity(0)),
        "frequencies_before_khz": before_frequency, "frequencies_after_khz": after_frequency,
        "cpu_ticks_before": {str(c): before_ticks.get(c) for c in cores},
        "cpu_ticks_after": {str(c): after_ticks.get(c) for c in cores},
        "core_busy_fraction": busy(before_ticks, after_ticks, core),
        "sibling_busy_fractions": {str(c): busy(before_ticks, after_ticks, c) for c in sibling_cpus},
    }
    sample["rejection_reasons"] = noise_reasons(sample, core, limits)
    sample["clean"] = not sample["rejection_reasons"]
    return result, sample


def noise_reasons(sample, core, limits):
    reasons = []
    sibling_loads = list(sample["sibling_busy_fractions"].values())
    if any(x is None or x > limits["sibling_busy_max"] for x in sibling_loads):
        reasons.append("SMT sibling busy or missing utilization interval")
    if sample["thread_cpu_wall_ratio"] < limits["thread_cpu_wall_min"]:
        reasons.append("thread CPU/wall below configured minimum")
    if limits["frequency_khz"] is not None:
        for key in ("frequencies_before_khz", "frequencies_after_khz"):
            value = sample[key][str(core)]
            if value is None or not limits["frequency_khz"][0] <= value <= limits["frequency_khz"][1]:
                reasons.append("frequency outside configured interval or unavailable")
                break
    if (sample["thread_rusage"]["major_faults"] > limits["major_faults_max"]
            or sample["process_rusage"]["major_faults"] > limits["major_faults_max"]):
        reasons.append("major page fault")
    if (sample["actual_cpu_before"] != core or sample["actual_cpu_after"] != core
            or sample["affinity_after"] != [core]):
        reasons.append("query left its single physical core")
    return reasons
