"""Small unittest fixtures for the canonical reproduction pipeline; no full graph loads."""

import argparse
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import resource
import struct
import types
import unittest
from unittest import mock
import uuid

import numpy as np

from exp_benchmark.dynamic import aggregate, canary, measurement, pipeline, protocol, runner, runtime
from exp_benchmark.dynamic.gt import ExactMixedDNF, MixedAttributes, array_digest, predicate_bounds


FIXTURES = Path(__file__).resolve().parent / ".dynamic-fixtures"


def directory(name):
    path = FIXTURES / f"{name}-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def brute_force(vectors, records, queries, predicates, active, k=10):
    labels = np.full((len(queries), k), -1, dtype=np.int64)
    distances = np.full((len(queries), k), np.inf, dtype=np.float32)
    counts = []
    for q, predicate in enumerate(predicates):
        eligible = []
        for row in range(len(vectors)):
            if not active[row]:
                continue
            matches = False
            for numeric, categorical in predicate:
                matches |= ((not numeric or numeric[0] <= records[row][0][0] <= numeric[1])
                            and all(label in records[row][1] for label in categorical))
            if matches:
                delta = vectors[row].astype(np.float64) - queries[q].astype(np.float64)
                eligible.append((float(np.sum(delta * delta)), row))
        counts.append(len(eligible))
        for pos, (distance, row) in enumerate(sorted(eligible)[:k]):
            labels[q, pos], distances[q, pos] = row, distance
    return labels, distances, np.asarray(counts, dtype=np.int64)


class DeletionTimingTests(unittest.TestCase):
    def stats(self, requested=3):
        result = dict.fromkeys(runtime.COUNTERS, 0)
        result["requested"] = result["marked"] = requested
        result["additional_work_counter"] = 7
        return result

    def mutation(self, operation="point_update"):
        phases = {}
        if operation != "insert":
            phases["delete"] = {"wall_s": 40.0, "scrub_wall_s": 10.0}
        if operation != "delete":
            phases["add"] = {"wall_s": 20.0}
        return {"wall_s": sum(phase["wall_s"] for phase in phases.values()),
                "phases": phases, "records": 1_000_000}

    def test_native_duration_is_separate_from_integer_work_counters(self):
        stats = {**self.stats(), "scrub_wall_s": 1.25}
        before = copy.deepcopy(stats)
        counters, duration = runtime.split_deletion_stats(stats, 3)
        self.assertEqual(counters, self.stats())
        self.assertEqual(duration, 1.25)
        self.assertEqual(stats, before)
        self.assertIs(runtime.validate_counters(counters, 3), counters)
        for value in (None, -1.0, float("nan"), float("inf"), True, "1.25"):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                runtime.split_deletion_stats({**self.stats(), "scrub_wall_s": value}, 3)
        for value in (1.25, -1, True):
            with self.subTest(counter=value), self.assertRaises(RuntimeError):
                runtime.split_deletion_stats(
                    {**self.stats(), "additional_work_counter": value, "scrub_wall_s": 0.5}, 3)
        with self.assertRaises(RuntimeError):
            runtime.split_deletion_stats([], 0)
        with self.assertRaises(RuntimeError):
            runtime.split_deletion_stats(self.stats(), 4)

    def test_old_native_unknown_and_measured_zero_are_distinct(self):
        self.assertEqual(runtime.split_deletion_stats(self.stats(), 3), (self.stats(), None))
        empty = {**self.stats(0), "scrub_wall_s": 0.0}
        self.assertEqual(runtime.split_deletion_stats(empty, 0), (self.stats(0), 0.0))
        for recorded in (False, True):
            mutation = self.mutation()
            if recorded:
                mutation["phases"]["delete"]["scrub_wall_s"] = None
            else:
                del mutation["phases"]["delete"]["scrub_wall_s"]
            result = runtime.mutation_timing(mutation)
            self.assertEqual(result["delete_wall_s"], 40.0)
            for name in ("delete_scrub_wall_s", "delete_non_scrub_wall_s",
                         "delete_scrub_fraction", "maintenance_scrub_fraction"):
                self.assertIsNone(result[name])
        mutation["phases"]["delete"]["scrub_wall_s"] = 0.0
        result = runtime.mutation_timing(mutation)
        self.assertEqual(result["delete_scrub_wall_s"], 0.0)
        self.assertEqual(result["delete_non_scrub_wall_s"], 40.0)
        self.assertEqual(result["delete_scrub_fraction"], 0.0)

    def test_delete_and_replacement_share_accounting_without_double_counting(self):
        for operation in ("delete", "point_update"):
            mutation = self.mutation(operation)
            before = copy.deepcopy(mutation)
            result = runtime.mutation_timing(mutation)
            self.assertEqual(result["delete_wall_s"], 40.0)
            self.assertEqual(result["delete_scrub_wall_s"], 10.0)
            self.assertEqual(result["delete_non_scrub_wall_s"], 30.0)
            self.assertEqual(result["delete_scrub_fraction"], 0.25)
            self.assertEqual(result["maintenance_wall_s"], mutation["wall_s"])
            self.assertAlmostEqual(result["maintenance_scrub_fraction"], 10.0 / mutation["wall_s"])
            self.assertEqual(result["maintenance_wall_s"],
                             result["delete_non_scrub_wall_s"] + result["delete_scrub_wall_s"]
                             + (result["add_wall_s"] or 0))
            self.assertEqual(mutation, before)
        self.assertTrue(all(value is None for value in runtime.mutation_timing(None).values()))
        result = runtime.mutation_timing(self.mutation("insert"))
        self.assertEqual(result["maintenance_wall_s"], 20.0)
        self.assertEqual(result["add_wall_s"], 20.0)
        self.assertTrue(all(value is None for name, value in result.items()
                            if name not in ("maintenance_wall_s", "add_wall_s")))

    def test_invalid_durations_and_extra_scrub_phase_are_rejected(self):
        for value in (-1.0, 40.001, float("inf"), float("nan"), True, "1"):
            mutation = self.mutation()
            mutation["phases"]["delete"]["scrub_wall_s"] = value
            with self.subTest(scrub=value), self.assertRaises(RuntimeError):
                runtime.mutation_timing(mutation)
        for target in ("total", "delete", "add"):
            for value in (0, -1.0, float("inf"), float("nan"), True, "40"):
                mutation = self.mutation()
                if target == "total":
                    mutation["wall_s"] = value
                else:
                    mutation["phases"][target]["wall_s"] = value
                with self.subTest(target=target, value=value), self.assertRaises(RuntimeError):
                    runtime.mutation_timing(mutation)
        mutation = self.mutation()
        mutation["wall_s"] += 10
        with self.assertRaisesRegex(RuntimeError, "exactly once"):
            runtime.mutation_timing(mutation)
        mutation["phases"]["scrub"] = {"wall_s": 10.0}
        with self.assertRaisesRegex(RuntimeError, "extra scrub phase"):
            runtime.mutation_timing(mutation)

    def test_paper_rows_include_breakdown_and_keep_work_counts_integer(self):
        mutation = self.mutation()
        mutation["counters"] = self.stats(1_000_000)
        stage = {"stage": 1, "progress": 1_000_000,
                 "state": {"occupied": 6_000_000, "deleted": 1_000_000, "live": 5_000_000},
                 "files": {"results": {}, "truth": {}, "checkpoint": {}}}
        summaries = dict.fromkeys(
            ("current", "initial"), measurement.summarize_curve([(10, .94, 120), (11, .96, 100)]))
        results = {"mutation": mutation, "ground_truth_details": {
            "selectivity_live_min": .1, "selectivity_live_mean": .1,
            "selectivity_live_max": .1, "matching_live_mean": 500_000}}
        row = aggregate.paper_row("point_update", stage, results, summaries, "native")
        self.assertEqual({key: row[key] for key in runtime.mutation_timing(mutation)},
                         runtime.mutation_timing(mutation))
        self.assertEqual(row["mutation_work_counts"], mutation["counters"])
        self.assertEqual(row["maintenance_records_s"], 1_000_000 / 60.0)


class TraceAndGTTests(unittest.TestCase):
    def test_insert_delete_and_complete_record_replacement_mapping(self):
        n, initial, step = 100, 50, 10
        for operation in protocol.OPERATIONS:
            trace = protocol.Trace(operation, n, initial, step)
            for previous, progress in zip(trace.progress, trace.progress[1:]):
                old, new = trace.mutation(previous, progress)
                state = trace.state(progress)
                if operation == "insert":
                    self.assertEqual(state["occupied"], initial + progress)
                    self.assertEqual(state["deleted"], 0)
                    self.assertEqual(new.tolist(), list(range(initial + previous, initial + progress)))
                elif operation == "delete":
                    self.assertEqual(old.tolist(), trace.order[previous:progress].tolist())
                    self.assertEqual(state["occupied"], n)
                    self.assertEqual(state["live"], n - progress)
                else:
                    self.assertEqual(state["live"], initial)
                    self.assertEqual(old.tolist(), list(range(previous, progress)))
                    self.assertEqual(new.tolist(), list(range(initial + previous, initial + progress)))
                    mapping = trace.arrays(progress)["logical_to_external"]
                    np.testing.assert_array_equal(trace.logical_labels(mapping), np.arange(initial))
                    self.assertTrue(state["active"][mapping].all())
                self.assertFalse(state["active"][old].any())
                self.assertTrue(state["active"][new].all())
                self.assertFalse(state["active"][state["occupied"]:].any())
                self.assertFalse(state["active"].flags.writeable)

    def test_exact_GT_overlap_bounds_retired_and_added_rows(self):
        rng = np.random.default_rng(3)
        vectors = rng.integers(0, 6, (60, 4)).astype(np.float32)
        records = [[[i % 7], ([9] if i % 2 else []) + ([12] if i % 3 else [])] for i in range(60)]
        queries = vectors[[3, 4, 10]]
        predicates = [[[[lo, hi], [9]], [[], [12]]] for lo, hi in ((0, 0), (1, 5), (6, 6))]
        oracle = ExactMixedDNF(vectors, MixedAttributes.from_records(records, 60), threads=1)
        for operation in protocol.OPERATIONS:
            trace = protocol.Trace(operation, 60, 30, 10)
            for progress in trace.progress:
                state = trace.state(progress)
                expected, distances, counts = brute_force(vectors, records, queries, predicates, state["active"], 5)
                exact = oracle.compute(queries, predicates, state["active"], 5, occupied_count=state["occupied"])
                np.testing.assert_array_equal(exact.distances, distances)
                np.testing.assert_array_equal(exact.matching_counts, counts)
                self.assertEqual(exact.metadata["occupied"], state["occupied"])
                self.assertAlmostEqual(exact.metadata["selectivity_live_mean"], counts.mean() / state["live"])
                self.assertTrue(state["active"][exact.labels].all())

    def test_ties_and_insufficient_results(self):
        vectors = np.asarray([[1, 0], [0, 1], [3, 0]], dtype=np.float32)
        oracle = ExactMixedDNF(vectors, MixedAttributes.from_records([[[i], [12]] for i in range(3)], 3), threads=1)
        predicates = [[[[0, 3], [9]], [[], [12]]]]
        queries = np.zeros((1, 2), dtype=np.float32)
        live = np.ones(3, dtype=bool)
        exact = oracle.compute(queries, predicates, live, 1)
        proof = oracle.validate_canonical(np.asarray([[1 - exact.labels[0, 0]]]), exact, queries, predicates, live)
        self.assertEqual(len(proof["boundary_tie_rows"]), 1)
        with self.assertRaises(ValueError):
            oracle.validate_canonical(np.asarray([[2]]), exact, queries, predicates, live)
        live[:] = False
        exact = oracle.compute(queries, predicates, live, 3)
        self.assertTrue(np.all(exact.labels == -1))
        self.assertTrue(np.isinf(exact.distances).all())
        with self.assertRaises(ValueError):
            predicate_bounds([[[9]]])

    def test_bounded_JSON_prefix_and_malformed_input(self):
        root = directory("json-prefix")
        rows = [[[i], [0, 9, 12]] for i in range(6000)]
        path = root / "attrs.json"
        path.write_text(json.dumps(rows))
        self.assertEqual(protocol.json_record_prefix(path, 5000), rows[:5000])
        with self.assertRaises(ValueError):
            protocol.json_record_prefix(path, 6001)
        path = root / "truncated.json"
        path.write_text("[[[1],[9]], [[2]")
        with self.assertRaises(ValueError):
            protocol.json_record_prefix(path, 2)

    def test_production_representative_level_plan_is_seeded_and_reusable(self):
        rng = np.random.default_rng(98)
        vectors = rng.integers(0, 128, (80, 4)).astype(np.float32)
        first, metadata = protocol.representative_levels(vectors, 1234, 1)
        second, repeated = protocol.representative_levels(vectors, 1234, 1)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(metadata, repeated)
        self.assertEqual(first.dtype, np.int32)
        self.assertEqual(int(first.sum()), int(np.sqrt(len(vectors))) * 4)
        self.assertNotIn("Bernoulli", json.dumps(metadata))


class ParserAndBoundaryTests(unittest.TestCase):
    def args(self, *extra):
        return pipeline.parse_args(["run", "--data-root", "/data", "--output", "/mnt/data/fresh",
                                    "--native-dir", "/native", "--expected-sha256", "a" * 64,
                                    "--numa-node", "0", *extra])

    def test_explicit_frequency_pilot_and_operation_surface(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.args()
        args = self.args("--frequency-khz", "3267000,3333000", "--operations", "insert,point_update")
        config = pipeline.configuration(args, {"node": 0})
        self.assertEqual(config["operations"], ["insert", "point_update"])
        self.assertEqual(config["index_format"], 10)
        self.assertEqual(config["numeric_marker_version"], 2)
        self.assertEqual(config["marker_owner_version"], 2)
        self.assertEqual(config["pickle_state_version"], 4)
        self.assertEqual(config["distance_order_version"], 1)
        self.assertEqual(config["candidate_order"], "vector-distance-only-v1")
        self.assertEqual(config["protocol"], "sift10m-mixed-dnf-dynamic-rebuild-v5")
        self.assertEqual(config["rebuild_policy"], "fix_rebuild_rerun_all")
        self.assertNotIn("initial_index", protocol.INPUT_SHA)
        self.assertEqual(config["query"]["target_recall"], .95)
        self.assertEqual(config["query"]["M"], 40)
        self.assertEqual(config["query"]["ef_construction"], 300)
        self.assertEqual(config["query"]["k"], 10)
        self.assertEqual(config["query"]["threads_query"], 1)
        self.assertEqual(config["query"]["ft_routing_min_deg"], 8)
        self.assertEqual(config["stage_progress"], list(range(0, 5_000_001, 1_000_000)))
        self.assertIsNone(runtime.DEFAULT_LIMITS["frequency_khz"])
        pilot = self.args("--pilot-size", "1000", "--through-stage", "1", "--resume")
        self.assertEqual(pipeline.configuration(pilot, {})["initial"], 500)
        for extra in (("--operations", "attr_update"), ("--operations", "insert,insert"),
                      ("--pilot-size", "10000000"), ("--pilot-size", "1000", "--reuse-delete", "/delete"),
                      ("--frequency-khz", "1,2", "--reuse-delete", "/delete"),
                      ("--frequency-khz", "1,2", "--retime-reused-delete"),
                      ("--frequency-khz", "1,2", "--reuse-delete", "/delete", "--retime-reused-delete"),
                      ("--frequency-khz", "1,2", "--initial-full", "/old/index"),
                      ("--frequency-khz", "2,1"), ("--threads_query", "32")):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.args(*extra)
        for native_sha in protocol.INCOMPATIBLE_GRAPH_NATIVES:
            with self.assertRaisesRegex(RuntimeError, "Superseded"):
                protocol.require_current_query_native(native_sha)

    def test_interpolation_above_minimum_boundary_and_static_default_unchanged(self):
        curve = [(57, .945, 1500), (58, .955, 1300)]
        self.assertAlmostEqual(measurement.summarize_curve(curve)["qps95"], 1400)
        self.assertEqual(measurement.summarize_curve(curve, target=.95)["recall_status"], "target95")
        value = measurement.summarize_curve([(10, .9673, 2292)])
        self.assertIsNone(value["qps95"])
        self.assertEqual(value["observed_qps"], 2292)
        self.assertEqual(value["target_recall"], .95)
        self.assertNotIn("qps90", value)
        self.assertIn(">=95%", value["recall_status"])
        self.assertEqual(measurement.summarize_curve([(10, .95, 2000)])["qps95"], 2000)
        with self.assertRaisesRegex(ValueError, "never relabel"):
            measurement.summarize_curve([(10, .9, 2000)], target=.90)
        for unsupported in (
                [], [(10, .9173, 2292)], [(57, .895, 1500), (58, .905, 1300)],
                [(10, .96, 2200), (11, .97, 2000)], [(10, .94, 2200), (12, .96, 2000)],
                [(10, float("nan"), 2000)], [(10, .95, float("inf"))], [(10, .95, 0)]):
            with self.subTest(curve=unsupported), self.assertRaises(ValueError):
                measurement.summarize_curve(unsupported)
        from exp_benchmark import aggregate as static
        self.assertIn("default=0.95", Path(static.__file__).read_text())

    def test_noise_policy_accepts_local_FAISS_only_with_bound_threads_and_pages(self):
        snapshot = {
            "anonymous_mapping_kib": {"N3": 100}, "anonymous_policies_kib": {"bind:3": 99, "local": 1},
            "default_memory_policy": {"mode": 2, "nodes": [3]}, "thread_affinities": {"1": [20, 21]},
        }
        with mock.patch.object(Path, "read_text", return_value="20-23"):
            runtime.require_local(snapshot, 3)
            for key, value in (
                    ("anonymous_mapping_kib", {"N3": 99, "N2": 1}),
                    ("default_memory_policy", {"mode": 0, "nodes": []}),
                    ("thread_affinities", {"1": [20, 90]})):
                with self.assertRaises(RuntimeError):
                    runtime.require_local({**snapshot, key: value}, 3)

    def test_categorical_and_attribute_legacy_entrypoints_removed(self):
        root = Path(pipeline.__file__).parent
        for name in ("attr_update.py", "attr_update.sh", "delete_patch.py", "delete_patch.sh", "update_common.py"):
            self.assertFalse((root / name).exists(), name)
        for name in ("incremental.py", "delete.py", "point_update.py"):
            self.assertIn("pipeline import main", (root / name).read_text())

    def test_legacy_numeric_run_and_retained_result_reuse_are_rejected(self):
        for native_sha in protocol.INCOMPATIBLE_GRAPH_NATIVES:
            with self.assertRaisesRegex(RuntimeError, "Superseded"):
                pipeline.run(argparse.Namespace(expected_sha256=native_sha))
            with self.assertRaisesRegex(RuntimeError, "Superseded"):
                aggregate.publish_paper(None, {}, expected_native=native_sha, current_path=None)
            with self.assertRaisesRegex(RuntimeError, "Superseded"):
                pipeline.native_canary(None, argparse.Namespace(expected_sha256=native_sha))
        self.assertFalse((Path(pipeline.__file__).parent / "retime.py").exists())
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            pipeline.parse_args(["validate-reuse", "--reuse-delete", "/old/delete"])
        config = {"capacity": 10_000_000, "query": protocol.QUERY_CONFIG, "mode": "official"}
        with self.assertRaisesRegex(RuntimeError, "never a legacy base index"):
            protocol.Dataset({"initial_index": Path("/protected/not-opened.index")}, config)

    def test_format10_header_and_rejection_of_all_older_attribute_formats(self):
        root = directory("numeric-formats")
        for version in range(1, 13):
            payload = bytearray()
            def put(fmt, *values):
                payload.extend(struct.pack("<" + fmt, *values))
            def vector(fmt, values):
                put("Q", len(values))
                for value in values:
                    put(fmt, value)
            put("8Q", 0, 2, 1, 2, 3412, 320, 3396, 2884)
            put("iI", 0, 0)
            put("5Q", 40, 80, 2, 16, 40)
            put("dQQii", 1.0, 300, 128, 1, 20)
            vector("i", [0, 1])
            vector("i", [0, 1])
            put("i", 2)
            vector("I", [])
            put("i", 0)
            for _ in range(3):
                vector("I", [])
            put("iQ", 0, 0)
            put("i", 2)
            vector("i", [0, 1])
            put("QQiii", 324, 3404, 4, 32, version)
            payload.extend(bytes(2 * 3412))
            path = root / f"format-{version}.header"
            path.write_bytes(payload)
            if version == 10:
                header = runtime.read_index_header(path)
                self.assertEqual(header["format"], 10)
                self.assertEqual(header["numeric_marker_semantics"], runtime.NUMERIC_MARKER_SEMANTICS)
                self.assertEqual(header["numeric_marker_version"], 2)
                self.assertEqual(header["marker_owner_version"], 2)
                self.assertEqual(header["distance_order_version"], 1)
                self.assertEqual(header["candidate_order"], "vector-distance-only-v1")
            else:
                with self.assertRaisesRegex(ValueError, "format10"):
                    runtime.read_index_header(path)


class CalibrationTests(unittest.TestCase):
    def evidence(self, recalls):
        index = mock.Mock()
        def query():
            ef = index.set_ef.call_args.args[0]
            return np.full((1, 10), ef, dtype=np.uint64), np.full((1, 10), ef, dtype=np.float32)
        def validate(labels, distances):
            return recalls[int(labels[0, 0]) - 10]
        return index, mock.Mock(side_effect=query), validate

    def test_calibration_crosses90_and_stops_only_at_first95_then_freezes_reference(self):
        root = directory("recall95-calibration")
        index, query, validate = self.evidence([.90, .94, .9499, .9501, .99])
        state, labels, distances = measurement.calibrate(index, query, validate, 10, 14, root, "initial")
        self.assertEqual([point["ef"] for point in state["points"]], [10, 11, 12, 13])
        self.assertEqual(state["target_recall"], .95)
        self.assertEqual(query.call_count, 4)
        self.assertEqual(labels.shape, (4, 1, 10))
        self.assertEqual(distances.shape, labels.shape)
        aggregate.validate_calibration(state)
        progress = json.loads((root / "initial-calibration-progress.json").read_text())
        self.assertEqual(progress["target_recall"], .95)
        self.assertEqual(progress["points"], state["points"])
        repeated, rl, rd = measurement.calibrate(index, query, validate, 10, 14, root, "current", state)
        self.assertEqual(repeated["points"], state["points"])
        self.assertTrue(repeated["initial_static_dynamic_identical"])
        np.testing.assert_array_equal(rl, labels)
        np.testing.assert_array_equal(rd, distances)

    def test_maximum_ef_below95_fails_without_timing_fallback(self):
        root = directory("recall95-no-fallback")
        index, query, validate = self.evidence([.90, .94, .9499])
        with self.assertRaisesRegex(RuntimeError, "reached95%.*no timing fallback"):
            measurement.calibrate(index, query, validate, 10, 12, root, "initial")
        self.assertEqual(query.call_count, 3)
        progress = json.loads((root / "initial-calibration-progress.json").read_text())
        self.assertEqual(progress["points"][-1]["recall"], .9499)
        self.assertNotIn("qps95", progress)

    def test_recall90_or_unversioned_reference_is_rejected_even_above95(self):
        root = directory("recall95-legacy-reference")
        index, query, validate = self.evidence([.97])
        for target in (.90, None):
            reference = {"points": [{"ef": 10, "recall": .97}]}
            if target is not None:
                reference["target_recall"] = target
            with self.subTest(target=target), self.assertRaisesRegex(RuntimeError, "recall95"):
                measurement.calibrate(index, query, validate, 10, 10, root, "current", reference)
        index.set_ef.assert_not_called()
        query.assert_not_called()
        self.assertFalse((root / "current-calibration-progress.json").exists())


class JournalTests(unittest.TestCase):
    def store(self):
        root = directory("journal")
        initial = root / "initial.index"
        initial.write_bytes(b"tiny initial")
        output = root / "operation"
        output.mkdir()
        manifest = {"config": {"mode": "pilot", "stage_progress": [0, 10, 20]},
                    "initial_checkpoint": runtime.file_info(initial)}
        return runtime.StageStore(output, manifest=manifest)

    def payload(self, store, stage, status="complete", reuse=None):
        root = Path(store.doc["pending"]["directory"])
        if reuse:
            files = copy.deepcopy(reuse["files"])
        else:
            if stage:
                checkpoint = root / "checkpoint.index"
                checkpoint.write_bytes(f"tiny {stage}".encode())
                info = runtime.file_info(checkpoint)
            else:
                info = store.manifest["initial_checkpoint"]
            files = {"checkpoint": info,
                     "truth": runtime.write_npz(root / "gt.npz", labels=np.asarray([[stage]])),
                     "calibration": runtime.write_npz(root / "cal.npz", labels=np.asarray([[stage]]))}
        files["results"] = runtime.immutable_json(root / "results.json", {"stage": stage})
        return {"stage": stage, "progress": stage * 10, "attempt_id": store.doc["pending"]["attempt_id"],
                "status": status, "files": files}

    def commit(self, store, stage, status="complete"):
        store.begin(stage, stage * 10)
        store.commit(self.payload(store, stage, status))

    def test_manifest_only_bootstrap_fault_and_IO_recovery_preserve_artifacts(self):
        for failure, resume in (("crash", True), ("IO", False)):
            with self.subTest(failure=failure):
                prior = self.store()
                self.commit(prior, 0)
                output = prior.output.parent / "next-operation"
                output.mkdir()
                manifest = copy.deepcopy(prior.manifest)
                manifest["config"]["operation"] = "point_update"
                original_atomic = runtime.atomic_json
                def fault(event):
                    if failure == "crash" and event == "after_manifest":
                        raise InterruptedError(event)
                def write(path, value):
                    if failure == "IO" and Path(path).name == "journal.json":
                        raise OSError("journal publication failed")
                    return original_atomic(path, value)
                with mock.patch.object(runtime, "atomic_json", write), self.assertRaises(OSError):
                    runtime.StageStore(output, manifest=manifest, fault=fault)
                self.assertFalse((output / "journal.json").exists())
                preserved = [runtime.file_info(output / "manifest.json"), runtime.file_info(prior.path),
                             prior.manifest["initial_checkpoint"], *prior.doc["stages"][0]["files"].values()]
                with self.assertRaisesRegex(RuntimeError, "requested immutable manifest"):
                    runtime.StageStore(output, resume=True)
                changed = copy.deepcopy(manifest)
                changed["config"]["stage_progress"] = [0, 20]
                with self.assertRaisesRegex(RuntimeError, "immutable identity"):
                    runtime.StageStore(output, manifest=changed, resume=resume)
                self.assertFalse((output / "journal.json").exists())
                restored = runtime.StageStore(output, manifest=manifest, resume=resume)
                self.assertTrue(restored.doc["bootstrap_recovered"])
                self.assertEqual(restored.doc["stages"], [])
                self.assertIsNone(restored.doc["pending"])
                self.assertEqual(restored.doc["status"], "running")
                for info in preserved:
                    runtime.file_info(info["path"], info)
                restored.verify_committed()
                runtime.StageStore(output, manifest=manifest, resume=True).verify_committed()

    def test_missing_journal_with_attempts_is_not_an_empty_bootstrap(self):
        prior = self.store()
        output = prior.output.parent / "lost-journal"
        attempt = output / "attempts" / "retained"
        attempt.mkdir(parents=True)
        runtime.immutable_json(output / "manifest.json", prior.manifest)
        evidence = runtime.immutable_json(attempt / "stage.json", {"status": "unknown transaction"})
        with self.assertRaisesRegex(RuntimeError, "existing attempts"):
            runtime.StageStore(output, manifest=prior.manifest, resume=True)
        self.assertFalse((output / "journal.json").exists())
        runtime.file_info(evidence["path"], evidence)

    def test_fault_before_commit_replays_previous_and_retains_pending(self):
        store = self.store()
        self.commit(store, 0)
        abandoned = store.begin(1, 10)
        payload = self.payload(store, 1)
        def fault(phase):
            if phase == "before_journal_commit":
                raise InterruptedError(phase)
        store.fault = fault
        with self.assertRaises(InterruptedError):
            store.commit(payload)
        restored = runtime.StageStore(store.output, resume=True)
        restored.verify_committed()
        restored.recover_pending()
        self.assertEqual(len(restored.doc["stages"]), 1)
        self.assertTrue((abandoned / "checkpoint.index").is_file())
        self.commit(restored, 1)
        self.assertTrue((abandoned / "checkpoint.index").is_file())

    def test_postcommit_fault_does_not_repeat_mutation(self):
        store = self.store()
        store.begin(0, 0)
        payload = self.payload(store, 0)
        def fault(phase):
            if phase == "after_journal_commit":
                raise InterruptedError(phase)
        store.fault = fault
        with self.assertRaises(InterruptedError):
            store.commit(payload)
        restored = runtime.StageStore(store.output, resume=True)
        restored.verify_committed()
        self.assertEqual(len(restored.doc["stages"]), 1)
        self.assertIsNone(restored.doc["pending"])

    def test_noise_resume_reuses_checkpoint_GT_and_rejects_corruption(self):
        store = self.store()
        self.commit(store, 0, "needs_timing")
        old = copy.deepcopy(store.doc["stages"][0])
        restored = runtime.StageStore(store.output, resume=True)
        restored.verify_committed()
        restored.recover_pending()
        restored.begin(0, 0, "retime")
        restored.commit(self.payload(restored, 0, reuse=old))
        self.assertEqual(restored.doc["stages"][0]["files"]["checkpoint"], old["files"]["checkpoint"])
        self.assertTrue(Path(old["files"]["results"]["path"]).exists())
        Path(old["files"]["results"]["path"]).write_text("corrupt")
        with self.assertRaises(RuntimeError):
            restored.verify_committed()

    def test_lock_and_manifest_mismatch(self):
        store = self.store()
        with runtime.OutputLock(store.output):
            with self.assertRaises(BlockingIOError):
                with runtime.OutputLock(store.output):
                    pass
        changed = copy.deepcopy(store.manifest)
        changed["config"]["stage_progress"] = [0, 20]
        with self.assertRaises(RuntimeError):
            runtime.StageStore(store.output, manifest=changed, resume=True)

    def test_recall90_operation_schema_is_rejected_without_rewriting_evidence(self):
        store = self.store()
        self.assertEqual(store.doc["schema"], "canonical-dynamic-operation-v4")
        legacy = {**store.doc, "schema": "canonical-dynamic-operation-v3"}
        runtime.atomic_json(store.path, legacy)
        journal = runtime.file_info(store.path)
        manifest = runtime.file_info(store.output / "manifest.json")
        with self.assertRaisesRegex(RuntimeError, "old recall protocols cannot be resumed"):
            runtime.StageStore(store.output, resume=True)
        runtime.file_info(journal["path"], journal)
        runtime.file_info(manifest["path"], manifest)

    def test_truncated_native_save_tail_is_not_committed(self):
        root = directory("truncated-native-tail")
        path = root / "checkpoint.index"
        header = {"records_offset": 0, "count": 2, "record_bytes": 4}
        path.write_bytes(b"12345678" + struct.pack("<I", 3) + b"abc" + struct.pack("<I", 0))
        self.assertTrue(runtime.verify_index_tail(path, header)["exact_file_boundary_verified"])
        for data in (path.read_bytes()[:-1], b"12345678" + struct.pack("<I", 500) + b"short"):
            path.write_bytes(data)
            with self.assertRaises(RuntimeError):
                runtime.verify_index_tail(path, header)


class PipelineBootstrapTests(unittest.TestCase):
    def test_recall90_pipeline_cannot_bootstrap_resume_or_publish_as95(self):
        root = directory("legacy-recall90-pipeline")
        manifest = {"schema": "sift10m-mixed-dnf-dynamic-rebuild-v4", "config": {
            "protocol": "sift10m-mixed-dnf-dynamic-rebuild-v4", "mode": "official",
            "query": {**protocol.QUERY_CONFIG, "target_recall": .90},
        }}
        info = runtime.immutable_json(root / "pipeline-manifest.json", manifest)
        with self.assertRaisesRegex(RuntimeError, "recall95"):
            pipeline.bootstrap_pipeline(root, manifest, resume=True)
        self.assertFalse((root / "pipeline.json").exists())
        runtime.immutable_json(root / "pipeline.json", {"status": "complete", "manifest": info})
        with self.assertRaisesRegex(RuntimeError, "pre-recall95"):
            pipeline.publish_existing(root, root / "current.json")
        self.assertFalse((root / "current.json").exists())
        runtime.file_info(info["path"], info)

    def test_manifest_only_bootstrap_fault_IO_and_identity_checks(self):
        for failure, resume in (("crash", True), ("IO", False)):
            with self.subTest(failure=failure):
                root = directory("pipeline-bootstrap")
                manifest = {
                    "schema": protocol.PROTOCOL, "config": {"operations": ["insert", "point_update"]},
                    "native": {"sha256": "a" * 64}, "source": {"runner": "b" * 64},
                    "inputs": {"source": "c" * 64},
                }
                def fault(event):
                    if event == "after_pipeline_manifest":
                        raise InterruptedError(event)
                if failure == "crash":
                    with self.assertRaises(InterruptedError):
                        pipeline.bootstrap_pipeline(root, manifest, fault=fault)
                else:
                    with mock.patch.object(pipeline, "atomic_json", side_effect=OSError("state publication failed")):
                        with self.assertRaises(OSError):
                            pipeline.bootstrap_pipeline(root, manifest)
                self.assertFalse((root / "pipeline.json").exists())
                info = runtime.file_info(root / "pipeline-manifest.json")
                for key in ("config", "native", "source", "inputs"):
                    changed = copy.deepcopy(manifest)
                    changed[key]["mismatch"] = True
                    with self.assertRaisesRegex(RuntimeError, "immutable identity"):
                        pipeline.bootstrap_pipeline(root, changed, resume=resume)
                    self.assertFalse((root / "pipeline.json").exists())
                    runtime.file_info(info["path"], info)
                state = pipeline.bootstrap_pipeline(root, manifest, resume=resume)
                self.assertTrue(state["bootstrap_recovered"])
                self.assertEqual(state["operations"], {})
                self.assertEqual(state["status"], "running")
                self.assertEqual(state["manifest"], info)
                self.assertEqual(pipeline.bootstrap_pipeline(root, manifest, resume=True), state)

    def test_prepare_output_accepts_matching_manifest_without_state_on_resume(self):
        path = Path("/mnt/data") / f"not-created-unit-fixture-{uuid.uuid4().hex}"
        with mock.patch.object(Path, "is_file", autospec=True,
                               side_effect=lambda p: p.name == "pipeline-manifest.json"), mock.patch.object(
                Path, "is_symlink", return_value=False), mock.patch.object(Path, "mkdir"), mock.patch.object(
                pipeline, "sync_directory"):
            self.assertEqual(pipeline.prepare_output(path, True), path)


class CanaryInvariantTests(unittest.TestCase):
    def test_approximate_recall_is_measured_but_invalid_records_and_distances_fail(self):
        vectors = np.arange(48, dtype=np.float32).reshape(24, 2)
        records = [[[i], [0, 12] if i < 23 else [0]] for i in range(24)]
        queries = np.zeros((1, 2), dtype=np.float32)
        predicates = [[[[0, 0], [9]], [[], [12]]]]
        active = np.ones(24, dtype=bool)
        active[22] = False
        oracle = ExactMixedDNF(vectors, MixedAttributes.from_records(records, 24), threads=1)
        def evidence(labels, distances=None):
            labels = np.asarray([labels], dtype=np.uint64)
            if distances is None:
                distances = np.sum(vectors[labels] ** 2, axis=2)
            index = types.SimpleNamespace(
                get_current_count=lambda: 24,
                hybrid_knn_query_dnf=lambda *args, **kwargs: (labels, distances),
            )
            return canary.query_evidence(index, vectors, records, queries, predicates, active, oracle)
        for first, recall in ((1, .9), (10, 0)):
            report = evidence(list(range(first, first + 10)))
            self.assertEqual(report["recall"], recall)
            self.assertTrue(report["independent_exact_gt_verified"])
        for last, reason in ((9, "duplicate"), (22, "inactive"), (23, "predicate-invalid")):
            with self.subTest(reason=reason), self.assertRaisesRegex(RuntimeError, reason):
                evidence([*range(1, 10), last])
        labels = np.arange(1, 11)
        distances = np.sum(vectors[labels] ** 2, axis=1)[None, :]
        distances[0, 0] = np.nextafter(distances[0, 0], np.float32(np.inf))
        with self.assertRaisesRegex(RuntimeError, "direct squared L2 for returned labels"):
            evidence(labels, distances)

    def test_pipeline_gate_requires_lifecycle_evidence_not_legacy_exactness_claim(self):
        root = directory("canary-consumer")
        args = argparse.Namespace(expected_sha256="a" * 64, native_dir=root)
        good = {
            "schema": canary.SCHEMA, "status": "complete", "native": {"sha256": args.expected_sha256},
            "lifecycle": dict.fromkeys(canary.LIFECYCLE_CHECKS, True),
            "ann": {"recall": .975}, "warnings": ["measured ANN miss"],
            "index_format": 10, "numeric_marker_semantics": runtime.NUMERIC_MARKER_SEMANTICS,
            "numeric_marker_version": 2, "marker_owner_version": 2, "pickle_state_version": 4,
            "distance_order_version": 1, "candidate_order": "vector-distance-only-v1", "attr_sort_alpha": 0,
        }
        for variant in ("measured", "legacy", "missing_check", "legacy_format", "legacy_marker",
                        "legacy_owner", "missing_owner", "legacy_pickle", "legacy_order", "missing_order_version",
                        "mixed_order", "missing_order_policy", "nonzero_alpha", "missing_alpha"):
            with self.subTest(variant=variant):
                report = copy.deepcopy(good)
                if variant == "legacy":
                    report.pop("ann")
                    report.pop("schema")
                    report["query_exact"] = True
                if variant == "missing_check":
                    report["lifecycle"].pop("returned_squared_l2")
                if variant == "legacy_format":
                    report["index_format"] = 8
                if variant == "legacy_marker":
                    report["numeric_marker_version"] = 1
                if variant == "legacy_pickle":
                    report["pickle_state_version"] = 3
                if variant == "legacy_owner":
                    report["marker_owner_version"] = 1
                if variant == "missing_owner":
                    report.pop("marker_owner_version")
                if variant == "legacy_order":
                    report["distance_order_version"] = 0
                if variant == "missing_order_version":
                    report.pop("distance_order_version")
                if variant == "mixed_order":
                    report["candidate_order"] = "vector-plus-attribute"
                if variant == "missing_order_policy":
                    report.pop("candidate_order")
                if variant == "nonzero_alpha":
                    report["attr_sort_alpha"] = 0.5
                if variant == "missing_alpha":
                    report.pop("attr_sort_alpha")
                def subprocess_result(command, **kwargs):
                    output = Path(command[-1])
                    output.mkdir()
                    runtime.immutable_json(output / "report.json", report)
                    return types.SimpleNamespace(returncode=0)
                with mock.patch.object(pipeline.subprocess, "run", subprocess_result):
                    if variant == "measured":
                        info = pipeline.native_canary(root, args)
                        self.assertEqual(json.loads(Path(info["path"]).read_text())["ann"]["recall"], .975)
                    elif variant in ("legacy", "missing_check"):
                        with self.assertRaisesRegex(RuntimeError, "identity/result mismatch"):
                            pipeline.native_canary(root, args)
                    else:
                        with self.assertRaisesRegex(RuntimeError, "format10"):
                            pipeline.native_canary(root, args)

    def test_optional_frozen_native_sources_match_recorded_build(self):
        root = directory("native-build-provenance")
        native = runtime.immutable_json(root / "native.json", {"fixture": "native"})
        (root / "hnswlib").mkdir()
        (root / "python_bindings").mkdir()
        header = runtime.immutable_json(root / "hnswlib/hnswalg.h", {"fixture": "header"})
        bindings = runtime.immutable_json(root / "python_bindings/bindings.cpp", {"fixture": "bindings"})
        setup = runtime.immutable_json(root / "setup.py", {"fixture": "setup"})
        record = {
            "native_sha256": native["sha256"], "hnswalg_sha256": header["sha256"],
            "bindings_sha256": bindings["sha256"], "numeric_marker_version": 2, "marker_owner_version": 2,
            "numeric_edge_file_format": 10, "categorical_edge_file_format": 10, "node_file_format": 9,
            "pickle_state_version": 4, "distance_order_version": 1,
            "candidate_order": "vector-distance-only-v1", "setup_sha256": setup["sha256"],
        }
        path = root / "provenance.json"
        runtime.immutable_json(path, record)
        sources = pipeline.native_source_inventory(root, native)
        self.assertEqual(len(sources), 4)
        self.assertEqual(sources["native/hnswlib/hnswalg.h"], header)
        old_record = {key: value for key, value in record.items() if key != "numeric_edge_file_format"}
        runtime.atomic_json(path, {**old_record, "edge_file_format": 10})
        self.assertEqual(len(pipeline.native_source_inventory(root, native)), 4)
        aliases = {"edge_format_version": "numeric_edge_file_format", "node_format_version": "node_file_format",
                   "serialization_version": "pickle_state_version", "header_sha256": "hnswalg_sha256"}
        alias_record = {key: value for key, value in record.items()
                        if key not in {*aliases.values(), "categorical_edge_file_format"}}
        alias_record.update({alias: record[key] for alias, key in aliases.items()})
        runtime.atomic_json(path, alias_record)
        alias_sources = pipeline.native_source_inventory(root, native)
        self.assertEqual(len(alias_sources), 4)
        self.assertEqual(alias_sources["native/hnswlib/hnswalg.h"], header)
        self.assertEqual(alias_sources["native/python_bindings/bindings.cpp"], bindings)
        self.assertEqual(alias_sources["native/setup.py"], setup)
        for key, wrong in (("numeric_marker_version", 1), ("marker_owner_version", 1),
                           ("pickle_state_version", 3), ("distance_order_version", 0),
                           ("candidate_order", "vector-plus-attribute"), ("native_sha256", "wrong"),
                           ("hnswalg_sha256", "wrong"), ("bindings_sha256", "wrong"),
                           ("numeric_edge_file_format", 8), ("categorical_edge_file_format", 8),
                           ("edge_file_format", 8), ("node_file_format", 7),
                           ("edge_format_version", 8), ("node_format_version", 7),
                           ("serialization_version", 3), ("header_sha256", "wrong"), ("setup_sha256", "wrong")):
            runtime.atomic_json(path, {**record, key: wrong})
            with self.subTest(field=key), self.assertRaises(RuntimeError):
                pipeline.native_source_inventory(root, native)
        for missing in ("numeric_marker_version", "marker_owner_version", "pickle_state_version",
                        "distance_order_version", "candidate_order", "setup_sha256"):
            runtime.atomic_json(path, {key: value for key, value in record.items() if key != missing})
            with self.subTest(missing=missing), self.assertRaises(RuntimeError):
                pipeline.native_source_inventory(root, native)
        for missing in aliases:
            runtime.atomic_json(path, {key: value for key, value in alias_record.items() if key != missing})
            with self.subTest(missing=missing), self.assertRaises(RuntimeError):
                pipeline.native_source_inventory(root, native)
        helpers = {
            name: runtime.immutable_json(root / name, {"fixture": name})
            for name in ("hnswlib/parallel_for.h", "hnswlib/deletion_cache.h")
        }
        auxiliary = {name: info["sha256"] for name, info in helpers.items()}
        runtime.atomic_json(path, {**record, "auxiliary_source_sha256": auxiliary})
        sources = pipeline.native_source_inventory(root, native)
        self.assertEqual(len(sources), 6)
        for name, info in helpers.items():
            self.assertEqual(sources[f"native/{name}"], info)
        (root / "hnswlib/link.h").symlink_to(root / "hnswlib/parallel_for.h")
        (root / "linked-headers").symlink_to(root / "hnswlib", target_is_directory=True)
        for invalid in (
                None, [], {"hnswlib/parallel_for.h": "0" * 64}, {"hnswlib/missing.h": "0" * 64},
                {str(root / "hnswlib/parallel_for.h"): "0" * 64}, {"../parallel_for.h": "0" * 64},
                {"hnswlib/../hnswlib/parallel_for.h": "0" * 64}, {"hnswlib//parallel_for.h": "0" * 64},
                {"": "0" * 64}, {"hnswlib/parallel_for.h": 42},
                {"hnswlib/link.h": auxiliary["hnswlib/parallel_for.h"]},
                {"linked-headers/parallel_for.h": auxiliary["hnswlib/parallel_for.h"]},
                {"hnswlib/hnswalg.h": "0" * 64}):
            runtime.atomic_json(path, {**record, "auxiliary_source_sha256": invalid})
            with self.subTest(auxiliary=invalid), self.assertRaises((RuntimeError, OSError)):
                pipeline.native_source_inventory(root, native)


class SamplingAndPublicationTests(unittest.TestCase):
    def fake_timing(self, accepted, *, pilot=False, above=False):
        cpus = sorted(os.sched_getaffinity(0))
        cores = cpus[:3]
        labels, distances = np.zeros((2, 10), dtype=np.uint64), np.zeros((2, 10), dtype=np.float32)
        class Index:
            def set_ef(self, ef):
                pass
            def hybrid_knn_query_dnf(self, *args, **kwargs):
                return labels, distances
        class Query:
            queries = np.zeros((2, 2), dtype=np.float32)
            predicates = [[[[0, 1], [9]], [[], [12]]]] * 2
            k, query_sha = 10, "test-query"
            gt_shas = {"initial": "test-initial", "current": "test-current"}
            def __call__(self, *args):
                return labels, distances
            def validate(self, *args):
                pass
        counter = 0
        def sampler(call, core, siblings, limits):
            nonlocal counter
            repeat = counter // (2 if above else 4)
            clean = accepted(repeat)
            counter += 1
            return call(), {"wall_s": .001 if repeat == 0 else 1.0, "clean": clean}
        points = [{"ef": 10, "recall": .97}] if above else [{"ef": 10, "recall": .94}, {"ef": 11, "recall": .96}]
        states = {owner: {"points": copy.deepcopy(points), "target_recall": .95}
                  for owner in ("initial", "current")}
        env = {"node": 0, "update_cpus": cpus, "query_candidates": cores,
               "siblings": {str(core): [] for core in cores}}
        return measurement.paired_timing(
            {owner: Index() for owner in states}, states, Query(), env, copy.deepcopy(runtime.DEFAULT_LIMITS),
            lambda report: None, sampler=sampler,
            picker=lambda remaining: (remaining[0], {remaining[0]: 0}), locality=False, pilot=pilot)

    def test_first_clean_pairs_not_fastest_and_no_core_pooling(self):
        report = self.fake_timing(lambda repeat: repeat != 0)
        self.assertEqual(report["target_recall"], .95)
        self.assertEqual(report["qps95_change_fraction"], 0)
        self.assertNotIn("qps90_change_fraction", report)
        self.assertAlmostEqual(report["summary"]["current"]["qps95"], 2)
        self.assertEqual(report["attempts"][0]["accepted_rounds"], list(range(1, 8)))
        self.assertLess(report["summary"]["current"]["observed_qps"], 3)
        report = self.fake_timing(lambda repeat: repeat % 21 < 6)
        self.assertIsNone(report["summary"])
        self.assertIsNone(report["qps95_change_fraction"])
        self.assertEqual(report["status"], "noise_contaminated")
        self.assertTrue(all(len(a["accepted_rounds"]) == 6 for a in report["attempts"]))

    def test_pilot_and_above_minimum_timing_never_claim_QPS95(self):
        for pilot, above in ((True, False), (True, True), (False, True)):
            with self.subTest(pilot=pilot, above=above):
                report = self.fake_timing(lambda repeat: not pilot, pilot=pilot, above=above)
                self.assertEqual(report["status"], "complete")
                self.assertIs(report["official_timing"], not pilot)
                self.assertIsNone(report["qps95_change_fraction"])
                for summary in report["summary"].values():
                    self.assertIsNone(summary["qps95"])
                    self.assertGreater(summary["observed_qps"], 0)
                    self.assertNotIn("qps90", summary)
                    if pilot:
                        self.assertEqual(summary["method"], "NONOFFICIAL_pilot_observation_NO_QPS95")
                if pilot:
                    self.assertIsNone(report["accepted_attempt"])
                    with self.assertRaisesRegex(RuntimeError, "official recall95"):
                        aggregate.validate_timing(report, {}, report["limits"], 0)

    def timing_evidence(self):
        states = {owner: {"target_recall": .95, "points": [
            {"ef": 10, "recall": .94, "labels_sha256": "labels10", "distances_sha256": "distances10"},
            {"ef": 11, "recall": .96, "labels_sha256": "labels11", "distances_sha256": "distances11"},
        ]} for owner in ("initial", "current")}
        limits = {**runtime.DEFAULT_LIMITS, "clean_rounds": 1, "max_rounds_per_core": 1, "max_cores": 1}
        samples, medians, summaries = {}, {}, {}
        for owner in states:
            for point in states[owner]["points"]:
                name = f"{owner}_ef{point['ef']}"
                qps = 1500.0 - 100 * (point["ef"] - 10) - (200 if owner == "current" else 0)
                samples[name] = [{
                    **point, "repeat": 0, "clean": True, "wall_s": 1000 / qps, "qps": qps,
                    "query_payload_sha256": protocol.QUERY_SHA, "sibling_busy_fractions": {},
                    "thread_cpu_wall_ratio": 1, "thread_rusage": {"major_faults": 0},
                    "process_rusage": {"major_faults": 0}, "actual_cpu_before": 0,
                    "actual_cpu_after": 0, "affinity_after": [0],
                }]
                medians[name] = qps
            summaries[owner] = measurement.summarize_curve([
                (p["ef"], p["recall"], medians[f"{owner}_ef{p['ef']}"]) for p in states[owner]["points"]])
        report = {
            "status": "complete", "accepted_attempt": 0, "official_timing": True, "target_recall": .95,
            "limits": limits, "numa_before": {}, "numa_after": {}, "summary": summaries,
            "qps95_change_fraction": summaries["current"]["qps95"] / summaries["initial"]["qps95"] - 1,
            "attempts": [{"core": 0, "samples": samples, "accepted_rounds": [0], "clean": True,
                          "median_qps": medians}],
        }
        return report, states, limits

    def test_timing_validation_requires_actual95_evidence_and_matching_metric_metadata(self):
        report, states, limits = self.timing_evidence()
        with mock.patch.object(aggregate, "require_local"):
            summaries = aggregate.validate_timing(report, states, limits, 0)
            self.assertAlmostEqual(summaries["initial"]["qps95"], 1450)
            self.assertAlmostEqual(summaries["current"]["qps95"], 1250)
            for invalid in ("target", "unversioned", "pilot", "summary_target", "old_metric",
                            "old_status", "old_method", "change", "old_change", "calibration"):
                timing, calibration = copy.deepcopy(report), copy.deepcopy(states)
                summary = timing["summary"]["current"]
                if invalid == "target":
                    timing["target_recall"] = .90
                elif invalid == "unversioned":
                    timing.pop("target_recall")
                elif invalid == "pilot":
                    timing["official_timing"] = False
                elif invalid == "summary_target":
                    summary["target_recall"] = .90
                elif invalid == "old_metric":
                    summary["qps90"] = summary.pop("qps95")
                elif invalid == "old_status":
                    summary["recall_status"] = "target90"
                elif invalid == "old_method":
                    summary["method"] = "NONOFFICIAL_pilot_observation_NO_QPS95"
                elif invalid == "change":
                    timing["qps95_change_fraction"] = 0
                elif invalid == "old_change":
                    timing["qps90_change_fraction"] = timing.pop("qps95_change_fraction")
                else:
                    calibration["current"]["target_recall"] = .90
                with self.subTest(invalid=invalid), self.assertRaises(RuntimeError):
                    aggregate.validate_timing(timing, calibration, limits, 0)

    def components(self):
        rows = []
        for i in range(6):
            rows.append({"operation": "", "stage": i, "live": 5_000_000,
                         "maintenance_wall_s": None if not i else 100.0, "recall": .96 if i == 5 else .95,
                         "ef": 10 if i == 5 else 58, "qps95": None if i == 5 else 1000.0,
                         "target_recall": .95, "paired_initial_qps95": 1000.0,
                         "paired_initial_recall": .95, "paired_initial_ef": 58,
                         "same_recall_qps95_change": None if i == 5 else 0.0})
        result = {}
        prefix, full = {"fixture": "shared-prefix"}, {"fixture": "insert-stage5-full"}
        point = {"ef": 10, "recall": 1.0, "labels_sha256": "same", "distances_sha256": "same"}
        for name in protocol.OPERATIONS:
            own = copy.deepcopy(rows)
            for row in own:
                row["operation"] = name
                phases = {}
                if row["stage"]:
                    if name != "insert":
                        phases["delete"] = {
                            "wall_s": 70.0 if name == "point_update" else 100.0, "scrub_wall_s": 20.0}
                    if name != "delete":
                        phases["add"] = {"wall_s": 30.0 if name == "point_update" else 100.0}
                row.update(runtime.mutation_timing({"wall_s": 100.0, "phases": phases} if phases else None))
                row.update(
                    operation_native_sha256="a" * 64, index_format=10, numeric_marker_version=2,
                    marker_owner_version=2, pickle_state_version=4, distance_order_version=1,
                    candidate_order="vector-distance-only-v1",
                    query_native_sha256="a" * 64, query_implementation=protocol.QUERY_IMPLEMENTATION,
                    queries_retimed=False,
                )
                row["checkpoint"] = (full if (name == "insert" and row["stage"] == 5)
                                     or (name == "delete" and row["stage"] == 0) else {"fixture": f"{name}-{row['stage']}"})
            native = {"sha256": "a" * 64}
            result[name] = {"operation": name, "status": "complete", "fully_verified": True,
                            "reused": False,
                            "native": native, "operation_native": copy.deepcopy(native),
                            "query_native": {"sha256": "a" * 64}, "query_implementation": protocol.QUERY_IMPLEMENTATION,
                            "queries_retimed": False, "rebuild_policy": protocol.REBUILD_POLICY, "index_format": 10,
                            "numeric_marker_version": 2, "marker_owner_version": 2, "pickle_state_version": 4,
                            "distance_order_version": 1, "candidate_order": "vector-distance-only-v1",
                            "initial_checkpoint": full if name == "delete" else prefix,
                            "initial_calibration": {"target_recall": .95, "points": [copy.deepcopy(point)]},
                            "final_calibration": {"target_recall": .95, "points": [copy.deepcopy(point)]},
                            "final_stage_file": {"fixture": f"{name}-final"},
                            "original_manifest": {"fixture": f"{name}-manifest"},
                            "original_journal": {"fixture": f"{name}-journal"},
                            "protocol": protocol.PROTOCOL, "target_recall": .95,
                            "input_sha256": protocol.INPUT_SHA.copy(), "rows": own}
        result["delete"]["initial_full_provenance"] = {
            "kind": "completed_insert_stage5", "checkpoint": full,
            "insert_stage": result["insert"]["final_stage_file"],
            "insert_manifest": result["insert"]["original_manifest"],
            "insert_journal": result["insert"]["original_journal"],
            "initial_query_reference": copy.deepcopy(result["insert"]["final_calibration"]),
        }
        return result

    def test_atomic_only_three_complete_operations_nullable_table_and_original_native(self):
        root = directory("paper")
        current = root / "current.json"
        parts = self.components()
        pointer = aggregate.publish_paper(root, parts, expected_native="a" * 64, current_path=current, latex=True)
        self.assertEqual(pointer["operations"], list(protocol.OPERATIONS))
        self.assertEqual(pointer["operation_native_sha256"]["delete"], "a" * 64)
        self.assertEqual(pointer["query_native_sha256"]["delete"], "a" * 64)
        self.assertEqual(pointer["schema"], "canonical-dynamic-paper-current-v6")
        self.assertEqual(pointer["target_recall"], .95)
        self.assertEqual(pointer["distance_order_version"], 1)
        self.assertEqual(pointer["candidate_order"], "vector-distance-only-v1")
        rows = json.loads((root / "paper.json").read_text())["rows"]
        self.assertEqual(len(rows), 18)
        self.assertIsNone(rows[5]["qps95"])
        self.assertIn("NA", (root / "paper.tsv").read_text())
        self.assertIn("paired_initial_qps95", (root / "paper.tsv").read_text())
        self.assertIn("delete_scrub_wall_s", (root / "paper.tsv").read_text())
        self.assertIn("delete_scrub_fraction", (root / "paper.tsv").read_text())
        self.assertNotIn("qps90", (root / "paper.tsv").read_text())
        self.assertIn("QPS95", (root / "paper.tex").read_text())
        self.assertIn("Scrub s", (root / "paper.tex").read_text())
        self.assertIn("20.000", (root / "paper.tex").read_text())
        self.assertNotIn("QPS90", (root / "paper.tex").read_text())
        original = current.read_bytes()
        for invalid in ("missing", "attribute", "partial", "native", "input", "extrapolated", "metadata_only",
                        "legacy_reuse", "old_query_native", "old_query_epoch", "row_query_native", "legacy_format",
                        "separate_delete_graph", "different_prefix", "query_parity", "old_marker_version",
                        "old_owner_version", "missing_owner_version", "old_state_version", "old_row_owner",
                        "old_order_version", "missing_order_version", "mixed_order", "missing_order_policy",
                        "old_row_order", "mixed_row_order", "old_protocol", "old_target", "old_row_target",
                        "old_metric", "missing_metric", "below_target", "baseline_extrapolated",
                        "retimed", "retimed_row"):
            components = self.components()
            if invalid == "missing":
                del components["delete"]
            elif invalid == "attribute":
                components["attr_update"] = components.pop("point_update")
            elif invalid == "partial":
                components["insert"]["status"] = "paused_noise"
            elif invalid == "native":
                components["point_update"]["native"]["sha256"] = "b" * 64
            elif invalid == "input":
                components["insert"]["input_sha256"]["predicates"] = "wrong"
            elif invalid == "extrapolated":
                components["delete"]["rows"][5]["qps95"] = 1200.0
            elif invalid == "metadata_only":
                components["delete"]["fully_verified"] = False
            elif invalid == "legacy_reuse":
                components["delete"]["reused"] = True
            elif invalid == "old_query_native":
                components["delete"]["query_native"]["sha256"] = aggregate.DELETE_NATIVE_SHA
            elif invalid == "old_query_epoch":
                components["delete"]["query_implementation"] = "legacy-upper-slot-exclusion"
            elif invalid == "row_query_native":
                components["delete"]["rows"][0]["query_native_sha256"] = aggregate.DELETE_NATIVE_SHA
            elif invalid == "legacy_format":
                components["delete"]["index_format"] = 8
            elif invalid == "separate_delete_graph":
                components["delete"]["initial_checkpoint"] = {"fixture": "independent full build"}
            elif invalid == "different_prefix":
                components["point_update"]["initial_checkpoint"] = {"fixture": "random surviving prefix"}
            elif invalid == "old_marker_version":
                components["delete"]["numeric_marker_version"] = 1
            elif invalid == "old_owner_version":
                components["delete"]["marker_owner_version"] = 1
            elif invalid == "missing_owner_version":
                components["delete"].pop("marker_owner_version")
            elif invalid == "old_state_version":
                components["delete"]["pickle_state_version"] = 3
            elif invalid == "old_row_owner":
                components["delete"]["rows"][0]["marker_owner_version"] = 1
            elif invalid == "old_order_version":
                components["delete"]["distance_order_version"] = 0
            elif invalid == "missing_order_version":
                components["delete"].pop("distance_order_version")
            elif invalid == "mixed_order":
                components["delete"]["candidate_order"] = "vector-plus-attribute"
            elif invalid == "missing_order_policy":
                components["delete"].pop("candidate_order")
            elif invalid == "old_row_order":
                components["delete"]["rows"][0]["distance_order_version"] = 0
            elif invalid == "mixed_row_order":
                components["delete"]["rows"][0]["candidate_order"] = "vector-plus-attribute"
            elif invalid == "old_protocol":
                components["delete"]["protocol"] = "sift10m-mixed-dnf-dynamic-rebuild-v4"
            elif invalid == "old_target":
                components["delete"]["target_recall"] = .90
            elif invalid == "old_row_target":
                components["delete"]["rows"][0]["target_recall"] = .90
            elif invalid == "old_metric":
                row = components["delete"]["rows"][0]
                row["qps90"] = row.pop("qps95")
            elif invalid == "missing_metric":
                components["delete"]["rows"][5].pop("qps95")
            elif invalid == "below_target":
                components["delete"]["rows"][0]["recall"] = .91
            elif invalid == "baseline_extrapolated":
                components["delete"]["rows"][0].update(paired_initial_ef=10, paired_initial_recall=.97)
            elif invalid == "retimed":
                components["delete"]["queries_retimed"] = True
            elif invalid == "retimed_row":
                components["delete"]["rows"][0]["queries_retimed"] = True
            else:
                components["delete"]["initial_calibration"]["points"][0]["labels_sha256"] = "different"
            with self.subTest(invalid=invalid), self.assertRaises(RuntimeError):
                aggregate.publish_paper(root, components, expected_native="a" * 64, current_path=current)
            self.assertEqual(current.read_bytes(), original)

    def test_new95_pointer_can_replace90_pointer_without_relabeling_old_output(self):
        root = directory("recall95-pointer-revision")
        legacy_paper = runtime.immutable_json(root / "paper90.json", {"target_recall": .90, "qps90": 1900})
        legacy_tsv = root / "paper90.tsv"
        legacy_tsv.write_text("qps90\n1900\n")
        old_table = runtime.file_info(legacy_tsv)
        current = root / "current.json"
        runtime.immutable_json(current, {
            "schema": "canonical-dynamic-paper-current-v5", "status": "complete",
            "protocol": "sift10m-mixed-dnf-dynamic-rebuild-v4", "operations": list(protocol.OPERATIONS),
            "paper_json": legacy_paper, "paper_tsv": old_table,
        })
        fresh = root / "fresh95"
        fresh.mkdir()
        pointer = aggregate.publish_paper(fresh, self.components(), expected_native="a" * 64, current_path=current)
        self.assertEqual(pointer["target_recall"], .95)
        self.assertEqual(pointer["schema"], aggregate.CURRENT_SCHEMA)
        runtime.file_info(legacy_paper["path"], legacy_paper)
        runtime.file_info(old_table["path"], old_table)

    def test_no_published_manifest_on_interrupted_export_or_nonpointer_target(self):
        root = directory("interrupted-paper")
        current = root / "current.json"
        with mock.patch.object(aggregate, "write_text_once", side_effect=InterruptedError("export interrupted")):
            with self.assertRaises(InterruptedError):
                aggregate.publish_paper(root, self.components(), expected_native="a" * 64, current_path=current)
        self.assertFalse(current.exists())
        aggregate.publish_paper(root, self.components(), expected_native="a" * 64, current_path=current)
        target = root / "unrelated.json"
        target.write_text('{"important":"unchanged"}')
        with self.assertRaises(RuntimeError):
            aggregate.publish_paper(root, self.components(), expected_native="a" * 64, current_path=target)
        self.assertEqual(json.loads(target.read_text()), {"important": "unchanged"})

    def test_calibration_and_partial_operation_cannot_enter_paper(self):
        valid = {"target_recall": .95, "points": [{"ef": 10, "recall": .94}, {"ef": 11, "recall": .95}]}
        aggregate.validate_calibration(valid)
        aggregate.validate_calibration({"target_recall": .95, "points": [{"ef": 10, "recall": .97}]})
        for points in ([{"ef": 10, "recall": .95}, {"ef": 11, "recall": .96}],
                       [{"ef": 11, "recall": .96}], [{"ef": 10, "recall": .94}],
                       [{"ef": 10, "recall": .89}, {"ef": 11, "recall": .91}]):
            with self.assertRaises(RuntimeError):
                aggregate.validate_calibration({"target_recall": .95, "points": points})
        for state in ({"points": valid["points"]}, {**valid, "target_recall": .90}):
            with self.assertRaises(RuntimeError):
                aggregate.validate_calibration(state)
        fake = types.SimpleNamespace(
            manifest={"config": {"operation": "insert", "mode": "pilot"}},
            doc={"schema": runtime.SCHEMA, "status": "complete"})
        with self.assertRaises(RuntimeError):
            aggregate.validate_operation(fake, {}, {})


class FakeIndex:
    """Independent tiny ANN stand-in: exact enumeration, with append-only occupied slots."""
    def __init__(self, space, dim):
        self.dim, self.rows, self.retired = dim, {}, set()
        self.M, self.ef_construction, self.capacity = 40, 300, 0
    def init_index(self, max_elements, **kwargs):
        self.capacity = max_elements
    def initAttrMapping(self, records):
        if len(records) != self.capacity:
            raise AssertionError("codebook must see all future complete records")
    def add_items(self, vectors, records, ids, replace_deleted=False, **kwargs):
        if replace_deleted:
            raise AssertionError("slot reuse forbidden")
        for vector, record, label in zip(vectors, records, ids):
            label = int(label)
            if label in self.rows or len(self.rows) >= self.capacity:
                raise RuntimeError("duplicate/reused/full")
            self.rows[label] = (vector.tolist(), copy.deepcopy(record))
    def delete_items(self, ids, **kwargs):
        if any(int(label) not in self.rows or int(label) in self.retired for label in ids):
            raise RuntimeError("bad deletion")
        self.retired.update(int(label) for label in ids)
        result = {name: 0 for name in runtime.COUNTERS}
        result["requested"] = result["marked"] = len(ids)
        return result
    def get_current_count(self):
        return len(self.rows)
    def get_max_elements(self):
        return self.capacity
    def get_deleted_ratio(self):
        return len(self.retired) / len(self.rows)
    def get_dirty_count(self):
        return 0
    def repair_candidates_size(self):
        return 0
    def index_file_size(self):
        return 100000
    def save_index(self, path):
        Path(path).write_text(json.dumps({"rows": self.rows, "retired": sorted(self.retired),
                                        "capacity": self.capacity, "format": 10}))
    def load_index(self, path, **kwargs):
        saved = json.loads(Path(path).read_text())
        self.rows = {int(key): value for key, value in saved["rows"].items()}
        self.retired, self.capacity = set(saved["retired"]), saved["capacity"]
    def hybrid_knn_query_dnf(self, queries, predicates, k, num_threads):
        if num_threads != 1:
            raise AssertionError("query threads must stay1")
        rows = sorted(self.rows)
        vectors = np.zeros((self.capacity, self.dim), dtype=np.float32)
        records = [[[0], []] for _ in range(self.capacity)]
        active = np.zeros(self.capacity, dtype=bool)
        for label in rows:
            vectors[label], records[label] = self.rows[label]
            active[label] = label not in self.retired
        labels, distances, _ = brute_force(vectors, records, queries, predicates, active, k)
        return labels.astype(np.uint64), distances
    def set_ef(self, ef):
        self.ef = ef
    def get_ft_stats(self):
        return {}
    def __getattr__(self, name):
        if name.startswith("set_") or name in ("generateAttrIndexes", "reset_ft_stats"):
            return lambda *args: None
        raise AttributeError(name)


def operation_fixture(operation, native, *, output_name="run"):
    root = directory(f"operation-{operation}")
    n, initial, step, dim = 120, 60, 12, 8
    rng = np.random.default_rng(41)
    vectors = rng.integers(0, 128, (n, dim)).astype(np.float32)
    records = [[[i % 101], sorted({0, i % 20} | ({9} if i % 2 else set())
                                | ({12} if i % 3 else set()))] for i in range(n)]
    predicates = [[[[i, i + 30], [9]], [[], [12]]] for i in range(4)]
    query = {**protocol.QUERY_CONFIG, "dimension": dim, "query_count": 4}
    inputs = {}
    for name in ("base_vectors", "attributes", "queries", "predicates", "canonical_gt0"):
        path = root / f"{name}.json"
        runtime.immutable_json(path, {"fixture": name})
        inputs[name] = runtime.file_info(path)
    data = types.SimpleNamespace(
        n=n, dim=dim, k=10, base=vectors, records=records, queries=vectors[10:14],
        predicates=predicates, bounds=predicate_bounds(predicates), inputs=inputs,
        attributes=MixedAttributes.from_records(records, n),
        identity={"fixture": True, "vectors_sha256": array_digest(vectors)},
        verify=lambda: [runtime.require_unchanged(info) for info in inputs.values()],
    )
    data.oracle = ExactMixedDNF(vectors, data.attributes, threads=1)
    cpus = sorted(os.sched_getaffinity(0))
    config = {
        "mode": "fixture", "operation": operation, "capacity": n, "initial": initial, "step": step,
        "stage_progress": list(range(0, initial + 1, step)), "threads_update": 1, "threads_build": 1,
        "threads_gt": 1, "level_seed": 1234, "add_chunk": 24, "query": query,
        "max_query_ef": 300, "limits": copy.deepcopy(runtime.DEFAULT_LIMITS),
        "environment": {"node": 0, "update_cpus": cpus, "query_candidates": [cpus[0]],
                        "siblings": {str(cpus[0]): runtime.siblings(cpus[0])}},
    }
    source = runtime.file_info(Path(runner.__file__))
    native_file = root / "native-identity.json"
    runtime.immutable_json(native_file, {"fixture_backend": str(native)})
    native_info = runtime.file_info(native_file)
    if hasattr(native, "__file__"):
        native_info = runtime.file_info(native.__file__)
    sources = {"runner.py": source}
    levels, plan = runner.prepare_levels(root, data, config)
    if operation == "delete":
        index = native.Index(space="l2", dim=dim)
        index.init_index(max_elements=n, top_elements=int(levels.sum()), M=40, ef_construction=300,
                         ft_bits=128, attr_type=[0, 1], max_cate_size=20, edge_level_ft=True,
                         allow_replace_deleted=False)
        index.initAttrMapping(records)
        runner.add_source_rows(index, data, levels, 0, n, 1, 24)
        checkpoint = root / "initial.index"
        index.save_index(str(checkpoint))
        initial_info = runtime.file_info(checkpoint)
    else:
        cache = runner.prefix_checkpoint(root, data, config, native, native_info, levels, plan, sources)
        initial_info = cache["checkpoint"]
    output = root / output_name
    output.mkdir()
    manifest = {"config": config, "initial_checkpoint": initial_info, "inputs": inputs,
                "native": native_info, "source": sources, "dataset": data.identity}
    return runtime.StageStore(output, manifest=manifest), data, native, levels


@contextlib.contextmanager
def fake_serialization():
    def header(path):
        content = json.loads(Path(path).read_text())
        return {"count": len(content["rows"]), "max_elements": content["capacity"],
                "M": 40, "ef_construction": 300, "format": content["format"]}
    def verify(path, header, active, attrs, vectors):
        content = json.loads(Path(path).read_text())
        ids = np.asarray(sorted(map(int, content["rows"])), dtype=np.int64)
        np.testing.assert_array_equal(ids, np.arange(header["count"]))
        retired = set(content["retired"])
        if any(active[i] == (i in retired) for i in ids):
            raise AssertionError("inline state mismatch")
        for i in ids:
            row = content["rows"][str(i)]
            np.testing.assert_array_equal(row[0], vectors[i])
            self_record = row[1]
            if self_record[0][0] != attrs.numeric[i] or (9 in self_record[1]) != attrs.label9[i] or (
                    12 in self_record[1]) != attrs.label12[i]:
                raise AssertionError("replacement did not use complete original attributes")
        return {"occupied": len(ids), "marked_count": len(retired), "live": int(active.sum())}
    with mock.patch.object(runner, "read_index_header", header), mock.patch.object(
            runner, "verify_index_records", verify), mock.patch.object(
            runner, "verify_index_tail", return_value={"fixture": True}):
        yield


def correctness_measure(worker, states, truth, initial_truth, directory):
    query = measurement.FrozenQuery(worker, {"initial": initial_truth, "current": truth})
    for owner in ("initial", "current"):
        for point in measurement.bracket(states[owner]):
            worker.indexes[owner].set_ef(point["ef"])
            query(owner, point)
    return {"status": "complete", "summary": None, "fixture_only": "no performance claim"}


class OperationIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.affinity = os.sched_getaffinity(0)
    def tearDown(self):
        os.sched_setaffinity(0, self.affinity)

    def test_all_operations_initial_save_load_mutation_GT_pause_resume(self):
        for operation in protocol.OPERATIONS:
            with self.subTest(operation=operation), fake_serialization(), mock.patch.object(
                    runner.OperationRunner, "measure", correctness_measure):
                store, data, native, levels = operation_fixture(operation, types.SimpleNamespace(Index=FakeIndex))
                worker = runner.OperationRunner(store, data, native, levels)
                self.assertFalse(worker.run(through_stage=1))
                self.assertEqual(store.doc["status"], "paused")
                retained = copy.deepcopy(store.doc["stages"][1])
                mutation = retained["details"]["mutation"]
                if operation != "insert":
                    self.assertIsNone(mutation["phases"]["delete"]["scrub_wall_s"])
                    self.assertIsNone(mutation["timing_breakdown"]["delete_scrub_wall_s"])
                    self.assertIsNone(mutation["timing_breakdown"]["delete_non_scrub_wall_s"])
                with np.load(retained["files"]["truth"]["path"]) as gt:
                    _, distances, counts = brute_force(data.base, data.records, data.queries, data.predicates,
                                                       worker.state["active"])
                    np.testing.assert_array_equal(gt["distances"], distances)
                    np.testing.assert_array_equal(gt["matching_counts"], counts)
                del worker
                restored = runtime.StageStore(store.output, resume=True)
                restored.verify_committed()
                restored.recover_pending()
                worker = runner.OperationRunner(restored, data, native, levels)
                self.assertTrue(worker.run())
                self.assertEqual(restored.doc["status"], "complete")
                self.assertEqual(len(restored.doc["stages"]), 6)
                self.assertTrue(Path(retained["files"]["checkpoint"]["path"]).exists())
                self.assertTrue(Path(retained["files"]["truth"]["path"]).exists())
                final = restored.doc["stages"][-1]["state"]
                self.assertEqual(final["occupied"], data.n)
                self.assertEqual(final["live"], data.n if operation == "insert" else data.n // 2)
                restored.verify_committed()

    def test_delete_and_update_persist_native_scrub_time_without_double_counting(self):
        original_delete = FakeIndex.delete_items

        def instrumented_delete(index, ids, **kwargs):
            return {**original_delete(index, ids, **kwargs), "scrub_wall_s": 10.0}

        def fixed_timing(call):
            result = call()
            return result, {"wall_s": 40.0 if isinstance(result, dict) else 20.0}

        for operation in ("delete", "point_update"):
            with self.subTest(operation=operation), fake_serialization(), mock.patch.object(
                    runner.OperationRunner, "measure", correctness_measure):
                store, data, native, levels = operation_fixture(operation, types.SimpleNamespace(Index=FakeIndex))
                worker = runner.OperationRunner(store, data, native, levels)
                with mock.patch.object(FakeIndex, "delete_items", instrumented_delete), mock.patch.object(
                        runner, "timed_mutation", fixed_timing):
                    self.assertFalse(worker.run(through_stage=1))
                stage = store.doc["stages"][1]
                mutation = stage["details"]["mutation"]
                self.assertEqual(mutation["phases"]["delete"]["scrub_wall_s"], 10.0)
                self.assertEqual(mutation["phases"]["delete"]["wall_s"], 40.0)
                self.assertEqual(mutation["wall_s"], 40.0 if operation == "delete" else 60.0)
                self.assertEqual(mutation["timing_breakdown"]["delete_non_scrub_wall_s"], 30.0)
                self.assertEqual(mutation["timing_breakdown"]["delete_scrub_fraction"], 0.25)
                self.assertNotIn("scrub_wall_s", mutation["counters"])
                runtime.validate_counters(mutation["counters"], 12)
                saved = json.loads(Path(stage["files"]["results"]["path"]).read_text())["mutation"]
                self.assertEqual(saved, mutation)
                restored = runtime.StageStore(store.output, resume=True)
                self.assertEqual(restored.doc["stages"][1]["details"]["mutation"], mutation)

    def test_crash_between_delete_and_fresh_add_never_commits_partial_update(self):
        with fake_serialization(), mock.patch.object(runner.OperationRunner, "measure", correctness_measure):
            store, data, native, levels = operation_fixture("point_update", types.SimpleNamespace(Index=FakeIndex))
            def fault(phase):
                if phase == "after_delete":
                    raise InterruptedError("between delete and add")
            worker = runner.OperationRunner(store, data, native, levels, fault=fault)
            with self.assertRaises(InterruptedError):
                worker.run()
            self.assertEqual(len(store.doc["stages"]), 1)
            self.assertEqual(store.doc["pending"]["progress"], 12)
            checkpoint = store.doc["stages"][0]["files"]["checkpoint"]
            restored = runtime.StageStore(store.output, resume=True)
            restored.verify_committed()
            restored.recover_pending()
            worker = runner.OperationRunner(restored, data, native, levels)
            self.assertTrue(worker.run())
            self.assertEqual(restored.doc["stages"][0]["files"]["checkpoint"], checkpoint)
            self.assertEqual(restored.doc["stages"][1]["details"]["mutation"]["counters"]["marked"], 12)

    def test_operation_noise_retime_does_not_repeat_replacement(self):
        with fake_serialization():
            store, data, native, levels = operation_fixture("point_update", types.SimpleNamespace(Index=FakeIndex))
            def noisy(worker, states, truth, initial_truth, path):
                result = correctness_measure(worker, states, truth, initial_truth, path)
                if worker.progress:
                    result["status"] = "noise_contaminated"
                return result
            with mock.patch.object(runner.OperationRunner, "measure", noisy):
                worker = runner.OperationRunner(store, data, native, levels)
                self.assertFalse(worker.run())
            self.assertEqual(store.doc["status"], "paused_noise")
            self.assertEqual(len(store.doc["stages"]), 2)
            previous = copy.deepcopy(store.doc["stages"][-1])
            restored = runtime.StageStore(store.output, resume=True)
            restored.verify_committed()
            restored.recover_pending()
            with mock.patch.object(runner.OperationRunner, "measure", correctness_measure), mock.patch.object(
                    runner.OperationRunner, "mutate", side_effect=AssertionError("retime repeated mutation")):
                worker = runner.OperationRunner(restored, data, native, levels)
                self.assertFalse(worker.run(through_stage=1))
            current = restored.doc["stages"][-1]
            self.assertEqual(current["files"]["checkpoint"], previous["files"]["checkpoint"])
            self.assertEqual(current["files"]["truth"], previous["files"]["truth"])
            self.assertEqual(current["details"]["mutation"], previous["details"]["mutation"])
            self.assertTrue(Path(previous["files"]["results"]["path"]).exists())
            restored.verify_committed()

    def test_explicit_missing_prefix_cache_never_silently_rebuilds(self):
        with fake_serialization():
            store, data, native, levels = operation_fixture("insert", types.SimpleNamespace(Index=FakeIndex))
            missing = directory("missing-prefix")
            with self.assertRaisesRegex(RuntimeError, "missing"):
                runner.prefix_checkpoint(missing, data, store.manifest["config"], native, store.manifest["native"],
                                         levels, {"details": {}}, store.manifest["source"], explicit=missing)

    @contextlib.contextmanager
    def pipeline_fixture(self):
        root = directory("pipeline-pilot")
        data_root = root / "dataset"
        label_root = data_root / "label/arbi_0_1_random"
        label_root.mkdir(parents=True)
        rng = np.random.default_rng(910)
        vectors = rng.integers(0, 128, (1000, 8)).astype(np.float32)
        records = [[[i % 101], [0] + ([9] if i % 2 else []) + ([12] if i % 3 else [])]
                   for i in range(len(vectors))]
        predicates = [[[[i, 50 + i], [9]], [[], [12]]] for i in range(4)]
        queries = vectors[10:14].copy()
        def fvecs(path, array):
            rows = np.empty((len(array), array.shape[1] + 1), dtype=np.int32)
            rows[:, 0] = array.shape[1]
            rows[:, 1:] = array.view(np.int32)
            path.write_bytes(rows.tobytes())
        fvecs(data_root / "sift10m.fvecs", vectors)
        fvecs(data_root / "sift10m_query.fvecs", queries)
        for name, value in (
                ("attr_arbi_0_1_random.json", records), ("predicate_dnf_or_T10.json", predicates),
                ("gt_dnf_or_T10.json", brute_force(vectors, records, queries, predicates,
                                                  np.ones(len(vectors), dtype=bool))[0].tolist())):
            runtime.immutable_json(label_root / name, value)
        native_file = root / "fake-native.json"
        runtime.immutable_json(native_file, {"backend": "unit-test only"})
        native_info = runtime.file_info(native_file)
        output = root / "output"
        cpus = sorted(self.affinity)
        environment = {"node": 0, "update_cpus": cpus, "query_candidates": [cpus[0]],
                       "siblings": {str(cpus[0]): runtime.siblings(cpus[0])}}
        args = pipeline.parse_args([
            "run", "--data-root", str(data_root), "--output", str(output),
            "--native-dir", str(root), "--expected-sha256", native_info["sha256"],
            "--numa-node", "0", "--pilot-size", "1000", "--threads-build", "1",
            "--threads-update", "1", "--threads-gt", "1", "--through-stage", "1",
        ])
        def prepare(path, resume):
            path.mkdir(exist_ok=True)
            return path
        query_config = {**protocol.QUERY_CONFIG, "dimension": 8, "query_count": 4}
        with fake_serialization(), mock.patch.object(pipeline, "prepare_output", prepare), mock.patch.object(
                pipeline, "QUERY_CONFIG", query_config), mock.patch.object(
                pipeline, "environment_config", return_value=environment), mock.patch.object(
                pipeline, "import_native", return_value=(types.SimpleNamespace(Index=FakeIndex), native_info)), mock.patch.object(
                pipeline, "native_canary", return_value=native_info), mock.patch.object(
                runner.OperationRunner, "measure", correctness_measure):
            yield output, args

    def test_complete_pipeline_pilot_pause_resume_and_shared_prefix(self):
        with self.pipeline_fixture() as (output, args), mock.patch.object(
                pipeline, "full_checkpoint", side_effect=AssertionError("combined run built an extra full index")):
            self.assertEqual(pipeline.run(args), 0)
            self.assertEqual(json.loads((output / "pipeline.json").read_text())["status"], "paused")
            self.assertFalse((output / "current.json").exists())
            args.resume, args.through_stage = True, None
            self.assertEqual(pipeline.run(args), 0)
        summary = json.loads((output / "pilot-summary.json").read_text())
        self.assertEqual(summary["status"], "complete")
        self.assertFalse(summary["official"])
        self.assertEqual(set(summary["operations"]), set(protocol.OPERATIONS))
        inserted = runtime.StageStore(output / "insert", resume=True)
        deleted = runtime.StageStore(output / "delete", resume=True)
        replaced = runtime.StageStore(output / "point_update", resume=True)
        self.assertEqual(inserted.manifest["initial_checkpoint"], replaced.manifest["initial_checkpoint"])
        self.assertEqual(inserted.doc["stages"][-1]["state"]["live"], 1000)
        self.assertEqual(replaced.doc["stages"][-1]["state"], {"occupied": 1000, "deleted": 500, "live": 500})
        self.assertEqual(deleted.manifest["initial_checkpoint"], inserted.doc["stages"][5]["files"]["checkpoint"])
        self.assertEqual(deleted.manifest["initial_full_provenance"]["kind"], "completed_insert_stage5")
        self.assertEqual(deleted.manifest["initial_full_provenance"]["distance_order_version"], 1)
        self.assertEqual(deleted.manifest["initial_full_provenance"]["candidate_order"], "vector-distance-only-v1")
        prefix = json.loads((output / "initial-prefix/prefix-cache.json").read_text())
        self.assertEqual(prefix["fingerprint"]["distance_order_version"], 1)
        self.assertEqual(prefix["fingerprint"]["candidate_order"], "vector-distance-only-v1")
        for store in (inserted, deleted, replaced):
            for stage in store.doc["stages"]:
                results = json.loads(Path(stage["files"]["results"]["path"]).read_text())
                self.assertEqual(results["distance_order_version"], 1)
                self.assertEqual(results["candidate_order"], "vector-distance-only-v1")
        reference = inserted.doc["stages"][5]["calibrations"]["current"]["points"]
        actual = deleted.doc["stages"][0]["calibrations"]["initial"]["points"]
        for a, b in zip(reference, actual):
            for key in ("ef", "recall", "labels_sha256", "distances_sha256"):
                self.assertEqual(a[key], b[key])
        runtime.file_info(deleted.manifest["initial_checkpoint"]["path"], deleted.manifest["initial_checkpoint"])
        self.assertFalse((output / "initial-full").exists())
        self.assertFalse((output / "paper.json").exists())

    def test_delete_alone_builds_and_resumes_one_fresh_full_cache(self):
        with self.pipeline_fixture() as (output, args):
            args.operations = ["delete"]
            self.assertEqual(pipeline.run(args), 0)
            self.assertFalse((output / "initial-prefix").exists())
            cache = json.loads((output / "initial-full/full-cache.json").read_text())
            self.assertEqual(cache["fingerprint"]["kind"], "full")
            self.assertEqual(cache["fingerprint"]["initial"], 1000)
            self.assertEqual(cache["fingerprint"]["index_format"], 10)
            self.assertEqual(cache["fingerprint"]["marker_owner_version"], 2)
            self.assertEqual(cache["fingerprint"]["pickle_state_version"], 4)
            self.assertEqual(cache["fingerprint"]["distance_order_version"], 1)
            self.assertEqual(cache["fingerprint"]["candidate_order"], "vector-distance-only-v1")
            self.assertEqual(cache["past_last_source_row"], 1000)
            args.resume, args.through_stage = True, None
            with mock.patch.object(FakeIndex, "init_index", side_effect=AssertionError("rebuilt initial full graph")):
                self.assertEqual(pipeline.run(args), 0)
            deleted = runtime.StageStore(output / "delete", resume=True)
            self.assertEqual(deleted.manifest["initial_checkpoint"], cache["checkpoint"])
            self.assertEqual(deleted.manifest["initial_full_provenance"]["kind"], "fresh_full_cache")
            self.assertEqual(deleted.doc["status"], "complete")
            runtime.file_info(cache["checkpoint"]["path"], cache["checkpoint"])

    def test_insert_delete_query_parity_failure_stops_before_deletion(self):
        with self.pipeline_fixture() as (output, args):
            args.operations, args.through_stage = ["insert", "delete"], None
            original = pipeline.completed_insert_checkpoint
            def changed(*values):
                result = original(*values)
                result["initial_query_reference"]["points"][0]["labels_sha256"] = "different result"
                return result
            with mock.patch.object(pipeline, "completed_insert_checkpoint", side_effect=changed), mock.patch.object(
                    FakeIndex, "delete_items", side_effect=AssertionError("deletion ran before parity verification")):
                with self.assertRaisesRegex(RuntimeError, "frozen same-graph reference"):
                    pipeline.run(args)
            inserted = runtime.StageStore(output / "insert", resume=True)
            self.assertEqual(inserted.doc["status"], "complete")
            deleted = runtime.StageStore(output / "delete", resume=True)
            self.assertEqual(deleted.doc["stages"], [])
            self.assertFalse((output / "paper.json").exists())

    def test_explicit_full_cache_missing_or_wrong_format_is_not_rebuilt(self):
        with self.pipeline_fixture() as (output, args):
            args.operations = ["delete"]
            args.initial_full = directory("missing-full-cache")
            with mock.patch.object(FakeIndex, "init_index", side_effect=AssertionError("silently rebuilt explicit cache")):
                with self.assertRaisesRegex(RuntimeError, "Explicit full cache is missing"):
                    pipeline.run(args)
        with fake_serialization(), mock.patch.object(runner.OperationRunner, "measure", correctness_measure):
            store, data, native, levels = operation_fixture("insert", types.SimpleNamespace(Index=FakeIndex))
            root = store.output.parent
            plan = json.loads((root / "levels.json").read_text())
            full = runner.full_checkpoint(root, data, store.manifest["config"], native,
                                          store.manifest["native"], levels, plan, store.manifest["source"])
            cache_path = root / "initial-full/full-cache.json"
            graph = Path(full["checkpoint"]["path"])
            for key, value in (("distance_order_version", None), ("candidate_order", "vector-plus-attribute")):
                stale = copy.deepcopy(full)
                if value is None:
                    stale["fingerprint"].pop(key)
                else:
                    stale["fingerprint"][key] = value
                runtime.atomic_json(cache_path, stale)
                with self.subTest(stale=key), mock.patch.object(
                        FakeIndex, "init_index", side_effect=AssertionError("rebuilt incompatible ranking cache")):
                    with self.assertRaisesRegex(RuntimeError, "Incompatible"):
                        runner.full_checkpoint(root, data, store.manifest["config"], native,
                                               store.manifest["native"], levels, plan, store.manifest["source"],
                                               explicit=cache_path.parent)
                runtime.file_info(graph, full["checkpoint"])
            value = json.loads(graph.read_text())
            value["format"] = 8
            graph.write_text(json.dumps(value))
            full["checkpoint"] = runtime.file_info(graph)
            runtime.atomic_json(cache_path, full)
            with mock.patch.object(FakeIndex, "init_index", side_effect=AssertionError("rebuilt incompatible full cache")):
                with self.assertRaisesRegex(RuntimeError, "incompatible"):
                    runner.full_checkpoint(root, data, store.manifest["config"], native,
                                           store.manifest["native"], levels, plan, store.manifest["source"],
                                           explicit=cache_path.parent)

    def test_pipeline_bootstrap_interruption_is_resumable_without_replacing_manifest(self):
        with self.pipeline_fixture() as (output, args):
            def fault(event):
                if event == "after_pipeline_manifest":
                    raise InterruptedError(event)
            with self.assertRaises(InterruptedError):
                pipeline.run(args, fault=fault)
            self.assertFalse((output / "pipeline.json").exists())
            self.assertFalse((output / "initial-prefix").exists())
            info = runtime.file_info(output / "pipeline-manifest.json")
            args.resume = True
            self.assertEqual(pipeline.run(args), 0)
            state = json.loads((output / "pipeline.json").read_text())
            self.assertTrue(state["bootstrap_recovered"])
            self.assertEqual(state["status"], "paused")
            runtime.file_info(info["path"], info)

    def test_bootstrap_recovery_preserves_built_prefix_prior_operation_and_rejects_mismatch(self):
        with self.pipeline_fixture() as (output, args):
            args.operations, args.through_stage = ["insert", "point_update"], None
            def fault(event):
                if event == "operation_point_update:after_manifest":
                    raise InterruptedError(event)
            with self.assertRaises(InterruptedError):
                pipeline.run(args, fault=fault)
            self.assertFalse((output / "point_update/journal.json").exists())
            prior = runtime.StageStore(output / "insert", resume=True)
            self.assertEqual(prior.doc["status"], "complete")
            preserved = [
                runtime.file_info(output / name) for name in (
                    "pipeline-manifest.json", "point_update/manifest.json", "insert/journal.json",
                    "initial-prefix/prefix-cache.json", "levels.json",
                )
            ]
            preserved += [info for stage in prior.doc["stages"] for info in stage["files"].values()]
            preserved.append(json.loads((output / "levels.json").read_text())["artifact"])
            # Retain the old root state as evidence while simulating a missing
            # state alongside already committed prefix/operation artifacts.
            (output / "pipeline.json").rename(output / "prior-pipeline-state.json")
            args.resume = True
            changed = copy.deepcopy(args)
            changed.level_seed += 1
            with self.assertRaisesRegex(RuntimeError, "immutable identity"):
                pipeline.run(changed)
            self.assertFalse((output / "pipeline.json").exists())
            with mock.patch.object(FakeIndex, "init_index", side_effect=AssertionError("silently rebuilt prefix")), mock.patch.object(
                    runner, "representative_levels", side_effect=AssertionError("silently rebuilt level plan")):
                self.assertEqual(pipeline.run(args), 0)
            state = json.loads((output / "pipeline.json").read_text())
            self.assertTrue(state["bootstrap_recovered"])
            self.assertEqual(state["status"], "scope_complete_no_paper")
            restored = runtime.StageStore(output / "point_update", resume=True)
            self.assertTrue(restored.doc["bootstrap_recovered"])
            self.assertEqual(restored.doc["status"], "complete")
            restored.verify_committed()
            for info in preserved:
                runtime.file_info(info["path"], info)
            self.assertFalse((output / "current.json").exists())

    def test_known_unfixed_native_blocked_before_load_add(self):
        args = argparse.Namespace(expected_sha256=aggregate.DELETE_NATIVE_SHA)
        with self.assertRaisesRegex(RuntimeError, "Superseded"):
            pipeline.native_canary(directory("known-old-native"), args)


class NativeSmallIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        native_dir, native_sha = os.environ.get("HASHANN_DYNAMIC_NATIVE_DIR"), os.environ.get("HASHANN_DYNAMIC_NATIVE_SHA")
        if not native_dir or not native_sha:
            raise unittest.SkipTest("set HASHANN_DYNAMIC_NATIVE_DIR/SHA to the distance-order1 native for tiny integration")
        protocol.require_current_query_native(native_sha)
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        cls.native, cls.native_info = runtime.import_native(native_dir, native_sha)

    def setUp(self):
        self.affinity = os.sched_getaffinity(0)
    def tearDown(self):
        os.sched_setaffinity(0, self.affinity)

    def test_capability_lifecycle_recall_and_known_boundary_sensitivity(self):
        output = directory("native-canary")
        report = canary.exercise(self.native, output)
        runtime.immutable_json(output / "report.json", {**report, "native": self.native_info})
        self.assertEqual(report["lifecycle"], dict.fromkeys(canary.LIFECYCLE_CHECKS, True))
        self.assertNotIn("query_exact", report)
        self.assertFalse(report["marker_fidelity_certified"])
        self.assertFalse(report["candidate_completeness_certified"])
        self.assertIn("max-bin", report["marker_coverage_limit"])
        self.assertIn("witness-to-owner", report["marker_coverage_limit"])
        self.assertEqual(report["after_replacement"], {"occupied": 99, "live": 96, "retired": 3})
        self.assertEqual(report["codebook_categorical_labels"], list(range(20)))
        runtime.validate_counters(report["deletion_work_counts"], 3)
        if report["deletion_scrub_wall_s"] is not None:
            self.assertGreaterEqual(report["deletion_scrub_wall_s"], 0)
        for stage, count in zip(report["stages"], (64, 96, 99)):
            self.assertEqual(stage["records"]["vector_mapping_sample_count"], count)
            self.assertTrue(stage["records"]["complete_categorical_bitsets_verified"])
            self.assertTrue(stage["save_load_results_identical"])
        self.assertEqual(report["index_format"], 10)
        self.assertEqual(report["numeric_marker_version"], 2)
        self.assertEqual(report["marker_owner_version"], 2)
        self.assertEqual(report["pickle_state_version"], 4)
        self.assertEqual(report["distance_order_version"], 1)
        self.assertEqual(report["candidate_order"], "vector-distance-only-v1")
        self.assertEqual(report["attr_sort_alpha"], 0)
        for stage in report["stages"]:
            self.assertEqual(stage["pickle_state_versions"], {
                "numeric_marker_version": 2, "marker_owner_version": 2, "distance_order_version": 1, "ser_version": 4})
            self.assertEqual(stage["attr_sort_alpha"], 0)
        for probe in report["diagnostics"].values():
            self.assertTrue(probe["eligible_set_unchanged"])
            self.assertTrue(probe["formal_query_restored"])

    def test_isolated_subprocess_canary_gate_accepts_measured_ANN_quality(self):
        args = argparse.Namespace(native_dir=Path(self.native_info["path"]).parent,
                                  expected_sha256=self.native_info["sha256"])
        info = pipeline.native_canary(directory("isolated-native-canary"), args)
        report = json.loads(Path(info["path"]).read_text())
        self.assertEqual(report["native"]["sha256"], self.native_info["sha256"])
        self.assertTrue(report["lifecycle"]["returned_squared_l2"])
        self.assertEqual(report["ann"]["recall"], report["ann"]["hits"] / report["ann"]["total"])

    def test_actual_operations_pause_and_reload_saved_native(self):
        for operation in protocol.OPERATIONS:
            with self.subTest(operation=operation), mock.patch.object(
                    runner.OperationRunner, "measure", correctness_measure):
                store, data, native, levels = operation_fixture(operation, self.native)
                worker = runner.OperationRunner(store, data, native, levels)
                self.assertFalse(worker.run(through_stage=1))
                del worker
                restored = runtime.StageStore(store.output, resume=True)
                restored.verify_committed()
                restored.recover_pending()
                worker = runner.OperationRunner(restored, data, native, levels)
                self.assertTrue(worker.run())
                self.assertEqual(restored.doc["stages"][-1]["state"]["occupied"], data.n)
                for stage in restored.doc["stages"][1:]:
                    mutation = stage["details"]["mutation"]
                    self.assertEqual(mutation["timing_breakdown"], runtime.mutation_timing(mutation))
                    if operation != "insert":
                        runtime.validate_counters(mutation["counters"], 12)
                        phase = mutation["phases"]["delete"]
                        if phase["scrub_wall_s"] is not None:
                            self.assertGreater(phase["scrub_wall_s"], 0)
                            self.assertLessEqual(phase["scrub_wall_s"], phase["wall_s"])
                restored.verify_committed()

    def test_actual_native_frozen_provenance_and_shared_three_operation_graphs(self):
        sources = pipeline.native_source_inventory(Path(self.native_info["path"]).parent, self.native_info)
        provenance_path = Path(self.native_info["path"]).parent / "provenance.json"
        if provenance_path.exists():
            auxiliary = json.loads(provenance_path.read_text()).get("auxiliary_source_sha256", {})
            expected = {"native/provenance.json", "native/hnswlib/hnswalg.h",
                        "native/python_bindings/bindings.cpp", "native/setup.py"}
            self.assertEqual(set(sources), expected | {f"native/{name}" for name in auxiliary})
            for name, digest in auxiliary.items():
                self.assertEqual(sources[f"native/{name}"]["sha256"], digest)
        with mock.patch.object(runner.OperationRunner, "measure", correctness_measure):
            inserted, data, native, levels = operation_fixture("insert", self.native, output_name="insert")
            worker = runner.OperationRunner(inserted, data, native, levels)
            self.assertTrue(worker.run())
            del worker
            config = {key: value for key, value in inserted.manifest["config"].items() if key != "operation"}
            origin = runner.completed_insert_checkpoint(
                inserted.output.parent, data, config, self.native_info, inserted.manifest["source"])
            prefix, full = inserted.manifest["initial_checkpoint"], origin["checkpoint"]
            for operation in ("delete", "point_update"):
                output = inserted.output.parent / operation
                output.mkdir()
                manifest = {
                    **inserted.manifest, "config": {**config, "operation": operation},
                    "initial_checkpoint": full if operation == "delete" else prefix,
                    "initial_full_provenance": origin if operation == "delete" else None,
                }
                store = runtime.StageStore(output, manifest=manifest)
                worker = runner.OperationRunner(store, data, native, levels)
                self.assertFalse(worker.run(through_stage=1))
                del worker
                restored = runtime.StageStore(output, manifest=manifest, resume=True)
                restored.verify_committed()
                restored.recover_pending()
                worker = runner.OperationRunner(restored, data, native, levels)
                self.assertTrue(worker.run())
                del worker
                self.assertEqual(restored.doc["stages"][-1]["state"], {"occupied": 120, "deleted": 60, "live": 60})
                restored.verify_committed()
                if operation == "delete":
                    for a, b in zip(inserted.doc["stages"][5]["calibrations"]["current"]["points"],
                                    restored.doc["stages"][0]["calibrations"]["initial"]["points"]):
                        self.assertTrue(all(a[key] == b[key] for key in
                                            ("ef", "recall", "labels_sha256", "distances_sha256")))
            runtime.file_info(prefix["path"], prefix)
            runtime.file_info(full["path"], full)
            self.assertFalse((inserted.output.parent / "initial-full").exists())


if __name__ == "__main__":
    unittest.main()
