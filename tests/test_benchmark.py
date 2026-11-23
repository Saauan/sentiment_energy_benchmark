"""Offline synthetic orchestration tests; no adapters, GPUs, or real timers start."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import benchmark
import storage
from adapters.base import Unavailable


PANEL = [(0, "bad", 0), (1, "ordinary", 1), (2, "good", 2)]
TRAIN = [(0, "bad", 0), (1, "ordinary", 1), (2, "good", 2)]
VALIDATION = [(3, "warm", 1)]
WARMUP = [(3, "warm", 1)]


def dataset_info():
    return {
        "name": "synthetic", "adapter": "line_pairs", "revision": None,
        "source_uri": None, "source_type": "other", "files_sha256": {},
        "test_ids": [0, 1, 2], "label_order": ["negative", "neutral", "positive"],
    }


def config(adapter="svm", *, batches=(1,), training=True, passes=None):
    model = "model"
    per_width = {str(batch): (passes or 2) for batch in batches}
    if adapter == "qev":
        per_width = {"1": 2} if 1 in batches else {}
    return {
        "phase": "synthetic", "execution_protocol": "engine-session-v1", "dataset": {"adapter": "line_pairs", "path": "unused"},
        "models": {model: {"adapter": adapter, "device": "cuda" if adapter in {"qev", "vllm"} else "cpu", "params": {}}},
        "batch_sizes": list(batches), "seed": 1729, "sample_size": 3,
        "warmup_rows": 1, "repetitions": 1, "passes": {model: per_width},
        "energy": {"country_iso_code": "FRA", "gpu_ids": [0], "sample_secs": 1, "min_samples": 5},
        "device_idle_s": 8, "loaded_idle_s": 12,
        "loaded_idle_adapters": [model], "measure_training": training, "settle_s": 0,
    }


class FakeAdapter:
    def __init__(self, *, setup_error=None, predict_error=None):
        self.setup_error = setup_error
        self.predict_error = predict_error
        self.setup_calls = 0
        self.predict_calls = 0
        self.closed = False

    def setup(self, train, validation):
        self.setup_calls += 1
        if self.setup_error:
            raise self.setup_error
        return {"adapter": "fake", "train_rows": len(train), "validation_rows": len(validation)}

    def predict(self, texts):
        self.predict_calls += 1
        if self.predict_error:
            raise self.predict_error
        labels = {"bad": "negative", "good": "positive", "ordinary": "neutral", "warm": "neutral"}
        return [{"label": labels.get(text, "neutral"), "raw": {"text": text}, "latency_s": 0.001}
                for text in texts]

    def set_submission_width(self, width):
        self.width = width
        return {"submission_width": width, "server_restarted": False}

    def close(self):
        self.closed = True


class FakeMeter:
    calls = []

    def __init__(self, out, task, cfg):
        self.out = Path(out)
        self.out.mkdir()
        self.task = task
        self.native = None

    def capture(self, work):
        result = work()
        number = len(type(self).calls) + 1
        window = 6.0
        self.native = {
            "run_id": f"native-{number}", "energy_kwh": 0.001 * number,
            "duration_s": window, "schema_pin": "synthetic",
            "source_units": {"energy": "kWh", "duration": "s"},
            "count": self.task["nbRequest"], "cpu_energy_kwh": 0.0005 * number,
            "gpu_energy_kwh": None, "ram_energy_kwh": 0.0005 * number,
            "source_signature": "same-sources", "sources": [],
            "native_report": "synthetic.json", "scheduled_power_samples": 6,
        }
        storage.atomic_json(self.out / "native.json", {"task": self.task, "native": self.native})
        type(self).calls.append(self)
        return result, window, self.native


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "artifacts"
        self.host = {"host_id": "synthetic-host", "gpu": [], "gpu_compute_apps": None}
        FakeMeter.calls = []
        self.adapters = []
        mock.patch.object(benchmark, "sleep", lambda _seconds: None).start()
        mock.patch.object(benchmark.hardware, "snapshot", lambda: dict(self.host)).start()
        mock.patch.object(benchmark.hardware, "preflight", lambda spec, host: ({"device": spec["device"]}, None)).start()
        mock.patch.object(benchmark.hardware, "looks_like_oom", lambda exc, attempt: False).start()
        mock.patch.object(benchmark.energy, "Meter", FakeMeter).start()

        def validate(out, expected):
            saved = storage.load(Path(out) / "native.json")
            if saved["task"] != expected:
                raise ValueError("native task mismatch")
            return saved["native"]

        mock.patch.object(benchmark.energy, "validate_run", side_effect=validate).start()

    def tearDown(self):
        mock.patch.stopall()
        self.temporary.cleanup()

    def data(self, cfg):
        return cfg, dataset_info(), TRAIN, VALIDATION, PANEL, WARMUP

    def plan(self, cfg):
        return benchmark.ensure_plan(self.root, cfg, dataset_info(), len(TRAIN))

    def use_adapter(self, adapter):
        self.adapters.append(adapter)
        return adapter

    def run_all(self, cfg, plan):
        selected = benchmark.select_trials(plan, self.root)
        return benchmark.run_selected(self.root, plan, selected, self.data(cfg))

    def test_plan_is_deterministic_and_training_native_count_is_zero(self):
        cfg = config()
        first = benchmark.build_plan(cfg, dataset_info(), len(TRAIN))
        second = benchmark.build_plan(cfg, dataset_info(), len(TRAIN))
        self.assertEqual(first, second)
        self.assertEqual([trial["kind"] for trial in first["trials"]],
                         ["device_idle", "training", "inference"])
        training = first["trials"][1]
        self.assertEqual(training["expected_count"], len(TRAIN))
        self.assertEqual(training["native_count"], 0)
        task = benchmark._native_task(training, cfg, dataset_info(), len(TRAIN))
        self.assertEqual(task["taskStage"], "training")
        self.assertEqual(task["nbRequest"], 0)
        self.assertEqual(task["dataset"][0]["dataQuantity"], len(TRAIN))
        inference = first["trials"][2]
        self.assertEqual(inference["expected_count"], 6)
        self.assertEqual(inference["native_count"], 6)
        self.assertEqual(inference["device_idle_run_id"], first["trials"][0]["run_id"])
        self.assertEqual(inference["training_run_id"], training["run_id"])

    def test_unsupported_qev_width_has_sentinel_without_pass_entry(self):
        cfg = config("qev", batches=(1, 16), training=False)
        plan = benchmark.build_plan(cfg, dataset_info(), len(TRAIN))
        inference = [t for t in plan["trials"] if t["kind"] == "inference"]
        self.assertEqual(len(inference), 2)
        unsupported = next(t for t in inference if t["batch_size"] == 16)
        self.assertEqual(unsupported["passes"], 0)
        self.assertEqual(unsupported["expected_count"], 0)
        self.assertEqual(unsupported["native_count"], 0)

    def test_packed_qev_counts_envelopes_and_checks_context_outside_inference(self):
        cfg = config("qev", batches=(4,), training=False)
        cfg["models"]["model"]["params"]["request_mode"] = "packed"
        cfg["sample_size"] = 5
        cfg["passes"]["model"] = {"4": 2}
        panel = PANEL + [(3, "good", 2), (4, "bad", 0)]
        info = {**dataset_info(), "test_ids": [row[0] for row in panel]}
        data = (cfg, info, TRAIN, VALIDATION, panel, WARMUP)
        plan = benchmark.ensure_plan(self.root, cfg, info, len(TRAIN))
        adapter = self.use_adapter(FakeAdapter())
        adapter.check_context = mock.Mock(side_effect=lambda texts: {
            "width": len(texts), "fit": True, "model_forwarding": False,
        })

        def pack(texts):
            rows = FakeAdapter.predict(adapter, texts)
            width = len(rows)
            ids = ["sentiment"] if width == 1 else [f"sentiment_{i + 1}" for i in range(width)]
            for i, (row, qid) in enumerate(zip(rows, ids, strict=True)):
                row["raw"].update(request_width=width, target_number=i + 1, question_id=qid)
            rows[0]["raw"]["request"] = {"width": width, "question_ids": ids}
            return rows

        adapter.predict = pack
        with mock.patch.object(benchmark, "_create_adapter", return_value=adapter):
            benchmark.run_selected(self.root, plan, benchmark.select_trials(plan, self.root), data)
        inference = plan["trials"][-1]
        attempt = self.root / inference["run_id"] / "attempt-0001"
        result = storage.load(attempt / "result.json")
        self.assertEqual(inference["expected_count"], 10)
        self.assertEqual(result["native_task"]["nbRequest"], 4)
        self.assertEqual(result["counts"]["qev_request_count"], 4)
        self.assertEqual(result["counts"]["qev_choice_count"], 10)
        self.assertEqual(result["counts"]["final_group_width"], 1)
        self.assertEqual(len(storage.load(attempt / "qev_context.json")), 3)
        self.assertEqual(adapter.check_context.call_count, 3)
        report = benchmark.verify_plan(self.root, plan, data)
        self.assertTrue(report["ok"], report["inconsistencies"])

    def test_packed_qev_context_rejection_never_calls_predict(self):
        cfg = config("qev", batches=(4,), training=False)
        cfg["models"]["model"]["params"]["request_mode"] = "packed"
        cfg["passes"]["model"] = {"4": 1}
        plan = self.plan(cfg)
        adapter = self.use_adapter(FakeAdapter())
        rejection = Unavailable("packed context exceeds 16384 tokens")
        rejection.context_receipt = {"fit": False, "max_branch_tokens": 16385}
        adapter.check_context = mock.Mock(side_effect=rejection)
        with mock.patch.object(benchmark, "_create_adapter", return_value=adapter):
            self.run_all(cfg, plan)
        inference = plan["trials"][-1]
        self.assertEqual(benchmark.status_report(self.root, plan)["counts"]["skipped"], 2)
        self.assertEqual(adapter.predict_calls, 0)
        receipt = storage.load(self.root / inference["run_id"] / "attempt-0001" / "qev_context.json")
        self.assertEqual(receipt[0]["detail"], rejection.context_receipt)

    def test_complete_synthetic_run_persists_audits_and_revalidates(self):
        cfg = config()
        plan = self.plan(cfg)
        adapter = self.use_adapter(FakeAdapter())
        with mock.patch.object(benchmark, "_create_adapter", return_value=adapter):
            summary = self.run_all(cfg, plan)
        self.assertFalse(summary["interrupted"])
        self.assertTrue(adapter.closed)
        statuses = benchmark.status_report(self.root, plan)["counts"]
        self.assertEqual(statuses["complete"], 3)
        self.assertEqual(statuses["pending"], 0)
        inference = plan["trials"][-1]
        result = storage.load(self.root / inference["run_id"] / "attempt-0001" / "result.json")
        self.assertEqual(result["counts"]["submitted"], 6)
        self.assertEqual(result["counts"]["retries"], 0)
        self.assertEqual(result["quality"]["basis_n"], 3)
        self.assertEqual(result["energy"]["scheduled_power_samples"], 6)
        self.assertTrue(result["gate"]["meets_minimum_window"])
        self.assertEqual(result["predictions"]["rows"], 6)
        self.assertIsNotNone(result["derived"]["device_idle_adjusted_j_per_tweet"])
        training = storage.load(self.root / plan["trials"][1]["run_id"] / "attempt-0001" / "result.json")
        self.assertEqual(training["energy"]["count"], 0)
        self.assertEqual(training["counts"]["submitted"], len(TRAIN))
        report = benchmark.verify_plan(self.root, plan, self.data(cfg))
        self.assertTrue(report["ok"], report["inconsistencies"])
        self.assertEqual(report["valid_complete"], 3)
        self.assertEqual(report["low_sampling"], [])
        # Completed trials are immutable and are not selected again.
        self.assertEqual(benchmark.select_trials(plan, self.root), [])
        self.assertEqual(adapter.setup_calls, 1)

    def test_verify_detects_prediction_tampering_without_rewriting(self):
        cfg = config(training=False)
        plan = self.plan(cfg)
        with mock.patch.object(benchmark, "_create_adapter", return_value=FakeAdapter()):
            self.run_all(cfg, plan)
        inference = plan["trials"][-1]
        predictions = self.root / inference["run_id"] / "attempt-0001" / "predictions.jsonl"
        before = predictions.read_bytes()
        predictions.write_bytes(before + b"{}\n")
        report = benchmark.verify_plan(self.root, plan, self.data(cfg))
        self.assertFalse(report["ok"])
        self.assertTrue(any("prediction digest mismatch" in item for item in report["inconsistencies"]))
        self.assertEqual(predictions.read_bytes(), before + b"{}\n")

    def test_interrupted_inference_retries_fresh_but_keeps_completed_calibration(self):
        cfg = config(training=False)
        plan = self.plan(cfg)
        interrupted = FakeAdapter(predict_error=benchmark.SignalReceived(15))
        with mock.patch.object(benchmark, "_create_adapter", return_value=interrupted):
            summary = self.run_all(cfg, plan)
        self.assertTrue(summary["interrupted"])
        self.assertTrue(interrupted.closed)
        idle, inference = plan["trials"]
        self.assertEqual(benchmark._state_status(self.root, idle), "complete")
        self.assertEqual(benchmark._state_status(self.root, inference), "incomplete")
        retry = FakeAdapter()
        with mock.patch.object(benchmark, "_create_adapter", return_value=retry):
            self.run_all(cfg, plan)
        self.assertEqual(benchmark._state_status(self.root, inference), "complete")
        state = storage.load(self.root / inference["run_id"] / "state.json")
        self.assertEqual(state["active_attempt"], "attempt-0002")
        self.assertEqual([entry["status"] for entry in state["history"]], ["incomplete", "complete"])
        self.assertFalse((self.root / idle["run_id"] / "attempt-0002").exists())

    def test_unavailable_is_not_automatically_retried_and_explicit_retry_is_fresh(self):
        cfg = config(training=False)
        plan = self.plan(cfg)
        unavailable = FakeAdapter(setup_error=Unavailable("missing runtime"))
        with mock.patch.object(benchmark, "_create_adapter", return_value=unavailable):
            self.run_all(cfg, plan)
        inference = plan["trials"][-1]
        self.assertEqual(benchmark._state_status(self.root, inference), "skipped")
        self.assertTrue(unavailable.closed)
        self.assertEqual(benchmark.select_trials(plan, self.root), [])
        with mock.patch.object(benchmark, "_create_adapter", return_value=FakeAdapter()):
            selected = benchmark.select_trials(plan, self.root, retry_skipped=True)
            self.assertEqual([t["run_id"] for t in selected], [inference["run_id"]])
            benchmark.run_selected(self.root, plan, selected, self.data(cfg))
        state = storage.load(self.root / inference["run_id"] / "state.json")
        self.assertEqual(state["active_attempt"], "attempt-0002")
        self.assertEqual([entry["status"] for entry in state["history"]], ["skipped", "complete"])

    def test_oom_is_skipped_only_after_clean_close(self):
        cfg = config(training=False)
        plan = self.plan(cfg)
        adapter = FakeAdapter(predict_error=RuntimeError("CUDA out of memory"))
        with mock.patch.object(benchmark, "_create_adapter", return_value=adapter), \
                mock.patch.object(benchmark.hardware, "looks_like_oom", return_value=True):
            self.run_all(cfg, plan)
        self.assertTrue(adapter.closed)
        self.assertEqual(benchmark._state_status(self.root, plan["trials"][-1]), "skipped")

    def test_unsupported_condition_skips_before_any_idle_or_meter(self):
        cfg = config("qev", batches=(16,), training=False)
        plan = self.plan(cfg)
        with mock.patch.object(benchmark, "_create_adapter") as create:
            self.run_all(cfg, plan)
        create.assert_not_called()
        self.assertEqual(FakeMeter.calls, [])
        statuses = benchmark.status_report(self.root, plan)
        self.assertEqual(statuses["counts"]["skipped"], 3)
        self.assertEqual(statuses["counts"]["complete"], 0)
        first = self.root / plan["trials"][0]["run_id"] / "attempt-0001" / "preflight.json"
        self.assertTrue(first.is_file())
        self.assertIn("preflight", storage.load(first))

    def test_calibration_only_does_not_preflight_model(self):
        cfg = config("qev", batches=(16,), training=False)
        plan = self.plan(cfg)
        idle = plan["trials"][0]
        with mock.patch.object(benchmark.hardware, "preflight") as preflight:
            benchmark.run_selected(self.root, plan, [idle], self.data(cfg))
        preflight.assert_not_called()
        self.assertEqual(benchmark._state_status(self.root, idle), "complete")
        self.assertEqual(len(FakeMeter.calls), 1)

    def test_sample_counter_and_host_matching_drive_flags_and_adjustment(self):
        cfg = config()
        trial = {"kind": "inference"}
        native = {"duration_s": 1.0, "scheduled_power_samples": 5}
        gate = benchmark._gate(trial, cfg, native)
        self.assertTrue(gate["meets_minimum_window"])
        self.assertFalse(gate["duration_meets_minimum"])
        self.assertTrue(gate["samples_meet_minimum"])
        native["scheduled_power_samples"] = 4
        self.assertTrue(benchmark._gate(trial, cfg, native)["low_sampling"])
        gross = {"source_signature": "same", "energy_kwh": 0.002, "duration_s": 2.0}
        baseline_energy = {"source_signature": "same", "energy_kwh": 0.001, "duration_s": 1.0}
        baseline = {"energy": baseline_energy, "host": self.host}
        self.assertIsNone(benchmark._adjusted(gross, baseline, {**self.host, "host_id": "other"}, 2))
        other_gpu = {"host_id": self.host["host_id"], "gpu": [{"uuid": "GPU-other"}]}
        self.assertIsNone(benchmark._adjusted(gross, baseline, other_gpu, 2))
        self.assertIsNotNone(benchmark._adjusted(gross, baseline, self.host, 2))

    def test_limit_and_kind_selection_include_dependencies_in_plan_order(self):
        cfg = config()
        plan = self.plan(cfg)
        selected = benchmark.select_trials(plan, self.root, kind="inference")
        self.assertEqual([t["kind"] for t in selected], ["device_idle", "training", "inference"])
        limited = benchmark.select_trials(plan, self.root, kind="inference", limit_new=2)
        self.assertEqual([t["kind"] for t in limited], ["device_idle", "training"])

    def test_sessions_share_one_engine_across_widths_and_repetitions(self):
        cfg = config("vllm", batches=(1, 4, 16), training=False, passes=1)
        cfg["repetitions"] = 2
        plan = self.plan(cfg)
        self.assertEqual(plan["format_version"], 2)
        self.assertEqual(plan, benchmark.build_plan(cfg, dataset_info(), len(TRAIN)))
        self.assertEqual([s["rep"] for s in plan["sessions"]], [0, 1])
        adapters, specs = [], []
        def create(spec, path):
            specs.append(spec)
            adapter = FakeAdapter()
            adapter.widths = []
            adapter.set_submission_width = lambda width: (adapter.widths.append(width) or {"submission_width": width, "server_restarted": False})
            adapters.append(adapter)
            return adapter
        with mock.patch.object(benchmark, "_create_adapter", side_effect=create):
            self.run_all(cfg, plan)
        self.assertEqual(len(adapters), 2)
        self.assertEqual([a.setup_calls for a in adapters], [1, 1])
        self.assertTrue(all(a.closed for a in adapters))
        self.assertEqual([a.widths for a in adapters], [s["widths"] for s in plan["sessions"]])
        self.assertEqual([s["submission_width"] for s in specs], [16, 16])
        for session in plan["sessions"]:
            ts = [t for t in plan["trials"] if t["session_id"] == session["session_id"]]
            idle = next(t for t in ts if t["kind"] == "device_idle")
            self.assertEqual(idle["batch_size"], 0)
            self.assertTrue(all(t["device_idle_run_id"] == idle["run_id"] for t in ts if t is not idle))
            self.assertEqual(len([t for t in ts if t["kind"] == "inference"]), 3)
        report = benchmark.verify_plan(self.root, plan, self.data(cfg))
        self.assertTrue(report["ok"], report["inconsistencies"])

    def test_seeded_order_is_deterministic_and_rep_major(self):
        cfg = config("vllm", batches=(1, 4, 16, 64, 256), training=False, passes=1)
        cfg["models"]["another"] = dict(cfg["models"]["model"])
        cfg["passes"]["another"] = dict(cfg["passes"]["model"])
        cfg["repetitions"] = 3
        plan = benchmark.build_plan(cfg, dataset_info(), len(TRAIN))
        self.assertEqual(plan, benchmark.build_plan(cfg, dataset_info(), len(TRAIN)))
        self.assertEqual([s["rep"] for s in plan["sessions"]], [0, 0, 1, 1, 2, 2])
        self.assertTrue(all(set(s["widths"]) == set(cfg["batch_sizes"]) for s in plan["sessions"]))
        other = {**cfg, "seed": cfg["seed"] + 1}
        other_plan = benchmark.build_plan(other, dataset_info(), len(TRAIN))
        self.assertNotEqual([s["widths"] for s in plan["sessions"]], [s["widths"] for s in other_plan["sessions"]])

    def test_training_once_for_all_widths_and_in_each_repetition(self):
        cfg = config("svm", batches=(1, 4, 16), training=True, passes=1)
        cfg["repetitions"] = 2
        plan = self.plan(cfg)
        adapters = []
        def create(spec, path):
            adapters.append(FakeAdapter())
            return adapters[-1]
        with mock.patch.object(benchmark, "_create_adapter", side_effect=create):
            self.run_all(cfg, plan)
        self.assertEqual(len(adapters), 2)
        self.assertEqual([a.setup_calls for a in adapters], [1, 1])
        self.assertEqual(sum(t["kind"] == "training" for t in plan["trials"]), 2)
        report = benchmark.verify_plan(self.root, plan, self.data(cfg))
        self.assertTrue(report["ok"], report["inconsistencies"])

    def test_partial_loaded_baseline_resumes_once_and_preserves_bytes(self):
        cfg = config("vllm", batches=(1, 4, 16), training=False, passes=1)
        plan = self.plan(cfg)
        selected = benchmark.select_trials(plan, self.root, limit_new=2)
        self.assertEqual([t["kind"] for t in selected], ["device_idle", "loaded_idle"])
        first = FakeAdapter()
        with mock.patch.object(benchmark, "_create_adapter", return_value=first):
            benchmark.run_selected(self.root, plan, selected, self.data(cfg))
        saved = {t["run_id"]: (self.root / t["run_id"] / "attempt-0001" / "result.json").read_bytes() for t in selected}
        second = FakeAdapter()
        with mock.patch.object(benchmark, "_create_adapter", return_value=second) as create:
            self.run_all(cfg, plan)
        self.assertEqual(create.call_count, 1)
        self.assertEqual(second.setup_calls, 1)
        for t in selected:
            self.assertEqual(saved[t["run_id"]], (self.root / t["run_id"] / "attempt-0001" / "result.json").read_bytes())
            self.assertFalse((self.root / t["run_id"] / "attempt-0002").exists())
        inf = next(t for t in plan["trials"] if t["kind"] == "inference" and t["batch_size"] == selected[1]["batch_size"])
        result = storage.load(self.root / inf["run_id"] / "attempt-0001" / "result.json")
        self.assertEqual(result["adapter"]["engine_session"]["segment_id"], "segment-0002")
        self.assertEqual(result["adapter"]["loaded_idle_engine_session"]["segment_id"], "segment-0001")
        for t in plan["trials"]:
            if t["kind"] == "inference":
                result = storage.load(self.root / t["run_id"] / "attempt-0001" / "result.json")
                self.assertEqual(result["adapter"]["engine_session"]["segment_id"], "segment-0002")
        report = benchmark.verify_plan(self.root, plan, self.data(cfg))
        self.assertTrue(report["ok"], report["inconsistencies"])

    def test_historical_plans_cannot_execute_even_directly(self):
        cfg = config()
        del cfg["execution_protocol"]
        plan = self.plan(cfg)
        self.assertEqual(plan["format_version"], 1)
        with mock.patch.object(benchmark, "_create_adapter") as create:
            with self.assertRaisesRegex(ValueError, "read/verify only"):
                self.run_all(cfg, plan)
            create.assert_not_called()
        self.assertTrue(benchmark.verify_plan(self.root, plan, self.data(cfg))["ok"])

    def test_qev_context_rejection_is_width_local_and_engine_reused(self):
        cfg = config("qev", batches=(1, 4), training=False, passes=1)
        cfg["models"]["model"]["params"]["request_mode"] = "packed"
        cfg["passes"]["model"] = {"1": 1, "4": 1}
        # Arrange the rejected width first, proving the healthy engine continues.
        while True:
            plan = benchmark.build_plan(cfg, dataset_info(), len(TRAIN))
            if plan["sessions"][0]["widths"][0] == 4:
                break
            cfg["seed"] += 1
        plan = self.plan(cfg)
        adapter = FakeAdapter()
        def context(texts):
            if len(texts) > 1:
                exc = Unavailable("too long")
                exc.context_receipt = {"fit": False}
                raise exc
            return {"width": len(texts), "fit": True, "model_forwarding": False}
        adapter.check_context = context
        original_predict = adapter.predict
        def pack(texts):
            rows = original_predict(texts)
            ids = ["sentiment"] if len(rows) == 1 else [f"sentiment_{i+1}" for i in range(len(rows))]
            for i, (row, qid) in enumerate(zip(rows, ids, strict=True)):
                row["raw"].update(request_width=len(rows), target_number=i + 1, question_id=qid)
            rows[0]["raw"]["request"] = {"width": len(rows), "question_ids": ids}
            return rows
        adapter.predict = pack
        with mock.patch.object(benchmark, "_create_adapter", return_value=adapter) as create:
            self.run_all(cfg, plan)
        self.assertEqual(create.call_count, 1)
        self.assertEqual(adapter.setup_calls, 1)
        self.assertTrue(adapter.closed)
        for t in plan["trials"]:
            if t["batch_size"] == 4:
                self.assertEqual(benchmark._state_status(self.root, t), "skipped")
            elif t["batch_size"] == 1:
                self.assertEqual(benchmark._state_status(self.root, t), "complete")
        report = benchmark.verify_plan(self.root, plan, self.data(cfg))
        self.assertTrue(report["ok"], report["inconsistencies"])

    def test_engine_failure_stops_remaining_widths(self):
        cfg = config("vllm", batches=(1, 4, 16), training=False, passes=1)
        plan = self.plan(cfg)
        adapter = FakeAdapter(predict_error=RuntimeError("engine disconnected"))
        with mock.patch.object(benchmark, "_create_adapter", return_value=adapter) as create:
            self.run_all(cfg, plan)
        self.assertEqual(create.call_count, 1)
        self.assertTrue(adapter.closed)
        self.assertEqual(adapter.setup_calls, 1)
        remaining = [t for t in plan["trials"] if t["kind"] == "inference"]
        self.assertTrue(all(benchmark._state_status(self.root, t) in {"failed", "skipped"} for t in remaining))
        self.assertEqual(sum(benchmark._state_status(self.root, t) == "failed" for t in remaining), 1)

    def test_gpu_process_filter_accepts_only_owned_process_group(self):
        self.host["gpu_compute_apps"] = "300, owned child, 20 MiB"
        adapter = FakeAdapter()
        adapter.compute_process_id = 200
        with mock.patch.object(benchmark.os, "getpgid", return_value=200):
            self.assertEqual(benchmark._owned_snapshot(adapter)["gpu_compute_apps"], self.host["gpu_compute_apps"])
        with mock.patch.object(benchmark.os, "getpgid", return_value=201):
            with self.assertRaisesRegex(Unavailable, "foreign GPU"):
                benchmark._owned_snapshot(adapter)
        self.host.update(gpu=[{"uuid": "GPU-1"}], gpu_compute_apps=None)
        with self.assertRaisesRegex(Unavailable, "query failed"):
            benchmark._owned_snapshot(adapter)

    def test_post_window_competing_compute_prevents_publication(self):
        cfg = config(training=False)
        plan = self.plan(cfg)
        capture = FakeMeter.capture
        def contaminated(meter, work):
            result = capture(meter, work)
            if meter.task["taskStage"] == "inference":
                self.host["gpu_compute_apps"] = "99999999, foreign, 100 MiB"
            return result
        with mock.patch.object(FakeMeter, "capture", contaminated), mock.patch.object(benchmark, "_create_adapter", return_value=FakeAdapter()):
            self.run_all(cfg, plan)
        inference = plan["trials"][-1]
        self.assertEqual(benchmark._state_status(self.root, inference), "skipped")
        self.assertFalse((self.root / inference["run_id"] / "attempt-0001" / "result.json").exists())

    def test_interpreter_paths_are_lexical_not_symlink_resolved(self):
        spec = {
            "adapter": "vllm", "python": ".scratch/venv/bin/python",
            "engine_safety": {"expected_python": ".scratch/venv/bin/python"},
            "params": {"python": ".scratch/qev/bin/python", "onejev_path": ".scratch/OneJev"},
        }
        resolved = benchmark._runtime_spec(spec)
        expected = str(benchmark.ROOT / ".scratch/venv/bin/python")
        self.assertEqual(resolved["python"], expected)
        self.assertEqual(resolved["engine_safety"]["expected_python"], expected)
        self.assertEqual(resolved["params"]["python"], str(benchmark.ROOT / ".scratch/qev/bin/python"))
        self.assertEqual(spec["python"], ".scratch/venv/bin/python")


if __name__ == "__main__":
    unittest.main()
