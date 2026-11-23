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
        "phase": "synthetic", "dataset": {"adapter": "line_pairs", "path": "unused"},
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
