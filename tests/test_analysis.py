"""Offline synthetic tests for analysis.py; no measurement code is exercised."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import load_artifacts, main, successful_inference


class AnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary_directory.name) / "artifacts"
        self.root.mkdir()

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def write_run(
        self,
        name: str,
        *,
        status: str = "completed",
        phase: str = "steady",
        kind: str = "inference",
        model: str = "model-a",
        batch_size=1,
        rep=0,
        active_attempt="attempt-0001",
        result: dict | None = None,
        host: str = "host-a",
        adapter: str = "adapter-a",
    ) -> Path:
        run_dir = self.root / name
        attempt_dir = run_dir / "attempt-0001"
        attempt_dir.mkdir(parents=True)
        state = {
            "run_id": f"run-{name}",
            "status": status,
            "trial": {
                "phase": phase,
                "kind": kind,
                "model": model,
                "batch_size": batch_size,
                "rep": rep,
            },
            "active_attempt": active_attempt,
        }
        (run_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
        if result is not None:
            result = dict(result)
            result.setdefault("kind", kind)
            result.setdefault("model", model)
            result.setdefault("batch_size", batch_size)
            result.setdefault("trial", dict(state["trial"]))
            result.setdefault("dataset", {"name": "tweeteval", "split": "test"})
            result.setdefault("host", {"id": host})
            result.setdefault("adapter", {"name": adapter})
            (attempt_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
        return run_dir

    @staticmethod
    def inference_result(**overrides) -> dict:
        result = {
            "counts": {
                "submitted": 10,
                "completed": 9,
                "valid": 8,
                "correct": 7,
                "retries": 1,
                "pass_count": 2,
                "panel_size": 5,
                "groups": 10,
                "retry_requests": 1,
                "http_request_count": 10,
                "qev_choice_count": 0,
                "final_group_width": 1,
            },
            "quality": {
                "macro_recall": 0.71,
                "macro_f1": 0.7,
                "accuracy": 0.72,
                "coverage": 0.8,
                "basis_pass": 0,
                "basis_n": 5,
                "repeat_disagreements_from_first": 1,
                "per_class": {
                    label: {"precision": value, "recall": value + 0.01, "f1": value + 0.02}
                    for label, value in (("negative", 0.6), ("neutral", 0.7), ("positive", 0.8))
                },
            },
            "performance": {
                "window_s": 12.0,
                "throughput_tweets_s": 0.75,
                "latency_p50_s": 0.9,
                "latency_p95_s": 2.1,
                "valid_throughput_tweets_s": 0.6,
                "group_latency_p50_s": 1.0,
                "group_latency_p95_s": 2.2,
                "latency_scopes": {"adapter request": 10},
            },
            "energy": {
                "energy_kwh": 0.00001,
                "cpu_energy_kwh": 0.000002,
                "gpu_energy_kwh": 0.000006,
                "ram_energy_kwh": 0.000001,
                "duration_s": 12.0,
                "run_id": "energy-run",
            },
            "derived": {
                "gross_j_per_tweet": 3.6,
                "device_idle_adjusted_j_per_tweet": 3.1,
                "loaded_idle_adjusted_j_per_tweet": 2.9,
            },
        }
        result.update(overrides)
        return result

    def test_loads_metrics_and_tidy_breakdowns_without_aggregation(self) -> None:
        self.write_run("inference-a", result=self.inference_result())

        tables = load_artifacts(self.root)

        self.assertEqual(len(tables.runs), 1)
        self.assertEqual(len(tables.trials), 1)
        trial = tables.trials.iloc[0]
        self.assertTrue(trial["analysis_eligible"])
        self.assertEqual(trial["quality_macro_recall"], 0.71)
        self.assertEqual(trial["count_submitted"], 10)
        self.assertEqual(trial["energy_kwh"], 0.00001)
        self.assertEqual(trial["gross_j_per_tweet"], 3.6)
        self.assertEqual(trial["throughput_tweets_s"], 0.75)
        self.assertEqual(trial["latency_p95_s"], 2.1)
        self.assertEqual(trial["count_pass_count"], 2)
        self.assertEqual(trial["count_retry_requests"], 1)
        self.assertEqual(trial["quality_basis_n"], 5)
        self.assertEqual(trial["quality_repeat_disagreements_from_first"], 1)
        self.assertEqual(trial["group_latency_p95_s"], 2.2)
        self.assertIn("adapter request", trial["performance_latency_scopes_json"])
        self.assertEqual(trial["energy_run_id"], "energy-run")

        components = tables.energy_components.set_index("component")["energy_kwh"]
        self.assertEqual(components["cpu"], 0.000002)
        self.assertEqual(components["gpu"], 0.000006)
        self.assertEqual(components["ram"], 0.000001)
        intensity = tables.energy_intensity.set_index("basis")["j_per_tweet"]
        self.assertEqual(intensity["gross"], 3.6)
        self.assertEqual(intensity["device_idle_adjusted"], 3.1)
        self.assertEqual(intensity["loaded_idle_adjusted"], 2.9)
        performance = tables.performance.set_index("metric")["value"]
        self.assertEqual(performance["window_s"], 12.0)
        self.assertEqual(performance["throughput_tweets_s"], 0.75)
        self.assertEqual(performance["valid_throughput_tweets_s"], 0.6)
        self.assertEqual(performance["group_latency_p95_s"], 2.2)
        self.assertEqual(len(tables.class_quality), 9)
        self.assertEqual(set(tables.class_quality["class_label"]), {"negative", "neutral", "positive"})

    def test_missing_records_and_fields_remain_null_and_noncomplete_is_visible(self) -> None:
        self.write_run("running", status="running", result=None)
        self.write_run("missing-fields", result={})

        tables = load_artifacts(self.root)

        self.assertEqual(len(tables.trials), 2)
        running = tables.trials.set_index("run_dir").loc[str(self.root / "running")]
        self.assertEqual(running["status"], "running")
        self.assertEqual(running["artifact_status"], "missing_result")
        self.assertFalse(running["analysis_eligible"])
        self.assertTrue(pd.isna(running["count_submitted"]))
        self.assertNotEqual(running["count_submitted"], 0)
        missing = tables.trials.set_index("run_id").loc["run-missing-fields"]
        self.assertTrue(missing["analysis_eligible"])
        self.assertTrue(pd.isna(missing["quality_macro_recall"]))
        self.assertTrue(pd.isna(missing["gpu_energy_kwh"]))
        self.assertTrue(pd.isna(missing["gross_j_per_tweet"]))
        self.assertNotEqual(missing["gpu_energy_kwh"], 0)
        non_complete_ids = set(
            tables.runs.loc[~tables.runs["is_complete"], "run_id"]
        )
        self.assertEqual(non_complete_ids, {"run-running"})
        missing_run_energy = tables.energy_components.loc[
            tables.energy_components["run_id"].eq("run-missing-fields"),
            "energy_kwh",
        ]
        self.assertEqual(len(missing_run_energy), 3)
        self.assertTrue(missing_run_energy.isna().all())

    def test_quality_selection_excludes_training_calibration_and_incomplete(self) -> None:
        quality = self.inference_result()["quality"]
        self.write_run("inference", result=self.inference_result())
        self.write_run("calibration", kind="calibration", result={"quality": quality})
        self.write_run("training", kind="training", phase="train", result={"quality": quality})
        self.write_run("running", kind="inference", status="running", result={"quality": quality})

        tables = load_artifacts(self.root)
        inference = successful_inference(tables.trials)

        self.assertEqual(list(inference["run_id"]), ["run-inference"])
        self.assertEqual(set(tables.class_quality["run_id"]), {"run-inference"})
        self.assertNotIn("run-calibration", set(tables.class_quality["run_id"]))
        self.assertNotIn("run-training", set(tables.class_quality["run_id"]))
        # Energy/performance tables still retain other kinds for separate facets.
        self.assertEqual(
            set(tables.energy_components["kind"]),
            {"inference", "calibration", "training"},
        )

    def test_condition_identity_preserves_incompatible_conditions(self) -> None:
        self.write_run("base", result={})
        self.write_run("phase", phase="warmup", result={})
        self.write_run("host", host="host-b", result={})
        self.write_run("adapter", adapter="adapter-b", result={})
        self.write_run("batch", batch_size=16, result={})
        self.write_run("rep", rep=1, result={})

        tables = load_artifacts(self.root)

        self.assertEqual(len(tables.trials), 6)
        self.assertEqual(tables.trials["condition_id"].nunique(), 6)
        self.assertEqual(tables.trials["trial_label"].nunique(), 6)
        self.assertEqual(len(successful_inference(tables.trials)), 6)

    def test_malformed_artifacts_are_reported_and_nested_states_are_ignored(self) -> None:
        malformed_state = self.root / "bad-state"
        malformed_state.mkdir()
        (malformed_state / "state.json").write_text("{not-json", encoding="utf-8")
        list_state = self.root / "list-state"
        list_state.mkdir()
        (list_state / "state.json").write_text("[]", encoding="utf-8")
        bad_result = self.write_run("bad-result", result={})
        (bad_result / "attempt-0001" / "result.json").write_text("not-json", encoding="utf-8")
        self.write_run("no-attempt", active_attempt=None, result={})
        nested = self.root / "nested" / "inner"
        nested.mkdir(parents=True)
        (nested / "state.json").write_text("{}", encoding="utf-8")

        tables = load_artifacts(self.root)

        self.assertEqual(len(tables.runs), 4)
        statuses = set(tables.runs["artifact_status"])
        self.assertEqual(
            statuses,
            {"state_read_error", "result_read_error", "invalid_active_attempt"},
        )
        self.assertTrue(tables.trials["analysis_eligible"].eq(False).all())
        self.assertTrue(tables.trials["result_error"].notna().all())

    def test_active_attempt_cannot_escape_run_directory(self) -> None:
        outside = self.root / "outside-result.json"
        outside.write_text("{}", encoding="utf-8")
        self.write_run("traversal", active_attempt="../outside-result.json", result={})

        tables = load_artifacts(self.root)

        trial = tables.trials.iloc[0]
        self.assertEqual(trial["artifact_status"], "invalid_active_attempt")
        self.assertFalse(trial["analysis_eligible"])
        self.assertTrue(pd.isna(trial["result_path"]))

    def test_empty_root_returns_typed_empty_tables(self) -> None:
        tables = load_artifacts(self.root)

        for frame in tables.as_dict().values():
            self.assertTrue(frame.empty)
        self.assertIn("quality_macro_recall", tables.trials.columns)
        self.assertIn("energy_kwh", tables.energy_components.columns)
        self.assertIn("j_per_tweet", tables.energy_intensity.columns)

    def test_cli_prints_noncomplete_runs_and_writes_six_csv_tables(self) -> None:
        self.write_run("complete", result=self.inference_result())
        self.write_run("running", status="running", result=None)
        output_dir = Path(self._temporary_directory.name) / "csv"
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            return_code = main([str(self.root), "--output-dir", str(output_dir)])

        self.assertEqual(return_code, 0)
        self.assertIn("Non-complete runs", stdout.getvalue())
        self.assertIn("run-running", stdout.getvalue())
        expected = {
            "runs.csv",
            "trials.csv",
            "class_quality.csv",
            "energy_components.csv",
            "energy_intensity.csv",
            "performance.csv",
        }
        self.assertEqual({path.name for path in output_dir.glob("*.csv")}, expected)
        trials = pd.read_csv(output_dir / "trials.csv")
        running = trials.loc[trials["run_id"].eq("run-running")].iloc[0]
        self.assertTrue(pd.isna(running["count_submitted"]))
        self.assertNotEqual(running["count_submitted"], 0)

    def test_cli_rejects_missing_root(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            return_code = main([str(self.root / "does-not-exist")])
        self.assertEqual(return_code, 2)
        self.assertIn("not a directory", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
