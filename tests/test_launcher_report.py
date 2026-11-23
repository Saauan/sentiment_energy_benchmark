"""Compact offline tests for launcher_report.py; no measurement is exercised."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from launcher_report import generate_report, main


class LauncherReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary_directory.name) / "run"
        self.root.mkdir()

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def write_launcher(self, *, mode: str = "full", selected=None, excluded=None, sha="config-a") -> None:
        (self.root / "launcher.json").write_text(
            json.dumps({
                "mode": mode,
                "discovery": {
                    "selected": [] if selected is None else selected,
                    "excluded": {} if excluded is None else excluded,
                },
                "config_sha256": sha,
            }),
            encoding="utf-8",
        )

    def write_plan(self, trials: list[dict], *, sha: str = "config-a", config=None,
                   repetitions: int | None = None) -> None:
        payload = {
            "format_version": 2,
            "execution_protocol": "engine-session-v1",
            "phase": "full",
            "config_sha256": sha,
            "trials": trials,
        }
        prepared = {} if config is None else dict(config)
        if repetitions is not None:
            prepared["repetitions"] = repetitions
        if prepared:
            payload["config"] = prepared
        (self.root / "plan.json").write_text(json.dumps(payload), encoding="utf-8")

    def write_verification(self, *, ok=True, valid_complete=0, runs=None, inconsistencies=None,
                           low_sampling=None) -> None:
        prepared = []
        for entry in [] if runs is None else runs:
            entry = dict(entry)
            entry.setdefault("active_attempt", "attempt-0001")
            prepared.append(entry)
        (self.root / "verification.json").write_text(
            json.dumps({
                "ok": ok,
                "valid_complete": valid_complete,
                "runs": prepared,
                "inconsistencies": [] if inconsistencies is None else inconsistencies,
                "low_sampling": [] if low_sampling is None else low_sampling,
            }),
            encoding="utf-8",
        )

    @staticmethod
    def trial(
        run_id: str,
        *,
        model: str = "model-a",
        width: int = 1,
        kind: str = "inference",
        expected: int = 10,
        adapter: str = "test",
        passes: int = 1,
        rep: int = 0,
    ) -> dict:
        return {
            "run_id": run_id,
            "phase": "full",
            "kind": kind,
            "model": model,
            "batch_size": width,
            "rep": rep,
            "adapter": adapter,
            "passes": passes if kind == "inference" else 0,
            "expected_count": expected,
            "native_count": expected,
            "panel_size": expected if kind == "inference" else 0,
        }

    @staticmethod
    def result(
        trial: dict,
        *,
        submitted: int = 10,
        valid_samples: int = 10,
        gross: float | None = 2.0,
        recall: float | None = 0.7,
        f1: float | None = 0.6,
        accuracy: float | None = 0.8,
        coverage: float | None = 1.0,
        valid: bool = True,
        performance: dict | None = None,
        gate: dict | None = None,
        device_adjusted: float | None = None,
        loaded_adjusted: float | None = None,
    ) -> dict:
        result = {
            "valid": valid,
            "trial": trial,
            "kind": trial["kind"],
            "model": trial["model"],
            "batch_size": trial["batch_size"],
            "counts": {
                "submitted": submitted,
                "completed": submitted,
                "valid": valid_samples,
                "correct": max(0, valid_samples - 1),
                "pass_count": 1,
                "panel_size": trial.get("panel_size", submitted),
            },
            "quality": {
                "macro_recall": recall,
                "macro_f1": f1,
                "accuracy": accuracy,
                "coverage": coverage,
                "basis_pass": 0,
                "basis_n": valid_samples,
                "repeat_disagreements_from_first": 0,
            },
            "performance": {"window_s": 1.0} if performance is None else performance,
            "energy": {"energy_kwh": 0.0001, "run_id": "energy-a"},
            "derived": {
                "gross_j_per_tweet": gross,
                "device_idle_adjusted_j_per_tweet": device_adjusted,
                "loaded_idle_adjusted_j_per_tweet": loaded_adjusted,
            },
        }
        if gate is not None:
            result["gate"] = gate
        return result

    def write_run(
        self,
        trial: dict,
        *,
        status: str = "complete",
        result: dict | None = None,
        reason: str | None = None,
        write_state: bool = True,
        malformed_state: bool = False,
    ) -> None:
        run_dir = self.root / trial["run_id"]
        if write_state:
            run_dir.mkdir(parents=True, exist_ok=True)
            if malformed_state:
                (run_dir / "state.json").write_text("{not-json", encoding="utf-8")
                return
            (run_dir / "state.json").write_text(
                json.dumps({
                    "run_id": trial["run_id"],
                    "trial": trial,
                    "status": status,
                    "active_attempt": "attempt-0001",
                    "reason": reason,
                    "history": [],
                }),
                encoding="utf-8",
            )
        attempt = run_dir / "attempt-0001"
        attempt.mkdir(parents=True, exist_ok=True)
        if result is not None:
            (attempt / "result.json").write_text(json.dumps(result), encoding="utf-8")

    def summary(self) -> str:
        return (self.root / "report" / "summary.md").read_text(encoding="utf-8")

    def condition_rows(self) -> list[dict]:
        path = self.root / "report" / "inference_medians.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def condition_row(self, model: str, width: int) -> dict:
        matches = [row for row in self.condition_rows()
                   if row["model"] == model and int(float(row["width"])) == width]
        self.assertEqual(len(matches), 1, f"expected one row for {model} width {width}: {matches}")
        return matches[0]

    @staticmethod
    def table_row(text: str, system: str, width: str) -> list[str] | None:
        section = text.split("## Results (inference conditions)", 1)
        if len(section) != 2:
            return None
        body = section[1].split("\n## Outputs", 1)[0]
        for line in body.splitlines():
            if line.startswith("| ") and f"| {system} | {width} |" in line:
                return [cell.strip() for cell in line.strip().strip("|").split(" | ")]
        return None

    def test_full_results_table_combines_verified_medians_and_counts(self) -> None:
        trials = [
            self.trial("r1", width=1),
            self.trial("r2", width=1),
            self.trial("r3", width=1),
            self.trial("r4", model="model-b", width=16, expected=5),
            self.trial("r5", width=1, kind="training", expected=0),
            self.trial("r6", model="model-c", expected=5),
        ]
        self.write_plan(trials, repetitions=3)
        self.write_launcher(selected=["model-a", "model-b"], excluded={"model-c": "does not fit VRAM"})
        latency = {
            "window_s": 1.0,
            "throughput_tweets_s": 10.0,
            "latency_p50_s": 0.1,
            "latency_p95_s": 0.2,
            "group_latency_p50_s": 0.1,
            "group_latency_p95_s": 0.2,
            "latency_scopes": {"adapter request": 10},
        }
        self.write_run(trials[0], result=self.result(
            trials[0], gross=2.0, recall=0.5, f1=0.4, accuracy=0.6, coverage=0.9, performance=latency))
        self.write_run(trials[1], result=self.result(
            trials[1], gross=4.0, recall=0.7, f1=0.6, accuracy=0.8, coverage=1.0, performance=latency))
        # r3 is planned but has no state.json.
        self.write_run(trials[3], result=self.result(
            trials[3], submitted=4, valid_samples=3, gross=8.0, recall=0.9, f1=0.85,
            accuracy=0.9, coverage=0.8))
        self.write_run(trials[4], result=self.result(
            trials[4], submitted=0, valid_samples=0, gross=0.0))
        self.write_run(trials[5], result=self.result(trials[5], submitted=5, valid_samples=5, gross=1.0))
        self.write_verification(
            ok=True,
            valid_complete=5,
            runs=[{"run_id": run_id, "status": "complete"} for run_id in ("r1", "r2", "r4", "r5", "r6")],
        )
        originals = {
            path: path.read_bytes()
            for path in (
                self.root / "launcher.json",
                self.root / "plan.json",
                self.root / "r1" / "state.json",
                self.root / "r1" / "attempt-0001" / "result.json",
            )
        }

        report = generate_report(self.root)
        text = self.summary()

        self.assertEqual(report["summary_path"], self.root / "report" / "summary.md")
        self.assertEqual(len(report["csv_paths"]), 7)
        self.assertTrue((self.root / "report" / "inference_medians.csv").is_file())
        self.assertIn("**Mode:** **full**", text)
        self.assertIn("Selected: `model-a`, `model-b`", text)
        self.assertIn("model-c", text)
        self.assertIn("does not fit VRAM", text)
        self.assertIn("Missing submission sample runs: 1.", text)
        self.assertIn("verified: 5", text)
        self.assertLess(text.index("## Results (inference conditions)"), text.index("## Status"))
        self.assertIn("First-pass valid-output coverage: 80.0%–100.0% (3/3 verified inference trials", text)
        self.assertNotIn("## Energy", text)
        self.assertNotIn("## Quality (first pass)", text)
        self.assertNotIn("SMOKE ONLY", text)

        row = self.table_row(text, "model-a", "1")
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row[2], "0.600")  # macro recall median of 0.5 and 0.7
        self.assertEqual(row[3], "0.500")  # macro-F1 median of 0.4 and 0.6
        self.assertEqual(row[4], "3.000 [2.000–4.000]")
        self.assertEqual(row[5], "10.0")
        self.assertEqual(row[6], "200.0")  # p95 ms, adapter request
        self.assertEqual(row[7], "adapter request")
        self.assertEqual(row[8:], ["3", "2", "2"])

        shortfall = self.table_row(text, "model-b", "16")
        self.assertIsNotNone(shortfall)
        assert shortfall is not None
        self.assertEqual(shortfall[4], "—")  # submitted 4 of 5: no energy value, not zero
        self.assertEqual(shortfall[8:], ["1", "1", "0"])

        condition = self.condition_row("model-a", 1)
        self.assertEqual(condition["engine"], "test")
        self.assertEqual(condition["adapter"], "test")
        self.assertEqual(condition["planned_reps"], "3")
        self.assertEqual(condition["complete_reps"], "2")
        self.assertEqual(condition["verified_reps"], "2")
        self.assertEqual(condition["energy_eligible_reps"], "2")
        self.assertEqual(condition["requested_reps"], "3")
        self.assertEqual(condition["macro_recall_n"], "2")
        self.assertEqual(condition["accuracy_n"], "2")
        self.assertEqual(condition["gross_j_per_tweet_n"], "2")
        self.assertAlmostEqual(float(condition["gross_j_per_tweet_median"]), 3.0)
        self.assertAlmostEqual(float(condition["gross_j_per_tweet_min"]), 2.0)
        self.assertAlmostEqual(float(condition["gross_j_per_tweet_max"]), 4.0)
        self.assertAlmostEqual(float(condition["latency_p95_ms_median"]), 200.0)
        # A training run and an out-of-scope model never enter the inference table.
        self.assertEqual({row["model"] for row in self.condition_rows()}, {"model-a", "model-b"})
        for path, content in originals.items():
            self.assertEqual(path.read_bytes(), content, f"mutated original artifact: {path}")

    def test_latency_scope_selection_across_vllm_packed_and_classical(self) -> None:
        trials = [
            self.trial("vllm16", model="qwen2b", width=16, adapter="vllm"),
            self.trial("qev256", model="joey1", width=16, adapter="qev"),
            self.trial("qev1", model="joey1", width=1, adapter="qev"),
            self.trial("svm16", model="svm", width=16, adapter="svm"),
        ]
        self.write_plan(trials, repetitions=1, config={"models": {
            "qwen2b": {"adapter": "vllm", "repo": "Qwen/Qwen3.5-2B"},
            "joey1": {"adapter": "qev", "repo": "baselabs/joey-1", "params": {"request_mode": "packed"}},
            "svm": {"adapter": "svm"},
        }})
        self.write_launcher(selected=["qwen2b", "joey1", "svm"])
        self.write_run(trials[0], result=self.result(trials[0], performance={
            "window_s": 1.0, "throughput_tweets_s": 100.0,
            "latency_p50_s": 0.12, "latency_p95_s": 0.2,
            "group_latency_p50_s": 0.3, "group_latency_p95_s": 0.5,
            "latency_scopes": {"adapter request": 40}}))
        # Packed QEV copies one envelope completion onto every tweet row; the bogus
        # per-tweet p95 below must not win over the group/envelope quantile.
        self.write_run(trials[1], result=self.result(trials[1], performance={
            "window_s": 1.0, "throughput_tweets_s": 60.0,
            "latency_p50_s": 0.0649, "latency_p95_s": 0.999,
            "group_latency_p50_s": 0.0649, "group_latency_p95_s": 0.0716,
            "latency_scopes": {"packed_request_completion": 512, "adapter request": 2}}))
        self.write_run(trials[2], result=self.result(trials[2], performance={
            "window_s": 1.0, "throughput_tweets_s": 20.0,
            "latency_p50_s": 0.046, "latency_p95_s": 0.0506,
            "group_latency_p50_s": 0.047, "group_latency_p95_s": 0.0507,
            "latency_scopes": {"adapter request": 1}}))
        self.write_run(trials[3], result=self.result(trials[3], performance={
            "window_s": 1.0, "throughput_tweets_s": 4000.0,
            "latency_p50_s": 0.000377, "latency_p95_s": 0.999,
            "group_latency_p50_s": 0.000377, "group_latency_p95_s": 0.0004289,
            "latency_scopes": {"closed-loop group completion": 58500}}))
        self.write_verification(
            ok=True, valid_complete=4,
            runs=[{"run_id": trial["run_id"], "status": "complete"} for trial in trials])

        generate_report(self.root)
        text = self.summary()

        vllm = self.table_row(text, "Qwen3.5-2B", "16")
        assert vllm is not None
        self.assertEqual(vllm[6], "200.0")
        self.assertEqual(vllm[7], "adapter request")
        packed = self.table_row(text, "Joey-1", "16")
        assert packed is not None
        self.assertEqual(packed[6], "71.60")
        self.assertEqual(packed[7], "packed envelope")
        width1 = self.table_row(text, "Joey-1", "1")
        assert width1 is not None
        self.assertEqual(width1[6], "50.60")
        self.assertEqual(width1[7], "adapter request")
        classical = self.table_row(text, "TF–IDF + LinearSVC", "16")
        assert classical is not None
        self.assertEqual(classical[6], "0.4289")
        self.assertEqual(classical[7], "group completion")

        vllm_row = self.condition_row("qwen2b", 16)
        self.assertEqual(vllm_row["engine"], "vllm")
        self.assertEqual(vllm_row["latency_scope"], "adapter request")
        self.assertAlmostEqual(float(vllm_row["latency_p95_ms_median"]), 200.0)
        self.assertAlmostEqual(float(vllm_row["group_latency_p95_ms_median"]), 500.0)

        packed_row = self.condition_row("joey1", 16)
        self.assertEqual(packed_row["engine"], "qev-packed")
        self.assertEqual(packed_row["latency_scope"], "packed_request_completion")
        self.assertAlmostEqual(float(packed_row["latency_p95_ms_median"]), 71.6)
        self.assertAlmostEqual(float(packed_row["group_latency_p95_ms_median"]), 71.6)
        self.assertAlmostEqual(float(packed_row["latency_p95_ms_n"]), 1.0)

        width1_row = self.condition_row("joey1", 1)
        self.assertEqual(width1_row["latency_scope"], "adapter request")
        self.assertAlmostEqual(float(width1_row["latency_p95_ms_median"]), 50.6)

        classical_row = self.condition_row("svm", 16)
        self.assertEqual(classical_row["engine"], "svm")
        self.assertEqual(classical_row["latency_scope"], "closed-loop group completion")
        self.assertAlmostEqual(float(classical_row["latency_p95_ms_median"]), 0.4289)
        # CSV keeps model ids; display names are only for Markdown.
        self.assertNotIn("Joey-1", (self.root / "report" / "inference_medians.csv").read_text(encoding="utf-8"))

    def test_partial_and_low_sampling_exclude_only_energy(self) -> None:
        trials = [self.trial("p1"), self.trial("p2"), self.trial("p3")]
        self.write_plan(trials, repetitions=3)
        self.write_launcher(selected=["model-a"])
        self.write_run(trials[0], result=self.result(
            trials[0], gross=2.0, recall=0.5, device_adjusted=-0.5,
            performance={"window_s": 1.0, "throughput_tweets_s": 10.0, "latency_p50_s": 0.1,
                          "latency_p95_s": 0.2, "group_latency_p50_s": 0.1, "group_latency_p95_s": 0.2,
                          "latency_scopes": {"adapter request": 10}},
            gate={"kind": "energy_sampling", "low_sampling": False, "meets_minimum_window": True}))
        # Fewer submitted tweets than planned: quality/throughput stay, energy is gated out.
        self.write_run(trials[1], result=self.result(
            trials[1], submitted=8, valid_samples=8, gross=4.0, recall=0.6, device_adjusted=0.25,
            performance={"window_s": 1.0, "throughput_tweets_s": 20.0, "latency_p50_s": 0.2,
                          "latency_p95_s": 0.3, "group_latency_p50_s": 0.2, "group_latency_p95_s": 0.3,
                          "latency_scopes": {"adapter request": 8}},
            gate={"kind": "energy_sampling", "low_sampling": False, "meets_minimum_window": True}))
        # Full count but the sampling gate failed: energy-only exclusion.
        self.write_run(trials[2], result=self.result(
            trials[2], gross=6.0, recall=0.7, device_adjusted=0.75,
            performance={"window_s": 1.0, "throughput_tweets_s": 30.0, "latency_p50_s": 0.3,
                          "latency_p95_s": 0.4, "group_latency_p50_s": 0.3, "group_latency_p95_s": 0.4,
                          "latency_scopes": {"adapter request": 10}},
            gate={"kind": "energy_sampling", "low_sampling": True, "meets_minimum_window": False}))
        self.write_verification(ok=True, valid_complete=3,
                                runs=[{"run_id": trial["run_id"], "status": "complete"} for trial in trials])

        generate_report(self.root)
        text = self.summary()
        row = self.table_row(text, "model-a", "1")
        assert row is not None
        self.assertEqual(row[2], "0.600")  # quality from all three verified replicas
        self.assertEqual(row[4], "2.000 [2.000–2.000]")  # energy from the one eligible replica
        self.assertEqual(row[5], "20.0")  # throughput median over all three
        self.assertEqual(row[8:], ["3", "3", "1"])
        self.assertIn("excluded from energy for low sampling", text)
        self.assertIn("excluded from energy for an incomplete submitted count", text)

        condition = self.condition_row("model-a", 1)
        self.assertEqual(condition["verified_reps"], "3")
        self.assertEqual(condition["energy_eligible_reps"], "1")
        self.assertEqual(condition["low_sampling_excluded"], "1")
        self.assertEqual(condition["count_shortfall_excluded"], "1")
        self.assertEqual(condition["gross_j_per_tweet_n"], "1")
        self.assertAlmostEqual(float(condition["gross_j_per_tweet_median"]), 2.0)
        self.assertEqual(condition["device_idle_adjusted_j_per_tweet_n"], "1")
        self.assertEqual(condition["macro_recall_n"], "3")
        self.assertEqual(condition["accuracy_n"], "3")
        self.assertEqual(condition["throughput_tweets_s_n"], "3")
        self.assertEqual(condition["latency_p50_ms_n"], "3")
        self.assertEqual(condition["latency_p95_ms_n"], "3")
        self.assertAlmostEqual(float(condition["latency_p95_ms_median"]), 300.0)

    def test_mixed_or_unknown_latency_scopes_are_not_pooled(self) -> None:
        mixed = [self.trial("m1"), self.trial("m2", rep=1)]
        unknown = [self.trial("u1", model="model-b")]
        self.write_plan(mixed + unknown, repetitions=2)
        self.write_launcher(selected=["model-a", "model-b"])
        self.write_run(mixed[0], result=self.result(mixed[0], performance={
            "window_s": 1.0, "throughput_tweets_s": 10.0,
            "latency_p50_s": 0.1, "latency_p95_s": 0.2,
            "group_latency_p50_s": 0.1, "group_latency_p95_s": 0.2,
            "latency_scopes": {"adapter request": 10}}))
        self.write_run(mixed[1], result=self.result(mixed[1], performance={
            "window_s": 1.0, "throughput_tweets_s": 10.0,
            "latency_p50_s": 0.9, "latency_p95_s": 0.99,
            "group_latency_p50_s": 0.05, "group_latency_p95_s": 0.06,
            "latency_scopes": {"packed_request_completion": 10}}))
        self.write_run(unknown[0], result=self.result(unknown[0], performance={
            "window_s": 1.0, "throughput_tweets_s": 10.0,
            "latency_p50_s": 0.1, "latency_p95_s": 0.2,
            "group_latency_p50_s": 0.1, "group_latency_p95_s": 0.2,
            "latency_scopes": {"mystery scope": 10}}))
        self.write_verification(ok=True, valid_complete=3,
                                runs=[{"run_id": trial["run_id"], "status": "complete"}
                                      for trial in mixed + unknown])

        generate_report(self.root)
        text = self.summary()

        mixed_row = self.table_row(text, "model-a", "1")
        assert mixed_row is not None
        self.assertEqual(mixed_row[6], "—")  # scopes are never co-mingled
        self.assertEqual(mixed_row[7], "—")
        self.assertIn("latency scopes differ across verified replicas", text)
        mixed_csv = self.condition_row("model-a", 1)
        self.assertEqual(mixed_csv["latency_scope"], "")
        self.assertEqual(mixed_csv["latency_p95_ms_n"], "0")
        self.assertEqual(mixed_csv["latency_p95_ms_median"], "")
        # Group/envelope quantiles remain a single consistent scope and are kept separately.
        self.assertEqual(mixed_csv["group_latency_p95_ms_n"], "2")
        self.assertAlmostEqual(float(mixed_csv["group_latency_p95_ms_median"]), 130.0)

        unknown_row = self.table_row(text, "model-b", "1")
        assert unknown_row is not None
        self.assertEqual(unknown_row[6], "—")
        self.assertEqual(unknown_row[7], "—")
        self.assertIn("no verified replica exposes a usable latency scope", text)
        unknown_csv = self.condition_row("model-b", 1)
        self.assertEqual(unknown_csv["latency_scope"], "")
        self.assertEqual(unknown_csv["latency_p95_ms_n"], "0")

    def test_receipt_low_sampling_excludes_energy_without_a_gate(self) -> None:
        trial = self.trial("legacy")
        self.write_plan([trial])
        self.write_launcher(selected=["model-a"])
        self.write_run(trial, result=self.result(trial, gross=2.5, recall=0.65))
        self.write_verification(
            ok=True, valid_complete=1, runs=[{"run_id": "legacy", "status": "complete"}],
            low_sampling=[{"run_id": "legacy", "window_ratio": 0.1, "sample_ratio": 0.2}])

        generate_report(self.root)
        text = self.summary()
        row = self.table_row(text, "model-a", "1")
        assert row is not None
        self.assertEqual(row[2], "0.650")
        self.assertEqual(row[4], "—")
        self.assertEqual(row[8:], ["1", "1", "0"])
        self.assertIn("excluded from energy for low sampling", text)
        condition = self.condition_row("model-a", 1)
        self.assertEqual(condition["low_sampling_excluded"], "1")
        self.assertEqual(condition["gross_j_per_tweet_n"], "0")
        self.assertEqual(condition["gross_j_per_tweet_median"], "")

    def test_missing_zero_and_tiny_values_are_not_conflated(self) -> None:
        trials = [
            self.trial("missing", model="model-missing"),
            self.trial("zero", model="model-zero"),
            self.trial("tiny", model="model-tiny"),
        ]
        self.write_plan(trials)
        self.write_launcher(selected=["model-missing", "model-zero", "model-tiny"])
        self.write_run(trials[0], result=self.result(
            trials[0], gross=None, recall=0.5, accuracy=None,
            performance={"window_s": 1.0, "throughput_tweets_s": None}))
        self.write_run(trials[1], result=self.result(trials[1], gross=0.0, recall=0.6))
        self.write_run(trials[2], result=self.result(
            trials[2], gross=4.5388275696708476e-07, recall=0.7,
            device_adjusted=-0.003855667280350627))
        self.write_verification(ok=True, valid_complete=3,
                                runs=[{"run_id": trial["run_id"], "status": "complete"} for trial in trials])

        generate_report(self.root)
        text = self.summary()

        missing = self.table_row(text, "model-missing", "1")
        assert missing is not None
        self.assertEqual(missing[2], "0.500")
        self.assertEqual(missing[4], "—")
        zero = self.table_row(text, "model-zero", "1")
        assert zero is not None
        self.assertEqual(zero[4], "0 [0–0]")
        tiny = self.table_row(text, "model-tiny", "1")
        assert tiny is not None
        self.assertEqual(tiny[4], "0.0000004539 [0.0000004539–0.0000004539]")
        self.assertNotEqual(tiny[4], "0")

        missing_row = self.condition_row("model-missing", 1)
        self.assertEqual(missing_row["gross_j_per_tweet_median"], "")
        self.assertEqual(missing_row["gross_j_per_tweet_n"], "0")
        self.assertEqual(missing_row["accuracy_median"], "")
        self.assertEqual(missing_row["accuracy_n"], "0")
        self.assertEqual(missing_row["macro_recall_n"], "1")
        self.assertEqual(missing_row["throughput_tweets_s_median"], "")
        self.assertEqual(missing_row["throughput_tweets_s_n"], "0")

        zero_row = self.condition_row("model-zero", 1)
        self.assertEqual(float(zero_row["gross_j_per_tweet_median"]), 0.0)
        self.assertEqual(zero_row["gross_j_per_tweet_n"], "1")

        tiny_row = self.condition_row("model-tiny", 1)
        self.assertAlmostEqual(float(tiny_row["gross_j_per_tweet_median"]), 4.5388275696708476e-07)
        self.assertLess(float(tiny_row["device_idle_adjusted_j_per_tweet_median"]), 0.0)

    def test_markdown_escapes_model_pipes(self) -> None:
        trial = self.trial("piped", model="model|x")
        self.write_plan([trial])
        self.write_launcher(selected=["model|x"])
        self.write_run(trial, result=self.result(trial))
        self.write_verification(ok=True, valid_complete=1,
                                runs=[{"run_id": "piped", "status": "complete"}])

        generate_report(self.root)
        text = self.summary()
        self.assertIn("| model\\|x | 1 |", text)

    def test_unknown_mode_suppresses_condition_table_and_csv(self) -> None:
        trial = self.trial("r1")
        self.write_plan([trial])
        # No launcher.json: the mode stays unconfirmed.
        self.write_run(trial, result=self.result(trial))
        self.write_verification(ok=True, valid_complete=1,
                                runs=[{"run_id": "r1", "status": "complete"}])

        report = generate_report(self.root)
        text = self.summary()
        self.assertIsNone(report["launcher"]["mode"])
        self.assertIn("Condition medians omitted: mode is not confirmed `full`.", text)
        self.assertFalse((self.root / "report" / "inference_medians.csv").exists())
        self.assertIsNone(report["results_csv"])

    def test_mini_is_smoke_only_and_suppresses_comparative_metrics(self) -> None:
        trial = self.trial("mini-run", expected=5)
        self.write_plan([trial], repetitions=1)
        self.write_launcher(mode="mini", selected=["model-a"])
        self.write_run(trial, result=self.result(trial, submitted=5, valid_samples=5, gross=1.5))
        self.write_verification(ok=True, valid_complete=1, runs=[{"run_id": "mini-run", "status": "complete"}])
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            self.assertEqual(main([str(self.root)]), 0)
        text = self.summary()

        self.assertIn("SMOKE ONLY — NOT A COMPARATIVE RESULT", text)
        self.assertIn("mini runs do not report comparative energy medians", text)
        self.assertIn("comparative quality medians are intentionally omitted", text)
        self.assertIn("## Smoke-only audit (mini)", text)
        self.assertNotIn("Macro recall", text)
        self.assertNotIn("## Results (inference conditions)", text)
        self.assertNotIn("median 1.5000", text)
        self.assertIn("No `inference_medians.csv` is written for a mini run.", text)
        self.assertFalse((self.root / "report" / "inference_medians.csv").exists())
        self.assertTrue((self.root / "report" / "energy_intensity.csv").is_file())
        self.assertIn("mode: mini", stdout.getvalue())

    def test_missing_verification_lists_conditions_with_blank_metrics(self) -> None:
        trial = self.trial("unverified", expected=5)
        self.write_plan([trial])
        self.write_launcher(mode="full", selected=["model-a"])
        self.write_run(trial, result=self.result(trial, submitted=5, valid_samples=5))

        report = generate_report(self.root)
        text = self.summary()
        row = self.table_row(text, "model-a", "1")
        assert row is not None

        self.assertIsNotNone(report["results_csv"])
        self.assertIn("Overall: **missing**", text)
        self.assertIn("verified: 0", text)
        self.assertEqual(row[2:8], ["—", "—", "—", "—", "—", "—"])
        self.assertEqual(row[8:], ["1", "0", "0"])
        self.assertIn("No condition has a receipt-verified complete inference replica", text)
        condition = self.condition_row("model-a", 1)
        self.assertEqual(condition["verified_reps"], "0")
        self.assertEqual(condition["gross_j_per_tweet_median"], "")

    def test_missing_plan_reports_unmeasured_and_omits_condition_csv(self) -> None:
        trial = self.trial("orphan", expected=5)
        self.write_launcher(mode="full")
        self.write_run(trial, result=self.result(trial, submitted=5, valid_samples=5))

        report = generate_report(self.root)
        text = self.summary()

        self.assertIsNone(report["plan"])
        self.assertIn("UNMEASURED", text)
        self.assertIn("plan.json not found", text)
        self.assertIn("Condition medians omitted: the run is unmeasured (no plan).", text)
        self.assertIsNone(report["results_csv"])
        self.assertFalse((self.root / "report" / "inference_medians.csv").exists())
        self.assertTrue((self.root / "report" / "trials.csv").is_file())

    def test_stale_or_attempt_mismatched_verification_restricts_full_medians(self) -> None:
        trials = [self.trial("r1", width=1), self.trial("r2", width=1)]
        self.write_plan(trials, repetitions=2)
        self.write_launcher(selected=["model-a"])
        self.write_run(trials[0], result=self.result(trials[0], gross=2.0))
        self.write_run(trials[1], result=self.result(trials[1], gross=4.0))

        # Prior good verification, then r2 completed without updating the receipt.
        self.write_verification(ok=True, valid_complete=1, runs=[{"run_id": "r1", "status": "complete"}])
        generate_report(self.root)
        text = self.summary()
        row = self.table_row(text, "model-a", "1")
        assert row is not None
        self.assertIn("Overall: **partial**", text)
        self.assertEqual(row[4], "2.000 [2.000–2.000]")
        self.assertEqual(row[8:], ["2", "1", "1"])
        self.assertEqual(self.condition_row("model-a", 1)["verified_reps"], "1")
        self.assertIn("excluded from every median: 1", text)

        # A changed active attempt invalidates the old receipt for r1 as well.
        self.write_verification(
            ok=True,
            valid_complete=2,
            runs=[
                {"run_id": "r1", "status": "complete", "active_attempt": "attempt-9999"},
                {"run_id": "r2", "status": "complete"},
            ],
        )
        generate_report(self.root)
        text = self.summary()
        row = self.table_row(text, "model-a", "1")
        assert row is not None
        self.assertIn("Overall: **partial**", text)
        self.assertEqual(row[4], "4.000 [4.000–4.000]")
        self.assertEqual(row[8:], ["2", "1", "1"])
        self.assertIn("active attempt differs from the receipt", text)

    def test_bad_states_partial_verification_and_failed_runs(self) -> None:
        trials = [
            self.trial("r1", width=1),
            self.trial("r2", width=1),
            self.trial("r3", width=16),
            self.trial("r4", model="model-b", kind="device_idle", expected=0),
            self.trial("r5", model="model-b", expected=10),
            self.trial("r6", model="model-c", expected=10),
            self.trial("r7", model="model-c", expected=10),
            self.trial("r8", model="model-c", expected=10),
        ]
        self.write_plan(trials)
        self.write_launcher()
        self.write_run(trials[0], result=self.result(trials[0]))
        self.write_run(trials[1], result=self.result(trials[1]))
        self.write_run(trials[2], result=self.result(trials[2]))
        self.write_run(trials[3], status="failed", reason="VRAM preflight failed")
        # r5 has no state directory.
        self.write_run(trials[5], malformed_state=True)
        self.write_run(trials[6], status="complete", result=None)
        self.write_run(trials[7], result=self.result(trials[7], valid=False))
        self.write_verification(
            ok=False,
            valid_complete=1,
            runs=[{"run_id": "r1", "status": "complete"}, {"run_id": "r3", "status": "complete"}],
            inconsistencies=["r3: prediction digest mismatch"],
        )

        generate_report(self.root)
        text = self.summary()

        self.assertIn("Overall: **failed**", text)
        self.assertIn("| complete | 3 |", text)
        self.assertIn("| invalid | 3 |", text)
        self.assertIn("| failed | 1 |", text)
        self.assertIn("| pending | 1 |", text)
        self.assertIn("verified: 0", text)
        self.assertIn("failed: 1", text)
        self.assertIn("unverified: 2", text)
        self.assertIn("VRAM preflight failed", text)
        self.assertIn("active result.json is missing", text)
        self.assertIn("result valid flag is not true", text)
        self.assertIn("No condition has a receipt-verified complete inference replica", text)
        r1 = self.condition_row("model-a", 1)
        self.assertEqual(r1["planned_reps"], "2")
        self.assertEqual(r1["complete_reps"], "2")
        self.assertEqual(r1["verified_reps"], "0")
        self.assertEqual(r1["gross_j_per_tweet_median"], "")
        r5 = self.condition_row("model-b", 1)
        self.assertEqual(r5["planned_reps"], "1")
        self.assertEqual(r5["complete_reps"], "0")
        model_c = self.condition_row("model-c", 1)
        self.assertEqual(model_c["planned_reps"], "3")
        self.assertEqual(model_c["complete_reps"], "0")

    def test_cli_rejects_missing_root(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            self.assertEqual(main([str(self.root / "does-not-exist")]), 2)
        self.assertIn("not a directory", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
