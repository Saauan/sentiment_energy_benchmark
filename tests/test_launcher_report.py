"""Compact offline tests for launcher_report.py; no measurement is exercised."""

from __future__ import annotations

import contextlib
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

    def write_plan(self, trials: list[dict], *, sha: str = "config-a") -> None:
        (self.root / "plan.json").write_text(
            json.dumps({
                "format_version": 2,
                "execution_protocol": "engine-session-v1",
                "phase": "full",
                "config_sha256": sha,
                "trials": trials,
            }),
            encoding="utf-8",
        )

    def write_verification(self, *, ok=True, valid_complete=0, runs=None, inconsistencies=None) -> None:
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
                "low_sampling": [],
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
    ) -> dict:
        return {
            "run_id": run_id,
            "phase": "full",
            "kind": kind,
            "model": model,
            "batch_size": width,
            "rep": 0,
            "adapter": "test",
            "passes": 1 if kind == "inference" else 0,
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
        gross: float = 2.0,
        recall: float = 0.7,
        f1: float = 0.6,
        accuracy: float = 0.8,
        coverage: float = 1.0,
        valid: bool = True,
    ) -> dict:
        return {
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
            "performance": {"window_s": 1.0},
            "energy": {"energy_kwh": 0.0001, "run_id": "energy-a"},
            "derived": {"gross_j_per_tweet": gross},
        }

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

    def test_full_report_uses_only_verified_medians_and_preserves_artifacts(self) -> None:
        trials = [
            self.trial("r1", width=1, expected=10),
            self.trial("r2", width=1, expected=10),
            self.trial("r3", model="model-b", width=16, expected=5),
            self.trial("r4", width=1, kind="training", expected=0),
        ]
        self.write_plan(trials)
        self.write_launcher(selected=["model-a", "model-b"], excluded={"model-c": "does not fit VRAM"})
        self.write_run(trials[0], result=self.result(trials[0], gross=2.0, recall=0.5))
        self.write_run(trials[1], result=self.result(trials[1], gross=4.0, recall=0.7))
        self.write_run(trials[2], result=self.result(trials[2], submitted=4, valid_samples=3, gross=8.0))
        self.write_run(trials[3], result=self.result(trials[3], submitted=0, valid_samples=0, gross=0.0))
        self.write_verification(
            ok=True,
            valid_complete=4,
            runs=[{"run_id": run_id, "status": "complete"} for run_id in ("r1", "r2", "r3", "r4")],
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
        self.assertEqual(len(report["csv_paths"]), 6)
        self.assertIn("**Mode:** **full**", text)
        self.assertIn("Selected: `model-a`, `model-b`", text)
        self.assertIn("model-c", text)
        self.assertIn("does not fit VRAM", text)
        self.assertIn("n=2 verified replica(s); median 3.0000 J/tweet", text)
        self.assertIn("Missing submission sample runs: 1.", text)
        self.assertIn("verified: 4", text)
        self.assertIn("Macro recall", text)
        self.assertNotIn("SMOKE ONLY", text)
        for path, content in originals.items():
            self.assertEqual(path.read_bytes(), content, f"mutated original artifact: {path}")

    def test_stale_or_attempt_mismatched_verification_restricts_full_medians(self) -> None:
        trials = [self.trial("r1", width=1), self.trial("r2", width=1)]
        self.write_plan(trials)
        self.write_launcher(selected=["model-a"])
        self.write_run(trials[0], result=self.result(trials[0], gross=2.0))
        self.write_run(trials[1], result=self.result(trials[1], gross=4.0))

        # Prior good verification, then r2 completed without updating the receipt.
        self.write_verification(ok=True, valid_complete=1, runs=[{"run_id": "r1", "status": "complete"}])
        generate_report(self.root)
        text = self.summary()
        self.assertIn("Overall: **partial**", text)
        self.assertIn("n=1 verified replica(s); median 2.0000 J/tweet", text)
        self.assertIn("Excluded as unverified: 1 eligible inference run(s).", text)

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
        self.assertIn("Overall: **partial**", text)
        self.assertIn("n=1 verified replica(s); median 4.0000 J/tweet", text)
        self.assertIn("active attempt differs from the receipt", text)
        self.assertIn("Excluded as unverified: 1 eligible inference run(s).", text)

    def test_mini_is_smoke_only_and_suppresses_comparative_metrics(self) -> None:
        trial = self.trial("mini-run", expected=5)
        self.write_plan([trial])
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
        self.assertNotIn("Macro recall", text)
        self.assertNotIn("median 1.5000", text)
        self.assertTrue((self.root / "report" / "energy_intensity.csv").is_file())
        self.assertIn("mode: mini", stdout.getvalue())

    def test_missing_verification_suppresses_full_medians(self) -> None:
        trial = self.trial("unverified", expected=5)
        self.write_plan([trial])
        self.write_launcher(mode="full", selected=["model-a"])
        self.write_run(trial, result=self.result(trial, submitted=5, valid_samples=5))

        generate_report(self.root)
        text = self.summary()

        self.assertIn("Overall: **missing**", text)
        self.assertIn("verified: 0", text)
        self.assertIn("Energy medians suppressed", text)
        self.assertIn("Quality medians suppressed", text)
        self.assertNotIn("n=1 verified replica", text)

    def test_missing_plan_reports_unmeasured(self) -> None:
        trial = self.trial("orphan", expected=5)
        self.write_launcher(mode="full")
        self.write_run(trial, result=self.result(trial, submitted=5, valid_samples=5))

        report = generate_report(self.root)
        text = self.summary()

        self.assertIsNone(report["plan"])
        self.assertIn("UNMEASURED", text)
        self.assertIn("plan.json not found", text)
        self.assertIn("Energy medians omitted", text)
        self.assertIn("Quality medians omitted", text)
        self.assertTrue((self.root / "report" / "trials.csv").is_file())

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
        self.assertIn("Energy medians suppressed", text)
        self.assertIn("Quality medians suppressed", text)

    def test_cli_rejects_missing_root(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            self.assertEqual(main([str(self.root / "does-not-exist")]), 2)
        self.assertIn("not a directory", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
