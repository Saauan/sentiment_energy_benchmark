"""Coordinator-facing, offline tests for the run_experiment.py launcher.

``launcher_plan`` and ``launcher_setup`` are in-memory fakes, and
``run_experiment.run_command`` is a recorder. No pinned install, model
download, engine start, measurement, or real benchmark process runs. The
recorder persists the JSON receipts that benchmark commands would print, which
lets these tests exercise setup, resume, and readiness without runtime work.
Every write stays under a temporary directory.
"""
from __future__ import annotations

from contextlib import ExitStack, redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import dataclass
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import benchmark
import run_experiment
import storage


CURRENT_MACHINE = "machine-current-test-host"
OTHER_MACHINE = "machine-from-another-host"
GPU_UUID = "GPU-11111111-2222-3333-4444-555555555555"
HOST_GUARD_MESSAGE = "resume requires the original host; start a new run here"

CFG = {
    "execution_protocol": "engine-session-v1",
    "energy": {"country_iso_code": "FRA", "gpu_ids": [0]},
    "phases": {"full": {"sample_size": 12, "batch_sizes": [1, 8], "repetitions": 2}},
    "models": {"model-a": {"adapter": "vllm", "device": "cuda", "params": {}}},
}
DISCOVERY = {
    "selected": ["model-a"],
    "excluded": {"model-b": "insufficient VRAM headroom"},
    "download_bytes": {"model-a": 2_000_000_000},
    "disk": {"free_bytes": 10_000_000_000, "required_bytes": 2_000_000_000},
    "machine": {"machine_id": CURRENT_MACHINE},
}
INFERENCE_ROW = {
    "run_id": "pilot-inference", "kind": "inference", "status": "complete",
    "active_attempt": "attempt-0001", "reason": None,
}


def counts(*, complete=1, incomplete=0, failed=0, skipped=0, pending=0, running=0):
    return {
        "pending": pending, "running": running, "complete": complete,
        "incomplete": incomplete, "failed": failed, "skipped": skipped,
    }


def verify_report(*, ok=True, report_counts=None, runs=(), low_sampling=(), inconsistencies=()):
    return {
        "ok": ok, "inconsistencies": list(inconsistencies),
        "low_sampling": list(low_sampling), "valid_complete": len(runs),
        "counts": report_counts or counts(), "runs": list(runs),
    }


GOOD_VERIFY = verify_report()
STATUS = {"root": "synthetic-root", "phase": "full", "counts": counts(), "runs": []}
READY_REPORT = verify_report(runs=(INFERENCE_ROW,))


class FakeTTY(io.StringIO):
    def isatty(self):
        return True


@dataclass(frozen=True)
class CommandCall:
    command: str
    root: Path
    config: Path
    flags: tuple[str, ...]
    log: Path


class Harness:
    """Install fakes and temporary roots for one launcher test."""

    def __init__(self, base: Path):
        self.base = base
        self.out = base / "out"
        self.work = base / "work"
        self.client = base / "fake-client"
        self.stdin = FakeTTY()
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        self.calls: list[CommandCall] = []
        self.pilot_report = deepcopy(READY_REPORT)
        self.main_report = deepcopy(GOOD_VERIFY)
        self.pilot_valid = 1
        self.pilot_run_code = 0
        self.main_run_code = 0
        self.writes_plan = True
        self._dont_write_bytecode = sys.dont_write_bytecode

        def prepare(mode, country):
            return deepcopy(CFG), deepcopy(DISCOVERY)

        self.prepare = mock.Mock(side_effect=prepare)
        self.check_resume = mock.Mock()
        self.provision = mock.Mock(return_value=self.client)
        self.run_command = mock.Mock(side_effect=self._fake_run_command)
        self.render = mock.Mock(return_value=subprocess.CompletedProcess([], 0))
        self.status_report = mock.Mock(return_value=deepcopy(STATUS))

        self._stack = ExitStack()
        self._stack.enter_context(mock.patch.object(run_experiment, "ROOT", self.base))
        self._stack.enter_context(mock.patch.object(run_experiment, "WORK", self.work))
        self._stack.enter_context(mock.patch.object(
            run_experiment, "platform",
            types.SimpleNamespace(system=lambda: "Linux", machine=lambda: "x86_64")))
        self._stack.enter_context(mock.patch.object(
            run_experiment, "subprocess", types.SimpleNamespace(
                run=self.render, SubprocessError=subprocess.SubprocessError)))
        self._stack.enter_context(mock.patch.object(run_experiment, "run_command", self.run_command))
        self._stack.enter_context(mock.patch.object(benchmark, "status_report", self.status_report))
        plan_module = types.ModuleType("launcher_plan")
        plan_module.prepare = self.prepare
        plan_module.check_resume = self.check_resume
        setup_module = types.ModuleType("launcher_setup")
        setup_module.provision = self.provision
        self._stack.enter_context(mock.patch.dict(
            sys.modules, {"launcher_plan": plan_module, "launcher_setup": setup_module}))
        self._stack.enter_context(mock.patch.object(sys, "stdin", self.stdin))
        self._stack.enter_context(redirect_stdout(self.stdout))
        self._stack.enter_context(redirect_stderr(self.stderr))

    def close(self):
        sys.dont_write_bytecode = self._dont_write_bytecode
        self._stack.close()

    def invoke(self, *argv):
        return run_experiment.main(list(argv))

    def queue_input(self, text="y\n"):
        self.stdin.seek(0)
        self.stdin.truncate(0)
        self.stdin.write(text)
        self.stdin.seek(0)

    def clear_output(self):
        for stream in (self.stdout, self.stderr):
            stream.seek(0)
            stream.truncate(0)

    def reset_calls(self):
        for item in (self.prepare, self.check_resume, self.provision,
                     self.run_command, self.render, self.status_report):
            item.reset_mock()
        self.provision.side_effect = None
        self.provision.return_value = self.client
        self.check_resume.side_effect = None
        self.render.return_value = subprocess.CompletedProcess([], 0)
        self.status_report.return_value = deepcopy(STATUS)
        self.calls.clear()

    def reset_runner(self):
        self.run_command.reset_mock()
        self.render.reset_mock()
        self.calls.clear()

    def seed_root(self, name="seeded", *, cfg=None, receipt=None):
        """Create saved launcher metadata without invoking the launcher."""
        out = self.base / name
        out.mkdir()
        saved_cfg = deepcopy(CFG) if cfg is None else cfg
        saved_receipt = {
            "format": 1, "mode": "mini", "discovery": deepcopy(DISCOVERY),
            "config_sha256": storage.identity(saved_cfg), "readiness_required": False,
        }
        if receipt:
            saved_receipt.update(receipt)
        storage.atomic_json(out / "resolved.json", saved_cfg)
        storage.atomic_json(out / "launcher.json", saved_receipt)
        return out

    def host_guard(self, _cfg, *, expected_machine_id, expected_gpu_uuid=None):
        if expected_machine_id != CURRENT_MACHINE:
            raise ValueError(HOST_GUARD_MESSAGE)

    def _fake_run_command(self, client, root, config_path, command, log, *flags):
        root = Path(root)
        config_path = Path(config_path)
        log = Path(log)
        self.calls.append(CommandCall(command, root, config_path, tuple(flags), log))
        log.touch(exist_ok=False)
        pilot = root.name == "readiness"
        if command == "run":
            if self.writes_plan:
                storage.atomic_json(root / "plan.json", {"synthetic": True, "root": root.name})
            return self.pilot_run_code if pilot else self.main_run_code
        if command == "verify":
            report = deepcopy(self.pilot_report if pilot else self.main_report)
            if pilot and report.get("ok") and report.get("runs"):
                row = next((item for item in report["runs"] if item["kind"] == "inference"), None)
                if row is not None:
                    attempt = root / row["run_id"] / row["active_attempt"]
                    attempt.mkdir(parents=True, exist_ok=True)
                    storage.atomic_json(attempt / "result.json", {"counts": {"valid": self.pilot_valid}})
            storage.atomic_json(log, report)
            return 0 if report.get("ok") else 1
        raise AssertionError(f"unexpected benchmark command: {command}")


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.h = Harness(self.base)
        self.h.queue_input("y\n")
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(self.h.close)

    # -- fresh runs -------------------------------------------------------

    def test_fresh_dry_run_previews_without_writes_or_setup(self):
        code = self.h.invoke("mini", "--country", "fra", "--out", str(self.h.out), "--dry-run")

        self.assertEqual(code, 0)
        self.h.prepare.assert_called_once_with("mini", "FRA")
        self.h.check_resume.assert_not_called()
        self.h.provision.assert_not_called()
        self.h.run_command.assert_not_called()
        self.h.render.assert_not_called()
        self.assertEqual(list(self.base.iterdir()), [])
        self.assertIn("Dry run: no installation, measurements, or writes.", self.h.stdout.getvalue())

    def test_confirmation_refusal_leaves_no_trace(self):
        self.h.queue_input("n\n")
        code = self.h.invoke("mini", "--country", "FRA", "--out", str(self.h.out))

        self.assertEqual(code, 0)
        self.h.check_resume.assert_not_called()
        self.h.provision.assert_not_called()
        self.h.run_command.assert_not_called()
        self.assertEqual(list(self.base.iterdir()), [])
        self.assertIn("Cancelled; nothing written.", self.h.stdout.getvalue())

    def test_non_interactive_run_without_yes_is_rejected(self):
        self.h.stdin.isatty = lambda: False
        code = self.h.invoke("mini", "--country", "FRA", "--out", str(self.h.out))

        self.assertEqual(code, 2)
        self.h.check_resume.assert_not_called()
        self.h.provision.assert_not_called()
        self.h.run_command.assert_not_called()
        self.assertEqual(list(self.base.iterdir()), [])
        self.assertIn("confirmation requires a terminal or explicit --yes", self.h.stderr.getvalue())

    def test_preflight_blockers_stop_even_a_dry_run_without_writes(self):
        blocked = deepcopy(DISCOVERY)
        blocked["blockers"] = ["insufficient free disk for weight staging"]
        self.h.prepare.side_effect = lambda mode, country: (deepcopy(CFG), blocked)

        code = self.h.invoke("mini", "--country", "FRA", "--out", str(self.h.out), "--dry-run")

        self.assertEqual(code, 2)
        self.h.check_resume.assert_not_called()
        self.h.provision.assert_not_called()
        self.h.run_command.assert_not_called()
        self.assertEqual(list(self.base.iterdir()), [])
        text = self.h.stdout.getvalue()
        self.assertIn("Blocked — insufficient free disk for weight staging", text)
        self.assertIn("Preflight blocked this run", text)

    def test_mini_lifecycle_saves_receipts_and_verifies(self):
        out = self.h.out
        code = self.h.invoke("mini", "--country", "FRA", "--out", str(out), "--yes")

        self.assertEqual(code, 0)
        self.assertEqual(storage.load(out / "resolved.json"), CFG)
        self.assertEqual(storage.load(out / "launcher.json"), {
            "format": 1, "mode": "mini", "discovery": DISCOVERY,
            "config_sha256": storage.identity(CFG), "readiness_required": False,
        })
        self.h.prepare.assert_called_once_with("mini", "FRA")
        self.h.check_resume.assert_called_once_with(
            CFG, expected_machine_id=CURRENT_MACHINE)
        self.h.provision.assert_called_once()
        self.assertEqual(self.h.provision.call_args.args[0], CFG)
        self.assertTrue(callable(self.h.provision.call_args.args[1]))
        self.assertEqual([call.command for call in self.h.calls], ["run", "verify"])
        self.assertTrue(all(call.root == out for call in self.h.calls))
        self.assertTrue(all(call.config == out / "resolved.json" for call in self.h.calls))
        self.assertEqual(self.h.calls[0].flags, ("--progress",))
        self.assertEqual(self.h.calls[1].flags, ())
        self.assertEqual(storage.load(out / "verification.json"), GOOD_VERIFY)
        self.assertTrue((out / "plan.json").is_file())
        self.assertTrue((out / "launcher-logs" / "invocation-0001" / "run.json").is_file())
        self.assertTrue((out / "launcher-logs" / "invocation-0001" / "verify.json").is_file())
        self.assertFalse((out / "readiness").exists())
        self.h.status_report.assert_not_called()
        self.h.render.assert_called_once()
        args, kwargs = self.h.render.call_args
        self.assertEqual(args[0], [str(self.h.client), "launcher_report.py", str(out)])
        self.assertEqual(kwargs["cwd"], self.base)

    def test_retry_flags_are_rejected_for_fresh_runs(self):
        for extra, message in (
            (("--retry-failed",), "retry flags require --resume"),
            (("--retry-skipped",), "retry flags require --resume"),
            (("--validated",), "--validated is for new full runs only"),
        ):
            with self.subTest(extra=extra):
                self.h.reset_calls()
                self.h.clear_output()
                code = self.h.invoke(
                    "mini", "--country", "FRA", "--out", str(self.h.out), "--yes", *extra)
                self.assertEqual(code, 2)
                self.assertIn(message, self.h.stderr.getvalue())
                self.h.prepare.assert_not_called()
                self.h.check_resume.assert_not_called()
                self.h.provision.assert_not_called()
                self.h.run_command.assert_not_called()
                self.assertEqual(list(self.base.iterdir()), [])

    # -- setup failure and resume ----------------------------------------

    def _fail_setup(self):
        self.h.provision.side_effect = RuntimeError("pinned setup failed")
        code = self.h.invoke("mini", "--country", "FRA", "--out", str(self.h.out), "--yes")
        self.h.provision.side_effect = None
        return code

    def test_setup_failure_freezes_config_and_resume_does_not_reselect(self):
        out = self.h.out
        code = self._fail_setup()

        self.assertEqual(code, 2)
        self.assertIn("pinned setup failed", self.h.stderr.getvalue())
        self.h.prepare.assert_called_once_with("mini", "FRA")
        self.h.check_resume.assert_called_once_with(
            CFG, expected_machine_id=CURRENT_MACHINE)
        self.h.run_command.assert_not_called()
        self.h.render.assert_not_called()
        self.assertFalse((out / "plan.json").exists())
        self.assertFalse((out / "verification.json").exists())
        self.assertTrue((out / "launcher-logs" / "invocation-0001").is_dir())
        frozen = ((out / "resolved.json").read_bytes(), (out / "launcher.json").read_bytes())

        self.h.reset_runner()
        self.h.clear_output()
        self.h.queue_input("y\n")
        code = self.h.invoke("--resume", str(out), "--yes")

        self.assertEqual(code, 0)
        self.h.prepare.assert_called_once_with("mini", "FRA")
        self.assertEqual(self.h.check_resume.call_count, 3)
        self.assertEqual(
            self.h.check_resume.call_args_list,
            [mock.call(CFG, expected_machine_id=CURRENT_MACHINE)] * 3,
        )
        self.assertEqual(self.h.provision.call_count, 2)
        self.assertEqual([call.command for call in self.h.calls], ["run", "verify"])
        self.assertTrue(all(call.config == out / "resolved.json" for call in self.h.calls))
        self.assertEqual(self.h.calls[0].flags, ("--progress",))
        self.assertEqual(self.h.calls[1].flags, ())
        self.assertEqual(
            ((out / "resolved.json").read_bytes(), (out / "launcher.json").read_bytes()),
            frozen,
        )
        self.assertTrue((out / "launcher-logs" / "invocation-0002").is_dir())
        self.assertEqual(storage.load(out / "verification.json"), GOOD_VERIFY)
        self.assertIn("Failed trials: leave unchanged; skipped trials: leave unchanged.",
                      self.h.stdout.getvalue())

    def test_resume_forwards_explicit_retry_flags_to_the_runner(self):
        out = self.h.out
        self.assertEqual(self._fail_setup(), 2)
        self.h.reset_calls()
        self.h.clear_output()
        self.h.queue_input("y\n")

        code = self.h.invoke(
            "--resume", str(out), "--yes", "--retry-failed", "--retry-skipped")

        self.assertEqual(code, 0)
        self.h.prepare.assert_not_called()
        self.assertEqual(self.h.check_resume.call_count, 2)
        self.assertEqual([call.command for call in self.h.calls], ["run", "verify"])
        self.assertEqual(self.h.calls[0].flags,
                         ("--progress", "--retry-failed", "--retry-skipped"))
        self.assertEqual(self.h.calls[1].flags, ())
        self.assertIn("Failed trials: retry; skipped trials: retry.", self.h.stdout.getvalue())

    def test_resume_reports_saved_status_without_rerunning_status(self):
        out = self.h.out
        self.h.main_report = verify_report(
            ok=False, report_counts=counts(complete=0, failed=1),
            inconsistencies=("synthetic verify failure",))
        self.assertEqual(
            self.h.invoke("mini", "--country", "FRA", "--out", str(out), "--yes"), 1)
        plan = storage.load(out / "plan.json")
        self.assertTrue(plan["synthetic"])

        self.h.main_report = deepcopy(GOOD_VERIFY)
        self.h.reset_calls()
        self.h.clear_output()
        self.h.queue_input("y\n")
        code = self.h.invoke("--resume", str(out), "--yes")

        self.assertEqual(code, 0)
        self.h.prepare.assert_not_called()
        self.h.status_report.assert_called_once_with(out, plan)
        self.assertEqual([call.command for call in self.h.calls], ["run", "verify"])
        self.assertTrue(all(call.config == out / "resolved.json" for call in self.h.calls))
        self.assertEqual(self.h.calls[0].flags, ("--progress",))
        self.assertEqual(self.h.calls[1].flags, ())

    def test_resume_dry_run_previews_status_without_writes_or_setup(self):
        plan = {"synthetic": True}
        out = self.h.seed_root()
        storage.atomic_json(out / "plan.json", plan)

        code = self.h.invoke("--resume", str(out), "--dry-run")

        self.assertEqual(code, 0)
        self.h.check_resume.assert_called_once_with(
            storage.load(out / "resolved.json"), expected_machine_id=CURRENT_MACHINE)
        self.h.status_report.assert_called_once_with(out, plan)
        self.h.provision.assert_not_called()
        self.h.run_command.assert_not_called()
        self.h.render.assert_not_called()
        self.assertFalse((out / "launcher-logs").exists())
        self.assertFalse((out / "verification.json").exists())
        self.assertIn("Dry run: no installation, measurements, or writes.",
                      self.h.stdout.getvalue())

    # -- resume guards ----------------------------------------------------

    def test_resume_rejects_new_run_options_before_reading_the_root(self):
        out = self.h.seed_root()
        for extra in (("mini",), ("--country", "FRA"), ("--validated",)):
            with self.subTest(extra=extra):
                self.h.reset_calls()
                self.h.clear_output()
                code = self.h.invoke("--resume", str(out), "--yes", *extra)
                self.assertEqual(code, 2)
                self.assertIn("--resume uses the saved mode, country, and readiness policy",
                              self.h.stderr.getvalue())
                self.h.check_resume.assert_not_called()
                self.h.provision.assert_not_called()
                self.h.run_command.assert_not_called()
                self.assertFalse((out / "launcher-logs").exists())

    def test_resume_rejects_unsupported_or_modified_receipts_before_setup(self):
        cases = {
            "unsupported-format": ({"format": 2}, "unsupported or was modified"),
            "unknown-mode": ({"mode": "quick"}, "unsupported or was modified"),
            "changed-hash": ({"config_sha256": "0" * 64}, "unsupported or was modified"),
        }
        for name, (receipt, message) in cases.items():
            with self.subTest(name=name):
                self.h.reset_calls()
                self.h.clear_output()
                out = self.h.seed_root(name, receipt=receipt)
                code = self.h.invoke("--resume", str(out), "--yes")
                self.assertEqual(code, 2)
                self.assertIn(message, self.h.stderr.getvalue())
                self.h.check_resume.assert_not_called()
                self.h.provision.assert_not_called()
                self.h.run_command.assert_not_called()
                self.assertFalse((out / "launcher-logs").exists())
                self.assertFalse((out / "verification.json").exists())

    def test_resume_rejects_a_modified_saved_configuration(self):
        out = self.h.seed_root()
        tampered = deepcopy(CFG)
        tampered["phases"]["full"]["repetitions"] = 9
        storage.atomic_json(out / "resolved.json", tampered)

        code = self.h.invoke("--resume", str(out), "--yes")

        self.assertEqual(code, 2)
        self.assertIn("unsupported or was modified", self.h.stderr.getvalue())
        self.h.check_resume.assert_not_called()
        self.h.provision.assert_not_called()
        self.h.run_command.assert_not_called()
        self.assertFalse((out / "launcher-logs").exists())

    def test_resume_checks_the_saved_machine_id_before_setup(self):
        foreign = deepcopy(DISCOVERY)
        foreign["machine"]["machine_id"] = OTHER_MACHINE
        out = self.h.seed_root("foreign-host", receipt={"discovery": foreign})
        self.h.check_resume.side_effect = self.h.host_guard

        code = self.h.invoke("--resume", str(out), "--yes")

        self.assertEqual(code, 2)
        self.assertIn(HOST_GUARD_MESSAGE, self.h.stderr.getvalue())
        self.h.check_resume.assert_called_once_with(
            CFG, expected_machine_id=OTHER_MACHINE)
        self.h.provision.assert_not_called()
        self.h.run_command.assert_not_called()
        self.assertFalse((out / "launcher-logs").exists())
        self.assertFalse((out / "verification.json").exists())

    def test_fresh_run_rechecks_the_host_before_creating_the_root(self):
        foreign = deepcopy(DISCOVERY)
        foreign["machine"]["machine_id"] = OTHER_MACHINE
        self.h.prepare.side_effect = lambda mode, country: (deepcopy(CFG), foreign)
        self.h.check_resume.side_effect = self.h.host_guard

        code = self.h.invoke("mini", "--country", "FRA", "--out", str(self.h.out), "--yes")

        self.assertEqual(code, 2)
        self.assertIn(HOST_GUARD_MESSAGE, self.h.stderr.getvalue())
        self.h.check_resume.assert_called_once_with(
            CFG, expected_machine_id=OTHER_MACHINE)
        self.h.provision.assert_not_called()
        self.h.run_command.assert_not_called()
        self.assertFalse(self.h.out.exists())

    def test_resume_rechecks_a_recorded_gpu_uuid(self):
        discovery = deepcopy(DISCOVERY)
        discovery["gpu"] = {"uuid": GPU_UUID}
        out = self.h.seed_root("gpu-host", receipt={"discovery": discovery})

        code = self.h.invoke("--resume", str(out), "--yes")

        self.assertEqual(code, 0)
        self.assertEqual(self.h.check_resume.call_count, 2)
        self.assertEqual(
            self.h.check_resume.call_args_list,
            [mock.call(CFG, expected_machine_id=CURRENT_MACHINE,
                       expected_gpu_uuid=GPU_UUID)] * 2,
        )
        self.h.provision.assert_called_once()
        self.assertEqual([call.command for call in self.h.calls], ["run", "verify"])

    def test_resume_lets_the_plan_helper_reject_saved_conditions_before_setup(self):
        out = self.h.seed_root()
        self.h.check_resume.side_effect = ValueError("saved conditions no longer supported")

        code = self.h.invoke("--resume", str(out), "--yes")

        self.assertEqual(code, 2)
        self.assertIn("saved conditions no longer supported", self.h.stderr.getvalue())
        self.h.provision.assert_not_called()
        self.h.run_command.assert_not_called()
        self.assertFalse((out / "launcher-logs").exists())

    # -- readiness --------------------------------------------------------

    def test_full_readiness_runs_in_a_separate_root_before_the_main_run(self):
        out = self.h.out
        code = self.h.invoke("full", "--country", "FRA", "--out", str(out), "--yes")

        self.assertEqual(code, 0)
        pilot = out / "readiness"
        self.assertEqual([call.command for call in self.h.calls], ["run", "verify", "run", "verify"])
        self.assertTrue(all(call.root == pilot for call in self.h.calls[:2]))
        self.assertTrue(all(call.config == pilot / "resolved.json" for call in self.h.calls[:2]))
        self.assertTrue(all(call.root == out for call in self.h.calls[2:]))
        self.assertTrue(all(call.config == out / "resolved.json" for call in self.h.calls[2:]))
        self.assertEqual(
            storage.load(pilot / "resolved.json"),
            run_experiment.configure_readiness(deepcopy(CFG)),
        )
        self.assertEqual(storage.load(pilot / "resolved.json")["phases"]["full"]["repetitions"], 1)
        self.assertEqual(storage.load(out / "resolved.json"), CFG)
        self.assertEqual(storage.load(pilot / "verification.json"), READY_REPORT)
        self.assertEqual(storage.load(out / "verification.json"), GOOD_VERIFY)
        self.assertTrue((pilot / "pilot-inference" / "attempt-0001" / "result.json").is_file())
        self.assertTrue(storage.load(out / "launcher.json")["readiness_required"])
        self.h.check_resume.assert_called_once_with(
            CFG, expected_machine_id=CURRENT_MACHINE)
        self.h.provision.assert_called_once()

    def test_full_validated_attestation_skips_readiness(self):
        out = self.h.out
        code = self.h.invoke(
            "full", "--country", "FRA", "--out", str(out), "--validated", "--yes")

        self.assertEqual(code, 0)
        self.assertEqual([call.command for call in self.h.calls], ["run", "verify"])
        self.assertTrue(all(call.root == out for call in self.h.calls))
        self.assertFalse((out / "readiness").exists())
        self.assertFalse(storage.load(out / "launcher.json")["readiness_required"])

    def test_readiness_failure_blocks_the_main_run_and_stays_separate(self):
        cases = {
            "verify-not-ok": (
                verify_report(ok=False, report_counts=counts(complete=0),
                              inconsistencies=("bad native report",)), 1),
            "incomplete-counts": (
                verify_report(report_counts=counts(complete=1, incomplete=1)), 1),
            "low-sampling": (
                verify_report(low_sampling=({"run_id": "pilot-inference",
                                             "window_ratio": 0.1, "sample_ratio": 0.2},),
                              runs=(INFERENCE_ROW,)), 1),
            "no-valid-predictions": (verify_report(runs=(INFERENCE_ROW,)), 0),
        }
        for name, (report, valid) in cases.items():
            with self.subTest(name=name):
                self.h.reset_calls()
                self.h.clear_output()
                self.h.pilot_report = report
                self.h.pilot_valid = valid
                out = self.base / name
                code = self.h.invoke("full", "--country", "FRA", "--out", str(out), "--yes")
                pilot = out / "readiness"

                self.assertEqual(code, 1)
                self.assertEqual([call.command for call in self.h.calls], ["run", "verify"])
                self.assertTrue(all(call.root == pilot for call in self.h.calls))
                self.assertTrue(all(call.config == pilot / "resolved.json"
                                    for call in self.h.calls))
                self.assertEqual(storage.load(pilot / "resolved.json"),
                                 run_experiment.configure_readiness(deepcopy(CFG)))
                self.assertEqual(storage.load(pilot / "verification.json"), report)
                self.assertEqual(storage.load(out / "resolved.json"), CFG)
                self.assertFalse((out / "plan.json").exists())
                self.assertFalse((out / "verification.json").exists())
                self.assertTrue((out / "launcher-logs" / "invocation-0001" /
                                 "readiness-run.json").is_file())
                self.assertTrue((out / "launcher-logs" / "invocation-0001" /
                                 "readiness-verify.json").is_file())
                self.assertIn("Full run not started: readiness failed or was interrupted.",
                              self.h.stdout.getvalue())
                self.h.render.assert_not_called()


class FakeProcess:
    def __init__(self, code):
        self.code = code
        self.waited = 0
        self.signals = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def wait(self):
        self.waited += 1
        return self.code

    def send_signal(self, signum):
        self.signals.append(signum)


class RunCommandTests(unittest.TestCase):
    """The subprocess boundary is faked; no benchmark client is executed."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "run"
        self.root.mkdir()
        self.config = self.root / "resolved.json"
        self.config.write_text("{}\n", encoding="utf-8")
        self.logs = self.base / "logs"
        self.logs.mkdir()
        self.client = Path("/opt/fake/bin/python")

    def fake_subprocess(self, *, run=None, popen=None):
        return types.SimpleNamespace(
            run=run or mock.Mock(),
            Popen=popen or mock.Mock(),
            SubprocessError=subprocess.SubprocessError,
        )

    def test_status_command_omits_config_and_appends_flags(self):
        run = mock.Mock(return_value=subprocess.CompletedProcess([], 0))
        log = self.logs / "status.json"

        with mock.patch.object(run_experiment, "subprocess", self.fake_subprocess(run=run)):
            code = run_experiment.run_command(
                self.client, self.root, self.config, "status", log, "--retry-failed")

        self.assertEqual(code, 0)
        args, kwargs = run.call_args
        self.assertEqual(args[0], [
            str(self.client), "-u", "benchmark.py", "status", "--phase", "full",
            "--out", str(self.root), "--retry-failed",
        ])
        self.assertEqual(kwargs["cwd"], run_experiment.ROOT)
        self.assertFalse(kwargs["check"])
        self.assertEqual(kwargs["stdout"].name, str(log))

    def test_verify_command_includes_the_frozen_config(self):
        run = mock.Mock(return_value=subprocess.CompletedProcess([], 1))
        log = self.logs / "verify.json"

        with mock.patch.object(run_experiment, "subprocess", self.fake_subprocess(run=run)):
            code = run_experiment.run_command(
                self.client, self.root, self.config, "verify", log)

        self.assertEqual(code, 1)
        args, _ = run.call_args
        self.assertEqual(args[0], [
            str(self.client), "-u", "benchmark.py", "verify", "--phase", "full",
            "--out", str(self.root), "--config", str(self.config),
        ])

    def test_run_uses_an_isolated_session_and_propagates_the_exit_code(self):
        process = FakeProcess(130)
        popen = mock.Mock(return_value=process)
        log = self.logs / "run.json"

        with mock.patch.object(run_experiment, "subprocess", self.fake_subprocess(popen=popen)):
            code = run_experiment.run_command(
                self.client, self.root, self.config, "run", log,
                "--progress", "--retry-skipped")

        self.assertEqual(code, 130)
        self.assertEqual(process.waited, 1)
        args, kwargs = popen.call_args
        self.assertEqual(args[0], [
            str(self.client), "-u", "benchmark.py", "run", "--phase", "full",
            "--out", str(self.root), "--config", str(self.config),
            "--progress", "--retry-skipped",
        ])
        self.assertEqual(kwargs["cwd"], run_experiment.ROOT)
        self.assertTrue(kwargs["start_new_session"])
        self.assertEqual(kwargs["stdout"].name, str(log))

    def test_invocation_log_is_never_reused(self):
        run = mock.Mock(return_value=subprocess.CompletedProcess([], 0))
        log = self.logs / "status.json"

        with mock.patch.object(run_experiment, "subprocess", self.fake_subprocess(run=run)):
            self.assertEqual(run_experiment.run_command(
                self.client, self.root, self.config, "status", log), 0)
            with self.assertRaises(FileExistsError):
                run_experiment.run_command(
                    self.client, self.root, self.config, "status", log)

        self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
