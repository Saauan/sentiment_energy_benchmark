"""Offline tests for opt-in progress streaming and planned-width selection.

No adapters, GPUs, model downloads, or real timers start; the energy meter,
hardware probes, and engine interfaces are faked exactly as in the existing
orchestration tests.
"""
from __future__ import annotations

from contextlib import nullcontext, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import shlex
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


def config(adapter="vllm", *, batches=(1, 4), training=False, passes=1):
    model = "model"
    per_width = {str(batch): passes for batch in batches}
    if adapter == "qev":
        per_width = {str(batch): passes for batch in batches if batch == 1}
    return {
        "phase": "synthetic", "execution_protocol": "engine-session-v1",
        "dataset": {"adapter": "line_pairs", "path": "unused"},
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
        self.widths = []
        self.closed = False

    def setup(self, train, validation):
        self.setup_calls += 1
        if self.setup_error:
            raise self.setup_error
        return {"adapter": "fake", "train_rows": len(train), "validation_rows": len(validation)}

    def predict(self, texts):
        if self.predict_error:
            raise self.predict_error
        labels = {"bad": "negative", "good": "positive", "ordinary": "neutral", "warm": "neutral"}
        return [{"label": labels.get(text, "neutral"), "raw": {"text": text}, "latency_s": 0.001}
                for text in texts]

    def set_submission_width(self, width):
        self.widths.append(width)
        return {"submission_width": width, "server_restarted": False}

    def close(self):
        self.closed = True


class FakeMeter:
    """Synthetic meter that flags any progress byte written inside a capture window."""

    calls = []
    violations = []
    probe = None

    def __init__(self, out, task, cfg):
        self.out = Path(out)
        self.out.mkdir()
        self.task = task

    def capture(self, work):
        before = type(self).probe() if type(self).probe is not None else None
        result = work()
        if before is not None and type(self).probe() != before:
            type(self).violations.append(self.task["taskStage"])
        number = len(type(self).calls) + 1
        window = 6.0
        native = {
            "run_id": f"native-{number}", "energy_kwh": 0.001 * number,
            "duration_s": window, "schema_pin": "synthetic",
            "source_units": {"energy": "kWh", "duration": "s"},
            "count": self.task["nbRequest"], "cpu_energy_kwh": 0.0005 * number,
            "gpu_energy_kwh": None, "ram_energy_kwh": 0.0005 * number,
            "source_signature": "same-sources", "sources": [],
            "native_report": "synthetic.json", "scheduled_power_samples": 6,
        }
        storage.atomic_json(self.out / "native.json", {"task": self.task, "native": native})
        type(self).calls.append(self)
        return result, window, native


def parse_progress(text: str) -> list[tuple[str, dict[str, str]]]:
    events = []
    for line in text.splitlines():
        if not line:
            continue
        if not line.startswith(benchmark.PROGRESS_PREFIX + " "):
            raise AssertionError(f"unexpected stderr line: {line!r}")
        tokens = shlex.split(line)
        if tokens[:2] != ["benchmark", "progress"]:
            raise AssertionError(f"unexpected progress prefix: {line!r}")
        events.append((tokens[2], dict(token.split("=", 1) for token in tokens[3:])))
    return events


class BenchmarkHarness(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "artifacts"
        self.host = {"host_id": "synthetic-host", "gpu": [], "gpu_compute_apps": None}
        FakeMeter.calls = []
        FakeMeter.violations = []
        FakeMeter.probe = None
        mock.patch.object(benchmark, "sleep", lambda _seconds: None).start()
        mock.patch.object(benchmark.hardware, "snapshot", lambda: dict(self.host)).start()
        mock.patch.object(benchmark.hardware, "preflight", lambda spec, host: ({"device": spec["device"]}, None)).start()
        mock.patch.object(benchmark.hardware, "looks_like_oom", lambda exc, attempt: False).start()
        mock.patch.object(benchmark.energy, "Meter", FakeMeter).start()

    def tearDown(self):
        mock.patch.stopall()
        self.temporary.cleanup()

    def data(self, cfg):
        return cfg, dataset_info(), TRAIN, VALIDATION, PANEL, WARMUP

    def plan(self, cfg):
        return benchmark.ensure_plan(self.root, cfg, dataset_info(), len(TRAIN))

    def run_with(self, cfg, plan, selected, progress=None):
        adapter = FakeAdapter()
        with mock.patch.object(benchmark, "_create_adapter", return_value=adapter):
            summary = benchmark.run_selected(self.root, plan, selected, self.data(cfg), progress=progress)
        return adapter, summary


class ProgressFormatTests(unittest.TestCase):
    def test_default_progress_is_disabled_and_silent(self):
        progress = benchmark.Progress()
        self.assertFalse(progress.enabled)
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            progress.event("session", model="model", widths=[1, 4])
        self.assertEqual(stderr.getvalue(), "")

    def test_enabled_progress_writes_one_stable_key_value_line(self):
        stream = io.StringIO()
        progress = benchmark.Progress(stream)
        self.assertTrue(progress.enabled)
        progress.event("session", widths=[1, 4], model="model", rep=0,
                       note="needs a space", empty="", missing=None)
        self.assertEqual(
            stream.getvalue(),
            'benchmark progress session widths=1,4 model=model rep=0 '
            'note="needs a space" empty=""\n',
        )
        self.assertEqual(parse_progress(stream.getvalue())[0][1],
                         {"widths": "1,4", "model": "model", "rep": "0",
                          "note": "needs a space", "empty": ""})

    def test_broken_stream_disables_reporting_without_raising(self):
        class Broken:
            def write(self, _text):
                raise OSError("stderr closed")

            def flush(self):
                raise OSError("stderr closed")

        progress = benchmark.Progress(Broken())
        progress.event("run", selected=1)
        self.assertFalse(progress.enabled)
        progress.event("run", selected=1)

    def test_disabled_progress_skips_outcome_state_scans(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = config(training=False)
            plan = benchmark.build_plan(cfg, dataset_info(), len(TRAIN))
            work = {trial["run_id"]: index for index, trial in enumerate(plan["trials"], 1)}
            with mock.patch.object(benchmark.storage, "load",
                                   side_effect=AssertionError("disabled progress scanned state")):
                counts = benchmark.Progress().report_outcomes(
                    root, plan["trials"], work, len(plan["trials"]))
        self.assertEqual(counts, {})

    def test_progress_values_never_split_a_line_for_whitespace_or_backslashes(self):
        unsafe = ["a b", "a\tb", "a\nb", "a\rb", "a\vb", "a\fb", "a\\b", 'a"b', "a=b",
                  "", "caf\u00e9 au lait", "a\u00a0b"]
        for text in unsafe:
            token = benchmark._progress_value(text)
            self.assertNotIn("\n", token)
            self.assertNotIn("\r", token)
            self.assertEqual(json.loads(token), text)
        self.assertEqual(benchmark._progress_value("plain"), "plain")
        self.assertEqual(benchmark._progress_value("caf\u00e9"), "caf\u00e9")
        reason = 'line one\nline two\t"quoted" and \\ escaped'
        stream = io.StringIO()
        benchmark.Progress(stream).event("outcome", reason=reason, model="mo d\u00e8le")
        expected = " ".join([benchmark.PROGRESS_PREFIX, "outcome",
                             "reason=" + json.dumps(reason, ensure_ascii=False),
                             "model=" + json.dumps("mo d\u00e8le", ensure_ascii=False)])
        self.assertEqual(stream.getvalue(), expected + "\n")
        self.assertEqual(len(stream.getvalue().splitlines()), 1)

    def test_report_outcomes_is_deduplicated_and_reports_location(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = config(training=False)
            plan = benchmark.build_plan(cfg, dataset_info(), len(TRAIN))
            idle, inference = plan["trials"][0], plan["trials"][-1]
            benchmark._skip(root, inference, "engine unreachable")
            stream = io.StringIO()
            progress = benchmark.Progress(stream)
            work = {trial["run_id"]: index for index, trial in enumerate(plan["trials"], 1)}
            counts = progress.report_outcomes(root, plan["trials"], work, len(plan["trials"]))
            progress.report_outcomes(root, plan["trials"], work, len(plan["trials"]))
            self.assertEqual(counts, {"pending": len(plan["trials"]) - 1, "skipped": 1})
            events = parse_progress(stream.getvalue())
            self.assertEqual(len(events), 1)
            name, fields = events[0]
            self.assertEqual(name, "outcome")
            self.assertEqual(fields["status"], "skipped")
            self.assertEqual(fields["reason"], "engine unreachable")
            self.assertEqual(fields["location"], str(root / inference["run_id"]))
            self.assertEqual(fields["kind"], "inference")
            self.assertEqual(fields["run_id"], inference["run_id"])
            self.assertEqual(fields["work"], str(work[inference["run_id"]]))
            self.assertEqual(benchmark._state_status(root, idle), "pending")


class ProgressRunTests(BenchmarkHarness):
    def test_run_without_progress_writes_nothing_to_stderr(self):
        cfg = config("svm", batches=(1,), training=False)
        plan = self.plan(cfg)
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            adapter, _ = self.run_with(cfg, plan, benchmark.select_trials(plan, self.root))
        self.assertEqual(stderr.getvalue(), "")
        self.assertTrue(adapter.closed)
        self.assertEqual(benchmark.status_report(self.root, plan)["counts"]["complete"], 2)

    def test_run_progress_covers_every_selected_trial_and_never_writes_inside_capture(self):
        cfg = config("vllm", batches=(1, 4), training=False)
        plan = self.plan(cfg)
        selected = benchmark.select_trials(plan, self.root)
        self.assertEqual(len(selected), 5)
        stream = io.StringIO()
        FakeMeter.probe = lambda: len(stream.getvalue())
        stderr, stdout = io.StringIO(), io.StringIO()
        with redirect_stderr(stderr), redirect_stdout(stdout):
            adapter, summary = self.run_with(cfg, plan, selected, progress=benchmark.Progress(stream))
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(FakeMeter.violations, [])
        self.assertFalse(summary["interrupted"])
        events = parse_progress(stream.getvalue())
        names = [name for name, _ in events]
        self.assertEqual(names[0], "session")
        self.assertEqual(names[-1], "run-done")
        session = events[0][1]
        self.assertEqual((session["model"], session["rep"], session["widths"]),
                         ("model", "0", ",".join(str(w) for w in plan["sessions"][0]["widths"])))
        self.assertEqual(session["planned"], "5")
        self.assertEqual(session["selected"], "5")
        engine = [fields for name, fields in events if name == "engine"]
        self.assertEqual([fields["state"] for fields in engine], ["start", "ready"])
        done = [fields for name, fields in events if name == "work-done"]
        self.assertEqual(sorted(int(fields["work"]) for fields in done), [1, 2, 3, 4, 5])
        self.assertTrue(all(fields["total"] == "5" and fields["status"] == "complete" for fields in done))
        self.assertEqual({fields["kind"] for fields in done},
                         {"device_idle", "loaded_idle", "inference"})
        self.assertTrue(all(fields["model"] == "model" and fields["rep"] == "0" for fields in done))
        self.assertTrue(all(fields["run_id"] in {t["run_id"] for t in selected} for fields in done))
        starts = [fields for name, fields in events if name == "work-start"]
        self.assertEqual([(fields["kind"], fields["width"]) for fields in starts],
                         [(fields["kind"], fields["width"]) for fields in done])
        finished = [fields for name, fields in events if name == "session-done"]
        self.assertEqual(len(finished), 1)
        self.assertEqual((finished[0]["outcome"], finished[0]["complete"], finished[0]["failed"]),
                         ("complete", "5", "0"))
        self.assertNotIn("finished", names)  # the JSON-summary event belongs to main()
        self.assertEqual(summary["selected"], [t["run_id"] for t in selected])

    def test_progress_reports_setup_skips_with_reason_and_location(self):
        cfg = config("vllm", batches=(1, 4), training=False)
        plan = self.plan(cfg)
        selected = benchmark.select_trials(plan, self.root)
        stream = io.StringIO()
        adapter = FakeAdapter(setup_error=Unavailable("missing runtime"))
        with mock.patch.object(benchmark, "_create_adapter", return_value=adapter):
            benchmark.run_selected(self.root, plan, selected, self.data(cfg),
                                   progress=benchmark.Progress(stream))
        events = parse_progress(stream.getvalue())
        outcomes = [fields for name, fields in events if name == "outcome"]
        self.assertEqual(len(outcomes), 4)
        self.assertEqual({fields["status"] for fields in outcomes}, {"skipped"})
        skipped_ids = {t["run_id"] for t in selected if t["kind"] != "device_idle"}
        self.assertEqual({fields["run_id"] for fields in outcomes}, skipped_ids)
        for fields in outcomes:
            self.assertEqual(fields["location"], str(self.root / fields["run_id"]))
            self.assertIn("missing runtime", fields["reason"])
            self.assertEqual(fields["active_attempt"], "attempt-0001")
        self.assertEqual([name for name, _ in events].count("session-done"), 1)

    def test_progress_reports_engine_failure_then_skipped_remainder(self):
        cfg = config("vllm", batches=(1, 4), training=False)
        plan = self.plan(cfg)
        selected = benchmark.select_trials(plan, self.root)
        first_runnable = next(t for t in selected if t["kind"] != "device_idle")
        stream = io.StringIO()
        adapter = FakeAdapter(setup_error=RuntimeError("engine disconnected"))
        with mock.patch.object(benchmark, "_create_adapter", return_value=adapter):
            benchmark.run_selected(self.root, plan, selected, self.data(cfg),
                                   progress=benchmark.Progress(stream))
        outcomes = [fields for name, fields in parse_progress(stream.getvalue()) if name == "outcome"]
        self.assertEqual([fields["status"] for fields in outcomes],
                         ["failed", "skipped", "skipped", "skipped"])
        failed = outcomes[0]
        self.assertEqual(failed["kind"], first_runnable["kind"])
        self.assertEqual(failed["run_id"], first_runnable["run_id"])
        self.assertIn("engine disconnected", failed["reason"])
        self.assertEqual(failed["location"], str(self.root / first_runnable["run_id"]))
        self.assertTrue(all(fields["location"] == str(self.root / fields["run_id"]) for fields in outcomes))
        done = [fields for name, fields in parse_progress(stream.getvalue()) if name == "session-done"]
        self.assertEqual((done[0]["outcome"], done[0]["failed"], done[0]["skipped"]),
                         ("failed", "1", "3"))


class SelectWidthTests(BenchmarkHarness):
    def test_width_keeps_shared_dependencies_in_plan_order(self):
        cfg = config("vllm", batches=(1, 4, 16), training=False)
        plan = benchmark.build_plan(cfg, dataset_info(), len(TRAIN))
        order = {trial["run_id"]: index for index, trial in enumerate(plan["trials"])}
        widths = plan["sessions"][0]["widths"]
        for width in widths:
            selected = benchmark.select_trials(plan, self.root, kind="inference", width=width)
            self.assertEqual([trial["kind"] for trial in selected],
                             ["device_idle", "loaded_idle", "inference"])
            self.assertEqual([trial["batch_size"] for trial in selected], [0, width, width])
            self.assertEqual(selected[0]["run_id"], plan["trials"][0]["run_id"])
            self.assertEqual([order[trial["run_id"]] for trial in selected],
                             sorted(order[trial["run_id"]] for trial in selected))
            inference = selected[-1]
            expected = [trial["run_id"] for trial in plan["trials"]
                        if trial["kind"] in {"loaded_idle", "inference"} and trial["batch_size"] == width]
            self.assertEqual([trial["run_id"] for trial in selected[1:]], expected)
            self.assertEqual(inference["loaded_idle_run_id"], selected[1]["run_id"])
            self.assertEqual(inference["device_idle_run_id"], selected[0]["run_id"])

    def test_width_zero_selects_only_shared_calibration_and_training(self):
        cfg = config("svm", batches=(1, 4), training=True)
        plan = benchmark.build_plan(cfg, dataset_info(), len(TRAIN))
        selected = benchmark.select_trials(plan, self.root, width=0)
        self.assertEqual([trial["kind"] for trial in selected], ["device_idle", "training"])
        self.assertEqual(benchmark.select_trials(plan, self.root, kind="inference", width=0), [])
        self.assertTrue(all(trial["batch_size"] == 0 for trial in selected))

    def test_width_combines_with_model_and_kind_without_changing_plan_order(self):
        cfg = config("vllm", batches=(1, 4), training=False)
        cfg["models"]["another"] = dict(cfg["models"]["model"])
        cfg["passes"]["another"] = dict(cfg["passes"]["model"])
        cfg["loaded_idle_adapters"].append("another")
        plan = benchmark.build_plan(cfg, dataset_info(), len(TRAIN))
        width = plan["sessions"][0]["widths"][0]
        selected = benchmark.select_trials(plan, self.root, model="another", kind="inference", width=width)
        self.assertEqual([trial["model"] for trial in selected], ["another"] * 3)
        self.assertEqual([trial["kind"] for trial in selected],
                         ["device_idle", "loaded_idle", "inference"])
        self.assertEqual(benchmark.select_trials(plan, self.root, width=99), [])
        limited = benchmark.select_trials(plan, self.root, model="another", width=width, limit_new=2)
        self.assertEqual([trial["kind"] for trial in limited], ["device_idle", "loaded_idle"])

    def test_unsupported_planned_width_is_still_selectable_for_a_readiness_probe(self):
        cfg = config("qev", batches=(1, 16), training=False)
        plan = benchmark.build_plan(cfg, dataset_info(), len(TRAIN))
        selected = benchmark.select_trials(plan, self.root, kind="inference", width=16)
        self.assertEqual([trial["kind"] for trial in selected], ["device_idle", "loaded_idle", "inference"])
        self.assertEqual([trial["passes"] for trial in selected], [0, 0, 0])

    def test_completed_shared_dependency_is_not_reselected_for_another_width(self):
        cfg = config("vllm", batches=(1, 4), training=False)
        plan = self.plan(cfg)
        first, second = plan["sessions"][0]["widths"]
        selected = benchmark.select_trials(plan, self.root, kind="inference", width=first)
        self.run_with(cfg, plan, selected)
        self.assertEqual(benchmark.select_trials(plan, self.root, width=first), [])
        remaining = benchmark.select_trials(plan, self.root, kind="inference", width=second)
        # The completed shared device idle is not reselected; the same-width
        # loaded baseline is a shared dependency of its inference trial.
        self.assertEqual([trial["kind"] for trial in remaining], ["loaded_idle", "inference"])
        self.assertTrue(all(trial["batch_size"] == second for trial in remaining))


class CliProgressTests(BenchmarkHarness):
    def setUp(self):
        super().setUp()
        self.cfg = config("vllm", batches=(1, 4), training=False)
        mock.patch.object(benchmark, "_configure",
                          lambda phase, config_path: self.data(self.cfg)).start()
        mock.patch.object(benchmark, "_create_adapter",
                          side_effect=lambda spec, path: FakeAdapter()).start()
        mock.patch.object(benchmark, "_signal_handlers", lambda: nullcontext()).start()
        mock.patch.object(benchmark.storage, "measurement_lock",
                          lambda path: nullcontext()).start()

    def invoke(self, root, *extra, err_stream=None):
        stdout, stderr = io.StringIO(), err_stream if err_stream is not None else io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = benchmark.main(["run", "--phase", "synthetic", "--out", str(root), *extra])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_progress_keeps_json_stdout_identical_and_reports_completion(self):
        plain = Path(self.temporary.name) / "plain"
        streamed = Path(self.temporary.name) / "streamed"
        code_plain, stdout_plain, stderr_plain = self.invoke(plain)
        code_streamed, stdout_streamed, stderr_streamed = self.invoke(streamed, "--progress")
        self.assertEqual((code_plain, code_streamed), (0, 0))
        self.assertEqual(stderr_plain, "")
        summary_plain = json.loads(stdout_plain)
        summary_streamed = json.loads(stdout_streamed)
        self.assertEqual(summary_streamed.pop("root"), str(streamed))
        summary_plain.pop("root")
        self.assertEqual(summary_streamed, summary_plain)
        events = parse_progress(stderr_streamed)
        self.assertEqual(events[0][0], "run")
        self.assertEqual((events[0][1]["planned"], events[0][1]["selected"]), ("5", "5"))
        finished = events[-1]
        self.assertEqual(finished[0], "finished")
        self.assertEqual((finished[1]["complete"], finished[1]["failed"], finished[1]["pending"]),
                         ("5", "0", "0"))
        names = [name for name, _ in events]
        self.assertIn("session", names)
        self.assertIn("work-start", names)
        self.assertIn("work-done", names)
        self.assertIn("run-done", names)

    def test_width_flag_runs_only_the_selected_planned_width(self):
        plan = benchmark.build_plan(self.cfg, dataset_info(), len(TRAIN))
        width, other = plan["sessions"][0]["widths"]
        stderr = io.StringIO()
        code, stdout, _ = self.invoke(self.root, "--progress", "--width", str(width),
                                      "--kind", "inference", err_stream=stderr)
        self.assertEqual(code, 0)
        summary = json.loads(stdout)
        self.assertEqual(summary["counts"]["complete"], 3)
        self.assertEqual(summary["counts"]["pending"], 2)
        self.assertEqual(len(summary["selected"]), 3)
        events = parse_progress(stderr.getvalue())
        done = [fields for name, fields in events if name == "work-done"]
        self.assertEqual({fields["width"] for fields in done}, {"0", str(width)})
        self.assertNotIn(str(other), {fields["width"] for fields in done})
        completed = {trial["run_id"]: trial for trial in plan["trials"]
                     if benchmark._state_status(self.root, trial) == "complete"}
        self.assertEqual({trial["batch_size"] for trial in completed.values()}, {0, width})

    def test_width_zero_flag_runs_shared_calibration_then_training(self):
        cfg = config("svm", batches=(1, 4), training=True)
        self.cfg = cfg
        code, stdout, stderr = self.invoke(self.root, "--progress", "--width", "0")
        self.assertEqual(code, 0)
        summary = json.loads(stdout)
        self.assertEqual(summary["counts"]["complete"], 2)
        self.assertEqual(summary["counts"]["pending"], 2)
        done = [fields for name, fields in parse_progress(stderr) if name == "work-done"]
        self.assertEqual([fields["kind"] for fields in done], ["device_idle", "training"])
        self.assertTrue(all(fields["width"] == "0" for fields in done))

    def test_width_flag_rejects_negative_values(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.invoke(self.root, "--width", "-1")


if __name__ == "__main__":
    unittest.main()
