"""Offline tests for the out-of-process meter worker: fake tracker, no hardware.

The real CodeCarbon tracker is never imported here: ``FakeTracker`` mirrors only
the pinned interfaces the worker touches, so every test runs on a laptop without
NVML/RAPL and without a measurement permission.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter, sleep
from types import SimpleNamespace

import pytest

import energy
import meter_worker

ROOT = Path(__file__).resolve().parent.parent
TESTS = Path(__file__).resolve().parent
CFG = {"country_iso_code": "FRA", "gpu_ids": [0], "sample_secs": 0.1, "min_samples": 5}
TASK = {"taskStage": "inference", "taskFamily": "text classification", "nbRequest": 7,
        "algorithms": [{"algorithmName": "fake"}], "dataset": [{"dataUsage": "input", "dataType": "text"}],
        "taskDescription": "fake receipt"}

DRIVER = '''\
import sys
sys.path.insert(0, {root!r})
sys.path.insert(0, {tests!r})
import meter_worker
import test_meter_worker
factory = getattr(test_meter_worker, {factory!r})
sys.exit(meter_worker.main(sys.argv[1:], tracker_factory=factory))
'''


class FakeScheduler:
    """Stands in for CodeCarbon's PeriodicScheduler: calls function on a thread."""

    def __init__(self, function, interval=0.005):
        self.function = function
        self.interval = interval
        self.calls = 0
        self._thread = None
        self._stop = False

    def start(self):
        import threading
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop:
            sleep(self.interval)
            try:
                self.function()
            except Exception:  # noqa: BLE001 - mirrors CodeCarbon's scheduler
                pass

    def stop(self):
        self._stop = True
        if self._thread is not None:
            self._thread.join(self.interval * 4)


class _Source:
    def __init__(self, **fields):
        self.__dict__.update(fields)

    def description(self):
        return f"{type(self).__name__}()"


class FakeTracker:
    """The subset of ``OfflineEmissionsTracker`` the worker uses."""

    def __init__(self, out, task, cfg):
        self.out = Path(out)
        self.task = task
        self._tracking_mode = "machine"
        self._pue = 1.0
        self._measure_power_secs = cfg["sample_secs"]
        self._gpu_ids = cfg.get("gpu_ids")
        self._start_time = None
        self._power_measurement_count = 0
        self._scheduler = FakeScheduler(self._tick)
        self._hardware = [
            _Source(_mode="cpu_load", _tdp=96),
            _Source(_force_ram_power=None),
            _Source(devices=SimpleNamespace(devices=[SimpleNamespace(gpu_index=0)])),
        ]
        self.ensured = []

    # -- pinned private preparations ---------------------------------------
    def _ensure_hardware_ready(self):
        self.ensured.append("hardware")

    def _ensure_geo_metadata(self):
        self.ensured.append("geo")

    def _ensure_emissions_engine(self):
        self.ensured.append("emissions")

    def _ensure_cloud_conf(self):
        self.ensured.append("cloud")

    def get_detected_hardware(self):
        return {"cpu_count": 24, "ram_total_size": 62.2, "gpu_count": 1}

    # -- window ------------------------------------------------------------
    def _tick(self):
        self._power_measurement_count += 1

    def start(self):
        self._start_time = perf_counter()
        self._scheduler.start()
        self._tick()

    def stop(self):
        with (self.out / "fake_stop.log").open("a", encoding="utf-8") as stream:
            stream.write("stop\n")
        if self._scheduler is not None:
            self._scheduler.stop()
        self._start_time = None


class FakeCPUTracker(FakeTracker):
    """Hardware records with CodeCarbon's own class names, for signature parity."""

    def __init__(self, out, task, cfg):
        super().__init__(out, task, cfg)
        self._hardware = [CPU(_mode="cpu_load", _tdp=96),
                          RAM(_force_ram_power=None),
                          GPU(devices=SimpleNamespace(devices=[SimpleNamespace(gpu_index=0)]))]


class CPU(_Source):
    """Named ``CPU``: the worker branches on CodeCarbon's class names."""

    def _get_gpu_ids(self):
        raise AssertionError("CPU must not be asked for GPU ids")


class RAM(_Source):
    pass


class GPU(_Source):
    def _get_gpu_ids(self):
        return [0]


class FailingTracker(FakeTracker):
    def __init__(self, out, task, cfg):
        raise RuntimeError("synthetic prepare failure")


class HangingStartTracker(FakeTracker):
    def start(self):
        sleep(30.0)


def driver(tmp_path, factory="FakeCPUTracker"):
    path = Path(tmp_path) / f"driver_{factory}.py"
    path.write_text(DRIVER.format(root=str(ROOT), tests=str(TESTS), factory=factory), encoding="utf-8")
    return [sys.executable, str(path)]


def fake_native(out, task):
    return {"run_id": "fake-native", "energy_kwh": 0.001, "duration_s": 0.5, "schema_pin": "fake",
            "source_units": {"energy": "kWh", "duration": "s"}, "count": task["nbRequest"],
            "cpu_energy_kwh": 0.0005, "gpu_energy_kwh": None, "ram_energy_kwh": 0.0005,
            "source_signature": "fake-sources", "sources": [], "native_report": "fake.json",
            "scheduled_power_samples": 3, "out": str(out)}


@pytest.fixture
def patched_validate(monkeypatch):
    """Replace the native-artifact gate: artifacts here are synthetic by design."""
    calls = []

    def validate(out, expected):
        calls.append((Path(out), expected))
        native = fake_native(out, expected)
        native.pop("out")
        return native

    monkeypatch.setattr(energy, "validate_run", validate)
    return calls


def test_worker_handshake_monitoring_and_reap(tmp_path, patched_validate):
    out = tmp_path / "energy"
    meter = energy.Meter(out, TASK, CFG, worker=driver(tmp_path))
    try:
        worker_pid = meter._process.pid
        assert meter.metadata["expected_task"] == TASK
        assert meter.metadata["sources"] == [
            {"class": "CPU", "description": "CPU()", "mode": "cpu_load", "tdp_watts": 96},
            {"class": "RAM", "description": "RAM()", "mode": "CodeCarbon RAM estimate"},
            {"class": "GPU", "description": "GPU()", "mode": None,
             "monitored_gpu_indices": [0], "devices": [{"index": 0, "backend": "SimpleNamespace"}]},
        ]
        assert meter.metadata["meter"]["mode"] == "separate_process"
        assert meter.metadata["meter"]["protocol"] == "meter-worker/1"

        def work():
            # monitoring.json exists before the window: the old bright-line contract.
            assert json.loads((out / "monitoring.json").read_text())["expected_task"] == TASK
            sleep(0.05)
            return "result"

        result, wall, native = meter.capture(work)
    finally:
        meter.close(force=True)

    assert result == "result"
    assert 0.04 <= wall < 5.0
    assert native["count"] == TASK["nbRequest"]
    assert patched_validate and patched_validate[-1][0] == out

    metadata = json.loads((out / "monitoring.json").read_text())
    timing = metadata["timing"]
    assert timing["window_s"] >= 0.04
    assert timing["meter"]["protocol"] == "meter-worker/1"
    assert timing["meter"]["worker_window_s"] >= timing["window_s"] - 1e-6
    assert timing["scheduled_power_samples"] >= 1
    assert timing["total_power_samples"] >= timing["scheduled_power_samples"]
    ticks = timing["meter"]["ticks"]
    assert ticks["calls"] >= 2 and ticks["max_gap_s"] is not None and len(ticks["gaps_s"]) >= 1
    assert (out / "fake_stop.log").read_text().count("stop") == 1
    assert "worker ready" in (out / "worker.log").read_text()
    # Our worker only: reaped, not left behind.
    with pytest.raises(ProcessLookupError):
        os.kill(worker_pid, 0)


def test_work_failure_keeps_primary_error_and_closes_window(tmp_path, patched_validate):
    out = tmp_path / "energy"
    meter = energy.Meter(out, TASK, CFG, worker=driver(tmp_path))
    try:
        def work():
            raise ValueError("workload died")

        with pytest.raises(ValueError, match="workload died"):
            meter.capture(work)
    finally:
        meter.close(force=True)
    metadata = json.loads((out / "monitoring.json").read_text())
    assert metadata["timing"]["window_s"] >= 0.0
    assert (out / "fake_stop.log").read_text().count("stop") == 1


def test_signal_style_baseexception_still_stops(tmp_path, patched_validate):
    class SignalReceived(BaseException):
        pass

    out = tmp_path / "energy"
    meter = energy.Meter(out, TASK, CFG, worker=driver(tmp_path))
    try:
        def work():
            raise SignalReceived()

        with pytest.raises(SignalReceived):
            meter.capture(work)
    finally:
        meter.close(force=True)
    assert "scheduled_power_samples" in json.loads((out / "monitoring.json").read_text())["timing"]
    assert (out / "fake_stop.log").read_text().count("stop") == 1


def test_prepare_failure_reports_and_reaps(tmp_path):
    out = tmp_path / "energy"
    with pytest.raises(energy.MeterError, match="prepare"):
        energy.Meter(out, TASK, CFG, worker=driver(tmp_path, "FailingTracker"))
    assert not (out / "monitoring.json").exists()
    assert "synthetic prepare failure" in (out / "worker.log").read_text()


def test_begin_timeout_is_bounded_and_cleans_up(tmp_path):
    out = tmp_path / "energy"
    meter = energy.Meter(out, TASK, CFG, worker=driver(tmp_path, "HangingStartTracker"),
                         timeouts={"begin": 0.5})
    worker_pid = meter._process.pid
    started = perf_counter()
    with pytest.raises(energy.MeterError, match="timed out"):
        meter.capture(lambda: "never")
    assert perf_counter() - started < 10.0
    with pytest.raises(ProcessLookupError):
        os.kill(worker_pid, 0)


def test_worker_exits_and_flushes_on_parent_eof(tmp_path):
    """No orphan when the parent dies: EOF on the command pipe closes the window."""
    out = tmp_path / "energy"
    out.mkdir()
    read_fd, write_fd = os.pipe()
    process = subprocess.Popen(
        [*driver(tmp_path), "--events-fd", str(write_fd), "--out", str(out),
         "--config", json.dumps(CFG), "--task", json.dumps(TASK)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        pass_fds=(write_fd,), close_fds=True, start_new_session=True)
    os.close(write_fd)
    events = meter_worker.ChannelReader(read_fd)
    try:
        assert events.read(10.0)["event"] == "ready"
        process.stdin.write(b'{"cmd": "begin"}\n')
        process.stdin.flush()
        assert events.read(10.0)["event"] == "started"
        process.stdin.close()  # parent "dies"
        deadline = perf_counter() + 10.0
        while perf_counter() < deadline and process.poll() is None:
            sleep(0.05)
        assert process.poll() is not None, "worker outlived its command pipe"
        assert (out / "fake_stop.log").read_text().count("stop") == 1
    finally:
        events.close()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_capture_once_and_version_pin(tmp_path, patched_validate, monkeypatch):
    out = tmp_path / "energy"
    meter = energy.Meter(out, TASK, CFG, worker=driver(tmp_path))
    try:
        assert meter.capture(lambda: 1)[0] == 1
        with pytest.raises(energy.MeterError, match="exactly one window"):
            meter.capture(lambda: 2)
    finally:
        meter.close(force=True)

    monkeypatch.setattr(energy, "version", lambda _name: "0.0.0")
    with pytest.raises(RuntimeError, match="required"):
        energy.Meter(tmp_path / "other", TASK, CFG, worker=driver(tmp_path))


def test_record_ticks_is_optional_and_safe():
    class NoScheduler:
        pass

    assert meter_worker.record_ticks(NoScheduler()) is None

    calls = []

    class Tracker:
        _scheduler = SimpleNamespace(function=lambda: calls.append(1))

    record = meter_worker.record_ticks(Tracker())
    assert record is not None
    Tracker._scheduler.function()
    Tracker._scheduler.function()
    assert calls == [1, 1]
    assert record["calls"] == 2 and len(record["gaps_s"]) == 1
    assert meter_worker.summarize_ticks(record)["calls"] == 2
