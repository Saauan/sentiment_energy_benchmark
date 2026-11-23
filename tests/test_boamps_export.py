"""Exercise the pinned BoAmps mapper without starting a CodeCarbon tracker."""
from types import SimpleNamespace
from uuid import uuid4

import pytest

import meter_worker
from boamps_export import TrackingBoAmpsOutput, tracking_modes
from energy import task_context, validator
from storage import load


def gpu_source(indices, *backends):
    return {"class": "GPU", "monitored_gpu_indices": indices,
            "devices": [{"index": i, "backend": backend} for i, backend in enumerate(backends)]}


@pytest.mark.parametrize("mode,expected", [("intel_rapl", "rapl"), ("cpu_load", "cpu_load"),
                                           ("constant", "constant"), (None, None)])
def test_cpu_mode_from_monitoring_source(mode, expected):
    assert tracking_modes([{"class": "CPU", "mode": mode}]) == (expected, None)


@pytest.mark.parametrize("indices,backends,expected", [
    ([0], ["NvidiaGPUDevice", "AMDGPUDevice"], "nvml"),
    ([1], ["NvidiaGPUDevice", "AMDGPUDevice"], "amdsmi"),
    ([0, 1], ["NvidiaGPUDevice", "AMDGPUDevice"], None),
    ([0, 1], ["NvidiaGPUDevice", "NvidiaGPUDevice"], "nvml"),
    ([0], ["UnknownDevice"], None),
    ([0, 1], ["NvidiaGPUDevice"], None),
    ([], ["NvidiaGPUDevice"], None),
])
def test_gpu_mode_only_from_selected_devices(indices, backends, expected):
    assert tracking_modes([gpu_source(indices, *backends)]) == (None, expected)


@pytest.mark.parametrize("enable,gpu_tracked", [(True, True), (True, False), (False, False)])
def test_worker_configures_boamps_report_from_runtime_sources(tmp_path, monkeypatch, enable, gpu_tracked):
    task = task_context("inference", "test", {"source_type": "public"}, 1, "test", 1)
    sources = [{"class": "CPU", "mode": "cpu_load"}]
    if gpu_tracked:
        sources.append(gpu_source([0], "NvidiaGPUDevice"))
    tracker = object()
    outputs = []

    def build_tracker(cfg, received_task, out, pin, *, boamps_output):
        assert received_task == task
        outputs.append(boamps_output)
        return tracker

    def prepare(cfg, received_task, received_tracker, schema_pin, codecarbon_pin):
        assert received_tracker is tracker
        return {"sources": sources}

    monkeypatch.setattr(meter_worker, "build_tracker", build_tracker)
    monkeypatch.setattr(meter_worker, "prepare", prepare)
    argv = ["--events-fd", "99", "--out", str(tmp_path), "--config", "{}", "--task", "{}"]
    if enable:
        argv.append("--export-tracking-modes")
    args = meter_worker.parse(argv)
    worker = meter_worker.Worker(tmp_path, {}, task, channel=None, export_tracking_modes=args.export_tracking_modes)
    worker.build()
    output, = outputs
    if not enable:
        assert output is None  # Already-running parents don't opt in.
        return
    assert isinstance(output, TrackingBoAmpsOutput)
    totals = SimpleNamespace(
        run_id=str(uuid4()), timestamp="2026-11-23T12:00:00", codecarbon_version="3.3.3",
        cpu_utilization_percent=12.0, gpu_utilization_percent=0.0, duration=4.0, energy_consumed=0.001,
        os="Linux", python_version="3.12.13", gpu_count=1, gpu_model="Detected NVIDIA GPU",
        cpu_count=16, cpu_model="CPU", ram_total_size=32.0, on_cloud="N", cloud_provider="",
        country_name="France", latitude=48.0, longitude=2.0, tracking_mode="machine",
    )
    output.out(totals, None)
    path, = tmp_path.glob("boamps_report_*.json")
    report = load(path)
    validator().validate(report)
    assert report["task"] == task
    assert report["header"]["reportId"] == totals.run_id
    measure, = report["measures"]
    assert measure["cpuTrackingMode"] == "cpu_load"
    assert measure["powerConsumption"] == 0.001
    assert measure["measurementDuration"] == 4.0
    if gpu_tracked:
        assert measure["gpuTrackingMode"] == "nvml"
    else:
        assert "gpuTrackingMode" not in measure  # Detected hardware is not a tracking backend.
