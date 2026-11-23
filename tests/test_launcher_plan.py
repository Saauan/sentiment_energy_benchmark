"""Focused offline tests for launcher_plan host preflight and config resolution."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import pytest

import configuration
import launcher_plan as plan

# Capture the real helper before the autouse fixture replaces it for determinism.
_REAL_SNAPSHOT = plan._snapshot
_TINY = 10 ** 12
GPU0 = "0, GPU-test-0, 570.26, 12227, 11774, 8.6, NVIDIA Test GPU"
GPU_BIG = "0, GPU-big, 595.99.02, 200000, 200000, 12.0, NVIDIA Big Test GPU"
GPU_IDS = ["qwen2b", "joey1", "qwen9b", "elephant1", "qwen27b", "mammoth1"]
SELECTED = ["qwen2b", "joey1", "svm", "nb", "fasttext"]


def _platform_ok():
    return {"system": "Linux", "machine": "x86_64", "platform": "Linux-test",
            "supported": True, "required": "Linux x86_64"}


@pytest.fixture(autouse=True)
def default_host(monkeypatch):
    monkeypatch.setattr(plan, "_platform", _platform_ok)
    monkeypatch.setattr(plan, "_free_space", lambda path: (plan.ROOT, _TINY))
    monkeypatch.setattr(plan, "machine_identity", lambda: {
        "system": "Linux", "machine": "x86_64", "hostname": "test-host",
        "machine_id": "a" * 64, "source": "machine-id"})
    monkeypatch.setattr(plan, "_snapshot", lambda local, weights: {
        "path": str(local), "verified": True, "download_bytes": 0, "reason": None})
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    def no_command(args):
        raise AssertionError(f"unexpected host command: {args!r}")

    monkeypatch.setattr(plan, "_run", no_command)


def _gpu(monkeypatch, rows=GPU0, apps="", gpu_error=None, apps_error=None):
    def fake(args):
        if args == plan.GPU_QUERY:
            if gpu_error is not None:
                raise gpu_error
            return rows
        if args == plan.COMPUTE_APPS_QUERY:
            if apps_error is not None:
                raise apps_error
            return apps
        raise AssertionError(f"unexpected command: {args!r}")

    monkeypatch.setattr(plan, "_run", fake)


def _configure(document):
    fd, path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(document, stream)
        return configuration.configure("full", path)
    finally:
        os.unlink(path)


def _row(*, index=0, driver="570.26", total=12227, free=11774, cap="8.6", name="NVIDIA Test GPU"):
    return f"{index}, GPU-test-{index}, {driver}, {total}, {free}, {cap}, {name}"


def test_prepare_contract_and_hardware_filtering(monkeypatch):
    _gpu(monkeypatch)
    source = plan._load(plan.EXPERIMENT)
    document, discovery = plan.prepare("full", "FRA")

    assert plan.ROOT == plan.PLAN_ROOT and plan.ROOT.is_dir()
    assert json.dumps(document, allow_nan=False)
    assert json.dumps(discovery, allow_nan=False)
    assert discovery["mode"] == "full" and discovery["phase"] == "full"
    assert discovery["selected"] == SELECTED
    assert set(discovery["excluded"]) == {"qwen9b", "elephant1", "qwen27b", "mammoth1"}
    assert all("exceed GPU 0 total VRAM" in reason for reason in discovery["excluded"].values())
    assert document["phases"]["full"]["model_ids"] == SELECTED
    assert list(document["models"]) == SELECTED
    assert document["energy"] == {**source["energy"], "country_iso_code": "FRA", "gpu_ids": [0]}
    assert document["phases"]["full"]["execution_protocol"] == "engine-session-v1"
    assert discovery["gpu"]["status"] == "ok"
    assert discovery["gpu"]["gpu_ids"] == [0]
    assert discovery["gpu"]["uuid"] == "GPU-test-0"
    assert discovery["gpu"]["metering"]["gpu_ids"] == [0]
    assert discovery["download_bytes"] == {"qwen2b": 0, "joey1": 0}
    assert discovery["download_total_bytes"] == 0
    assert discovery["blockers"] == []
    assert discovery["config_sha256"] == plan._storage().identity(document)
    assert set(_configure(document)["models"]) == set(SELECTED)


def test_full_copies_active_phase_when_all_models_fit(monkeypatch):
    _gpu(monkeypatch, rows=GPU_BIG)
    source = plan._load(plan.EXPERIMENT)
    document, discovery = plan.prepare("full", "FRA")

    assert discovery["excluded"] == {}
    assert discovery["selected"] == source["phases"]["full"]["model_ids"]
    assert document["phases"]["full"] == source["phases"]["full"]
    assert set(discovery["download_bytes"]) == set(GPU_IDS)
    assert all(value == 0 for value in discovery["download_bytes"].values())
    assert discovery["blockers"] == []
    _configure(document)


def test_mini_contract_and_classical_protocol(monkeypatch):
    _gpu(monkeypatch)
    source = plan._load(plan.EXPERIMENT)
    document, discovery = plan.prepare("mini", "fra")
    phase = document["phases"]["full"]

    assert discovery["mode"] == "mini" and discovery["phase"] == "full"
    assert discovery["country"] == "FRA"
    assert phase["model_ids"] == discovery["selected"] == SELECTED
    assert phase["batch_sizes"] == [1]
    assert phase["sample_size"] == 9 and phase["warmup_rows"] == 3
    assert phase["repetitions"] == 1 and phase["measure_training"] is False
    assert phase["device_idle_s"] == 2 and phase["loaded_idle_s"] == 2 and phase["settle_s"] == 2
    assert phase["passes"] == {model_id: {"1": 1} for model_id in SELECTED}
    assert phase["seed"] == source["phases"]["full"]["seed"]
    assert document["energy"]["sample_secs"] == source["energy"]["sample_secs"]
    assert document["energy"]["min_samples"] == source["energy"]["min_samples"]
    for model_id in ("svm", "nb", "fasttext"):
        assert document["models"][model_id] == source["models"][model_id]
    assert document["models"]["qwen2b"]["params"] == source["models"]["qwen2b"]["params"]
    assert document["models"]["qwen2b"]["local_snapshot"] == source["models"]["qwen2b"]["local_snapshot"]
    _configure(document)


def test_engine_paths_and_legacy_snapshot(monkeypatch):
    _gpu(monkeypatch)
    document, _ = plan.prepare("full", "FRA")
    qwen2b = document["models"]["qwen2b"]
    joey = document["models"]["joey1"]

    assert qwen2b["python"] == plan.VLLM_PYTHON
    assert qwen2b["engine_safety"]["expected_python"] == plan.VLLM_PYTHON
    assert qwen2b["local_snapshot"] == "../inspect_pilot_study/.scratch/gpu/model"
    assert joey["params"]["python"] == plan.QEV_PYTHON
    assert joey["params"]["onejev_path"] == plan.QEV_ONEJEV
    assert joey["local_snapshot"] == ".scratch/models/joey-1"


def test_cpu_only_fallback_keeps_supported_gpu_meter(monkeypatch):
    _gpu(monkeypatch, rows=_row(total=1024, free=1024))
    document, discovery = plan.prepare("full", "FRA")

    assert discovery["selected"] == ["svm", "nb", "fasttext"]
    assert set(discovery["excluded"]) == set(GPU_IDS)
    assert all("exceed GPU 0 total VRAM" in reason for reason in discovery["excluded"].values())
    assert discovery["gpu"]["gpu_ids"] == [0]
    assert discovery["gpu"]["metering"]["gpu_ids"] == [0]
    assert document["energy"]["gpu_ids"] == [0]
    assert discovery["download_bytes"] == {}
    plan.check_resume(document, expected_gpu_uuid=discovery["gpu"]["uuid"])


def test_cpu_only_fallback_without_gpu(monkeypatch):
    _gpu(monkeypatch, gpu_error=plan.ProbeUnavailable("nvidia-smi is not installed"))
    document, discovery = plan.prepare("full", "FRA")

    assert discovery["selected"] == ["svm", "nb", "fasttext"]
    assert set(discovery["excluded"]) == set(GPU_IDS)
    assert all("CPU-only fallback" in reason for reason in discovery["excluded"].values())
    assert discovery["gpu"]["status"] == "absent"
    assert discovery["gpu"]["gpu_ids"] == []
    assert discovery["gpu"]["metering"]["gpu_ids"] == []
    assert discovery["gpu"]["uuid"] is None
    assert document["energy"]["gpu_ids"] == []
    assert discovery["download_bytes"] == {}
    assert discovery["blockers"] == []
    assert any("CPU-only fallback" in note for note in discovery["notes"])
    plan.check_resume(document)


@pytest.mark.parametrize(
    ("rows", "needle"),
    [
        (_row(cap="7.5"), "Ampere/BF16"),
        (_row(driver="525.60.13"), "CUDA 12.8"),
        (_row() + "\n" + _row(index=1), "exactly one visible GPU"),
        (_row(index=1), "GPU index 0"),
    ],
)
def test_unsupported_gpu_falls_back_to_cpu(monkeypatch, rows, needle):
    _gpu(monkeypatch, rows=rows)
    document, discovery = plan.prepare("full", "FRA")

    assert discovery["selected"] == ["svm", "nb", "fasttext"]
    assert discovery["gpu"]["status"] == "unsupported"
    assert discovery["gpu"]["gpu_ids"] == []
    assert document["energy"]["gpu_ids"] == []
    assert all(needle in reason for reason in discovery["excluded"].values())
    assert discovery["blockers"] == []
    plan.check_resume(document)


def test_malformed_gpu_query_fails_closed(monkeypatch):
    _gpu(monkeypatch, rows="not,a,valid,row")
    document, discovery = plan.prepare("full", "FRA")

    assert discovery["gpu"]["status"] == "unknown"
    assert discovery["selected"] == ["svm", "nb", "fasttext"]
    assert discovery["gpu"]["gpu_ids"] == []
    assert document["energy"]["gpu_ids"] == []
    assert any("could not be verified" in blocker for blocker in discovery["blockers"])
    assert any("could not be verified" in reason for reason in discovery["excluded"].values())
    with pytest.raises(plan.ResumeCheckError, match="could not be verified"):
        plan.check_resume(document)


def test_foreign_compute_blocks_without_downselecting(monkeypatch):
    _gpu(monkeypatch, apps="4242\n")
    document, discovery = plan.prepare("full", "FRA")

    assert discovery["selected"] == SELECTED
    assert set(discovery["excluded"]) == {"qwen9b", "elephant1", "qwen27b", "mammoth1"}
    assert discovery["gpu"]["gpu_ids"] == [0]
    assert any("foreign GPU compute" in blocker for blocker in discovery["blockers"])
    with pytest.raises(plan.ResumeCheckError, match="foreign GPU compute"):
        plan.check_resume(document)


def test_compute_apps_query_failure_blocks_without_downselecting(monkeypatch):
    _gpu(monkeypatch, apps_error=plan.ProbeError("synthetic apps failure"))
    document, discovery = plan.prepare("full", "FRA")

    assert discovery["selected"] == SELECTED
    assert discovery["gpu"]["gpu_ids"] == [0]
    assert any("compute-app query failed" in blocker for blocker in discovery["blockers"])
    with pytest.raises(plan.ResumeCheckError, match="compute-app query failed"):
        plan.check_resume(document)


def test_free_vram_shortage_is_blocker_not_exclusion(monkeypatch):
    _gpu(monkeypatch, rows=_row(free=1000))
    document, discovery = plan.prepare("full", "FRA")

    assert discovery["selected"] == SELECTED
    assert set(discovery["excluded"]) == {"qwen9b", "elephant1", "qwen27b", "mammoth1"}
    assert discovery["models"]["qwen2b"]["fits_total"] is True
    assert discovery["models"]["qwen2b"]["fits_free_now"] is False
    assert any("insufficient currently free VRAM" in blocker for blocker in discovery["blockers"])
    assert any("transient occupancy" in blocker for blocker in discovery["blockers"])
    with pytest.raises(plan.ResumeCheckError, match="are free"):
        plan.check_resume(document)


def test_disk_shortage_is_blocker_and_uses_snapshot_paths(monkeypatch):
    seen = []

    def free_space(path):
        seen.append(str(path))
        return Path(plan.ROOT), 1

    monkeypatch.setattr(plan, "_free_space", free_space)
    monkeypatch.setattr(plan, "_snapshot", lambda local, weights: {
        "path": str(local), "verified": False,
        "download_bytes": sum(weights.values()), "reason": "missing"})
    _gpu(monkeypatch)
    source = plan._load(plan.EXPERIMENT)
    document, discovery = plan.prepare("full", "FRA")

    expected = source["models"]["qwen2b"]["engine_safety"]["weight_shards_bytes"]
    expected_bytes = sum(expected.values())
    assert discovery["selected"] == SELECTED
    assert discovery["download_bytes"]["qwen2b"] == expected_bytes
    assert set(discovery["download_bytes"]) == {"qwen2b", "joey1"}
    assert discovery["disk"]["ok"] is False
    assert discovery["disk"]["required_bytes"] == sum(
        value + value // 20 for value in discovery["download_bytes"].values())
    assert any("insufficient disk" in blocker for blocker in discovery["blockers"])
    assert any("inspect_pilot_study/.scratch/gpu/model" in path for path in seen)
    with pytest.raises(plan.ResumeCheckError, match="insufficient disk"):
        plan.check_resume(document)


def test_download_bytes_only_for_selected_gpu_models(monkeypatch):
    calls = []

    def snapshot(local, weights):
        calls.append(local)
        return {"path": str(local), "verified": False,
                "download_bytes": sum(weights.values()), "reason": "missing"}

    monkeypatch.setattr(plan, "_snapshot", snapshot)
    _gpu(monkeypatch)
    _document, discovery = plan.prepare("full", "FRA")

    assert set(calls) == {"../inspect_pilot_study/.scratch/gpu/model", ".scratch/models/joey-1"}
    assert set(discovery["download_bytes"]) == {"qwen2b", "joey1"}
    assert all(value > 0 for value in discovery["download_bytes"].values())


def test_snapshot_verification_is_conservative(tmp_path):
    weights = {"part.safetensors": 10}

    verified = tmp_path / "verified"
    verified.mkdir()
    (verified / "config.json").write_text("{}", encoding="utf-8")
    (verified / "part.safetensors").write_bytes(b"x" * 10)
    result = _REAL_SNAPSHOT(verified, weights)
    assert result["verified"] is True and result["download_bytes"] == 0

    missing_config = tmp_path / "missing-config"
    missing_config.mkdir()
    (missing_config / "part.safetensors").write_bytes(b"x" * 10)
    result = _REAL_SNAPSHOT(missing_config, weights)
    assert result["verified"] is False and result["download_bytes"] == 10
    assert "config.json" in result["reason"]

    wrong_size = tmp_path / "wrong-size"
    wrong_size.mkdir()
    (wrong_size / "config.json").write_text("{}", encoding="utf-8")
    (wrong_size / "part.safetensors").write_bytes(b"x" * 9)
    result = _REAL_SNAPSHOT(wrong_size, weights)
    assert result["verified"] is False and result["download_bytes"] == 10
    assert "!=" in result["reason"]

    missing_shard = tmp_path / "missing-shard"
    missing_shard.mkdir()
    (missing_shard / "config.json").write_text("{}", encoding="utf-8")
    result = _REAL_SNAPSHOT(missing_shard, weights)
    assert result["verified"] is False and result["download_bytes"] == 10


def test_country_and_mode_are_explicit(monkeypatch):
    _gpu(monkeypatch)
    document, discovery = plan.prepare("MINI", "fra")
    assert discovery["mode"] == "mini" and discovery["country"] == "FRA"
    assert document["energy"]["country_iso_code"] == "FRA"

    for bad in ("", "F", "FR", "FRA1", "XXX", None, 123):
        with pytest.raises(ValueError):
            plan.prepare("full", bad)
    for bad in ("", "smoke", None, 3):
        with pytest.raises(ValueError):
            plan.prepare(bad, "FRA")


def test_manifest_pin_tamper_fails_closed(monkeypatch):
    _gpu(monkeypatch)
    manifest = plan._load(plan.MODEL_MANIFEST)
    manifest = json.loads(json.dumps(manifest))
    manifest["models"]["qwen2b"]["weight_shards_bytes"] = {
        "model.safetensors-00001-of-00001.safetensors": 1}
    monkeypatch.setattr(plan, "_load", lambda path: manifest if path == plan.MODEL_MANIFEST
                        else plan._storage().load(path))
    with pytest.raises(plan.LauncherError, match="manifest pin"):
        plan.prepare("full", "FRA")


def test_unsupported_platform_blocks_and_falls_back(monkeypatch):
    monkeypatch.setattr(plan, "_platform", lambda: {
        "system": "Darwin", "machine": "arm64", "platform": "Darwin-test",
        "supported": False, "required": "Linux x86_64"})
    document, discovery = plan.prepare("full", "FRA")

    assert discovery["selected"] == ["svm", "nb", "fasttext"]
    assert discovery["gpu"]["status"] == "unsupported"
    assert discovery["gpu"]["gpu_ids"] == []
    assert document["energy"]["gpu_ids"] == []
    assert any("unsupported" in blocker for blocker in discovery["blockers"])
    for reason in discovery["excluded"].values():
        assert "CPU-only fallback" in reason
    with pytest.raises(plan.ResumeCheckError, match="unsupported"):
        plan.check_resume(document)


def test_cuda_visible_devices_is_a_blocker(monkeypatch):
    _gpu(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    document, discovery = plan.prepare("full", "FRA")

    assert discovery["selected"] == SELECTED
    assert any("CUDA_VISIBLE_DEVICES" in blocker for blocker in discovery["blockers"])
    with pytest.raises(plan.ResumeCheckError, match="CUDA_VISIBLE_DEVICES"):
        plan.check_resume(document)


def test_check_resume_pins_machine_and_gpu_uuid(monkeypatch):
    _gpu(monkeypatch)
    document, discovery = plan.prepare("full", "FRA")
    machine_id = discovery["machine"]["machine_id"]
    gpu_uuid = discovery["gpu"]["uuid"]

    plan.check_resume(document, expected_machine_id=machine_id, expected_gpu_uuid=gpu_uuid)
    with pytest.raises(plan.ResumeCheckError, match="machine identity"):
        plan.check_resume(document, expected_machine_id="b" * 64)
    with pytest.raises(plan.ResumeCheckError, match="GPU UUID"):
        plan.check_resume(document, expected_gpu_uuid="GPU-other")


def test_check_resume_absent_gpu_and_metering_ids(monkeypatch):
    _gpu(monkeypatch)
    document, _ = plan.prepare("full", "FRA")
    _gpu(monkeypatch, gpu_error=plan.ProbeUnavailable("nvidia-smi is not installed"))
    with pytest.raises(plan.ResumeCheckError, match="no NVIDIA GPU"):
        plan.check_resume(document)

    _gpu(monkeypatch)
    bad = json.loads(json.dumps(document))
    bad["energy"]["gpu_ids"] = [1]
    with pytest.raises(plan.ResumeCheckError, match="only GPU 0"):
        plan.check_resume(bad)

    bad = json.loads(json.dumps(document))
    bad["energy"]["gpu_ids"] = []
    with pytest.raises(plan.ResumeCheckError, match="energy.gpu_ids is not \\[0\\]"):
        plan.check_resume(bad)

    with pytest.raises(plan.ResumeCheckError):
        plan.check_resume({})


def test_check_resume_accepts_missing_selected_snapshot(monkeypatch):
    _gpu(monkeypatch)
    document, _ = plan.prepare("full", "FRA")
    monkeypatch.setattr(plan, "_snapshot", lambda local, weights: {
        "path": str(local), "verified": False,
        "download_bytes": sum(weights.values()), "reason": "missing"})
    monkeypatch.setattr(plan, "_free_space", lambda path: (Path(plan.ROOT), _TINY))
    plan.check_resume(document)


def test_parsers_reject_malformed_rows_and_allow_commas_in_name():
    row = plan._gpu_rows("0, GPU-1, 570.26, 12227, 11774, 12.0, NVIDIA RTX, Laptop")[0]
    assert row["name"] == "NVIDIA RTX, Laptop"
    assert row["memory_total_bytes"] == 12227 * plan.MIB

    with pytest.raises(plan.ProbeError):
        plan._gpu_rows("0, GPU-1, 570.26, 12227, 11774, 12.0, One\n"
                       "0, GPU-2, 570.26, 12227, 11774, 12.0, Two")
    with pytest.raises(plan.ProbeError):
        plan._gpu_rows("0, GPU-1, 570.26, N/A, 11774, 12.0, One")
    with pytest.raises(plan.ProbeError):
        plan._compute_apps("not-a-pid")
    assert plan._compute_apps("12\n34\n") == [12, 34]
    assert plan._compute_apps("No running processes found\n") == []
    assert (570, 26) >= plan.MIN_CUDA_128_DRIVER
    assert (570, 25) < plan.MIN_CUDA_128_DRIVER
    assert (7, 5) < plan.MIN_COMPUTE_CAPABILITY
