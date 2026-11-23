from copy import deepcopy
import json
from pathlib import Path

import pytest

import configuration
import dataset
from storage import load


RAW_KEYS = {
    "phase", "dataset", "models", "batch_sizes", "seed", "sample_size", "warmup_rows",
    "repetitions", "passes", "energy", "device_idle_s", "loaded_idle_s",
    "loaded_idle_adapters", "measure_training", "settle_s",
}


def document():
    return load(configuration.DEFAULT_CONFIG)


def write_document(tmp_path, value):
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf-8")
    return path


def test_gpu_pilot_contract():
    cfg = configuration.configure("gpu_pilot")
    assert set(cfg) == RAW_KEYS
    assert list(cfg["models"]) == ["qwen2b", "joey1"]
    assert (cfg["sample_size"], cfg["warmup_rows"], cfg["seed"], cfg["repetitions"]) == (49, 17, 1729, 1)
    assert cfg["batch_sizes"] == [1, 16]
    assert cfg["passes"] == {"qwen2b": {"1": 3, "16": 2}, "joey1": {"1": 3}}
    assert cfg["energy"] == {"country_iso_code": "FRA", "gpu_ids": [0], "sample_secs": 1, "min_samples": 5}
    assert (cfg["device_idle_s"], cfg["loaded_idle_s"], cfg["settle_s"]) == (8, 12, 5)
    assert cfg["measure_training"] is False


def test_cpu_pilot_contract_and_sampling_workload():
    cfg = configuration.configure("cpu_pilot")
    assert list(cfg["models"]) == ["svm", "nb", "fasttext"]
    assert cfg["sample_size"] == 65 and cfg["repetitions"] == 1
    assert cfg["passes"]["svm"] == {"1": 450, "16": 4000}
    assert cfg["passes"]["nb"] == {"1": 450, "16": 4000}
    assert cfg["passes"]["fasttext"] == {"1": 1000, "16": 1000}
    assert cfg["measure_training"] is True
    for model in cfg["models"].values():
        assert model["device"] == "cpu"


def test_full_contract_all_nine_and_qev_width16_remains_requested():
    cfg = configuration.configure("full")
    expected = ["qwen2b", "joey1", "qwen9b", "elephant1", "qwen27b", "mammoth1", "svm", "nb", "fasttext"]
    assert list(cfg["models"]) == expected
    assert (cfg["sample_size"], cfg["warmup_rows"], cfg["seed"], cfg["repetitions"]) == (257, 17, 1729, 3)
    assert cfg["batch_sizes"] == [1, 16]
    assert cfg["passes"]["qwen2b"] == {"1": 2, "16": 4}
    assert cfg["passes"]["qwen9b"] == {"1": 2, "16": 4}
    assert cfg["passes"]["qwen27b"] == {"1": 2, "16": 4}
    assert cfg["passes"]["svm"] == {"1": 200, "16": 1200}
    assert cfg["passes"]["nb"] == {"1": 200, "16": 1200}
    for model_id in ("joey1", "elephant1", "mammoth1"):
        assert cfg["models"][model_id]["adapter"] == "qev"
        assert cfg["passes"][model_id] == {"1": 1}
    assert cfg["measure_training"] is True


def test_gpu_preflight_fields_and_exact_weight_totals():
    cfg = configuration.configure("full")
    expected = {
        "qwen2b": 4_548_221_488,
        "joey1": 5_443_677_680,
        "qwen9b": 19_306_310_880,
        "elephant1": 18_819_722_392,
        "qwen27b": 55_563_022_432,
        "mammoth1": 54_713_608_144,
    }
    for model_id, total in expected.items():
        spec = cfg["models"][model_id]
        assert spec["device"] == "cuda"
        safety = spec["engine_safety"]
        assert sum(safety["weight_shards_bytes"].values()) == total
        assert safety["reserve_mib"] == (3500 if spec["adapter"] == "vllm" else 2500)
        assert safety["gpu_index"] == 0
        assert len(spec["revision"]) == 40


def test_runtime_paths_stay_relative_and_deterministic():
    cfg = configuration.configure("full")
    for spec in cfg["models"].values():
        if "local_snapshot" in spec:
            assert not Path(spec["local_snapshot"]).is_absolute()
        if "python" in spec:
            assert not Path(spec["python"]).is_absolute()
        if spec["adapter"] == "qev":
            assert not Path(spec["params"]["python"]).is_absolute()
            assert spec["params"]["device"] == "cuda:0"
            assert spec["params"]["head_dtype"] == "float32"


def test_returned_configuration_is_detached():
    first = configuration.configure("gpu_pilot")
    first["dataset"]["path"] = "changed"
    first["models"]["qwen2b"]["params"]["max_tokens"] = 99
    first["passes"]["qwen2b"]["1"] = 99
    second = configuration.configure("gpu_pilot")
    assert second["dataset"]["path"] == "data/tweeteval"
    assert second["models"]["qwen2b"]["params"]["max_tokens"] == 8
    assert second["passes"]["qwen2b"]["1"] == 3


def test_dataset_selection_replays_exact_test_and_warmup_ids():
    cfg = configuration.configure("full")
    kwargs = {"sample_size": cfg["sample_size"], "warmup_rows": cfg["warmup_rows"], "seed": cfg["seed"]}
    first, *_ = dataset.configure(cfg["dataset"], **kwargs)
    second, *_ = dataset.configure(cfg["dataset"], **kwargs)
    assert first == second
    assert len(first["test_ids"]) == 257 and len(first["warmup_val_ids"]) == 17
    assert first["test_ids"] == sorted(set(first["test_ids"]))
    assert first["warmup_val_ids"] == sorted(set(first["warmup_val_ids"]))
    assert sum(first["test_support"].values()) == 257


def test_unknown_phase_is_rejected():
    with pytest.raises(ValueError, match="unknown phase"):
        configuration.configure("not-a-phase")


@pytest.mark.parametrize("mutation, match", [
    (lambda d: d.update(extra=True), "unknown keys"),
    (lambda d: d["phases"]["full"].update(extra=True), "unknown keys"),
    (lambda d: d["phases"]["full"]["model_ids"].append("missing"), "unknown model"),
    (lambda d: d["phases"]["full"]["model_ids"].append("svm"), "unique list"),
    (lambda d: d["phases"]["full"].update(measure_training=1), "must be boolean"),
    (lambda d: d["phases"]["full"]["loaded_idle_adapters"].append("missing"), "loaded_idle_adapters"),
])
def test_phase_validation(tmp_path, mutation, match):
    value = document()
    mutation(value)
    with pytest.raises(ValueError, match=match):
        configuration.configure("full", write_document(tmp_path, value))


@pytest.mark.parametrize("model_id,width,count", [
    ("qwen2b", "16", 0),
    ("svm", "1", 1000000),
])
def test_pass_counts_are_positive_and_memory_bounded(tmp_path, model_id, width, count):
    value = document()
    value["phases"]["full"]["passes"][model_id][width] = count
    with pytest.raises(ValueError, match="integer|in-memory prediction bound"):
        configuration.configure("full", write_document(tmp_path, value))


def test_every_supported_requested_width_needs_passes(tmp_path):
    value = document()
    del value["phases"]["full"]["passes"]["svm"]["16"]
    with pytest.raises(ValueError, match="exactly cover supported requested widths"):
        configuration.configure("full", write_document(tmp_path, value))


def test_qev_cannot_be_given_fake_independent_tweet_batching(tmp_path):
    value = document()
    value["phases"]["full"]["passes"]["joey1"]["16"] = 1
    with pytest.raises(ValueError, match="exactly cover supported requested widths"):
        configuration.configure("full", write_document(tmp_path, value))


@pytest.mark.parametrize("mutation, match", [
    (lambda d: d["models"]["qwen2b"]["params"].update(quantization="awq"), "quantization"),
    (lambda d: d["models"]["qwen2b"].update(revision="0" * 40), "manifest pin"),
    (lambda d: d["models"]["qwen2b"]["engine_safety"]["weight_shards_bytes"].update({"model.safetensors-00001-of-00001.safetensors": 1}), "manifest pin"),
    (lambda d: d["models"]["qwen2b"].update(local_snapshot="/tmp/model"), "project-relative"),
    (lambda d: d["models"]["qwen2b"].update(device="cpu"), "must be cuda"),
    (lambda d: d["models"]["joey1"]["params"].update(device="cuda"), "must remain 'cuda:0'"),
    (lambda d: d["models"]["joey1"]["params"].update(head_dtype="bfloat16"), "head_dtype"),
    (lambda d: d["models"]["qwen2b"]["engine_safety"].update(reserve_mib=1), "reserve_mib"),
    (lambda d: d["models"]["fasttext"]["params"]["model"].update(thread=2), "thread=1"),
])
def test_model_pins_cannot_be_silently_weakened(tmp_path, mutation, match):
    value = document()
    mutation(value)
    with pytest.raises(ValueError, match=match):
        configuration.configure("full", write_document(tmp_path, value))


def test_duplicate_json_keys_are_rejected(tmp_path):
    text = configuration.DEFAULT_CONFIG.read_text(encoding="utf-8")
    text = text.replace('"seed": 1729,', '"seed": 1729, "seed": 1730,', 1)
    path = tmp_path / "experiment.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        configuration.configure("full", path)


def test_nonfinite_json_is_rejected(tmp_path):
    text = configuration.DEFAULT_CONFIG.read_text(encoding="utf-8").replace('"sample_secs": 1', '"sample_secs": NaN', 1)
    path = tmp_path / "experiment.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        configuration.configure("full", path)


def test_configuration_does_no_runtime_or_network_work(monkeypatch):
    # A failure here would expose accidental subprocess/network setup during load.
    import socket
    import subprocess

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("subprocess")))
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    assert configuration.configure("full")["sample_size"] == 257
