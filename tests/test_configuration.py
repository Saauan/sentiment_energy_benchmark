from copy import deepcopy
from hashlib import sha256
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
PROTOCOL = configuration.EXECUTION_PROTOCOL
LEGACY_PROFILE = configuration.ROOT / "profiles" / "legacy-experiment.json"
FULL_ISOLATED_PROFILE = configuration.ROOT / "profiles" / "laptop-full-isolated-02.json"
CPU_ISOLATED_PROFILE = configuration.ROOT / "profiles" / "laptop-cpu-isolated-02.json"
WIDTH_PROFILE = configuration.ROOT / "profiles" / "laptop-gpu-widths.json"
WIDE_WIDTHS = (1, 4, 16, 64, 256)
ACTIVE_WIDTHS = [1, 16, 64]
ALL_MODELS = ["qwen2b", "joey1", "qwen9b", "elephant1", "qwen27b", "mammoth1", "svm", "nb", "fasttext"]
QEV_SIBLINGS = ("joey1", "elephant1", "mammoth1")


def width_document():
    return load(WIDTH_PROFILE)


def legacy_document():
    return load(LEGACY_PROFILE)


def document():
    return load(configuration.DEFAULT_CONFIG)


def write_document(tmp_path, value):
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf-8")
    return path


def test_active_config_is_one_full_phase_with_the_engine_session_protocol():
    active = document()
    assert list(active["phases"]) == ["full"]
    cfg = configuration.configure("full")
    assert set(cfg) == RAW_KEYS | {"execution_protocol"}
    assert cfg["execution_protocol"] == PROTOCOL == "engine-session-v1"
    assert cfg["phase"] == "full"
    assert (cfg["sample_size"], cfg["warmup_rows"], cfg["seed"], cfg["repetitions"]) == (257, 17, 1729, 3)
    assert cfg["batch_sizes"] == ACTIVE_WIDTHS
    # The active stage reuses the reported laptop full panel: isolated 0.2 s
    # tracker cadence with five scheduled samples, not the pilot cadence.
    assert cfg["energy"] == {"country_iso_code": "FRA", "gpu_ids": [0], "sample_secs": 0.2, "min_samples": 5}
    assert (cfg["device_idle_s"], cfg["loaded_idle_s"], cfg["settle_s"]) == (8, 12, 5)
    assert cfg["loaded_idle_adapters"] == ALL_MODELS
    assert cfg["measure_training"] is True
    # There is no second runnable phase: the legacy definitions live only in the
    # archived profile, which is resolved for verification and refused by the runner.
    with pytest.raises(ValueError, match="unknown phase"):
        configuration.configure("gpu_pilot")
    with pytest.raises(ValueError, match="unknown phase"):
        configuration.configure("cpu_pilot")
    with pytest.raises(ValueError, match="unknown phase"):
        configuration.configure("gpu_widths")


def test_full_contract_all_nine_widths_and_pass_targets():
    cfg = configuration.configure("full")
    assert list(cfg["models"]) == ALL_MODELS
    assert cfg["passes"] == {
        "qwen2b": {"1": 2, "16": 4, "64": 4},
        "joey1": {"1": 1, "16": 4, "64": 4},
        "qwen9b": {"1": 2, "16": 4, "64": 4},
        "elephant1": {"1": 1, "16": 4, "64": 4},
        "qwen27b": {"1": 2, "16": 4, "64": 4},
        "mammoth1": {"1": 1, "16": 4, "64": 4},
        "svm": {"1": 200, "16": 1200, "64": 1200},
        "nb": {"1": 200, "16": 1200, "64": 1200},
        "fasttext": {"1": 1750, "16": 1750, "64": 1750},
    }
    for model_id in ALL_MODELS:
        assert set(cfg["passes"][model_id]) == {str(width) for width in ACTIVE_WIDTHS}
        for count in cfg["passes"][model_id].values():
            assert cfg["sample_size"] * count <= configuration.MAX_PREDICTIONS_PER_CONDITION


def test_all_pinned_qev_siblings_are_packed_at_every_width():
    cfg = configuration.configure("full")
    for model_id in QEV_SIBLINGS:
        spec = cfg["models"][model_id]
        assert spec["adapter"] == "qev"
        assert spec["params"]["request_mode"] == "packed"
        assert configuration.supported_batches(spec) == configuration.REQUESTED_WIDTHS
        assert set(ACTIVE_WIDTHS) <= set(configuration.supported_batches(spec))
    for model_id in ("qwen2b", "qwen9b", "qwen27b"):
        assert configuration.supported_batches(cfg["models"][model_id]) == configuration.REQUESTED_WIDTHS
        assert "request_mode" not in cfg["models"][model_id]["params"]


def test_legacy_gpu_pilot_contract_is_verification_only():
    cfg = configuration.configure("gpu_pilot", LEGACY_PROFILE)
    assert set(cfg) == RAW_KEYS
    assert "execution_protocol" not in cfg
    assert list(cfg["models"]) == ["qwen2b", "joey1"]
    assert (cfg["sample_size"], cfg["warmup_rows"], cfg["seed"], cfg["repetitions"]) == (49, 17, 1729, 1)
    assert cfg["batch_sizes"] == [1, 16]
    assert cfg["passes"] == {"qwen2b": {"1": 3, "16": 2}, "joey1": {"1": 3}}
    assert cfg["energy"] == {"country_iso_code": "FRA", "gpu_ids": [0], "sample_secs": 1, "min_samples": 5}
    assert (cfg["device_idle_s"], cfg["loaded_idle_s"], cfg["settle_s"]) == (8, 12, 5)
    assert cfg["measure_training"] is False


def test_legacy_cpu_pilot_contract_and_sampling_workload():
    cfg = configuration.configure("cpu_pilot", LEGACY_PROFILE)
    assert set(cfg) == RAW_KEYS
    assert "execution_protocol" not in cfg
    assert list(cfg["models"]) == ["svm", "nb", "fasttext"]
    assert cfg["sample_size"] == 65 and cfg["repetitions"] == 1
    assert cfg["passes"]["svm"] == {"1": 450, "16": 4000}
    assert cfg["passes"]["nb"] == {"1": 450, "16": 4000}
    assert cfg["passes"]["fasttext"] == {"1": 1000, "16": 1000}
    assert cfg["measure_training"] is True
    for model in cfg["models"].values():
        assert model["device"] == "cpu"


def test_legacy_full_phase_still_resolves_for_verification():
    cfg = configuration.configure("full", LEGACY_PROFILE)
    assert set(cfg) == RAW_KEYS
    assert "execution_protocol" not in cfg
    assert cfg["batch_sizes"] == [1, 16]
    assert cfg["passes"]["qwen2b"] == {"1": 2, "16": 4}
    assert cfg["passes"]["joey1"] == {"1": 1}
    assert cfg["passes"]["fasttext"] == {"1": 1000, "16": 1000}
    assert cfg["energy"]["sample_secs"] == 1
    for model_id in QEV_SIBLINGS:
        assert "request_mode" not in cfg["models"][model_id]["params"]
        assert configuration.supported_batches(cfg["models"][model_id]) == (1,)


def test_reported_laptop_full_profile_keeps_its_isolated_panel():
    cfg = configuration.configure("full", FULL_ISOLATED_PROFILE)
    assert set(cfg) == RAW_KEYS
    assert cfg["energy"]["sample_secs"] == 0.2
    assert cfg["passes"]["fasttext"] == {"1": 1750, "16": 1750}
    assert cfg["passes"]["svm"] == {"1": 200, "16": 1200}
    assert cfg["batch_sizes"] == [1, 16]


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
    first = configuration.configure("full")
    first["dataset"]["path"] = "changed"
    first["models"]["qwen2b"]["params"]["max_tokens"] = 99
    first["passes"]["qwen2b"]["1"] = 99
    first["execution_protocol"] = "changed"
    second = configuration.configure("full")
    assert second["dataset"]["path"] == "data/tweeteval"
    assert second["models"]["qwen2b"]["params"]["max_tokens"] == 8
    assert second["passes"]["qwen2b"]["1"] == 2
    assert second["execution_protocol"] == PROTOCOL


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
    ("qwen2b", "64", 0),
    ("svm", "64", 1000000),
    ("fasttext", "1", 0),
])
def test_pass_counts_are_positive_and_memory_bounded(tmp_path, model_id, width, count):
    value = document()
    value["phases"]["full"]["passes"][model_id][width] = count
    with pytest.raises(ValueError, match="integer|in-memory prediction bound"):
        configuration.configure("full", write_document(tmp_path, value))


def test_every_supported_requested_width_needs_passes(tmp_path):
    value = document()
    del value["phases"]["full"]["passes"]["svm"]["64"]
    with pytest.raises(ValueError, match="exactly cover supported requested widths"):
        configuration.configure("full", write_document(tmp_path, value))


def test_qev_cannot_be_given_fake_independent_tweet_batching(tmp_path):
    # Without the explicit packed opt-in the wider widths stay requested but
    # unsupported, so wider pass entries no longer cover the capability exactly.
    value = document()
    del value["models"]["joey1"]["params"]["request_mode"]
    with pytest.raises(ValueError, match="exactly cover supported requested widths"):
        configuration.configure("full", write_document(tmp_path, value))


def test_unpacked_qev_keeps_wider_widths_visible_without_passes(tmp_path):
    value = document()
    del value["models"]["joey1"]["params"]["request_mode"]
    value["phases"]["full"]["passes"]["joey1"] = {"1": 1}
    cfg = configuration.configure("full", write_document(tmp_path, value))
    assert cfg["batch_sizes"] == ACTIVE_WIDTHS
    assert cfg["passes"]["joey1"] == {"1": 1}
    assert configuration.supported_batches(cfg["models"]["joey1"]) == (1,)


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
    text = configuration.DEFAULT_CONFIG.read_text(encoding="utf-8").replace('"sample_secs": 0.2', '"sample_secs": NaN', 1)
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
    assert configuration.configure("full", LEGACY_PROFILE)["sample_size"] == 257


def test_supported_batches_policy_is_coarse_and_deterministic():
    cfg = configuration.configure("full")
    for _ in range(2):
        for model_id, spec in cfg["models"].items():
            value = configuration.supported_batches(spec)
            if spec["adapter"] in configuration.CLASSICAL:
                assert value is None
            else:
                assert isinstance(value, tuple) and value
    assert configuration.supported_batches(cfg["models"]["qwen2b"]) == WIDE_WIDTHS
    # Every pinned QEV sibling opts into packed requests in the active stage.
    for model_id in QEV_SIBLINGS:
        assert configuration.supported_batches(cfg["models"][model_id]) == WIDE_WIDTHS
    with pytest.raises(ValueError, match="unsupported adapter"):
        configuration.supported_batches({"adapter": "mystery"})
    with pytest.raises(ValueError, match="must be an object"):
        configuration.supported_batches(None)


def test_execution_protocol_is_optional_validated_and_non_breaking(tmp_path):
    legacy = legacy_document()
    before = configuration.configure("full", write_document(tmp_path, legacy))
    assert "execution_protocol" not in before
    legacy["phases"]["full"]["execution_protocol"] = PROTOCOL
    after = configuration.configure("full", write_document(tmp_path, legacy))
    assert set(after) - set(before) == {"execution_protocol"}
    assert after["execution_protocol"] == PROTOCOL
    assert {key: value for key, value in after.items() if key != "execution_protocol"} == before


@pytest.mark.parametrize("protocol", ["engine-session-v2", "", None, True, 1, ["engine-session-v1"]])
def test_execution_protocol_values_are_a_fixed_allowlist(tmp_path, protocol):
    value = document()
    value["phases"]["full"]["execution_protocol"] = protocol
    with pytest.raises(ValueError, match="execution_protocol"):
        configuration.configure("full", write_document(tmp_path, value))


def test_only_a_declared_protocol_is_executable():
    assert configuration.may_execute(configuration.configure("full")) is True
    for relative, phase in (
        (LEGACY_PROFILE, "gpu_pilot"),
        (LEGACY_PROFILE, "cpu_pilot"),
        (LEGACY_PROFILE, "full"),
        (FULL_ISOLATED_PROFILE, "full"),
        (CPU_ISOLATED_PROFILE, "cpu_pilot"),
        (WIDTH_PROFILE, "gpu_widths"),
    ):
        cfg = configuration.configure(phase, relative)
        assert configuration.may_execute(cfg) is False
        # The protocol is never inferred: a resolution cannot be executed unless
        # the phase declared it.
        assert "execution_protocol" not in cfg
    with pytest.raises(ValueError, match="must be an object"):
        configuration.may_execute(None)


def test_execution_protocol_is_phase_level_not_document_level(tmp_path):
    value = document()
    del value["phases"]["full"]["execution_protocol"]
    value["execution_protocol"] = PROTOCOL
    with pytest.raises(ValueError, match="unknown keys"):
        configuration.configure("full", write_document(tmp_path, value))


def test_historical_profiles_stay_protocol_free_for_the_runner():
    # A historical profile is verification-only: it must stay protocol-free so the
    # runner can recognise it instead of executing it.
    for relative in (LEGACY_PROFILE, FULL_ISOLATED_PROFILE, CPU_ISOLATED_PROFILE, WIDTH_PROFILE):
        value = load(relative)
        for phase in value["phases"].values():
            assert "execution_protocol" not in phase
    assert document()["phases"]["full"]["execution_protocol"] == PROTOCOL


def test_gpu_widths_profile_contract():
    cfg = configuration.configure("gpu_widths", WIDTH_PROFILE)
    assert set(cfg) == RAW_KEYS
    assert cfg["phase"] == "gpu_widths"
    assert list(cfg["models"]) == ["qwen2b", "joey1"]
    assert cfg["batch_sizes"] == list(WIDE_WIDTHS)
    assert (cfg["sample_size"], cfg["warmup_rows"], cfg["seed"], cfg["repetitions"]) == (257, 17, 1729, 3)
    assert cfg["passes"] == {
        "qwen2b": {"1": 1, "4": 1, "16": 4, "64": 4, "256": 4},
        "joey1": {"1": 1, "4": 2, "16": 4, "64": 4, "256": 4},
    }
    assert cfg["energy"] == {"country_iso_code": "FRA", "gpu_ids": [0], "sample_secs": 0.2, "min_samples": 5}
    assert (cfg["device_idle_s"], cfg["loaded_idle_s"], cfg["settle_s"]) == (8, 12, 5)
    assert cfg["loaded_idle_adapters"] == ["qwen2b", "joey1"]
    assert cfg["measure_training"] is False
    # A 0.2 s tracker cadence needs >= 5 scheduled samples; every condition's
    # conservative duration estimate is at least one second per pass set.
    assert cfg["energy"]["sample_secs"] == 0.2 and cfg["energy"]["min_samples"] == 5
    for model_id in cfg["models"]:
        assert set(cfg["passes"][model_id]) == {str(width) for width in WIDE_WIDTHS}
        for count in cfg["passes"][model_id].values():
            assert cfg["sample_size"] * count <= configuration.MAX_PREDICTIONS_PER_CONDITION


def test_packed_qev_is_the_explicit_wider_width_opt_in(tmp_path):
    cfg = configuration.configure("gpu_widths", WIDTH_PROFILE)
    assert cfg["models"]["joey1"]["params"]["request_mode"] == "packed"
    assert configuration.supported_batches(cfg["models"]["joey1"]) == WIDE_WIDTHS
    # Removing the opt-in returns the historical capability, so the wider pass
    # entries no longer cover it exactly.
    value = width_document()
    del value["models"]["joey1"]["params"]["request_mode"]
    with pytest.raises(ValueError, match="exactly cover supported requested widths"):
        configuration.configure("gpu_widths", write_document(tmp_path, value))


def test_explicit_single_qev_mode_keeps_historical_capability(tmp_path):
    value = width_document()
    value["models"]["joey1"]["params"]["request_mode"] = "single"
    value["phases"]["gpu_widths"]["passes"]["joey1"] = {"1": 1}
    cfg = configuration.configure("gpu_widths", write_document(tmp_path, value))
    assert cfg["models"]["joey1"]["params"]["request_mode"] == "single"
    assert configuration.supported_batches(cfg["models"]["joey1"]) == (1,)


@pytest.mark.parametrize("mutation", [
    lambda d: d["models"]["joey1"]["params"].update(request_mode="auto"),
    lambda d: d["models"]["joey1"]["params"].update(request_mode=True),
    lambda d: d["models"]["qwen2b"]["params"].update(request_mode="packed"),
])
def test_request_mode_is_a_strict_qev_only_opt_in(tmp_path, mutation):
    value = width_document()
    mutation(value)
    with pytest.raises(ValueError, match="request_mode|unknown keys"):
        configuration.configure("gpu_widths", write_document(tmp_path, value))


@pytest.mark.parametrize("width", [0, 2, 32, 512, True, 4.0, "4"])
def test_requested_widths_are_a_fixed_allowlist(tmp_path, width):
    value = width_document()
    value["phases"]["gpu_widths"]["batch_sizes"].append(width)
    value["phases"]["gpu_widths"]["passes"]["qwen2b"][str(width)] = 1
    with pytest.raises(ValueError, match="batch_sizes"):
        configuration.configure("gpu_widths", write_document(tmp_path, value))


def test_published_configs_and_profiles_are_unchanged():
    # The historical files keep their content hashes: the active stage is the new
    # experiment.json, and the previous one is archived byte-for-byte.
    expected = {
        "experiment.json": "0301ccd6eb5e3d2d1d6d21ea0f425201d66969184420369fa4ee5114fa9d29ea",
        "profiles/legacy-experiment.json": "d69b8d5b1d4063dda167ba0dbf02ac5586a0ac03a6fe426ae3b9e358f11abc4c",
        "profiles/laptop-full-isolated-02.json": "02f11e29e70a6bc8223faa66e0be5f0f8ba8572babec399759f8bdd113ca135a",
        "profiles/laptop-cpu-isolated-02.json": "84f333434eb237e2e7d18bed66aa95e9c2979190b7d1e0560a5c8a6e86afe346",
        "profiles/laptop-gpu-widths.json": "2fbee0e66bb53c4a93e336e38155aa383d685ca127d14a99ea12d44b2cf199b1",
    }
    for relative, digest in expected.items():
        path = configuration.ROOT / relative
        assert sha256(path.read_bytes()).hexdigest() == digest
    # Historical roots keep resolving at widths 1/16 for verification.
    for path, phase in (
        (LEGACY_PROFILE, "gpu_pilot"),
        (LEGACY_PROFILE, "cpu_pilot"),
        (LEGACY_PROFILE, "full"),
        (FULL_ISOLATED_PROFILE, "full"),
        (CPU_ISOLATED_PROFILE, "cpu_pilot"),
    ):
        cfg = configuration.configure(phase, path)
        assert cfg["batch_sizes"] == [1, 16]
        assert "execution_protocol" not in cfg
    assert configuration.configure("full")["batch_sizes"] == ACTIVE_WIDTHS
