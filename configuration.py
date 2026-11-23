"""Static, offline validation for the pinned sentiment-energy configurations.

``configure`` is the runner's sole configuration entry point. It reads local
JSON only: it never checks model availability, imports an inference engine, or
downloads weights. Paths deliberately remain relative in the returned mapping.
The runner must normalize them against its project root at runtime, preserving
virtual-environment interpreter symlinks with ``os.path.abspath`` rather than
``Path.resolve``.

A phase may declare ``execution_protocol`` (see :data:`EXECUTION_PROTOCOL`).
The key is optional so that historical configurations stay byte-identical and
resolvable for verification: this module never refuses a phase, it only reports
what the phase declares. Deciding whether a resolved configuration may be
executed is the runner's job (an absent protocol means verification only).
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path, PurePosixPath
import re

from storage import load

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "experiment.json"
MODEL_MANIFEST = ROOT / "setup" / "model_manifest.json"
MAX_PREDICTIONS_PER_CONDITION = 500_000
CLASSICAL = {"svm", "nb", "fasttext"}
GPU = {"vllm", "qev"}
# Requested widths are client-side concurrency conditions for independent tweets.
# The historical protocol requested only 1 and 16; the wider stage adds 4, 64 and
# 256. Keep this an explicit allowlist so a mistyped width cannot silently become
# a different condition.
REQUESTED_WIDTHS = (1, 4, 16, 64, 256)
# Coarse adapter capability. It is not a per-model promise: it only decides which
# requested widths a phase may run instead of recording as skipped/unsupported.
# None (classical adapters) means every requested width can be formed from
# independent tweets in static groups. vLLM has always accepted independent
# requests at the historical widths 1 and 16; the tuple also lists the wider
# widths of the gpu_widths stage. QEV serves one independent tweet per request
# unless the model spec explicitly opts into packed requests.
_SUPPORTED_BATCHES = {"vllm": REQUESTED_WIDTHS, "qev": (1,)}
_PACKED_SUPPORTED_BATCHES = REQUESTED_WIDTHS
_QEV_REQUEST_MODES = {"single", "packed"}
# The only execution protocol this configuration vocabulary understands. It is a
# phase-level opt-in: a phase that omits it describes a historical stage that may
# still be resolved and verified, but not executed by the runner.
EXECUTION_PROTOCOL = "engine-session-v1"
_EXECUTION_PROTOCOLS = {EXECUTION_PROTOCOL}
REVISION = re.compile(r"[0-9a-f]{40}\Z")

_TOP_KEYS = {"schema_version", "dataset", "energy", "models", "phases"}
_PHASE_KEYS = {
    "model_ids", "batch_sizes", "seed", "sample_size", "warmup_rows", "repetitions",
    "passes", "device_idle_s", "loaded_idle_s", "loaded_idle_adapters", "measure_training", "settle_s",
}
# Optional phase keys. ``execution_protocol`` is validated when present and
# propagated to the resolved mapping only then, so historical configurations keep
# their exact resolved shape and therefore their hashes.
_OPTIONAL_PHASE_KEYS = {"execution_protocol"}
_ENERGY_KEYS = {"country_iso_code", "gpu_ids", "sample_secs", "min_samples"}
_COMMON_MODEL_KEYS = {"adapter", "device", "params"}
_HUB_MODEL_KEYS = _COMMON_MODEL_KEYS | {"repo", "revision", "local_snapshot", "engine_safety"}
_VLLM_MODEL_KEYS = _HUB_MODEL_KEYS | {"python"}
_VLLM_PARAMS = {
    "served_model_name", "dtype", "quantization", "max_model_len", "gpu_memory_utilization",
    "max_num_seqs", "max_tokens", "temperature", "top_p", "seed", "enable_thinking",
    "tensor_parallel_size", "enforce_eager", "host", "prompt",
}
_QEV_PARAMS = {
    "onejev_path", "python", "device", "dtype", "head_dtype", "fork_mode", "fork_tolerance",
    "fork_token_budget", "max_branch_tokens", "max_request_tokens", "debias", "debug",
    "init_timeout_s", "request_timeout_s", "close_timeout_s", "request_mode",
}
_SAFETY_KEYS = {
    "weight_shards_bytes", "reserve_mib", "gpu_index", "cuda_visible_devices", "expected_python",
    "version_timeout_s", "startup_timeout_s", "request_timeout_s", "shutdown_timeout_s",
}
_TFIDF_KEYS = {
    "analyzer", "ngram_range", "min_df", "max_df", "max_features", "lowercase", "strip_accents",
    "token_pattern", "binary", "norm", "use_idf", "smooth_idf", "sublinear_tf",
}
_SVM_KEYS = {"C", "penalty", "loss", "dual", "tol", "max_iter", "class_weight", "random_state"}
_NB_KEYS = {"alpha", "fit_prior", "class_prior"}
_FASTTEXT_KEYS = {
    "lr", "dim", "ws", "epoch", "minCount", "minCountLabel", "minn", "maxn", "neg", "wordNgrams",
    "loss", "bucket", "thread", "lrUpdateRate", "t", "seed", "verbose",
}


def _mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def supported_batches(spec):
    """Return the independent requested widths an adapter can serve, or ``None``.

    ``None`` mirrors the classical adapters: any requested width can be formed
    from independent tweets as static groups. The result is a coarse adapter
    capability, independent of what a phase actually requests, so it stays
    deterministic for a given spec. QEV advertises only width 1 unless the spec
    opts in with ``params.request_mode == "packed"``; an absent ``request_mode``
    (or the explicit historical ``"single"``) keeps single-tweet-per-request.

    The runner is expected to call this for planning and preflight: a requested
    width absent from the returned tuple must be recorded as
    skipped/unsupported rather than silently dropped from the plan.
    """
    spec = _mapping(spec, "model spec")
    kind = spec.get("adapter")
    if kind in CLASSICAL:
        return None
    if kind in GPU:
        if kind == "qev":
            params = _mapping(spec.get("params") or {}, "model params")
            if params.get("request_mode") == "packed":
                return _PACKED_SUPPORTED_BATCHES
        return _SUPPORTED_BATCHES[kind]
    raise ValueError(f"unsupported adapter for supported_batches: {kind!r}")


def may_execute(cfg):
    """Return whether a resolved phase declares a runnable execution protocol.

    Historical configurations deliberately resolve so their pins and roots stay
    verifiable; only a phase that declares :data:`EXECUTION_PROTOCOL` may be
    executed. The runner owns the refusal decision; this is the single source of
    truth for it, so no caller has to hardcode the protocol string.
    """
    protocol = _mapping(cfg, "resolved configuration").get("execution_protocol")
    return isinstance(protocol, str) and not isinstance(protocol, bool) and protocol in _EXECUTION_PROTOCOLS


def _exact_keys(value, allowed, name, *, required=()):
    _mapping(value, name)
    unknown = set(value) - set(allowed)
    missing = set(required) - set(value)
    if unknown or missing:
        raise ValueError(f"{name}: unknown keys={sorted(unknown)}, missing keys={sorted(missing)}")


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value, name, *, minimum=0, strict=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < minimum or (strict and value == minimum):
        raise ValueError(f"{name} is below its allowed minimum")
    return value


def _string(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _relative_path(value, name):
    _string(value, name)
    path = PurePosixPath(value)
    if path.is_absolute() or "\\" in value or value.startswith("~") or any(part in {"", "."} for part in path.parts):
        raise ValueError(f"{name} must be a normalized project-relative POSIX path")
    if ".." in path.parts and name != "Qwen prior-pilot path":
        # Only the explicitly referenced prior Qwen environment/snapshot may leave ROOT.
        if value not in {
            "../inspect_pilot_study/.scratch/gpu/model",
            "../inspect_pilot_study/.scratch/gpu/venv/bin/python",
        }:
            raise ValueError(f"{name} may not escape the project except for the pinned Qwen pilot paths")
    return value


def _revision(value, name):
    if not isinstance(value, str) or not REVISION.fullmatch(value):
        raise ValueError(f"{name} must be an exact lowercase 40-character commit")
    return value


def _validate_energy(energy):
    _exact_keys(energy, _ENERGY_KEYS, "energy", required=_ENERGY_KEYS)
    country = _string(energy["country_iso_code"], "energy.country_iso_code")
    if len(country) != 3 or country != country.upper():
        raise ValueError("energy.country_iso_code must be an uppercase ISO 3166-1 alpha-3 code")
    _number(energy["sample_secs"], "energy.sample_secs", strict=True)
    _integer(energy["min_samples"], "energy.min_samples", minimum=1)
    gpu_ids = energy["gpu_ids"]
    if not isinstance(gpu_ids, list) or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("energy.gpu_ids must be a unique list")
    for index in gpu_ids:
        _integer(index, "energy.gpu_ids entry")


def _validate_dataset(dataset):
    keys = {"adapter", "name", "path", "revision"}
    _exact_keys(dataset, keys, "dataset", required=keys)
    if dataset["adapter"] != "tweeteval":
        raise ValueError("this pinned experiment requires the tweeteval dataset adapter")
    _string(dataset["name"], "dataset.name")
    _relative_path(dataset["path"], "dataset.path")
    _revision(dataset["revision"], "dataset.revision")


def _validate_weights(weights, name):
    _mapping(weights, name)
    if not weights:
        raise ValueError(f"{name} may not be empty")
    for filename, size in weights.items():
        path = PurePosixPath(_string(filename, f"{name} filename"))
        if path.is_absolute() or ".." in path.parts or not filename.endswith(".safetensors"):
            raise ValueError(f"{name} has an unsafe or non-safetensors filename: {filename!r}")
        _integer(size, f"{name}[{filename!r}]", minimum=1)


def _validate_classical(model_id, spec):
    _exact_keys(spec, _COMMON_MODEL_KEYS, f"models.{model_id}", required=_COMMON_MODEL_KEYS)
    if spec["device"] != "cpu":
        raise ValueError(f"models.{model_id}.device must be cpu")
    params = spec["params"]
    _exact_keys(params, {"preprocessing", "model", "training"}, f"models.{model_id}.params")
    for section in ("preprocessing", "model", "training"):
        _mapping(params[section], f"models.{model_id}.params.{section}")
    _exact_keys(params["training"], {"seed"}, f"models.{model_id}.params.training", required={"seed"})
    _integer(params["training"]["seed"], f"models.{model_id}.params.training.seed")
    kind = spec["adapter"]
    if kind == "fasttext":
        _exact_keys(params["preprocessing"], {"lowercase", "strip_accents"}, f"models.{model_id}.params.preprocessing")
        _exact_keys(params["model"], _FASTTEXT_KEYS, f"models.{model_id}.params.model")
        if params["model"].get("thread", 1) != 1:
            raise ValueError("fastText must remain deterministic with thread=1")
    else:
        _exact_keys(params["preprocessing"], _TFIDF_KEYS, f"models.{model_id}.params.preprocessing")
        _exact_keys(params["model"], _SVM_KEYS if kind == "svm" else _NB_KEYS, f"models.{model_id}.params.model")
        ngram = params["preprocessing"].get("ngram_range", [1, 2])
        if ngram != [1, 2]:
            raise ValueError(f"models.{model_id} is pinned to word unigram/bigram TF-IDF")


def _validate_vllm_params(model_id, params):
    _exact_keys(params, _VLLM_PARAMS, f"models.{model_id}.params")
    pinned = {
        "dtype": "bfloat16", "quantization": None, "max_model_len": 1024, "max_num_seqs": 16,
        "max_tokens": 8, "temperature": 0.0, "top_p": 1.0, "enable_thinking": False,
        "tensor_parallel_size": 1, "enforce_eager": False, "host": "127.0.0.1",
    }
    for key, expected in pinned.items():
        if key in params and (params[key] != expected or isinstance(params[key], bool) != isinstance(expected, bool)):
            raise ValueError(f"models.{model_id}.params.{key} must remain {expected!r}")
    if "gpu_memory_utilization" in params:
        _number(params["gpu_memory_utilization"], f"models.{model_id}.params.gpu_memory_utilization", strict=True)
        if params["gpu_memory_utilization"] > 1:
            raise ValueError("gpu_memory_utilization must be <= 1")
    for key in ("served_model_name", "prompt"):
        if key in params:
            _string(params[key], f"models.{model_id}.params.{key}")
    if "seed" in params:
        _integer(params["seed"], f"models.{model_id}.params.seed")


def _validate_qev_params(model_id, params):
    _exact_keys(params, _QEV_PARAMS, f"models.{model_id}.params", required={"onejev_path", "python"})
    _relative_path(params["onejev_path"], f"models.{model_id}.params.onejev_path")
    _relative_path(params["python"], f"models.{model_id}.params.python")
    pinned = {"device": "cuda:0", "dtype": "bfloat16", "head_dtype": "float32", "fork_mode": "auto", "debias": 1, "debug": False}
    for key, expected in pinned.items():
        if key in params and (params[key] != expected or isinstance(params[key], bool) != isinstance(expected, bool)):
            raise ValueError(f"models.{model_id}.params.{key} must remain {expected!r}")
    for key in ("init_timeout_s", "request_timeout_s", "close_timeout_s"):
        if key in params:
            _number(params[key], f"models.{model_id}.params.{key}", strict=True)
    if "request_mode" in params:
        mode = params["request_mode"]
        if not isinstance(mode, str) or mode not in _QEV_REQUEST_MODES:
            raise ValueError(
                f"models.{model_id}.params.request_mode must be one of "
                f"{sorted(_QEV_REQUEST_MODES)} when present; omit it for the historical default"
            )


def _validate_gpu(model_id, spec, manifest):
    kind = spec["adapter"]
    keys = _VLLM_MODEL_KEYS if kind == "vllm" else _HUB_MODEL_KEYS
    _exact_keys(spec, keys, f"models.{model_id}", required=keys)
    if spec["device"] != "cuda":
        raise ValueError(f"models.{model_id}.device must be cuda")
    _string(spec["repo"], f"models.{model_id}.repo")
    _revision(spec["revision"], f"models.{model_id}.revision")
    _relative_path(spec["local_snapshot"], f"models.{model_id}.local_snapshot")
    if kind == "vllm":
        _relative_path(spec["python"], f"models.{model_id}.python")
        _validate_vllm_params(model_id, spec["params"])
    else:
        _validate_qev_params(model_id, spec["params"])
    safety = spec["engine_safety"]
    _exact_keys(safety, _SAFETY_KEYS, f"models.{model_id}.engine_safety", required={"weight_shards_bytes", "reserve_mib", "gpu_index"})
    _validate_weights(safety["weight_shards_bytes"], f"models.{model_id}.engine_safety.weight_shards_bytes")
    expected_reserve = 3500 if kind == "vllm" else 2500
    if safety["reserve_mib"] != expected_reserve:
        raise ValueError(f"models.{model_id}.engine_safety.reserve_mib must be {expected_reserve}")
    if safety["gpu_index"] != 0 or isinstance(safety["gpu_index"], bool):
        raise ValueError("this protocol supports only gpu_index=0")
    if "expected_python" in safety:
        _relative_path(safety["expected_python"], f"models.{model_id}.engine_safety.expected_python")
    if kind == "vllm" and safety.get("expected_python") != spec["python"]:
        raise ValueError(f"models.{model_id} python and expected_python must match")
    for key in ("version_timeout_s", "startup_timeout_s", "request_timeout_s", "shutdown_timeout_s"):
        if key in safety:
            _number(safety[key], f"models.{model_id}.engine_safety.{key}", strict=True)
    pinned = manifest["models"].get(model_id)
    if not isinstance(pinned, dict):
        raise ValueError(f"models.{model_id} is absent from setup/model_manifest.json")
    observed = {
        "adapter": kind, "repo": spec["repo"], "revision": spec["revision"],
        "local_snapshot": spec["local_snapshot"], "weight_shards_bytes": safety["weight_shards_bytes"],
    }
    expected = {key: pinned.get(key) for key in observed}
    if observed != expected:
        raise ValueError(f"models.{model_id} differs from its exact Hub manifest pin")


def _validate_models(models, manifest):
    _mapping(models, "models")
    if not models:
        raise ValueError("models may not be empty")
    for model_id, spec in models.items():
        _string(model_id, "model id")
        _mapping(spec, f"models.{model_id}")
        kind = spec.get("adapter")
        if kind in CLASSICAL:
            _validate_classical(model_id, spec)
        elif kind in GPU:
            _validate_gpu(model_id, spec, manifest)
        else:
            raise ValueError(f"models.{model_id}.adapter is unsupported: {kind!r}")


def _validate_execution_protocol(name, phase):
    if "execution_protocol" not in phase:
        return None
    protocol = phase["execution_protocol"]
    if not isinstance(protocol, str) or protocol not in _EXECUTION_PROTOCOLS:
        raise ValueError(
            f"phases.{name}.execution_protocol must be {sorted(_EXECUTION_PROTOCOLS)} when present; "
            "omit it for a historical phase that is verification-only"
        )
    return protocol


def _validate_phase(name, phase, models):
    _exact_keys(phase, _PHASE_KEYS | _OPTIONAL_PHASE_KEYS, f"phases.{name}", required=_PHASE_KEYS)
    _validate_execution_protocol(name, phase)
    model_ids = phase["model_ids"]
    if not isinstance(model_ids, list) or not model_ids or len(set(model_ids)) != len(model_ids):
        raise ValueError(f"phases.{name}.model_ids must be a nonempty unique list")
    if any(not isinstance(model_id, str) or model_id not in models for model_id in model_ids):
        raise ValueError(f"phases.{name}.model_ids contains an unknown model")
    widths = phase["batch_sizes"]
    if not isinstance(widths, list) or not widths or len(set(widths)) != len(widths) or any(
            isinstance(width, bool) or width not in REQUESTED_WIDTHS for width in widths):
        raise ValueError(
            f"phases.{name}.batch_sizes must contain unique requested widths from {list(REQUESTED_WIDTHS)}"
        )
    _integer(phase["seed"], f"phases.{name}.seed")
    sample = _integer(phase["sample_size"], f"phases.{name}.sample_size", minimum=1)
    if sample > 12_284:
        raise ValueError(f"phases.{name}.sample_size exceeds the TweetEval test split")
    warmup = _integer(phase["warmup_rows"], f"phases.{name}.warmup_rows")
    if warmup > 2_000:
        raise ValueError(f"phases.{name}.warmup_rows exceeds the TweetEval validation split")
    _integer(phase["repetitions"], f"phases.{name}.repetitions", minimum=1)
    for key in ("device_idle_s", "loaded_idle_s"):
        _number(phase[key], f"phases.{name}.{key}", strict=True)
    _number(phase["settle_s"], f"phases.{name}.settle_s")
    if not isinstance(phase["measure_training"], bool):
        raise ValueError(f"phases.{name}.measure_training must be boolean")
    loaded = phase["loaded_idle_adapters"]
    if not isinstance(loaded, list) or len(set(loaded)) != len(loaded) or any(model_id not in model_ids for model_id in loaded):
        raise ValueError(f"phases.{name}.loaded_idle_adapters must be unique selected model ids")
    passes = phase["passes"]
    _exact_keys(passes, set(model_ids), f"phases.{name}.passes", required=set(model_ids))
    for model_id in model_ids:
        per_width = passes[model_id]
        _mapping(per_width, f"phases.{name}.passes.{model_id}")
        capability = supported_batches(models[model_id])
        required = set(widths) if capability is None else set(widths) & set(capability)
        if set(per_width) != {str(width) for width in required}:
            raise ValueError(f"phases.{name}.passes.{model_id} must exactly cover supported requested widths {sorted(required)}")
        for width, count in per_width.items():
            _integer(count, f"phases.{name}.passes.{model_id}.{width}", minimum=1)
            if sample * count > MAX_PREDICTIONS_PER_CONDITION:
                raise ValueError(f"phases.{name}.{model_id} width {width} exceeds the in-memory prediction bound")


def _document(config_path):
    document = _mapping(load(config_path), "experiment")
    _exact_keys(document, _TOP_KEYS, "experiment", required=_TOP_KEYS)
    if document["schema_version"] != 1 or isinstance(document["schema_version"], bool):
        raise ValueError("experiment.schema_version must be 1")
    _validate_dataset(document["dataset"])
    _validate_energy(document["energy"])
    manifest = _mapping(load(MODEL_MANIFEST), "model manifest")
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("models"), dict):
        raise ValueError("invalid setup/model_manifest.json")
    _validate_models(document["models"], manifest)
    phases = _mapping(document["phases"], "phases")
    if not phases:
        raise ValueError("phases may not be empty")
    for name, phase in phases.items():
        _string(name, "phase name")
        _validate_phase(name, phase, document["models"])
    return document


def configure(phase, config_path=DEFAULT_CONFIG):
    """Return a detached, validated raw configuration for ``phase``.

    The returned shape is the runner contract. A requested width that the model's
    adapter cannot serve (see :func:`supported_batches`) stays in ``batch_sizes``
    and is deliberately absent from that model's ``passes``; the runner must
    record it as skipped/unsupported rather than removing it from the phase.

    ``execution_protocol`` appears in the result only when the phase declares it.
    An absent key means the caller resolved a historical configuration: it stays
    usable for verification, and refusing to execute it is the runner's decision.
    """
    document = _document(Path(config_path))
    if phase not in document["phases"]:
        raise ValueError(f"unknown phase {phase!r}; choose from {sorted(document['phases'])}")
    selected = document["phases"][phase]
    model_ids = selected["model_ids"]
    resolved = deepcopy({
        "phase": phase,
        "dataset": document["dataset"],
        "models": {model_id: document["models"][model_id] for model_id in model_ids},
        "batch_sizes": selected["batch_sizes"],
        "seed": selected["seed"],
        "sample_size": selected["sample_size"],
        "warmup_rows": selected["warmup_rows"],
        "repetitions": selected["repetitions"],
        "passes": selected["passes"],
        "energy": document["energy"],
        "device_idle_s": selected["device_idle_s"],
        "loaded_idle_s": selected["loaded_idle_s"],
        "loaded_idle_adapters": selected["loaded_idle_adapters"],
        "measure_training": selected["measure_training"],
        "settle_s": selected["settle_s"],
    })
    if "execution_protocol" in selected:
        resolved["execution_protocol"] = selected["execution_protocol"]
    return resolved
