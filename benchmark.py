#!/usr/bin/env python3
"""Exclusive, resumable orchestration for the sentiment energy benchmark.

``plan`` and ``verify`` never construct an adapter or start a measurement. ``run``
holds the project measurement lock for its whole lifetime. Completed attempts are
immutable; interrupted and incomplete attempts are resumed in fresh directories.
``run --progress`` streams opt-in progress lines on stderr between measurement
operations; the single JSON report on stdout is unchanged.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
from time import perf_counter, sleep
from typing import Any, Iterable, TextIO

import configuration
import dataset
import energy
import hardware
import storage
import workload
from adapters.base import Unavailable

ROOT = Path(__file__).resolve().parent
LOCK = ROOT / ".scratch" / "measurement.lock"
KIND_ORDER = ("device_idle", "training", "loaded_idle", "inference")
CLASSICAL = {"svm", "nb", "fasttext"}
REFERENCE_FIELDS = ("device_idle_run_id", "training_run_id", "loaded_idle_run_id")


class StopRequested(BaseException):
    """A stop file appeared between measurement operations."""


class SignalReceived(BaseException):
    """SIGINT/SIGTERM is converted to an incomplete active attempt."""

    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(signal.Signals(signum).name)


PROGRESS_PREFIX = "benchmark progress"


def _progress_value(value: Any) -> str:
    """Render one key=value token; quote any value that would not survive a token split.

    Quoted values are JSON string literals (non-ASCII kept literal), so any
    whitespace, line break, backslash, quote, or '=' inside a value can never
    split or escape a progress line.
    """
    if isinstance(value, (list, tuple)):
        value = ",".join(str(item) for item in value)
    text = str(value)
    if not text or any(character.isspace() or character in {'"', '=', '\\'} for character in text):
        return json.dumps(text, ensure_ascii=False)
    return text


def _trial_fields(trial: dict) -> dict:
    return {"run_id": trial["run_id"], "kind": trial["kind"], "model": trial["model"],
            "rep": trial["rep"], "width": trial["batch_size"]}


class Progress:
    """Opt-in, synchronous progress reporting; disabled unless a stream is supplied.

    Events are emitted only between measurement operations, never from inside an
    energy capture window, so enabling progress adds no work to a measured window.
    A broken stream disables reporting instead of failing an experiment.
    """

    def __init__(self, stream: TextIO | None = None):
        self._stream = stream
        self._reported: set[str] = set()

    @property
    def enabled(self) -> bool:
        return self._stream is not None

    def event(self, name: str, **fields: Any) -> None:
        stream = self._stream
        if stream is None:
            return
        parts = [PROGRESS_PREFIX, name]
        parts.extend(f"{key}={_progress_value(value)}" for key, value in fields.items() if value is not None)
        try:
            stream.write(" ".join(parts) + "\n")
            stream.flush()
        except (OSError, ValueError):
            self._stream = None

    def report_outcomes(self, root: Path, trials: Iterable[dict], work: dict[str, int],
                        total: int) -> dict[str, int]:
        """Emit each failed/skipped/incomplete selected trial once, with its location."""
        if not self.enabled:
            # Historical default: no stream means no reporting and no state scans.
            return {}
        counts: dict[str, int] = {}
        for trial in trials:
            path = root / trial["run_id"] / "state.json"
            state = None
            if path.exists():
                try:
                    state = storage.load(path)
                except (OSError, ValueError, KeyError, TypeError):
                    state = None
            status = state["status"] if state else "pending"
            counts[status] = counts.get(status, 0) + 1
            if (state is None or status not in {"incomplete", "failed", "skipped"}
                    or trial["run_id"] in self._reported):
                continue
            self._reported.add(trial["run_id"])
            self.event("outcome", work=work.get(trial["run_id"]), total=total, status=status,
                       reason=state.get("reason"), active_attempt=state.get("active_attempt"),
                       location=str(root / trial["run_id"]), **_trial_fields(trial))
        return counts


def _configure(phase: str, config_path: Path):
    """Resolve the two external contracts exactly once for this command."""
    cfg = configuration.configure(phase, config_path)
    dataset_info, train, validation, panel, warmup = dataset.configure(
        cfg["dataset"],
        sample_size=cfg["sample_size"],
        warmup_rows=cfg["warmup_rows"],
        seed=cfg["seed"],
    )
    return cfg, dataset_info, train, validation, panel, warmup


def _output_root(phase: str, out: Path | None) -> Path:
    return Path(out) if out is not None else ROOT / "artifacts" / phase


def _is_cuda(spec: dict) -> bool:
    return spec.get("device", "cuda" if spec["adapter"] in {"qev", "vllm"} else "cpu") == "cuda"


def _passes(cfg: dict, model: str, batch_size: int) -> int:
    return cfg["passes"][model][str(batch_size)]


def _packed_qev(spec: dict) -> bool:
    return spec["adapter"] == "qev" and (spec.get("params") or {}).get("request_mode") == "packed"


def _audit_kind(cfg: dict, trial: dict) -> str:
    spec = cfg["models"][trial["model"]]
    return "qev-packed" if _packed_qev(spec) else spec["adapter"]

def _secs(cfg: dict, stem: str) -> float:
    # The on-disk validator currently emits the historical *_s spellings;
    # accept the agreed *_secs spellings too so either raw contract works.
    return cfg[stem + "_secs"] if stem + "_secs" in cfg else cfg[stem + "_s"]


def _runtime_spec(spec: dict) -> dict:
    """Resolve configured project-relative paths without changing plan identity."""
    resolved = deepcopy(spec)
    for key in ("local_snapshot", "python", "venv", "cache_dir"):
        value = resolved.get(key)
        if value and not Path(value).is_absolute():
            # Preserve venv interpreter symlinks; resolve only lexical '..' parts.
            resolved[key] = os.path.abspath(ROOT / value)
    safety = resolved.get("engine_safety") or {}
    if safety.get("expected_python") and not Path(safety["expected_python"]).is_absolute():
        safety["expected_python"] = os.path.abspath(ROOT / safety["expected_python"])
    params = resolved.get("params") or {}
    for key in ("python", "onejev_path"):
        if params.get(key) and not Path(params[key]).is_absolute():
            params[key] = os.path.abspath(ROOT / params[key])
    return resolved


def _create_adapter(spec: dict, attempt: Path):
    kind = spec["adapter"]
    if kind in CLASSICAL:
        from adapters import classical
        return classical.create(spec, attempt)
    if kind == "vllm":
        from adapters import vllm_local
        return vllm_local.create(spec, attempt)
    if kind == "qev":
        from adapters import qev_local
        return qev_local.create(spec, attempt)
    raise Unavailable(f"unknown adapter: {kind}")


def _new_run_id(config_sha: str, dataset_sha: str, trial: dict) -> str:
    return storage.identity({"config": config_sha, "dataset": dataset_sha, "trial": trial})[:24]


def build_plan(cfg: dict, dataset_info: dict, train_count: int) -> dict:
    """Build a deterministic plan, retaining exact historical format-1 identities."""
    protocol = cfg.get("execution_protocol")
    if protocol == "engine-session-v1":
        return _build_session_plan(cfg, dataset_info, train_count)
    if protocol is not None:
        raise ValueError(f"unknown execution protocol: {protocol!r}")
    config_sha = storage.identity(cfg)
    dataset_sha = storage.identity(dataset_info)
    conditions = [
        (model, batch, rep)
        for model in sorted(cfg["models"])
        for batch in sorted(set(cfg["batch_sizes"]))
        for rep in range(cfg["repetitions"])
    ]
    # Seeded order avoids systematic condition order while retaining a total,
    # reproducible sort (including when two model names share a prefix).
    conditions.sort(key=lambda item: (
        storage.identity({"seed": cfg["seed"], "condition": item}), item
    ))
    trials: list[dict] = []
    blocks: list[dict] = []
    for block_index, (model, batch, rep) in enumerate(conditions):
        spec = cfg["models"][model]
        block_key = {"model": model, "batch_size": batch, "rep": rep}
        block_id = storage.identity({"config": config_sha, "dataset": dataset_sha, "block": block_key})[:24]
        block = {
            "block_id": block_id, "block_index": block_index, **block_key, "trials": [],
        }
        kinds = ["device_idle"]
        if cfg["measure_training"] and spec["adapter"] in CLASSICAL:
            kinds.append("training")
        if _is_cuda(spec) and model in cfg["loaded_idle_adapters"]:
            kinds.append("loaded_idle")
        kinds.append("inference")
        made: dict[str, dict] = {}
        supported = configuration.supported_batches(spec)
        for kind in kinds:
            if kind == "inference" and supported is not None and batch not in supported:
                # Keep unsupported requested conditions visible without requiring
                # a meaningless pass target (QEV batch 16 intentionally omits it).
                passes = 0
            else:
                passes = _passes(cfg, model, batch) if kind == "inference" else 0
            expected = {
                "device_idle": 0,
                "loaded_idle": 0,
                "training": train_count,
                "inference": cfg["sample_size"] * passes,
            }[kind]
            native_count = 0 if kind == "training" else expected
            if kind == "inference" and _packed_qev(spec):
                native_count = ((cfg["sample_size"] + batch - 1) // batch) * passes
            trial = {
                "run_id": "", "phase": cfg["phase"], "kind": kind, "model": model,
                "batch_size": batch, "rep": rep, "block_id": block_id,
                "block_index": block_index, "adapter": spec["adapter"],
                "device": "cuda" if _is_cuda(spec) else "cpu", "passes": passes,
                "expected_count": expected, "native_count": native_count,
                "panel_size": cfg["sample_size"] if kind == "inference" else 0,
                "config_sha256": config_sha, "dataset_sha256": dataset_sha,
                "device_idle_run_id": None, "training_run_id": None, "loaded_idle_run_id": None,
            }
            if kind != "device_idle":
                trial["device_idle_run_id"] = made["device_idle"]["run_id"]
            if kind == "inference" and "training" in made:
                trial["training_run_id"] = made["training"]["run_id"]
            if kind == "loaded_idle" and "training" in made:
                trial["training_run_id"] = made["training"]["run_id"]
            if kind == "inference" and "loaded_idle" in made:
                trial["loaded_idle_run_id"] = made["loaded_idle"]["run_id"]
            identity_fields = {key: value for key, value in trial.items() if key != "run_id"}
            trial["run_id"] = _new_run_id(config_sha, dataset_sha, identity_fields)
            made[kind] = trial
            trials.append(trial)
            block["trials"].append(trial["run_id"])
        blocks.append(block)
    return {
        "format_version": 1, "phase": cfg["phase"], "config_sha256": config_sha,
        "dataset_sha256": dataset_sha, "dataset": dataset_info,
        "train_count": train_count,
        "train_sha256": storage.identity({"count": train_count, "files": dataset_info["files_sha256"]}),
        "block_order": "sha256(seed, condition), then condition", "config": cfg,
        "blocks": blocks, "trials": trials,
    }


def _build_session_plan(cfg: dict, dataset_info: dict, train_count: int) -> dict:
    config_sha, dataset_sha = storage.identity(cfg), storage.identity(dataset_info)
    trials, sessions = [], []
    for rep in range(cfg["repetitions"]):
        models = sorted(cfg["models"], key=lambda model: (
            storage.identity({"seed": cfg["seed"], "rep": rep, "model": model}), model))
        for model in models:
            spec = cfg["models"][model]
            key = {"model": model, "rep": rep}
            session_id = storage.identity({"config": config_sha, "dataset": dataset_sha, "session": key})[:24]
            widths = sorted(set(cfg["batch_sizes"]), key=lambda width: (
                storage.identity({"seed": cfg["seed"], "session": key, "width": width}), width))
            session = {"session_id": session_id, "block_id": session_id,
                       "session_index": len(sessions), "block_index": len(sessions),
                       **key, "widths": widths, "trials": []}
            supported = configuration.supported_batches(spec)

            def add(kind, width, device=None, training=None, loaded=None):
                passes = (_passes(cfg, model, width) if kind == "inference"
                          and (supported is None or width in supported) else 0)
                expected = train_count if kind == "training" else cfg["sample_size"] * passes if kind == "inference" else 0
                native = 0 if kind == "training" else expected
                if kind == "inference" and _packed_qev(spec):
                    native = ((cfg["sample_size"] + width - 1) // width) * passes
                trial = {"run_id": "", "phase": cfg["phase"], "kind": kind, "model": model,
                         "batch_size": width, "rep": rep, "session_id": session_id,
                         "session_index": session["session_index"], "block_id": session_id,
                         "block_index": session["block_index"], "adapter": spec["adapter"],
                         "device": "cuda" if _is_cuda(spec) else "cpu", "passes": passes,
                         "expected_count": expected, "native_count": native,
                         "panel_size": cfg["sample_size"] if kind == "inference" else 0,
                         "config_sha256": config_sha, "dataset_sha256": dataset_sha,
                         "device_idle_run_id": device, "training_run_id": training,
                         "loaded_idle_run_id": loaded}
                trial["run_id"] = _new_run_id(config_sha, dataset_sha, {k: v for k, v in trial.items() if k != "run_id"})
                trials.append(trial)
                session["trials"].append(trial["run_id"])
                return trial["run_id"]

            idle = add("device_idle", 0)
            training = add("training", 0, idle) if cfg["measure_training"] and spec["adapter"] in CLASSICAL else None
            for width in widths:
                loaded = add("loaded_idle", width, idle, training) if _is_cuda(spec) and model in cfg["loaded_idle_adapters"] else None
                add("inference", width, idle, training, loaded)
            sessions.append(session)
    return {"format_version": 2, "execution_protocol": "engine-session-v1", "phase": cfg["phase"],
            "config_sha256": config_sha, "dataset_sha256": dataset_sha, "dataset": dataset_info,
            "train_count": train_count,
            "train_sha256": storage.identity({"count": train_count, "files": dataset_info["files_sha256"]}),
            "session_order": "rep ascending, sha256(seed, rep, model), then model",
            "width_order": "sha256(seed, session, width), then width", "config": cfg,
            "sessions": sessions, "blocks": sessions, "trials": trials}


def _immutable_plan(path: Path, plan: dict) -> None:
    """Atomically link a fully synced new plan; never replace an existing one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write((json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
        storage.sync_dir(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)
    existing = storage.load(path)
    if existing != plan:
        raise ValueError(f"immutable plan differs from selected configuration: {path}")


def ensure_plan(root: Path, cfg: dict, dataset_info: dict, train_count: int) -> dict:
    plan = build_plan(cfg, dataset_info, train_count)
    _immutable_plan(root / "plan.json", plan)
    return plan


def _native_task(trial: dict, cfg: dict, dataset_info: dict, train_count: int) -> dict:
    kind = trial["kind"]
    quantity = train_count if kind == "training" else trial["panel_size"] if kind == "inference" else 0
    caption = (
        f"{trial['phase']} {kind}; model={trial['model']}; "
        f"batch={trial['batch_size']}; rep={trial['rep']}"
    )
    if kind == "inference" and _audit_kind(cfg, trial) == "qev-packed":
        caption += f"; counted requests=packed envelopes; tweet decisions={trial['expected_count']}"
    return energy.task_context(
        kind, trial["model"], dataset_info, trial["native_count"], caption,
        data_quantity=quantity,
    )


def _gate(trial: dict, cfg: dict, native: dict) -> dict:
    duration = native["duration_s"]
    if trial["kind"] in {"device_idle", "loaded_idle"}:
        intended = _secs(cfg, trial["kind"])
        tolerance = max(0.05, intended * 0.01)
        meets = duration + tolerance >= intended
        gate = {
            "kind": "idle_duration", "duration_s": duration,
            "minimum_window_s": intended, "intended_window_s": intended,
            "tolerance_s": tolerance, "window_ratio": duration / intended,
            "sample_ratio": None, "meets_minimum_window": meets,
        }
    else:
        sample_secs = cfg["energy"]["sample_secs"]
        min_samples = cfg["energy"]["min_samples"]
        minimum = sample_secs * min_samples
        scheduled = native.get("scheduled_power_samples")
        if type(scheduled) not in (int, float) or scheduled < 0:
            scheduled = None
        duration_meets = duration >= minimum
        meets = duration_meets if scheduled is None else scheduled >= min_samples
        gate = {
            "kind": "energy_sampling", "duration_s": duration,
            "minimum_window_s": minimum, "sample_secs": sample_secs,
            "min_samples": min_samples, "window_ratio": duration / minimum,
            "sample_ratio": duration / sample_secs,
            "scheduled_power_samples": scheduled,
            "duration_meets_minimum": duration_meets,
            "samples_meet_minimum": None if scheduled is None else scheduled >= min_samples,
            "meets_minimum_window": meets,
        }
    gate["low_sampling"] = not meets
    return gate


def _empty_counts(count: int = 0) -> dict:
    return {
        "submitted": count, "completed": count, "valid": 0, "correct": 0,
        "retries": 0, "retry_requests": 0,
    }


def _inference_counts(counts: dict) -> dict:
    return {**counts, "retries": counts.get("retry_requests", 0)}


def _reference_result(root: Path, run_id: str | None) -> dict | None:
    if run_id is None:
        return None
    state = storage.load(root / run_id / "state.json")
    if state["status"] != "complete":
        raise ValueError(f"baseline is not complete: {run_id}")
    attempt = state["active_attempt"]
    return storage.load(root / run_id / attempt / "result.json")


def _gpu_uuids(host: dict) -> list[str]:
    return sorted(row["uuid"] for row in host.get("gpu", []) if row.get("uuid"))


def _adjusted(native: dict, baseline: dict | None, host: dict, count: int) -> float | None:
    if not count or not baseline:
        return None
    baseline_host = baseline.get("host", {})
    if (host.get("host_id") != baseline_host.get("host_id")
            or _gpu_uuids(host) != _gpu_uuids(baseline_host)):
        return None
    return energy.idle_adjusted(native, baseline["energy"], count)


def _derived(trial: dict, native: dict, root: Path, host: dict) -> dict:
    count = trial["expected_count"]
    gross = energy.J_PER_KWH * native["energy_kwh"] / count if count else None
    device = _reference_result(root, trial["device_idle_run_id"])
    loaded = _reference_result(root, trial["loaded_idle_run_id"])
    return {
        "gross_j_per_tweet": gross,
        "device_idle_adjusted_j_per_tweet": _adjusted(native, device, host, count),
        "loaded_idle_adjusted_j_per_tweet": _adjusted(native, loaded, host, count),
    }


def _base_result(trial: dict, cfg: dict, dataset_info: dict, host: dict, native: dict,
                 task: dict, adapter_metadata: dict) -> dict:
    return {
        "run_id": trial["run_id"], "trial": trial, "phase": trial["phase"],
        "kind": trial["kind"], "model": trial["model"], "batch_size": trial["batch_size"],
        "rep": trial["rep"], "config_sha256": trial["config_sha256"],
        "dataset_sha256": trial["dataset_sha256"], "dataset": dataset_info, "host": host,
        "adapter": adapter_metadata, "native_task": task, "energy": native,
        "gate": _gate(trial, cfg, native), "valid": True,
    }


def _start(root: Path, trial: dict) -> tuple[storage.Run, Path]:
    run = storage.Run(root, trial)
    run.recover()
    return run, run.start()


def _publish(run: storage.Run, attempt: Path, result: dict) -> None:
    storage.atomic_json(attempt / "result.json", result)
    run.finish("complete")


def _measure_idle(root: Path, trial: dict, cfg: dict, dataset_info: dict, train_count: int,
                  host: dict, active: dict, adapter_metadata: dict | None = None,
                  check_after=None) -> None:
    run = active.get(trial["run_id"])
    if run is None:
        run, attempt = _start(root, trial)
        active[run.trial["run_id"]] = run
    else:
        attempt = run.path / run.state["active_attempt"]
    outside_start = perf_counter()
    sleep(_secs(cfg, "settle"))
    settle_s = perf_counter() - outside_start
    task = _native_task(trial, cfg, dataset_info, train_count)
    meter = energy.Meter(attempt / "energy", task, cfg["energy"])
    _, window_s, native = meter.capture(lambda: sleep(_secs(cfg, trial["kind"])))
    if check_after is not None:
        check_after()
    result = _base_result(
        trial, cfg, dataset_info, host, native, task,
        adapter_metadata or {"adapter": None, "role": "unloaded device idle"},
    )
    result.update(
        counts=_empty_counts(), quality={},
        performance={"window_s": window_s, "throughput_tweets_s": None,
                     "latency_p50_s": None, "latency_p95_s": None},
        derived=_derived(trial, native, root, host),
        outside_window_s={"settle": settle_s},
    )
    _publish(run, attempt, result)
    active.pop(trial["run_id"], None)


def _measure_training(root: Path, trial: dict, cfg: dict, dataset_info: dict, train_count: int,
                      train: list, validation: list, host: dict, active: dict, adapter,
                      provenance: dict | None = None, check_after=None) -> dict:
    run = active.get(trial["run_id"])
    if run is None:
        run, attempt = _start(root, trial)
        active[run.trial["run_id"]] = run
    else:
        attempt = run.path / run.state["active_attempt"]
    started = perf_counter()
    sleep(_secs(cfg, "settle"))
    settle_s = perf_counter() - started
    task = _native_task(trial, cfg, dataset_info, train_count)
    meter = energy.Meter(attempt / "energy", task, cfg["energy"])
    metadata, window_s, native = meter.capture(lambda: adapter.setup(train, validation))
    if check_after is not None:
        check_after()
    if provenance is not None:
        metadata = {**metadata, "engine_session": provenance}
    counts = _empty_counts(train_count)
    counts.update(valid=None, correct=None)
    result = _base_result(trial, cfg, dataset_info, host, native, task, metadata)
    result.update(
        counts=counts, quality={},
        performance={"window_s": window_s, "throughput_tweets_s": train_count / window_s,
                     "latency_p50_s": None, "latency_p95_s": None},
        derived=_derived(trial, native, root, host),
        outside_window_s={"settle": settle_s},
    )
    if native["count"] != 0 or counts["submitted"] != train_count or metadata.get("train_rows") != train_count:
        raise ValueError("training actual/native count mismatch")
    _publish(run, attempt, result)
    active.pop(trial["run_id"], None)
    return metadata


def _measure_inference(root: Path, trial: dict, cfg: dict, dataset_info: dict, train_count: int,
                       panel: list, host: dict, active: dict, adapter, adapter_metadata: dict,
                       outside: dict, check_after=None) -> None:
    run = active.get(trial["run_id"])
    if run is None:
        run, attempt = _start(root, trial)
        active[run.trial["run_id"]] = run
    else:
        attempt = run.path / run.state["active_attempt"]
    started = perf_counter()
    sleep(_secs(cfg, "settle"))
    outside["settle"] = perf_counter() - started
    task = _native_task(trial, cfg, dataset_info, train_count)
    meter = energy.Meter(attempt / "energy", task, cfg["energy"])
    rows, window_s, native = meter.capture(
        lambda: workload.run_workload(adapter, panel, trial["batch_size"], trial["passes"])
    )
    if check_after is not None:
        check_after()
    started = perf_counter()
    predictions = attempt / "predictions.jsonl"
    digest = workload.persist(predictions, rows)
    outside["persist"] = perf_counter() - started
    started = perf_counter()
    counts, quality, performance = workload.audit(
        predictions, test_ids=dataset_info["test_ids"],
        gold_by_id={row[0]: dataset.LABELS[row[2]] for row in panel},
        batch_size=trial["batch_size"], passes=trial["passes"],
        adapter_kind=_audit_kind(cfg, trial), window_s=window_s,
    )
    outside["audit"] = perf_counter() - started
    counts = _inference_counts(counts)
    actual_requests = counts.get("qev_request_count", counts["submitted"])
    if not (counts["submitted"] == len(rows) == trial["expected_count"]
            and native["count"] == actual_requests == trial["native_count"]):
        raise ValueError("inference saved/actual tweet or native request count mismatch")
    result = _base_result(trial, cfg, dataset_info, host, native, task, adapter_metadata)
    result.update(
        counts=counts, quality=quality, performance=performance,
        derived=_derived(trial, native, root, host),
        predictions={"file": predictions.name, "sha256": digest, "rows": len(rows)},
        outside_window_s=outside,
    )
    _publish(run, attempt, result)
    active.pop(trial["run_id"], None)


def _state_status(root: Path, trial: dict) -> str:
    path = root / trial["run_id"] / "state.json"
    return storage.load(path)["status"] if path.exists() else "pending"


def _eligible(status: str, retry_failed: bool, retry_skipped: bool) -> bool:
    return status in {"pending", "incomplete", "running"} or (
        status == "failed" and retry_failed
    ) or (status == "skipped" and retry_skipped)


def select_trials(plan: dict, root: Path, *, model: str | None = None, kind: str | None = None,
                  width: int | None = None, limit_new: int | None = None,
                  retry_failed: bool = False, retry_skipped: bool = False) -> list[dict]:
    """Select eligible planned trials in plan order, including shared dependencies.

    ``width`` matches the planned model width (``batch_size``); ``0`` selects the
    shared calibration/training rows. Sharing is a plan property: the selected
    trial's same-session device idle, training, and same-width loaded idle are
    still added, and the result keeps plan order (the seeded v2 session/width
    order). ``limit_new`` is applied last, exactly as without ``width``.
    """
    by_id = {trial["run_id"]: trial for trial in plan["trials"]}
    statuses = {trial["run_id"]: _state_status(root, trial) for trial in plan["trials"]}
    selected = {
        trial["run_id"] for trial in plan["trials"]
        if (model is None or trial["model"] == model)
        and (kind is None or trial["kind"] == kind)
        and (width is None or trial["batch_size"] == width)
        and _eligible(statuses[trial["run_id"]], retry_failed, retry_skipped)
    }
    # Calibration/training dependencies precede inference in every block. Add
    # them even under --kind inference, then apply the global new-attempt cap.
    changed = True
    while changed:
        changed = False
        for run_id in tuple(selected):
            for field in REFERENCE_FIELDS:
                reference = by_id[run_id].get(field)
                if reference and reference not in selected and statuses[reference] != "complete" and _eligible(
                        statuses[reference], retry_failed, retry_skipped):
                    selected.add(reference)
                    changed = True
    ordered = [trial for trial in plan["trials"] if trial["run_id"] in selected]
    return ordered if limit_new is None else ordered[:limit_new]


def _check_stop(path: Path | None) -> None:
    if path is not None and path.exists():
        raise StopRequested(f"stop file exists: {path}")


def _skip(root: Path, trial: dict, reason: str, receipt: dict | None = None) -> None:
    run, attempt = _start(root, trial)
    if receipt is not None:
        storage.atomic_json(attempt / "preflight.json", receipt)
    run.finish("skipped", reason)


def _failure_status(exc: BaseException, close_error: BaseException | None, attempt: Path | None) -> tuple[str, str]:
    reason = f"{type(exc).__name__}: {exc}"
    if close_error is not None:
        reason += f"; adapter close failed: {type(close_error).__name__}: {close_error}"
    if isinstance(exc, (StopRequested, SignalReceived, KeyboardInterrupt)):
        return "incomplete", reason
    if close_error is not None:
        return "failed", reason
    if isinstance(exc, Unavailable):
        return "skipped", reason
    if attempt is not None and hardware.looks_like_oom(exc, attempt):
        return "skipped", "OOM after clean adapter shutdown: " + reason
    return "failed", reason


def _check_packed_context(adapter, panel: list, warmup: list, width: int,
                          path: Path, stop_file: Path | None) -> list[dict]:
    """Tokenize every panel and warmup envelope once, before measuring inference."""
    checks: list[dict] = []
    try:
        for split, rows in (("test", panel), ("warmup", warmup)):
            for group_index, group in enumerate(workload.groups(rows, width)):
                _check_stop(stop_file)
                try:
                    detail = adapter.check_context([row[1] for row in group])
                except Unavailable as exc:
                    checks.append({"split": split, "group_index": group_index,
                                   "detail": getattr(exc, "context_receipt", None), "error": str(exc)})
                    raise
                if detail.get("fit") is not True or detail.get("model_forwarding") is not False:
                    raise ValueError("packed context preflight did not confirm fit without forwarding")
                checks.append({"split": split, "group_index": group_index, "detail": detail})
    finally:
        storage.atomic_json(path, checks)
    return checks


def _owned_snapshot(adapter=None) -> dict:
    """Reject foreign compute; only the current process and owned engine PGID qualify."""
    host = hardware.snapshot()
    if host.get("gpu") and host.get("gpu_compute_apps") is None:
        raise Unavailable("GPU compute-app query failed; measurement refused")
    process = getattr(adapter, "_process", None) if adapter is not None else None
    pid = getattr(adapter, "compute_process_id", None) if adapter is not None else None
    if pid is None and process is not None and process.poll() is None:
        pid = process.pid
    foreign = []
    for line in (host.get("gpu_compute_apps") or "").splitlines():
        try:
            app_pid = int(line.split(",", 1)[0].strip())
            owned = app_pid == os.getpid() or (pid is not None and (app_pid == pid or os.getpgid(app_pid) == pid))
        except (ValueError, ProcessLookupError, PermissionError, OSError):
            owned = False
        if not owned:
            foreign.append(line)
    if foreign:
        raise Unavailable("foreign GPU compute app; measurement refused: " + "; ".join(foreign))
    if adapter is not None and getattr(adapter, "_closed", False):
        raise RuntimeError("owned engine is closed")
    if process is not None and process.poll() is not None:
        raise RuntimeError("owned engine process exited")
    return host


def _new_segment(root: Path, session: dict) -> tuple[Path, dict]:
    directory = root / session["session_id"] / "engine"
    directory.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        name = f"segment-{number:04d}"
        path = directory / name
        try:
            path.mkdir()
            break
        except FileExistsError:
            number += 1
    storage.sync_dir(directory)
    provenance = {"session_id": session["session_id"], "segment_id": name,
                  "log_dir": str(path.relative_to(root)), "model": session["model"],
                  "rep": session["rep"], "device_idle_policy": "shared planned session baseline across segments"}
    storage.atomic_json(path / "segment.json", {**provenance, "started_at": storage.now()})
    return path, provenance


def _execute_session(root: Path, plan: dict, session: dict, selected: list[dict], data: tuple,
                     stop_file: Path | None, *, progress: "Progress | None" = None,
                     work: dict[str, int] | None = None) -> str:
    cfg, dataset_info, train, validation, panel, warmup = data
    progress = progress if progress is not None else Progress()
    if work is None:
        work = {trial["run_id"]: index for index, trial in enumerate(selected, 1)}
    total = len(work)

    def starting(trial: dict) -> None:
        progress.event("work-start", work=work[trial["run_id"]], total=total, **_trial_fields(trial))

    def finished(trial: dict) -> None:
        progress.event("work-done", work=work[trial["run_id"]], total=total,
                       status="complete", **_trial_fields(trial))

    active: dict[str, storage.Run] = {}
    adapter = None
    segment_path = None
    receipt = None
    setup_done = False
    current = None
    outcome = "complete"
    selected_by_id = {t["run_id"]: t for t in selected}
    try:
        _check_stop(stop_file)
        host = _owned_snapshot()
        spec = _runtime_spec(cfg["models"][session["model"]])
        supported = configuration.supported_batches(spec)
        # Unsupported width pairs never cause a model load or a meaningless baseline.
        model_selected = [t for t in selected if t["kind"] != "device_idle"]
        valid_widths = [w for w in session["widths"] if supported is None or w in supported]
        if model_selected:
            evidence, reason = hardware.preflight(spec, host)
            receipt = {"host": host, "preflight": evidence}
            if reason:
                raise Unavailable(reason)
        else:
            evidence = None
        unsupported_ids = set()
        for t in model_selected:
            if t["batch_size"] and t["batch_size"] not in valid_widths:
                _skip(root, t, f"{spec['adapter']} does not support independent batch {t['batch_size']}", receipt)
                unsupported_ids.add(t["run_id"])
        model_selected = [t for t in model_selected if t["run_id"] not in unsupported_ids]
        idle = next((t for t in selected if t["kind"] == "device_idle"), None)
        if idle is not None:
            if not model_selected and any(t["kind"] != "device_idle" for t in selected):
                _skip(root, idle, "no supported selected width", receipt)
            else:
                _check_stop(stop_file)
                starting(idle)
                _measure_idle(root, idle, cfg, dataset_info, plan["train_count"], _owned_snapshot(), active,
                              check_after=_owned_snapshot)
                finished(idle)
        if not model_selected:
            return outcome
        # Do not consume new attempts for trials whose prerequisites are still missing.
        runnable = []
        for t in model_selected:
            missing = [t[f] for f in REFERENCE_FIELDS if t.get(f)
                       and _state_status(root, {"run_id": t[f]}) != "complete"
                       and t[f] not in selected_by_id]
            if missing:
                _skip(root, t, "required calibration/training reference is not complete: " + ", ".join(missing))
            else:
                runnable.append(t)
        if not runnable:
            return outcome
        _check_stop(stop_file)
        host = _owned_snapshot()
        evidence, reason = hardware.preflight(spec, host)
        receipt = {"host": host, "preflight": evidence}
        if reason:
            raise Unavailable(reason)
        segment_path, provenance = _new_segment(root, session)
        progress.event("engine", state="start", model=session["model"], rep=session["rep"],
                       segment=provenance["segment_id"])
        if spec["adapter"] == "vllm":
            # Capacity for the largest scheduled supported width; not the trial width.
            spec["submission_width"] = max(valid_widths)
        current = runnable[0]
        run, _ = _start(root, current)
        active[current["run_id"]] = run
        adapter = _create_adapter(spec, segment_path)
        training = next((t for t in runnable if t["kind"] == "training"), None)
        outside_setup = {"preflight": evidence}
        if training is not None:
            current = training
            _check_stop(stop_file)
            starting(training)
            metadata = _measure_training(root, training, cfg, dataset_info, plan["train_count"],
                                         train, validation, _owned_snapshot(adapter), active, adapter, provenance,
                                         check_after=lambda: _owned_snapshot(adapter))
            finished(training)
            outside_setup.update(setup_s=None, setup_measured_in=training["run_id"])
        else:
            started = perf_counter()
            metadata = adapter.setup(train, validation)
            metadata = {**metadata, "engine_session": provenance}
            outside_setup.update(setup_s=perf_counter() - started, setup_measured_in=None,
                                 previous_training_run_id=current.get("training_run_id"))
        setup_done = True
        progress.event("engine", state="ready", model=session["model"], rep=session["rep"],
                       segment=provenance["segment_id"], setup_s=outside_setup.get("setup_s"))
        _owned_snapshot(adapter)
        for width in session["widths"]:
            pair = [t for t in runnable if t["batch_size"] == width and t["kind"] in {"loaded_idle", "inference"}]
            if not pair:
                continue
            _check_stop(stop_file)
            current = next((t for t in pair if t["kind"] == "inference"), pair[0])
            if any(t.get(f) and _state_status(root, {"run_id": t[f]}) != "complete"
                   for t in pair for f in ("device_idle_run_id", "training_run_id")):
                for t in pair:
                    if t["run_id"] in active:
                        active.pop(t["run_id"]).finish("skipped", "required session dependency is not complete")
                    else:
                        _skip(root, t, "required session dependency is not complete")
                continue
            _owned_snapshot(adapter)
            width_receipt = adapter.set_submission_width(width) if spec["adapter"] == "vllm" else {}
            if spec["adapter"] == "vllm" and (not isinstance(width_receipt, dict) or width_receipt.get("submission_width") != width):
                raise ValueError("vLLM width selection did not return the requested policy receipt")
            width_metadata = {**metadata, "trial_width_policy": {
                **width_receipt, "submission_width": width,
                "semantics": "packed QEV envelopes" if _packed_qev(spec) else "independent tweet groups"}}
            outside = dict(outside_setup)
            inference = next((t for t in pair if t["kind"] == "inference"), None)
            loaded = next((t for t in pair if t["kind"] == "loaded_idle"), None)
            if spec["adapter"] == "qev":
                # Also gate a loaded-only selection against its paired inference workload.
                context_trial = inference or loaded
                run = active.get(context_trial["run_id"])
                if run is None:
                    run, attempt = _start(root, context_trial)
                    active[context_trial["run_id"]] = run
                else:
                    attempt = run.path / run.state["active_attempt"]
                context_file = attempt / "qev_context.json"
                started = perf_counter()
                try:
                    context = _check_packed_context(adapter, panel, warmup, width, context_file, stop_file)
                except Unavailable as exc:
                    if not isinstance(getattr(exc, "context_receipt", None), dict) or exc.context_receipt.get("fit") is not False:
                        raise
                    _owned_snapshot(adapter)  # A closed/failed worker is session-fatal.
                    for t in pair:
                        if t["run_id"] in active:
                            active.pop(t["run_id"]).finish("skipped", str(exc))
                        else:
                            _skip(root, t, "paired context preflight: " + str(exc))
                    continue
                outside["context_preflight_s"] = perf_counter() - started
                outside["context_preflight"] = {"file": context_file.name, "groups": len(context),
                                                "sha256": storage.content_hash(context_file)}
            started = perf_counter()
            if warmup:
                workload.run_workload(adapter, warmup, width, 1)
            outside.update(warmup_s=perf_counter() - started, warmup_rows=len(warmup))
            _owned_snapshot(adapter)
            if loaded is not None:
                current = loaded
                _check_stop(stop_file)
                starting(loaded)
                _measure_idle(root, loaded, cfg, dataset_info, plan["train_count"], _owned_snapshot(adapter), active, width_metadata,
                              check_after=lambda: _owned_snapshot(adapter))
                finished(loaded)
                _owned_snapshot(adapter)
            if inference is not None:
                current = inference
                _check_stop(stop_file)
                # A completed loaded baseline may belong to a prior segment on resume.
                baseline = _reference_result(root, inference.get("loaded_idle_run_id"))
                width_metadata["loaded_idle_engine_session"] = (baseline or {}).get("adapter", {}).get("engine_session")
                starting(inference)
                _measure_inference(root, inference, cfg, dataset_info, plan["train_count"], panel,
                                   _owned_snapshot(adapter), active, adapter, width_metadata, outside,
                                   check_after=lambda: _owned_snapshot(adapter))
                finished(inference)
                _owned_snapshot(adapter)
        adapter.close()
        adapter = None
        return outcome
    except BaseException as exc:
        close_error = None
        if adapter is not None:
            try:
                adapter.close()
            except BaseException as close_exc:
                close_error = close_exc
            adapter = None
        # After successful setup, do not scan shared logs containing earlier-width warnings.
        error_path = segment_path if not setup_done else (
            root / current["run_id"] / "no-current-logs" if current is not None else root / "no-current-logs")
        status, reason = _failure_status(exc, close_error, error_path)
        if status == "incomplete":
            outcome = "interrupted" if isinstance(exc, (SignalReceived, KeyboardInterrupt)) else "stopped"
        else:
            outcome = status
        if current is not None and current["run_id"] not in active and _state_status(root, current) in {"pending", "running"}:
            run, _ = _start(root, current)
            active[current["run_id"]] = run
        for run in list(active.values()):
            if run.state["status"] == "running":
                attempt = run.path / run.state["active_attempt"]
                storage.atomic_json(attempt / "failure.json", {"error": reason, "status": status, "preflight": receipt})
                run.finish(status, reason)
        if status != "incomplete":
            for trial in selected:
                if _state_status(root, trial) in {"pending", "running"}:
                    _skip(root, trial, "session dependency/engine did not complete: " + reason, receipt)
        return outcome
    finally:
        if adapter is not None:
            adapter.close()
        if segment_path is not None:
            storage.atomic_json(segment_path / "end.json", {"finished_at": storage.now(), "outcome": outcome})


def run_selected(root: Path, plan: dict, selected: list[dict], data: tuple,
                 stop_file: Path | None = None, progress: "Progress | None" = None) -> dict:
    if plan.get("format_version") != 2 or plan.get("execution_protocol") != "engine-session-v1":
        raise ValueError("archived format-1 plans are read/verify only; execution requires engine-session-v1")
    progress = progress if progress is not None else Progress()
    work = {trial["run_id"]: index for index, trial in enumerate(selected, 1)}
    by_session: dict[str, list[dict]] = {}
    for trial in selected:
        by_session.setdefault(trial["session_id"], []).append(trial)
    interrupted = stopped = False
    for position, session in enumerate(plan["sessions"], 1):
        trials = by_session.get(session["session_id"])
        if not trials:
            continue
        progress.event("session", index=position, total=len(plan["sessions"]),
                       model=session["model"], rep=session["rep"], widths=session["widths"],
                       planned=len(session["trials"]), selected=len(trials))
        outcome = _execute_session(root, plan, session, trials, data, stop_file,
                                   progress=progress, work=work)
        counts = progress.report_outcomes(root, trials, work, len(selected))
        progress.event("session-done", index=position, total=len(plan["sessions"]),
                       model=session["model"], rep=session["rep"], outcome=outcome,
                       complete=counts.get("complete", 0), failed=counts.get("failed", 0),
                       skipped=counts.get("skipped", 0), incomplete=counts.get("incomplete", 0),
                       pending=counts.get("pending", 0))
        interrupted |= outcome == "interrupted"
        stopped |= outcome == "stopped"
        if interrupted or stopped:
            break
    progress.report_outcomes(root, selected, work, len(selected))
    progress.event("run-done", selected=len(selected), planned=len(plan["trials"]),
                   interrupted=interrupted, stopped=stopped)
    return {"selected": [t["run_id"] for t in selected], "interrupted": interrupted, "stopped": stopped}


def status_report(root: Path, plan: dict) -> dict:
    counts = {status: 0 for status in ("pending", "running", "complete", "incomplete", "failed", "skipped")}
    runs = []
    for trial in plan["trials"]:
        path = root / trial["run_id"] / "state.json"
        state = storage.load(path) if path.exists() else None
        status = state["status"] if state else "pending"
        counts[status] = counts.get(status, 0) + 1
        runs.append({
            "run_id": trial["run_id"], "kind": trial["kind"], "model": trial["model"],
            "batch_size": trial["batch_size"], "rep": trial["rep"], "status": status,
            "reason": state.get("reason") if state else None,
            "active_attempt": state.get("active_attempt") if state else None,
        })
    return {"root": str(root), "phase": plan["phase"], "counts": counts, "runs": runs}


def _attempt_path(root: Path, state: dict) -> Path:
    name = state["active_attempt"]
    if not isinstance(name, str) or not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("invalid active_attempt")
    return root / state["run_id"] / name


def _derived_from_results(trial: dict, results: dict[str, dict]) -> dict:
    count = trial["expected_count"]
    native = results[trial["run_id"]]["energy"]
    host = results[trial["run_id"]].get("host", {})
    device = results.get(trial["device_idle_run_id"])
    loaded = results.get(trial["loaded_idle_run_id"])
    return {
        "gross_j_per_tweet": energy.J_PER_KWH * native["energy_kwh"] / count if count else None,
        "device_idle_adjusted_j_per_tweet": _adjusted(native, device, host, count),
        "loaded_idle_adjusted_j_per_tweet": _adjusted(native, loaded, host, count),
    }


def verify_plan(root: Path, plan: dict, data: tuple) -> dict:
    """Read-only independent validation of state, native energy and predictions."""
    cfg, dataset_info, train, _, _, _ = data
    inconsistencies: list[str] = []
    results: dict[str, dict] = {}
    low_sampling = []
    expected_ids = {trial["run_id"] for trial in plan["trials"]}
    for path in sorted(root.glob("*/state.json")):
        if path.parent.name not in expected_ids:
            inconsistencies.append(f"unexpected run state: {path.parent.name}")
    for trial in plan["trials"]:
        run_id = trial["run_id"]
        state_path = root / run_id / "state.json"
        if not state_path.exists():
            continue
        try:
            state = storage.load(state_path)
            if state.get("run_id") != run_id or state.get("trial") != trial:
                raise ValueError("state identity/trial differs from deterministic plan")
            if state.get("status") != "complete":
                continue
            attempt = _attempt_path(root, state)
            result = storage.load(attempt / "result.json")
            if result.get("run_id") != run_id or result.get("trial") != trial or result.get("valid") is not True:
                raise ValueError("result identity/trial/valid flag mismatch")
            if plan.get("format_version") == 2 and trial["kind"] != "device_idle":
                provenance = result.get("adapter", {}).get("engine_session") or {}
                segment = provenance.get("segment_id")
                expected_dir = Path(trial["session_id"]) / "engine" / str(segment)
                if (not isinstance(segment, str) or not segment.startswith("segment-")
                        or Path(segment).name != segment or provenance.get("session_id") != trial["session_id"]
                        or provenance.get("model") != trial["model"] or provenance.get("rep") != trial["rep"]
                        or provenance.get("log_dir") != str(expected_dir)):
                    raise ValueError("engine segment provenance mismatch")
                saved_segment = storage.load(root / expected_dir / "segment.json")
                if any(saved_segment.get(k) != v for k, v in provenance.items()):
                    raise ValueError("engine segment receipt mismatch")
                if trial["kind"] in {"loaded_idle", "inference"} and result["adapter"].get("trial_width_policy", {}).get("submission_width") != trial["batch_size"]:
                    raise ValueError("actual trial width policy mismatch")
            task = _native_task(trial, cfg, dataset_info, plan["train_count"])
            if result.get("native_task") != task:
                raise ValueError("saved native task differs from deterministic task")
            native = energy.validate_run(attempt / "energy", task)
            if result.get("energy") != native:
                raise ValueError("result energy differs from native JSON/CSV revalidation")
            counts = result.get("counts") or {}
            if counts.get("submitted") != trial["expected_count"] or native["count"] != trial["native_count"]:
                raise ValueError("actual/native expected count mismatch")
            if result.get("gate") != _gate(trial, cfg, native):
                raise ValueError("sampling/duration gate mismatch")
            if trial["kind"] == "inference":
                prediction_info = result.get("predictions") or {}
                if prediction_info.get("file") != "predictions.jsonl":
                    raise ValueError("unexpected predictions location")
                predictions = attempt / "predictions.jsonl"
                if storage.content_hash(predictions) != prediction_info.get("sha256"):
                    raise ValueError("prediction digest mismatch")
                actual_counts, quality, performance = workload.audit(
                    predictions, test_ids=dataset_info["test_ids"],
                    gold_by_id={row[0]: dataset.LABELS[row[2]] for row in data[4]},
                    batch_size=trial["batch_size"], passes=trial["passes"],
                    adapter_kind=_audit_kind(cfg, trial), window_s=result["performance"]["window_s"],
                )
                if actual_counts.get("qev_request_count", actual_counts["submitted"]) != native["count"]:
                    raise ValueError("native request count differs from audited envelope count")
                if counts != _inference_counts(actual_counts) or result.get("quality") != quality or result.get("performance") != performance:
                    raise ValueError("saved prediction audit differs from exported metrics")
                if prediction_info.get("rows") != actual_counts["submitted"]:
                    raise ValueError("prediction row metadata mismatch")
                if (_audit_kind(cfg, trial) == "qev-packed" or
                        (plan.get("format_version") == 2 and trial["adapter"] == "qev")):
                    context = result["outside_window_s"]["context_preflight"]
                    expected = [(split, i, len(group)) for split, rows in (("test", data[4]), ("warmup", data[5]))
                                for i, group in enumerate(workload.groups(rows, trial["batch_size"]))]
                    path = attempt / "qev_context.json"
                    if (context.get("file") != path.name or context.get("groups") != len(expected)
                            or context.get("sha256") != storage.content_hash(path)):
                        raise ValueError("packed context receipt identity mismatch")
                    checks = storage.load(path)
                    if ([(c["split"], c["group_index"], c["detail"]["width"]) for c in checks] != expected
                            or any(c["detail"].get("fit") is not True
                                   or c["detail"].get("model_forwarding") is not False for c in checks)):
                        raise ValueError("packed context receipt group or fit mismatch")
            elif trial["kind"] == "training":
                if result.get("adapter", {}).get("train_rows") != len(train):
                    raise ValueError("training row metadata mismatch")
            results[run_id] = result
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            inconsistencies.append(f"{run_id}: {type(exc).__name__}: {exc}")
    for trial in plan["trials"]:
        run_id = trial["run_id"]
        if run_id not in results:
            continue
        for field in REFERENCE_FIELDS:
            reference = trial.get(field)
            if reference and reference not in results:
                inconsistencies.append(f"{run_id}: required complete/valid reference missing: {reference}")
        expected_derived = _derived_from_results(trial, results)
        if results[run_id].get("derived") != expected_derived:
            inconsistencies.append(f"{run_id}: derived energy intensity mismatch")
        if plan.get("format_version") == 2 and trial["kind"] == "inference":
            baseline = results.get(trial.get("loaded_idle_run_id")) or {}
            expected_segment = baseline.get("adapter", {}).get("engine_session")
            if results[run_id].get("adapter", {}).get("loaded_idle_engine_session") != expected_segment:
                inconsistencies.append(f"{run_id}: loaded idle segment provenance mismatch")
        if trial["kind"] == "inference" and not results[run_id]["gate"]["meets_minimum_window"]:
            gate = results[run_id]["gate"]
            low_sampling.append({"run_id": run_id, "window_ratio": gate["window_ratio"],
                                 "sample_ratio": gate["sample_ratio"]})
    report = status_report(root, plan)
    report.update(ok=not inconsistencies, inconsistencies=inconsistencies,
                  low_sampling=low_sampling, valid_complete=len(results))
    return report


@contextmanager
def _signal_handlers():
    previous = {}

    def handler(signum, _frame):
        raise SignalReceived(signum)

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, handler)
        yield
    finally:
        for signum, old in previous.items():
            signal.signal(signum, old)


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _non_negative(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(name: str, *, config: bool = True):
        command = sub.add_parser(name)
        command.add_argument("--phase", required=True, help="phase name in the selected configuration")
        if config:
            command.add_argument("--config", type=Path, default=ROOT / "experiment.json")
        command.add_argument("--out", type=Path, help="default: artifacts/<phase>/")
        return command

    common("plan")
    run = common("run")
    run.add_argument("--limit-new", type=_positive, help="maximum newly attempted planned runs, including calibration")
    run.add_argument("--model", help="select one configured model id")
    run.add_argument("--kind", choices=KIND_ORDER)
    run.add_argument("--width", type=_non_negative,
                     help="select one planned model width; 0 selects the shared calibration/training rows")
    run.add_argument("--retry-failed", action="store_true")
    run.add_argument("--retry-skipped", action="store_true")
    run.add_argument("--stop-file", type=Path)
    run.add_argument("--progress", action="store_true",
                     help="stream opt-in progress lines to stderr; the stdout JSON report is unchanged")
    common("status", config=False)
    common("verify")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = _output_root(args.phase, args.out)
    try:
        if args.command == "status":
            report = status_report(root, storage.load(root / "plan.json"))
            print(json.dumps(report, indent=2, allow_nan=False))
            return 0
        cfg, dataset_info, train, validation, panel, warmup = _configure(args.phase, args.config)
        data = (cfg, dataset_info, train, validation, panel, warmup)
        if args.command == "plan":
            with storage.measurement_lock(LOCK):
                plan = ensure_plan(root, cfg, dataset_info, len(train))
            print(json.dumps({"plan": str(root / "plan.json"), "runs": len(plan["trials"]),
                              "blocks": len(plan["blocks"])}, indent=2))
            return 0
        if args.command == "verify":
            generated = build_plan(cfg, dataset_info, len(train))
            saved = storage.load(root / "plan.json")
            report = verify_plan(root, generated, data)
            if saved != generated:
                report["ok"] = False
                report["inconsistencies"].insert(0, "saved plan.json differs from deterministic selected configuration")
            print(json.dumps(report, indent=2, allow_nan=False))
            return 0 if report["ok"] else 1
        if args.command == "run":
            if cfg.get("execution_protocol") != "engine-session-v1":
                raise ValueError("archived configurations are read/verify only; run requires engine-session-v1")
            progress = Progress(sys.stderr) if args.progress else Progress()
            with storage.measurement_lock(LOCK), _signal_handlers():
                plan = ensure_plan(root, cfg, dataset_info, len(train))
                selected = select_trials(
                    plan, root, model=args.model, kind=args.kind, width=args.width,
                    limit_new=args.limit_new, retry_failed=args.retry_failed,
                    retry_skipped=args.retry_skipped,
                )
                progress.event("run", phase=cfg["phase"], root=str(root),
                               planned=len(plan["trials"]), selected=len(selected),
                               model=args.model, kind=args.kind, width=args.width)
                summary = run_selected(root, plan, selected, data, args.stop_file, progress=progress)
                summary.update(status_report(root, plan))
            counts = summary["counts"]
            progress.event("finished", complete=counts.get("complete"), failed=counts.get("failed"),
                           skipped=counts.get("skipped"), incomplete=counts.get("incomplete"),
                           pending=counts.get("pending"), interrupted=summary["interrupted"],
                           stopped=summary["stopped"])
            print(json.dumps(summary, indent=2, allow_nan=False))
            if summary["interrupted"]:
                return 130
            return 1 if summary["counts"].get("failed") else 0
        raise AssertionError(args.command)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"benchmark: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
