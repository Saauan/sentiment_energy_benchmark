#!/usr/bin/env python3
"""Exclusive, resumable orchestration for the sentiment energy benchmark.

``plan`` and ``verify`` never construct an adapter or start a measurement. ``run``
holds the project measurement lock for its whole lifetime. Completed attempts are
immutable; interrupted and incomplete attempts are resumed in fresh directories.
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
from typing import Any, Iterable

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
SUPPORTED_BATCHES = {"qev": (1,), "vllm": (1, 16)}
REFERENCE_FIELDS = ("device_idle_run_id", "training_run_id", "loaded_idle_run_id")


class StopRequested(BaseException):
    """A stop file appeared between measurement operations."""


class SignalReceived(BaseException):
    """SIGINT/SIGTERM is converted to an incomplete active attempt."""

    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(signal.Signals(signum).name)


def _configure(phase: str, config_path: Path):
    """Resolve the two external contracts exactly once for this command."""
    import configuration

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
    """Build deterministic seeded blocks; no wall-clock field enters identity."""
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
        supported = SUPPORTED_BATCHES.get(spec["adapter"])
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
            trial = {
                "run_id": "", "phase": cfg["phase"], "kind": kind, "model": model,
                "batch_size": batch, "rep": rep, "block_id": block_id,
                "block_index": block_index, "adapter": spec["adapter"],
                "device": "cuda" if _is_cuda(spec) else "cpu", "passes": passes,
                "expected_count": expected, "native_count": 0 if kind == "training" else expected,
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
                  host: dict, active: dict, adapter_metadata: dict | None = None) -> None:
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
                      train: list, validation: list, host: dict, active: dict, adapter) -> dict:
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
                       outside: dict) -> None:
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
    started = perf_counter()
    predictions = attempt / "predictions.jsonl"
    digest = workload.persist(predictions, rows)
    outside["persist"] = perf_counter() - started
    started = perf_counter()
    counts, quality, performance = workload.audit(
        predictions, test_ids=dataset_info["test_ids"],
        gold_by_id={row[0]: dataset.LABELS[row[2]] for row in panel},
        batch_size=trial["batch_size"], passes=trial["passes"],
        adapter_kind="vllm" if trial["adapter"] == "vllm" else "qev" if trial["adapter"] == "qev" else "classical",
        window_s=window_s,
    )
    outside["audit"] = perf_counter() - started
    counts = _inference_counts(counts)
    if not (native["count"] == counts["submitted"] == len(rows) == trial["expected_count"]):
        raise ValueError("inference saved/actual/native prediction count mismatch")
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
                  limit_new: int | None = None, retry_failed: bool = False,
                  retry_skipped: bool = False) -> list[dict]:
    by_id = {trial["run_id"]: trial for trial in plan["trials"]}
    statuses = {trial["run_id"]: _state_status(root, trial) for trial in plan["trials"]}
    selected = {
        trial["run_id"] for trial in plan["trials"]
        if (model is None or trial["model"] == model)
        and (kind is None or trial["kind"] == kind)
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


def _references_ready(root: Path, trial: dict, selected_ids: set[str]) -> bool:
    for field in REFERENCE_FIELDS:
        reference = trial.get(field)
        if reference and reference not in selected_ids and _state_status(root, {"run_id": reference}) != "complete":
            return False
    return True


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


def _preflight_block(cfg: dict, block: dict, host: dict) -> tuple[dict, dict, dict, str | None]:
    spec = _runtime_spec(cfg["models"][block["model"]])
    evidence, reason = hardware.preflight(spec, host)
    receipt = {"host": host, "preflight": evidence}
    supported = SUPPORTED_BATCHES.get(spec["adapter"])
    if reason is None and supported is not None and block["batch_size"] not in supported:
        reason = f"{spec['adapter']} does not support independent batch {block['batch_size']}"
        receipt["supported_batches"] = supported
    return spec, evidence, receipt, reason


def _execute_block(root: Path, plan: dict, block: dict, selected: list[dict], data: tuple,
                   stop_file: Path | None) -> str:
    cfg, dataset_info, train, validation, panel, warmup = data
    active: dict[str, storage.Run] = {}
    adapter = None
    adapter_attempt = None
    receipt = None
    outcome = "complete"
    try:
        host = hardware.snapshot()
        if host.get("gpu_compute_apps"):
            raise Unavailable("other active GPU compute app; block refused")
        model_trials = [t for t in selected if t["kind"] != "device_idle"]
        # Refuse infeasible model conditions before spending an idle window.
        # A calibration-only selection deliberately skips this model preflight.
        if model_trials:
            _, _, receipt, reason = _preflight_block(cfg, block, host)
            if reason:
                raise Unavailable(reason)
        idle = next((t for t in selected if t["kind"] == "device_idle"), None)
        if idle is not None:
            _check_stop(stop_file)
            _measure_idle(root, idle, cfg, dataset_info, plan["train_count"], host, active)
        selected_ids = {trial["run_id"] for trial in selected}
        runnable = []
        for trial in model_trials:
            if _references_ready(root, trial, selected_ids):
                runnable.append(trial)
            else:
                _skip(root, trial, "a required calibration/training reference is not complete")
        if not runnable:
            return "complete"
        _check_stop(stop_file)
        host = hardware.snapshot()
        if host.get("gpu_compute_apps"):
            raise Unavailable("other active GPU compute app; model setup refused")
        # Recheck free VRAM and competing compute after the idle window.
        spec, evidence, receipt, reason = _preflight_block(cfg, block, host)
        if reason:
            raise Unavailable(reason)
        home = runnable[0]
        run, attempt = _start(root, home)
        active[home["run_id"]] = run
        adapter_attempt = attempt
        adapter = _create_adapter(spec, attempt)
        metadata: dict = {}
        outside: dict = {"preflight": evidence}
        training = next((t for t in runnable if t["kind"] == "training"), None)
        inference = next((t for t in runnable if t["kind"] == "inference"), None)
        loaded = next((t for t in runnable if t["kind"] == "loaded_idle"), None)
        if training is not None:
            _check_stop(stop_file)
            metadata = _measure_training(
                root, training, cfg, dataset_info, plan["train_count"], train, validation,
                host, active, adapter,
            )
            outside["setup_s"] = None
            outside["setup_measured_in"] = training["run_id"]
        elif inference is not None or loaded is not None:
            started = perf_counter()
            metadata = adapter.setup(train, validation)
            outside["setup_s"] = perf_counter() - started
            outside["setup_measured_in"] = home.get("training_run_id")
        if inference is not None:
            _check_stop(stop_file)
            started = perf_counter()
            if warmup:
                workload.run_workload(adapter, warmup, inference["batch_size"], 1)
            outside["warmup_s"] = perf_counter() - started
            outside["warmup_rows"] = len(warmup)
        if loaded is not None:
            _check_stop(stop_file)
            _measure_idle(root, loaded, cfg, dataset_info, plan["train_count"], host, active, metadata)
        if inference is not None:
            _check_stop(stop_file)
            _measure_inference(
                root, inference, cfg, dataset_info, plan["train_count"], panel,
                host, active, adapter, metadata, outside,
            )
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
        status, reason = _failure_status(exc, close_error, adapter_attempt)
        if status == "incomplete":
            outcome = "interrupted" if isinstance(exc, (SignalReceived, KeyboardInterrupt)) else "stopped"
        for run in list(active.values()):
            if run.state["status"] == "running":
                attempt = run.path / run.state["active_attempt"]
                storage.atomic_json(attempt / "failure.json",
                                    {"error": reason, "status": status, "preflight": receipt})
                run.finish(status, reason)
        active.clear()
        if status != "incomplete":
            for trial in selected:
                if _state_status(root, trial) in {"pending", "running"}:
                    _skip(root, trial, "block dependency did not complete: " + reason, receipt)
        return outcome
    finally:
        if adapter is not None:
            adapter.close()


def run_selected(root: Path, plan: dict, selected: list[dict], data: tuple,
                 stop_file: Path | None = None) -> dict:
    by_block: dict[str, list[dict]] = {}
    for trial in selected:
        by_block.setdefault(trial["block_id"], []).append(trial)
    interrupted = False
    stopped = False
    for block in plan["blocks"]:
        trials = by_block.get(block["block_id"])
        if not trials:
            continue
        trials.sort(key=lambda trial: KIND_ORDER.index(trial["kind"]))
        outcome = _execute_block(root, plan, block, trials, data, stop_file)
        interrupted |= outcome == "interrupted"
        stopped |= outcome == "stopped"
        if interrupted or stopped:
            break
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
                    adapter_kind="vllm" if trial["adapter"] == "vllm" else "qev" if trial["adapter"] == "qev" else "classical",
                    window_s=result["performance"]["window_s"],
                )
                if counts != _inference_counts(actual_counts) or result.get("quality") != quality or result.get("performance") != performance:
                    raise ValueError("saved prediction audit differs from exported metrics")
                if prediction_info.get("rows") != actual_counts["submitted"]:
                    raise ValueError("prediction row metadata mismatch")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(name: str, *, config: bool = True):
        command = sub.add_parser(name)
        command.add_argument("--phase", required=True, help="gpu_pilot, cpu_pilot, or full")
        if config:
            command.add_argument("--config", type=Path, default=ROOT / "experiment.json")
        command.add_argument("--out", type=Path, help="default: artifacts/<phase>/")
        return command

    common("plan")
    run = common("run")
    run.add_argument("--limit-new", type=_positive, help="maximum newly attempted planned runs, including calibration")
    run.add_argument("--model", help="select one configured model id")
    run.add_argument("--kind", choices=KIND_ORDER)
    run.add_argument("--retry-failed", action="store_true")
    run.add_argument("--retry-skipped", action="store_true")
    run.add_argument("--stop-file", type=Path)
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
            with storage.measurement_lock(LOCK), _signal_handlers():
                plan = ensure_plan(root, cfg, dataset_info, len(train))
                selected = select_trials(
                    plan, root, model=args.model, kind=args.kind, limit_new=args.limit_new,
                    retry_failed=args.retry_failed, retry_skipped=args.retry_skipped,
                )
                summary = run_selected(root, plan, selected, data, args.stop_file)
                summary.update(status_report(root, plan))
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
