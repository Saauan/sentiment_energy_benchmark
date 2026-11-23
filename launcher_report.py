#!/usr/bin/env python3
"""Write RUN_ROOT/report/summary.md and analysis CSVs for a launcher-owned run.

Full runs also get report/inference_medians.csv: one row per planned model x width
inference condition, aggregated only from receipt-verified replicas. Read-only with
respect to benchmark artifacts; only RUN_ROOT/report/ is written. No model is loaded
and no measurement is started.

Usage:
    python launcher_report.py RUN_ROOT
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import storage
from analysis import load_artifacts, write_tables

STATUS_ORDER = ("complete", "pending", "running", "incomplete", "failed", "skipped", "invalid")
VERIFIED, FAILED, UNVERIFIED, NOT_APPLICABLE = "verified", "failed verification", "unverified", "not applicable"

# Display names follow the pinned protocol identities in RESULTS.md; CSV keeps model ids.
PINNED_DISPLAY_NAMES = {
    "svm": "TF–IDF + LinearSVC",
    "nb": "TF–IDF + multinomial NB",
    "fasttext": "fastText",
}
LATENCY_SCOPE_LABELS = {
    "adapter request": "adapter request",
    "closed-loop group completion": "group completion",
    "packed_request_completion": "packed envelope",
}


def _text(value: Any) -> str | None:
    return str(value).strip() or None if value is not None else None

def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None

def _count(value: Any) -> int | None:
    number = _num(value)
    return int(round(number)) if number is not None and number >= 0 else None

def _map(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}

def _load(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = storage.load(path)
    except FileNotFoundError:
        return None, "not found"
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return (value, None) if isinstance(value, dict) else (None, "JSON root must be an object")

def _stamp(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
    except OSError:
        return None

def _sig(value: float | None, digits: int = 4) -> str:
    """Format by significant digits so tiny energy is never rounded to zero."""
    if value is None:
        return "—"
    if value == 0:
        return "0"
    magnitude = math.floor(math.log10(abs(value)))
    decimals = digits - 1 - magnitude
    if decimals > 12:
        return f"{value:.{digits}g}"
    decimals = max(decimals, 0)
    return f"{value:,.{decimals}f}" if magnitude >= 0 else f"{value:.{decimals}f}"

def _ratio(numerator: int, denominator: int) -> str:
    return "—" if denominator <= 0 else f"{numerator}/{denominator}"

def _cell(value: Any) -> str:
    return ("—" if value is None else str(value)).replace("|", "\\|").replace("\n", " ")

def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    result = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    result.extend("| " + " | ".join(_cell(value) for value in row) + " |" for row in rows)
    return result

def _groups(items: Iterable[Any], key) -> dict[Any, list[Any]]:
    grouped: dict[Any, list[Any]] = defaultdict(list)
    for item in items:
        grouped[key(item)].append(item)
    return grouped

def _counts(trials: Iterable[Any]) -> dict[str, int]:
    return {status: sum(1 for trial in trials if trial.status == status) for status in STATUS_ORDER}

# ---------------------------------------------------------------------------
# Launcher, plan, verification, and trial state


def load_launcher(root: Path) -> dict[str, Any]:
    path = root / "launcher.json"
    data, error = _load(path) if path.is_file() else (None, "launcher.json not found")
    discovery = _map(data.get("discovery")) if data else {}
    selected = discovery.get("selected") if isinstance(discovery.get("selected"), list) else []
    excluded = discovery.get("excluded") if isinstance(discovery.get("excluded"), dict) else {}
    return {
        "path": path, "data": data, "error": error,
        "mode": data.get("mode") if data and data.get("mode") in {"full", "mini"} else None,
        "selected": [str(item) for item in selected if isinstance(item, str)],
        "excluded": {str(key): value for key, value in excluded.items()},
        "config_sha256": _text(data.get("config_sha256")) if data else None,
    }

def load_plan(root: Path) -> tuple[dict[str, Any] | None, str | None]:
    path = root / "plan.json"
    if not path.is_file():
        return None, "plan.json not found"
    plan, error = _load(path)
    if plan is None:
        return None, error
    return (plan, None) if isinstance(plan.get("trials"), list) else (None, "plan.json has no trials list")

def load_verification(root: Path) -> dict[str, Any]:
    path = root / "verification.json"
    if not path.is_file():
        return {"path": path, "data": None, "error": "verification.json not found"}
    data, error = _load(path)
    if data is None:
        return {"path": path, "data": None, "error": error}
    if not isinstance(data.get("ok"), bool):
        return {"path": path, "data": None, "error": "verification receipt has no boolean ok flag"}
    return {"path": path, "data": data, "error": None}

@dataclass
class Trial:
    run_id: str
    model: str
    width: Any
    kind: str
    status: str
    expected: int | None = None
    submitted: int | None = None
    valid: int | None = None
    gross: float | None = None
    recall: float | None = None
    f1: float | None = None
    accuracy: float | None = None
    coverage: float | None = None
    adapter: str | None = None
    passes: int | None = None
    device_adjusted: float | None = None
    loaded_adjusted: float | None = None
    window_s: float | None = None
    throughput: float | None = None
    latency_p50: float | None = None
    latency_p95: float | None = None
    group_latency_p50: float | None = None
    group_latency_p95: float | None = None
    latency_scopes: Mapping[str, Any] | None = None
    gate: Mapping[str, Any] | None = None
    active_attempt: str | None = None
    issue: str | None = None
    state_status: str = "missing"
    in_scope: bool = True
    verification: str = NOT_APPLICABLE
    detail: str | None = None
    @property
    def complete(self) -> bool:
        return self.status == "complete"
    @property
    def missing_samples(self) -> bool:
        return self.complete and self.expected is not None and self.submitted is not None and self.submitted < self.expected
    @property
    def bad_samples(self) -> bool:
        return self.complete and self.submitted is not None and self.valid is not None and self.valid < self.submitted
    @property
    def low_sampling(self) -> bool | None:
        """Sampling flag from the result gate; None when the artifact has no gate."""
        gate = self.gate
        if gate is None:
            return None
        flag = gate.get("low_sampling")
        if isinstance(flag, bool):
            return flag
        meets = gate.get("meets_minimum_window")
        if isinstance(meets, bool):
            return not meets
        return True  # A present but unusable gate never qualifies for energy.
    @property
    def energy_eligible(self) -> bool:
        return (self.complete and self.kind == "inference" and self.gross is not None
                and self.expected is not None and self.submitted == self.expected)

def _plan_trials(plan: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if plan is None:
        return []
    return [dict(trial) for trial in plan.get("trials", []) if isinstance(trial, Mapping) and isinstance(trial.get("run_id"), str) and trial["run_id"]]

def _join(*values: str | None) -> str | None:
    return "; ".join(value for value in values if value) or None

def build_trials(root: Path, plan: Mapping[str, Any] | None) -> list[Trial]:
    trials: list[Trial] = []
    for planned in _plan_trials(plan):
        run_id = planned["run_id"]
        base = dict(run_id=run_id, model=_text(planned.get("model")) or "<model:missing>",
                    width=planned.get("batch_size"), kind=_text(planned.get("kind")) or "<kind:missing>",
                    expected=_count(planned.get("expected_count")),
                    adapter=_text(planned.get("adapter")), passes=_count(planned.get("passes")))
        state_path = root / run_id / "state.json"
        if not state_path.is_file():
            trials.append(Trial(**base, status="pending"))
            continue
        state, error = _load(state_path)
        if state is None:
            trials.append(Trial(**base, status="invalid", state_status="unreadable", issue=error))
            continue
        state_status = _text(state.get("status")) or "invalid"
        issue = _text(state.get("reason"))
        attempt = _text(state.get("active_attempt"))
        if state_status != "complete":
            status = state_status if state_status in STATUS_ORDER else "invalid"
            trials.append(Trial(**base, status=status, state_status=state_status,
                                issue=issue if status != "pending" else None, active_attempt=attempt))
            continue
        result = None
        if not attempt or Path(attempt).name != attempt or attempt in {".", ".."}:
            issue = _join(issue, "invalid active_attempt")
        else:
            result_path = root / run_id / attempt / "result.json"
            if not result_path.is_file():
                issue = _join(issue, "active result.json is missing")
            else:
                result, error = _load(result_path)
                issue = _join(issue, f"unreadable result.json: {error}" if result is None else None)
        if result is None or result.get("valid") is not True:
            issue = _join(issue, "result valid flag is not true" if result is not None else None)
            trials.append(Trial(**base, status="invalid", state_status=state_status,
                                issue=issue, active_attempt=attempt))
            continue
        counts, quality, derived = _map(result.get("counts")), _map(result.get("quality")), _map(result.get("derived"))
        performance = _map(result.get("performance"))
        scopes = performance.get("latency_scopes")
        gate = result.get("gate")
        trials.append(Trial(**base, status="complete", state_status=state_status,
                            submitted=_count(counts.get("submitted")), valid=_count(counts.get("valid")),
                            gross=_num(derived.get("gross_j_per_tweet")), recall=_num(quality.get("macro_recall")),
                            f1=_num(quality.get("macro_f1")), accuracy=_num(quality.get("accuracy")),
                            coverage=_num(quality.get("coverage")),
                            device_adjusted=_num(derived.get("device_idle_adjusted_j_per_tweet")),
                            loaded_adjusted=_num(derived.get("loaded_idle_adjusted_j_per_tweet")),
                            window_s=_num(performance.get("window_s")),
                            throughput=_num(performance.get("throughput_tweets_s")),
                            latency_p50=_num(performance.get("latency_p50_s")),
                            latency_p95=_num(performance.get("latency_p95_s")),
                            group_latency_p50=_num(performance.get("group_latency_p50_s")),
                            group_latency_p95=_num(performance.get("group_latency_p95_s")),
                            latency_scopes=dict(scopes) if isinstance(scopes, Mapping) else None,
                            gate=dict(gate) if isinstance(gate, Mapping) else None,
                            active_attempt=attempt, issue=issue))
    return trials

def resolve_selection(launcher: Mapping[str, Any], trials: Sequence[Trial]) -> tuple[str, set[str], str]:
    selected = set(launcher["selected"])
    models = {trial.model for trial in trials}
    if selected & models:
        return "models", selected & models, "Status counts cover launcher-selected model ids."
    if selected:
        return "all", set(), "Selected entries match no plan model id; whole plan shown."
    return "all", set(), "No selected entries recorded; whole plan shown."

def apply_scope(trials: Sequence[Trial], kind: str, values: set[str]) -> None:
    for trial in trials:
        trial.in_scope = kind == "all" or (kind == "models" and trial.model in values)

def _receipt_runs(receipt: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    runs = _map(receipt.get("data")).get("runs")
    return {item["run_id"]: item for item in runs if isinstance(item, Mapping) and isinstance(item.get("run_id"), str)} if isinstance(runs, list) else {}

def _inconsistency_ids(receipt: Mapping[str, Any]) -> set[str]:
    items = _map(receipt.get("data")).get("inconsistencies")
    return {str(item).split(":", 1)[0].strip() for item in items if str(item).strip()} if isinstance(items, list) else set()

def verification_state(receipt: Mapping[str, Any], trials: Sequence[Trial]) -> tuple[str, str]:
    data = receipt.get("data")
    if data is None:
        state = "missing" if receipt.get("error") == "verification.json not found" else "unreadable"
        return state, receipt.get("error") or "verification receipt is unreadable"
    if data.get("ok") is not True:
        items = data.get("inconsistencies")
        return "failed", f"verification reported {len(items) if isinstance(items, list) else 0} inconsistency item(s)"
    current = {trial.run_id: trial.active_attempt for trial in trials if trial.complete}
    runs = _receipt_runs(receipt)
    receipt_complete = {run_id: _text(entry.get("active_attempt")) for run_id, entry in runs.items() if _text(entry.get("status")) == "complete"}
    missing, extra = set(current) - set(receipt_complete), set(receipt_complete) - set(current)
    mismatched = [run_id for run_id in set(current) & set(receipt_complete) if current[run_id] != receipt_complete[run_id]]
    if missing or extra or mismatched:
        detail = []
        if missing:
            detail.append(f"{len(missing)} complete run(s) absent from receipt")
        if extra:
            detail.append(f"{len(extra)} receipt-complete run(s) not currently complete")
        if mismatched:
            detail.append(f"{len(mismatched)} active-attempt mismatch(es)")
        return "partial", "verification is partial or stale: " + ", ".join(detail)
    return "ok", f"{len(current)} complete run(s) and active attempts match the receipt"

def apply_verification(trials: Sequence[Trial], receipt: Mapping[str, Any]) -> None:
    data, runs, inconsistent = receipt.get("data"), _receipt_runs(receipt), _inconsistency_ids(receipt)
    for trial in trials:
        if not trial.complete:
            continue
        if data is None:
            trial.verification, trial.detail = UNVERIFIED, "no usable verification receipt"
        elif trial.run_id in inconsistent:
            trial.verification, trial.detail = FAILED, "run appears in verification inconsistencies"
        elif data.get("ok") is not True:
            trial.verification, trial.detail = UNVERIFIED, "overall verification receipt is not ok"
        else:
            entry = runs.get(trial.run_id)
            if entry is None:
                trial.verification, trial.detail = UNVERIFIED, "run is absent from the receipt"
            elif _text(entry.get("status")) != "complete":
                trial.verification, trial.detail = UNVERIFIED, "receipt run is not complete"
            elif _text(entry.get("active_attempt")) != trial.active_attempt:
                trial.verification, trial.detail = UNVERIFIED, "active attempt differs from the receipt"
            else:
                trial.verification = VERIFIED

# ---------------------------------------------------------------------------
# Condition aggregation (verified inference replicas only)

QUALITY_SOURCES = (
    ("accuracy", "accuracy"),
    ("coverage", "coverage"),
    ("macro_recall", "recall"),
    ("macro_f1", "f1"),
)
ENERGY_SOURCES = (
    ("gross_j_per_tweet", "gross"),
    ("device_idle_adjusted_j_per_tweet", "device_adjusted"),
    ("loaded_idle_adjusted_j_per_tweet", "loaded_adjusted"),
)


@dataclass(frozen=True)
class Metric:
    """Observed replica statistics for one metric; empty when nothing was observed."""
    median: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    n: int = 0


def _metric(values: Iterable[float | None]) -> Metric:
    numbers = [value for value in values if value is not None]
    return Metric(statistics.median(numbers), min(numbers), max(numbers), len(numbers)) if numbers else Metric()


def _milliseconds(value: float | None) -> float | None:
    return None if value is None else value * 1000.0


@dataclass
class Condition:
    """One model x width inference condition; never pooled with another width."""
    model: str
    width: Any
    system: str
    engine: str
    adapter: str | None
    passes: str
    requested_reps: int | None
    planned: int
    complete: int
    verified: int
    energy_reps: int
    low_sampling_excluded: int
    count_shortfall_excluded: int
    latency_scope: str | None
    quality: dict[str, Metric]
    energy: dict[str, Metric]
    performance: dict[str, Metric]
    note: str | None = None


def receipt_low_sampling(receipt: Mapping[str, Any]) -> set[str]:
    """Run ids that the verification receipt flagged for insufficient sampling."""
    items = _map(receipt.get("data")).get("low_sampling")
    identifiers = set()
    for item in items if isinstance(items, list) else ():
        run_id = _text(item.get("run_id")) if isinstance(item, Mapping) else _text(item)
        if run_id:
            identifiers.add(run_id)
    return identifiers


def _display_name(model: str, plan: Mapping[str, Any] | None) -> str:
    """Pinned display name where known; the model id stays the identity in CSV."""
    if model in PINNED_DISPLAY_NAMES:
        return PINNED_DISPLAY_NAMES[model]
    models = _map(_map(_map(plan).get("config")).get("models")) if plan else {}
    repo = _text(_map(models.get(model)).get("repo"))
    if repo:
        name = repo.rstrip("/").rsplit("/", 1)[-1]
        if name and not any(character.isupper() for character in name):
            name = "-".join(part.capitalize() for part in name.split("-"))
        if name:
            return name
    return model


def _engine(model: str, planned: Sequence[Trial], plan: Mapping[str, Any] | None) -> tuple[str, str | None]:
    models = _map(_map(_map(plan).get("config")).get("models")) if plan else {}
    spec = _map(models.get(model))
    adapter = _text(spec.get("adapter")) or next((trial.adapter for trial in planned if trial.adapter), None)
    request_mode = _text(_map(spec.get("params")).get("request_mode"))
    if adapter == "qev" and request_mode == "packed":
        return "qev-packed", adapter
    return (adapter or "—"), adapter


def _width_number(value: Any) -> Any:
    number = _num(value)
    if number is None:
        return "" if value is None else str(value)
    return int(number) if number.is_integer() else number


def _width_order(value: Any) -> tuple[int, Any]:
    number = _num(value)
    return (0, number) if number is not None else (1, str(value))


def _width_cell(value: Any) -> str:
    if value is None:
        return "—"
    return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)


def _passes(planned: Sequence[Trial]) -> str:
    values = sorted({trial.passes for trial in planned if trial.passes is not None})
    return "/".join(str(value) for value in values) if values else ""


def _trial_scope(trial: Trial) -> str | None:
    """Selected latency scope for one replica; None when the artifact is ambiguous."""
    scopes = set(trial.latency_scopes or {})
    if "packed_request_completion" in scopes:
        return "packed_request_completion"
    if "closed-loop group completion" in scopes:
        return "closed-loop group completion"
    return "adapter request" if scopes == {"adapter request"} else None


def _sampling_excluded(trial: Trial, receipt_ids: set[str]) -> bool:
    return trial.run_id in receipt_ids or trial.low_sampling is True


def _selected_latency(verified: Sequence[Trial]) -> tuple[str | None, Metric, Metric, str | None]:
    """One scope per condition; group quantiles for packed/classical, request quantiles otherwise."""
    if not verified:
        return None, Metric(), Metric(), None
    selected, unscoped = [], 0
    for trial in verified:
        scope = _trial_scope(trial)
        if scope is None:
            unscoped += 1
        else:
            selected.append((scope, trial))
    scopes = sorted({scope for scope, _ in selected})
    if len(scopes) > 1:
        return None, Metric(), Metric(), "latency scopes differ across verified replicas (" + ", ".join(scopes) + "); not pooled"
    if not scopes:
        return None, Metric(), Metric(), "no verified replica exposes a usable latency scope"
    scope = scopes[0]
    if scope == "adapter request":
        p50 = [_milliseconds(trial.latency_p50) for _, trial in selected]
        p95 = [_milliseconds(trial.latency_p95) for _, trial in selected]
    else:
        # Group/envelope quantiles are one observation per submission envelope. The
        # per-tweet latency field repeats that completion and is never pooled as if it
        # were an independent request latency.
        p50 = [_milliseconds(trial.group_latency_p50) for _, trial in selected]
        p95 = [_milliseconds(trial.group_latency_p95) for _, trial in selected]
    note = f"latency scope missing for {unscoped} verified replica(s)" if unscoped else None
    return scope, _metric(p50), _metric(p95), note


def build_conditions(trials: Sequence[Trial], plan: Mapping[str, Any] | None,
                     receipt: Mapping[str, Any]) -> list[Condition]:
    """Aggregate planned inference conditions from verified replicas only."""
    low_ids = receipt_low_sampling(receipt)
    requested = _count(_map(_map(plan).get("config")).get("repetitions")) if plan else None
    planned_groups = _groups((trial for trial in trials if trial.in_scope and trial.kind == "inference"),
                             lambda trial: (trial.model, trial.width))
    conditions: list[Condition] = []
    for (model, width), planned in sorted(
            planned_groups.items(),
            key=lambda item: (_display_name(item[0][0], plan).lower(), _width_order(item[0][1]))):
        complete = [trial for trial in planned if trial.complete]
        verified = [trial for trial in complete if trial.verification == VERIFIED]
        energy = [trial for trial in verified
                  if trial.energy_eligible and not _sampling_excluded(trial, low_ids)]
        low_excluded = sum(1 for trial in verified
                           if trial.energy_eligible and _sampling_excluded(trial, low_ids))
        count_excluded = sum(1 for trial in verified if trial.gross is not None and not trial.energy_eligible)
        no_energy = sum(1 for trial in verified if trial.gross is None)
        scope, p50, p95, latency_note = _selected_latency(verified)
        engine, adapter = _engine(model, planned, plan)
        notes = [latency_note]
        if low_excluded:
            notes.append(f"{low_excluded} verified replica(s) excluded from energy for low sampling")
        if count_excluded:
            notes.append(f"{count_excluded} verified replica(s) excluded from energy for an incomplete submitted count")
        if no_energy:
            notes.append(f"{no_energy} verified replica(s) have no gross energy value")
        conditions.append(Condition(
            model=model, width=width, system=_display_name(model, plan), engine=engine, adapter=adapter,
            passes=_passes(planned), requested_reps=requested, planned=len(planned), complete=len(complete),
            verified=len(verified), energy_reps=len(energy), low_sampling_excluded=low_excluded,
            count_shortfall_excluded=count_excluded, latency_scope=scope,
            quality={name: _metric(getattr(trial, source) for trial in verified) for name, source in QUALITY_SOURCES},
            energy={name: _metric(getattr(trial, source) for trial in energy) for name, source in ENERGY_SOURCES},
            performance={
                "throughput_tweets_s": _metric(trial.throughput for trial in verified),
                "latency_p50_ms": p50,
                "latency_p95_ms": p95,
                "group_latency_p50_ms": _metric(_milliseconds(trial.group_latency_p50) for trial in verified),
                "group_latency_p95_ms": _metric(_milliseconds(trial.group_latency_p95) for trial in verified),
            },
            note="; ".join(item for item in notes if item) or None,
        ))
    return conditions


def _range(metric: Metric, digits: int = 4) -> str:
    if metric.median is None:
        return "—"
    return f"{_sig(metric.median, digits)} [{_sig(metric.minimum, digits)}–{_sig(metric.maximum, digits)}]"


def _metric_cells(metric: Metric) -> list[Any]:
    return [metric.median, metric.minimum, metric.maximum, metric.n]


def condition_csv(conditions: Sequence[Condition]) -> tuple[list[str], list[list[Any]]]:
    """Tabular form of the same aggregation as the Markdown results table."""
    headers = [
        "model", "engine", "adapter", "width", "passes", "requested_reps",
        "planned_reps", "complete_reps", "verified_reps", "energy_eligible_reps",
        "low_sampling_excluded", "count_shortfall_excluded",
    ]
    for name in ("accuracy", "coverage", "macro_recall", "macro_f1"):
        headers += [f"{name}_median", f"{name}_min", f"{name}_max", f"{name}_n"]
    for name in ("gross_j_per_tweet", "device_idle_adjusted_j_per_tweet",
                 "loaded_idle_adjusted_j_per_tweet"):
        headers += [f"{name}_median", f"{name}_min", f"{name}_max", f"{name}_n"]
    headers += ["throughput_tweets_s_median", "throughput_tweets_s_min",
                "throughput_tweets_s_max", "throughput_tweets_s_n"]
    headers += ["latency_scope", "latency_p50_ms_median", "latency_p50_ms_min",
                "latency_p50_ms_max", "latency_p50_ms_n", "latency_p95_ms_median",
                "latency_p95_ms_min", "latency_p95_ms_max", "latency_p95_ms_n"]
    headers += ["group_latency_scope", "group_latency_p50_ms_median", "group_latency_p50_ms_min",
                "group_latency_p50_ms_max", "group_latency_p50_ms_n", "group_latency_p95_ms_median",
                "group_latency_p95_ms_min", "group_latency_p95_ms_max", "group_latency_p95_ms_n",
                "notes"]
    rows: list[list[Any]] = []
    for condition in conditions:
        row: list[Any] = [
            condition.model, condition.engine, condition.adapter or "", _width_number(condition.width),
            condition.passes, condition.requested_reps if condition.requested_reps is not None else "",
            condition.planned, condition.complete, condition.verified, condition.energy_reps,
            condition.low_sampling_excluded, condition.count_shortfall_excluded,
        ]
        for name, _ in QUALITY_SOURCES:
            row += _metric_cells(condition.quality[name])
        for name, _ in ENERGY_SOURCES:
            row += _metric_cells(condition.energy[name])
        row += _metric_cells(condition.performance["throughput_tweets_s"])
        row += [condition.latency_scope or ""]
        row += _metric_cells(condition.performance["latency_p50_ms"])
        row += _metric_cells(condition.performance["latency_p95_ms"])
        row += ["submission envelope completion"]
        row += _metric_cells(condition.performance["group_latency_p50_ms"])
        row += _metric_cells(condition.performance["group_latency_p95_ms"])
        row += [condition.note or ""]
        rows.append(row)
    return headers, rows


def write_condition_csv(path: Path, conditions: Sequence[Condition]) -> Path:
    headers, rows = condition_csv(conditions)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(headers)
    for row in rows:
        writer.writerow(["" if value is None else value for value in row])
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(buffer.getvalue(), encoding="utf-8")
    temporary.replace(path)
    return path


def _throughput(value: float | None) -> str:
    if value is None:
        return "—"
    if value == 0:
        return "0"
    return _sig(value, 3) if abs(value) < 0.05 else f"{value:,.1f}"


def _results_table(conditions: Sequence[Condition]) -> list[str]:
    rows = []
    for condition in conditions:
        rows.append((
            condition.system, _width_cell(condition.width),
            _sig(condition.quality["macro_recall"].median, 3),
            _sig(condition.quality["macro_f1"].median, 3),
            _range(condition.energy["gross_j_per_tweet"], 4),
            _throughput(condition.performance["throughput_tweets_s"].median),
            _sig(condition.performance["latency_p95_ms"].median, 4),
            LATENCY_SCOPE_LABELS.get(condition.latency_scope, "—"),
            condition.planned, condition.verified, condition.energy_reps,
        ))
    return _table(("System", "Width", "Macro recall", "Macro-F1", "Gross J/tweet [min–max]",
                   "Tweets/s", "p95 ms", "p95 scope", "Planned", "Verified", "Energy-eligible"), rows)


def _results_notes(conditions: Sequence[Condition], results_csv: Path | None) -> list[str]:
    lines = []
    if results_csv is not None:
        lines += [f"Machine-readable table: `{results_csv}`.", ""]
    lines.append("- A condition is one model and one submission width. Planned, Verified, and Energy-eligible "
                 "count planned, receipt-verified, and energy-eligible inference replicas (trials, not tweets); "
                 "a width is never pooled with another width.")
    lines.append("- Quality uses the first measured pass only (`basis_pass = 0`); medians are across replicas, not tweets.")
    observed = [condition.quality["coverage"] for condition in conditions if condition.quality["coverage"].n]
    n_coverage = sum(metric.n for metric in observed)
    n_verified = sum(condition.verified for condition in conditions)
    if n_coverage:
        lo = min(metric.minimum for metric in observed)
        hi = max(metric.maximum for metric in observed)
        coverage = f"{lo:.1%}" if lo == hi else f"{lo:.1%}–{hi:.1%}"
        lines.append(f"- First-pass valid-output coverage: {coverage} "
                     f"({n_coverage}/{n_verified} verified inference trials with coverage recorded).")
    lines.append("- Gross J/tweet = CodeCarbon CPU + GPU + RAM energy / all submitted tweet decisions. "
                 "Device- and loaded-idle-adjusted J/tweet are separate signed estimates in the CSV.")
    lines.append("- Tweets/s = completed tweets / measured work-window seconds. p95 (and p50 in the CSV) select "
                 "adapter-request quantiles for vLLM and QEV width 1, and one group/envelope observation for "
                 "classical static groups and packed QEV widths > 1; packed envelope time is never repeated per "
                 "tweet, and a column never mixes scopes.")
    lines.append("- Energy-eligible needs a full submitted count and a passing sampling gate. Low-sampling and "
                 "sample-shortfall replicas are excluded only from energy; accuracy uses the quality subset, p50 "
                 "the performance subset, and adjusted energy the energy subset, so `*_n` can differ by column.")
    unverified = sum(condition.complete - condition.verified for condition in conditions)
    if unverified:
        lines.append(f"- Complete in-scope inference replicas not verified and excluded from every median: {unverified}.")
    for condition in conditions:
        if condition.note:
            lines.append("- " + _cell(f"{condition.system} width {_width_cell(condition.width)}: {condition.note}") + ".")
    return lines


# ---------------------------------------------------------------------------
# Markdown summary


def render_summary(root: Path, launcher: Mapping[str, Any], plan: Mapping[str, Any] | None,
                   plan_error: str | None, receipt: Mapping[str, Any], trials: Sequence[Trial],
                   scope_note: str, csv_paths: Sequence[Path], summary_path: Path,
                   conditions: Sequence[Condition] = (), results_csv: Path | None = None) -> str:
    in_scope = [trial for trial in trials if trial.in_scope]
    mode = launcher.get("mode")
    overall, overall_detail = verification_state(receipt, trials)
    complete_count = sum(1 for trial in trials if trial.complete)
    verified_count = sum(1 for trial in trials if trial.verification == VERIFIED)
    failed_count = sum(1 for trial in trials if trial.verification == FAILED)
    unverified_count = sum(1 for trial in trials if trial.verification == UNVERIFIED)
    lines = ["# Sentiment energy benchmark launcher report", "", f"**Run root:** `{root}`"]
    lines.append(f"**Launcher:** unreadable — {_cell(launcher['error'])} (`{launcher['path']}`)" if launcher["error"] else f"**Launcher:** `{launcher['path']}`")
    lines.append(f"**Mode:** **{mode}**" if mode else "**Mode:** **unknown** (launcher.json missing or invalid)")
    plan_sha = _text(plan.get("config_sha256")) if plan else None
    if launcher["config_sha256"] or plan_sha:
        lines.append(f"**Config SHA-256:** launcher `{launcher['config_sha256'] or '—'}`, plan `{plan_sha or '—'}`")
        if launcher["config_sha256"] and plan_sha and launcher["config_sha256"] != plan_sha:
            lines.append("> **Warning:** launcher and plan config hashes differ.")
    lines.append(f"**Plan:** **UNMEASURED** — {_cell(plan_error)}" if plan is None else f"**Plan:** {len(_plan_trials(plan))} planned trial(s); phase `{plan.get('phase', 'unknown')}`")
    lines.append("")
    if plan is None:
        lines += ["> **UNMEASURED:** `plan.json` is missing or unreadable. Artifacts on disk", "> are not evidence of a planned benchmark protocol; no energy or quality", "> result is presented.", ""]
    if mode == "mini":
        lines += ["> **SMOKE ONLY — NOT A COMPARATIVE RESULT.** This `mini` run checks plumbing,", "> coverage, and failures only. Do not compare quality, latency, throughput, or", "> energy between systems, and do not publish these numbers as a benchmark result.", "> CSV tables are labelled smoke-only audit output.", ""]

    lines += ["## Discovery and scope", ""]
    lines.append("- Selected: " + ", ".join(f"`{item}`" for item in launcher["selected"]) if launcher["selected"] else "- Selected: none recorded.")
    if launcher["excluded"]:
        lines.append("- Excluded by the launcher:")
        lines += [f"  - `{identifier}`: {_cell(reason)}" for identifier, reason in sorted(launcher["excluded"].items())]
    else:
        lines.append("- Excluded: none recorded.")
    lines += [f"- {scope_note}", f"- In scope: {len(in_scope)} trial(s).", ""]

    if mode != "mini":
        lines += ["## Results (inference conditions)", ""]
        if plan is None:
            lines.append("Condition medians omitted: the run is unmeasured (no plan).")
        elif mode != "full":
            lines.append("Condition medians omitted: mode is not confirmed `full`.")
        elif not conditions:
            lines.append("Condition medians suppressed: the plan contains no in-scope inference condition.")
        else:
            lines += _results_table(conditions) + [""]
            lines += _results_notes(conditions, results_csv)
            if not any(condition.verified for condition in conditions):
                lines.append("No condition has a receipt-verified complete inference replica; every metric is missing (`—`), not zero.")
        lines.append("")

    lines += ["## Status", ""]
    counts = _counts(in_scope)
    lines += _table(("Status", "Count"), [(status, counts[status]) for status in STATUS_ORDER])
    lines += ["", f"- Pending without `state.json`: {sum(1 for trial in in_scope if trial.state_status == 'missing')}.",
              f"- Missing/unreadable state files: {sum(1 for trial in in_scope if trial.state_status in {'missing', 'unreadable'})}.",
              f"- Missing submission sample runs: {sum(1 for trial in in_scope if trial.missing_samples)}.",
              f"- Invalid sample runs: {sum(1 for trial in in_scope if trial.bad_samples)}.", "",
              "### Status by model, width, and workload kind", ""]
    rows = []
    for (model, width, kind), group in sorted(_groups(in_scope, lambda t: (t.model, str(t.width), t.kind)).items()):
        group_counts = _counts(group)
        gaps = sum(1 for trial in group if trial.missing_samples or trial.bad_samples) if kind == "inference" else "—"
        rows.append((model, width, kind, *[group_counts[status] for status in STATUS_ORDER], len(group), gaps))
    lines += _table(("Model", "Width", "Kind", *STATUS_ORDER, "Total", "Sample gaps"), rows) + [""]

    lines += ["### Issues", ""]
    issues = [trial for trial in in_scope if trial.issue]
    if issues:
        lines += [f"- `{trial.run_id}` {trial.model} width {trial.width} {trial.kind} ({trial.status}): {_cell(trial.issue)}" for trial in issues[:30]]
        if len(issues) > 30:
            lines.append(f"- … and {len(issues) - 30} more issue(s).")
    else:
        lines.append("No state reasons or artifact errors recorded.")
    lines.append("")

    lines += ["## Verification", ""]
    stamp = _stamp(receipt["path"])
    lines += [f"- Receipt: `{receipt['path']}` (stamp {stamp or 'unavailable'})", f"- Overall: **{overall}** — {_cell(overall_detail)}.",
              f"- Complete runs: {complete_count}; verified: {verified_count}; failed: {failed_count}; unverified: {unverified_count}."]
    inconsistencies = _map(receipt.get("data")).get("inconsistencies")
    if isinstance(inconsistencies, list) and inconsistencies:
        lines.append(f"- Inconsistencies: {len(inconsistencies)}")
        lines += [f"  - {_cell(item)}" for item in inconsistencies[:20]]
        if len(inconsistencies) > 20:
            lines.append(f"  - … and {len(inconsistencies) - 20} more.")
    if not (complete_count == 0 or verified_count == complete_count):
        lines.append("> Only receipt-listed complete run ids with matching active attempts enter full medians.")
    attention = [trial for trial in trials if trial.complete and trial.verification != VERIFIED]
    if attention:
        lines += ["", "Complete runs not verified:"]
        lines += [f"- `{trial.run_id}`: {trial.verification} — {_cell(trial.detail)}" for trial in attention[:10]]
        if len(attention) > 10:
            lines.append(f"- … and {len(attention) - 10} more.")
    lines.append("")

    if mode == "mini":
        lines += ["## Smoke-only audit (mini)", ""]
        complete_inference = [trial for trial in in_scope if trial.kind == "inference" and trial.complete]
        lines += ["**SMOKE ONLY:** mini runs do not report comparative energy medians. `energy_components.csv` and `energy_intensity.csv` are raw audit tables only.", ""]
        rows = []
        for (model, width), group in sorted(_groups(complete_inference, lambda t: (t.model, str(t.width))).items()):
            metrics = sum(1 for trial in group if None not in (trial.recall, trial.f1, trial.accuracy, trial.coverage))
            rows.append((model, width, len(group), _ratio(metrics, len(group)), sum(1 for trial in group if trial.verification == VERIFIED)))
        lines += ["**SMOKE ONLY:** comparative quality medians are intentionally omitted.", ""]
        lines += _table(("Model", "Width", "Complete runs", "Quality fields", "Verified"), rows)
        lines += ["", "Only metric availability/coverage is shown; raw values remain in the smoke-only CSV audit tables.",
                  "No `inference_medians.csv` is written for a mini run."]
        lines.append("")

    lines += ["## Outputs", ""]
    lines.append(f"- Summary: `{summary_path}`")
    audit_paths = [path for path in csv_paths if path.name != "inference_medians.csv"]
    lines.append("- Standard `analysis.py` audit tables:")
    lines += [f"  - `{path}`" for path in audit_paths]
    if results_csv is not None:
        lines.append(f"- Condition medians (verified inference): `{results_csv}`")
    lines += ["", "Standard CSV tables retain incomplete/unverified rows; they are audit tables, not",
              "evidence that every row is a valid verified result."]
    if results_csv is not None:
        lines.append("`inference_medians.csv` uses only receipt-verified complete inference replicas;"
                     " an empty field means missing, never zero.")
    if mode == "mini":
        lines.append("For this `mini` root all CSV tables are smoke-only.")
    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Orchestration and CLI


def generate_report(root: str | Path) -> dict[str, Any]:
    root_path = Path(root).expanduser()
    if not root_path.is_dir():
        raise NotADirectoryError(f"Run root is not a directory: {root_path}")
    launcher, receipt = load_launcher(root_path), load_verification(root_path)
    plan, plan_error = load_plan(root_path)
    trials = build_trials(root_path, plan)
    kind, values, scope_note = resolve_selection(launcher, trials)
    apply_scope(trials, kind, values)
    apply_verification(trials, receipt)
    tables = load_artifacts(root_path)
    report_dir = root_path / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_paths = tuple(write_tables(tables, report_dir))
    conditions = build_conditions(trials, plan, receipt)
    results_csv = None
    if launcher.get("mode") == "full" and plan is not None and conditions:
        results_csv = write_condition_csv(report_dir / "inference_medians.csv", conditions)
        csv_paths += (results_csv,)
    summary_path = report_dir / "summary.md"
    temporary = summary_path.with_name(f".{summary_path.name}.tmp")
    temporary.write_text(render_summary(root_path, launcher, plan, plan_error, receipt, trials, scope_note,
                                        csv_paths, summary_path, conditions, results_csv), encoding="utf-8")
    temporary.replace(summary_path)
    return {"root": root_path, "launcher": launcher, "plan": plan, "plan_error": plan_error, "verification": receipt,
            "trials": trials, "conditions": conditions, "results_csv": results_csv,
            "summary_path": summary_path, "csv_paths": csv_paths}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write RUN_ROOT/report/summary.md, inference_medians.csv, and analysis CSVs.")
    parser.add_argument("run_root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = generate_report(args.run_root)
    except NotADirectoryError as exc:
        print(f"launcher_report: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"launcher_report: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    trials = report["trials"]
    in_scope = [trial for trial in trials if trial.in_scope]
    counts = _counts(in_scope)
    overall, _ = verification_state(report["verification"], trials)
    complete = sum(1 for trial in trials if trial.complete)
    verified = sum(1 for trial in trials if trial.verification == VERIFIED)
    print(f"launcher report: {report['root']}")
    print(f"mode: {report['launcher'].get('mode') or 'unknown'}")
    print(f"plan: {'UNMEASURED' if report['plan'] is None else str(len(trials)) + ' planned trial(s)'}")
    print(f"verification: {overall}; {verified}/{complete} complete run(s) verified")
    print("status: " + ", ".join(f"{status}={counts[status]}" for status in STATUS_ORDER))
    print(f"summary: {report['summary_path']}")
    print(f"tables: {report['summary_path'].parent}/*.csv ({len(report['csv_paths'])} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
