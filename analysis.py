#!/usr/bin/env python3
"""Load benchmark artifacts into run-level pandas tables.

The input root is scanned non-recursively for ``*/state.json``.  No benchmark
code is imported and no measurements are started.  Missing or malformed data
is reported in the status table; absent metric values stay null.

Usage:
    python analysis.py /path/to/artifacts
    python analysis.py /path/to/artifacts --output-dir /path/to/tables

Without ``--output-dir`` the command only prints summaries.  With it, the CLI
writes tidy CSV files suitable for further analysis.  It never aggregates over
phase, kind, model, batch size, repetition, dataset, host, or adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

SUCCESS_STATUSES = frozenset({"complete", "completed", "success", "succeeded"})
COUNT_FIELDS = (
    "submitted",
    "completed",
    "valid",
    "correct",
    "retries",
    "pass_count",
    "panel_size",
    "groups",
    "retry_requests",
    "http_request_count",
    "qev_choice_count",
    "qev_request_count",
    "final_group_width",
)
QUALITY_FIELDS = (
    "macro_recall",
    "macro_f1",
    "accuracy",
    "coverage",
    "basis_pass",
    "basis_n",
    "repeat_disagreements_from_first",
)
PERFORMANCE_FIELDS = (
    "window_s",
    "throughput_tweets_s",
    "valid_throughput_tweets_s",
    "latency_p50_s",
    "latency_p95_s",
    "group_latency_p50_s",
    "group_latency_p95_s",
)
ENERGY_FIELDS = (
    "energy_kwh",
    "cpu_energy_kwh",
    "gpu_energy_kwh",
    "ram_energy_kwh",
    "duration_s",
)
DERIVED_FIELDS = (
    "gross_j_per_tweet",
    "device_idle_adjusted_j_per_tweet",
    "loaded_idle_adjusted_j_per_tweet",
)
IDENTITY_FIELDS = (
    "run_id",
    "run_dir",
    "state_path",
    "status",
    "active_attempt",
    "result_path",
    "artifact_status",
    "result_error",
    "analysis_eligible",
    "phase",
    "kind",
    "model",
    "batch_size",
    "rep",
    "condition_id",
    "trial_label",
)
CONTEXT_FIELDS = (
    "state_trial_json",
    "result_trial_json",
    "dataset_json",
    "host_json",
    "adapter_json",
    "energy_run_id",
    "performance_latency_scopes_json",
    "outside_window_json",
    "gate_json",
)
RUN_COLUMNS = (
    "run_id",
    "run_dir",
    "state_path",
    "status",
    "is_complete",
    "active_attempt",
    "result_path",
    "artifact_status",
    "result_error",
    "analysis_eligible",
    "phase",
    "kind",
    "model",
    "batch_size",
    "rep",
)
TRIAL_COLUMNS = (
    IDENTITY_FIELDS
    + CONTEXT_FIELDS
    + tuple(f"count_{field}" for field in COUNT_FIELDS)
    + tuple(f"quality_{field}" for field in QUALITY_FIELDS)
    + PERFORMANCE_FIELDS
    + ENERGY_FIELDS
    + DERIVED_FIELDS
)
LONG_ID_FIELDS = (
    "run_id",
    "run_dir",
    "status",
    "analysis_eligible",
    "condition_id",
    "trial_label",
    "phase",
    "kind",
    "model",
    "batch_size",
    "rep",
    "dataset_json",
    "host_json",
    "adapter_json",
)
NUMERIC_COLUMNS = (
    "batch_size",
    *(f"count_{field}" for field in COUNT_FIELDS),
    *(f"quality_{field}" for field in QUALITY_FIELDS if field != "basis_pass"),
    *PERFORMANCE_FIELDS,
    *ENERGY_FIELDS,
    *DERIVED_FIELDS,
)


@dataclass(frozen=True)
class AnalysisTables:
    """Run-level and tidy tables produced from one artifact root."""

    runs: pd.DataFrame
    trials: pd.DataFrame
    class_quality: pd.DataFrame
    energy_components: pd.DataFrame
    energy_intensity: pd.DataFrame
    performance: pd.DataFrame

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {
            "runs": self.runs,
            "trials": self.trials,
            "class_quality": self.class_quality,
            "energy_components": self.energy_components,
            "energy_intensity": self.energy_intensity,
            "performance": self.performance,
        }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _value(mapping: Mapping[str, Any], key: str) -> Any:
    value = mapping.get(key)
    return None if value is None else value


def _json(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError) as exc:
        return json.dumps({"serialization_error": str(exc)})


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(value, dict):
        return None, "JSON root must be an object"
    return value, None


def _attempt_result_path(run_dir: Path, active_attempt: Any) -> tuple[Path | None, str | None]:
    if not isinstance(active_attempt, str) or not active_attempt.strip():
        return None, "active_attempt is missing or is not a nonempty string"
    candidate = Path(active_attempt)
    if candidate.is_absolute() or candidate.name != active_attempt or active_attempt in {".", ".."}:
        return None, "active_attempt must be one relative directory name"
    return run_dir / candidate / "result.json", None


def _is_success(status: Any) -> bool:
    return isinstance(status, str) and status.strip().lower() in SUCCESS_STATUSES


def _condition(trial: Mapping[str, Any]) -> tuple[str, str]:
    """Identify a full condition and one observed trial without aggregating."""
    payload = {
        key: trial.get(key)
        for key in (
            "phase",
            "kind",
            "model",
            "batch_size",
            "rep",
            "state_trial_json",
            "result_trial_json",
            "dataset_json",
            "host_json",
            "adapter_json",
        )
    }
    encoded = _json(payload) or "{}"
    condition_id = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
    model = trial.get("model") or "<model:null>"
    batch = trial.get("batch_size")
    rep = trial.get("rep")
    # The run directory keeps labels unique even if an artifact reuses run_id.
    run = Path(str(trial.get("run_dir", "<run>"))).name
    label = f"{model} | batch={batch if batch is not None else 'null'} | rep={rep if rep is not None else 'null'} | run={run}"
    return condition_id, label


def _base_trial(run_dir: Path, state_path: Path) -> dict[str, Any]:
    row = {column: None for column in TRIAL_COLUMNS}
    row.update(
        run_dir=str(run_dir),
        state_path=str(state_path),
        artifact_status="state_read_error",
        analysis_eligible=False,
    )
    return row


def _load_one(state_path: Path) -> dict[str, Any]:
    run_dir = state_path.parent
    row = _base_trial(run_dir, state_path)
    state, error = _read_json(state_path)
    if error:
        row["result_error"] = error
        return row

    assert state is not None  # For type checkers; _read_json returns a dict on success.
    state_trial = _mapping(state.get("trial"))
    row.update(
        run_id=_value(state, "run_id"),
        status=_value(state, "status"),
        active_attempt=_value(state, "active_attempt"),
        phase=_value(state_trial, "phase"),
        kind=_value(state_trial, "kind"),
        model=_value(state_trial, "model"),
        batch_size=_value(state_trial, "batch_size"),
        rep=_value(state_trial, "rep"),
        state_trial_json=_json(state.get("trial")),
    )

    result_path, error = _attempt_result_path(run_dir, state.get("active_attempt"))
    if error:
        row.update(artifact_status="invalid_active_attempt", result_error=error)
        return row
    assert result_path is not None
    row["result_path"] = str(result_path)
    if not result_path.is_file():
        row.update(
            artifact_status="missing_result",
            result_error="Active attempt result.json does not exist",
        )
        return row

    result, error = _read_json(result_path)
    if error:
        row.update(artifact_status="result_read_error", result_error=error)
        return row
    assert result is not None

    result_trial = _mapping(result.get("trial"))
    merged_trial = dict(state_trial)
    merged_trial.update(result_trial)
    counts = _mapping(result.get("counts"))
    quality = _mapping(result.get("quality"))
    performance = _mapping(result.get("performance"))
    energy = _mapping(result.get("energy"))
    derived = _mapping(result.get("derived"))

    row.update(
        phase=_value(merged_trial, "phase"),
        kind=_value(result, "kind") or _value(merged_trial, "kind"),
        model=_value(result, "model") or _value(merged_trial, "model"),
        batch_size=(
            _value(result, "batch_size")
            if _value(result, "batch_size") is not None
            else _value(merged_trial, "batch_size")
        ),
        rep=_value(merged_trial, "rep"),
        result_trial_json=_json(result.get("trial")),
        dataset_json=_json(result.get("dataset")),
        host_json=_json(result.get("host")),
        adapter_json=_json(result.get("adapter")),
        energy_run_id=_value(energy, "run_id"),
        performance_latency_scopes_json=_json(performance.get("latency_scopes")),
        outside_window_json=_json(result.get("outside_window_s")),
        gate_json=_json(result.get("gate")),
        artifact_status="ok",
        result_error=None,
        analysis_eligible=_is_success(state.get("status")),
    )
    for field in COUNT_FIELDS:
        row[f"count_{field}"] = _value(counts, field)
    for field in QUALITY_FIELDS:
        row[f"quality_{field}"] = _value(quality, field)
    for field in PERFORMANCE_FIELDS:
        row[field] = _value(performance, field)
    for field in ENERGY_FIELDS:
        row[field] = _value(energy, field)
    for field in DERIVED_FIELDS:
        row[field] = _value(derived, field)
    row["condition_id"], row["trial_label"] = _condition(row)
    return row


def _frame(rows: list[dict[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(columns))


def _numeric_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    for column in columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _long_table(
    trials: pd.DataFrame,
    value_columns: Mapping[str, str],
    metric_name: str,
) -> pd.DataFrame:
    columns = list(LONG_ID_FIELDS) + [metric_name, "value"]
    if trials.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for _, trial in trials.iterrows():
        identity = {field: trial[field] for field in LONG_ID_FIELDS}
        for source, metric in value_columns.items():
            rows.append({**identity, metric_name: metric, "value": trial[source]})
    result = _frame(rows, columns)
    result["value"] = pd.to_numeric(result["value"], errors="coerce")
    return result


def _class_quality_table(trials: pd.DataFrame, raw_per_class: Mapping[str, Any]) -> pd.DataFrame:
    columns = list(LONG_ID_FIELDS) + ["class_label", "metric", "value"]
    rows: list[dict[str, Any]] = []
    for _, trial in trials.iterrows():
        # Calibration and training records must never enter quality summaries.
        if not trial["analysis_eligible"] or trial["kind"] != "inference":
            continue
        per_class = _mapping(raw_per_class.get(trial["result_path"]))
        for class_label, metrics in per_class.items():
            if isinstance(metrics, Mapping):
                items = metrics.items()
            else:
                items = (("value", metrics),)
            for metric, value in items:
                rows.append(
                    {
                        **{field: trial[field] for field in LONG_ID_FIELDS},
                        "class_label": str(class_label),
                        "metric": str(metric),
                        "value": value,
                    }
                )
    result = _frame(rows, columns)
    if not result.empty:
        result["value"] = pd.to_numeric(result["value"], errors="coerce")
    return result


def load_artifacts(root: str | Path) -> AnalysisTables:
    """Scan ``root/*/state.json`` and return tables without modifying artifacts."""
    root_path = Path(root).expanduser()
    if not root_path.is_dir():
        raise NotADirectoryError(f"Artifact root is not a directory: {root_path}")

    state_paths = sorted(
        (path for path in root_path.glob("*/state.json") if path.is_file()),
        key=lambda path: str(path.relative_to(root_path)),
    )
    trial_rows = [_load_one(path) for path in state_paths]
    trials = _frame(trial_rows, TRIAL_COLUMNS)
    trials = _numeric_columns(trials, NUMERIC_COLUMNS)

    run_rows = [{column: row.get(column) for column in RUN_COLUMNS} for row in trial_rows]
    for row in run_rows:
        row["is_complete"] = _is_success(row["status"])
    runs = _frame(run_rows, RUN_COLUMNS)

    raw_per_class: dict[str, Any] = {}
    for state_path in state_paths:
        # Re-read only result.json files already validated by _load_one. This
        # keeps per-class metrics long without hiding their original structure.
        trial = next(row for row in trial_rows if row["state_path"] == str(state_path))
        if trial["artifact_status"] == "ok" and trial["result_path"]:
            result, error = _read_json(Path(trial["result_path"]))
            if error is None and result is not None:
                raw_per_class[trial["result_path"]] = _mapping(result.get("quality")).get("per_class")

    class_quality = _class_quality_table(trials, raw_per_class)
    energy_components = _long_table(
        trials,
        {
            "cpu_energy_kwh": "cpu",
            "gpu_energy_kwh": "gpu",
            "ram_energy_kwh": "ram",
        },
        "component",
    ).rename(columns={"value": "energy_kwh"})
    energy_intensity = _long_table(
        trials,
        {
            "gross_j_per_tweet": "gross",
            "device_idle_adjusted_j_per_tweet": "device_idle_adjusted",
            "loaded_idle_adjusted_j_per_tweet": "loaded_idle_adjusted",
        },
        "basis",
    ).rename(columns={"value": "j_per_tweet"})
    performance = _long_table(
        trials,
        {
            "window_s": "window_s",
            "throughput_tweets_s": "throughput_tweets_s",
            "latency_p50_s": "latency_p50_s",
            "latency_p95_s": "latency_p95_s",
            "valid_throughput_tweets_s": "valid_throughput_tweets_s",
            "group_latency_p50_s": "group_latency_p50_s",
            "group_latency_p95_s": "group_latency_p95_s",
        },
        "metric",
    )
    return AnalysisTables(
        runs=runs,
        trials=trials,
        class_quality=class_quality,
        energy_components=energy_components,
        energy_intensity=energy_intensity,
        performance=performance,
    )


def successful_inference(trials: pd.DataFrame) -> pd.DataFrame:
    """Return only complete, readable inference trials for quality analysis."""
    if trials.empty:
        return trials.copy()
    mask = trials["analysis_eligible"].fillna(False) & trials["kind"].eq("inference")
    return trials.loc[mask].copy()


def write_tables(tables: AnalysisTables, output_dir: str | Path) -> list[Path]:
    """Write all tables as CSV, using empty fields rather than zero for nulls."""
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, frame in tables.as_dict().items():
        path = destination / f"{name}.csv"
        frame.to_csv(path, index=False)
        written.append(path)
    return written


def _display(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(no rows)"
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        180,
        "display.max_colwidth",
        60,
    ):
        return frame.to_string(index=False)


def print_summary(tables: AnalysisTables) -> None:
    """Print status and unaggregated metric breakdowns."""
    run_display = [
        "run_id",
        "run_dir",
        "status",
        "phase",
        "kind",
        "model",
        "batch_size",
        "rep",
        "artifact_status",
        "result_error",
    ]
    metric_display = [
        "run_id",
        "phase",
        "kind",
        "model",
        "batch_size",
        "rep",
        "count_submitted",
        "count_completed",
        "count_valid",
        "count_correct",
        "quality_macro_recall",
        "quality_macro_f1",
        "quality_accuracy",
        "quality_coverage",
        "energy_kwh",
        "gross_j_per_tweet",
        "device_idle_adjusted_j_per_tweet",
        "loaded_idle_adjusted_j_per_tweet",
        "throughput_tweets_s",
        "latency_p50_s",
        "latency_p95_s",
    ]
    # A completed state with a missing/broken active result is also unusable.
    non_complete = tables.runs.loc[~tables.runs["analysis_eligible"].fillna(False)]
    inference = successful_inference(tables.trials)
    print("Discovered runs (all statuses)")
    print(_display(tables.runs[run_display]))
    print("\nNon-complete runs")
    print(_display(non_complete[run_display]))
    print("\nSuccessful inference trials (one row per trial; no pooled conditions)")
    print(_display(inference[metric_display]))
    print("\nEnergy components (kWh; missing components remain null)")
    energy_display = [
        "run_id",
        "phase",
        "kind",
        "model",
        "batch_size",
        "rep",
        "component",
        "energy_kwh",
    ]
    print(_display(tables.energy_components[energy_display]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize sentiment benchmark artifacts without pooling conditions."
    )
    parser.add_argument("root", type=Path, help="Directory containing run-dir/state.json files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory for six tidy CSV tables; default only prints summaries",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        tables = load_artifacts(args.root)
    except NotADirectoryError as exc:
        print(f"analysis: {exc}", file=sys.stderr)
        return 2
    print_summary(tables)
    if args.output_dir is not None:
        written = write_tables(tables, args.output_dir)
        print("\nWrote CSV tables:")
        for path in written:
            print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
