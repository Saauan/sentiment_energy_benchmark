#!/usr/bin/env python3
"""Write RUN_ROOT/report/summary.md and analysis CSVs for a launcher-owned run.

Read-only with respect to benchmark artifacts; only RUN_ROOT/report/ is written.
No model is loaded and no measurement is started.

Usage:
    python launcher_report.py RUN_ROOT
"""

from __future__ import annotations

import argparse
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

def _median(values: Iterable[float | None]) -> float | None:
    numbers = [value for value in values if value is not None]
    return statistics.median(numbers) if numbers else None

def _fmt(value: float | int | None) -> str:
    return "—" if value is None else (f"{float(value):.4f}" if abs(float(value)) < 1000 else f"{float(value):,.1f}")

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
                    expected=_count(planned.get("expected_count")))
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
        trials.append(Trial(**base, status="complete", state_status=state_status,
                            submitted=_count(counts.get("submitted")), valid=_count(counts.get("valid")),
                            gross=_num(derived.get("gross_j_per_tweet")), recall=_num(quality.get("macro_recall")),
                            f1=_num(quality.get("macro_f1")), accuracy=_num(quality.get("accuracy")),
                            coverage=_num(quality.get("coverage")), active_attempt=attempt, issue=issue))
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
# Markdown summary


def render_summary(root: Path, launcher: Mapping[str, Any], plan: Mapping[str, Any] | None,
                   plan_error: str | None, receipt: Mapping[str, Any], trials: Sequence[Trial],
                   scope_note: str, csv_paths: Sequence[Path], summary_path: Path) -> str:
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

    lines += ["## Energy", ""]
    if plan is None:
        lines.append("Energy medians omitted: the run is unmeasured (no plan).")
    elif mode == "mini":
        lines.append("**SMOKE ONLY:** mini runs do not report comparative energy medians. `energy_components.csv` and `energy_intensity.csv` are raw audit tables only.")
    elif mode != "full":
        lines.append("Energy medians omitted: mode is not confirmed `full`.")
    else:
        eligible = [trial for trial in in_scope if trial.kind == "inference" and trial.energy_eligible]
        verified = [trial for trial in eligible if trial.verification == VERIFIED]
        if not verified:
            lines.append("Energy medians suppressed: no eligible inference run is explicitly verified by the receipt.")
            if eligible:
                lines.append(f"{len(eligible)} eligible inference run(s) excluded because they are unverified.")
        else:
            for (model, width), group in sorted(_groups(verified, lambda t: (t.model, str(t.width))).items()):
                values = [trial.gross for trial in group if trial.gross is not None]
                lines.append(f"- {model} width {width}: n={len(group)} verified replica(s); median {_fmt(_median(values))} J/tweet (min {_fmt(min(values))}, max {_fmt(max(values))}).")
            lines += ["", "Median is over verified eligible full inference runs (a measured run is the energy", "replicate). Calibration, training, invalid, incomplete, sample-shortfall, and", "unverified runs are excluded; widths are separate conditions and never pooled."]
            if len(eligible) > len(verified):
                lines.append(f"Excluded as unverified: {len(eligible) - len(verified)} eligible inference run(s).")
    lines.append("")

    lines += ["## Quality (first pass)", ""]
    complete_inference = [trial for trial in in_scope if trial.kind == "inference" and trial.complete]
    if plan is None:
        lines.append("Quality medians omitted: the run is unmeasured (no plan).")
    elif mode == "mini":
        rows = []
        for (model, width), group in sorted(_groups(complete_inference, lambda t: (t.model, str(t.width))).items()):
            metrics = sum(1 for trial in group if None not in (trial.recall, trial.f1, trial.accuracy, trial.coverage))
            rows.append((model, width, len(group), _ratio(metrics, len(group)), sum(1 for trial in group if trial.verification == VERIFIED)))
        lines += ["**SMOKE ONLY:** comparative quality medians are intentionally omitted.", ""]
        lines += _table(("Model", "Width", "Complete runs", "Quality fields", "Verified"), rows)
        lines += ["", "Only metric availability/coverage is shown; raw values remain in the smoke-only CSV audit tables."]
    elif mode != "full":
        lines.append("Quality medians omitted: mode is not confirmed `full`.")
    else:
        verified = [trial for trial in complete_inference if trial.verification == VERIFIED]
        if not verified:
            lines.append("Quality medians suppressed: no complete inference run is explicitly verified by the receipt.")
            if complete_inference:
                lines.append(f"{len(complete_inference)} complete inference run(s) excluded because they are unverified.")
        else:
            rows = []
            for (model, width), group in sorted(_groups(verified, lambda t: (t.model, str(t.width))).items()):
                metrics = sum(1 for trial in group if None not in (trial.recall, trial.f1, trial.accuracy, trial.coverage))
                rows.append((model, width, len(group), _ratio(metrics, len(group)), _fmt(_median(t.recall for t in group)),
                             _fmt(_median(t.f1 for t in group)), _fmt(_median(t.accuracy for t in group)), _fmt(_median(t.coverage for t in group))))
            lines += _table(("Model", "Width", "Verified runs", "Quality fields", "Macro recall", "Macro-F1", "Accuracy", "Coverage"), rows)
            lines += ["", "First-pass quality only; later passes diagnose repeat disagreement. `Quality fields`", "is the fraction of verified runs containing all four quality fields. Missing values", "stay missing, never zero. Only explicitly verified run ids with matching active", "attempts enter these medians."]
            if len(complete_inference) > len(verified):
                lines.append(f"Excluded as unverified: {len(complete_inference) - len(verified)} complete inference run(s).")
    lines.append("")

    lines += ["## Outputs", ""]
    lines.append(f"- Summary: `{summary_path}`")
    lines += [f"- Table: `{path}`" for path in csv_paths]
    lines += ["", "CSV tables are standard `analysis.py` outputs and retain incomplete/unverified rows;", "they are audit tables, not evidence that every row is a valid verified result."]
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
    summary_path = report_dir / "summary.md"
    temporary = summary_path.with_name(f".{summary_path.name}.tmp")
    temporary.write_text(render_summary(root_path, launcher, plan, plan_error, receipt, trials, scope_note, csv_paths, summary_path), encoding="utf-8")
    temporary.replace(summary_path)
    return {"root": root_path, "launcher": launcher, "plan": plan, "plan_error": plan_error, "verification": receipt,
            "trials": trials, "summary_path": summary_path, "csv_paths": csv_paths}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write RUN_ROOT/report/summary.md and analysis CSVs.")
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
