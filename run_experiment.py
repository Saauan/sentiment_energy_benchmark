#!/usr/bin/env python3
"""Interactive bootstrap and resume front end; benchmark.py owns measurements."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import platform
import shlex
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
WORK = ROOT / ".scratch"
RETRY_OPTIONS = ("--retry-failed", "--retry-skipped")


def banner(title: str) -> None:
    print(f"\n== {title} ==", flush=True)


def say(message: str) -> None:
    print(message, flush=True)


def confirm(message: str, yes: bool) -> bool:
    if yes:
        return True
    if not sys.stdin.isatty():
        raise ValueError("confirmation requires a terminal or explicit --yes")
    return input(f"{message} [y/N] ").strip().lower() in {"y", "yes"}


def country_code(value: str) -> str:
    value = value.upper()
    if len(value) != 3 or not value.isascii() or not value.isalpha():
        raise argparse.ArgumentTypeError("use a three-letter ISO 3166-1 country code, e.g. FRA or USA")
    return value


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("full", "mini"), nargs="?")
    p.add_argument("--country", type=country_code, help="host's ISO 3166-1 alpha-3 country; required for new runs")
    paths = p.add_mutually_exclusive_group()
    paths.add_argument("--out", type=Path, help="new output directory (default: a timestamped directory under artifacts/)")
    paths.add_argument("--resume", type=Path, help="resume a previous launcher run from its saved configuration")
    p.add_argument("--dry-run", action="store_true", help="preview without downloads, installs, or writes")
    p.add_argument("--yes", action="store_true", help="approve the displayed plan and its downloads without prompting")
    p.add_argument("--validated", action="store_true", help="full only: attest this exact setup/conditions already passed readiness")
    for flag in RETRY_OPTIONS:
        p.add_argument(flag, action="store_true", help=f"on resume, explicitly {flag[2:].replace('-', ' ')} trials")
    return p


def show_plan(root: Path, receipt: dict, cfg: dict) -> None:
    banner(f"TweetEval sentiment benchmark · {receipt['mode'].upper()}")
    phase = cfg["phases"]["full"]
    discovery = receipt["discovery"]
    say(f"Output: {root}\nMeter country: {cfg['energy']['country_iso_code']} | GPU IDs: {cfg['energy']['gpu_ids']}")
    say(f"Workload: {phase['sample_size']} test tweets; widths {phase['batch_sizes']}; {phase['repetitions']} repetition(s)")
    gpu = discovery.get("gpu", {})
    if devices := gpu.get("devices"):
        say("Detected NVIDIA GPU(s): " + "; ".join(str(device.get("name", "unknown")) for device in devices))
    sizes = discovery["download_bytes"]
    for model in discovery["selected"]:
        note = "" if model not in sizes else f" — {'weights cached' if not sizes[model] else f'~{sizes[model] / 1e9:.2f} GB weights to fetch'}"
        say(f"  Include {model}{note}")
    for model, reason in discovery["excluded"].items():
        say(f"  Exclude {model} — {reason}")
    say(f"Weight estimate at selection: {sum(sizes.values()) / 1e9:.2f} GB (excludes Python, packages and model metadata).")
    if disk := discovery.get("disk"):
        say(f"Disk at selection: {disk['free_bytes'] / 1e9:.2f} GB free; estimated weight staging needs {disk['required_bytes'] / 1e9:.2f} GB.")
    for reason in discovery.get("blockers", []):
        say(f"  Blocked — {reason}")
    if receipt["mode"] == "mini":
        say("Smoke test only; not comparable to full results. Classical training still uses the full training split.")
    elif receipt["readiness_required"]:
        say("Readiness: one separate pilot repetition at each selected model's planned widths before the full run.")
    else:
        say("Readiness: already validated on this setup (operator attestation).")


def next_log_dir(root: Path) -> Path:
    parent = root / "launcher-logs"
    parent.mkdir(exist_ok=True)
    for index in range(1, 1_000_000):
        path = parent / f"invocation-{index:04d}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            pass
    raise RuntimeError("launcher log namespace exhausted")


@contextmanager
def forward_signal(proc):
    """Let the runner finish and persist its interrupted attempt before returning."""
    def forward(signum, _frame):
        say("\nStopping the benchmark; waiting for it to save its checkpoint…")
        try:
            proc.send_signal(signum)
        except ProcessLookupError:
            pass

    saved = {sig: signal.signal(sig, forward) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)


def run_command(client: Path, root: Path, config_path: Path, command: str, log: Path, *flags: str) -> int:
    argv = [str(client), "-u", "benchmark.py", command, "--phase", "full", "--out", str(root)]
    if command != "status":
        argv.extend(("--config", str(config_path)))
    argv.extend(flags)
    with log.open("x", encoding="utf-8") as stdout:
        if command != "run":
            return subprocess.run(argv, cwd=ROOT, stdout=stdout, check=False).returncode
        with subprocess.Popen(argv, cwd=ROOT, stdout=stdout, start_new_session=True) as proc:
            with forward_signal(proc):
                return proc.wait()


def show_status(report: dict) -> None:
    say("  " + " · ".join(f"{n} {status}" for status, n in report["counts"].items()))
    for row in report["runs"]:
        if row["status"] in {"failed", "skipped", "incomplete", "running"}:
            say(f"    {row['run_id']}: {row['status']} — {row.get('reason') or 'see trial log'}")


def has_working_predictions(report: dict, root: Path) -> bool:
    import storage

    inferred = 0
    for row in report["runs"]:
        if row["kind"] == "inference":
            result = storage.load(root / row["run_id"] / row["active_attempt"] / "result.json")
            if not result["counts"].get("valid"):
                say(f"No valid predictions: {row['run_id']}")
                return False
            inferred += 1
    return inferred > 0


def ready(report: dict, root: Path) -> bool:
    if not report.get("ok") or any(n for status, n in report["counts"].items() if status != "complete"):
        return False
    if report["low_sampling"]:
        say("Readiness found insufficient sampling: " + ", ".join(row["run_id"] for row in report["low_sampling"]))
        return False
    return has_working_predictions(report, root)


def configure_readiness(cfg: dict) -> dict:
    pilot = deepcopy(cfg)
    pilot["phases"]["full"]["repetitions"] = 1
    return pilot


def verify(client: Path, root: Path, config_path: Path, log: Path) -> dict:
    import storage

    code = run_command(client, root, config_path, "verify", log)
    if code not in (0, 1):
        # Do not let a stale success receipt represent the current artifacts.
        report = {"ok": False, "inconsistencies": [f"verify exited {code}; see {log}"]}
    else:
        report = storage.load(log)
    storage.atomic_json(root / "verification.json", report)
    return report


def run_readiness(client: Path, root: Path, cfg: dict, logs: Path, flags: list[str]) -> bool:
    import storage

    banner("Readiness check · separate from benchmark results")
    pilot = root / "readiness"
    pilot.mkdir(exist_ok=True)
    path = pilot / "resolved.json"
    expected = configure_readiness(cfg)
    if path.exists():
        if storage.load(path) != expected:
            raise ValueError("stored readiness configuration was modified")
    else:
        storage.atomic_json(path, expected)
    if (pilot / "plan.json").exists():
        with storage.measurement_lock(WORK / "measurement.lock"):
            report = verify(client, pilot, path, logs / "readiness-before-verify.json")
            if ready(report, pilot):
                say("Saved readiness results passed; not repeating completed trials.")
                return True
            if not report.get("ok") or report.get("low_sampling"):
                raise ValueError(f"readiness cannot be resumed unchanged; check {pilot / 'verification.json'}")
    code = run_command(client, pilot, path, "run", logs / "readiness-run.json", "--progress", *flags)
    if code in (130, -signal.SIGINT, -signal.SIGTERM):
        raise KeyboardInterrupt
    with storage.measurement_lock(WORK / "measurement.lock"):
        report = verify(client, pilot, path, logs / "readiness-verify.json")
        if "counts" in report:
            show_status(report)
        return code == 0 and ready(report, pilot)


def resume_hint(root: Path) -> str:
    return f"python3 run_experiment.py --resume {shlex.quote(str(root))}"


def recheck(config: dict, discovery: dict) -> None:
    from launcher_plan import check_resume

    pins = {"expected_machine_id": discovery["machine"]["machine_id"]}
    if gpu_uuid := discovery.get("gpu", {}).get("uuid"):
        pins["expected_gpu_uuid"] = gpu_uuid
    check_resume(config, **pins)


def launcher(args) -> int:
    import launcher_plan
    import launcher_setup
    import storage

    if args.resume:
        if args.mode or args.country or args.validated:
            raise ValueError("--resume uses the saved mode, country, and readiness policy")
        root = args.resume.resolve()
        receipt = storage.load(root / "launcher.json")
        cfg = storage.load(root / "resolved.json")
        if (receipt.get("format") != 1 or receipt.get("mode") not in {"full", "mini"}
                or receipt.get("config_sha256") != storage.identity(cfg)):
            raise ValueError("saved launcher metadata/configuration is unsupported or was modified")
    else:
        if not args.mode or not args.country:
            raise ValueError("new runs require a mode (full or mini) and --country")
        if (args.retry_failed or args.retry_skipped) or (args.validated and args.mode != "full"):
            raise ValueError("retry flags require --resume; --validated is for new full runs only")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        root = (args.out or ROOT / "artifacts" / f"{args.mode}-{stamp}").resolve()
        if root.exists():
            raise ValueError(f"output already exists: {root}; use --resume for a launcher run or a new --out")
        cfg, discovery = launcher_plan.prepare(args.mode, args.country)
        receipt = {"format": 1, "mode": args.mode, "discovery": discovery,
                   "config_sha256": storage.identity(cfg),
                   "readiness_required": args.mode == "full" and not args.validated}

    show_plan(root, receipt, cfg)
    if args.resume:
        say("Resuming saved conditions. Pending/interrupted work continues; completed trials are unchanged.")
        say(f"Failed trials: {'retry' if args.retry_failed else 'leave unchanged'}; skipped trials: {'retry' if args.retry_skipped else 'leave unchanged'}.")
        plan = root / "plan.json"
        if plan.exists():
            from benchmark import status_report

            show_status(status_report(root, storage.load(plan)))
        recheck(cfg, receipt["discovery"])
    if receipt["discovery"].get("blockers"):
        say("Preflight blocked this run; resolve the issues above and retry. Nothing was changed.")
        return 2
    if args.dry_run:
        say("Dry run: no installation, measurements, or writes.")
        return 0
    if not confirm("Proceed? This can install pinned dependencies, download weights and start measurements.", args.yes):
        say("Cancelled; nothing written.")
        return 0

    config_path = root / "resolved.json"
    flags = [f for f in RETRY_OPTIONS if getattr(args, f[2:].replace("-", "_"))]
    with storage.measurement_lock(WORK / "launcher.lock"):
        with storage.measurement_lock(WORK / "measurement.lock"):
            recheck(cfg, receipt["discovery"])
            if not args.resume:
                root.mkdir(parents=True, exist_ok=False)
                storage.atomic_json(config_path, cfg)
                storage.atomic_json(root / "launcher.json", receipt)
            logs = next_log_dir(root)
            banner("Pinned setup")
            say(f"Run root (also for resume): {root}")
            client = launcher_setup.provision(cfg, say)
        if receipt["readiness_required"] and not run_readiness(client, root, cfg, logs, flags):
            say(f"Full run not started: readiness failed or was interrupted. See {root / 'readiness'}.")
            return 1
        banner("Run benchmark")
        run_code = run_command(client, root, config_path, "run", logs / "run.json", "--progress", *flags)
        if run_code in (130, -signal.SIGINT, -signal.SIGTERM):
            say(f"Interrupted. Continue with: {resume_hint(root)}")
            return 130
        banner("Verify and build local report")
        with storage.measurement_lock(WORK / "measurement.lock"):
            report = verify(client, root, config_path, logs / "verify.json")
            rendered = subprocess.run([str(client), "launcher_report.py", str(root)], cwd=ROOT, check=False)
        say(f"Invocation logs: {logs}")
        counts = report.get("counts", {})
        if run_code or rendered.returncode or not report.get("ok") or any(n for status, n in counts.items() if status != "complete"):
            say(f"Run is not fully complete and verified. Resume with: {resume_hint(root)}")
            if counts.get("failed") or counts.get("skipped"):
                say("After addressing the cause, opt in to retries with --retry-failed and/or --retry-skipped.")
            return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    # Keep a first-time dry run read-only, including Python's local import cache.
    sys.dont_write_bytecode = True
    args = parser().parse_args(argv)
    if platform.system() != "Linux" or platform.machine() != "x86_64" or sys.version_info < (3, 11):
        print("Launcher requires Linux x86_64 and Python 3.11 or newer to bootstrap.", file=sys.stderr)
        return 2
    try:
        return launcher(args)
    except (ValueError, OSError, EOFError, RuntimeError, KeyError, subprocess.SubprocessError) as exc:
        print(f"Launcher: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nLauncher interrupted. If a run root was created, use --resume with that directory.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
