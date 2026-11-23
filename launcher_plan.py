"""Host preflight and executable configuration resolution for the launcher.

Pre-bootstrap module: module-level imports are standard library only; it never
imports an engine, downloads a model, installs anything, or initializes CUDA. The
resolved document is validated in memory by the project validator.

``prepare(mode, country)`` returns ``(config_document, discovery)``. Mode is
``"full"`` or ``"mini"``; country is a required ISO 3166-1 alpha-3 code. The
document is ``experiment.json``-shaped, always uses phase ``"full"``, and is
validated by ``configuration.validate_document`` before return. ``check_resume(config,
*, expected_machine_id=None, expected_gpu_uuid=None)`` re-checks a saved document
without re-selecting; failures raise ``ResumeCheckError``. ``machine_identity()``
returns a stable hashed Linux identity for host pinning.

A GPU model is selected only when exactly one compatible GPU 0 is visible and
pinned weights + reserve fit its *total* VRAM. Transient occupancy never changes
the selection: foreign compute, unverifiable queries, insufficient free VRAM and
insufficient disk are ``blockers`` (do not start); ``excluded`` holds persistent
hardware reasons only. ``energy.gpu_ids`` is ``[0]`` whenever a compatible GPU 0
is present (including CPU-only fallback), else ``[]``. Downloads count full pinned
shards unless a snapshot is fully verified. Disk is checked at each actual
snapshot path with a 5% margin; Qwen2B keeps its legacy sibling path.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import os
from pathlib import Path
import platform as platform_module
import re
import shutil
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PLAN_ROOT = ROOT
EXPERIMENT = ROOT / "experiment.json"
MODEL_MANIFEST = ROOT / "setup" / "model_manifest.json"
PHASE_NAME = "full"
MODES = ("full", "mini")
EXECUTION_PROTOCOL = "engine-session-v1"
VLLM_PYTHON = ".scratch/launcher/vllm/venv/bin/python"
QEV_PYTHON = ".scratch/launcher/qev-runtime/venv/bin/python"
QEV_ONEJEV = ".scratch/launcher/qev-runtime/OneJev"
MIB = 1024 ** 2
MIN_COMPUTE_CAPABILITY = (8, 0)  # Ampere or newer, pinned BF16 condition.
MIN_CUDA_128_DRIVER = (570, 26)  # CUDA 12.8 GA floor; fail closed below it.
GPU_FIELDS = ("index", "uuid", "driver_version", "memory.total", "memory.free", "compute_cap", "name")
GPU_QUERY = ("nvidia-smi", "--query-gpu=" + ",".join(GPU_FIELDS), "--format=csv,noheader,nounits")
COMPUTE_APPS_QUERY = ("nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits")
CLASSICAL_ADAPTERS = frozenset({"svm", "nb", "fasttext"})
GPU_ADAPTERS = frozenset({"vllm", "qev"})
MINI = {"batch_sizes": [1], "sample_size": 9, "warmup_rows": 3, "repetitions": 1,
        "device_idle_s": 2, "loaded_idle_s": 2, "settle_s": 2, "measure_training": False}
__all__ = ["ROOT", "PLAN_ROOT", "prepare", "check_resume", "machine_identity",
           "LauncherError", "ResumeCheckError"]

class LauncherError(RuntimeError):
    """The launcher cannot construct a trustworthy executable configuration."""

class ProbeError(LauncherError):
    """A bounded host query failed or returned malformed output."""

class ProbeUnavailable(ProbeError):
    """The query tool is not installed (distinct from a malformed query)."""

class ResumeCheckError(LauncherError):
    """A saved configuration no longer satisfies its runtime preconditions."""

def _configuration():
    try:
        import configuration  # noqa: PLC0415 - deliberately lazy
    except Exception as exc:  # pragma: no cover
        raise LauncherError(f"cannot import configuration validator: {exc}") from None
    return configuration

def _storage():
    try:
        import storage  # noqa: PLC0415 - deliberately lazy
    except Exception as exc:  # pragma: no cover
        raise LauncherError(f"cannot import storage: {exc}") from None
    return storage

def _load(path):
    try:
        return _storage().load(path)
    except (OSError, ValueError) as exc:
        raise LauncherError(f"cannot load {path}: {exc}") from None

def _validate(document):
    try:
        _configuration().validate_document(document)
    except ValueError as exc:
        raise LauncherError(f"resolved configuration failed validation: {exc}") from None

def _run(args):
    try:
        proc = subprocess.run(list(args), capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=10, check=False)
    except FileNotFoundError:
        raise ProbeUnavailable(f"{args[0]} is not installed") from None
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProbeError(f"{args[0]} query failed: {exc}") from None
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[:240]
        raise ProbeError(f"{args[0]} exited {proc.returncode}" + (f": {detail}" if detail else ""))
    return proc.stdout or ""

def _platform():
    system, machine = platform_module.system(), platform_module.machine()
    ok = sys.platform.startswith("linux") and system == "Linux" and machine.lower() in {"x86_64", "amd64"}
    return {"system": system, "machine": machine, "platform": platform_module.platform(),
            "supported": ok, "required": "Linux x86_64"}

def machine_identity():
    system, machine, hostname = platform_module.system(), platform_module.machine(), socket.gethostname()
    raw = ""
    for candidate in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        try:
            raw = Path(candidate).read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if raw:
            break
    seed = f"{system}\0{machine}\0{raw or hostname}"
    return {"system": system, "machine": machine, "hostname": hostname,
            "machine_id": hashlib.sha256(seed.encode("utf-8")).hexdigest(),
            "source": "machine-id" if raw else "hostname"}

def _gpu(status, reason=None, devices=(), device=None, apps=None):
    return {"status": status, "reason": reason, "devices": list(devices), "device": device,
            "apps": apps or {"status": "unknown", "pids": [], "reason": reason}}

def _gpu_rows(text):
    rows, seen = [], set()
    for line in str(text).splitlines():
        line = line.strip()
        if not line or line.lower().startswith("no devices were found"):
            continue
        # Name is last so a comma in the model name cannot shift fields.
        parts = [part.strip() for part in line.split(",", len(GPU_FIELDS) - 1)]
        if len(parts) != len(GPU_FIELDS):
            raise ProbeError(f"malformed nvidia-smi GPU row: {line!r}")
        row = dict(zip(GPU_FIELDS, parts, strict=True))
        try:
            index, total, free = int(row["index"]), int(row["memory.total"]), int(row["memory.free"])
        except ValueError:
            raise ProbeError(f"malformed numeric nvidia-smi GPU row: {line!r}") from None
        if (index < 0 or index in seen or total <= 0 or not 0 <= free <= total
                or not all(row[key] for key in ("uuid", "driver_version", "compute_cap", "name"))):
            raise ProbeError(f"implausible or incomplete nvidia-smi GPU row: {line!r}")
        seen.add(index)
        rows.append({"index": index, "uuid": row["uuid"], "driver_version": row["driver_version"],
                     "memory_total_mib": total, "memory_free_mib": free,
                     "memory_total_bytes": total * MIB, "memory_free_bytes": free * MIB,
                     "compute_cap": row["compute_cap"], "name": row["name"]})
    return sorted(rows, key=lambda item: item["index"])

def _compute_apps(text):
    pids = []
    for line in str(text).splitlines():
        line = line.strip()
        if not line or line.lower().startswith("no running processes"):
            continue
        try:
            pid = int(line)
        except ValueError:
            raise ProbeError(f"malformed nvidia-smi compute-app row: {line!r}") from None
        if pid <= 0:
            raise ProbeError(f"implausible compute-app pid: {line!r}")
        pids.append(pid)
    return pids

def _probe_gpu():
    try:
        text = _run(GPU_QUERY)
    except ProbeUnavailable as exc:
        return _gpu("absent", str(exc))
    except ProbeError as exc:
        return _gpu("absent" if "no devices were found" in str(exc).lower() else "unknown", str(exc))
    if not text.strip() or all(line.strip().lower().startswith("no devices were found")
                               for line in text.splitlines() if line.strip()):
        return _gpu("absent", "nvidia-smi reports no devices")
    try:
        devices = _gpu_rows(text)
    except ProbeError as exc:
        return _gpu("unknown", str(exc))
    try:
        pids = _compute_apps(_run(COMPUTE_APPS_QUERY))
        apps = {"status": "active" if pids else "none", "pids": pids, "reason": None}
    except ProbeError as exc:
        apps = {"status": "unknown", "pids": [], "reason": str(exc)}
    if len(devices) != 1:
        return _gpu("unsupported", f"exactly one visible GPU at index 0 is required; found {len(devices)}",
                    devices, apps=apps)
    device = devices[0]
    if device["index"] != 0:
        return _gpu("unsupported", f"GPU index 0 is unavailable; found index {device['index']}",
                    devices, device, apps)
    try:
        driver = tuple(int(part) for part in device["driver_version"].split("."))
        capability = tuple(int(part) for part in device["compute_cap"].split("."))
        if len(capability) != 2:
            raise ValueError
    except ValueError:
        return _gpu("unknown", "nvidia-smi returned an unparsable driver/compute capability",
                    devices, device, apps)
    device["compute_capability"] = list(capability)
    if capability < MIN_COMPUTE_CAPABILITY:
        return _gpu("unsupported", f"compute capability {device['compute_cap']} is below "
                    f"Ampere/BF16 floor {MIN_COMPUTE_CAPABILITY[0]}.{MIN_COMPUTE_CAPABILITY[1]}",
                    devices, device, apps)
    if len(driver) < 2 or driver[:2] < MIN_CUDA_128_DRIVER:
        return _gpu("unsupported", f"driver {device['driver_version']} is below CUDA 12.8 GA floor "
                    f"{MIN_CUDA_128_DRIVER[0]}.{MIN_CUDA_128_DRIVER[1]}", devices, device, apps)
    return _gpu("ok", None, devices, device, apps)

def _free_space(path):
    target = Path(path)
    while not target.exists() and target.parent != target:
        target = target.parent
    return target, shutil.disk_usage(target).free

def _margin(weight_bytes):
    return weight_bytes + weight_bytes // 20

def _disk(entries):
    """Check actual filesystems; entries are (label, path, required_bytes)."""
    groups, problems = {}, []
    try:
        for label, path, required in entries:
            target, free = _free_space(path)
            group = groups.setdefault(os.stat(target).st_dev, {
                "path": str(target), "free_bytes": free, "required_bytes": 0, "models": []})
            group["required_bytes"] += required
            group["models"].append(label)
    except OSError as exc:
        return {"ok": False, "free_bytes": 0, "required_bytes": sum(item[2] for item in entries),
                "checks": [], "error": str(exc)}, [f"disk check failed: {exc}"]
    for group in groups.values():
        group["ok"] = group["free_bytes"] >= group["required_bytes"]
        group["headroom_bytes"] = group["free_bytes"] - group["required_bytes"]
        if not group["ok"]:
            problems.append(f"insufficient disk at {group['path']}: {group['free_bytes']} bytes free but "
                            f"{group['required_bytes']} bytes required (weights + 5% margin)")
    checks = sorted(groups.values(), key=lambda item: item["path"])
    return {"ok": all(group["ok"] for group in checks) if checks else None,
            "free_bytes": min((group["free_bytes"] for group in checks), default=None),
            "required_bytes": sum(group["required_bytes"] for group in checks),
            "checks": checks,
            "policy": "weights + weights//20 per unverified snapshot, grouped by filesystem"}, problems

def _manifest_entry(manifest, model_id, spec):
    entry, safety = manifest["models"].get(model_id), spec.get("engine_safety") or {}
    expected = {key: spec.get(key) for key in ("adapter", "repo", "revision", "local_snapshot")}
    if (not isinstance(entry, dict) or any(entry.get(key) != value for key, value in expected.items())
            or entry.get("weight_shards_bytes") != safety.get("weight_shards_bytes")
            or not isinstance(entry.get("weight_shards_bytes"), dict)):
        raise LauncherError(f"models.{model_id} differs from its exact Hub manifest pin")
    return entry


def _snapshot(local_snapshot, weights):
    path = Path(local_snapshot)
    path = path if path.is_absolute() else Path(os.path.abspath(ROOT / local_snapshot))
    total, missing = sum(weights.values()), []
    if not (path / "config.json").is_file():
        missing.append("config.json")
    for name, expected in weights.items():
        try:
            actual = (path / name).stat().st_size
        except OSError:
            missing.append(name)
            continue
        if actual != expected:
            missing.append(f"{name} ({actual} != {expected})")
    return {"path": str(local_snapshot), "verified": not missing,
            "download_bytes": 0 if not missing else total,
            "reason": None if not missing else "not fully verified: " + ", ".join(missing[:3])}

def _select(document, manifest, gpu):
    phase, models = document["phases"][PHASE_NAME], document["models"]
    device = gpu["device"]
    total = device["memory_total_bytes"] if device else None
    free = device["memory_free_bytes"] if device else None
    selected, excluded, details = [], {}, {}
    for model_id in phase["model_ids"]:
        spec = models[model_id]
        adapter = spec["adapter"]
        if adapter in CLASSICAL_ADAPTERS:
            selected.append(model_id)
            details[model_id] = {"adapter": adapter, "selected": True, "reason": None}
            continue
        entry = _manifest_entry(manifest, model_id, spec)
        weights = entry["weight_shards_bytes"]
        weights_bytes = sum(weights.values())
        reserve_bytes = int(spec["engine_safety"]["reserve_mib"]) * MIB
        required = weights_bytes + reserve_bytes
        fits_total = None if total is None else required <= total
        fits_free = None if free is None else required <= free
        detail = {"adapter": adapter, "selected": False, "reason": None,
                  "required_bytes": required, "fits_total": fits_total,
                  "fits_free_now": fits_free, "snapshot": None, "download_bytes": None}
        if gpu["status"] == "ok" and fits_total:
            selected.append(model_id)
            detail.update(selected=True, snapshot=_snapshot(entry["local_snapshot"], weights))
            detail["download_bytes"] = detail["snapshot"]["download_bytes"]
        else:
            detail["reason"] = (
                f"GPU capability could not be verified ({gpu['reason']}); CPU-only fallback"
                if gpu["status"] == "unknown" else
                f"{gpu['reason']}; CPU-only fallback" if gpu["status"] != "ok" else
                f"pinned weights {weights_bytes / MIB:.0f} MiB + reserve {reserve_bytes / MIB:.0f} MiB "
                f"exceed GPU 0 total VRAM {total / MIB:.0f} MiB; no download or load")
            excluded[model_id] = detail["reason"]
        details[model_id] = detail
    return selected, excluded, details

def _normalize_mode(mode):
    if not isinstance(mode, str) or mode.strip().lower() not in MODES:
        raise ValueError(f"mode must be one of {list(MODES)}")
    return mode.strip().lower()

def _normalize_country(country):
    code = country.strip().upper() if isinstance(country, str) else ""
    if not re.fullmatch(r"[A-Z]{3}", code) or code == "XXX":
        raise ValueError("country is required as an explicit ISO 3166-1 alpha-3 code (no default)")
    return code

def _build(source, mode, country, gpu_ids, selected):
    document = deepcopy(source)
    phase, original = document["phases"][PHASE_NAME], deepcopy(document["phases"][PHASE_NAME])
    if original.get("execution_protocol") != EXECUTION_PROTOCOL:
        raise LauncherError(f"phases.{PHASE_NAME}.execution_protocol must be {EXECUTION_PROTOCOL!r}")
    document["models"] = {model_id: deepcopy(source["models"][model_id]) for model_id in selected}
    phase["model_ids"] = list(selected)
    if mode == "mini":
        phase.update(deepcopy(MINI))
    passes = {model_id: ({"1": 1} if mode == "mini" else deepcopy(original["passes"][model_id]))
              for model_id in selected}
    phase["passes"] = passes
    phase["loaded_idle_adapters"] = [model_id for model_id in original["loaded_idle_adapters"]
                                     if model_id in selected]
    document["phases"] = {PHASE_NAME: phase}
    document["energy"]["country_iso_code"] = country
    document["energy"]["gpu_ids"] = list(gpu_ids)
    for model_id, spec in document["models"].items():
        if spec["adapter"] == "vllm":
            spec["python"] = spec["engine_safety"]["expected_python"] = VLLM_PYTHON
        elif spec["adapter"] == "qev":
            spec["params"]["python"], spec["params"]["onejev_path"] = QEV_PYTHON, QEV_ONEJEV
    return document

def _blockers(host, selected, details):
    gpu, apps, status = host["gpu"], host["gpu"]["apps"], host["gpu"]["status"]
    gpu_selected = [model_id for model_id in selected if details[model_id]["adapter"] in GPU_ADAPTERS]
    out = []
    if not host["platform"]["supported"]:
        out.append(f"host platform {host['platform']['system']} {host['platform']['machine']} "
                   "is unsupported; Linux x86_64 required")
    if status == "unknown":
        out.append(f"GPU capability could not be verified: {gpu['reason']}; failing closed")
    elif status in {"ok", "unsupported"} and apps["status"] == "unknown":
        out.append(f"GPU compute-app query failed: {apps['reason']}; measurement refused")
    elif status in {"ok", "unsupported"} and apps["status"] == "active":
        out.append("foreign GPU compute is active (pid(s) " + ", ".join(str(pid) for pid in apps["pids"])
                   + "); stop it before starting. Selection is unchanged.")
    if status == "ok" and gpu_selected:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible not in {None, "0", gpu["device"]["uuid"]}:
            out.append(f"CUDA_VISIBLE_DEVICES={visible!r} does not expose only GPU 0")
    if status == "ok" and apps["status"] == "none":
        short = [model_id for model_id in gpu_selected if details[model_id]["fits_free_now"] is False]
        if short:
            out.append("insufficient currently free VRAM for " + ", ".join(short)
                       + f" (only {gpu['device']['memory_free_mib']} MiB free); "
                       "transient occupancy, selection unchanged")
    return out

def _notes(selected, excluded, details, downloads, mode):
    notes = ["bootstrap must provide the .scratch/launcher engine environments"]
    if excluded:
        notes.append("hardware exclusions: " + ", ".join(excluded))
    if not any(details[model_id]["adapter"] in GPU_ADAPTERS for model_id in selected):
        notes.append("CPU-only fallback: every GPU model is excluded")
    if downloads:
        notes.append("download_bytes counts full pinned shards unless a snapshot is fully verified; "
                     "Hugging Face cache reuse is not assumed" if any(downloads.values())
                     else "all selected GPU snapshots are fully verified locally; download_bytes=0")
    if "qwen2b" in selected:
        notes.append("Qwen2B keeps the legacy sibling snapshot ../inspect_pilot_study/"
                     ".scratch/gpu/model; its parent must be writable and disk is checked there")
    if mode == "mini":
        notes.append("mini keeps phase 'full' and the full classical train split/protocol; one "
                     "repetition, width 1, 9 samples/3 warmup, passes=1, 2 s idle/settle, "
                     "measure_training=false (classical fitting still occurs in setup)")
    return notes

def prepare(mode: str, country: str) -> tuple[dict, dict]:
    """Probe the host and return (executable document, JSON-safe discovery)."""
    mode, country = _normalize_mode(mode), _normalize_country(country)
    source, manifest = _load(EXPERIMENT), _load(MODEL_MANIFEST)
    platform = _platform()
    gpu = (_probe_gpu() if platform["supported"] else
           _gpu("unsupported", "host platform is unsupported",
                apps={"status": "absent", "pids": [], "reason": "platform unsupported"}))
    host = {"platform": platform, "gpu": gpu}
    selected, excluded, details = _select(source, manifest, gpu)
    if not selected:
        raise LauncherError("no model in the active phase can run on this host")
    gpu_ids = [0] if gpu["status"] == "ok" else []
    downloads = {model_id: details[model_id]["download_bytes"] for model_id in selected
                 if details[model_id]["adapter"] in GPU_ADAPTERS}
    entries = [(model_id, os.path.abspath(ROOT / source["models"][model_id]["local_snapshot"]),
                _margin(details[model_id]["download_bytes"]))
               for model_id in selected if details[model_id]["adapter"] in GPU_ADAPTERS]
    disk, disk_problems = _disk(entries or [(PHASE_NAME, ROOT, 0)])
    blockers = _blockers(host, selected, details) + disk_problems
    document = _build(source, mode, country, gpu_ids, selected)
    _validate(document)
    metering = ({"gpu_ids": [0], "policy": "GPU 0 is metered for every trial, including CPU-only work"}
                if gpu_ids else
                {"gpu_ids": [], "policy": "no GPU meter: " + (gpu["reason"] or gpu["status"])})
    discovery = {
        "schema_version": 1, "mode": mode, "phase": PHASE_NAME, "country": country,
        "selected": list(selected), "excluded": dict(excluded),
        "download_bytes": downloads, "download_total_bytes": sum(downloads.values()),
        "disk": disk, "machine": machine_identity(), "platform": platform,
        "gpu": {"status": gpu["status"], "reason": gpu["reason"], "gpu_ids": list(gpu_ids),
                "uuid": gpu["device"]["uuid"] if gpu["status"] == "ok" else None,
                "metering": metering, "devices": gpu["devices"]},
        "compute_apps": gpu["apps"], "models": details,
        "engine_paths": {"vllm": {"python": VLLM_PYTHON, "expected_python": VLLM_PYTHON}, "qev": {"python": QEV_PYTHON, "onejev_path": QEV_ONEJEV}},
        "checks": {"platform_ok": platform["supported"], "gpu_status": gpu["status"], "compute_apps_status": gpu["apps"]["status"], "disk_ok": disk["ok"]},
        "blockers": blockers, "notes": _notes(selected, excluded, details, downloads, mode),
        "config_sha256": _storage().identity(document),
    }
    return document, discovery

def _saved(config):
    if not isinstance(config, dict):
        raise ResumeCheckError("saved configuration must be a JSON object")
    phases, models, energy = config.get("phases"), config.get("models"), config.get("energy")
    if not isinstance(phases, dict) or not isinstance(phases.get(PHASE_NAME), dict):
        raise ResumeCheckError(f"saved document has no phases.{PHASE_NAME} object")
    model_ids = phases[PHASE_NAME].get("model_ids")
    if (not isinstance(models, dict) or not isinstance(model_ids, list) or not model_ids
            or not isinstance(energy, dict)):
        raise ResumeCheckError("saved document has no selected models or energy object")
    for model_id in model_ids:
        if not isinstance(model_id, str) or not isinstance(models.get(model_id), dict):
            raise ResumeCheckError(f"saved document references an unknown model: {model_id!r}")
    gpu_ids = energy.get("gpu_ids")
    if (not isinstance(gpu_ids, list) or any(type(value) is not int for value in gpu_ids)
            or gpu_ids not in ([], [0])):
        raise ResumeCheckError(f"only GPU 0 is supported; saved energy.gpu_ids={gpu_ids}")
    return models, model_ids, gpu_ids

def _resume_problems(gpu, gpu_ids, gpu_models, models, expected_gpu_uuid):
    problems, status, apps = [], gpu["status"], gpu["apps"]
    if expected_gpu_uuid is not None and expected_gpu_uuid not in {row["uuid"] for row in gpu["devices"]}:
        problems.append(f"expected GPU UUID {expected_gpu_uuid!r} is not visible")
    if status == "unknown":
        return problems + [f"GPU capability could not be verified: {gpu['reason']}"]
    if status == "absent":
        if gpu_models:
            problems.append("saved selection contains GPU models but no NVIDIA GPU is visible")
        if gpu_ids:
            problems.append(f"saved energy.gpu_ids={gpu_ids} but no NVIDIA GPU is visible")
        return problems
    app_problem = (f"GPU compute-app query failed: {apps['reason']}" if apps["status"] == "unknown" else
                   "foreign GPU compute is active: pid(s) " + ", ".join(str(pid) for pid in apps["pids"])
                   if apps["status"] == "active" else None)
    if status == "unsupported":
        if gpu_models or gpu_ids:
            problems.append(f"GPU 0 is not compatible with the saved configuration: {gpu['reason']}")
        if app_problem:
            problems.append(app_problem)
        return problems
    device = gpu["device"]
    if expected_gpu_uuid is not None and device["uuid"] != expected_gpu_uuid:
        problems.append(f"GPU UUID mismatch: saved {expected_gpu_uuid!r}, current {device['uuid']!r}")
    live = {row["index"] for row in gpu["devices"]}
    if any(index not in live for index in gpu_ids):
        problems.append(f"saved energy.gpu_ids={gpu_ids} do not match live GPU indices {sorted(live)}")
    if app_problem:
        problems.append(app_problem)
    if not gpu_models:
        if not gpu_ids:
            problems.append("GPU 0 is present but saved energy.gpu_ids is empty; metering policy changed")
        return problems
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in {None, "0", device["uuid"]}:
        problems.append(f"CUDA_VISIBLE_DEVICES={visible!r} does not expose only GPU 0")
    if gpu_ids != [0]:
        problems.append("saved selection contains GPU models but energy.gpu_ids is not [0]")
    for model_id in gpu_models:
        safety = models[model_id].get("engine_safety") or {}
        weights = safety.get("weight_shards_bytes")
        try:
            required = sum(weights.values()) + int(safety["reserve_mib"]) * MIB
        except (AttributeError, KeyError, TypeError, ValueError):
            problems.append(f"saved model {model_id} has an unreadable VRAM preflight")
            continue
        if required > device["memory_total_bytes"]:
            problems.append(f"saved model {model_id} does not fit GPU 0 total VRAM")
        if required > device["memory_free_bytes"]:
            problems.append(f"saved model {model_id} needs {required / MIB:.0f} MiB but only "
                            f"{device['memory_free_bytes'] / MIB:.0f} MiB are free")
    return problems

def check_resume(config: dict, *, expected_machine_id: str | None = None,
                 expected_gpu_uuid: str | None = None) -> None:
    """Re-check a saved launcher document without changing its selection.

    Pass saved ``discovery["machine"]["machine_id"]`` / ``discovery["gpu"]["uuid"]``
    to pin host and GPU. Missing selected snapshots may be reprovisioned by the
    installer, subject to the disk check.
    """
    models, model_ids, gpu_ids = _saved(config)
    platform = _platform()
    problems = []
    if not platform["supported"]:
        problems.append(f"host platform {platform['system']} {platform['machine']} is unsupported")
    if expected_machine_id is not None and machine_identity()["machine_id"] != str(expected_machine_id):
        problems.append("machine identity mismatch")
    if problems:
        raise ResumeCheckError("resume check failed: " + "; ".join(problems))
    gpu = _probe_gpu()
    gpu_models = [model_id for model_id in model_ids if models[model_id].get("adapter") in GPU_ADAPTERS]
    for model_id in model_ids:
        if models[model_id].get("adapter") not in GPU_ADAPTERS | CLASSICAL_ADAPTERS:
            problems.append(f"saved model {model_id} has an unsupported adapter")
    problems.extend(_resume_problems(gpu, gpu_ids, gpu_models, models, expected_gpu_uuid))
    entries = []
    for model_id in gpu_models:
        spec = models[model_id]
        weights = (spec.get("engine_safety") or {}).get("weight_shards_bytes")
        local = spec.get("local_snapshot")
        if isinstance(weights, dict) and weights and isinstance(local, str):
            entries.append((model_id, os.path.abspath(ROOT / local),
                            _margin(_snapshot(local, weights)["download_bytes"])))
    if entries:
        problems.extend(_disk(entries)[1])
    if problems:
        raise ResumeCheckError("resume check failed: " + "; ".join(problems))
