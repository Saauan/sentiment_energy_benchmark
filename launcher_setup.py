#!/usr/bin/env python3
"""User-space bootstrap for the benchmark client and selected engines.

Call :func:`provision` only after user confirmation, under the coordinator's
launcher/benchmark exclusion, with the *frozen* ``launcher_plan.prepare``
document.  That document is never mutated: engine paths must already be the
launcher paths below or provisioning stops.  No measurement lock is taken here,
and the pinned setup helpers do not take it either.

``uv`` is an explicit prerequisite; no sudo and no curl|sh.  After confirmation
uv may fetch exact Python 3.12.13 in user space.  Existing launcher-owned
environments are reused via content-addressed receipts; unrelated directories
are never overwritten and a failed setup is safe to retry.  Only selected
engines install.  Inputs are hash-checked before the pinned fetch.  Models use
``setup/gpu/fetch_models.sh`` one at a time, so its VRAM/disk/foreign-compute
guard runs immediately before each download; any post-confirmation guard
failure is hard and actionable, never a silent skip.

QEV is the demonstrated fallback: torch/torchvision 2.10.0/0.25.0+cu128,
Transformers 5.19.0, exact OneJev commit plus the checkpoint ``serve.patch``
(git blob validated), and no causal-conv1d (native F.conv1d stays active).
The snapshot's ``setup.sh`` is never executed.  The returned path is the lexical
``.venv/bin/python`` anchored to this repository, never symlink-resolved.  One
append-only log is ``.scratch/launcher/provision.log``; receipts are beside it in
``.scratch/launcher/receipts.json``.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from typing import Any, Callable, Sequence

import storage

ROOT = Path(__file__).resolve().parent
SUPPORTED_PYTHON = "3.12.13"
ONEJEV_REVISION = "81ce62f1597c91e46767d4d02d6ac2e18534fe94"
ONEJEV_PATCH_BLOB = "7a432e8d254a9838b04ca3a155547ec3b3c70e78"
ONEJEV_REPOSITORY = "https://github.com/OmniJev/OneJev.git"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
PYPI_INDEX = "https://pypi.org/simple"

CLIENT_PYTHON_REL = ".venv/bin/python"
VLLM_PYTHON_REL = ".scratch/launcher/vllm/venv/bin/python"
QEV_PYTHON_REL = ".scratch/launcher/qev-runtime/venv/bin/python"
QEV_ONEJEV_REL = ".scratch/launcher/qev-runtime/OneJev"
VLLM_ROOT_REL = ".scratch/launcher/vllm"
QEV_ROOT_REL = ".scratch/launcher/qev-runtime"

_ADAPTERS = {"vllm", "qev", "svm", "nb", "fasttext"}
_ENGINES = ("vllm", "qev")
_CLIENT_PROBE = """
import json
from importlib.metadata import version
names = ["codecarbon", "jsonschema", "numpy", "pandas", "scikit-learn",
         "fasttext-wheel", "threadpoolctl", "altair", "marimo", "pytest"]
observed = {name: version(name) for name in names}
import codecarbon, fasttext, jsonschema, numpy, pandas, sklearn
print(json.dumps(observed, sort_keys=True))
"""
_VLLM_PROBE = """
import json
from importlib.metadata import version
import torch
print(json.dumps({"vllm": version("vllm"), "torch_pkg": version("torch"),
                  "torch": torch.__version__, "cuda": torch.version.cuda,
                  "transformers": version("transformers"),
                  "huggingface_hub": version("huggingface-hub")}, sort_keys=True))
"""
_QEV_DEPS_PROBE = """
import importlib.util, json
from importlib.metadata import version
import torch, torchvision, transformers
print(json.dumps({"torch": torch.__version__, "torchvision": torchvision.__version__,
                  "cuda": torch.version.cuda, "transformers": transformers.__version__,
                  "flash_linear_attention": version("flash-linear-attention"),
                  "fla_core": version("fla-core"),
                  "causal_conv1d": importlib.util.find_spec("causal_conv1d") is not None}, sort_keys=True))
"""
_QEV_SOURCE_PROBE = """
import json
from importlib.metadata import version
import qev.readout
from qev.engine import DecisionEngine
print(json.dumps({"qev": version("qev"), "readout": qev.readout.__name__,
                  "engine": DecisionEngine.__name__}, sort_keys=True))
"""


class SetupError(RuntimeError):
    """Actionable bootstrap failure; the coordinator reports and stops."""


class _Log:
    """One append-only setup log; every line is also exported through ``emit``."""

    def __init__(self, path: Path, emit: Callable[[str], None]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = path.open("a", encoding="utf-8")
        self._emit = emit

    def line(self, text: str, *, detail: bool = False) -> None:
        for part in str(text).splitlines() or [""]:
            stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self._stream.write(f"{stamp} {'| ' if detail else ''}{part}\n")
            self._stream.flush()
            self._emit(part)

    def command(self, argv: Sequence[object]) -> None:
        self.line("$ " + shlex.join(str(item) for item in argv))

    def close(self) -> None:
        self._stream.close()


def _path(root: Path, relative: str) -> Path:
    return Path(root) / relative


def _lexical(path: Path) -> Path:
    """Absolute path without resolving a venv interpreter symlink."""
    return Path(os.path.abspath(os.fspath(path)))


def _hash(path: Path) -> str:
    try:
        return storage.content_hash(Path(path))
    except OSError as exc:
        raise SetupError(f"cannot hash required file {path}: {exc}") from None


def _tail(output: str, limit: int = 20) -> str:
    lines = [line for line in str(output).strip().splitlines() if line.strip()]
    return "\n".join(lines[-limit:]) if lines else "(no output)"


def _run(
    argv: Sequence[object],
    log: _Log,
    *,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    check: bool = True,
    quiet: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = [str(item) for item in argv]
    cwd = Path(ROOT) if cwd is None else Path(cwd)
    env = _env(ROOT) if env is None else env
    if not quiet:
        log.command(command)
    try:
        result = subprocess.run(command, cwd=str(cwd), env=env,
                                capture_output=True, text=True, check=False)
    except OSError as exc:
        raise SetupError(f"could not execute {shlex.join(command)}: {exc}") from None
    output = result.stdout + result.stderr
    if not quiet:
        for line in output.splitlines():
            log.line(line, detail=True)
    if check and result.returncode:
        raise SetupError(f"command failed with exit {result.returncode}: "
                         f"{shlex.join(command)}\n{_tail(output)}")
    return result


def _env(root: Path) -> dict[str, str]:
    launcher = _path(root, ".scratch/launcher")
    env = os.environ.copy()
    env["UV_CACHE_DIR"] = str(launcher / "uv-cache")
    env["UV_PYTHON_INSTALL_DIR"] = str(launcher / "uv-python")
    env["UV_PYTHON_DOWNLOADS"] = "automatic"
    env["HF_HOME"] = str(launcher / "hf")
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    return env


def _which(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise SetupError(
            f"{name!r} is required on PATH; install it in user space "
            "(the launcher never uses sudo or curl|sh)"
        )
    return found


def _receipts(root: Path) -> dict[str, Any]:
    path = _path(root, ".scratch/launcher/receipts.json")
    if not path.exists():
        return {}
    try:
        value = storage.load(path)
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _receipt_matches(root: Path, name: str, key: dict[str, Any]) -> bool:
    entry = _receipts(root).get(name)
    return isinstance(entry, dict) and entry.get("key") == key


def _record(root: Path, name: str, key: dict[str, Any], observed: Any) -> None:
    receipts = _receipts(root)
    receipts[name] = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      "key": key, "observed": observed}
    path = _path(root, ".scratch/launcher/receipts.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    storage.atomic_json(path, receipts)


def _probe(root: Path, python: Path, code: str, label: str, log: _Log) -> dict[str, Any]:
    result = _run([python, "-c", code], log, cwd=root, check=False, quiet=True)
    try:
        value = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        value = None
    if result.returncode or not isinstance(value, dict):
        raise SetupError(f"{label} probe failed using {python}\n{_tail(result.stdout + result.stderr)}")
    return value


def _pins(observed: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    for key, value in expected.items():
        if observed.get(key) != value:
            raise SetupError(f"{label} mismatch: {key}={observed.get(key)!r}, expected {value!r}")


VLLM_PINS = {"vllm": "0.19.0", "torch_pkg": "2.10.0", "torch": "2.10.0+cu128",
             "cuda": "12.8", "transformers": "4.57.6", "huggingface_hub": "0.36.2"}
QEV_ENV_PINS = {"torch": "2.10.0+cu128", "torchvision": "0.25.0+cu128", "cuda": "12.8",
                "transformers": "5.19.0", "flash_linear_attention": "0.5.2",
                "fla_core": "0.5.2", "causal_conv1d": False}
QEV_SOURCE_PINS = {"qev": "0.1.0", "readout": "qev.readout", "engine": "DecisionEngine"}


def _find_python(root: Path, uv: str, log: _Log) -> bool:
    found = _run([uv, "python", "find", SUPPORTED_PYTHON], log, cwd=root, check=False, quiet=True)
    return found.returncode == 0 and any(
        Path(line.strip()).is_absolute() and Path(line.strip()).exists()
        for line in reversed(found.stdout.splitlines())
    )


def _ensure_python(root: Path, log: _Log) -> None:
    uv = _which("uv")
    if _find_python(root, uv, log):
        return
    log.line(f"Exact Python {SUPPORTED_PYTHON} is absent; uv will fetch it into "
             f"{_path(root, '.scratch/launcher/uv-python')} (user space, no sudo).")
    _run([uv, "python", "install", SUPPORTED_PYTHON], log, cwd=root)
    if not _find_python(root, uv, log):
        raise SetupError(f"uv could not provide the exact supported Python {SUPPORTED_PYTHON}")


def _python_version(root: Path, python: Path, log: _Log) -> str | None:
    if not Path(python).exists():
        return None
    try:
        result = _run([python, "-c", "import platform; print(platform.python_version())"],
                      log, cwd=root, check=False, quiet=True)
    except SetupError:
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return lines[-1] if result.returncode == 0 and lines else None


def _owned_venv(
    root: Path, directory: Path, venv: Path, python: Path, label: str, log: _Log
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / ".launcher-owned"
    if venv.exists():
        if venv.is_symlink() or not venv.is_dir():
            raise SetupError(f"{label} venv path is not a directory: {venv}")
        version = _python_version(root, python, log) if python.exists() else None
        if version == SUPPORTED_PYTHON:
            marker.write_text(f"launcher_setup: {label}\n", encoding="utf-8")
            return
        if not marker.is_file():
            raise SetupError(
                f"{label} environment {venv} is not a usable Python {SUPPORTED_PYTHON} "
                "launcher-owned environment; refusing to overwrite it"
            )
        log.line(f"removing incomplete launcher-owned {label} environment at {venv}")
        shutil.rmtree(venv)
    marker.write_text(f"launcher_setup: {label}\n", encoding="utf-8")
    _ensure_python(root, log)
    _run([_which("uv"), "venv", "--python", SUPPORTED_PYTHON, str(venv)], log, cwd=root)
    version = _python_version(root, python, log)
    if version != SUPPORTED_PYTHON:
        raise SetupError(f"{label} environment has Python {version!r}, expected {SUPPORTED_PYTHON}")


def _ensure_client(root: Path, log: _Log) -> Path:
    venv = _path(root, ".venv")
    python = _path(root, CLIENT_PYTHON_REL)
    marker = _path(root, ".scratch/launcher/client-owned")
    lock = _path(root, "requirements.lock")
    if not lock.is_file():
        raise SetupError(f"pinned client lock is missing: {lock}")
    if venv.exists():
        if venv.is_symlink() or not venv.is_dir():
            raise SetupError(f"client venv path is not a directory: {venv}")
        if python.exists() and _python_version(root, python, log) != SUPPORTED_PYTHON:
            if not marker.is_file():
                raise SetupError(f"existing client interpreter is not Python {SUPPORTED_PYTHON}; "
                                 f"refusing to replace {venv}")
            shutil.rmtree(venv)
        elif not python.exists() and any(venv.iterdir()):
            if not marker.is_file():
                raise SetupError(f"client venv {venv} has no bin/python but is not empty; "
                                 "refusing to overwrite it")
            shutil.rmtree(venv)
        elif not python.exists():
            venv.rmdir()
    if not python.exists():
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("launcher_setup: client\n", encoding="utf-8")
        _ensure_python(root, log)
        _run([_which("uv"), "venv", "--python", SUPPORTED_PYTHON, str(venv)], log, cwd=root)
        if _python_version(root, python, log) != SUPPORTED_PYTHON:
            raise SetupError(f"uv created a client environment without Python {SUPPORTED_PYTHON}")
    key = {"requirements_sha256": _hash(lock), "python": SUPPORTED_PYTHON}
    if _receipt_matches(root, "client", key):
        try:
            _probe(root, python, _CLIENT_PROBE, "client", log)
            log.line(f"client environment reused at {python}")
            return _lexical(python)
        except SetupError as exc:
            log.line(f"client receipt did not validate ({exc}); resynchronizing")
    _run([_which("uv"), "pip", "sync", "--python", str(python), str(lock)], log, cwd=root)
    observed = _probe(root, python, _CLIENT_PROBE, "client", log)
    _record(root, "client", key, observed)
    log.line(f"client environment ready at {python}")
    return _lexical(python)


def _ensure_inputs(root: Path, log: _Log) -> None:
    manifest = _path(root, "setup/inputs.sha256")
    if not manifest.is_file():
        raise SetupError(f"pinned input manifest is missing: {manifest}")
    check = [_which("sha256sum"), "--check", "--strict", "--quiet", str(manifest)]
    verified = _run(check, log, cwd=root, check=False)
    if verified.returncode:
        log.line("pinned inputs need fetching")
        script = _path(root, "setup/fetch_inputs.sh")
        if not script.is_file():
            raise SetupError(f"pinned input fetch script is missing: {script}")
        _run([_which("bash"), str(script)], log, cwd=root)
        verified = _run(check, log, cwd=root, check=False)
        if verified.returncode:
            raise SetupError("pinned dataset/schema inputs still mismatch after fetch\n"
                             + _tail(verified.stdout + verified.stderr))
    log.line("pinned dataset/schema inputs verified")


def _validate_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict) or not isinstance(config.get("models"), dict):
        raise SetupError("provision requires the frozen resolved configuration document")
    dataset = config.get("dataset")
    if not isinstance(dataset, dict) or dataset.get("path") != "data/tweeteval":
        raise SetupError("the launcher supports only the pinned dataset path 'data/tweeteval'")
    for model_id, spec in config["models"].items():
        adapter = spec.get("adapter") if isinstance(spec, dict) else None
        if adapter not in _ADAPTERS:
            raise SetupError(f"model {model_id!r} has unsupported adapter {adapter!r}")
        if adapter == "vllm":
            safety = spec.get("engine_safety") or {}
            if spec.get("python") != VLLM_PYTHON_REL \
                    or safety.get("expected_python") != VLLM_PYTHON_REL:
                raise SetupError(
                    f"model {model_id!r} does not point at the frozen launcher vLLM "
                    f"interpreter {VLLM_PYTHON_REL}; launcher_plan.prepare must resolve it"
                )
        elif adapter == "qev":
            params = spec.get("params") or {}
            if params.get("python") != QEV_PYTHON_REL \
                    or params.get("onejev_path") != QEV_ONEJEV_REL:
                raise SetupError(
                    f"model {model_id!r} does not point at the frozen launcher QEV "
                    f"paths ({QEV_PYTHON_REL}, {QEV_ONEJEV_REL}); "
                    "launcher_plan.prepare must resolve them"
                )


def _ensure_vllm(root: Path, log: _Log) -> None:
    directory = _path(root, VLLM_ROOT_REL)
    venv = directory / "venv"
    python = _path(root, VLLM_PYTHON_REL)
    _owned_venv(root, directory, venv, python, "vLLM", log)
    requirements = _path(root, "setup/gpu/requirements.txt")
    installer = _path(root, "setup/gpu/install.sh")
    key = {"requirements_sha256": _hash(requirements), "install_sha256": _hash(installer),
           "python": SUPPORTED_PYTHON, "gpu_root": str(directory)}
    if _receipt_matches(root, "vllm", key):
        try:
            _pins(_probe(root, python, _VLLM_PROBE, "vLLM", log), VLLM_PINS, "vLLM")
            if not (python.parent / "hf").is_file():
                raise SetupError("reused vLLM environment has no hf CLI")
            log.line(f"pinned vLLM environment reused at {python}")
            return
        except SetupError as exc:
            log.line(f"vLLM receipt did not validate ({exc}); reinstalling")
    env = _env(root)
    env["GPU_ROOT"] = str(directory)
    log.line(f"installing pinned vLLM into {directory} via documented setup/gpu/install.sh")
    _run([_which("bash"), str(installer), "--install-vllm"], log, env=env, cwd=root)
    observed = _probe(root, python, _VLLM_PROBE, "vLLM", log)
    _pins(observed, VLLM_PINS, "vLLM")
    if not (python.parent / "hf").is_file():
        raise SetupError(f"vLLM environment does not provide executable hf at {python.parent / 'hf'}")
    _record(root, "vllm", key, observed)
    log.line(f"pinned vLLM environment ready at {python}")


def _ensure_qev_env(root: Path, log: _Log) -> None:
    directory = _path(root, QEV_ROOT_REL)
    venv = directory / "venv"
    python = _path(root, QEV_PYTHON_REL)
    _owned_venv(root, directory, venv, python, "QEV", log)
    requirements = _path(root, "qev_setup/launcher-requirements.txt")
    key = {"requirements_sha256": _hash(requirements), "python": SUPPORTED_PYTHON}
    if _receipt_matches(root, "qev-env", key):
        try:
            _pins(_probe(root, python, _QEV_DEPS_PROBE, "QEV dependencies", log), QEV_ENV_PINS, "QEV")
            log.line(f"pinned QEV dependency environment reused at {python}")
            return
        except SetupError as exc:
            log.line(f"QEV receipt did not validate ({exc}); reinstalling")
    log.line("installing exact QEV fallback dependencies (torch/torchvision +cu128 "
             "from the CUDA index; all remaining freeze pins from PyPI; "
             "causal-conv1d intentionally absent)")
    lines = [line.strip() for line in requirements.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    torch_pins = [line for line in lines
                  if line.startswith("torch==") or line.startswith("torchvision==")]
    other_pins = [line for line in lines if line not in torch_pins]
    if not any(line.startswith("torch==") for line in torch_pins) \
            or not any(line.startswith("torchvision==") for line in torch_pins):
        raise SetupError(f"{requirements} must pin torch and torchvision")
    _run([_which("uv"), "pip", "install", "--python", str(python), "--no-deps",
          "--index-url", TORCH_INDEX, *torch_pins], log, cwd=root)
    _run([_which("uv"), "pip", "install", "--python", str(python), "--no-deps",
          "--index-url", PYPI_INDEX, *other_pins], log, cwd=root)
    _run([_which("uv"), "pip", "check", "--python", str(python)], log, cwd=root)
    observed = _probe(root, python, _QEV_DEPS_PROBE, "QEV dependencies", log)
    _pins(observed, QEV_ENV_PINS, "QEV")
    _record(root, "qev-env", key, observed)
    log.line(f"pinned QEV dependency environment ready at {python}")


def _fetch_models(
    root: Path, models: Sequence[tuple[str, dict[str, Any]]], engines: Sequence[str], log: _Log
) -> None:
    client = _path(root, CLIENT_PYTHON_REL)
    helper = _path(root, "setup/gpu/model_io.py")
    script = _path(root, "setup/gpu/fetch_models.sh")
    if not client.exists():
        raise SetupError(f"client interpreter is missing before model fetch: {client}")
    for required in (helper, script):
        if not required.is_file():
            raise SetupError(f"pinned model setup helper is missing: {required}")
    hf = _path(root, VLLM_PYTHON_REL if engines[0] == "vllm" else QEV_PYTHON_REL).parent / "hf"
    if not hf.is_file() or not os.access(hf, os.X_OK):
        raise SetupError(f"selected engine environment has no executable hf CLI at {hf}")
    for model_id, spec in models:
        log.line(f"ensuring pinned {model_id}: {spec['repo']}@{spec['revision']} "
                 f"into {spec['local_snapshot']}")
        env = _env(root)
        env["PYTHON"] = str(client)
        env["ALLOW_LARGE_MODEL_DOWNLOAD"] = "YES"
        env["PATH"] = str(hf.parent) + os.pathsep + env.get("PATH", "")
        env.pop("HF_HUB_OFFLINE", None)
        env.pop("TRANSFORMERS_OFFLINE", None)
        result = _run([_which("bash"), str(script), model_id], log,
                      env=env, cwd=root, check=False)
        if result.returncode:
            raise SetupError(
                f"pinned model fetch for {model_id} was blocked or failed. The frozen "
                "selection is not silently skipped; resolve the VRAM/foreign-compute/disk "
                f"issue and resume. Output:\n{_tail(result.stdout + result.stderr)}"
            )
        confirmed = _run([client, str(helper), "verify", model_id], log,
                         cwd=root, check=False, quiet=True)
        if confirmed.returncode:
            raise SetupError(f"fetch reported success for {model_id}, but the pinned "
                             f"snapshot failed verification\n{_tail(confirmed.stdout + confirmed.stderr)}")
        log.line(f"model {model_id} verified")


def _git_blob(path: Path) -> str:
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise SetupError(f"cannot read git blob source {path}: {exc}") from None
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def _git(repo: Path, *args: str, log: _Log, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run([_which("git"), "-C", str(repo), *args], log, cwd=repo.parent,
                check=check, quiet=True)


def _onejev_ready(repo: Path, patch: Path, log: _Log) -> bool:
    head = _git(repo, "rev-parse", "HEAD", log=log, check=False)
    blob = _git(repo, "hash-object", str(patch), log=log, check=False)
    reverse = _git(repo, "apply", "--reverse", "--check", str(patch), log=log, check=False)
    return (head.returncode == 0 and head.stdout.strip() == ONEJEV_REVISION
            and blob.returncode == 0 and blob.stdout.strip() == ONEJEV_PATCH_BLOB
            and reverse.returncode == 0)


def _onejev_origin(repo: Path, log: _Log) -> str:
    result = _git(repo, "remote", "get-url", "origin", log=log, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def _prepare_onejev(root: Path, patch: Path, log: _Log) -> None:
    if _git_blob(patch) != ONEJEV_PATCH_BLOB:
        raise SetupError(f"serve.patch has git blob {_git_blob(patch)}, expected {ONEJEV_PATCH_BLOB}")
    directory = _path(root, QEV_ROOT_REL)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / ".launcher-onejev-owned"
    repo = directory / "OneJev"
    if repo.exists() and _onejev_ready(repo, patch, log):
        log.line(f"exact patched OneJev source reused at {repo}")
        return
    if repo.exists() and not (repo / ".git").exists():
        if repo.is_symlink() or not repo.is_dir():
            raise SetupError(f"OneJev path is not a directory: {repo}")
        if not marker.is_file() and any(repo.iterdir()):
            raise SetupError(
                f"{repo} is not an exact patched OneJev checkout and is not launcher-owned; "
                "refusing to reset it. Move it aside (or use a fresh launcher root) and retry."
            )
        shutil.rmtree(repo)
    if not repo.exists():
        marker.write_text("launcher_setup: OneJev\n", encoding="utf-8")
        log.line(f"cloning pinned OneJev into {repo}")
        _run([_which("git"), "clone", "--quiet", ONEJEV_REPOSITORY, str(repo)], log, cwd=root)
    elif not marker.is_file():
        raise SetupError(
            f"{repo} exists but is not the exact patched pinned source and is not "
            "launcher-owned; refusing to reset it. Move it aside (or use a fresh launcher "
            "root) and retry."
        )
    if "OmniJev/OneJev" not in _onejev_origin(repo, log):
        raise SetupError(f"{repo} origin is not OmniJev/OneJev; refusing to reset or use it")
    present = _git(repo, "cat-file", "-e", f"{ONEJEV_REVISION}^{{commit}}", log=log, check=False)
    if present.returncode:
        _git(repo, "fetch", "--quiet", "origin", ONEJEV_REVISION, log=log)
    _git(repo, "checkout", "--quiet", "-f", ONEJEV_REVISION, log=log)
    _git(repo, "clean", "-fdq", log=log)
    _git(repo, "apply", "--check", str(patch), log=log)
    _git(repo, "apply", str(patch), log=log)
    if not _onejev_ready(repo, patch, log):
        raise SetupError("OneJev did not validate after applying the pinned serve.patch")
    log.line(f"OneJev checked out at {ONEJEV_REVISION} and patched ({ONEJEV_PATCH_BLOB})")


def _ensure_qev_source(root: Path, models: Sequence[tuple[str, dict[str, Any]]], log: _Log) -> None:
    if not models:
        raise SetupError("QEV engine selected without a selected QEV model")
    model_id, spec = models[0]
    patch = _path(root, spec["local_snapshot"]) / "serve.patch"
    if not patch.is_file() or _git_blob(patch) != ONEJEV_PATCH_BLOB:
        raise SetupError(f"verified QEV snapshot {model_id} has no valid pinned serve.patch")
    _prepare_onejev(root, patch, log)
    python = _path(root, QEV_PYTHON_REL)
    key = {"onejev_revision": ONEJEV_REVISION, "patch_blob": ONEJEV_PATCH_BLOB,
           "requirements_sha256": _hash(_path(root, "qev_setup/launcher-requirements.txt")),
           "python": SUPPORTED_PYTHON}
    if _receipt_matches(root, "qev-source", key):
        try:
            _pins(_probe(root, python, _QEV_SOURCE_PROBE, "QEV source", log), QEV_SOURCE_PINS, "QEV source")
            log.line(f"editable patched QEV source reused at {_path(root, QEV_ONEJEV_REL)}")
            return
        except SetupError as exc:
            log.line(f"QEV source receipt did not validate ({exc}); reinstalling editable source")
    _run([_which("uv"), "pip", "install", "--python", str(python), "--no-deps",
          "--no-build-isolation", "-e", str(_path(root, QEV_ONEJEV_REL))], log, cwd=root)
    _run([_which("uv"), "pip", "check", "--python", str(python)], log, cwd=root)
    observed = _probe(root, python, _QEV_SOURCE_PROBE, "QEV source", log)
    _pins(observed, QEV_SOURCE_PINS, "QEV source")
    _record(root, "qev-source", key, observed)
    log.line(f"exact patched QEV source ready at {_path(root, QEV_ONEJEV_REL)}")


def provision(config: dict[str, Any], emit: Callable[[str], None]) -> Path:
    """Install/reuse the client and selected engines; return lexical client python."""
    if not callable(emit):
        raise TypeError("emit must be callable")
    root = Path(ROOT)
    log = _Log(_path(root, ".scratch/launcher/provision.log"), emit)
    try:
        _validate_config(config)
        _which("uv")
        engines = tuple(name for name in _ENGINES
                        if any(spec["adapter"] == name for spec in config["models"].values()))
        log.line(f"launcher provisioning start (repo={root}); engines: "
                 f"{', '.join(engines) if engines else 'none (classical only)'}")
        _ensure_inputs(root, log)
        client = _ensure_client(root, log)
        if "vllm" in engines:
            _ensure_vllm(root, log)
        if "qev" in engines:
            _ensure_qev_env(root, log)
        gpu_models = [(model_id, spec) for model_id, spec in config["models"].items()
                      if spec["adapter"] in {"vllm", "qev"}] if engines else []
        if gpu_models:
            _fetch_models(root, gpu_models, engines, log)
        if "qev" in engines:
            _ensure_qev_source(root, [item for item in gpu_models
                                      if item[1]["adapter"] == "qev"], log)
        log.line(f"launcher provisioning complete; client python: {_lexical(client)}")
        return _lexical(client)
    except SetupError as exc:
        log.line(f"launcher provisioning failed: {exc}")
        raise
    except Exception as exc:
        log.line(f"launcher provisioning failed unexpectedly: {type(exc).__name__}: {exc}")
        raise SetupError(f"launcher provisioning failed unexpectedly: {type(exc).__name__}: {exc}") from exc
    finally:
        log.close()
