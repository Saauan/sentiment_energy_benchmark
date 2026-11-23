"""Lazy, offline adapter for Baseten decision models served by patched OneJev QEV.

Only batch size one is advertised: QEV batches question branches from one state,
not independent tweets. Model loading and all QEV/Transformers imports happen in
a dedicated worker during :meth:`QEVLocalAdapter.setup`, never at module import
or adapter construction.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

from .base import Unavailable

LABELS = ("negative", "neutral", "positive")
ONEJEV_REVISION = "81ce62f1597c91e46767d4d02d6ac2e18534fe94"
PATCH_BLOB = "7a432e8d254a9838b04ca3a155547ec3b3c70e78"
SETUP_BLOB = "1d00a2676f8f1da5d25295bf727a85491ff95d51"
READOUT_BLOB = "444b7bf8313d43415e97988a79740a93154fdd3b"
WORKER_PATH = Path(__file__).resolve().parent.parent / "qev_setup" / "qev_worker.py"
_MAX_MESSAGE_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class ModelSetup:
    revision: str
    config_blob: str
    weights: dict[str, int]


# Exact model-specific mappings from the Hugging Face API at the pinned refs.
MODEL_SETUP = {
    "baselabs/joey-1": ModelSetup(
        revision="db5c2a631f2da93c4173378438d8246966e25159",
        config_blob="5878a46a4364f966cc6e877a4f3f3f80d6230be9",
        weights={"model.safetensors": 5_443_677_680},
    ),
    "baselabs/elephant-1": ModelSetup(
        revision="2e80345f1e41b2c1eaff8ae2538c0ac657210c44",
        config_blob="98295053fab64c0a71b1c2bbba06c2e31ed33cfb",
        weights={"model.safetensors": 18_819_722_392},
    ),
    "baselabs/mammoth-1": ModelSetup(
        revision="858aab610d4d124315ebf72d2f36c5e7341f758f",
        config_blob="7c80d74dfe54ed2de25007a979c06743dbc29518",
        weights={
            "model-00001-of-00002.safetensors": 49_825_162_976,
            "model-00002-of-00002.safetensors": 4_888_445_168,
        },
    ),
}

_ENGINE_KEYS = {
    "device",
    "dtype",
    "head_dtype",
    "fork_mode",
    "fork_tolerance",
    "fork_token_budget",
    "max_branch_tokens",
    "max_request_tokens",
}
_CONTROL_KEYS = {
    "python",
    "onejev_path",
    "debias",
    "debug",
    "init_timeout_s",
    "request_timeout_s",
    "close_timeout_s",
}


def _git_blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"params[{name!r}] must be a positive number")
    return float(value)


class QEVLocalAdapter:
    supported_batches = (1,)

    def __init__(self, spec: dict[str, Any], attempt: Path, setup: ModelSetup) -> None:
        self.repo = spec["repo"]
        self.revision = spec["revision"]
        self.model_setup = setup
        self.local_snapshot = spec.get("local_snapshot")
        self.attempt = Path(attempt)
        self.params = dict(spec.get("params") or {})
        unknown = set(self.params) - _ENGINE_KEYS - _CONTROL_KEYS
        if unknown:
            raise ValueError(f"unknown QEV params: {sorted(unknown)}")
        self.debias = self.params.get("debias", 1)
        if isinstance(self.debias, bool) or not isinstance(self.debias, int) or self.debias < 1:
            raise ValueError("params['debias'] must be an integer >= 1")
        self.debug = self.params.get("debug", False)
        if not isinstance(self.debug, bool):
            raise ValueError("params['debug'] must be boolean")
        self.init_timeout_s = _positive_number(self.params.get("init_timeout_s", 1800), "init_timeout_s")
        self.request_timeout_s = _positive_number(self.params.get("request_timeout_s", 300), "request_timeout_s")
        self.close_timeout_s = _positive_number(self.params.get("close_timeout_s", 10), "close_timeout_s")

        self._process: subprocess.Popen[bytes] | None = None
        self._request_fd: int | None = None
        self._response_fd: int | None = None
        self._response_buffer = b""
        self._log = None
        self._lock = threading.RLock()
        self._closed = False
        self._atexit_registered = False

    def _engine_options(self) -> dict[str, Any]:
        options = {key: self.params[key] for key in _ENGINE_KEYS if key in self.params}
        options.setdefault("device", "cuda:0")
        options.setdefault("dtype", "bfloat16")
        options.setdefault("head_dtype", "float32")
        options.setdefault("fork_mode", "auto")
        return options

    def _validate_snapshot(self) -> tuple[Path, dict[str, Any]]:
        if not self.local_snapshot:
            raise Unavailable(
                "QEV requires spec['local_snapshot']; Hub loading would ignore the checkpoint's "
                "block-bidirectional attention and slot_lm overrides"
            )
        path = Path(self.local_snapshot).expanduser().resolve()
        if not path.is_dir():
            raise Unavailable(f"QEV local snapshot is not a directory: {path}")
        expected_blobs = {
            "config.json": self.model_setup.config_blob,
            "readout.json": READOUT_BLOB,
            "serve.patch": PATCH_BLOB,
            "setup.sh": SETUP_BLOB,
        }
        actual_blobs: dict[str, str] = {}
        for name, expected in expected_blobs.items():
            file_path = path / name
            try:
                actual = _git_blob(file_path)
            except OSError as exc:
                raise Unavailable(f"QEV snapshot is missing {name}: {exc}") from exc
            if actual != expected:
                raise Unavailable(
                    f"QEV snapshot {name} has git blob {actual}, expected {expected} at {self.revision}"
                )
            actual_blobs[name] = actual
        for name, expected_size in self.model_setup.weights.items():
            try:
                actual_size = (path / name).stat().st_size
            except OSError as exc:
                raise Unavailable(f"QEV snapshot is missing weights {name}: {exc}") from exc
            if actual_size != expected_size:
                raise Unavailable(
                    f"QEV weights {name} are incomplete: {actual_size} bytes, expected {expected_size}"
                )

        try:
            config = json.loads((path / "config.json").read_text(encoding="utf-8"))
            readout = json.loads((path / "readout.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise Unavailable(f"cannot read QEV checkpoint metadata: {exc}") from exc
        overrides = config.get("modeling_overrides", {})
        attention = overrides.get("attention", {})
        expected_attention = {"full_attention": "block_bidirectional", "linear_attention": "causal"}
        if attention != expected_attention or overrides.get("readout") != "slot_lm":
            raise Unavailable(f"unsupported QEV modeling_overrides: {overrides!r}")
        if readout.get("readout") != "slot_lm":
            raise Unavailable(f"unsupported QEV readout.json: {readout!r}")
        evidence = {
            "local_snapshot": str(path),
            "file_git_blobs": actual_blobs,
            "weight_bytes": dict(self.model_setup.weights),
            "modeling_overrides": overrides,
        }
        return path, evidence

    def setup(self, train, validation) -> dict[str, Any]:
        """Load the pinned local checkpoint and verify the reference decision runtime.

        No training, validation adaptation, threshold fitting, or generative decoding is
        performed. The supplied splits are intentionally unused.
        """
        del train, validation
        with self._lock:
            if self._closed:
                raise Unavailable("QEV adapter is already closed")
            if self._process is not None:
                raise RuntimeError("QEV setup may only be called once")
            try:
                snapshot, snapshot_evidence = self._validate_snapshot()
                runtime_value = self.params.get("onejev_path") or os.environ.get("QEV_ONEJEV_PATH")
                if not runtime_value:
                    raise Unavailable("set params['onejev_path'] (or QEV_ONEJEV_PATH) to patched OneJev")
                runtime = Path(runtime_value).expanduser().resolve()
                if not runtime.is_dir():
                    raise Unavailable(f"OneJev runtime is not a directory: {runtime}")
                if not WORKER_PATH.is_file():
                    raise Unavailable(f"QEV worker is missing: {WORKER_PATH}")
                self._spawn(runtime)
                response = self._rpc(
                    "init",
                    {
                        "model_path": str(snapshot),
                        "onejev_path": str(runtime),
                        "onejev_revision": ONEJEV_REVISION,
                        "patch_path": str(snapshot / "serve.patch"),
                        "patch_blob": PATCH_BLOB,
                        "engine_options": self._engine_options(),
                    },
                    self.init_timeout_s,
                )
                metadata = {
                    "adapter": "qev",
                    "repo": self.repo,
                    "revision": self.revision,
                    "onejev_revision": ONEJEV_REVISION,
                    "serve_patch_blob": PATCH_BLOB,
                    "supported_batches": list(self.supported_batches),
                    "independent_tweets_per_request": 1,
                    "debias": self.debias,
                    "debug": self.debug,
                    "engine_options": self._engine_options(),
                    "snapshot": snapshot_evidence,
                    "runtime": response,
                    "train_validation_use": "none; fixed structured sentiment choice",
                }
                # This is also a hard JSON-serializability check for run manifests.
                json.dumps(metadata, allow_nan=False)
                return metadata
            except Unavailable:
                self.close()
                raise
            except Exception as exc:
                self.close()
                raise Unavailable(f"QEV setup failed: {type(exc).__name__}: {exc}") from exc

    def _spawn(self, runtime: Path) -> None:
        self.attempt.mkdir(parents=True, exist_ok=True)
        python = str(self.params.get("python") or sys.executable)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(runtime) + os.pathsep + env.get("PYTHONPATH", "")
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_DATASETS_OFFLINE"] = "1"
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["PYTHONUNBUFFERED"] = "1"
        self._log = (self.attempt / "qev_worker.log").open("ab")
        request_read, request_write = os.pipe()
        response_read, response_write = os.pipe()
        try:
            self._process = subprocess.Popen(
                [python, str(WORKER_PATH), "--request-fd", str(request_read), "--response-fd", str(response_write)],
                stdin=subprocess.DEVNULL,
                stdout=self._log,
                stderr=subprocess.STDOUT,
                pass_fds=(request_read, response_write),
                start_new_session=True,
                env=env,
            )
        except Exception:
            for fd in (request_read, request_write, response_read, response_write):
                try:
                    os.close(fd)
                except OSError:
                    pass
            if self._log is not None:
                self._log.close()
                self._log = None
            raise
        os.close(request_read)
        os.close(response_write)
        self._request_fd = request_write
        self._response_fd = response_read
        atexit.register(self.close)
        self._atexit_registered = True

    def _read_message(self, timeout: float) -> dict[str, Any]:
        if self._response_fd is None:
            raise RuntimeError("QEV response pipe is closed")
        deadline = time.monotonic() + timeout
        while b"\n" not in self._response_buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"QEV worker did not answer within {timeout:.1f}s")
            readable, _, _ = select.select([self._response_fd], [], [], remaining)
            if not readable:
                raise TimeoutError(f"QEV worker did not answer within {timeout:.1f}s")
            chunk = os.read(self._response_fd, 65_536)
            if not chunk:
                raise RuntimeError("QEV worker exited or closed its response pipe; see qev_worker.log")
            self._response_buffer += chunk
            if len(self._response_buffer) > _MAX_MESSAGE_BYTES:
                raise RuntimeError("QEV worker response exceeds 32 MiB")
        line, self._response_buffer = self._response_buffer.split(b"\n", 1)
        response = json.loads(line)
        if not isinstance(response, dict):
            raise RuntimeError("invalid QEV worker response")
        return response

    def _rpc(self, operation: str, payload: dict[str, Any] | None = None, timeout: float | None = None) -> Any:
        with self._lock:
            if self._process is None or self._request_fd is None or self._process.poll() is not None:
                raise RuntimeError("QEV worker is not running")
            message = {"operation": operation, "payload": payload or {}}
            data = (json.dumps(message, allow_nan=False) + "\n").encode("utf-8")
            view = memoryview(data)
            while view:
                try:
                    written = os.write(self._request_fd, view)
                except InterruptedError:
                    continue
                view = view[written:]
            response = self._read_message(self.request_timeout_s if timeout is None else timeout)
            if not response.get("ok"):
                error = response.get("error") or {}
                kind = error.get("type", "RuntimeError")
                text = error.get("message", "unknown worker error")
                if kind in {"OutOfMemoryError", "MemoryError"} or "out of memory" in text.lower():
                    raise Unavailable(f"QEV memory precondition failed: {kind}: {text}")
                raise RuntimeError(f"QEV worker {kind}: {text}")
            return response.get("result")

    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        if not texts:
            return []
        if len(texts) != 1:
            raise Unavailable("QEV supports only one independent tweet per request; batch 16 is unverified")
        if not isinstance(texts[0], str):
            raise TypeError("QEV predict expects strings")
        started = time.perf_counter()
        try:
            result = self._rpc(
                "predict",
                {"text": texts[0], "debias": self.debias, "debug": self.debug},
                self.request_timeout_s,
            )
            answer = result["answer"]
            choice = answer.get("choice")
            label = choice if choice in LABELS else None
            output = {"label": label, "raw": result, "latency_s": time.perf_counter() - started}
            json.dumps(output, allow_nan=False)
            return [output]
        except Unavailable:
            self.close()
            raise
        except Exception:
            # A failed or timed-out request must not leave a GPU worker behind.
            self.close()
            raise

    def _close_fds(self) -> None:
        for attr in ("_request_fd", "_response_fd"):
            fd = getattr(self, attr)
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
                setattr(self, attr, None)
        if self._log is not None:
            self._log.close()
            self._log = None

    def _stop_process(self) -> None:
        process = self._process
        if process is not None:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=self.close_timeout_s)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=self.close_timeout_s)
            else:
                process.wait(timeout=0)
        self._process = None
        self._close_fds()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._process is not None:
                try:
                    self._rpc("close", timeout=self.close_timeout_s)
                except Exception:
                    pass
                self._stop_process()
            else:
                self._close_fds()
            if self._atexit_registered:
                try:
                    atexit.unregister(self.close)
                except Exception:
                    pass
                self._atexit_registered = False


def create(spec: dict[str, Any], attempt: Path) -> QEVLocalAdapter:
    """Construct lazily; no snapshot access, runtime imports, or process start occurs here."""
    if spec.get("adapter") != "qev":
        raise ValueError("qev_local.create requires spec['adapter'] == 'qev'")
    repo = spec.get("repo")
    revision = spec.get("revision")
    setup = MODEL_SETUP.get(repo)
    if setup is None:
        raise Unavailable(f"unsupported QEV decision model: {repo!r}")
    if revision != setup.revision:
        raise Unavailable(f"{repo} is pinned to revision {setup.revision}, got {revision!r}")
    return QEVLocalAdapter(spec, Path(attempt), setup)
