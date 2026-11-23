"""Lifecycle-owning, loopback vLLM OpenAI-server adapter.

Construction is side-effect free.  ``setup()`` performs the explicit version
check and starts a child process; ``close()`` terminates only that child's
process group.  The client uses the standard library so the client environment
need not contain vLLM, OpenAI, or Inspect.
"""

from __future__ import annotations

import atexit
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from .base import Unavailable


LABELS = ("negative", "neutral", "positive")
ENGINE_VERSION = "0.19.0"
PROMPT = (
    "Classify the sentiment of this tweet. Reply with exactly one word: "
    "negative, neutral, or positive.\n\nTweet: "
)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_PYTHON = (
    _PROJECT_ROOT.parent
    / "inspect_pilot_study"
    / ".scratch"
    / "gpu"
    / "venv"
    / "bin"
    / "python"
)
_OPENER = build_opener(ProxyHandler({}))
_MEMORY_MARKERS = (
    "out of memory",
    "cuda oom",
    "insufficient memory",
    "no available memory",
    "not enough memory",
    "free memory on device",
)


def _jsonable(value: Any) -> Any:
    """Return a defensive JSON value, rejecting non-finite/custom objects."""
    return json.loads(json.dumps(value, allow_nan=False))


def _parse_label(content: str) -> str | None:
    label = content.strip().lower()
    return label if label in LABELS else None


def _path(value: os.PathLike[str] | str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (_PROJECT_ROOT / path).resolve()


def _interpreter_path(value: os.PathLike[str] | str) -> Path:
    """Normalize the invoked path without discarding a venv's Python symlink."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return Path(os.path.abspath(path))


def create(spec: dict, attempt: Path) -> "VLLMLocalAdapter":
    """Create an unstarted adapter; all process and filesystem work is deferred."""
    return VLLMLocalAdapter(spec, Path(attempt))


class VLLMLocalAdapter:
    supported_batches = (1, 16)

    def __init__(self, spec: dict, attempt: Path) -> None:
        if not isinstance(spec, dict):
            raise TypeError("vLLM spec must be a dictionary")
        self.spec = spec
        self.attempt = attempt
        self.params = dict(spec.get("params") or {})
        self.safety = dict(spec.get("engine_safety") or {})
        self._process: subprocess.Popen[bytes] | None = None
        self._server_output = None
        self._base_url: str | None = None
        self._ready = False
        self._closed = False
        self._lock = threading.RLock()
        self._request_policy: dict[str, Any] = {}
        self._served_name = "sentiment-vllm"
        self._pid: int | None = None
        self._atexit_registered = False

    def _resolve_python(self) -> Path:
        configured = self.spec.get("python")
        if configured:
            python = _interpreter_path(configured)
        elif self.spec.get("venv"):
            venv = _interpreter_path(self.spec["venv"])
            python = venv / "bin" / "python"
        else:
            python = _interpreter_path(_DEFAULT_PYTHON)
        if not python.is_file() or not os.access(python, os.X_OK):
            raise Unavailable(f"Pinned vLLM Python is not executable: {python}")
        expected = self.safety.get("expected_python")
        if expected and python != _interpreter_path(expected):
            raise Unavailable(
                f"vLLM Python {python} does not match expected_python "
                f"{_interpreter_path(expected)}"
            )
        return python

    def _check_engine_version(self, python: Path) -> None:
        # importlib.metadata does not import vLLM or initialise CUDA.  Hide GPUs
        # anyway so this secondary guard cannot accidentally touch a device.
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        try:
            result = subprocess.run(
                [
                    str(python),
                    "-c",
                    "import importlib.metadata; print(importlib.metadata.version('vllm'))",
                ],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=float(self.safety.get("version_timeout_s", 15)),
                check=False,
                env=env,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise Unavailable(f"Could not verify vLLM {ENGINE_VERSION}: {exc}") from exc
        version = result.stdout.strip()
        if result.returncode != 0 or version != ENGINE_VERSION:
            detail = result.stderr.strip()[-500:]
            raise Unavailable(
                f"Expected vLLM=={ENGINE_VERSION}, observed {version!r} "
                f"(exit {result.returncode}){': ' + detail if detail else ''}"
            )

    def _validate_configuration(self) -> None:
        dtype = str(self.params.get("dtype", "bfloat16")).lower()
        if dtype not in {"bfloat16", "bf16"}:
            raise Unavailable(f"vLLM condition requires BF16, not dtype={dtype!r}")
        quantization = self.params.get("quantization")
        if quantization not in (None, "", "none", "None"):
            raise Unavailable(f"Quantization is forbidden for this condition: {quantization!r}")
        if int(self.params.get("max_num_seqs", 16)) != 16:
            raise Unavailable("The closed-loop width-16 condition requires max_num_seqs=16")
        if int(self.params.get("max_tokens", 8)) < 1:
            raise Unavailable("max_tokens must be positive")
        if int(self.params.get("max_model_len", 1024)) < 1:
            raise Unavailable("max_model_len must be positive")
        memory = float(self.params.get("gpu_memory_utilization", 0.85))
        if not 0 < memory <= 1:
            raise Unavailable("gpu_memory_utilization must be in (0, 1]")
        if self.params.get("host", "127.0.0.1") != "127.0.0.1":
            raise Unavailable("The adapter only permits a 127.0.0.1 loopback server")
        if float(self.params.get("temperature", 0)) != 0:
            raise Unavailable("The primary condition requires temperature=0")
        if float(self.params.get("top_p", 1)) != 1:
            raise Unavailable("The primary condition requires top_p=1")
        if self.params.get("enable_thinking", False) is not False:
            raise Unavailable("The primary condition requires enable_thinking=false")
        snapshot = self.spec.get("local_snapshot")
        if snapshot and not _path(snapshot).is_dir():
            raise Unavailable(f"local_snapshot is not a directory: {_path(snapshot)}")
        if not snapshot and not self.spec.get("repo"):
            raise Unavailable("vLLM spec requires repo or local_snapshot")

    def _environment(self) -> dict[str, str]:
        cache = _path(self.spec.get("cache_dir", self.attempt / "cache"))
        directories = {
            "HF_HOME": cache / "hf",
            "XDG_CACHE_HOME": cache / "xdg",
            "TRITON_CACHE_DIR": cache / "triton",
            "VLLM_CACHE_ROOT": cache / "vllm",
        }
        for directory in directories.values():
            directory.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({key: str(value) for key, value in directories.items()})
        # Do not inherit client-thread limits into a server unintentionally.
        env["TOKENIZERS_PARALLELISM"] = "false"
        if self.spec.get("local_snapshot"):
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
        cuda_devices = self.safety.get("cuda_visible_devices")
        if cuda_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)
        return env

    def _command(self, python: Path, port: int) -> list[str]:
        snapshot = self.spec.get("local_snapshot")
        model = str(_path(snapshot)) if snapshot else str(self.spec["repo"])
        served = str(self.params.get("served_model_name", "sentiment-vllm"))
        self._served_name = served
        command = [
            str(python),
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--served-model-name",
            served,
            "--dtype",
            "bfloat16",
            "--max-model-len",
            str(int(self.params.get("max_model_len", 1024))),
            "--gpu-memory-utilization",
            str(float(self.params.get("gpu_memory_utilization", 0.85))),
            "--max-num-seqs",
            "16",
            "--seed",
            str(int(self.params.get("seed", 0))),
            "--generation-config",
            "vllm",
            "--no-enable-prefix-caching",
        ]
        if not snapshot:
            revision = self.spec.get("revision")
            if not revision:
                raise Unavailable("Hub-served vLLM models require an explicit revision")
            command.extend(["--revision", str(revision)])
        tensor_parallel = int(self.params.get("tensor_parallel_size", 1))
        if tensor_parallel < 1:
            raise Unavailable("tensor_parallel_size must be positive")
        command.extend(["--tensor-parallel-size", str(tensor_parallel)])
        if self.params.get("enforce_eager", False):
            command.append("--enforce-eager")
        return command

    @staticmethod
    def _free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def _log(self, event: str, **fields: Any) -> None:
        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
        line = json.dumps(_jsonable(record), sort_keys=True) + "\n"
        with (self.attempt / "vllm-adapter.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(line)

    @staticmethod
    def _open(request: Request, timeout: float):
        return _OPENER.open(request, timeout=timeout)

    def _get_json(self, path: str, timeout: float = 5) -> Any:
        assert self._base_url is not None
        request = Request(self._base_url + path, method="GET")
        with self._open(request, timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _server_log_tail(self, characters: int = 4000) -> str:
        path = self.attempt / "vllm-server.log"
        try:
            with path.open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                size = stream.tell()
                stream.seek(max(0, size - characters))
                return stream.read().decode("utf-8", errors="replace")
        except OSError:
            return ""

    def _startup_error(self, prefix: str) -> Exception:
        tail = self._server_log_tail()
        message = f"{prefix}\n{vllm_log}" if (vllm_log := tail.strip()) else prefix
        if any(marker in tail.lower() for marker in _MEMORY_MARKERS):
            return Unavailable(message)
        return RuntimeError(message)

    def _wait_for_server(self, timeout: float) -> None:
        assert self._process is not None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            returncode = self._process.poll()
            if returncode is not None:
                raise self._startup_error(f"vLLM exited during startup with status {returncode}")
            try:
                assert self._base_url is not None
                request = Request(self._base_url + "/health", method="GET")
                with self._open(
                    request, timeout=min(2, max(0.1, deadline - time.monotonic()))
                ) as response:
                    response.read()
                # Recheck after a response so an unrelated listener can never be
                # adopted when the owned child has just exited.
                if self._process.poll() is None:
                    return
            except (HTTPError, URLError, TimeoutError, OSError, ValueError):
                pass
            time.sleep(min(0.25, max(0, deadline - time.monotonic())))
        raise self._startup_error(f"vLLM did not become ready within {timeout:g}s")

    def _check_served_model(self) -> None:
        payload = self._get_json("/v1/models")
        models = payload.get("data", []) if isinstance(payload, dict) else []
        ids = {item.get("id") for item in models if isinstance(item, dict)}
        if self._served_name not in ids:
            raise RuntimeError(
                f"vLLM did not advertise expected model {self._served_name!r}; observed {sorted(ids)!r}"
            )

    def setup(self, train, validation) -> dict[str, Any]:
        """Verify, launch and await readiness; no inference warmup is done here."""
        with self._lock:
            if self._ready:
                raise RuntimeError("vLLM adapter setup() may only be called once")
            if self._closed:
                raise RuntimeError("vLLM adapter is closed")
            self.attempt.mkdir(parents=True, exist_ok=True)
            try:
                self._validate_configuration()
                python = self._resolve_python()
                self._check_engine_version(python)
                configured_port = int(self.params.get("port", 0))
                port = self._free_port() if configured_port == 0 else configured_port
                if not 1 <= port <= 65535:
                    raise Unavailable(f"Invalid loopback port: {port}")
                command = self._command(python, port)
                env = self._environment()
                self._base_url = f"http://127.0.0.1:{port}"
                self._request_policy = {
                    "endpoint": "/v1/chat/completions",
                    "independent_request_threads": 16,
                    "closed_loop": True,
                    "retries": 0,
                    "timeout_s": float(self.safety.get("request_timeout_s", 120)),
                    "max_tokens": int(self.params.get("max_tokens", 8)),
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "seed": int(self.params.get("seed", 0)),
                    "stream": False,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "prompt": str(self.params.get("prompt", PROMPT)),
                    "strict_labels": list(LABELS),
                }
                self._log(
                    "server_start",
                    command=command,
                    engine_version=ENGINE_VERSION,
                    cache_dir=str(_path(self.spec.get("cache_dir", self.attempt / "cache"))),
                    cuda_visible_devices=env.get("CUDA_VISIBLE_DEVICES"),
                    request_policy=self._request_policy,
                )
                self._server_output = (self.attempt / "vllm-server.log").open("ab", buffering=0)
                try:
                    self._process = subprocess.Popen(
                        command,
                        stdin=subprocess.DEVNULL,
                        stdout=self._server_output,
                        stderr=subprocess.STDOUT,
                        env=env,
                        start_new_session=True,
                        cwd=str(self.attempt),
                    )
                except OSError as exc:
                    raise Unavailable(f"Could not start pinned vLLM server: {exc}") from exc
                self._pid = self._process.pid
                atexit.register(self.close)
                self._atexit_registered = True
                self._wait_for_server(float(self.safety.get("startup_timeout_s", 900)))
                self._check_served_model()
                self._ready = True
                metadata = {
                    "adapter": "vllm",
                    "engine": "vLLM",
                    "engine_version": ENGINE_VERSION,
                    "repo": self.spec.get("repo"),
                    "revision": self.spec.get("revision"),
                    "local_snapshot": str(_path(self.spec["local_snapshot"]))
                    if self.spec.get("local_snapshot")
                    else None,
                    "python": str(python),
                    "python_realpath": str(python.resolve()),
                    "server_pid": self._pid,
                    "base_url": self._base_url,
                    "served_model_name": self._served_name,
                    "dtype": "bfloat16",
                    "quantization": None,
                    "max_model_len": int(self.params.get("max_model_len", 1024)),
                    "max_num_seqs": 16,
                    "gpu_memory_utilization": float(
                        self.params.get("gpu_memory_utilization", 0.85)
                    ),
                    "tensor_parallel_size": int(self.params.get("tensor_parallel_size", 1)),
                    "enforce_eager": bool(self.params.get("enforce_eager", False)),
                    "prefix_caching": False,
                    "cache_dir": str(_path(self.spec.get("cache_dir", self.attempt / "cache"))),
                    "server_log": str(self.attempt / "vllm-server.log"),
                    "adapter_log": str(self.attempt / "vllm-adapter.jsonl"),
                    "request_policy": self._request_policy,
                    "train_examples": len(train),
                    "validation_examples": len(validation),
                    "warmup_performed_by_adapter": False,
                }
                self._log("server_ready", metadata=metadata)
                return _jsonable(metadata)
            except Exception as exc:
                self._log("setup_failed", error=f"{type(exc).__name__}: {exc}")
                self.close()
                raise

    def _predict_one(self, text: str) -> dict[str, Any]:
        if not isinstance(text, str):
            raise TypeError("vLLM predict texts must be strings")
        assert self._base_url is not None
        started = time.perf_counter()
        body = {
            "model": self._served_name,
            "messages": [
                {"role": "user", "content": self._request_policy["prompt"] + text}
            ],
            "max_tokens": self._request_policy["max_tokens"],
            "temperature": self._request_policy["temperature"],
            "top_p": self._request_policy["top_p"],
            "seed": self._request_policy["seed"],
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request = Request(
            self._base_url + "/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Connection": "close",
            },
            method="POST",
        )
        try:
            with self._open(request, self._request_policy["timeout_s"]) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read(1000).decode("utf-8", errors="replace")
            raise RuntimeError(f"vLLM HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"vLLM request failed without retry: {exc}") from exc
        try:
            choice = payload["choices"][0]
            content = choice["message"].get("content") or ""
            if not isinstance(content, str):
                raise TypeError("message.content is not a string")
            raw = {
                "content": content,
                "finish_reason": choice.get("finish_reason"),
                "response_id": payload.get("id"),
                "response_model": payload.get("model"),
                "usage": payload.get("usage"),
            }
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Malformed vLLM chat response: {exc}") from exc
        return {
            "label": _parse_label(content),
            "raw": _jsonable(raw),
            "latency_s": time.perf_counter() - started,
        }

    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        """Submit independent requests in closed-loop groups of at most 16."""
        with self._lock:
            if not self._ready or self._closed or self._process is None:
                raise RuntimeError("vLLM adapter is not set up")
            if not texts:
                return []
            group_sizes = [len(texts[start : start + 16]) for start in range(0, len(texts), 16)]
            self._log("predict", tweets=len(texts), group_sizes=group_sizes)
            results: list[dict[str, Any]] = []
            for start in range(0, len(texts), 16):
                group = texts[start : start + 16]
                # A fresh executor per closed-loop group makes the barrier
                # explicit. executor.map preserves input order.
                with ThreadPoolExecutor(
                    max_workers=len(group), thread_name_prefix="vllm-request"
                ) as executor:
                    results.extend(executor.map(self._predict_one, group))
            return results

    def close(self) -> None:
        """Idempotently terminate only the owned server process group."""
        with self._lock:
            self._ready = False
            process = self._process
            self._process = None
            try:
                if process is not None:
                    try:
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGTERM)
                            try:
                                process.wait(timeout=float(self.safety.get("shutdown_timeout_s", 20)))
                            except subprocess.TimeoutExpired:
                                os.killpg(process.pid, signal.SIGKILL)
                                process.wait(timeout=10)
                    except ProcessLookupError:
                        pass
                    finally:
                        self._log(
                            "server_stopped",
                            pid=process.pid,
                            returncode=process.poll(),
                        )
            finally:
                if self._server_output is not None:
                    self._server_output.close()
                    self._server_output = None
                self._closed = True
                self._base_url = None
                if self._atexit_registered:
                    try:
                        atexit.unregister(self.close)
                    except Exception:
                        pass
                    self._atexit_registered = False
