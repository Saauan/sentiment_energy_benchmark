"""CPU-only tests; no vLLM import, model load, GPU, or network egress."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from types import SimpleNamespace
import pytest

from adapters.base import Unavailable
import adapters.vllm_local as module


class FakeProcess:
    def __init__(self, pid=43210, returncode=None):
        self.pid = pid
        self.returncode = returncode
        self.wait_calls = []

    def poll(self):
        return self.returncode

    def wait(self, timeout):
        self.wait_calls.append(timeout)
        if self.returncode is None:
            self.returncode = -signal.SIGTERM
        return self.returncode


def executable(tmp_path: Path) -> Path:
    path = tmp_path / "pinned-python"
    path.write_text("#!/bin/sh\nexit 1\n")
    path.chmod(0o755)
    return path


def spec_for(python: Path, **changes):
    spec = {
        "adapter": "vllm",
        "repo": "Qwen/Qwen3.5-2B",
        "revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
        "python": str(python),
        "params": {},
        "engine_safety": {"startup_timeout_s": 1, "shutdown_timeout_s": 1},
    }
    spec.update(changes)
    return spec


def ready_metadata(adapter):
    return {
        "endpoint": "/v1/chat/completions",
        "independent_request_threads": 16,
        "closed_loop": True,
        "retries": 0,
        "timeout_s": 2,
        "max_tokens": 8,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "prompt": module.PROMPT,
        "strict_labels": list(module.LABELS),
    }


def test_create_is_side_effect_free(tmp_path, monkeypatch):
    attempt = tmp_path / "not-created"
    run_calls = []
    popen_calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: run_calls.append(a))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: popen_calls.append(a))

    adapter = module.create(spec_for(executable(tmp_path)), attempt)

    assert adapter.supported_batches == (1, 16)
    assert not attempt.exists()
    assert run_calls == []
    assert popen_calls == []


def test_relative_python_preserves_venv_symlink(tmp_path, monkeypatch):
    project = tmp_path / "project"
    bin_dir = project / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    global_python = executable(tmp_path)
    venv_python = bin_dir / "python"
    venv_python.symlink_to(global_python)
    monkeypatch.setattr(module, "_PROJECT_ROOT", project)
    spec = spec_for(Path("venv/bin/python"))
    spec["engine_safety"]["expected_python"] = "venv/bin/python"

    adapter = module.create(spec, tmp_path / "attempt")
    observed = adapter._resolve_python()

    assert observed == venv_python
    assert observed.is_symlink()
    assert observed.resolve() == global_python.resolve()
    assert observed != global_python.resolve()


def test_setup_command_metadata_and_owned_group_cleanup(tmp_path, monkeypatch):
    python = executable(tmp_path)
    attempt = tmp_path / "attempt"
    fake = FakeProcess()
    calls = {}
    kills = []

    def version_probe(command, **kwargs):
        calls["version"] = (command, kwargs)
        return SimpleNamespace(returncode=0, stdout="0.19.0\n", stderr="")

    def launch(command, **kwargs):
        calls["launch"] = (command, kwargs)
        return fake

    monkeypatch.setattr(subprocess, "run", version_probe)
    monkeypatch.setattr(subprocess, "Popen", launch)
    monkeypatch.setattr(os, "killpg", lambda pid, sig: kills.append((pid, sig)))
    adapter = module.create(spec_for(python), attempt)
    monkeypatch.setattr(adapter, "_wait_for_server", lambda timeout: None)
    monkeypatch.setattr(adapter, "_check_served_model", lambda: None)

    metadata = adapter.setup([("train-0", "a tweet", 0)], [])
    command, launch_kwargs = calls["launch"]
    _, version_kwargs = calls["version"]

    assert command[:4] == [str(python), "-m", "vllm.entrypoints.cli.main", "serve"]
    assert command[command.index("--dtype") + 1] == "bfloat16"
    assert command[command.index("--max-model-len") + 1] == "1024"
    assert command[command.index("--max-num-seqs") + 1] == "16"
    assert command[command.index("--revision") + 1] == spec_for(python)["revision"]
    assert "--no-enable-prefix-caching" in command
    assert "--quantization" not in command
    assert launch_kwargs["start_new_session"] is True
    assert launch_kwargs["cwd"] == str(attempt)
    assert version_kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert launch_kwargs["env"]["HF_HOME"] == str(attempt / "cache" / "hf")
    assert launch_kwargs["env"]["VLLM_CACHE_ROOT"] == str(attempt / "cache" / "vllm")
    assert metadata["engine_version"] == "0.19.0"
    assert metadata["dtype"] == "bfloat16"
    assert metadata["quantization"] is None
    assert metadata["prefix_caching"] is False
    assert metadata["request_policy"]["retries"] == 0
    assert metadata["request_policy"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert metadata["train_examples"] == 1
    assert metadata["validation_examples"] == 0
    assert metadata["warmup_performed_by_adapter"] is False
    json.dumps(metadata, allow_nan=False)
    events = [json.loads(line)["event"] for line in (attempt / "vllm-adapter.jsonl").read_text().splitlines()]
    assert events == ["server_start", "server_ready"]

    adapter.close()
    adapter.close()
    assert kills == [(fake.pid, signal.SIGTERM)]
    assert fake.wait_calls == [1.0]
    assert adapter._server_output is None
    assert json.loads((attempt / "vllm-adapter.jsonl").read_text().splitlines()[-1])["event"] == "server_stopped"


def test_setup_rejects_quantization_before_version_or_launch(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: called.append("run"))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: called.append("popen"))
    spec = spec_for(executable(tmp_path), params={"quantization": "awq"})
    adapter = module.create(spec, tmp_path / "attempt")

    with pytest.raises(Unavailable, match="Quantization is forbidden"):
        adapter.setup([], [])
    assert called == []


def test_early_cuda_oom_is_unavailable_and_cleans_up(tmp_path, monkeypatch):
    fake = FakeProcess(returncode=1)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="0.19.0\n", stderr=""),
    )
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: fake)
    adapter = module.create(spec_for(executable(tmp_path)), tmp_path / "attempt")

    def write_oom(*args, **kwargs):
        (adapter.attempt / "vllm-server.log").write_text("torch.cuda.OutOfMemoryError: CUDA out of memory")

    monkeypatch.setattr(adapter, "_log", write_oom)
    with pytest.raises(Unavailable, match="out of memory"):
        adapter.setup([], [])
    assert adapter._process is None
    assert adapter._closed is True


@pytest.mark.parametrize(
    "content,expected",
    [
        ("negative", "negative"),
        ("  NEUTRAL\n", "neutral"),
        ("positive", "positive"),
        ("positive.", None),
        ("positive neutral", None),
        ("```json\n{\"label\":\"positive\"}\n```", None),
        ("", None),
    ],
)
def test_strict_label_parser(content, expected):
    assert module._parse_label(content) == expected


class RecordingServer:
    def __init__(self, status=200):
        self.status = status
        self.requests = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()
        self.first_group_arrived = threading.Barrier(16)
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                body = json.loads(self.rfile.read(length))
                text = body["messages"][0]["content"].removeprefix(module.PROMPT)
                index = int(text.rsplit("-", 1)[-1])
                with owner.lock:
                    owner.active += 1
                    owner.max_active = max(owner.max_active, owner.active)
                    owner.requests.append((index, time.monotonic(), body, self.path))
                try:
                    if owner.status == 200 and index < 16:
                        # Hold the first wave until all independent requests are
                        # inside the HTTP handler; client thread startup alone
                        # need not produce 16 simultaneous fake-server arrivals.
                        owner.first_group_arrived.wait(timeout=2)
                    time.sleep(0.025)
                    if owner.status == 200:
                        response = {
                            "id": f"response-{index}",
                            "model": "sentiment-vllm",
                            "choices": [
                                {
                                    "message": {"role": "assistant", "content": module.LABELS[index % 3]},
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {"prompt_tokens": 20, "completion_tokens": 1, "total_tokens": 21},
                        }
                        payload = json.dumps(response).encode()
                    else:
                        payload = b'{"error":"injected failure"}'
                    self.send_response(owner.status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                finally:
                    with owner.lock:
                        owner.active -= 1

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join()


def loopback_adapter(tmp_path, server):
    adapter = module.create(spec_for(tmp_path / "unused-python"), tmp_path / "loopback-attempt")
    adapter.attempt.mkdir()
    adapter._base_url = f"http://127.0.0.1:{server.httpd.server_port}"
    adapter._request_policy = ready_metadata(adapter)
    adapter._served_name = "sentiment-vllm"
    adapter._process = FakeProcess()
    adapter._ready = True
    return adapter


def test_predict_width_16_is_ordered_independent_and_closed_loop(tmp_path):
    with RecordingServer() as server:
        adapter = loopback_adapter(tmp_path, server)
        results = adapter.predict([f"tweet-{index}" for index in range(17)])

        assert [row["label"] for row in results] == [module.LABELS[i % 3] for i in range(17)]
        assert [row["raw"]["response_id"] for row in results] == [f"response-{i}" for i in range(17)]
        assert all(row["latency_s"] > 0 for row in results)
        assert server.max_active == 16
        assert len(server.requests) == 17
        # The seventeenth request cannot start before every first-group response
        # has been recorded and its handler sleep has completed.
        seventeenth_start = next(start for i, start, body, path in server.requests if i == 16)
        first_group_starts = [start for i, start, body, path in server.requests if i < 16]
        assert seventeenth_start >= max(first_group_starts) + 0.024
        for index, start, body, path in server.requests:
            assert path == "/v1/chat/completions"
            assert body["model"] == "sentiment-vllm"
            assert body["messages"] == [{"role": "user", "content": module.PROMPT + f"tweet-{index}"}]
            assert body["max_tokens"] == 8
            assert body["temperature"] == 0.0
            assert body["top_p"] == 1.0
            assert body["seed"] == 0
            assert body["stream"] is False
            assert body["chat_template_kwargs"] == {"enable_thinking": False}
        json.dumps(results, allow_nan=False)
        event = json.loads((adapter.attempt / "vllm-adapter.jsonl").read_text().splitlines()[-1])
        assert event == {**event, "event": "predict", "tweets": 17, "group_sizes": [16, 1]}


def test_predict_does_not_retry_http_failure(tmp_path):
    with RecordingServer(status=500) as server:
        adapter = loopback_adapter(tmp_path, server)
        with pytest.raises(RuntimeError, match="HTTP 500"):
            adapter.predict(["tweet-0"])
        assert len(server.requests) == 1


def test_health_accepts_empty_200_body(tmp_path, monkeypatch):
    adapter = module.create(spec_for(tmp_path / "unused-python"), tmp_path / "health-attempt")
    adapter._base_url = "http://127.0.0.1:1"
    adapter._process = FakeProcess()
    opened = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return b""

    monkeypatch.setattr(adapter, "_open", lambda request, timeout: opened.append(request) or Response())
    adapter._wait_for_server(1)
    assert len(opened) == 1
