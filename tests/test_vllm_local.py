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
        "independent_request_threads": module.MAX_SUBMISSION_WIDTH,
        "closed_loop": True,
        "retries": 0,
        "timeout_s": 10,
        "scheduler_max_num_seqs": module.SCHEDULER_MAX_NUM_SEQS,
        "supported_submission_widths": list(module.SUBMISSION_WIDTHS),
        "submission_width": None,
        "submission_width_source": None,
        "max_concurrent_client_requests": module.MAX_SUBMISSION_WIDTH,
        "max_queue_waves": module.MAX_SUBMISSION_WIDTH // module.SCHEDULER_MAX_NUM_SEQS,
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

    # Wider widths are client bursts over the same fixed 16-sequence window.
    assert adapter.supported_batches == (1, 4, 16, 64, 256)
    assert module.SCHEDULER_MAX_NUM_SEQS == 16
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
    assert metadata["max_num_seqs"] == 16
    assert metadata["request_policy"]["scheduler_max_num_seqs"] == 16
    assert metadata["request_policy"]["submission_width"] is None
    assert metadata["request_policy"]["max_concurrent_client_requests"] == 256
    assert metadata["request_policy"]["max_queue_waves"] == 16
    assert metadata["client_burst"] == {
        "supported_widths": [1, 4, 16, 64, 256],
        "declared_width_source": None,
        "declared_width": None,
        "max_concurrent_requests": 256,
        "scheduler_window": 16,
        "max_queue_waves": 16,
        "required_fds": 256 + module._CLIENT_FD_MARGIN,
        "rlimit_nofile_soft": metadata["client_burst"]["rlimit_nofile_soft"],
    }
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


@pytest.mark.parametrize("width", [4, 16, 64, 256])
def test_setup_pins_fixed_scheduler_window_for_wider_widths(tmp_path, monkeypatch, width):
    spec = spec_for(executable(tmp_path), params={"submission_width": width})
    adapter = module.create(spec, tmp_path / "attempt")
    commands = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="0.19.0\n", stderr=""))
    monkeypatch.setattr(subprocess, "Popen", lambda command, **k: commands.append(command) or FakeProcess())
    monkeypatch.setattr(os, "killpg", lambda pid, sig: None)
    monkeypatch.setattr(adapter, "_wait_for_server", lambda timeout: None)
    monkeypatch.setattr(adapter, "_check_served_model", lambda: None)

    metadata = adapter.setup([], [])

    command = commands[0]
    assert command[command.index("--max-num-seqs") + 1] == "16"
    assert metadata["max_num_seqs"] == 16
    assert metadata["client_burst"]["declared_width"] == width
    assert metadata["client_burst"]["max_concurrent_requests"] == width
    assert metadata["client_burst"]["scheduler_window"] == 16
    assert metadata["client_burst"]["max_queue_waves"] == _queue_waves(width)
    assert metadata["request_policy"]["submission_width"] == width
    assert metadata["request_policy"]["max_concurrent_client_requests"] == width
    assert metadata["request_policy"]["independent_request_threads"] == width
    adapter.close()


def test_setup_rejects_widened_scheduler_window(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: called.append("run"))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: called.append("popen"))
    spec = spec_for(executable(tmp_path), params={"max_num_seqs": 64, "submission_width": 64})
    adapter = module.create(spec, tmp_path / "attempt")

    with pytest.raises(Unavailable, match="fixed scheduler window"):
        adapter.setup([], [])
    assert called == []


@pytest.mark.parametrize("width", [0, 32, 128, True, "64", 2.0])
def test_setup_rejects_unsupported_submission_width(tmp_path, monkeypatch, width):
    called = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: called.append("run"))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: called.append("popen"))
    spec = spec_for(executable(tmp_path), params={"submission_width": width})
    adapter = module.create(spec, tmp_path / "attempt")

    with pytest.raises(Unavailable, match="submission_width must be one of"):
        adapter.setup([], [])
    assert called == []


def test_runtime_spec_width_overrides_static_param(tmp_path, monkeypatch):
    """The runner injects the per-block width at the top level of the spec."""
    spec = spec_for(executable(tmp_path), submission_width=64, params={"submission_width": 16})
    adapter = module.create(spec, tmp_path / "attempt")
    commands = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="0.19.0\n", stderr=""))
    monkeypatch.setattr(subprocess, "Popen", lambda command, **k: commands.append(command) or FakeProcess())
    monkeypatch.setattr(os, "killpg", lambda pid, sig: None)
    monkeypatch.setattr(adapter, "_wait_for_server", lambda timeout: None)
    monkeypatch.setattr(adapter, "_check_served_model", lambda: None)

    metadata = adapter.setup([], [])

    assert metadata["client_burst"]["declared_width"] == 64
    assert metadata["client_burst"]["declared_width_source"] == "spec"
    # The file-descriptor budget follows the width that will actually run.
    assert metadata["client_burst"]["max_concurrent_requests"] == 64
    assert metadata["client_burst"]["required_fds"] == 64 + module._CLIENT_FD_MARGIN
    assert metadata["request_policy"]["submission_width_source"] == "spec"
    assert commands[0][commands[0].index("--max-num-seqs") + 1] == "16"
    adapter.close()


def test_setup_fails_fast_when_file_limit_blocks_the_burst(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(module.resource, "getrlimit", lambda *_: (200, 200))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: called.append("run"))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: called.append("popen"))
    spec = spec_for(executable(tmp_path), params={"submission_width": 256})
    adapter = module.create(spec, tmp_path / "attempt")

    with pytest.raises(Unavailable, match="requires at least 320 open files"):
        adapter.setup([], [])
    # The measurement window must not open only to fail inside the first burst.
    assert called == []
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


def _queue_waves(width):
    return -(-width // module.SCHEDULER_MAX_NUM_SEQS)


class BurstHTTPServer(ThreadingHTTPServer):
    # The stdlib default listen backlog is 5; a 64/256-request burst needs room in
    # the accept queue or simultaneous loopback connects stall or time out.
    request_queue_size = 256
    daemon_threads = True


class FakeEngine:
    """Loopback fake OpenAI server that holds a whole burst inside the handlers.

    ``batched`` additionally models the engine's fixed scheduler window: nothing
    is answered until every burst member arrived, and requests are then served in
    consecutive batches of at most ``window``, so a wider client burst queues.
    """

    def __init__(self, burst, *, status=200, batched=False, hold_s=0.025, timeout=20.0):
        self.burst = burst
        self.status = status
        self.batched = batched
        self.hold_s = hold_s
        self.timeout = timeout
        self.window = module.SCHEDULER_MAX_NUM_SEQS
        self.requests = []
        self.active = 0
        self.max_active = 0
        self.batches = 0
        self.batch_sizes = []
        self.pending = []
        self.stopping = False
        self.lock = threading.Lock()
        self.ready = threading.Condition(self.lock)
        self.arrived = threading.Barrier(burst)
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
                    if owner.status == 200 and index < owner.burst:
                        # Hold every burst member until all of them are inside the
                        # handler; a wave-by-wave client cannot reach this point.
                        owner.arrived.wait(timeout=owner.timeout)
                    if owner.batched:
                        ticket = threading.Event()
                        with owner.lock:
                            owner.pending.append(ticket)
                            owner.ready.notify()
                        assert ticket.wait(timeout=owner.timeout), "fake engine never served"
                    else:
                        time.sleep(owner.hold_s)
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
                    try:
                        self.send_response(owner.status)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(payload)))
                        self.end_headers()
                        self.wfile.write(payload)
                    except (BrokenPipeError, ConnectionResetError):
                        # The client abandoned the request; nothing to report here.
                        pass
                finally:
                    with owner.lock:
                        owner.active -= 1

            def log_message(self, *args):
                pass

        self.httpd = BurstHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.dispatch = threading.Thread(target=self._serve_batches, daemon=True)

    def _serve_batches(self):
        sealed = False
        while True:
            with self.lock:
                # Nothing is answered before the whole burst arrived, then
                # requests drain in consecutive windows of at most 16.
                while not self.stopping:
                    if sealed and self.pending:
                        break
                    if not sealed and len(self.pending) >= self.burst:
                        sealed = True
                        break
                    self.ready.wait(timeout=0.05)
                if not self.pending:
                    return
                batch, self.pending = self.pending[: self.window], self.pending[self.window :]
                self.batches += 1
                self.batch_sizes.append(len(batch))
            time.sleep(self.hold_s)
            for ticket in batch:
                ticket.set()

    def __enter__(self):
        self.thread.start()
        if self.batched:
            self.dispatch.start()
        return self

    def __exit__(self, *args):
        with self.lock:
            self.stopping = True
            self.ready.notify_all()
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join()
        if self.batched:
            self.dispatch.join(timeout=5)


def loopback_adapter(tmp_path, server, submission_width=None):
    adapter = module.create(spec_for(tmp_path / "unused-python"), tmp_path / "loopback-attempt")
    adapter.attempt.mkdir()
    adapter._base_url = f"http://127.0.0.1:{server.httpd.server_port}"
    adapter._request_policy = ready_metadata(adapter)
    adapter._served_name = "sentiment-vllm"
    adapter._submission_width = submission_width
    adapter._burst_width = submission_width or module.MAX_SUBMISSION_WIDTH
    adapter._process = FakeProcess()
    adapter._ready = True
    return adapter


def offline_adapter(tmp_path, submission_width=None):
    """A ready adapter whose requests never leave the process."""
    adapter = module.create(spec_for(tmp_path / "unused-python"), tmp_path / "offline-attempt")
    adapter.attempt.mkdir()
    adapter._base_url = "http://127.0.0.1:1"
    adapter._request_policy = ready_metadata(adapter)
    adapter._served_name = "sentiment-vllm"
    adapter._submission_width = submission_width
    adapter._burst_width = submission_width or module.MAX_SUBMISSION_WIDTH
    adapter._process = FakeProcess()
    adapter._ready = True
    return adapter


def assert_request_bodies(server, count):
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
    assert len(server.requests) == count


@pytest.mark.parametrize("width", [1, 4, 16, 64, 256])
def test_predict_burst_queues_through_fixed_scheduler_window(tmp_path, width):
    with FakeEngine(burst=width, batched=True, hold_s=0.004) as server:
        adapter = loopback_adapter(tmp_path, server)
        started = time.monotonic()
        results = adapter.predict([f"tweet-{index}" for index in range(width)])
        elapsed = time.monotonic() - started

    assert [row["label"] for row in results] == [module.LABELS[i % 3] for i in range(width)]
    assert [row["raw"]["response_id"] for row in results] == [f"response-{i}" for i in range(width)]
    assert all(row["latency_s"] > 0 for row in results)
    # The whole burst was in flight at once: the fake server held each request
    # until every other member of the group also arrived.
    assert server.max_active == width
    # Wider bursts queue into 16-sequence admission waves instead of widening
    # the engine's resident batch.
    waves = _queue_waves(width)
    assert server.batches == waves
    assert all(size <= module.SCHEDULER_MAX_NUM_SEQS for size in server.batch_sizes)
    assert elapsed >= waves * 0.004 * 0.5
    assert_request_bodies(server, width)
    json.dumps(results, allow_nan=False)
    event = json.loads((adapter.attempt / "vllm-adapter.jsonl").read_text().splitlines()[-1])
    assert event["event"] == "predict"
    assert event["tweets"] == width
    assert event["submission_width"] == width
    assert event["submission_width_source"] == "caller_group"
    assert event["group_sizes"] == [width]
    assert event["queue_waves"] == [waves]
    assert event["scheduler_max_num_seqs"] == 16


def test_declared_width_never_splits_a_legal_group(tmp_path):
    """A per-block width keeps every group of that width or less in one burst."""
    with FakeEngine(burst=4) as server:
        adapter = loopback_adapter(tmp_path, server, submission_width=4)
        full = adapter.predict([f"tweet-{index}" for index in range(4)])
        partial = adapter.predict([f"tweet-{index}" for index in range(4, 7)])
        wide = adapter.predict([f"tweet-{index}" for index in range(8, 16)])

        assert server.max_active == 4
        assert (len(full), len(partial), len(wide)) == (4, 3, 8)
        events = [
            json.loads(line)
            for line in (adapter.attempt / "vllm-adapter.jsonl").read_text().splitlines()
        ]
        predicts = [event for event in events if event["event"] == "predict"]
        assert [(event["submission_width_source"], event["group_sizes"]) for event in predicts] == [
            ("declared", [4]),
            ("declared", [3]),
            ("declared_split", [4, 4]),
        ]
        assert [event["queue_waves"] for event in predicts] == [[1], [1], [1, 1]]


def test_predict_burst_splits_input_beyond_largest_supported_width(tmp_path):
    with FakeEngine(burst=256, hold_s=0.001) as server:
        adapter = loopback_adapter(tmp_path, server)
        results = adapter.predict([f"tweet-{index}" for index in range(257)])

    assert len(results) == 257
    assert server.max_active == 256
    event = json.loads((adapter.attempt / "vllm-adapter.jsonl").read_text().splitlines()[-1])
    assert event["event"] == "predict"
    assert event["submission_width"] == 256
    assert event["submission_width_source"] == "capped"
    assert event["group_sizes"] == [256, 1]


def test_declared_width_preserves_historical_chunking(tmp_path):
    with FakeEngine(burst=16) as server:
        adapter = loopback_adapter(tmp_path, server, submission_width=16)
        results = adapter.predict([f"tweet-{index}" for index in range(17)])

        assert [row["label"] for row in results] == [module.LABELS[i % 3] for i in range(17)]
        assert server.max_active == 16
        assert len(server.requests) == 17
        # The seventeenth request cannot start before every first-group response
        # has been recorded and its handler sleep has completed.
        seventeenth_start = next(start for i, start, body, path in server.requests if i == 16)
        first_group_starts = [start for i, start, body, path in server.requests if i < 16]
        assert seventeenth_start >= max(first_group_starts) + 0.024
        assert_request_bodies(server, 17)
        json.dumps(results, allow_nan=False)
        event = json.loads((adapter.attempt / "vllm-adapter.jsonl").read_text().splitlines()[-1])
        assert event["event"] == "predict"
        assert event["submission_width"] == 16
        # Callers that hand in more texts than the declared width still see the
        # historical 16-request groups, now labelled as an explicit split.
        assert event["submission_width_source"] == "declared_split"
        assert event["group_sizes"] == [16, 1]


def test_declared_width_256_keeps_output_order(tmp_path):
    with FakeEngine(burst=256, batched=True, hold_s=0.002) as server:
        adapter = loopback_adapter(tmp_path, server, submission_width=256)
        results = adapter.predict([f"tweet-{index}" for index in range(256)])

    assert [row["raw"]["response_id"] for row in results] == [f"response-{i}" for i in range(256)]
    assert [row["label"] for row in results] == [module.LABELS[i % 3] for i in range(256)]


def test_predict_does_not_retry_http_failure(tmp_path):
    with FakeEngine(burst=1, status=500) as server:
        adapter = loopback_adapter(tmp_path, server)
        with pytest.raises(RuntimeError, match="HTTP 500"):
            adapter.predict(["tweet-0"])
        assert len(server.requests) == 1


def test_predict_refuses_burst_beyond_file_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(module.resource, "getrlimit", lambda *_: (200, 200))
    adapter = offline_adapter(tmp_path)
    with pytest.raises(Unavailable, match="RLIMIT_NOFILE soft limit is 200"):
        adapter.predict([f"tweet-{index}" for index in range(256)])
    assert adapter._process is not None


def test_predict_reports_thread_start_failure_without_retry(tmp_path, monkeypatch):
    class NoThreads:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("can't start new thread")

    monkeypatch.setattr(module, "ThreadPoolExecutor", NoThreads)
    adapter = offline_adapter(tmp_path)
    with pytest.raises(RuntimeError, match="Could not start 2 client request threads"):
        adapter.predict(["tweet-0", "tweet-1"])


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
