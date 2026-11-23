from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

from adapters.base import Unavailable
from adapters import qev_local


def spec_for(repo="baselabs/joey-1", **changes):
    revision = changes.pop("revision", qev_local.MODEL_SETUP.get(repo, qev_local.ModelSetup("main", "", {})).revision)
    spec = {"adapter": "qev", "repo": repo, "revision": revision, "params": {}}
    spec.update(changes)
    return spec


def test_exact_model_setup_mapping():
    assert qev_local.ONEJEV_REVISION == "81ce62f1597c91e46767d4d02d6ac2e18534fe94"
    assert {
        repo: (setup.revision, sum(setup.weights.values()))
        for repo, setup in qev_local.MODEL_SETUP.items()
    } == {
        "baselabs/joey-1": ("db5c2a631f2da93c4173378438d8246966e25159", 5_443_677_680),
        "baselabs/elephant-1": ("2e80345f1e41b2c1eaff8ae2538c0ac657210c44", 18_819_722_392),
        "baselabs/mammoth-1": ("858aab610d4d124315ebf72d2f36c5e7341f758f", 54_713_608_144),
    }
    assert set(qev_local.MODEL_SETUP["baselabs/mammoth-1"].weights) == {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }
    assert {repo: setup.config_blob for repo, setup in qev_local.MODEL_SETUP.items()} == {
        "baselabs/joey-1": "5878a46a4364f966cc6e877a4f3f3f80d6230be9",
        "baselabs/elephant-1": "98295053fab64c0a71b1c2bbba06c2e31ed33cfb",
        "baselabs/mammoth-1": "7c80d74dfe54ed2de25007a979c06743dbc29518",
    }


def test_create_is_lazy_and_only_supports_independent_batch_one(tmp_path, monkeypatch):
    def forbidden_spawn(*args, **kwargs):
        raise AssertionError("create must not spawn a process")

    monkeypatch.setattr(qev_local.subprocess, "Popen", forbidden_spawn)
    adapter = qev_local.create(spec_for(local_snapshot=tmp_path / "not-read-at-create"), tmp_path / "attempt")
    assert adapter.supported_batches == (1,)
    assert adapter._process is None
    assert not (tmp_path / "attempt").exists()
    with pytest.raises(Unavailable, match="batch 16"):
        adapter.predict(["one tweet"] * 16)
    adapter.close()


def test_create_rejects_unpinned_or_unknown_models(tmp_path):
    with pytest.raises(Unavailable, match="pinned"):
        qev_local.create(spec_for(revision="main"), tmp_path)
    with pytest.raises(Unavailable, match="unsupported"):
        qev_local.create(spec_for(repo="Qwen/Qwen3.5-2B", revision="main"), tmp_path)
    with pytest.raises(ValueError, match="adapter"):
        qev_local.create({"adapter": "vllm"}, tmp_path)


@pytest.mark.parametrize("repo", list(qev_local.MODEL_SETUP))
def test_packed_is_explicit_opt_in_for_every_pinned_model(tmp_path, repo):
    adapter = qev_local.create(spec_for(repo=repo, params={"request_mode": "packed"}), tmp_path)
    assert adapter.supported_batches == (1, 4, 16, 64, 256)
    assert adapter.request_mode == "packed"
    assert adapter.compute_process_id is None
    with pytest.raises(Unavailable, match="pinned"):
        qev_local.create(spec_for(repo=repo, revision="main", params={"request_mode": "packed"}), tmp_path)
    with pytest.raises(Unavailable, match="unpinned"):
        qev_local.QEVLocalAdapter(spec_for(repo=repo, revision="main", params={"request_mode": "packed"}),
                                  tmp_path, qev_local.MODEL_SETUP[repo])
    adapter.close()


def test_setup_requires_pinned_local_snapshot_before_spawn(tmp_path, monkeypatch):
    def forbidden_spawn(*args, **kwargs):
        raise AssertionError("missing snapshot must fail before spawn")

    monkeypatch.setattr(qev_local.subprocess, "Popen", forbidden_spawn)
    adapter = qev_local.create(spec_for(), tmp_path)
    with pytest.raises(Unavailable, match="local_snapshot"):
        adapter.setup([], [])


def test_invalid_choice_becomes_none_and_raw_is_json(tmp_path, monkeypatch):
    adapter = qev_local.create(spec_for(), tmp_path)
    raw = {
        "answer": {"type": "choice", "choice": "none of the options listed here", "probabilities": {}},
        "usage": {"input_tokens": 4, "output_tokens": 1},
        "runtime": {"fork_mode": "sequential"},
    }
    monkeypatch.setattr(adapter, "_rpc", lambda *args: raw)
    result = adapter.predict(["a tweet"])[0]
    assert result["label"] is None
    assert result["raw"] is raw
    assert result["latency_s"] >= 0
    json.dumps(result, allow_nan=False)
    adapter.close()


def _git(cwd: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(cwd), *args], text=True).strip()


def _fake_runtime_and_snapshot(tmp_path: Path, monkeypatch):
    """Build a tiny patched source tree and QEV API fake; no ML dependencies/CUDA."""
    runtime = tmp_path / "OneJev"
    snapshot = tmp_path / "snapshot"
    (runtime / "qev").mkdir(parents=True)
    snapshot.mkdir()
    (runtime / "README.md").write_text("base\n", encoding="utf-8")
    (runtime / "qev" / "__init__.py").write_text("", encoding="utf-8")
    (runtime / "qev" / "schema.py").write_text(
        "class SystemOneRequest:\n"
        "    def __init__(self, state, questions):\n"
        "        self.state, self.questions = state, questions\n",
        encoding="utf-8",
    )
    (runtime / "qev" / "engine.py").write_text(
        "from dataclasses import dataclass\n"
        "_qev_noncausal = True\n"
        "@dataclass\n"
        "class EngineInfo:\n"
        "    extra: dict\n"
        "class Config:\n"
        "    _attn_implementation = 'sdpa'\n"
        "class Model:\n"
        "    config = Config()\n"
        "class ReadoutConfig:\n"
        "    noncausal = True\n"
        "    readout = 'slot_lm'\n"
        "class Readout:\n"
        "    config = ReadoutConfig()\n"
        "    head = None\n"
        "class Value:\n"
        "    def __init__(self, value): self.value = value\n"
        "    def model_dump(self, mode): return self.value\n"
        "class Response:\n"
        "    def __init__(self, choice):\n"
        "        self.answers = {'sentiment': Value({'type': 'choice', 'choice': choice, "
        "'probabilities': {'negative': 0.1, 'neutral': 0.2, 'positive': 0.7}, 'confidence': 0.4})}\n"
        "        self.usage = Value({'input_tokens': 12, 'output_tokens': 1})\n"
        "class DecisionEngine:\n"
        "    def __init__(self, model_path, **options):\n"
        "        self.model, self.readout, self.slot_ids = Model(), Readout(), [1, 2, 3]\n"
        "        self.info = EngineInfo(extra={'attention': 'noncausal', 'readout': 'slot_lm'})\n"
        "        self.options = options\n"
        "    def decide(self, request, debias=1, debug=False):\n"
        "        assert request.questions['sentiment']['type'] == 'choice'\n"
        "        assert tuple(request.questions['sentiment']['criteria']) == ('negative', 'neutral', 'positive')\n"
        "        return Response('positive'), {'prompt_version': 'fake-qev', 'debias': debias, 'debug': debug}\n",
        encoding="utf-8",
    )
    _git(runtime, "init", "-q")
    _git(runtime, "config", "user.email", "qev-test@example.invalid")
    _git(runtime, "config", "user.name", "QEV Test")
    _git(runtime, "add", ".")
    _git(runtime, "commit", "-qm", "fake upstream")
    head = _git(runtime, "rev-parse", "HEAD")

    (runtime / "README.md").write_text("base\npatched\n", encoding="utf-8")
    patch = _git(runtime, "diff")
    (snapshot / "serve.patch").write_text(patch + "\n", encoding="utf-8")
    (snapshot / "setup.sh").write_text("# fake pinned installer\n", encoding="utf-8")
    overrides = {
        "attention": {"full_attention": "block_bidirectional", "linear_attention": "causal"},
        "readout": "slot_lm",
        "head_dim": 256,
        "n_slots": 256,
    }
    (snapshot / "config.json").write_text(json.dumps({"modeling_overrides": overrides}), encoding="utf-8")
    (snapshot / "readout.json").write_text(json.dumps({"readout": "slot_lm"}), encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"tiny fake weights")

    monkeypatch.setattr(qev_local, "ONEJEV_REVISION", head)
    monkeypatch.setattr(qev_local, "PATCH_BLOB", qev_local._git_blob(snapshot / "serve.patch"))
    monkeypatch.setattr(qev_local, "SETUP_BLOB", qev_local._git_blob(snapshot / "setup.sh"))
    monkeypatch.setattr(qev_local, "READOUT_BLOB", qev_local._git_blob(snapshot / "readout.json"))
    monkeypatch.setitem(
        qev_local.MODEL_SETUP,
        "baselabs/joey-1",
        qev_local.ModelSetup(
            revision=head,
            config_blob=qev_local._git_blob(snapshot / "config.json"),
            weights={"model.safetensors": (snapshot / "model.safetensors").stat().st_size},
        ),
    )
    return runtime, snapshot


def test_fake_qev_subprocess_setup_predict_and_bounded_close(tmp_path, monkeypatch):
    runtime, snapshot = _fake_runtime_and_snapshot(tmp_path, monkeypatch)
    spec = spec_for(
        local_snapshot=str(snapshot),
        params={
            "onejev_path": str(runtime),
            "python": sys.executable,
            "device": "cpu",
            "init_timeout_s": 10,
            "request_timeout_s": 5,
            "close_timeout_s": 2,
        },
    )
    adapter = qev_local.create(spec, tmp_path / "attempt")
    metadata = adapter.setup([("unused", "train", 0)], [("unused", "validation", 1)])
    assert metadata["repo"] == "baselabs/joey-1"
    assert metadata["supported_batches"] == [1]
    assert metadata["runtime"]["engine_info"]["extra"] == {
        "attention": "noncausal",
        "readout": "slot_lm",
    }
    assert metadata["runtime"]["runtime_source"]["serve_patch_reverse_check"] is True
    assert metadata["runtime"]["smoke_answer"]["choice"] == "positive"
    assert metadata["runtime"]["generative_decoding"] is False
    assert metadata["engine_options"]["device"] == "cpu"
    json.dumps(metadata, allow_nan=False)

    process = adapter._process
    assert process is not None and process.poll() is None
    assert adapter.compute_process_id == process.pid == metadata["worker_pid"]
    prediction = adapter.predict(["a happy tweet"])[0]
    assert prediction["label"] == "positive"
    assert prediction["raw"]["runtime"] == {"prompt_version": "fake-qev", "debias": 1, "debug": False}
    assert prediction["latency_s"] >= 0
    assert os.environ.get("HF_HUB_OFFLINE") != "1"  # offline override is child-only
    adapter.close()
    assert process.poll() is not None
    assert adapter._process is None
    assert adapter.compute_process_id is None
    adapter.close()  # idempotent


@pytest.fixture
def packed_worker(monkeypatch):
    from qev_setup import qev_worker

    class Request:
        def __init__(self, state, questions):
            self.state = state
            self.questions = {k: types.SimpleNamespace(**v) for k, v in questions.items()}

    def render(qid, question, order=None, style=None):
        return types.SimpleNamespace(qid=qid, n_slots=3, suffix=(question.instructions, order))

    monkeypatch.setitem(sys.modules, "qev", types.ModuleType("qev"))
    monkeypatch.setitem(sys.modules, "qev.schema", types.SimpleNamespace(SystemOneRequest=Request))
    monkeypatch.setitem(sys.modules, "qev.prompt", types.SimpleNamespace(render_question=render))

    class Value:
        def __init__(self, value): self.value = value
        def model_dump(self, mode): return self.value

    class Engine:
        max_slots, style, max_branch_tokens, max_request_tokens = 3, None, 32768, 65536
        calls = 0
        decisions = 0
        def _prefix_and_suffix_ids(self, state, rendered):
            self.calls += 1
            return [0] * (len(json.dumps(state)) + 1), [[0] * (10 + len(r.suffix[0])) for r in rendered]
        def decide(self, request, debias, debug):
            self.decisions += 1
            import logging
            logging.getLogger("qev.engine").warning("retrying with chunk 2")
            answers = {qid: Value({"type": "choice", "choice": "positive", "probabilities": {"positive": 1.0}})
                       for qid in request.questions}
            return types.SimpleNamespace(answers=answers, usage=Value({"input_tokens": 1})), {"timing": {"forward_s": 0.1}}

    worker = qev_worker.Worker()
    worker.engine = Engine()
    return qev_worker, worker


def test_packed_preflight_and_predict_cache(packed_worker):
    module, worker = packed_worker
    texts = ["text 1", "text 2", "\"tweet 3\"", "text 4"]
    request = module._request(texts)
    assert request.state["tweets"][2] == {"number": 3, "text": texts[2]}
    assert "tweet number 3" in request.questions["sentiment_3"].instructions
    assert module._request(["tail"]).state == "tail"
    assert list(module._request(["tail"]).questions) == ["sentiment"]
    payload = {"texts": texts, "debias": 2, "debug": False, "request_mode": "packed"}
    receipt = worker.check_context(payload)
    assert receipt["fit"] and receipt["width"] == 4 and len(receipt["branch_tokens"]) == 8
    assert receipt["model_forwarding"] is False and worker.engine.decisions == 0
    assert worker.engine.calls == 1
    result = worker.predict_packed(payload)
    assert worker.engine.calls == 1 and worker.engine.decisions == 1
    assert list(result["answers"]) == [f"sentiment_{i}" for i in range(1, 5)]
    assert result["runtime"]["fallback_events"] == ["retrying with chunk 2"]
    assert result["answers"]["sentiment_3"]["probabilities"]["positive"] == 1.0


def test_context_limit_receipt_and_no_forward(packed_worker, tmp_path, monkeypatch):
    _, worker = packed_worker
    worker.engine.max_request_tokens = 1
    payload = {"texts": ["one", "two"], "debias": 1, "request_mode": "packed", "debug": False}
    receipt = worker.check_context(payload)
    assert not receipt["fit"] and receipt["checks"][2]["measured"] > 1
    assert worker.predict_packed(payload) == {"context": receipt}
    assert worker.engine.decisions == 0
    adapter = qev_local.create(spec_for(params={"request_mode": "packed"}), tmp_path)
    monkeypatch.setattr(adapter, "_rpc", lambda *args: receipt)
    with pytest.raises(Unavailable, match="max_request_tokens.* > 1") as exc:
        adapter.check_context(payload["texts"])
    assert exc.value.context_receipt == receipt


def test_packed_adapter_mapping_and_no_fake_latencies(packed_worker, tmp_path, monkeypatch):
    _, worker = packed_worker
    adapter = qev_local.create(spec_for(params={"request_mode": "packed"}), tmp_path)
    monkeypatch.setattr(adapter, "_rpc", lambda op, payload, *args: worker.predict_packed(payload))
    assert adapter.supported_batches == (1, 4, 16, 64, 256)
    outputs = adapter.predict(["x", "y", "z", "w"])
    assert [item["raw"]["target_number"] for item in outputs] == [1, 2, 3, 4]
    assert all(item["label"] == "positive" and "latency_s" not in item for item in outputs)
    assert "context" in outputs[0]["raw"] and "context" not in outputs[1]["raw"]
    assert adapter.predict(["tail"])[0]["latency_s"] >= 0


def test_card_branch_limit_is_independent_of_runtime_limit(packed_worker):
    _, worker = packed_worker
    payload = {"texts": ["x" * 17000], "debias": 1, "request_mode": "packed"}
    receipt = worker.check_context(payload)
    assert receipt["max_branch_tokens"] > 16384
    assert receipt["checks"][0]["fit"] is False
    assert receipt["checks"][1]["fit"] is True
    assert not receipt["fit"] and worker.engine.decisions == 0


def test_packed_answer_mapping_mismatch_is_rejected(packed_worker, tmp_path, monkeypatch):
    _, worker = packed_worker
    adapter = qev_local.create(spec_for(params={"request_mode": "packed"}), tmp_path)
    def bad_rpc(op, payload, *args):
        result = worker.predict_packed(payload)
        del result["answers"]["sentiment_2"]
        return result
    monkeypatch.setattr(adapter, "_rpc", bad_rpc)
    with pytest.raises(RuntimeError, match="answer mapping"):
        adapter.predict(["one", "two"])
