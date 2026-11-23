from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

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
    prediction = adapter.predict(["a happy tweet"])[0]
    assert prediction["label"] == "positive"
    assert prediction["raw"]["runtime"] == {"prompt_version": "fake-qev", "debias": 1, "debug": False}
    assert prediction["latency_s"] >= 0
    assert os.environ.get("HF_HUB_OFFLINE") != "1"  # offline override is child-only
    adapter.close()
    assert process.poll() is not None
    assert adapter._process is None
    adapter.close()  # idempotent
