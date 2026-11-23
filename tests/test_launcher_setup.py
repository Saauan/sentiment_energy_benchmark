from __future__ import annotations

import copy
import os
from pathlib import Path
import subprocess

import pytest

import launcher_setup as launcher


class Log:
    def __init__(self):
        self.lines: list[str] = []

    def line(self, text, *, detail=False):
        self.lines.append(str(text))

    def command(self, argv):
        self.lines.append("$ " + " ".join(str(item) for item in argv))

    def close(self):
        pass


def completed(argv, code=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(list(argv), code, stdout, stderr)


def fake_run(monkeypatch, handler):
    calls = []

    def run(argv, log, **kwargs):
        calls.append(([str(item) for item in argv], kwargs))
        return handler([str(item) for item in argv], kwargs)

    monkeypatch.setattr(launcher, "_run", run)
    return calls


def dataset():
    return {"path": "data/tweeteval", "revision": launcher.ONEJEV_REVISION}


def vllm_spec(model_id="qwen2b", **changes):
    spec = {
        "adapter": "vllm",
        "repo": "Qwen/Qwen3.5-2B",
        "revision": "1" * 40,
        "local_snapshot": "models/qwen2b",
        "python": launcher.VLLM_PYTHON_REL,
        "params": {},
        "engine_safety": {"expected_python": launcher.VLLM_PYTHON_REL},
    }
    spec.update(changes)
    return spec


def qev_spec(model_id="joey1", **changes):
    spec = {
        "adapter": "qev",
        "repo": "baselabs/joey-1",
        "revision": "2" * 40,
        "local_snapshot": "models/joey1",
        "params": {"python": launcher.QEV_PYTHON_REL,
                   "onejev_path": launcher.QEV_ONEJEV_REL},
    }
    spec.update(changes)
    return spec


def classical_spec():
    return {"adapter": "svm"}


def document(**models):
    return {"dataset": dataset(), "models": models}


def write_lock(root: Path):
    (root / "requirements.lock").write_text("pinned==1\n", encoding="utf-8")


def make_client(root: Path):
    python = root / launcher.CLIENT_PYTHON_REL
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    return python


def test_qev_requirements_match_demonstrated_fallback():
    text = (launcher.ROOT / "qev_setup/launcher-requirements.txt").read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines()]
    assert "torch==2.10.0+cu128" in lines
    assert "torchvision==0.25.0+cu128" in lines
    assert "transformers==5.19.0" in lines
    assert "flash-linear-attention==0.5.2" in lines
    assert not any(line.startswith("causal-conv1d") for line in lines)
    assert not any(line.startswith("vllm") for line in lines)
    assert not any(line.startswith("-e ") or line.startswith("git+") for line in lines)


def test_git_blob_matches_git_algorithm(tmp_path):
    path = tmp_path / "serve.patch"
    path.write_bytes(b"hello\n")
    assert launcher._git_blob(path) == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_validate_config_accepts_frozen_launcher_paths():
    config = document(qwen=vllm_spec(), joey=qev_spec(), svm=classical_spec())
    before = copy.deepcopy(config)
    launcher._validate_config(config)
    assert config == before


def test_validate_config_rejects_unfrozen_engine_paths():
    config = document(qwen=vllm_spec(python="old/python"))
    with pytest.raises(launcher.SetupError, match="launcher vLLM"):
        launcher._validate_config(config)
    config = document(joey=qev_spec(params={"python": "old/python",
                                             "onejev_path": launcher.QEV_ONEJEV_REL}))
    with pytest.raises(launcher.SetupError, match="launcher QEV"):
        launcher._validate_config(config)


def test_validate_config_rejects_noncanonical_dataset():
    config = document(svm=classical_spec())
    config["dataset"]["path"] = "elsewhere"
    with pytest.raises(launcher.SetupError, match="data/tweeteval"):
        launcher._validate_config(config)


def test_provision_returns_repo_lexical_client_and_does_not_mutate(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    write_lock(root)
    monkeypatch.setattr(launcher, "ROOT", root)
    monkeypatch.setattr(launcher, "_which", lambda name: f"/fake/{name}")
    monkeypatch.setattr(launcher, "_ensure_inputs", lambda root, log: None)
    monkeypatch.setattr(launcher, "_ensure_vllm", lambda root, log: calls.append("vllm"))
    monkeypatch.setattr(launcher, "_ensure_qev_env", lambda root, log: calls.append("qev"))
    monkeypatch.setattr(launcher, "_ensure_qev_source",
                        lambda root, models, log: calls.append("qev-source"))
    monkeypatch.setattr(launcher, "_fetch_models",
                        lambda root, models, engines, log: calls.append("models"))
    calls: list[str] = []

    def client(root, log):
        python = make_client(root)
        return launcher._lexical(python)

    monkeypatch.setattr(launcher, "_ensure_client", client)
    qwen, joey = vllm_spec(), qev_spec()
    config = document(qwen2b=qwen, joey1=joey, svm=classical_spec())
    before = copy.deepcopy(config)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    previous = Path.cwd()
    os.chdir(elsewhere)
    try:
        result = launcher.provision(config, lambda line: None)
    finally:
        os.chdir(previous)
    assert result == launcher._lexical(root / launcher.CLIENT_PYTHON_REL)
    assert result.is_absolute()
    assert config == before
    assert not (root / ".scratch/measurement.lock").exists()
    assert calls == ["vllm", "qev", "models", "qev-source"]


def test_provision_classical_only_does_not_touch_engines(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    write_lock(root)
    monkeypatch.setattr(launcher, "ROOT", root)
    monkeypatch.setattr(launcher, "_which", lambda name: "/fake/uv")
    monkeypatch.setattr(launcher, "_ensure_inputs", lambda root, log: None)
    monkeypatch.setattr(launcher, "_ensure_client", lambda root, log: make_client(root))

    def forbidden(*args, **kwargs):
        raise AssertionError("no engine/model work for classical-only configuration")

    for name in ("_ensure_vllm", "_ensure_qev_env", "_ensure_qev_source", "_fetch_models"):
        monkeypatch.setattr(launcher, name, forbidden)
    config = document(svm=classical_spec())
    assert launcher.provision(config, lambda line: None) == launcher._lexical(
        root / launcher.CLIENT_PYTHON_REL)
    assert config == document(svm=classical_spec())


def test_provision_rejects_bad_paths_before_client_install(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(launcher, "ROOT", root)
    monkeypatch.setattr(launcher, "_which", lambda name: "/fake/uv")

    def forbidden(*args, **kwargs):
        raise AssertionError("must not install with an unfrozen configuration")

    monkeypatch.setattr(launcher, "_ensure_client", forbidden)
    monkeypatch.setattr(launcher, "_ensure_inputs", forbidden)
    with pytest.raises(launcher.SetupError, match="launcher vLLM"):
        launcher.provision(document(qwen=vllm_spec(python="old/python")), lambda line: None)


def test_client_reuses_matching_receipt(monkeypatch, tmp_path):
    root = tmp_path
    write_lock(root)
    python = make_client(root)
    key = {"requirements_sha256": launcher._hash(root / "requirements.lock"),
           "python": launcher.SUPPORTED_PYTHON}
    launcher._record(root, "client", key, {"ok": True})
    monkeypatch.setattr(launcher, "_python_version", lambda root, python, log: launcher.SUPPORTED_PYTHON)
    monkeypatch.setattr(launcher, "_probe", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(launcher, "_run", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("valid receipt must skip uv")))
    assert launcher._ensure_client(root, Log()) == launcher._lexical(python)


def test_client_missing_venv_creates_and_syncs(monkeypatch, tmp_path):
    root = tmp_path
    write_lock(root)
    monkeypatch.setattr(launcher, "_which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(launcher, "_ensure_python", lambda root, log: None)
    monkeypatch.setattr(launcher, "_python_version", lambda root, python, log: launcher.SUPPORTED_PYTHON)
    monkeypatch.setattr(launcher, "_probe", lambda *args, **kwargs: {"ok": True})
    calls = fake_run(monkeypatch, lambda argv, kwargs: completed(argv))
    python = launcher._ensure_client(root, Log())
    commands = [" ".join(call[0]) for call in calls]
    assert any(" venv " in command for command in commands)
    assert any(" pip sync " in command for command in commands)
    assert python == launcher._lexical(root / launcher.CLIENT_PYTHON_REL)


def test_client_refuses_wrong_existing_python(monkeypatch, tmp_path):
    root = tmp_path
    write_lock(root)
    make_client(root)
    monkeypatch.setattr(launcher, "_python_version", lambda root, python, log: "3.11.0")
    monkeypatch.setattr(launcher, "_run", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("must not overwrite a foreign client environment")))
    with pytest.raises(launcher.SetupError, match="refusing to replace"):
        launcher._ensure_client(root, Log())
    assert (root / ".venv" / "bin" / "python").exists()


def make_inputs(root: Path):
    (root / "setup").mkdir(parents=True)
    (root / "setup/inputs.sha256").write_text("0" * 64 + "  data/tweeteval/mapping.txt\n",
                                              encoding="utf-8")
    (root / "setup/fetch_inputs.sh").write_text("#!/bin/sh\n", encoding="utf-8")


def test_inputs_verified_without_fetch(monkeypatch, tmp_path):
    make_inputs(tmp_path)
    monkeypatch.setattr(launcher, "_which", lambda name: f"/usr/bin/{name}")
    calls = fake_run(monkeypatch, lambda argv, kwargs: completed(argv))
    launcher._ensure_inputs(tmp_path, Log())
    assert len(calls) == 1
    assert calls[0][0][1:4] == ["--check", "--strict", "--quiet"]


def test_inputs_fetch_only_when_needed(monkeypatch, tmp_path):
    make_inputs(tmp_path)
    monkeypatch.setattr(launcher, "_which", lambda name: f"/usr/bin/{name}")
    checks = 0

    def run(argv, log, **kwargs):
        nonlocal checks
        if "sha256sum" in argv[0]:
            checks += 1
            return completed(argv, 1 if checks == 1 else 0)
        assert argv[-1].endswith("fetch_inputs.sh")
        return completed(argv)

    fake_run(monkeypatch, lambda argv, kwargs: run(argv, None, **kwargs))
    launcher._ensure_inputs(tmp_path, Log())
    assert checks == 2


def test_inputs_still_bad_after_fetch_fails(monkeypatch, tmp_path):
    make_inputs(tmp_path)
    monkeypatch.setattr(launcher, "_which", lambda name: f"/usr/bin/{name}")
    fake_run(monkeypatch, lambda argv, kwargs: completed(argv, 1 if "sha256sum" in argv[0] else 0))
    with pytest.raises(launcher.SetupError, match="still mismatch after fetch"):
        launcher._ensure_inputs(tmp_path, Log())

def test_owned_venv_refuses_unowned_invalid_directory(monkeypatch, tmp_path):
    directory = tmp_path / "vllm"
    venv = directory / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").write_text("", encoding="utf-8")
    monkeypatch.setattr(launcher, "_python_version", lambda root, python, log: "3.11.0")
    with pytest.raises(launcher.SetupError, match="refusing to overwrite"):
        launcher._owned_venv(tmp_path, directory, venv, venv / "bin/python", "vLLM", Log())
    assert not (directory / ".launcher-owned").exists()


def test_owned_venv_replaces_launcher_owned_incomplete(monkeypatch, tmp_path):
    directory = tmp_path / "vllm"
    venv = directory / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").write_text("", encoding="utf-8")
    (directory / ".launcher-owned").write_text("launcher_setup: vLLM\n", encoding="utf-8")
    versions = iter(["3.11.0", launcher.SUPPORTED_PYTHON])
    monkeypatch.setattr(launcher, "_python_version", lambda root, python, log: next(versions))
    monkeypatch.setattr(launcher, "_ensure_python", lambda root, log: None)
    monkeypatch.setattr(launcher, "_which", lambda name: "/usr/bin/uv")
    calls = fake_run(monkeypatch, lambda argv, kwargs: completed(argv))
    launcher._owned_venv(tmp_path, directory, venv, venv / "bin/python", "vLLM", Log())
    assert any("venv" in command for command, _ in calls)


def test_vllm_install_uses_documented_gpu_root_and_reuses_receipt(monkeypatch, tmp_path):
    root = tmp_path
    setup = root / "setup/gpu"
    setup.mkdir(parents=True)
    (setup / "requirements.txt").write_text("vllm==0.19.0\n", encoding="utf-8")
    (setup / "install.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    hf = root / ".scratch/launcher/vllm/venv/bin/hf"
    hf.parent.mkdir(parents=True)
    hf.write_text("#!/bin/sh\n", encoding="utf-8")
    hf.chmod(0o755)
    monkeypatch.setattr(launcher, "_owned_venv", lambda *args, **kwargs: None)
    monkeypatch.setattr(launcher, "_which", lambda name: f"/usr/bin/{name}")
    observed = {"vllm": "0.19.0", "torch_pkg": "2.10.0", "torch": "2.10.0+cu128",
                "cuda": "12.8", "transformers": "4.57.6", "huggingface_hub": "0.36.2"}
    monkeypatch.setattr(launcher, "_probe", lambda *args, **kwargs: observed)
    calls = fake_run(monkeypatch, lambda argv, kwargs: completed(argv))
    launcher._ensure_vllm(root, Log())
    installs = [call for call in calls if "--install-vllm" in call[0]]
    assert len(installs) == 1
    assert installs[0][1]["env"]["GPU_ROOT"] == str(root / ".scratch/launcher/vllm")
    launcher._ensure_vllm(root, Log())
    assert len([call for call in calls if "--install-vllm" in call[0]]) == 1


def test_qev_install_uses_exact_cu128_fallback_and_reuses_receipt(monkeypatch, tmp_path):
    root = tmp_path
    requirements = root / "qev_setup/launcher-requirements.txt"
    requirements.parent.mkdir(parents=True)
    requirements.write_text("torch==2.10.0+cu128\ntorchvision==0.25.0+cu128\ntransformers==5.19.0\n",
                            encoding="utf-8")
    monkeypatch.setattr(launcher, "_owned_venv", lambda *args, **kwargs: None)
    monkeypatch.setattr(launcher, "_which", lambda name: f"/usr/bin/{name}")
    observed = {"torch": "2.10.0+cu128", "torchvision": "0.25.0+cu128", "cuda": "12.8",
                "transformers": "5.19.0", "flash_linear_attention": "0.5.2",
                "fla_core": "0.5.2", "causal_conv1d": False}
    monkeypatch.setattr(launcher, "_probe", lambda *args, **kwargs: observed)
    calls = fake_run(monkeypatch, lambda argv, kwargs: completed(argv))
    launcher._ensure_qev_env(root, Log())
    installs = [call for call in calls if "install" in call[0]]
    assert len(installs) == 2
    torch_install, other_install = installs
    assert launcher.TORCH_INDEX in torch_install[0]
    assert "torch==2.10.0+cu128" in torch_install[0]
    assert "torchvision==0.25.0+cu128" in torch_install[0]
    assert "--no-deps" in torch_install[0]
    assert launcher.PYPI_INDEX in other_install[0]
    assert "transformers==5.19.0" in other_install[0]
    assert "--no-deps" in other_install[0]
    assert launcher.TORCH_INDEX not in other_install[0]
    assert any("check" in call[0] for call in calls)
    launcher._ensure_qev_env(root, Log())
    assert len([call for call in calls if "install" in call[0]]) == 2


def test_fetch_models_delegates_cached_check_to_pinned_script(monkeypatch, tmp_path):
    root = tmp_path
    make_client(root)
    (root / "setup/gpu").mkdir(parents=True)
    (root / "setup/gpu/model_io.py").write_text("", encoding="utf-8")
    (root / "setup/gpu/fetch_models.sh").write_text("", encoding="utf-8")
    hf = root / ".scratch/launcher/vllm/venv/bin/hf"
    hf.parent.mkdir(parents=True)
    hf.write_text("", encoding="utf-8")
    hf.chmod(0o755)
    spec = vllm_spec()
    calls = fake_run(monkeypatch, lambda argv, kwargs: completed(argv, 0))
    launcher._fetch_models(root, [("qwen2b", spec)], ("vllm",), Log())
    assert sum("fetch_models.sh" in " ".join(call[0]) for call in calls) == 1
    assert sum("verify" in call[0] for call in calls) == 1


def test_fetch_models_success_sets_explicit_approval_and_preflight_guard(monkeypatch, tmp_path):
    root = tmp_path
    make_client(root)
    (root / "setup/gpu").mkdir(parents=True)
    (root / "setup/gpu/model_io.py").write_text("", encoding="utf-8")
    (root / "setup/gpu/fetch_models.sh").write_text("", encoding="utf-8")
    hf = root / ".scratch/launcher/vllm/venv/bin/hf"
    hf.parent.mkdir(parents=True)
    hf.write_text("", encoding="utf-8")
    hf.chmod(0o755)
    spec = vllm_spec()

    def run(argv, log, **kwargs):
        return completed(argv, 0)

    calls = fake_run(monkeypatch, lambda argv, kwargs: run(argv, None, **kwargs))
    launcher._fetch_models(root, [("qwen2b", spec)], ("vllm",), Log())
    fetch = [call for call in calls if "fetch_models.sh" in " ".join(call[0])]
    assert len(fetch) == 1
    assert fetch[0][1]["env"]["ALLOW_LARGE_MODEL_DOWNLOAD"] == "YES"
    assert fetch[0][1]["env"]["PYTHON"] == str(root / launcher.CLIENT_PYTHON_REL)
    assert fetch[0][1]["env"]["PATH"].startswith(str(hf.parent))


def test_fetch_models_guard_failure_is_hard_not_silent(monkeypatch, tmp_path):
    root = tmp_path
    make_client(root)
    (root / "setup/gpu").mkdir(parents=True)
    (root / "setup/gpu/model_io.py").write_text("", encoding="utf-8")
    (root / "setup/gpu/fetch_models.sh").write_text("", encoding="utf-8")
    hf = root / ".scratch/launcher/vllm/venv/bin/hf"
    hf.parent.mkdir(parents=True)
    hf.write_text("", encoding="utf-8")
    hf.chmod(0o755)
    spec = vllm_spec()

    def run(argv, log, **kwargs):
        if "verify" in argv:
            return completed(argv, 1)
        return completed(argv, 1, "weights + reserve require 6000 MiB, but only 5000 MiB are free; no download")

    fake_run(monkeypatch, lambda argv, kwargs: run(argv, None, **kwargs))
    with pytest.raises(launcher.SetupError, match="not silently skipped"):
        launcher._fetch_models(root, [("qwen2b", spec)], ("vllm",), Log())


def test_qev_source_rejects_wrong_patch_blob(monkeypatch, tmp_path):
    spec = qev_spec()
    snapshot = tmp_path / "models/joey1"
    snapshot.mkdir(parents=True)
    (snapshot / "serve.patch").write_text("not the patch", encoding="utf-8")
    monkeypatch.setattr(launcher, "_prepare_onejev",
                        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not prepare")))
    with pytest.raises(launcher.SetupError, match="valid pinned serve.patch"):
        launcher._ensure_qev_source(tmp_path, [("joey1", spec)], Log())


def test_prepare_onejev_never_executes_snapshot_setup_script(monkeypatch, tmp_path):
    repo = tmp_path / launcher.QEV_ONEJEV_REL
    (repo / ".git").mkdir(parents=True)
    patch = tmp_path / "serve.patch"
    patch.write_bytes(b"patch")
    monkeypatch.setattr(launcher, "_git_blob", lambda path: launcher.ONEJEV_PATCH_BLOB)
    monkeypatch.setattr(launcher, "_onejev_ready", lambda *args, **kwargs: True)
    calls = fake_run(monkeypatch, lambda argv, kwargs: completed(argv, 0,
                                                                "https://github.com/OmniJev/OneJev.git\n"))
    launcher._prepare_onejev(tmp_path, patch, Log())
    assert all("setup.sh" not in " ".join(call[0]) for call in calls)


def test_run_failure_is_actionable(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher.subprocess, "run",
                        lambda *args, **kwargs: completed(["false"], 7, "bad output\n"))
    with pytest.raises(launcher.SetupError, match="command failed with exit 7"):
        launcher._run(["false"], Log(), cwd=tmp_path)
    monkeypatch.setattr(launcher.subprocess, "run",
                        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("missing")))
    with pytest.raises(launcher.SetupError, match="could not execute"):
        launcher._run(["missing"], Log(), cwd=tmp_path)


def test_prepare_onejev_refuses_unowned_dirty_same_origin_checkout(monkeypatch, tmp_path):
    repo = tmp_path / launcher.QEV_ONEJEV_REL
    (repo / ".git").mkdir(parents=True)
    patch = tmp_path / "serve.patch"
    patch.write_bytes(b"patch")
    monkeypatch.setattr(launcher, "_git_blob", lambda path: launcher.ONEJEV_PATCH_BLOB)
    monkeypatch.setattr(launcher, "_onejev_ready", lambda *args, **kwargs: False)
    calls = fake_run(monkeypatch, lambda argv, kwargs: completed(
        argv, 0, "https://github.com/OmniJev/OneJev.git\n"))
    with pytest.raises(launcher.SetupError, match="refusing to reset"):
        launcher._prepare_onejev(tmp_path, patch, Log())
    commands = [" ".join(call[0]) for call in calls]
    assert not any("checkout" in command or "clean" in command or "apply" in command
                   for command in commands)


def test_prepare_onejev_resets_only_launcher_owned_checkout(monkeypatch, tmp_path):
    repo = tmp_path / launcher.QEV_ONEJEV_REL
    (repo / ".git").mkdir(parents=True)
    directory = repo.parent
    (directory / ".launcher-onejev-owned").write_text("launcher_setup: OneJev\n", encoding="utf-8")
    patch = tmp_path / "serve.patch"
    patch.write_bytes(b"patch")
    monkeypatch.setattr(launcher, "_git_blob", lambda path: launcher.ONEJEV_PATCH_BLOB)
    states = iter([False, True])
    monkeypatch.setattr(launcher, "_onejev_ready", lambda *args, **kwargs: next(states))

    def run(argv, log, **kwargs):
        if "get-url" in argv:
            return completed(argv, 0, "https://github.com/OmniJev/OneJev.git\n")
        return completed(argv, 0)

    calls = fake_run(monkeypatch, lambda argv, kwargs: run(argv, None, **kwargs))
    launcher._prepare_onejev(tmp_path, patch, Log())
    commands = [" ".join(call[0]) for call in calls]
    assert any("checkout" in command for command in commands)
    assert any("clean" in command for command in commands)
    assert any(" apply " in command for command in commands)
    assert not any("setup.sh" in command for command in commands)


def test_qev_editable_install_checks_deps_and_reuses_receipt(monkeypatch, tmp_path):
    spec = qev_spec()
    snapshot = tmp_path / spec["local_snapshot"]
    snapshot.mkdir(parents=True)
    (snapshot / "serve.patch").write_bytes(b"patch")
    requirements = tmp_path / "qev_setup/launcher-requirements.txt"
    requirements.parent.mkdir(parents=True)
    requirements.write_text("torch==2.10.0+cu128\n", encoding="utf-8")
    monkeypatch.setattr(launcher, "_git_blob", lambda path: launcher.ONEJEV_PATCH_BLOB)
    monkeypatch.setattr(launcher, "_prepare_onejev", lambda *args, **kwargs: None)
    monkeypatch.setattr(launcher, "_which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(launcher, "_probe", lambda *args, **kwargs: {
        "qev": "0.1.0", "readout": "qev.readout", "engine": "DecisionEngine"})
    calls = fake_run(monkeypatch, lambda argv, kwargs: completed(argv))
    launcher._ensure_qev_source(tmp_path, [("joey1", spec)], Log())
    installs = [call for call in calls if "install" in call[0]]
    assert len(installs) == 1
    assert any("check" in call[0] for call in calls)
    launcher._ensure_qev_source(tmp_path, [("joey1", spec)], Log())
    assert len([call for call in calls if "install" in call[0]]) == 1


def test_frozen_launcher_plan_paths_match_bootstrap_contract():
    import launcher_plan

    assert launcher_plan.VLLM_PYTHON == launcher.VLLM_PYTHON_REL
    assert launcher_plan.QEV_PYTHON == launcher.QEV_PYTHON_REL
    assert launcher_plan.QEV_ONEJEV == launcher.QEV_ONEJEV_REL


def test_client_repairs_only_launcher_owned_partial_venv(monkeypatch, tmp_path):
    root = tmp_path
    write_lock(root)
    (root / ".venv/bin").mkdir(parents=True)
    (root / ".venv/pyvenv.cfg").write_text("home = /missing\n", encoding="utf-8")
    monkeypatch.setattr(launcher, "_ensure_python", lambda root, log: None)
    monkeypatch.setattr(launcher, "_python_version", lambda root, python, log: launcher.SUPPORTED_PYTHON)
    monkeypatch.setattr(launcher, "_probe", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(launcher, "_which", lambda name: f"/usr/bin/{name}")
    with pytest.raises(launcher.SetupError, match="refusing to overwrite"):
        launcher._ensure_client(root, Log())
    (root / ".scratch/launcher").mkdir(parents=True)
    (root / ".scratch/launcher/client-owned").write_text("launcher_setup: client\n", encoding="utf-8")
    calls = fake_run(monkeypatch, lambda argv, kwargs: completed(argv))
    launcher._ensure_client(root, Log())
    assert any("venv" in call[0] for call in calls)
