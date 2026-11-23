#!/usr/bin/env python3
"""Dedicated-process QEV worker; launched only by adapters.qev_local.

Protocol: one JSON object per line on inherited request/response file descriptors.
QEV, torch, and Transformers are imported only for the init operation. All model
loads use a pinned local snapshot with Hub/Transformers offline mode set by the
parent. Unexpected diagnostics are written to the worker log, not the protocol.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any

QUESTION_ID = "sentiment"
QUESTION = {
    "type": "choice",
    "instructions": "Which sentiment best describes the tweet?",
    "criteria": {
        "negative": "The tweet expresses negative sentiment.",
        "neutral": "The tweet expresses neutral sentiment or no clear positive or negative sentiment.",
        "positive": "The tweet expresses positive sentiment.",
    },
}
LABELS = frozenset(QUESTION["criteria"])
SMOKE_TEXT = "The service was excellent."


def _run_git(onejev: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(onejev), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=15,
    )
    return completed.stdout.strip()


def _check_runtime(onejev: Path, expected_revision: str, patch: Path, patch_blob: str) -> dict[str, Any]:
    head = _run_git(onejev, "rev-parse", "HEAD")
    if head != expected_revision:
        raise RuntimeError(f"OneJev HEAD is {head}, expected {expected_revision}")
    actual_blob = _run_git(onejev, "hash-object", str(patch))
    if actual_blob != patch_blob:
        raise RuntimeError(f"serve.patch has git blob {actual_blob}, expected {patch_blob}")
    # Reverse application succeeds only when the exact patch is present in the
    # worktree. This guards against using unpatched upstream QEV, whose default
    # ReadoutConfig silently falls back to causal attention for Hub-like loads.
    _run_git(onejev, "apply", "--reverse", "--check", str(patch))
    return {"head": head, "serve_patch_git_blob": actual_blob, "serve_patch_reverse_check": True}


def _verify_engine(engine: Any) -> dict[str, Any]:
    info = asdict(engine.info)
    extra = info.get("extra") or {}
    if extra.get("attention") != "noncausal":
        raise RuntimeError(f"QEV did not activate noncausal attention: {extra!r}")
    if extra.get("readout") != "slot_lm":
        raise RuntimeError(f"QEV did not activate slot_lm readout: {extra!r}")
    readout = getattr(engine, "readout", None)
    config = getattr(readout, "config", None)
    if config is None or not config.noncausal or config.readout != "slot_lm":
        raise RuntimeError("instantiated QEV readout configuration does not match noncausal slot_lm")
    if getattr(readout, "head", None) is not None:
        raise RuntimeError("slot_lm checkpoint unexpectedly has a learned readout head")
    module = importlib.import_module(type(engine.model).__module__)
    if not getattr(module, "_qev_noncausal", False):
        raise RuntimeError("patched block-bidirectional attention hook is not installed")
    implementation = getattr(engine.model.config, "_attn_implementation", None)
    if implementation not in {"sdpa", "eager", "flex_attention"}:
        raise RuntimeError(f"block-bidirectional attention requires sdpa/eager/flex_attention, got {implementation}")
    if len(engine.slot_ids) < 3 or len(set(engine.slot_ids)) != len(engine.slot_ids):
        raise RuntimeError("QEV answer-slot verification failed")
    return info


def _decide(engine: Any, text: str, debias: int, debug: bool) -> dict[str, Any]:
    from qev.schema import SystemOneRequest

    request = SystemOneRequest(state=text, questions={QUESTION_ID: QUESTION})
    response, runtime = engine.decide(request, debias=debias, debug=debug)
    answer = response.answers[QUESTION_ID]
    result = {
        "answer": answer.model_dump(mode="json"),
        "usage": response.usage.model_dump(mode="json"),
        "runtime": runtime,
    }
    # Reject numpy scalars, arbitrary objects, NaN, and Infinity at the worker
    # boundary so every adapter raw result is safe for the JSON run artifacts.
    json.dumps(result, allow_nan=False)
    return result


class Worker:
    def __init__(self) -> None:
        self.engine = None

    def init(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.engine is not None:
            raise RuntimeError("QEV engine is already initialized")
        onejev = Path(payload["onejev_path"]).resolve()
        runtime_evidence = _check_runtime(
            onejev,
            payload["onejev_revision"],
            Path(payload["patch_path"]),
            payload["patch_blob"],
        )
        sys.path.insert(0, str(onejev))
        from qev.engine import DecisionEngine

        engine = DecisionEngine(payload["model_path"], **payload["engine_options"])
        try:
            info = _verify_engine(engine)
            smoke = _decide(engine, SMOKE_TEXT, debias=1, debug=True)
            if smoke["answer"].get("choice") not in LABELS:
                raise RuntimeError(f"invalid smoke-test choice: {smoke['answer']!r}")
        except Exception:
            del engine
            raise
        self.engine = engine
        return {
            "runtime_source": runtime_evidence,
            "engine_info": info,
            "question": QUESTION,
            "smoke_text": SMOKE_TEXT,
            "smoke_answer": smoke["answer"],
            "generative_decoding": False,
        }

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.engine is None:
            raise RuntimeError("QEV engine is not initialized")
        return _decide(self.engine, payload["text"], payload["debias"], payload["debug"])

    def close(self) -> None:
        self.engine = None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-fd", type=int, required=True)
    parser.add_argument("--response-fd", type=int, required=True)
    args = parser.parse_args()
    worker = Worker()
    with os.fdopen(args.request_fd, "r", encoding="utf-8") as requests, os.fdopen(
        args.response_fd, "w", encoding="utf-8"
    ) as responses:
        for line in requests:
            try:
                message = json.loads(line)
                operation = message["operation"]
                payload = message.get("payload") or {}
                if operation == "init":
                    result = worker.init(payload)
                elif operation == "predict":
                    result = worker.predict(payload)
                elif operation == "close":
                    worker.close()
                    responses.write(json.dumps({"ok": True, "result": None}) + "\n")
                    responses.flush()
                    break
                else:
                    raise ValueError(f"unknown worker operation {operation!r}")
                response = {"ok": True, "result": result}
            except Exception as exc:
                traceback.print_exc()
                response = {"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}}
            responses.write(json.dumps(response, allow_nan=False) + "\n")
            responses.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
