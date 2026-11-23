#!/usr/bin/env python3
"""Metadata-only snapshot verification and pre-download VRAM/disk guards.

This utility never initializes CUDA or downloads a model. The fetch shell script
calls ``preflight`` immediately before its explicit ``hf download`` command.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "setup" / "model_manifest.json"
MIB = 1024 ** 2


def entry(model_id):
    with MANIFEST.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    try:
        model = manifest["models"][model_id]
    except KeyError:
        raise SystemExit(f"unknown model {model_id!r}; choose from {sorted(manifest['models'])}") from None
    model["local_snapshot"] = ROOT / model["local_snapshot"]
    model["reserve_mib"] = 3500 if model["adapter"] == "vllm" else 2500
    return model


def show(model_id):
    model = entry(model_id)
    print(f"{model['repo']}\t{model['revision']}\t{model['local_snapshot']}")


def verify(model_id):
    model = entry(model_id)
    snapshot = model["local_snapshot"]
    missing = []
    if not (snapshot / "config.json").is_file():
        missing.append("config.json")
    for name, expected in model["weight_shards_bytes"].items():
        path = snapshot / name
        try:
            actual = path.stat().st_size
        except OSError:
            missing.append(name)
        else:
            if actual != expected:
                missing.append(f"{name} ({actual} != {expected} bytes)")
    if missing:
        print(f"incomplete {model_id} snapshot {snapshot}: {', '.join(missing)}", file=sys.stderr)
        return 1
    print(f"verified {model_id} {model['revision']} at {snapshot}")
    return 0


def nvidia(*args):
    try:
        result = subprocess.run(
            ["nvidia-smi", *args], check=True, capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"VRAM preflight requires readable nvidia-smi: {exc}") from None
    return result.stdout.strip()


def preflight(model_id):
    model = entry(model_id)
    weights = sum(model["weight_shards_bytes"].values())
    required = weights + model["reserve_mib"] * MIB
    rows = nvidia(
        "--query-gpu=index,memory.total,memory.free", "--format=csv,noheader,nounits",
    ).splitlines()
    try:
        parsed = [tuple(int(value.strip()) for value in row.split(",")) for row in rows]
        index, total, free = next(row for row in parsed if row[0] == 0)
    except (ValueError, StopIteration):
        raise SystemExit("could not identify GPU index 0 and its total/free MiB") from None
    if len(parsed) != 1:
        raise SystemExit("this pinned preflight supports exactly one visible GPU")
    apps = nvidia("--query-compute-apps=pid", "--format=csv,noheader,nounits")
    if apps:
        raise SystemExit("active GPU compute application; refusing model download/preflight")
    if weights > total * MIB:
        raise SystemExit(f"{model_id}: weight bytes alone exceed GPU total; no download")
    if required > free * MIB:
        raise SystemExit(
            f"{model_id}: weights + reserve require {required / MIB:.0f} MiB, "
            f"but only {free} MiB are free; no download"
        )
    snapshot = model["local_snapshot"]
    parent = snapshot.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    available = shutil.disk_usage(parent).free
    if available < weights + weights // 20:
        raise SystemExit(f"{model_id}: insufficient disk space for weights plus 5% margin")
    print(
        f"preflight estimate passed for {model_id}: weights={weights} bytes, "
        f"reserve={model['reserve_mib']} MiB, free={free} MiB, disk_free={available} bytes; "
        "this is not evidence of runtime fit"
    )
    return 0


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in {"show", "verify", "preflight"}:
        raise SystemExit(f"usage: {sys.argv[0]} show|verify|preflight MODEL_ID")
    action, model_id = sys.argv[1:]
    if action == "show":
        show(model_id)
        return 0
    return verify(model_id) if action == "verify" else preflight(model_id)


if __name__ == "__main__":
    raise SystemExit(main())
