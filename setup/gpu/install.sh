#!/usr/bin/env bash
# Explicit coordinator-owned installation; not run by configuration validation.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "$0")/../.." && pwd)
GPU_ROOT=${GPU_ROOT:-"$ROOT/../inspect_pilot_study/.scratch/gpu"}
VENV=$GPU_ROOT/venv
if [[ ${1:-} != "--install-vllm" ]]; then
    echo "Refusing implicit installation. After coordinator handoff: $0 --install-vllm" >&2
    exit 2
fi
command -v uv >/dev/null || { echo "uv is required; no system installation will be attempted" >&2; exit 1; }
mkdir -p "$ROOT/.scratch/gpu/uv-cache" "$ROOT/.scratch/gpu/hf" "$ROOT/.scratch/gpu/cache"
export UV_CACHE_DIR="$ROOT/.scratch/gpu/uv-cache"
export HF_HOME="$ROOT/.scratch/gpu/hf"
export XDG_CACHE_HOME="$ROOT/.scratch/gpu/cache"
export UV_PYTHON_DOWNLOADS=never
if [[ -e $VENV/bin/python ]]; then
    [[ $($VENV/bin/python -c 'import platform; print(platform.python_version())') == 3.12.13 ]] || {
        echo "Existing $VENV is not Python 3.12.13; refusing to replace it" >&2; exit 1; }
else
    uv venv --python 3.12.13 "$VENV"
fi
uv pip install --python "$VENV/bin/python" -r "$ROOT/setup/gpu/requirements.txt"
"$VENV/bin/python" - <<'PY'
from importlib.metadata import version
import platform
assert platform.python_version() == "3.12.13", platform.python_version()
assert version("vllm") == "0.19.0", version("vllm")
assert version("torch") == "2.10.0", version("torch")
print("Pinned vLLM environment verified without importing vLLM, torch, or initializing CUDA.")
PY
