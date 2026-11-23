#!/usr/bin/env bash
# Explicit, per-model weight fetch. No arguments means no network or filesystem changes.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT"
PYTHON=${PYTHON:-"$ROOT/.venv/bin/python"}
if [[ $# -eq 0 ]]; then
    echo "No models requested. Usage: $0 qwen2b [joey1 ...]" >&2
    echo "Models above 8 GiB also require ALLOW_LARGE_MODEL_DOWNLOAD=YES and a successful VRAM/disk preflight." >&2
    exit 2
fi
command -v hf >/dev/null || { echo "hf CLI is required; no installation will be attempted" >&2; exit 1; }
[[ -x $PYTHON ]] || { echo "Pinned base interpreter missing: $PYTHON" >&2; exit 1; }
for model_id in "$@"; do
    if "$PYTHON" setup/gpu/model_io.py verify "$model_id" 2>/dev/null; then
        continue
    fi
    weight_bytes=$("$PYTHON" - "$model_id" <<'PY'
import json, sys
with open("setup/model_manifest.json", encoding="utf-8") as stream:
    model = json.load(stream)["models"][sys.argv[1]]
print(sum(model["weight_shards_bytes"].values()))
PY
)
    if (( weight_bytes > 8 * 1000 * 1000 * 1000 )) && [[ ${ALLOW_LARGE_MODEL_DOWNLOAD:-} != YES ]]; then
        echo "$model_id exceeds 8 GB; refusing without ALLOW_LARGE_MODEL_DOWNLOAD=YES" >&2
        exit 1
    fi
    "$PYTHON" setup/gpu/model_io.py preflight "$model_id"
    mapfile -t pin < <("$PYTHON" setup/gpu/model_io.py show "$model_id")
    IFS=$'\t' read -r repo revision destination <<< "${pin[0]}"
    echo "Explicitly fetching $repo@$revision into $destination"
    hf download "$repo" --revision "$revision" --local-dir "$destination"
    "$PYTHON" setup/gpu/model_io.py verify "$model_id"
done
