# Pinned GPU model and vLLM setup

These are **explicit setup instructions**, not actions performed by
`configuration.configure`. Configuration validation reads local JSON only and
never imports an engine, creates a CUDA context, installs packages, or downloads
weights. The benchmark does not use Inspect.

## Environments

- Benchmark client: project `.venv`, Python **3.12.13**, CodeCarbon **3.3.3**.
- vLLM server: `../inspect_pilot_study/.scratch/gpu/venv`, Python **3.12.13**,
  vLLM **0.19.0**, PyTorch **2.10.0+cu128**. `requirements.txt` is an exact copy
  of the prior pilot's GPU pip freeze; it is a reproducibility record, not a
  minimal dependency list.
- QEV decision runtime: `.scratch/qev-runtime/venv`, with OneJev under
  `.scratch/qev-runtime/OneJev`. Its independently owned contract and validation
  evidence are in `qev_setup/README.md` and `qev_setup/RUNTIME_SMOKE.md`; do not
  replace that setup with the vLLM environment.

After obtaining exclusive installation ownership, recreate or update only the
pinned vLLM environment with:

```bash
setup/gpu/install.sh --install-vllm
```

The script refuses to run without that explicit flag, uses no `sudo`, disables
automatic Python downloads, and checks versions through `importlib.metadata`
without importing vLLM or torch. An existing interpreter at another Python
version is not replaced. Set `GPU_ROOT` only when deliberately staging an
equivalent environment elsewhere; an executed configuration must then be
updated before any run, never retroactively.

## Exact model snapshots

`setup/model_manifest.json` pins the repository, full commit, local path and
every weight-shard size in decimal bytes for the three Qwen and three decision
models. The metadata was obtained from read-only Hugging Face Hub API responses
with `blobs=true` on 2026-11-23. Local snapshots are checked for `config.json`
and exact shard sizes before loading; configuration itself does not require
snapshots to exist, so an unavailable/infeasible full-experiment model can be
recorded as skipped.

Fetch only explicitly named models:

```bash
# Small pilot pair (still explicit downloads if snapshots are absent).
setup/gpu/fetch_models.sh qwen2b joey1

# Metadata/path lookup and local verification; neither downloads anything.
.venv/bin/python setup/gpu/model_io.py show qwen2b
.venv/bin/python setup/gpu/model_io.py verify qwen2b
```

There is no default model download. Models over 8 GB additionally require
`ALLOW_LARGE_MODEL_DOWNLOAD=YES`. Every actual fetch first uses `nvidia-smi`
read-only to check GPU 0 weight bytes plus the configured reserve (vLLM 3500
MiB, QEV 2500 MiB), absence of compute applications, and disk space. Thus the
9B/27B variants are guarded against accidental download on the current 12 GB
GPU. Weight-plus-reserve arithmetic is a lower-bound preflight, **not evidence
of runtime fit**. No quantization, offload, model substitution, or Hub revision
fallback is allowed.

## vLLM condition

All Qwen variants use BF16, one GPU, a 1,024-token context, eight output tokens,
temperature zero, top-p one, seed zero, thinking disabled, at most 16 sequences,
and prefix caching disabled. Width 16 is a closed-loop burst of independent
requests, not a claim about an internal GPU batch. The prior Qwen2B snapshot is
`../inspect_pilot_study/.scratch/gpu/model`; the other snapshots are under
`.scratch/models/`.

## Joey/QEV condition and provenance

Joey decision system: local `baselabs/joey-1@db5c2a631f2da93c4173378438d8246966e25159`;
OneJev `qev` 0.1.0 at `81ce62f1597c91e46767d4d02d6ac2e18534fe94` plus
`serve.patch` blob `7a432e8d254a9838b04ca3a155547ec3b3c70e78`;
Python 3.12.13; torch 2.10.0+cu128 / torchvision 0.25.0+cu128;
Transformers 5.19.0; FLA 0.5.2; BF16 backbone, FP32 `slot_lm` head;
block-bidirectional full attention/causal linear attention; no generative
decoding. `causal-conv1d` was unavailable (no `nvcc`), so pinned Transformers
uses its native `F.conv1d` fallback. This actual torch **2.10.0+cu128** runtime
differs from the model-card image's torch **2.8.0** and must remain visible in
reports. FLA's optimized `chunk_gated_delta_rule` was enabled; `fork_mode=auto`
resolves to batched **branches of one tweet**, never independent-tweet width 16.

The Joey smoke checks closed their workers and left no compute process. Those
observations are not benchmark results, and peak memory was not separately
sampled. Normal pilot preflight and loaded-idle/warmup gates remain necessary.
Elephant and Mammoth use their own exact snapshots/revisions and are not
silently replaced by Joey's working runtime condition. See
[qev_setup/README.md](../../qev_setup/README.md) and
[qev_setup/RUNTIME_SMOKE.md](../../qev_setup/RUNTIME_SMOKE.md) for the full
acceptance evidence.

## Runtime path normalization

Stored configuration paths are deterministic relative strings. The runner
normalizes them only at execution. For `python`, QEV `params.python`, and vLLM
`engine_safety.expected_python`, use `os.path.abspath` and preserve the lexical
`venv/bin/python` symlink; `Path.resolve` can change virtual-environment
interpreter behavior. Snapshot and OneJev paths must likewise be made absolute
before adapter setup/preflight.
