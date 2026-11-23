# QEV decision-model adapter: pinned setup and evidence

## Scope

`adapters/qev_local.py` supports the Baseten decision models through their
**reference OneJev QEV runtime plus the checkpoint's `serve.patch`**. It does not
load them as generative Qwen models, sample tokens, tune prompts on TweetEval, or
silently quantize/offload. `qev_worker.py` isolates QEV's runtime monkey patch and
heavy dependencies in a dedicated, offline subprocess with a bounded lifecycle.

The default mode supports only one independent tweet per request. QEV's fork
handles question branches over **one state**, not multiple independent states.
The active protocol ("engine-session-v1") opts every pinned QEV sibling
(Joey-1, Elephant-1, Mammoth-1) into `params.request_mode="packed"` at widths 1,
16, and 64. It puts the group's tweets in one request, using a numbered list
and one three-option question per tweet. Singleton requests keep the baseline
format. This is shared context, not independent batching: other tweets can
affect each answer. The model may evaluate the branches in multiple chunks. The
historical ten-condition sweep packed Joey only, at widths 1, 4, 16, 64, and
256.

## Exact model-specific mapping

The following values came from the Hugging Face model API with `?blobs=true` and
small metadata files fetched read-only on 2026-11-23. No weights were downloaded
for this inspection.

| Model | Exact model revision | BF16 parameters | Weight file bytes | `config.json` git blob |
|---|---|---:|---:|---|
| `baselabs/joey-1` | `db5c2a631f2da93c4173378438d8246966e25159` | 2,721,801,024 | 5,443,677,680 | `5878a46a4364f966cc6e877a4f3f3f80d6230be9` |
| `baselabs/elephant-1` | `2e80345f1e41b2c1eaff8ae2538c0ac657210c44` | 9,409,813,744 | 18,819,722,392 | `98295053fab64c0a71b1c2bbba06c2e31ed33cfb` |
| `baselabs/mammoth-1` | `858aab610d4d124315ebf72d2f36c5e7341f758f` | 27,356,728,560 | 49,825,162,976 + 4,888,445,168 = 54,713,608,144 | `7c80d74dfe54ed2de25007a979c06743dbc29518` |

These are decimal weight-byte sizes, not measured peak VRAM. Actual runtime
memory needs a runtime fit check under the README readiness rule even when the
weight-only envelope fits; Mammoth in particular. Weight-byte preflight alone
never authorizes a measured run, and the packed widths are new fit/sampling
conditions rather than inherited evidence.

All three cards and all three identical `setup.sh` files specify:

- base image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`, one CUDA GPU;
- OneJev commit `81ce62f1597c91e46767d4d02d6ac2e18534fe94`;
- the same `serve.patch`, git blob `7a432e8d254a9838b04ca3a155547ec3b3c70e78`;
- `setup.sh` git blob `1d00a2676f8f1da5d25295bf727a85491ff95d51`;
- `readout.json` git blob `444b7bf8313d43415e97988a79740a93154fdd3b`;
- maximum trained sequence length 16,384 tokens; over-limit requests must be
  refused rather than truncated;
- BF16 model execution, with the reference engine's default FP32 output head.

All three pinned `config.json` files declare:

```json
"modeling_overrides": {
  "attention": {
    "full_attention": "block_bidirectional",
    "linear_attention": "causal"
  },
  "readout": "slot_lm",
  "head_dim": 256,
  "n_slots": 256
}
```

Their `readout.json` files also declare `slot_lm`. Thus elephant and mammoth use
the same runtime commit/patch as Joey but their own exact model revisions,
configs, and weight layouts above; the adapter does not substitute Joey's setup
or revision for either sibling.

## Upstream evidence and why local snapshots are mandatory

Pinned source URLs:

- Model cards: <https://huggingface.co/baselabs/joey-1/blob/db5c2a631f2da93c4173378438d8246966e25159/README.md>,
  <https://huggingface.co/baselabs/elephant-1/blob/2e80345f1e41b2c1eaff8ae2538c0ac657210c44/README.md>,
  <https://huggingface.co/baselabs/mammoth-1/blob/858aab610d4d124315ebf72d2f36c5e7341f758f/README.md>.
- Reference installer: each pinned model's `setup.sh`, `constraints.txt`, and
  `serve.patch`. The installer itself names the full OneJev commit.
- Checkpoint wrapper: <https://huggingface.co/baselabs/joey-1/blob/db5c2a631f2da93c4173378438d8246966e25159/engine.py>
  (identical git blob in elephant/mammoth), which constructs
  `qev.engine.DecisionEngine` and reports attention/readout provenance.
- Unpatched upstream engine/schema/prompt:
  <https://github.com/OmniJev/OneJev/tree/81ce62f1597c91e46767d4d02d6ac2e18534fe94/qev>.
- QEV dependency baseline at that commit:
  <https://github.com/OmniJev/OneJev/blob/81ce62f1597c91e46767d4d02d6ac2e18534fe94/pyproject.toml>
  (`qev` 0.1.0, Python >=3.10, Transformers >=5.0, torch extra >=2.6).
  The checkpoint constraints are stricter and pin Transformers 5.19.0.

The patch adds `qev/readout.py` and wires it into `DecisionEngine`. For
noncausal checkpoints, full-attention layers attend bidirectionally within the
shared prefix block and within the question suffix block, while linear-attention
layers stay causal. `slot_lm` reads the slot letters' LM-head logits; no text is
generated. Upstream `decide(..., debias=1)` makes one option ordering; values
above one average cyclic option-order shifts.

The patched loader uses `ReadoutConfig.load(model_path)` **only when
`Path(model_path).exists()`**. Otherwise it constructs default causal `slot_lm`
configuration. Consequently a Hub repo ID can silently serve these weights with
the wrong attention. This adapter requires `spec['local_snapshot']`, sets
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, verifies small-file git blobs
and weight sizes, and never falls back to a Hub download.

## Runtime setup

Dependency installation, compilation, model loading, smoke inference, and GPU
benchmarking require exclusive installation/runtime ownership and a quiet host.
Use the pinned checkpoint's own installer in the documented image (or an
equivalently pinned isolated environment):

```bash
# SNAPSHOT is the fully downloaded, absolute local model snapshot.
# RUNTIME_ROOT is a dedicated writable work directory.
bash "$SNAPSHOT/setup.sh" "$RUNTIME_ROOT"
# Patched source: "$RUNTIME_ROOT/OneJev"
# Use the Python executable from that same isolated environment/image.
```

The installer clones OneJev at the exact commit, applies the model snapshot's
patch, and installs the card's constrained runtime. It also installs the
Decision Index kit even though this benchmark does not use it; do not replace
this with an arbitrary newer OneJev or system Transformers installation. Keep
the base image's CUDA PyTorch build as the installer directs.

At adapter setup, the worker additionally checks:

1. OneJev `HEAD` equals the exact commit.
2. The snapshot patch has the expected git blob and passes
   `git apply --reverse --check` against the runtime worktree.
3. Instantiated `engine.info.extra` reports `attention=noncausal` and
   `readout=slot_lm`; the attached readout config agrees and has no learned head.
4. The noncausal modeling hook is installed, attention implementation is
   `sdpa`, `eager`, or `flex_attention`, and answer slot IDs are present/unique.
5. A fixed smoke request (`"The service was excellent."`) returns one of the
   three labels. This load/smoke work occurs in setup, outside measured
   inference windows, and only with exclusive GPU ownership.

Worker stdout/stderr is retained in the attempt's `qev_worker.log`. Close uses a
bounded graceful stop followed by SIGTERM/SIGKILL only for the worker's own
process group. Request/init timeouts are bounded; a failed prediction closes the
worker rather than leaving a GPU process behind.

## Adapter specification and pinned sentiment choice

Minimal Joey example:

```python
spec = {
    "adapter": "qev",
    "repo": "baselabs/joey-1",
    "revision": "db5c2a631f2da93c4173378438d8246966e25159",
    "local_snapshot": "/absolute/path/to/joey-1",
    "params": {
        "onejev_path": "/absolute/path/to/runtime/OneJev",
        "python": "/absolute/path/to/isolated/bin/python",
        "device": "cuda:0",
        "dtype": "bfloat16",
        "head_dtype": "float32",
        "fork_mode": "auto",
        "debias": 1,
        "debug": False,
    },
}
```

Use the corresponding table row for elephant/mammoth; any other revision is
rejected. `onejev_path` may instead come from `QEV_ONEJEV_PATH`. Optional engine
arguments are `fork_tolerance`, `fork_token_budget`, `max_branch_tokens`, and
`max_request_tokens`. Lifecycle controls are `init_timeout_s` (default 1800),
`request_timeout_s` (300), and `close_timeout_s` (10). Defaults otherwise follow
the pinned reference engine; changing dtype/device/limits is explicit manifest
metadata, never an automatic fit strategy.

The fixed request is the raw tweet as `state` and one structured choice:

```text
Question: Which sentiment best describes the tweet?
negative: The tweet expresses negative sentiment.
neutral:  The tweet expresses neutral sentiment or no clear positive or negative sentiment.
positive: The tweet expresses positive sentiment.
```

Python insertion order pins option order to `negative`, `neutral`, `positive`
(slots A, B, C). `debias=1` is the pinned default; any cyclic-shift debiasing is
an explicit separate condition. Predictions return the QEV choice, calibrated
probabilities, confidence, token usage, fork/prompt diagnostics, and independent
request wall latency. Invalid choices map to `label=None` without inventing a
fourth benchmark class.

## Runtime acceptance status

The Joey installation and GPU smoke passed; see
[`RUNTIME_SMOKE.md`](RUNTIME_SMOKE.md) for exact versions, paths, outputs, and
cleanup evidence. That fallback uses torch 2.10.0+cu128 and Transformers'
native convolution fallback rather than the card image's PyTorch 2.8 stack and
optional causal-conv1d extension. This complete-system difference must remain
explicit in run metadata.

Before measured inference, perform the normal warmup and run preflight. Peak GPU
memory was not separately sampled during the smoke. Under the active protocol the
packed condition checks every distinct test and warmup request for every packed
sibling using the pinned tokenizer and renderer before inference. It records
longest branch and total shared-prefix-plus-suffix token counts in
`qev_context.json` and refuses over-limit inputs without truncation. An
over-context group skips only that width's loaded-idle/inference pair inside the
engine session; the session's other widths continue. This gate does not certify
that context-fitting requests fit GPU memory.
