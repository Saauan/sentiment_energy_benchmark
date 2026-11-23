# Joey runtime installation and GPU smoke evidence — 2026-11-23

## Result

**READY AND CLEAN** for coordinator handoff. The pinned Joey model loaded and
completed structured decisions on `cuda:0`; all workers were closed and the final
`nvidia-smi --query-compute-apps` output was empty. No CodeCarbon or benchmark
measurement window was run, and no other model weights were downloaded.

## Paths and source identity

- Snapshot: `.scratch/models/joey-1`
  (repo `baselabs/joey-1`, revision
  `db5c2a631f2da93c4173378438d8246966e25159`).
- Patched source: `.scratch/qev-runtime/OneJev`, HEAD
  `81ce62f1597c91e46767d4d02d6ac2e18534fe94`.
- Patch: snapshot `serve.patch`, git blob
  `7a432e8d254a9838b04ca3a155547ec3b3c70e78`.
  `git apply --reverse --check` passed in the adapter worker.
- Interpreter: `.scratch/qev-runtime/venv/bin/python`, Python 3.12.13.
- Smoke attempts/logs: `.scratch/qev-runtime/smoke-attempt`,
  `.scratch/qev-runtime/smoke-predict-attempt`, and the final
  `.scratch/qev-runtime/smoke-final-attempt`.
- Full machine-readable setup metadata: `.scratch/qev-runtime/smoke-metadata.json`.
- Three prediction records: `.scratch/qev-runtime/smoke-predictions.json`.
- Preserved installed package inventory: [`laptop-runtime.freeze.txt`](laptop-runtime.freeze.txt). The editable QEV entry identifies the upstream base commit; the checkpoint's exact `serve.patch` must also be applied as described above.

The dedicated venv reuses only torch/torchvision (including torch's top-level
`functorch` package), NVIDIA libraries, and Triton from the prior read-only CUDA
12.8 environment through symlinks. That prior
environment was not mutated. All QEV/Transformers/Python dependencies are local
to `.scratch/qev-runtime/venv`; `pip check` reports no broken requirements. No
sudo, Docker, or privileged operation was used.

## Resolved package/runtime versions

| Component | Resolved version |
|---|---|
| qev | 0.1.0 (editable pinned OneJev source + patch) |
| torch | package 2.10.0; runtime 2.10.0+cu128 |
| torchvision | package 0.25.0; runtime 0.25.0+cu128 |
| Transformers | 5.19.0 |
| Accelerate | 1.15.0 |
| safetensors | 0.8.0 |
| Pydantic | 2.14.1 |
| flash-linear-attention / fla-core | 0.5.2 / 0.5.2 |
| huggingface_hub / hf_xet | 2.2.0 / 1.7.0 |
| causal-conv1d | **not installed** |

Resolved engine options were `device=cuda:0`, `dtype=bfloat16`,
`head_dtype=float32`, `fork_mode=auto`, `debias=1`. Setup metadata reports
`model_type=qwen3_5_text`, prompt `qev-labels-v2`, calibration `none`, attention
`noncausal`, readout `slot_lm`, and no generative decoding. Snapshot small-file
hashes and the exact 5,443,677,680-byte weight size passed before loading.

## Deviations from the card's reference environment

1. The card names the `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` image. This
   user-space fallback instead reused torch/torchvision **2.10.0/0.25.0+cu128**
   with the same CUDA 12.8 family. Results must identify this complete runtime,
   not claim the card's exact PyTorch image.
2. The official `causal-conv1d==1.7.0` source build could not run because
   `/usr/local/cuda/bin/nvcc` is absent. Transformers 5.19.0's pinned Qwen3.5
   implementation provides native fallbacks: `causal_conv1d_fn` and
   `causal_conv1d_update` use `torch.nn.functional.conv1d`. Hub kernel loading
   was disabled by the adapter's offline environment. The successful internal
   fork comparison and predictions below verify this path structurally, but the
   convolution kernel remains a runtime difference from the card and must be
   recorded in manifests. The final worker log contains only this causal-conv
   fallback warning plus harmless Triton `_POSIX_C_SOURCE` compile warnings.
3. The unnecessary Decision Index kit was not installed. The benchmark calls
   `qev.engine.DecisionEngine` directly and does not import that kit.
4. Installation reproduced the official clone/commit/apply and constrained
   `OneJev[train]` package steps manually. The exact OneJev source and patch are
   unchanged; no model code or weights were patched by this project.

The first broad package-overlay attempt also exposed the prior vLLM dependency
constraints and was rejected by pip. It was replaced by the narrow CUDA-only
symlink overlay before the successful install. An initial smoke then showed
that FLA's optimized import also needed torch's top-level `functorch` package;
adding that small read-only symlink enabled FLA 0.5.2's optimized
`chunk_gated_delta_rule`, confirmed by its absence from the final fallback log.
No vLLM server was started.

## Actual verification and smoke outputs

Adapter setup instantiated the patched engine and verified the source HEAD,
patch, local config/readout, noncausal hook, attention implementation, slot IDs,
and fixed smoke response. The reference engine's own `fork_mode=auto` check
selected `batched` question-branch forking with:

```text
fork_max_abs_diff = 0.0023944793911302442 (tolerance 0.05)
```

This is branch batching within one request, not independent-tweet batching.
`supported_batches` remains `(1,)`.

Fixed setup smoke (`The service was excellent.`):

```json
{
  "choice": "positive",
  "probabilities": {"negative": 0.000565, "neutral": 0.004667, "positive": 0.994769},
  "confidence": 0.992153
}
```

Three subsequent calls each submitted exactly one independent tweet:

| Text | Label | P(negative) | P(neutral) | P(positive) | Adapter latency (s) |
|---|---|---:|---:|---:|---:|
| `I absolutely loved this!` | positive | 0.000486 | 0.005917 | 0.993597 | 0.046267 |
| `This was terrible.` | negative | 0.977153 | 0.022517 | 0.000330 | 0.046225 |
| `The meeting is at noon.` | neutral | 0.003533 | 0.993355 | 0.003112 | 0.046337 |

These latencies are smoke observations only, not benchmark results. No warmup,
steady-state timing, energy measurement, or quality claim is made. Actual model
load and all three classes' structured outputs succeeded on the 12,227 MiB GPU;
peak memory was not separately sampled, so preflight reserve settings remain
estimates rather than fit evidence.

An intermediate test script mistakenly passed three texts in one `predict`
call. The adapter correctly raised `Unavailable` for unsupported independent
batching and its `finally` block closed the worker. The corrected sequential run
above passed; this script error is not a model failure.

## Tests and handoff gates

- `tests/test_qev_local.py`: 6 passed under system Python and project `.venv`.
- Full project suite before runtime installation: 38 passed, 9 subtests passed.
- Real snapshot metadata validation: passed.
- Real Joey setup/fixed smoke/three sequential predictions: passed.
- Worker close/idempotent cleanup: passed in the fake-process integration test;
  real workers also closed and left no compute process.

The runtime execution owner must still perform normal warmup and measured-run
preflight, retain the runtime differences above in run metadata, and leave
independent batch 16 unsupported.
