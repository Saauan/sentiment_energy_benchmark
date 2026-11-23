# Laptop full run — preserved detailed report

> **Historical record.** This is the original 23 November 2026 report for the
> laptop `full` phase, preserved verbatim apart from relative link paths. It
> describes the completed baseline run and its engineering pilots; it is not
> the current top-level results summary. See [RESULTS.md](../../RESULTS.md) for
> the latest `gpu_widths` results and for links to the run-level CSVs in this
> directory. The named pilot phases are no longer required steps; see the
> readiness rule in [DESIGN.md](../../DESIGN.md#9-acceptance-sequence).

# TweetEval sentiment energy benchmark — laptop run

**23 November 2026.** All conditions that fit the laptop and were supported by their inference engine completed. **27 inference trials measured, 27 skipped, none failed or pending.** The full run contains 81 validated native BoAmps JSON/CodeCarbon CSV pairs: 27 inference, 27 unloaded device idle, 9 model-loaded idle, and 18 classical training trials. Another 81 plan entries were skipped, including the calibration entries associated with the 27 skipped inference trials. Nothing was submitted upstream.

- [Full run artifacts](../../artifacts/full_isolated_02/) · [Run-level CSVs](./) · [Medians and ranges CSV](inference_medians.csv) · [Rendered marimo notebook](notebook.html) · [Verification receipt](verification.json)
- [Measured full configuration](../../profiles/laptop-full-isolated-02.json) · [Design and limitations](../../DESIGN.md) · [Usage and workstation rerun](../../README.md#run-on-another-host)

## 1. Coverage and conditions

The host was a laptop with an Intel Core Ultra 9 275HX and one NVIDIA RTX PRO 3000 Blackwell Laptop GPU, 12,227 MiB total VRAM (driver 595.99.02). The fixed [TweetEval sentiment](https://github.com/cardiffnlp/tweeteval/tree/4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66/datasets/sentiment) test panel had 257 tweets: 76 negative, 129 neutral, and 52 positive. All five runnable systems saw these same test IDs in the same order. Three independently measured repetitions were planned per model and width in a seed-determined interleaving. Training used the published training split; hyperparameters and prompts were fixed rather than tuned on validation or test. Validation examples were used for warmup. Scores use the exact persisted **first measured pass per trial**, not all repeated submissions. Energy and performance use the actual full measured workload. Across 27 completed inference trials there were 4,862,697 submissions, not 4,862,697 independently sampled tweets.

| System | Completed inference widths | Submissions per trial, widths 1 / 16 | Note |
|---|---:|---:|---|
| Qwen3.5-2B / vLLM 0.19.0 | 1, 16 | 514 / 1,028 | BF16, no quantization/thinking/prefix caching; closed-loop bursts of independent requests |
| Joey-1 / patched OneJev QEV | 1 | 257 / — | Verified noncausal full attention and `slot_lm`; BF16 backbone and FP32 head |
| TF–IDF + LinearSVC | 1, 16 | 51,400 / 308,400 | Static groups, train-only baseline variant |
| TF–IDF + multinomial NB | 1, 16 | 51,400 / 308,400 | Static groups, shared feature recipe with SVM |
| fastText | 1, 16 | 449,750 / 449,750 | Static groups; trained from scratch, not the published pretrained-embedding baseline |

Joey-1 width 16 was skipped in all three repetitions: QEV can batch branches *inside* one decision, not 16 verified independent tweets. These larger models were evaluated by preflight and skipped without downloading or loading weights. Each accounts for six inference skips (two widths × three repetitions); the weight sizes alone exceed total laptop VRAM, before runtime reserve or cache:

| Checkpoint | Pinned safetensors total |
|---|---:|
| Qwen3.5-9B | 19.31 GB |
| Elephant-1 | 18.82 GB |
| Qwen3.5-27B | 55.56 GB |
| Mammoth-1 | 54.71 GB |

No quantized, offloaded, or substitute instances were run. Joey used the exact pinned source and model patch, but a local CUDA 12.8 runtime with PyTorch **2.10.0+cu128**, not the card's 2.8 image. The missing `causal-conv1d` extension fell back to native Torch convolution; optimized FLA gated-delta execution was enabled. See [the live QEV smoke](../../qev_setup/RUNTIME_SMOKE.md) and [the actual package freeze](../../qev_setup/laptop-runtime.freeze.txt).

## 2. Inference results

Values are medians across three trials. Gross energy uses CodeCarbon CPU + GPU + RAM, divided by all submitted tweets. Throughput is completed tweets per measured work-window second. p95 is per request for vLLM/QEV, and group-completion latency for static CPU groups; those are different latency units of work despite both being milliseconds. All trials had **100% valid-output coverage**. The [CSV](inference_medians.csv) has min/max, accuracy, p50, and idle-adjusted estimates.

| System | Width | Macro recall | Macro-F1 | Gross J/tweet [min–max] | Tweets/s | p95, ms |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-2B | 1 | 0.589 | 0.423 | 4.731 [4.589–4.864] | 24.35 | 43.62 |
| Qwen3.5-2B | 16 | 0.589 | 0.424 | 0.766 [0.759–0.772] | 147.70 | 113.90 |
| Joey-1 | 1 | 0.646 | 0.647 | 5.582 [5.582–5.708] | 20.23 | 53.36 |
| TF–IDF + LinearSVC | 1 | 0.591 | 0.587 | 0.00885 [0.00843–0.00887] | 4,055 | 0.254 |
| TF–IDF + LinearSVC | 16 | 0.591 | 0.587 | 0.00104 [0.00101–0.00105] | 35,150 | 0.424 |
| TF–IDF + multinomial NB | 1 | 0.493 | 0.432 | 0.02018 [0.01996–0.02036] | 1,772 | 0.588 |
| TF–IDF + multinomial NB | 16 | 0.493 | 0.432 | 0.00173 [0.00171–0.00178] | 20,185 | 0.750 |
| fastText | 1 | 0.540 | 0.546 | 0.000225 [0.000212–0.000235] | 157,376 | 0.00580 |
| fastText | 16 | 0.540 | 0.546 | 0.000184 [0.000175–0.000186] | 195,339 | 0.0607 |

Within Qwen3.5-2B, width 16 had about **6.2× lower gross energy per tweet** and **6.1× higher throughput**, but about **2.6× higher p95 per-request latency**. The two widths had the same median macro recall to three decimals. Width 16 had 2–5 disagreements with its own first-pass predictions across the three later passes of each run; deterministic request parameters do not guarantee identical results under concurrency. Across the runnable width-1 systems, Joey had the highest observed macro recall on this panel, not an established advantage across datasets or hardware. The classical pipelines used less gross energy per submission on this host and workload; the model, engine, formatting, work unit, and hardware-use differences do not identify an architecture-only effect.

The 27 unloaded idle windows gave a median gross component estimate of **35.97 W** [35.01–39.33 W]; no model requests, fitting, or scoring happened in the idle callback. With a compatible model-loaded baseline, median adjusted J/tweet was 2.900 (Qwen width 1), 0.463 (Qwen width 16), and 3.448 (Joey width 1). There was no loaded-idle adjustment for CPU inference. Classical device-idle-adjusted values were near zero and sometimes negative: the difference is sensitive to the baseline and the CPU estimation floor and does **not** resolve the workload's incremental energy. Signed values remain in the artifacts. GPU 0 remained powered and metered for CPU trials.

## 3. Pilot corrections, restart, and verification

1. **[GPU engineering pilot](../../artifacts/gpu_pilot/):** an initial idle attempt failed validation because CodeCarbon initializes hardware discovery lazily; initialization was moved before the measurement window and the retry passed. Qwen and Joey then completed native exports and first-pass scoring. A real SIGTERM during Qwen width-1 inference left an incomplete attempt; resume wrote a fresh attempt, and invoking the runner again left completed bytes unchanged. The width-16 GPU pilot was too short for scheduled samples and is not used for quantitative inference comparisons.
2. **Classical pilots:** the [1 s in-process pilot](../../artifacts/cpu_pilot/) and [0.1 s in-process rescue](../../artifacts/cpu_pilot_resized/) produced schema-valid native reports but exposed missed scheduler ticks and under-integration. In the worst rescue trial (SVM width 1), only 2 scheduled samples arrived in 14.2 s and the fixed RAM estimate integrated about 3.5% of the expected energy. Scheduled count alone was insufficient: both rescue fastText windows exceeded five ticks but still showed poor RAM integration. The exact cause of the stalls was not isolated. These profiles are engineering evidence, not primary energy data.
3. **[Corrected CPU pilot](../../artifacts/cpu_pilot_isolated_02/) and full run:** the CodeCarbon scheduler moved to its own interpreter; the final cadence was 0.2 s for all models. The corrected CPU pilot passed both gates in all six conditions (10–82 scheduled samples, 0.949–0.972 fixed-RAM coverage). The full run passed them in all 27 inference trials (11–145 samples, coverage 0.955–0.976, max observed inter-tick gap 0.2113 s). Coverage here is `3,600,000 × RAM kWh / (reported RAM W × native seconds)`, only a consistency check for the constant 20 W RAM estimate. The full run also stopped at a time budget during one idle calibration and resumed under the unchanged plan; the incomplete attempt remains recorded.
4. **Independent checks:** 104 offline tests and 9 subtests passed. The final `verify` command re-audited all 81 native pairs against five hash-checked Draft-04 schemas pinned at BoAmps `af1680c56ce2cefd5777fcac1c57591b685d6be5`, agreeing on unique native IDs, duration, component and total energy, task metadata, and counts. It also rehashed and rescored predictions, and recomputed adjustments. Native `powerConsumption` is retained in kWh; J/tweet is derived, not written back to the native report. No active meter, benchmark, serving or GPU compute process remained at handoff.

## 4. Limits and the planned workstation run

This was a single-host study, not controlled wall-power measurement. CodeCarbon used a machine CPU-load/TDP estimate (96 W nominal TDP, observed CPU estimate near the 9.6 W model floor on many CPU runs), a RAM model, and NVIDIA NVML on this laptop. Native schema/consistency checks and stable ticks do not establish absolute energy accuracy or eliminate CodeCarbon overhead. The system remained a laptop with desktop activity possible; clocks and thermals were observed, not locked. Model loading and warmup are excluded from steady-state energy. The separate training trials should not be pooled with inference, and prior pretraining energy is outside the measured boundary. The 257-item panel, three runs per condition, and possible pretrained-model exposure to TweetEval limit generalization.

On the planned better machine, use a [fresh hardware profile and artifact root](../../README.md#run-on-another-host). Re-run the pilots and fit gates; reuse the pinned panel and comparable engine settings when possible. Pay particular attention to the metered GPU IDs, CPU/RAM backends, idle conditions, and sufficient inference-window lengths on faster hardware. Tests of the larger model pairs remain open until that run. Before any BoAmps contribution, review identifiers, dataset terms, and the native hardware fields against observed inventory; passing the schema does not test downstream ingestion or publication readiness.
