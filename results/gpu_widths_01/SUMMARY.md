# gpu_widths — condition summary (results only)

Phase `gpu_widths` from `artifacts/gpu_widths_01`, exported to `results/gpu_widths_01`. This summary contains **only this phase's inference trials**. It does not pool with `gpu_pilot`, `cpu_pilot*`, or `full_isolated_02`, and no prior phase is used as a baseline or reference. Artifacts are read-only inputs.

## 1. Coverage and status

90 planned trial entries across 30 blocks (`device_idle` 30, `loaded_idle` 30, `inference` 30).

| kind | planned | complete | skipped | incomplete | failed | pending/running |
|---|---|---|---|---|---|---|
| device_idle | 30 | 30 | 0 | 0 | 0 | 0 |
| loaded_idle | 30 | 30 | 0 | 0 | 0 | 0 |
| inference | 30 | 30 | 0 | 0 | 0 | 0 |

No skipped, failed, incomplete, or pending trials.

## 2. Measurement bases and definitions

- Condition = model x submission width. Medians and ranges are over the **actual completed repetitions** for that exact condition (`complete_reps` and metric-specific `*_n_reps` columns); planned repetitions that were skipped are counted separately and never imputed. `rep` is the stored zero-based repetition index (0, 1, 2).
- Quality fields use the exact persisted **first measured pass** per trial (`basis_pass = 0`, `basis_n = panel size`). Repeated passes in the same trial are not independent samples and never enter quality medians.
- Gross J/tweet = CodeCarbon CPU + GPU + RAM kWh x 3,600,000 / submitted tweet decisions. Loaded-adjusted J/tweet subtracts the compatible model-loaded idle baseline for that model; device-idle-adjusted remains separate. Adjustment bases are never collapsed.
- Throughput = completed tweets / measured work-window second.
- Two latency units stay separate: `group p95` is closed-loop completion of one submission envelope (all independent requests in the width, or the packed QEV envelope); `per-request p95` is the adapter-reported per-request latency where the engine reports one. Both are in milliseconds.
- Sampling: CodeCarbon scheduled samples per window and the gate's minimum window, from each attempt's `monitoring.json` / `gate`.

## 3. Measured inference conditions

| model | width | passes | reps | macro recall | macro F1 | accuracy | gross J/tweet | loaded-adj J/tweet | tweets/s | group p95 ms | per-request p95 ms | submissions |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| joey1 | 1 | 1 | 3/3 | 0.646 [0.646–0.646] | 0.647 [0.647–0.647] | 0.654 [0.654–0.654] | 5.63729 [5.57346–5.65202] | 3.46152 [3.34335–3.51917] | 20.2 [20.1–20.3] | 53.92 [53.81–54.19] | 53.90 [53.78–54.17] | 257 |
| joey1 | 4 | 2 | 3/3 | 0.589 [0.589–0.589] | 0.599 [0.599–0.599] | 0.634 [0.634–0.634] | 1.88974 [1.85241–1.91128] | 1.14319 [1.12425–1.16945] | 60.5 [59.9–60.6] | 71.65 [70.52–72.27] | NA (see note) | 514 |
| joey1 | 16 | 4 | 3/3 | 0.582 [0.582–0.582] | 0.586 [0.586–0.586] | 0.638 [0.638–0.638] | 1.49635 [1.44820–1.51658] | 0.91232 [0.86981–0.93105] | 76.7 [76.3–76.8] | 216.62 [215.86–221.13] | NA (see note) | 1,028 |
| joey1 | 64 | 4 | 3/3 | 0.593 [0.593–0.593] | 0.598 [0.598–0.598] | 0.634 [0.634–0.634] | 1.95770 [1.95123–2.63754] | 1.10201 [1.09677–1.43967] | 51.9 [36.7–52.2] | 1,323.69 [1,323.17–2,121.73] | NA (see note) | 1,028 |
| joey1 | 256 | 4 | 3/3 | 0.578 [0.578–0.578] | 0.574 [0.574–0.574] | 0.630 [0.630–0.630] | 3.29248 [3.27559–3.88610] | 1.35612 [1.34711–1.65048] | 22.9 [19.9–23.2] | 11,315.17 [11,218.67–14,655.86] | NA (see note) | 1,028 |
| qwen2b | 1 | 1 | 3/3 | 0.589 [0.589–0.589] | 0.423 [0.423–0.423] | 0.455 [0.455–0.455] | 4.57403 [4.57352–4.80869] | 2.78795 [2.72430–2.93231] | 24.4 [24.0–25.4] | 46.23 [44.73–47.10] | 45.12 [44.08–45.87] | 257 |
| qwen2b | 4 | 1 | 3/3 | 0.595 [0.589–0.601] | 0.428 [0.424–0.432] | 0.459 [0.455–0.463] | 1.41072 [1.29754–1.46228] | 0.85304 [0.78512–0.85405] | 81.3 [73.7–87.2] | 64.67 [59.05–65.86] | 61.59 [56.41–63.21] | 257 |
| qwen2b | 16 | 4 | 3/3 | 0.595 [0.595–0.601] | 0.429 [0.427–0.431] | 0.459 [0.459–0.463] | 0.76606 [0.76283–0.78133] | 0.46098 [0.45820–0.47696] | 147.0 [145.4–147.7] | 120.02 [118.77–120.10] | 115.56 [112.98–116.71] | 1,028 |
| qwen2b | 64 | 4 | 3/3 | 0.595 [0.595–0.601] | 0.429 [0.427–0.432] | 0.459 [0.459–0.463] | 0.73069 [0.70218–0.73399] | 0.44162 [0.42249–0.44466] | 156.5 [155.0–162.3] | 434.41 [420.47–490.23] | 408.77 [399.38–429.41] | 1,028 |
| qwen2b | 256 | 4 | 3/3 | 0.595 [0.595–0.601] | 0.429 [0.428–0.431] | 0.459 [0.459–0.463] | 0.69458 [0.68690–0.71372] | 0.41585 [0.41469–0.43200] | 161.1 [159.2–165.4] | 1,588.84 [1,578.09–1,607.62] | 1,477.35 [1,443.83–1,492.70] | 1,028 |

Per-request p95 is `NA` for the packed QEV widths 4, 16, 64, 256: the engine reports one completion time for the whole packed envelope and it is copied onto every tweet row, so a row-weighted p95 would be an envelope time weighted by envelope width, not a per-request latency. Use group p95 there; row latencies remain in `trials.csv`/`performance.csv` unchanged.

Fixed panel support (same 257 test IDs in every condition): negative 76, neutral 129, positive 52. Panels and widths are not re-sampled.


Counts, validity, group/envelope counts, per-class support and the full sampling columns are in `inference_medians.csv` (which keeps unmeasured conditions too); per-trial rows are in `trials.csv`.


### Sampling adequacy

| model | width | scheduled samples min | median | max | gate minimum s | gate met (all reps) | worst tick gap s | repeat-vs-first-pass disagreements (sum) |
|---|---|---|---|---|---|---|---|---|
| joey1 | 1 | 63 | 63 | 63 | 1.000 | yes | 0.208 | 0 |
| joey1 | 4 | 42 | 42 | 42 | 1.000 | yes | 0.206 | 0 |
| joey1 | 16 | 66 | 67 | 67 | 1.000 | yes | 0.210 | 0 |
| joey1 | 64 | 98 | 99 | 139 | 1.000 | yes | 0.215 | 0 |
| joey1 | 256 | 221 | 224 | 258 | 1.000 | yes | 0.216 | 0 |
| qwen2b | 1 | 50 | 52 | 53 | 1.000 | yes | 0.220 | 0 |
| qwen2b | 4 | 14 | 15 | 17 | 1.000 | yes | 0.209 | 0 |
| qwen2b | 16 | 34 | 34 | 35 | 1.000 | yes | 0.211 | 6 |
| qwen2b | 64 | 31 | 32 | 33 | 1.000 | yes | 0.212 | 3 |
| qwen2b | 256 | 31 | 31 | 32 | 1.000 | yes | 0.206 | 8 |

## 4. Accuracy versus width 1 on the same fixed panel

Each row pairs width w repetition r with width-1 repetition r on the **same 257 fixed test IDs**, first measured pass only. `b` = only width w correct, `c` = only width 1 correct. These are descriptive paired counts over one fixed panel; **no significance test and no p-value are reported**, because the three repetitions re-measure the same panel and repeated passes within a trial are not independent. A majority-of-repetitions consensus per tweet is shown as a secondary check.

| model | width | rep | n | agree | agree rate | b (w only) | c (w1 only) | acc w | acc w1 | delta acc | delta macro recall | delta macro F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| joey1 | 4 | 0 | 257 | 198 | 0.770 | 25 | 30 | 0.634 | 0.654 | -0.019 | -0.057 | -0.048 |
| joey1 | 4 | 1 | 257 | 198 | 0.770 | 25 | 30 | 0.634 | 0.654 | -0.019 | -0.057 | -0.048 |
| joey1 | 4 | 2 | 257 | 198 | 0.770 | 25 | 30 | 0.634 | 0.654 | -0.019 | -0.057 | -0.048 |
| joey1 | 16 | 0 | 257 | 200 | 0.778 | 25 | 29 | 0.638 | 0.654 | -0.016 | -0.064 | -0.061 |
| joey1 | 16 | 1 | 257 | 200 | 0.778 | 25 | 29 | 0.638 | 0.654 | -0.016 | -0.064 | -0.061 |
| joey1 | 16 | 2 | 257 | 200 | 0.778 | 25 | 29 | 0.638 | 0.654 | -0.016 | -0.064 | -0.061 |
| joey1 | 64 | 0 | 257 | 196 | 0.763 | 25 | 30 | 0.634 | 0.654 | -0.019 | -0.054 | -0.049 |
| joey1 | 64 | 1 | 257 | 196 | 0.763 | 25 | 30 | 0.634 | 0.654 | -0.019 | -0.054 | -0.049 |
| joey1 | 64 | 2 | 257 | 196 | 0.763 | 25 | 30 | 0.634 | 0.654 | -0.019 | -0.054 | -0.049 |
| joey1 | 256 | 0 | 257 | 188 | 0.732 | 29 | 35 | 0.630 | 0.654 | -0.023 | -0.068 | -0.073 |
| joey1 | 256 | 1 | 257 | 188 | 0.732 | 29 | 35 | 0.630 | 0.654 | -0.023 | -0.068 | -0.073 |
| joey1 | 256 | 2 | 257 | 188 | 0.732 | 29 | 35 | 0.630 | 0.654 | -0.023 | -0.068 | -0.073 |
| qwen2b | 4 | 0 | 257 | 255 | 0.992 | 1 | 0 | 0.459 | 0.455 | +0.004 | +0.006 | +0.005 |
| qwen2b | 4 | 1 | 257 | 256 | 0.996 | 0 | 0 | 0.455 | 0.455 | +0.000 | +0.000 | +0.001 |
| qwen2b | 4 | 2 | 257 | 254 | 0.988 | 2 | 0 | 0.463 | 0.455 | +0.008 | +0.013 | +0.009 |
| qwen2b | 16 | 0 | 257 | 255 | 0.992 | 1 | 0 | 0.459 | 0.455 | +0.004 | +0.006 | +0.006 |
| qwen2b | 16 | 1 | 257 | 255 | 0.992 | 2 | 0 | 0.463 | 0.455 | +0.008 | +0.013 | +0.008 |
| qwen2b | 16 | 2 | 257 | 256 | 0.996 | 1 | 0 | 0.459 | 0.455 | +0.004 | +0.006 | +0.004 |
| qwen2b | 64 | 0 | 257 | 254 | 0.988 | 2 | 0 | 0.463 | 0.455 | +0.008 | +0.013 | +0.009 |
| qwen2b | 64 | 1 | 257 | 255 | 0.992 | 1 | 0 | 0.459 | 0.455 | +0.004 | +0.006 | +0.006 |
| qwen2b | 64 | 2 | 257 | 256 | 0.996 | 1 | 0 | 0.459 | 0.455 | +0.004 | +0.006 | +0.004 |
| qwen2b | 256 | 0 | 257 | 255 | 0.992 | 1 | 0 | 0.459 | 0.455 | +0.004 | +0.006 | +0.005 |
| qwen2b | 256 | 1 | 257 | 255 | 0.992 | 2 | 0 | 0.463 | 0.455 | +0.008 | +0.013 | +0.008 |
| qwen2b | 256 | 2 | 257 | 255 | 0.992 | 1 | 0 | 0.459 | 0.455 | +0.004 | +0.006 | +0.006 |

Consensus over the paired repetitions (one vote per tweet):

| model | width | reps paired | reps used | n | agree | agree rate | b | c | delta acc | delta macro recall | ties |
|---|---|---|---|---|---|---|---|---|---|---|---|
| joey1 | 4 | 0,1,2 | 3 | 257 | 198 | 0.770 | 25 | 30 | -0.019 | -0.057 | 0 |
| joey1 | 16 | 0,1,2 | 3 | 257 | 200 | 0.778 | 25 | 29 | -0.016 | -0.064 | 0 |
| joey1 | 64 | 0,1,2 | 3 | 257 | 196 | 0.763 | 25 | 30 | -0.019 | -0.054 | 0 |
| joey1 | 256 | 0,1,2 | 3 | 257 | 188 | 0.732 | 29 | 35 | -0.023 | -0.068 | 0 |
| qwen2b | 4 | 0,1,2 | 3 | 257 | 255 | 0.992 | 1 | 0 | +0.004 | +0.006 | 0 |
| qwen2b | 16 | 0,1,2 | 3 | 257 | 256 | 0.996 | 1 | 0 | +0.004 | +0.006 | 0 |
| qwen2b | 64 | 0,1,2 | 3 | 257 | 255 | 0.992 | 1 | 0 | +0.004 | +0.006 | 0 |
| qwen2b | 256 | 0,1,2 | 3 | 257 | 254 | 0.988 | 2 | 0 | +0.008 | +0.013 | 0 |

Per-tweet disagreements behind these counts are in `width1-disagreements.csv`. A disagreement here is a label change under a different submission width; it is not evidence of a better model.

Repeat behaviour on the first pass: bit-identical across repetitions for joey1 width 1, joey1 width 4, joey1 width 16, joey1 width 64, joey1 width 256, qwen2b width 1. Not identical across repetitions for qwen2b width 4, qwen2b width 16, qwen2b width 64, qwen2b width 256 (concurrent serving does not guarantee identical outputs). Identical repetitions therefore also produce identical paired counts; they add measurement repeats, not independent observations.

## 5. QEV packed envelopes and context maxima

| model | width | passes | test groups/pass | envelopes | tweet decisions | tweets/envelope | max prefix tokens | max branch tokens (test) | max branch tokens (warmup) | card branch limit | runtime branch limit | max request tokens (test) | request limit | min branch headroom (card) | min request headroom | all groups fit | no forwarding in preflight |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| joey1 | 1 | 1 | 257 | 257 | 257 | 1.000 | 91 | 164 | 163 | 16,384 | 32,768 | 164 | 65,536 | 16,220 | 65,372 | yes | yes |
| joey1 | 4 | 2 | 65 | 130 | 514 | 3.954 | 269 | 348 | 363 | 16,384 | 32,768 | 585 | 65,536 | 16,036 | 64,951 | yes | yes |
| joey1 | 16 | 4 | 17 | 68 | 1,028 | 15.118 | 801 | 881 | 940 | 16,384 | 32,768 | 2,072 | 65,536 | 15,503 | 63,464 | yes | yes |
| joey1 | 64 | 4 | 5 | 20 | 1,028 | 51.400 | 2,934 | 3,014 | 989 | 16,384 | 32,768 | 8,045 | 65,536 | 13,370 | 57,491 | yes | yes |
| joey1 | 256 | 4 | 2 | 8 | 1,028 | 128.500 | 11,620 | 11,701 | 989 | 16,384 | 32,768 | 32,248 | 65,536 | 4,683 | 33,288 | yes | yes |

Envelopes are packed QEV requests, each answering a whole width group in one shared state; they are **not** independent tweet batches. Tweet decisions are the prediction rows. Token maxima are tokenizer-only preflight receipts (`qev_context.json`); no model forward ran during preflight. The binding branch limit is the card limit 16,384 tokens, which is what the branch headroom uses; the runtime limit (32,768 here) is wider and not binding. Table maxima are the **test panel only**: at widths 4 and 16 the warmup envelope is larger (363 vs 348 and 940 vs 881 branch tokens), while at 64 and 256 the test envelope dominates; combined test+warmup maxima are in `packed-context.csv` and never exceed the card limit.

## 6. Deviations and open items

- qwen2b width 4: count_correct varies across reps: [117, 118, 119]
- qwen2b width 16: count_correct varies across reps: [472, 473]
- qwen2b width 64: count_correct varies across reps: [472, 474, 475]
- qwen2b width 256: count_correct varies across reps: [472, 473, 475]

## 7. Files

Standard `analysis.py` export: `runs.csv`, `trials.csv`, `class_quality.csv`, `energy_components.csv`, `energy_intensity.csv`, `performance.csv` (`analysis-cli.log` has the CLI transcript). Added here: `inference_medians.csv`, `skips.csv`, `packed-context.csv`, `width1-comparison.csv`, `width1-disagreements.csv`, `SUMMARY.md`, `verification.json`.

The top-level [results report](../../RESULTS.md) links this sweep and the separate baseline.
