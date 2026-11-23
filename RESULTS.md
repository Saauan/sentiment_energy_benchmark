# TweetEval sentiment energy benchmark — results

**23 November 2026.** Two completed measurement sets on one laptop: a
ten-condition GPU width sweep (`gpu_widths`, latest) and the five-system
baseline matrix (`full`). Both use the same fixed 257-tweet panel and first-pass
quality. They are different workloads and are reported separately, never pooled.

- **Latest:** [width sweep summary](results/gpu_widths_01/SUMMARY.md) ·
  [inference medians](results/gpu_widths_01/inference_medians.csv) ·
  [verification receipt](results/gpu_widths_01/verification.json)
- **Baseline:** [preserved detailed report](results/full_isolated_02/DETAILS.md) ·
  [inference medians](results/full_isolated_02/inference_medians.csv) ·
  [notebook](results/full_isolated_02/notebook.html) ·
  [verification receipt](results/full_isolated_02/verification.json)
- **Protocol:** [DESIGN.md](DESIGN.md) · **Setup:** [README.md](README.md)

## 1. GPU width sweep (`gpu_widths_01`)

**Status:** all 90 planned entries completed — 30 inference, 30 unloaded idle,
and 30 model-loaded idle — with no skips, failures, or incomplete attempts. No
pilot phase was run. Every inference trial passed its sampling gate
(14–258 scheduled ticks; worst observed tick gap 0.220 s).

**Conditions:** Qwen3.5-2B/vLLM and Joey-1/patched OneJev QEV, widths
1/4/16/64/256, three repetitions each, same 257 test IDs (76 negative, 129
neutral, 52 positive). Qwen submits independent requests with vLLM's scheduler
capped at 16. Joey receives one packed envelope per width group: a shared,
numbered tweet list, so **packed decisions are one shared state, not an
independent batch**. Gross J/tweet divides by all submitted tweet decisions;
quality uses the first measured pass only.

| Model | Width | Macro recall | Macro-F1 | Accuracy | Gross J/tweet | Tweets/s | Group p95, ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Joey-1 | 1 | 0.646 | 0.647 | 0.654 | 5.637 | 20.2 | 54 |
| Joey-1 | 4 | 0.589 | 0.599 | 0.634 | 1.890 | 60.5 | 72 |
| Joey-1 | 16 | 0.582 | 0.586 | 0.638 | **1.496** | **76.7** | 217 |
| Joey-1 | 64 | 0.593 | 0.598 | 0.634 | 1.958 | 51.9 | 1,324 |
| Joey-1 | 256 | 0.578 | 0.574 | 0.630 | 3.292 | 22.9 | 11,315 |
| Qwen3.5-2B | 1 | 0.589 | 0.423 | 0.455 | 4.574 | 24.4 | 46 |
| Qwen3.5-2B | 4 | 0.595 | 0.428 | 0.459 | 1.411 | 81.3 | 65 |
| Qwen3.5-2B | 16 | 0.595 | 0.429 | 0.459 | 0.766 | 147.0 | 120 |
| Qwen3.5-2B | 64 | 0.595 | 0.429 | 0.459 | 0.731 | 156.5 | 434 |
| Qwen3.5-2B | 256 | 0.595 | 0.429 | 0.459 | **0.695** | **161.1** | 1,589 |

Medians of three repetitions; ranges and all counters are in
`inference_medians.csv`. Group p95 is the closed-loop completion of one
submission envelope: all requests in a width for Qwen, or Joey's packed request.
For Qwen, per-request (HTTP) p95 was 45/62/116/409/1,477 ms for widths 1–256.
Joey's envelope is a single adapter request, so group p95 is the comparable
measure; the export reports an adapter-request p95 only at width 1 (53.9 ms).

### Findings

- **Joey:** width 16 is the most energy-efficient packed condition
  (1.496 J/tweet, 76.7 tweets/s). Width 1 has the best quality (macro-F1 0.647,
  accuracy 0.654); packing reduces macro-F1 by 0.048–0.073 and accuracy by
  0.016–0.024. Packing changes each answer's context, so this is a workload and
  quality trade-off, not a free speedup.
- **Qwen:** gross energy per tweet falls 6.6× from width 1 to 256
  (4.574 → 0.695). Most of the gain already occurs by width 16 (0.766); going
  from 16 to 256 lowers J/tweet only ~9% while group p95 grows ~13×
  (120 → 1,589 ms) and HTTP p95 grows ~13× (116 → 1,477 ms). Quality is flat
  across widths (macro-F1 0.423–0.429, accuracy 0.455–0.459).
- Loaded-idle-adjusted J/tweet, a separate basis, is 3.462/1.143/0.912/1.102/1.356
  for Joey widths 1–256 and 2.788/0.853/0.461/0.442/0.416 for Qwen.
- **Context:** every Joey envelope fit. At width 256 the longest branch used
  11,701 of 16,384 tokens, and the largest request used 32,248 of 65,536 tokens.
  Token checks ran before inference; no input was truncated.
- **Engine retry behaviour (audited):** 45 of 48 full width-64 Joey envelopes
  (15 of 16 per repetition) hit an internal OOM retry and halved the branch
  chunk from 64 to 32; there was no sequential fallback. Width 256 used planned
  token-budget chunking (up to 17 branches per chunk, 16 chunks per full envelope) with no OOM
  retries. Both are internal to one request envelope and change neither the
  envelope nor tweet-decision count.
- Repeat disagreements with each trial's own first pass were 0 for every Joey
  width; for Qwen they were 0 (widths 1 and 4), 6 (16), 3 (64), and 8 (256)
  across later passes. Deterministic request parameters do not guarantee
  identical labels under concurrency.

## 2. Baseline laptop matrix (`full_isolated_02`)

Five systems were measured on the same panel and the same first-pass basis:
Qwen3.5-2B/vLLM, Joey-1/QEV, TF–IDF + linear SVM, TF–IDF + multinomial NB, and
fastText. 27 inference trials and 81 validated native BoAmps/CodeCarbon pairs
completed with no failures; four larger checkpoints were skipped on fit
without downloading or altering them.

- Within Qwen3.5-2B, width 16 had ~6.2× lower gross J/tweet, ~6.1× higher
  throughput, and ~2.6× higher per-request p95 than width 1, with unchanged
  median macro recall.
- Joey width 1 had the highest observed macro recall (0.646) among width-1
  systems on this panel.
- Classical pipelines used far less gross energy per submission on this host
  (0.00018–0.020 J/tweet), but engine, formatting, work unit, and hardware use
  also differ; this is not an architecture-only effect.

Full conditions, tables, pilot corrections, and limits are preserved verbatim
in [results/full_isolated_02/DETAILS.md](results/full_isolated_02/DETAILS.md).
The baseline is **not pooled** with the width sweep. Qwen width 16 was repeated
in the new sweep; packed Joey widths above 1 were not measured in the baseline.

## 3. Historical engineering evidence

`results/gpu_pilot/` and `results/cpu_pilot_isolated_02/` are preserved
engineering exports from meter and restart checks. They are not required steps
and are not quantitative results. The current readiness rule is in
[DESIGN.md](DESIGN.md#9-acceptance-sequence).

## 4. Limits

- Single laptop host; CodeCarbon CPU/GPU/RAM estimates with PUE 1, not
  wall-meter or whole-system energy. Clocks and thermals were observed, not
  locked, and desktop activity was possible.
- Quality is first-pass only; later passes diagnose disagreement and are not
  independent quality samples. Energy divides by all submitted tweet decisions,
  so repeated passes count as workload, not as new tweets.
- A run is the energy experimental unit; tweets and passes are not independent
  replicates. Width-1 comparisons on one 257-item panel limit generalization,
  and pretrained models may have seen TweetEval text.
- Qwen bursts are independent requests under a scheduler cap of 16; Joey
  envelopes are shared-context packed requests. Client width never proves an
  internal hardware batch. No result isolates an architecture-only effect.
- One Joey width-64 repetition took ~28.0 s versus ~19.7 s in the other two.
  The retry pattern and first-pass predictions were identical across repetitions.

Nothing was submitted upstream to BoAmps. Before any contribution, review
identifiers, dataset terms, and native hardware metadata against observed
inventory.

## Files

| Path | Contents |
|---|---|
| `results/gpu_widths_01/SUMMARY.md` | Width-sweep conditions, coverage, and comparisons |
| `results/gpu_widths_01/audit-notes.md` | Export-side audit of sampling, retries, determinism, and corrections |
| `results/gpu_widths_01/inference_medians.csv` | Condition medians, ranges, counters, sampling |
| `results/gpu_widths_01/packed-context.csv` | Per-run Joey context maxima and digests |
| `results/gpu_widths_01/width1-comparison.csv` | Descriptive paired width-vs-1 agreement counts (no p-values) |
| `results/gpu_widths_01/width1-disagreements.csv` | Per-tweet label changes by width |
| `results/gpu_widths_01/runs.csv`, `trials.csv`, `class_quality.csv`, `energy_components.csv`, `energy_intensity.csv`, `performance.csv` | Standard analysis tables |
| `results/gpu_widths_01/verification.json` | Native-report and prediction audit |
| `results/full_isolated_02/DETAILS.md` | Preserved baseline report |
| `results/full_isolated_02/` | Baseline CSVs, notebook, and verification receipt |
