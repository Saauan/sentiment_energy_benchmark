# gpu_widths — audit notes (export-side checks)

Read-only audits requested with the GO, run against `artifacts/gpu_widths_01/` after
the standard export. These notes are descriptive engineering evidence, not a
hypothesis test; the artifacts themselves are unchanged.

## A. Scheduled power samples versus meter clock ticks

Scope: all 30 completed inference trials (plus their attempts). Source:
`attempt-*/energy/monitoring.json` (`timing`, `timing.meter.ticks`) and the
persisted `gate`.

| check | result |
|---|---|
| `scheduled_power_samples == ticks.calls` | 30 / 30 inference trials |
| `total_power_samples >= scheduled_power_samples` | 30 / 30 |
| `ticks.truncated` true | none |
| scheduled samples per inference window | 14 – 258 (matches the coordinator's independent 14–258 scan) |
| worst inter-tick gap across all inference windows | 0.2204 s (cadence 0.2 s) |
| `scheduled / (window_s / 0.2)` | 0.949 – 0.999 |
| sampling gate met (`meets_minimum_window`) | 30 / 30 |
| `benchmark.py verify` `low_sampling` list | empty |

The shortest window is qwen2b width 4 (14 scheduled samples in ~2.95 s); it is
the only case near the 0.949 ratio. Sample counts are consistent between the
scheduled counter and the meter's own tick calls everywhere, and no window was
truncated.

## B. QEV width 64 memory retry versus width 256 planned chunking

The OneJev engine pre-fills the shared prefix once and runs the question
branches in chunks of `min(branches, 200000 / prefix_tokens)`, halving the chunk
on `torch.OutOfMemoryError` and logging
`batched branches out of memory at prefix ...; retrying with chunk N`
(`engine.py::_run_batched`). Chunking is internal to one request envelope: it
does not change the envelope count or the tweet-decision count.

| width | test max prefix tokens | initial chunk | planned chunks / envelope | OOM retries observed | outcome |
|---:|---:|---:|---:|---|---|
| 1 | 91 | 1 | 1 | 0 | one branch per envelope |
| 4 | 269 | 4 | 1 | 0 | single batch |
| 16 | 801 | 16 | 1 | 0 | single batch |
| 64 | 2,934 | 64 | 1 | **15 of 16 multi-tweet envelopes per repetition, in all 3 repetitions (45 total)**, always `chunk 64 → 32` | adaptive retry, no sequential fallback |
| 256 | 11,620 | 17 | 16 | 0 | planned token-budget chunking, no retry |

Details for width 64: the retry messages name prefixes 2,848 / 2,896 / 2,929 /
2,934 tokens and always halve once (64 → 32). Only a search of all three
`predictions.jsonl` files for width 64 contains the OOM string; width 256
contains none, and no trial recorded `using sequential branches`. No trial has a
second attempt, a `failure.json`, or an interrupted state.

Width 64 also contains the widest measurement ranges in the phase
(gross 1.952–2.638 J/tweet, throughput 36.7–52.2 tweets/s, group p95
1,323–2,122 ms). That spread comes from one uniformly slower repetition
(window ~28.0 s versus ~19.7–19.8 s at the same width), not from a different
retry path: the retry pattern and the first-pass predictions are identical in
all three repetitions. Both effects (adaptive 64→32 forward retry and the slower
repetition) are measurement conditions, not silent model or width changes.

Envelope versus decision counts remain as planned: width 64 -> 20 envelopes and
1,028 tweet decisions per trial; width 256 -> 8 envelopes and 1,028 tweet
decisions, i.e. 128.5 decisions per envelope (see `packed-context.csv`).

## C. Repetition determinism (cross-reference)

First-pass predictions are bit-identical across the three repetitions for every
joey1 width and for qwen2b width 1. qwen2b widths 4/16/64/256 differ slightly
across repetitions under concurrent serving. This is reported in `SUMMARY.md`
section 4; it is why identical paired counts repeat across repetitions and why
no significance test is reported.

## D. Presentation corrections applied in this export

1. McNemar p-values removed from `width1-comparison.csv` and `SUMMARY.md`; the
   earlier values were wrong (a two-sided binomial p was multiplied by two and
   clamped). Only descriptive paired counts (agreement, b, c, deltas) remain,
   with an explicit note that no p-value is reported.
2. `per-request p95` is `NA` for packed QEV widths 4/16/64/256: the engine
   reports one completion time per packed envelope and it is copied onto every
   tweet row, so a row-weighted p95 is an envelope time weighted by envelope
   width. `group p95` is the correct unit there. Raw row latencies stay
   unchanged in `trials.csv` and `performance.csv`.
3. The packed context table now shows the **card** branch limit (16,384) next to
   the runtime limit (32,768) and states that branch headroom uses the card
   limit. It also states that table maxima are test-panel only; warmup envelopes
   are larger at widths 4 (363 vs 348) and 16 (940 vs 881), and test envelopes
   dominate at 64 and 256.
4. Cross-repetition count differences in the deviations list are plain integers.

## E. Independent checks behind the handoff

- `benchmark.py verify`: ok, 90 valid complete runs, 0 inconsistencies,
  0 low-sampling entries, saved plan equals the deterministic plan.
- First-pass quality re-derived from `predictions.jsonl` matches the exported
  `quality_accuracy` / `quality_macro_recall` / `quality_macro_f1` for every
  complete inference trial (no mismatch notes).
- Submission counts per condition match the plan (`257` at passes 1, `514` at
  passes 2, `1,028` at passes 4); `qev_request_count` equals the audited
  envelope count and native `nbRequest`.
- Fixed panel support is 76 negative / 129 neutral / 52 positive for all
  conditions.
