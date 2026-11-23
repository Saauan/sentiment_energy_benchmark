# Experimental design and reporting protocol

## 1. Objective and estimand

The benchmark measures the quality, end-to-end serving performance, and
CodeCarbon-estimated CPU/GPU/RAM energy of complete systems performing the same
TweetEval sentiment task. A **system condition** includes the model checkpoint,
runtime/engine and version, adapter, input format, output/readout mechanism,
decoding or choice settings, precision/quantization, hardware, workload width,
cache settings, and software environment.

The primary estimand is run-level gross component energy per submitted tweet
for a fixed, documented inference workload, accompanied by macro recall,
macro-F1, accuracy, coverage, throughput, and end-to-end latency. Idle-adjusted
energy is secondary and remains separately labelled. Training and calibration
energy are distinct workload kinds, not inference energy.

This is not a model-only comparison when engines or output mechanisms differ.
No result may be used to claim an energy effect of architecture alone unless all
other condition elements are genuinely comparable.

## 2. Reproducibility cards

### 2.1 Client and analysis environment

| Field | Pin/protocol |
|---|---|
| Client Python | 3.12.13 |
| Dependency lock | `requirements.lock`, generated with `uv pip compile --generate-hashes` |
| CodeCarbon | 3.3.3 |
| pandas / Altair / marimo | 3.1.1 / 6.3.0 / 0.25.2 |
| scikit-learn / fastText wheel | 1.9.1 / 0.9.2 |
| JSON Schema / referencing | 4.26.0 / 0.37.0 |
| Native BoAmps schema | BoAmps commit `af1680c56ce2cefd5777fcac1c57591b685d6be5` |
| Native BoAmps format | `formatVersion=0.1` |

The resolved interpreter, package versions, thread environment, CPU affinity,
host platform, power settings where observable, and GPU metadata are captured in
run artifacts. GPU engine dependencies have separate pinned environments; they
must not silently drift with the client environment.

### 2.2 Dataset card

| Field | Value |
|---|---|
| Task | TweetEval sentiment, three-class single-label text classification |
| Source | <https://github.com/cardiffnlp/tweeteval> |
| Revision | `4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66` |
| Labels and order | `negative`, `neutral`, `positive` (integer IDs 0, 1, 2) |
| Train / validation / test sizes | 45,615 / 2,000 / 12,284 |
| Integrity | Per-file SHA-256 in `setup/inputs.sha256`, rechecked by the offline adapter |
| Full-phase test panel | 257 distinct zero-based test IDs, sampled with seed 1729 and sorted |
| Warmup panel | 17 validation IDs, sampled using seed 1730 and sorted |
| GPU pilot panel | 49 test IDs, same panel-selection rule and configured seed |
| CPU pilot panel | 65 test IDs, same panel-selection rule and configured seed |

The manifest retains selected test IDs, warmup validation IDs, file hashes,
panel hash, split sizes, label order, seed, and class support. All systems in a
phase see the same panel and order. Training uses only the training split.
Validation supplies warmup rows and would be the only permitted split for any
future prompt/hyperparameter selection; the current classical configuration
performs no validation-based tuning. Test predictions and quality are reserved
for final evaluation of the fixed condition.

TweetEval source terms and potential restrictions must be reviewed before
redistribution. Selected IDs and hashes provide reproducibility without
redefining the upstream dataset license.

### 2.3 Model and engine cards

| Family | Systems | Runtime and condition | Independent width support |
|---|---|---|---|
| Traditional | TF-IDF + linear SVM; TF-IDF + multinomial NB | scikit-learn 1.9.1; from-scratch train-split fit; resolved TF-IDF/estimator settings persisted | Static groups 1 and 16 |
| Small neural baseline | fastText | fasttext-wheel 0.9.2; from scratch, single thread, no pretrained vectors or autotuning | Static groups 1 and 16 |
| Generative Qwen | Qwen3.5 2B, 9B, and conditionally 27B | vLLM 0.19.0, BF16, no quantization, deterministic one-label generation | Closed-loop independent requests 1 and 16 |
| Decision | Joey-1, Elephant-1, and conditionally Mammoth-1 | Model-specific pinned local snapshot in patched OneJev QEV reference runtime; structured three-way choice | 1 only |

The exact Hub repository/revision, resolved local snapshot, weight metadata,
adapter settings, and any recorded skip are authoritative in `experiment.json`
and the run manifest. The generative pins are `Qwen/Qwen3.5-2B` at
`15852e8c16360a2fea060d615a32b45270f8a8fc`, `Qwen/Qwen3.5-9B` at
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`, and `Qwen/Qwen3.5-27B` at
`81a287ca484a5e71148f5beb7f367bb5286d70f9`. Larger models enter only after a
memory/runtime pilot. Preflight weight-byte fit is not proof of successful
runtime fit.

The SVM and fastText conditions are **baseline variants**, not claims of exact
TweetEval-paper reproduction: this SVM configuration does not implement the
published word-plus-character recipe, and fastText does not use the published
pretrained Twitter embeddings.

#### vLLM generative card

The primary vLLM condition fixes:

- engine 0.19.0, one visible GPU, BF16, no quantization or implicit offload;
- temperature 0, `top_p=1`, fixed seed, at most 8 output tokens;
- thinking disabled and a short instruction to answer exactly one of the three
  labels;
- `max_model_len=1024`, `max_num_seqs=16`, one server on loopback;
- cross-request prefix caching disabled (`--no-enable-prefix-caching`), with no
  complete-input or prediction cache across repetitions;
- closed-loop groups: all independent requests in a group finish before the
  next group is submitted; final partial groups remain visible in predictions.

A cache-enabled, thinking-enabled, quantized, different-precision, multi-GPU,
or alternative-engine condition would require a new explicit label and could
not be pooled with the primary condition.

#### QEV decision card

| Model | Exact revision |
|---|---|
| `baselabs/joey-1` | `db5c2a631f2da93c4173378438d8246966e25159` |
| `baselabs/elephant-1` | `2e80345f1e41b2c1eaff8ae2538c0ac657210c44` |
| `baselabs/mammoth-1` | `858aab610d4d124315ebf72d2f36c5e7341f758f` |

The reference setup uses OneJev commit
`81ce62f1597c91e46767d4d02d6ac2e18534fe94`, checkpoint patch git blob
`7a432e8d254a9838b04ca3a155547ec3b3c70e78`, BF16 weights, the reference FP32
output head, and the model cards' pinned environment constraints. Local
snapshots are mandatory because loading only a Hub ID can omit the checkpoint's
block-bidirectional attention and `slot_lm` overrides.

The fixed structured choice uses raw tweet text as state, the pinned sentiment
question and option descriptions, label order negative/neutral/positive, and
`debias=1`. A different debiasing or prompt is another condition. QEV's branch
batching concerns options for one state; it is not a 16-tweet batch and tweets
must never be concatenated to simulate one.

Before measurement, inspect the worker evidence and verify: exact OneJev HEAD;
applied patch; local config/readout blobs; block-bidirectional full attention
and causal linear attention; `slot_lm`; permitted attention implementation;
unique answer slots; and a fixed smoke classification. See
`qev_setup/README.md` for the full provenance and acceptance checklist.

#### Optional cross-engine experiment

Running a decision checkpoint under vLLM is a separate cross-engine experiment,
not a drop-in replication. First verify model-specific attention and readout,
then compare labels and probabilities with the reference runtime on a fixed
sample. Label artifacts with both engine and reference-comparison outcome and
never pool them with primary QEV or generative-vLLM conditions.

## 3. Phases, workloads, and repetitions

### 3.1 Planned phase matrix

The validated `experiment.json` is authoritative. Initial planned values are:

| Phase | Models | Test panel | Repetitions | Initial inference passes per supported width |
|---|---|---:|---:|---|
| `gpu_pilot` | Qwen 2B + Joey-1 | 49 | 1 | Qwen 2B: 3 at width 1 and 2 at width 16; Joey-1: 3 at width 1 |
| `cpu_pilot` | SVM, NB, fastText | 65 | 1 | SVM/NB: 450 at width 1, 4000 at width 16; fastText: 1000/1000 |
| `full` | All nine configured systems, subject to recorded fit/support skips | 257 | 3 | vLLM systems: 2/4; QEV systems: 1 at width 1; SVM/NB: 200/1200; fastText: 1000/1000 |

Pass counts are nested by model and stringified width, for example
`passes["svm"]["16"]`. They were chosen from preliminary timing estimates to
obtain measurable windows; fastText and any other uncertain timing must be
calibrated **before** the corresponding execution. Once execution starts, pass
counts, panel, and conditions are fixed. Do not change them afterward to obtain
a preferred energy or quality result.

The initial energy settings are `country_iso_code=FRA`, `sample_secs=1`,
`gpu_ids=[0]`, and `min_samples=5`. Planned device-idle and loaded-idle windows
are 8 s and 12 s, with a 5 s settling delay. Resolved values and actual
durations are recorded per attempt. Current model-loaded idle trials are
scheduled for CUDA systems; classical loaded-idle adapter names in the
configuration do not create such a trial.

The initial defaults are retained for provenance. After the CPU sampling
pilot, the final laptop run used [the measured profile](profiles/laptop-full-isolated-02.json):
0.2 s cadence with an isolated tracker, and 1,750 fastText passes at each width.
The corrected [CPU pilot profile](profiles/laptop-cpu-isolated-02.json) uses the
same sampler/cadence. These were timing and metering changes, not label-quality
or prompt tuning; their artifact roots are distinct from the initial pilots.

Width 1 is the common primary workload. Width 16 is measured only where
independent-tweet grouping is supported. One QEV choice is always one tweet.
Actual group indices, widths, final partial groups, request counts, and pass
indices are persisted.

### 3.2 Experimental unit and order

A measured run/repetition is the energy experimental unit. Tweets within a run
and repeated passes over the same panel are not independent energy replicates.
Retain every repetition and summarize run-level variation; do not treat the
larger number of tweet predictions as a larger energy sample size.

The runner deterministically interleaves model/width/repetition blocks by
`SHA256(seed, condition)`, with the condition as a tie-breaker, and records the
block order in `plan.json`. Device idle, any training, loaded idle, and
inference retain their dependency order within each block. Record actual
execution order, clocks/temperatures/power limits where observable, and
background-host state. Seeded interleaving reduces systematic condition-order
effects but does not eliminate thermal or temporal confounding.

## 4. Execution protocol

### 4.1 Host and preflight

1. Obtain the single runtime/measurement ownership handoff.
2. Confirm the resolved lockfiles, dataset/schema hashes, local model snapshots,
   exact revisions, engine paths, and output roots.
3. Stop unrelated background activity. Record CPU affinity/thread variables,
   CPU governor/preferences, AC state, memory, load, GPU IDs, driver, free/total
   VRAM, power state/limit, temperature/clocks where available, and existing GPU
   compute applications.
4. Preflight weight bytes plus configured reserve before any GPU model download
   or load. Refuse unsupported multi-GPU visibility or active competing GPU
   work. Record a skip if the specified condition cannot run.
5. Never silently quantize, offload, lower precision, replace a checkpoint, or
   change hardware. Any explicit alternative becomes a different condition.

### 4.2 Setup, smoke checks, and warmup

Model construction is lazy. Model loading, training, engine startup, and smoke
checks happen only during an authorized runtime phase. Inspect engine logs and
resolved adapter metadata before accepting the run.

For each model and each supported width, warm up after loading and before the
steady-state measured window using the configured validation rows. Record
warmup workload, duration, stabilization evidence, and cache state. Warmup is
excluded from steady-state energy and quality. Model-loading/cold-start energy
is outside the primary boundary unless a separately labelled measurement is
later implemented; it is not zero.

### 4.3 Measured inference bright line

Use one fresh CodeCarbon `OfflineEmissionsTracker` per measured trial in machine
tracking mode on the single host containing both client and inference
server. Resolve hardware metadata, output methods, schema validator, and task
metadata before the bright line. The final meter uses `meter-worker/1`, a
separate interpreter, so the scheduler does not share the client's Python
interpreter. Preparation precedes measurement; `begin`/`end` handshakes bound
only the work callback, not fitting, scoring, or file serialization for an
inference trial. The worker's short handshakes and sampling load still fall
within the machine-level energy boundary. `monitoring.json` preserves sampler
mode, scheduled counts, tick gaps and tracker/work-window timing.

The timed workload starts immediately before the first request and ends after
the last result has completed and its output has been parsed into the adapter's
label/raw record. It includes preprocessing, client/queueing, model execution,
server work, output parsing, and any permitted request overhead. Prediction
serialization, the persisted-prediction audit, scoring, and report analysis
happen outside this window and retain separate `outside_window_s` timings.

Persist monotonic-clock start/end and tracker start/stop overhead. The tracker
uses `pue=1.0`; reported energy is the sum of estimated CodeCarbon components,
not whole-system or wall-meter energy. The full host has no wall meter. Record
whether an unused GPU remains powered and metered during CPU-only work.

Retries are disabled in the primary request policy. If a future condition allows
them, its requests and energy stay inside the same measured attempt and retry
counts must be explicit.

### 4.4 Interruption and immutability

`state.json` and attempt histories are atomically persisted. SIGINT/SIGTERM or
an abandoned running checkpoint produces an incomplete/recoverable state.
Completed trials are immutable and never silently repeated or overwritten.
Retries create fresh `attempt-NNNN` directories. Failed and skipped states need
explicit retry flags; all prior logs and native artifacts remain available for
audit.

## 5. Energy measurements and calibration

### 5.1 Component and gross energy

Retain native CodeCarbon CPU, GPU, and RAM energies and total energy in kWh.
An unavailable component is null, not zero; a detected zero and an unavailable
measurement are different. Record actual measurement/estimation backends,
included GPU indices, sample interval, source signature, and resolved hardware.

For inference with `N_submitted > 0`, gross intensity is:

\[
E_{gross} = \frac{3{,}600{,}000 \times E_{kWh}}{N_{submitted}}
\quad \mathrm{J/submitted\ tweet}.
\]

`N_submitted` counts actual tweet submissions across the fixed panel and all
passes, not API calls, groups, or only valid predictions. Keep completed, valid,
and correct counts for alternative denominators. Native BoAmps
`powerConsumption` is energy in kWh despite its name; do not reinterpret it as
watts. For training results, the runner's legacy derived-intensity keys use
training rows as the denominator; they are not tweet energy and are excluded
from the notebook's per-tweet inference figure.

### 5.2 Device-idle calibration

A `device_idle` trial performs no model workload. It measures the same host and
CodeCarbon source configuration for the configured duration, with device state,
unused-GPU state, and background conditions recorded. Device idle is a
calibration kind and never enters inference-quality figures.

For a compatible baseline with the same source signature, host identifier, and
GPU UUID set:

\[
E_{device-adjusted} = \frac{3{,}600{,}000}{N_{submitted}}
\left(E_{gross,kWh} - E_{device,kWh}
\frac{T_{gross}}{T_{device}}\right).
\]

### 5.3 Model-loaded idle calibration

A `loaded_idle` trial loads the specified model/runtime but submits no inference
workload. Use the same model condition and compatible measurement source
signature to estimate the additional idle baseline. Apply the analogous
duration-scaled subtraction and label it `loaded_idle_adjusted_j_per_tweet`.

Do not substitute device idle for loaded idle or pool baselines across models,
engines, hosts, GPU sets, or source signatures. If a matching baseline is absent
or incompatible, the adjusted result is null. Signed adjusted values are retained:
a negative value can expose baseline mismatch or measurement noise and must not
be clipped to zero.

Gross, device-idle-adjusted, and loaded-idle-adjusted intensities are different
measurement bases and must appear as separate series/table rows.

### 5.4 Training and prior training

Task-specific classical/fastText training is measured as `training`, separately
from inference and any hyperparameter tuning. Record training rows, resolved
parameters, threads, duration, and native energy artifacts. Pretraining energy
for Qwen/decision models is outside the measured boundary or unavailable, never
zero. No inference energy ratio should include training unless a separately
defined amortization assumption is stated.

## 6. Performance and quality scoring

### 6.1 Persisted predictions and counts

Every submitted tweet produces a durable prediction record containing test ID,
gold label, predicted label or null, raw output, completion state, latency and
latency scope, pass index, group index, and actual group width. Cardinality and
panel/group order are audited before a result is accepted.

Report at least:

- submitted, completed, valid, and correct tweet counts;
- panel size, pass count, group count, and final group width;
- retry requests (zero in the primary condition);
- HTTP request count for vLLM and QEV choice count for QEV, as applicable;
- repeat disagreements from the first pass.

Invalid or missing predictions remain in the denominator and count as wrong.
Coverage is the fraction with one of the three valid labels. A failure marker
is never a fourth sentiment class.

### 6.2 Quality basis

Quality is computed outside the energy window from persisted predictions. The
primary basis is **first pass only** (`basis_pass=0`, `basis_n=panel_size`), so
repeated energy passes do not replicate test labels or artificially change the
quality denominator. Later passes are used for repeat-disagreement diagnostics,
not pooled into primary quality.

Restrict macro averages to negative, neutral, and positive:

- macro recall across the three gold classes is the official TweetEval
  sentiment metric;
- also report macro-F1, per-class precision/recall/F1/support/predicted/correct,
  accuracy, and coverage;
- denominators with no predictions or no gold support use the runner's explicit
  scoring rule and must be interpreted with the panel support table.

The quality–energy plot therefore pairs first-pass macro recall with gross
energy per actual submitted tweet over the full repeated workload.

### 6.3 Latency and throughput

End-to-end latency includes queueing. Persist adapter request latency when
available; otherwise the fallback is closed-loop group completion and is
labelled as such, not presented as per-tweet model latency. Report p50 and p95
with their scope and completed-request basis. Group p50/p95 may be reported
separately.

Main throughput is:

\[
\mathrm{throughput}_{tweets/s} =
\frac{N_{completed}}{T_{full\ measurement\ window}}.
\]

Valid throughput may additionally use `N_valid`, with an explicit denominator
label. The runner's legacy `throughput_tweets_s` key contains training rows/s
for a training result; analysis must not relabel that as tweet throughput, and
the notebook's tweet-throughput figure is inference-only. Do not compare batch 1
and batch 16 without showing actual grouping, final partial groups, engine
scheduling settings, and latency scope.

## 7. Native BoAmps export and verification

Each measured trial emits both CodeCarbon CSV and native BoAmps JSON. Task
metadata identifies inference/training/calibration stage, text-classification
or device-idle family, actual request count, a nonempty algorithm array, and a
nonempty dataset array with usage/type and source information. Calibration and
task-specific training have native `nbRequest=0` (training row quantity remains
in dataset metadata); inference may not silently substitute zero for a missing
count.

Verification is an acceptance gate, not a warning:

1. Exactly one native report exists and validates offline against the five
   SHA-256-checked BoAmps schemas at the pinned commit.
2. Local schema references resolve without network retrieval using the
   upstream-declared Draft-04 dialect. For Draft-04 metaschema compatibility,
   only `algorithm_schema.json`'s validation-inert `required: []` keyword is
   omitted from the in-memory validator copy; source bytes and exports are
   unchanged.
3. Native format version is 0.1. Report ID, CSV run UUID, report filename, and
   native header agree.
4. Native and CSV duration and total energy agree within strict numeric
   tolerance; total equals the detected component sum.
5. Tracking mode, CodeCarbon version, expected task metadata, PUE, monitoring
   metadata, and schema pin agree.
6. Components without a detected monitoring source are represented as
   unavailable in the derived result; invalid native output fails the pipeline.
7. Inference meets the low-sampling gate: at least `min_samples=5` scheduled
   CodeCarbon samples when the scheduled-sample counter is available, with the
   duration-based proxy retained as supporting evidence. Short/low-sampling
   trials remain explicitly flagged rather than silently accepted as precise.
   The laptop execution also checked integration coverage against the observed
   constant RAM power estimate (`3,600,000 × RAM kWh / (RAM W × native duration)`
   ≥ 0.85 for inference); this is an internal consistency check for that
   backend, not an accuracy guarantee or a test for variable-power RAM sensors.

Retain the native report, CSV, monitoring metadata, CodeCarbon log, predictions,
resolved manifest, and schema pin together. Any later transformation required
for BoAmps ingestion must preserve native units and be documented separately.

## 8. Analysis and reporting rules

- Run `analysis.py` or `analysis_notebook.py` on one phase artifact root at a
  time. Keep phase/kind labels if a later report deliberately combines roots.
- Include all statuses, failed/incomplete artifacts, and recorded skips in the
  run-status report. Successful figures require a readable active result and a
  complete state.
- Do not pool phases, workload kinds, models, widths, repetitions, datasets,
  hosts, adapters, engines, precision, cache settings, or calibration bases.
- Preserve individual runs and report run-level variation. Do not use tweets as
  independent energy replicates.
- Missing fields and unavailable components remain null. Missing data never
  becomes zero for plotting or aggregation.
- Keep gross and adjusted energy separate, inference separate from training and
  calibration, and quality restricted to successful inference.
- Show quality, energy components, energy intensity, throughput, and p50/p95
  latency with model/width/phase/kind context. Include the quality–gross-energy
  trade-off without implying causality.
- Preserve exact raw artifacts alongside any figure, CSV, or aggregate. The
  coordinator's separate results report must contain no fabricated or
  placeholder values.

## 9. Pilot and acceptance sequence

### Gate A — offline implementation

- Locked client imports and offline tests pass.
- Dataset/schema hashes and configuration validate.
- Plans resolve the intended models, panels, widths, passes, repetitions,
  calibrations, and output paths.
- Analysis tests and synthetic/empty marimo export succeed without launching a
  benchmark.

### Gate B — GPU pilot

Run one generative Qwen and one QEV decision model. Before continuing:

- actual fit and peak-memory behavior are acceptable, with no silent condition
  changes;
- vLLM served revision and deterministic/output behavior are verified;
- QEV attention, readout, patch, structured choice, and smoke outputs are
  verified;
- batch support matches the declared independent-tweet semantics;
- warmup, steady-state windows, device/model-loaded idle calibrations, and
  interruption/resumption behave as specified;
- predictions audit and first-pass scoring succeed;
- native BoAmps/CSV validation and actual workload counts pass;
- analysis displays all metrics and missing components correctly.

### Gate C — classical pilot

Add SVM, NB, and fastText. Verify training-only provenance, deterministic
configuration where claimed, static groups at both widths, sufficient tracker
sampling duration, persisted prediction quality, repeat behavior, native
exports, and timing-based pass counts fixed before execution.

### Gate D — full experiment

Attempt all nine configured models and supported width conditions with three
repetitions. Add 27B systems only after their memory/runtime pilots. Record any
fit skip with its exact configuration and evidence. Complete verification and
retain raw reports before beginning the results report.

## 10. Limitations and threats to validity

1. **Complete-system confounding.** Model, runtime, prompt/choice interface,
   readout, hardware, and batching can vary together; energy differences are not
   architecture-only causal effects.
2. **Estimated boundary.** CodeCarbon CPU/GPU/RAM is a component estimate with
   PUE 1, not whole-system energy. There is no wall meter on the full host.
   RAM/CPU values may be model-based, and an unused powered GPU can affect CPU
   runs.
3. **Sampling and short windows.** Tracker interval and overhead can materially
   affect fast classifiers. Timing-derived pass counts mitigate but do not remove
   sampling error; report window durations and run variation.
4. **Calibration instability.** Idle subtraction can be noisy or negative.
   Duration scaling and source-signature matching do not eliminate temporal,
   thermal, or model-loaded baseline mismatch.
5. **Workload equivalence.** Static CPU groups, closed-loop vLLM bursts, and
   single-choice QEV requests have different scheduling/readout semantics.
   Client width does not prove an internal hardware batch of 16.
6. **Repetition independence.** Repeated passes share tweets and a run; they are
   not independent energy replicates. Seeded block interleaving reduces order
   effects but cannot remove thermal, drift, or within-block dependency effects.
7. **Baseline fidelity.** The SVM and fastText systems are variants, not exact
   published TweetEval baselines. No validation tuning is currently performed.
8. **Panel generalization.** Fixed subsets of 49/65/257 tweets support controlled
   comparison but not automatic generalization to all TweetEval or other text.
   Report class support and first-pass quality.
9. **Test contamination.** Pretrained models may have seen TweetEval or related
   text. A newer held-out sentiment dataset would be needed for complementary
   contamination-resistant evaluation.
10. **Runtime acceptance pending evidence.** Static tests and weight-byte
    preflight do not prove CUDA fit, QEV numerical behavior, or attention/readout
    correctness. Unexecuted or failed systems remain planned/skipped, not
    measured.
11. **Training boundary.** Only task-specific training can be measured here;
    large-model pretraining is excluded/unavailable. No total lifecycle-energy
    claim is supported.
12. **Host and software specificity.** Results apply to the recorded host,
    power state, threads, package locks, engine commits, and revisions. Another
    environment requires a new labelled condition.
13. **Latency scope.** Adapter request latency and closed-loop group-completion
    fallback are not interchangeable; group latency and request latency must be
    distinguished.
14. **Quality repeat basis.** First-pass quality avoids test-label pooling but
    does not characterize all repeat variability; report repeat disagreements
    separately.
