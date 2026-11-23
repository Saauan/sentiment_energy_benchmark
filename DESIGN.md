# Experimental design and reporting protocol

This is the benchmark's protocol reference. See [RESULTS.md](RESULTS.md) for
measurements and [README.md](README.md) for setup and reproduction commands.

## Protocol at a glance

| Decision | Protocol |
|---|---|
| Comparison | Complete systems on the same TweetEval panel, not isolated model architectures |
| Primary endpoints | Gross component J/submitted tweet, macro recall, throughput, and end-to-end latency |
| Energy denominator | Actual tweet submissions across the full repeated workload |
| Quality denominator | First pass only; later passes measure repeat disagreement |
| Experimental unit | One measured run/repetition, not a tweet or pass |
| Execution | Validated setup: measure directly; new host or relevant runtime/meter change: targeted readiness check |
| Execution protocol | `engine-session-v1`: one engine process per model per repetition, widths run inside it in seed-determined order; model order is seed-determined within each repetition, not blocked by model |
| Measurement boundary | CodeCarbon CPU/GPU/RAM with PUE 1; no wall-meter claim |

## 1. Objective and estimand

The benchmark measures quality, end-to-end serving performance, and
CodeCarbon-estimated CPU/GPU/RAM energy for complete systems performing the
same sentiment task. A **system condition** includes:

- checkpoint, runtime/engine and version, adapter, and software environment;
- input format, output/readout mechanism, and decoding or choice settings;
- precision, hardware, workload width, and cache settings.

The primary estimand is run-level gross component energy per submitted tweet
for a fixed inference workload. Report it with macro recall, macro-F1, accuracy,
coverage, throughput, and end-to-end latency. Idle-adjusted energy is secondary
and separately labelled; training and calibration are separate workload kinds.

Do not make an architecture-only causal claim when engines, interfaces,
hardware, or other condition elements differ.

## 2. Fixed conditions

### 2.1 Software and measurement environment

| Field | Pin/protocol |
|---|---|
| Client Python | 3.12.13 |
| Dependency lock | `requirements.lock`, generated with `uv pip compile --generate-hashes` |
| CodeCarbon | 3.3.3 |
| pandas / Altair / marimo | 3.1.1 / 6.3.0 / 0.25.2 |
| scikit-learn / fastText wheel | 1.9.1 / 0.9.2 |
| JSON Schema / referencing | 4.26.0 / 0.37.0 |
| BoAmps schema commit | `af1680c56ce2cefd5777fcac1c57591b685d6be5` |
| Native BoAmps format | `formatVersion=0.1` |

Artifacts capture the resolved interpreter, packages, thread environment, CPU
affinity, host/power metadata where observable, and GPU metadata. GPU engines
use separate pinned environments:

- [setup/gpu/README.md](setup/gpu/README.md): vLLM installation and snapshots.
- [qev_setup/README.md](qev_setup/README.md): QEV reference setup.
- [qev_setup/RUNTIME_SMOKE.md](qev_setup/RUNTIME_SMOKE.md): actual Joey runtime
  evidence and deviations.

### 2.2 Dataset

| Field | Value |
|---|---|
| Task | TweetEval sentiment, three-class single-label classification |
| Source / revision | <https://github.com/cardiffnlp/tweeteval> at `4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66` |
| Label order | `negative`, `neutral`, `positive` (IDs 0, 1, 2) |
| Train / validation / test | 45,615 / 2,000 / 12,284 rows |
| Integrity | Per-file SHA-256 in `setup/inputs.sha256`, rechecked offline |
| Full test panel | 257 distinct zero-based IDs, seed 1729, sorted |
| Warmup panel | 17 validation IDs, derived seed 1730, sorted |
| Legacy GPU / CPU pilot panels | 49 / 65 test IDs, historical; same selection rule and configured seeds |

Every system in a phase sees the same panel and order. Manifests retain selected
IDs, file and panel hashes, split sizes, labels, seeds, and class support.

- **Training:** training split only.
- **Validation:** warmup rows; the only split permitted for future prompt or
  hyperparameter selection. Current conditions perform no validation tuning.
- **Test:** evaluation of fixed conditions only.

Review TweetEval terms before redistribution.

### 2.3 Models and engines

| Family | Systems | Runtime | Configured widths (1, 16, 64) |
|---|---|---|---|
| Traditional | TF–IDF + linear SVM; TF–IDF + multinomial NB | scikit-learn 1.9.1, from-scratch train-split fit | Static groups 1, 16, 64 |
| Small neural | fastText | fasttext-wheel 0.9.2, single thread, no pretrained vectors or autotuning | Static groups 1, 16, 64 |
| Generative | Qwen3.5 2B, 9B, 27B | vLLM 0.19.0, BF16, deterministic one-label generation | Closed-loop requests 1, 16, 64, scheduler cap 16 |
| Decision | Joey-1, Elephant-1, Mammoth-1 | Pinned local snapshot in patched OneJev QEV | Packed requests 1, 16, 64 |

Exact checkpoint revisions:

| Model | Revision |
|---|---|
| `Qwen/Qwen3.5-2B` | `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| `Qwen/Qwen3.5-27B` | `81a287ca484a5e71148f5beb7f367bb5286d70f9` |
| `baselabs/joey-1` | `db5c2a631f2da93c4173378438d8246966e25159` |
| `baselabs/elephant-1` | `2e80345f1e41b2c1eaff8ae2538c0ac657210c44` |
| `baselabs/mammoth-1` | `858aab610d4d124315ebf72d2f36c5e7341f758f` |

The selected configuration and run manifest are authoritative for snapshots,
weights, adapter settings, and skips. On an unvalidated setup, larger models
require fit and prediction readiness checks; weight-byte fit alone is
insufficient.

The SVM and fastText systems are **baseline variants**, not exact TweetEval
paper reproductions: the SVM lacks the published word-plus-character recipe,
and fastText lacks the published pretrained Twitter embeddings.

#### vLLM condition

The primary condition fixes:

- one visible GPU, BF16, no quantization or implicit offload;
- temperature 0, `top_p=1`, fixed seed, at most 8 output tokens;
- thinking disabled, `max_model_len=1024`, `max_num_seqs=16`;
- one loopback server and cross-request prefix caching disabled;
- no complete-input or prediction cache across repetitions.

Closed-loop groups finish before the next group starts; final partial groups
remain visible. The active protocol uses client bursts of 1, 16, and 64 with the
same scheduler cap of 16; the historical `gpu_widths` sweep also used 4 and
256. Wider bursts can queue: they are not wider GPU batches or evidence of
simultaneous execution.

Any cache, thinking, precision, quantization, GPU-topology, or engine change is
a separate condition.

#### QEV condition

The reference setup pins OneJev commit
`81ce62f1597c91e46767d4d02d6ac2e18534fe94`, checkpoint `serve.patch` blob
`7a432e8d254a9838b04ca3a155547ec3b3c70e78`, BF16 weights, the reference FP32
output head, and checkpoint-specific environment constraints.

Local snapshots are mandatory: loading only a Hub ID can omit the checkpoint's
block-bidirectional attention and `slot_lm` overrides. The QEV setup guide gives
the provenance and acceptance checklist.

The baseline structured choice uses:

- raw tweet text as state and the pinned sentiment question/options;
- label order negative/neutral/positive and `debias=1`;
- one independent tweet per request. Batching question branches for one state
  is not independent-tweet batching.

The active protocol opts all pinned QEV siblings (Joey-1, Elephant-1,
Mammoth-1) into **packed requests**: one shared, numbered tweet list with one
targeted choice question per tweet, at widths 1, 16, and 64. Singleton groups
retain the baseline request. QEV may batch or chunk branches within an envelope,
but does not batch independent states. The historical `gpu_widths` sweep packed
Joey only.

Packing places other tweets in every answer's context, so it can change quality.
It is a distinct workload, not a transparent speedup. Exact tokenizer checks
reject envelopes exceeding the model-card branch limit or either runtime limit
before measurement; there is no truncation.

Running a decision checkpoint under vLLM is a separate cross-engine experiment.
It requires attention/readout verification and fixed-sample label/probability
comparisons with the reference runtime; its results cannot be pooled with the
primary conditions.

## 3. Workloads and experimental units

### 3.1 Phases and configured pass counts

The selected JSON configuration is authoritative. The active phase is `full`
with `execution_protocol='engine-session-v1'` under the existing
`schema_version`:

| Phase | Status | Models | Panel | Reps | Widths |
|---|---|---|---:|---:|---|
| `full` | Active protocol | Nine systems, subject to fit/support skips | 257 | 3 | 1, 16, 64 |
| `gpu_pilot` | Legacy, historical | Qwen 2B + Joey-1 | 49 | 1 | 1, 16 |
| `cpu_pilot` | Legacy, historical | SVM, NB, fastText | 65 | 1 | 1, 16 |

The named `gpu_pilot` and `cpu_pilot` phases are legacy definitions, not
required execution steps. Their profiles, artifacts, and findings remain
historical engineering evidence, not prerequisites or quantitative results.

Pass counts are nested by model and stringified width in the selected
configuration. Choose them for measurable windows and fix them before
execution; never tune them afterward to improve a result. Historical pass
counts were fixed for widths 1 and 16 and are not evidence for the width-64
condition.

Configured energy/calibration settings (resolved values and actual durations
are recorded per attempt):

| Setting | Value |
|---|---|
| Country / GPUs | `country_iso_code=FRA`, `gpu_ids=[0]` |
| Sampling | `sample_secs` and `min_samples` as configured (active config: isolated tracker at 0.2 s, 5 samples) |
| Device idle / loaded idle / settling | As configured (active config: 8 s / 12 s / 5 s) |

Classical loaded-idle adapter names do not create a loaded-idle trial.

Profile provenance is historical and read-only:

- `profiles/laptop-full-isolated-02.json`: frozen reported full-run condition.
- `profiles/laptop-cpu-isolated-02.json`: historical engineering profile.
- `profiles/laptop-gpu-widths.json`: historical ten-condition sweep, never
  re-executed or pooled with the engine-session protocol.
- `profiles/legacy-experiment.json`: frozen legacy-format configuration without
  `execution_protocol`, kept for historical read/verify and configuration
  checks.

Versions, hashes, and published artifacts of these profiles are immutable, and
there is no runtime toggle back to their execution protocol.

Timing/metering changes use distinct artifact roots and do not tune labels or
prompts. Width 1 is the common primary workload. Persist actual group
indices/widths, final partial groups, request counts, and pass indices.

### 3.2 Experimental unit and execution order

A measured run/repetition is the energy experimental unit. Tweets and repeated
passes within it are not independent energy replicates. Retain every repetition
and summarize run-level variation.

Order is seed-determined and recorded in `plan.json`. Under
`engine-session-v1` the plan describes sessions keyed by model and repetition:
repetitions ascend, and model order within a repetition and width order within a
session are seeded. A configuration without `execution_protocol` remains a
historical plan: readable and verifiable, but `run` refuses it.

Within a repetition, one engine session per model starts in a fresh process. It
covers one unloaded idle baseline (no batch width) before setup, shared by all
widths of the session; one measured classical training/setup trial where
configured, likewise shared; then, per width, an optional loaded-idle trial
followed by inference. Widths stay separate conditions; the shared session does
not make their results poolable or ordered by speed. Engine startup capacity
never defines a measured width; the width is applied explicitly per condition.

The plan links each trial to its session and dependencies, and records the
realised width order. Adapter output is segment-scoped: a fresh log segment per
process or resume, with its own segment provenance. Results link session,
segment, log directory, and the realised width policy.

Record actual order, clocks, temperatures, power limits, and background state
where observable. Seeded interleaving reduces, but does not eliminate, order,
thermal, and temporal confounding.

## 4. Execution protocol

### 4.1 Ownership, preflight, and warmup

1. Obtain exclusive runtime/measurement ownership and stop unrelated activity.
2. Confirm lockfiles, hashes, snapshots, revisions, engine paths, and fresh
   output roots. Record CPU/thread settings, power state, memory, load, GPU
   inventory, free VRAM, and existing compute applications.
3. Before a GPU download or load, preflight weight bytes plus reserve. Refuse
   unsupported GPU visibility or competing work; record exact skips.
4. Never silently quantize, offload, lower precision, substitute a checkpoint,
   or change hardware. An explicit alternative is a new condition.
5. Inspect engine logs and adapter metadata. After loading a session, warm up
   every configured width with validation rows and the configured settle
   period; record workload, duration, stabilization, and cache state. Apply the
   section 9 readiness rule rather than a mandatory pilot sequence.
6. GPU checks before and after each measured window exclude the run's own
   adapter process group. Foreign compute prevents the trial from completing.
   These are point-in-time checks, not continuous interference monitoring.

Warmup is excluded from steady-state energy and quality. Model loading and cold
start are outside the primary boundary, not zero-energy events.

### 4.2 Measured inference boundary

Use one fresh CodeCarbon `OfflineEmissionsTracker` per trial in machine tracking
mode on the single host containing client and server. Resolve hardware, output
methods, schema validation, and task metadata before measurement. The final
meter uses `meter-worker/1` in a separate interpreter so its scheduler does not
share the client interpreter.

| Inside the timed window | Outside the timed window |
|---|---|
| Preprocessing and client queueing | Prediction serialization and audit |
| Model and server execution | Quality scoring |
| Output parsing into label/raw record | Report analysis |
| Permitted request overhead and worker sampling | Model fitting/loading and warmup |

The window starts immediately before the first request and ends after the last
result completes and is parsed. Outside work retains separate
`outside_window_s` timings. Worker handshakes and sampling load remain inside
the machine-level energy boundary.

Additional rules:

- Persist monotonic start/end and tracker overhead.
- Use `pue=1.0`; energy is the estimated component sum, not wall energy.
- Record whether an unused GPU remains powered and metered during CPU work.
- Primary requests do not retry. Any future retry-enabled condition must keep
  retries and energy in the same attempt and report retry counts.

### 4.3 Interruption and immutability

Atomically persist `state.json` and attempt histories. SIGINT/SIGTERM or an
abandoned checkpoint produces an incomplete/recoverable state. Completed trials
are never overwritten or silently repeated. Retries use fresh `attempt-NNNN`
directories; failed and skipped states require explicit retry flags. Preserve
all logs and native artifacts.

Resume continues the same session in a fresh log segment, so segment provenance
distinguishes processes. Completed trials are immutable. A resumed segment
may reuse a completed calibration from the same planned session; the loaded
idle's segment is linked explicitly. A previously measured classical pipeline
is fitted again outside measurement if it must be rebuilt on resume.

A QEV over-context envelope skips only that width's loaded-idle/inference
pair. A fatal engine failure stops the session: the active trial records its
failure or fit skip, and remaining dependent trials are skipped. No completed
trial is rewritten.

## 5. Energy measurements and calibration

### 5.1 Gross component energy

Retain native CodeCarbon CPU, GPU, RAM, and total energies in kWh. Record
backends, GPU indices, sample interval, source signature, and resolved hardware.
An unavailable component is **null**, not zero.

For `N_submitted > 0`:

\[
E_{gross} = \frac{3{,}600{,}000 \times E_{kWh}}{N_{submitted}}
\quad \mathrm{J/submitted\ tweet}.
\]

`N_submitted` counts actual tweet submissions across the panel and all passes—
not API calls, groups, envelopes, or valid predictions only. Keep completed,
valid, and correct counts for explicitly labelled alternatives.

Native BoAmps `powerConsumption` is energy in kWh despite its name. Training's
legacy derived intensity uses training rows and must not be labelled tweet
energy.

### 5.2 Idle adjustments

`device_idle` measures the same host/source configuration with no model
workload. For a compatible baseline with the same source signature, host, and
GPU UUID set:

\[
E_{device-adjusted} = \frac{3{,}600{,}000}{N_{submitted}}
\left(E_{gross,kWh} - E_{device,kWh}
\frac{T_{gross}}{T_{device}}\right).
\]

`loaded_idle` measures the already-loaded engine without submitting inference.
Apply analogous duration-scaled subtraction and label it
`loaded_idle_adjusted_j_per_tweet`.

- Never substitute device idle for loaded idle or pool baselines across models,
  engines, hosts, GPU sets, or source signatures.
- One unloaded `device_idle` (no batch width) per model/repetition serves every
  width in that session; `loaded_idle` is measured per width only where
  configured. A baseline never crosses sessions, repetitions, or models.
- Without a compatible baseline, adjusted energy is null.
- Retain signed values; negative results can expose mismatch or noise and must
  not be clipped.
- Keep gross, device-adjusted, and loaded-adjusted series separate. Device idle
  never enters inference-quality figures.

### 5.3 Training boundary

Measure task-specific classical/fastText training as `training`, separately
from inference and tuning. Record rows, parameters, threads, duration, and
native artifacts. Qwen/decision pretraining is outside the boundary or
unavailable, never zero. Include training in an inference ratio only under an
explicit amortization assumption.

A classical model/repetition is fitted once and reused across that session's
widths, analogous to one GPU engine session. The training trial is shared
reference work, not duplicated or re-counted per width; each width trial links
the shared trial.

## 6. Quality and performance

### 6.1 Prediction audit and denominators

Every submitted tweet has a durable record containing test ID, gold label,
prediction or null, raw output, completion state, latency and scope, pass index,
group index, and actual width. Audit cardinality and panel/group order before
acceptance.

Report:

- submitted, completed, valid, and correct counts;
- panel size, passes, groups, and final group width;
- retries, vLLM HTTP requests, and applicable QEV choice/envelope counts;
- repeat disagreements from the first pass.

Invalid or missing predictions remain in the denominator and count as wrong.
Coverage is the fraction with a valid three-class label; failure is not a fourth
class.

Primary quality uses the **first pass only** (`basis_pass=0`,
`basis_n=panel_size`) and is computed outside the energy window. Later passes
diagnose disagreement, not additional independent quality observations.

Report official TweetEval macro recall over negative/neutral/positive, plus
macro-F1, per-class precision/recall/F1/support/predicted/correct, accuracy, and
coverage. Interpret empty-denominator scoring with the panel support table. The
quality–energy plot pairs first-pass macro recall with gross J/submitted tweet
over the full repeated workload.

### 6.2 Latency and throughput

End-to-end latency includes queueing. Use adapter request latency when
available; otherwise explicitly label closed-loop group completion as the
fallback. Report p50/p95 with scope and completed-request basis.

\[
\mathrm{throughput}_{tweets/s} =
\frac{N_{completed}}{T_{full\ measurement\ window}}.
\]

Valid throughput may use `N_valid` only with an explicit label. Training's
legacy `throughput_tweets_s` contains training rows/s, not tweet throughput.

For the width comparisons at 1, 16, and 64:

- compare engines using group p95, one observation per closed-loop group;
- do not repeat packed-request latency for every output;
- report independent HTTP request p95 separately for vLLM;
- show final partial groups, scheduler settings, session order, and width order.

## 7. Native BoAmps export and verification

Every trial emits CodeCarbon CSV and native BoAmps JSON. Metadata identifies
stage/task family, actual request count, a nonempty algorithm array, and a
nonempty dataset array with usage/type and source.

Request-count rules:

- Calibration and training use native `nbRequest=0`; inference may not replace
  a missing count with zero.
- Packed QEV counts request envelopes, including final partial groups, not
  tweet decisions. The verifier checks envelope receipts in predictions.
- Gross and adjusted J/tweet still divide by submitted tweets.
- In vLLM, request and tweet counts coincide.

Verification is an acceptance gate:

1. Exactly one native report validates offline against five hash-checked BoAmps
   schemas at the pinned commit, with local Draft-04 references. Only
   `algorithm_schema.json`'s validation-inert `required: []` is omitted from
   the in-memory validator copy for metaschema compatibility; source bytes and
   exports are unchanged.
2. Native format is 0.1; report ID, CSV UUID, filename, and native header agree.
3. Native/CSV duration and total energy agree within strict tolerance; total
   equals the detected component sum.
4. Tracking mode, CodeCarbon version, task metadata, PUE, monitoring metadata,
   and schema pin agree.
5. Components without a detected source remain unavailable; invalid native
   output fails the pipeline.
6. Inference has at least `min_samples=5` scheduled samples when that counter
   is available. Short/low-sampling trials remain flagged.
7. The laptop run additionally checks fixed-RAM integration coverage:
   `3,600,000 × RAM kWh / (RAM W × native duration) >= 0.85` for inference.
   This checks backend consistency, not absolute accuracy or variable-power
   RAM sensors.

Retain native reports, CSV, monitoring metadata, logs, predictions, manifests,
and schema pins together. Later BoAmps transformations must preserve native
units and be documented separately.

## 8. Analysis and reporting

- Analyze one phase root at a time; label any deliberate join.
- Keep phase, kind, model, width, repetition, session, segment, host, engine,
  and condition.
- Include failed/incomplete artifacts and skips in status reports. Successful
  figures require a readable active result and complete state.
- Preserve run-level variation; never treat tweets or passes as independent
  energy replicates.
- Keep missing values null and unavailable components distinct from zero.
- Separate inference/training/calibration, gross/adjusted energy, and
  request/group latency. Quality figures include successful inference only.
- Show quality, components, intensity, throughput, and latency with condition
  context. A quality–energy trade-off does not establish causality.
- Preserve raw artifacts beside every figure, CSV, or aggregate. Never fabricate
  or use placeholder results.

## 9. Acceptance sequence

The named `gpu_pilot` and `cpu_pilot` phases are historical. Use this readiness
rule for the active engine-session protocol:

- **Validated setup:** proceed directly to the intended measurement.
- **New host or relevant runtime/meter change:** run a short readiness check
  scoped to what changed. Do not repeat unrelated pilots.

The readiness check must establish that:

1. Each intended affected model fits without silent condition changes and
   returns valid predictions. Relevant engine/source, attention/readout, and
   output checks still apply.
2. The shortest planned measured window for each affected condition collects
   adequate samples and shows no obvious under-integration. Faster hosts can
   shorten windows, so a previously adequate longer window is not sufficient
   evidence.
3. Configuration, input integrity, native export, and prediction audits pass
   for the affected path. Static tests and weight-byte preflight alone do not
   establish runtime or measurement readiness.
4. A session loads its model once and serves every configured width, and each
   width's shortest window samples adequately. Fit and sampling evidence for the
   new widths remain future checks: they are not inherited from width 1 or 16,
   and not from a historical sweep.

Every session retains its unloaded (`device_idle`) calibration and every
configured width its applicable model-loaded (`loaded_idle`) calibration as
analysis baselines; these are not pilots. Complete the artifact verification in
section 7 before reporting.

## 10. Limitations and threats to validity

1. **Complete-system confounding.** Model, runtime, interface, readout, hardware,
   and batching can vary together; differences are not architecture-only causal
   effects.
2. **Estimated boundary.** CodeCarbon CPU/GPU/RAM with PUE 1 is not whole-system
   or wall energy. CPU/RAM values may be model-based; an unused powered GPU can
   affect CPU trials.
3. **Sampling.** Tracker interval and overhead can materially affect fast
   classifiers. Fixed pass counts do not remove sampling error; report durations,
   actual samples, and run variation.
4. **Calibration.** Idle subtraction can be noisy or negative. Duration scaling
   and signature matching do not remove temporal, thermal, or loaded-baseline
   mismatch.
5. **Workload equivalence.** Static CPU groups, independent vLLM bursts,
   baseline QEV choices, and packed QEV envelopes have different scheduling,
   context, and readout semantics. Client width does not prove hardware batching.
6. **Dependence and order.** Repeated passes, widths of one session, and runs
   share tweets and process state. Seeded interleaving and per-session width
   ordering cannot remove thermal, drift, or within-session dependencies.
7. **Baseline fidelity and tuning.** SVM and fastText are variants, not exact
   published baselines; no validation tuning is performed.
8. **Panel generalization.** Fixed subsets of 49/65/257 tweets support controlled
   comparison, not automatic generalization. Report support and first-pass
   quality.
9. **Contamination.** Pretrained models may have seen TweetEval or related text;
   a newer held-out dataset is needed for contamination-resistant evaluation.
10. **Runtime evidence.** Unexecuted or failed systems are planned/skipped, not
    measured. Static checks do not prove CUDA fit or QEV numerical correctness.
11. **Training boundary.** Large-model pretraining is excluded/unavailable; no
    lifecycle-energy claim is supported.
12. **Host specificity.** Results apply to the recorded host, power state,
    threads, locks, commits, and revisions. Another environment is a new
    condition.
13. **Latency scope.** Request latency and group-completion fallback are not
    interchangeable; packed and independent requests need distinct reporting.
14. **Repeat quality.** First-pass scoring avoids label pooling but does not
    characterize all repeat variability; report disagreements separately.
15. **Session sharing.** One engine process, one unloaded idle, and one classical
    fit serve several widths. This is by design and keeps reload/refit work out
    of the measured window; it is not a measured speedup or energy saving, and
    it means widths share allocator state, caches, and thermal history. A
    session-scoped baseline is not evidence for another session or model.
