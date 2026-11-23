# Sentiment energy benchmark

Compare quality, latency, throughput, and CodeCarbon-estimated CPU/GPU/RAM
energy for complete sentiment-classification systems on TweetEval. This bespoke
benchmark does **not** use Inspect.

A result belongs to a **whole system**—model, engine, input format, output
mechanism, hardware, and workload—not to a model in isolation.

## At a glance

| | |
|---|---|
| Task | Three-class TweetEval sentiment classification |
| Completed measurements | Historical/verification only: GPU width sweep (10 conditions, Qwen/Joey, 30 inference trials) and baseline matrix (5 systems, 27 inference trials) |
| Shared test panel | 257 tweets; larger checkpoints that did not fit were skipped, not altered |
| Systems | Qwen3.5-2B/vLLM, Joey-1/OneJev QEV, linear SVM, multinomial naive Bayes, fastText |
| **Measurements** | [RESULTS.md](RESULTS.md): findings, run-level tables, and verification evidence |
| **Protocol** | [DESIGN.md](DESIGN.md): exact conditions, metrics, readiness rule, and limitations |

## How to read the measurements

| Measure | Basis |
|---|---|
| Primary energy | Gross CodeCarbon CPU + GPU + RAM Joules per **submitted tweet across all passes** |
| Adjusted energy | Device-idle- or loaded-idle-adjusted; reported separately, never pooled with gross energy |
| Primary quality | **First pass only**; later passes diagnose repeat disagreement |
| Experimental unit | One measured run; tweets and repeated passes are not independent energy replicates |

Training and calibration are separate from inference. CodeCarbon provides
component estimates with PUE 1, not wall-meter or whole-system energy.

### What “width” means

Width is not a common internal GPU batch size:

- **Classical systems:** static groups of tweets.
- **Qwen:** closed-loop bursts of independent HTTP requests at widths 1, 16, and
  64, while vLLM's scheduler remains capped at 16; wider bursts can queue. The
  historical sweep also used 4 and 256.
- **QEV single-request mode:** the adapter default, one independent tweet per
  choice request; not the active phase's condition. Branch batching inside a
  decision is not independent-tweet batching.
- **Packed QEV (active protocol):** one numbered, shared tweet list with one
  targeted question per tweet, available to every pinned QEV sibling (Joey-1,
  Elephant-1, Mammoth-1) at widths 1, 16, and 64. Other tweets enter the
  context, so this distinct condition can change quality. Exact tokenizer checks
  reject over-limit envelopes before inference; inputs are never truncated to
  fit. The historical sweep packed Joey only.

Width 1 is the common primary comparison. See
[DESIGN.md](DESIGN.md#10-limitations-and-threats-to-validity) before comparing
across engines or making architecture-level claims.

## Reproduce

Two routes exist. The **guided launcher** previews a plan, provisions the pinned
client and offline inputs, checks readiness, measures, verifies, and writes a
local report. The **manual route** keeps the individual `benchmark.py` commands
for auditing, for another host, and for finer control; the launcher does not
replace it.

### Before you start

Use one owner at a time for GPU work, downloads, installation/compilation, and
every measured window, including CPU-only and idle calibration. Ensure a quiet
host and never start while another measurement is active.

Completed trials and native reports are immutable. Always use a **new artifact
root** for a new measurement; an existing root resumes its stored plan.

Older published profiles (`profiles/laptop-full-isolated-02.json`,
`profiles/laptop-cpu-isolated-02.json`, `profiles/laptop-gpu-widths.json`) and
their artifact roots are **historical/verification inputs only**: read-only,
with immutable versions and hashes and no switch back to a previous execution
protocol.

### Guided route: the launcher

`run_experiment.py` bootstraps the pinned client and the selected engines, checks
readiness, measures, verifies, and reports; `benchmark.py` still owns every
measurement, and the launcher adds no experimental condition.

#### Prerequisites

- Linux `x86_64` and a **bootstrap Python 3.11+** (standard library only), with
  `uv` on `PATH`; `uv` may fetch the pinned 3.12.13 client interpreter.
- GPU systems need one visible NVIDIA GPU 0 with compute capability 8.0+ and the
  CUDA 12.8 GA driver (570.26+). With no compatible GPU, only the CPU systems are
  selected and no GPU is metered.
- Kernel driver, CUDA compatibility, free disk, and network access are yours to
  provide. The launcher uses no `sudo` and uploads nothing.
- Engine environments (vLLM, patched OneJev QEV) and model weights are **not**
  prerequisites: they are installed or fetched only after you approve the plan.

#### Quick start

```bash
# Recommended first: a short end-to-end smoke test of the whole pipeline
python3 run_experiment.py mini --country FRA --dry-run  # plan only: no installs, downloads, writes, measurements
python3 run_experiment.py mini --country FRA            # confirm, then install + measure + verify + report

# Then the measurement
python3 run_experiment.py full --country FRA --out artifacts/full_engine_session_02
```

`--country` is a required ISO 3166-1 alpha-3 code for a new run and is recorded as
`energy.country_iso_code`. `--out` must not exist; the default root is a
UTC-stamped `artifacts/<mode>-<stamp>Z`, and the printed root is what you resume.
Without a terminal, add `--yes` to approve the installs and downloads up front. The
mode and `--country` are rejected with `--resume`, which reuses the saved ones.

#### What the launcher does

1. **Plan.** Lists `Include`/`Exclude` models with reasons and a conservative
   weight-download estimate. Selection uses pinned weight bytes plus the configured
   reserve against GPU 0's total VRAM, so transient occupancy never drops a model;
   active foreign compute, a current free-VRAM shortfall, insufficient disk, or
   an unreadable GPU probe blocks the run. If GPU execution is unsupported, the
   launcher offers the classical systems only.
2. **Install, after approval only.** Hash-checked dataset/schema inputs, the
   client `.venv` (pinned 3.12.13), the selected engine environments under
   `.scratch/launcher/`, then weights one model at a time through
   `setup/gpu/fetch_models.sh`, whose VRAM, disk, and foreign-compute guard runs
   again before each download. A post-approval shortfall fails setup instead of
   silently skipping a model.
3. **Readiness, measurement, verification, and report** as described below.

Provisioning logs to `.scratch/launcher/provision.log`. It reuses validated
launcher environments by receipt and refuses to replace unrelated environments. Qwen2B keeps its historical `local_snapshot`
from the frozen manifest, a path in this checkout's writable parent
(`../inspect_pilot_study/.scratch/gpu/model`), so those weights land there rather
than inside the benchmark repository.

#### Full-run readiness

A new `full` run without `--validated` first measures **one separate repetition of
the same test workload** — selected models, selected widths, configured panel and
pass counts — under `<root>/readiness/`, with its own resolved configuration and
verification. It implements the [DESIGN.md §9](DESIGN.md#9-acceptance-sequence)
rule instead of a named pilot, can cost about a third of the run, and is not a
result: it is never pooled with or reported beside the measured repetitions, and
resuming the same root reuses completed readiness trials. Measurement starts only
if every readiness trial completes, verification passes, sampling is adequate, and
inference returned valid predictions.

`--validated` replaces that repetition with your attestation that this exact setup
and these conditions already passed readiness on this host. If a completed
readiness trial fails the sampling gate, review its logs and use an explicitly
revised configuration and fresh root via the manual route; do not bypass the gate.

#### `mini` is a smoke test

`mini` keeps phase `full` and the full classical training split but shrinks the
measured workload to width 1, 9 test tweets (3 warmup), one pass, one repetition,
and 2 s idle/settle; classical fits still happen in session setup, and training is
not measured (`measure_training=false`). It exercises installation, engines,
metering, exports, verification, reporting, and resume end to end.

Its energy, latency, throughput, and quality numbers are **not comparable** with a
full run, with [RESULTS.md](RESULTS.md), or with each other, even when verification
passes; its report is marked `SMOKE ONLY`. It also does not establish that the full
workload fits or samples adequately, so it is not readiness evidence.

#### Artifacts

`launcher.json` (mode, selection/exclusions, same-host guard, readiness policy,
config digest), `resolved.json` (the frozen configuration; resume requires it
unchanged), `plan.json` and one directory per trial with `attempt-NNNN/` (result,
predictions, CodeCarbon CSV, native BoAmps JSON, QEV receipts, logs), `readiness/`,
`verification.json`, `launcher-logs/invocation-NNNN/*.json`, and
`report/summary.md` plus `report/*.csv`.

Excluded models are not attempted: report them as such, never as zero energy, a
failed measurement, or evidence about a model. The download estimate counts a
selected GPU model as zero only when `config.json` and every pinned shard match
their manifest sizes; otherwise it counts the full shard sum, and the disk
requirement adds the 5% margin of `setup/gpu/fetch_models.sh`. It excludes Python,
packages, and model metadata. Approve only after checking that the selection fits:
nothing is silently quantized, offloaded, or substituted.

#### Later and reporting

```bash
python3 run_experiment.py --resume artifacts/full_engine_session_02
python3 run_experiment.py --resume artifacts/full_engine_session_02 --retry-failed
.venv/bin/python benchmark.py status --phase full --out artifacts/full_engine_session_02
.venv/bin/python launcher_report.py artifacts/full_engine_session_02
```

Resume is same-host only and requires the unchanged `resolved.json`. Completed
trials are reused; failed or skipped trials are retried only with an explicit
`--retry-failed` or `--retry-skipped`, in fresh attempts, without changing the
condition. `launcher_report.py` reads artifacts, never measures, and writes only
inside `report/`; `analysis.py` and the notebook remain available for tables and
figures.

Exit status: `0` successful (including a preview or cancellation), `1` readiness
failed or the run is not fully complete and verified, `2` configuration, setup,
or preflight error, `130` interrupted.

### Manual route (unchanged)

These commands remain supported for read-only verification, another host, and
finer control.

#### 1. Install the client and fetch offline inputs

The supported client is **Python 3.12.13**:

```bash
uv venv --python 3.12.13 .venv
uv pip sync --python .venv/bin/python requirements.lock
.venv/bin/python -c 'import sys, pandas, altair, marimo; print(sys.version); print(pandas.__version__, altair.__version__, marimo.__version__)'
```

After obtaining download permission:

```bash
bash setup/fetch_inputs.sh
sha256sum -c setup/inputs.sha256
```

This fetches no model weights. It pins TweetEval to
`4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66` and BoAmps schemas to
`af1680c56ce2cefd5777fcac1c57591b685d6be5`; the runner rechecks their sizes and
hashes offline.

Keep engine dependencies out of the client environment:

- **Qwen/vLLM:** follow [setup/gpu/README.md](setup/gpu/README.md).
- **Joey/QEV:** follow [qev_setup/README.md](qev_setup/README.md), including the
  exact local snapshot, OneJev commit, checkpoint patch, and runtime checks.
- **Classical systems:** use the client environment and train only on the
  training split.

Preflight free VRAM before downloading or loading weights. Never silently
quantize, offload, substitute a model, or change precision to make a condition
fit.

#### 2. Verify historical published runs (read-only)

These commands audit existing predictions and native energy reports; they do
not load a model and produce no new measurements:

```bash
.venv/bin/python benchmark.py verify --phase full \
  --config profiles/laptop-full-isolated-02.json \
  --out artifacts/full_isolated_02
```

These need the published artifact roots to be present on the current host.

The reported full profile uses the isolated sampler at 0.2 s. That run and the
ten-condition width sweep (`profiles/laptop-gpu-widths.json`) are
historical/verification only and are not re-executed under the current
protocol. The corrected CPU pilot (`profiles/laptop-cpu-isolated-02.json` and
`artifacts/cpu_pilot_isolated_02`) and earlier pilots are preserved as
historical engineering evidence, not prerequisites or quantitative results.

#### 3. Start a fresh run

`experiment.json` defines the active `full` phase. The original pilot
configurations are archived in [`profiles/legacy-experiment.json`](profiles/legacy-experiment.json)
for verification only.

**Active protocol (`engine-session-v1`).** The full run schedules all nine
models at widths 1, 16, and 64. Each model gets one fresh engine per repetition:
a new worker process for GPU models, or a new fit for classical pipelines. That
engine serves all three widths in seeded order. Model order is also seeded
within each repetition. Each session shares one unloaded idle calibration and,
for classical models, one measured fit. Widths retain separate warmup, settle,
inference, and model-loaded idle measurements where configured.

Widths remain separate conditions: session reuse removes excluded setup work
(reloading, refitting), which is not a measured energy or performance gain, and
width results are never pooled.

**Readiness rule:** on a validated setup, go directly to measurement. On a new
host, or after a relevant runtime or meter change, do a short readiness check
only for what changed. Confirm that the intended models fit and return valid
predictions, and that the shortest planned measured window collects adequate
samples without obvious under-integration. Faster hosts can shorten windows,
so check the shortest affected condition rather than assuming the previously
used window length is sufficient. See
[DESIGN.md](DESIGN.md#9-acceptance-sequence).

Per-session unloaded (`device_idle`) and per-width model-loaded (`loaded_idle`)
calibrations where configured remain part of measurement as analysis baselines;
they are not pilots.

A fresh active-protocol full run uses a new root:

```bash
OUT=artifacts/full_engine_session_01
.venv/bin/python benchmark.py plan --phase full --config experiment.json --out "$OUT"
.venv/bin/python benchmark.py run --phase full --config experiment.json --out "$OUT"
.venv/bin/python benchmark.py verify --phase full --config experiment.json --out "$OUT"
```

`plan` resolves metadata without loading models; it does not prove runtime fit.
`run` includes loading, warmup, calibration, training, and inference. Check its
status without launching models:

```bash
.venv/bin/python benchmark.py status --phase full --out "$OUT"
```

To resume, reissue the same `run` command. Incomplete trials receive fresh
`attempt-NNNN` directories. Failed or skipped trials require `--retry-failed`
or `--retry-skipped`; a retry does not authorize changing the condition.

##### Historical width sweep (verification only)

The ten-condition sweep at widths 1, 4, 16, 64, and 256 used a per-width engine
lifecycle and packed Joey only; it predates the engine-session protocol. Do not
re-run or pool it. Audit the published root read-only:

```bash
.venv/bin/python benchmark.py verify --phase gpu_widths \
  --config profiles/laptop-gpu-widths.json \
  --out artifacts/gpu_widths_01
```

Apply the readiness rule above to any future run; no named pilot phase is
required.

## Analyze a run

Each phase is a separate analysis root. Its `plan.json`, run states, attempt
histories, predictions, CodeCarbon CSV, monitoring metadata, logs, and native
BoAmps JSON form the audit trail. Analysis reads only the active attempt's
`result.json`; earlier attempts remain available for failure auditing.

Print trial summaries or export six tidy CSV tables:

```bash
.venv/bin/python analysis.py artifacts/full_engine_session_01
.venv/bin/python analysis.py artifacts/full_engine_session_01 \
  --output-dir analysis_tables/full_engine_session_01
```

Open the run-level notebook and select one artifact root:

```bash
.venv/bin/marimo edit analysis_notebook.py
# Or serve a read-only session:
.venv/bin/marimo run analysis_notebook.py
```

Missing values remain null, not zero. Figures keep phases and workload kinds
separate, distinguish gross and adjusted energy, and retain request versus
group latency. Run the offline analysis tests with:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/test_analysis.py -q
```

Analysis and tests do not authorize inference or downloads. Avoid exports and
other background work during measurement.

## Run on another host

Create a hardware-specific profile and fresh artifact root. Apply the targeted
readiness rule above to the intended models, widths, and shortest planned
windows; no named pilot phases are required. Previous validation and idle
baselines do not transfer, so establish fresh per-session calibrations.
Preserve the panel, checkpoints, engine versions, precision, prompts, widths,
and `execution_protocol` where possible, and label every difference.

The guided route above does this on a new host: it resolves a fresh root, applies
the fit-based selection, records the new host's metadata, and runs an automatic
readiness repetition — or accepts `--validated` when the setup and conditions are
unchanged. Never copy a launcher root between hosts; resume is same-host only.

This checkout targets one NVIDIA GPU at index 0. Multi-GPU use requires an
explicit, verified measurement boundary. Faster hardware may need more passes
for adequate sampling; fix counts before execution and report actual samples
and cadence.

## Reproducibility boundary

Git tracks code, profiles, documentation, and derived analysis in `results/`.
Raw artifacts, datasets, weights, environments, and notebook caches stay on the
benchmark host. A fresh clone can read derived results but needs the raw
artifacts for full verification, rehashing, and rescoring.

Before redistribution, review host metadata, local identifiers, prediction
text, dataset terms, and native BoAmps metadata. Keep historical engineering
evidence distinct from final measurements, and never turn missing or failed
measurements into zero or fabricated results.
