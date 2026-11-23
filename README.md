# Sentiment energy benchmark

Bespoke, offline-artifact tooling for comparing the quality, latency, throughput,
and CodeCarbon-estimated component energy of complete sentiment-classification
systems on TweetEval. The benchmark does **not** use Inspect and does not
attribute differences to a model alone when its runtime, input format, hardware,
or output mechanism differs.

This repository currently contains the implementation and analysis workflow,
not a results report. See [DESIGN.md](DESIGN.md) for the experimental protocol,
acceptance gates, metric definitions, and limitations.

## Safety and ownership

One coordinator-approved owner at a time may perform GPU work, model downloads,
dependency installation/compilation, or any tracked window, including CPU-only
and idle calibration. Do not launch a measured run while another owner is
measuring. Avoid unrelated background activity on the benchmark host.

- `plan`, `status`, `verify`, the analysis CLI, tests, and offline marimo export
  do not launch inference models. `plan` and `run` use
  `.scratch/measurement.lock`; `status` and `verify` are read-only.
- `run` requires an explicit runtime/measurement handoff. Model setup, smoke
  inference, warmups, calibration, training, and inference are all runtime work.
- Preflight GPU capacity before downloading or loading weights. A failed fit is
  recorded as a skip; precision, quantization, offload, hardware, and model
  identity are never silently changed.
- Completed trials and native reports are immutable. Retry only in a new
  `attempt-NNNN` directory.
- Do not redistribute artifacts before reviewing host metadata, local paths and
  identifiers, prediction text, dataset terms, and native BoAmps metadata.

## Repository map

| Path | Purpose |
|---|---|
| `experiment.json` | Validated phase, model, workload, repetition, and energy configuration |
| `configuration.py` | Configuration selection and validation |
| `benchmark.py` | `plan`, `run`, `status`, and `verify` command-line entry point |
| `dataset.py` | Pinned offline TweetEval adapter and deterministic panel selection |
| `adapters/` | Classical, vLLM, and reference-QEV model adapters |
| `workload.py` | Closed-loop workloads, persisted predictions, and out-of-window scoring |
| `energy.py`, `meter_worker.py` | Isolated CodeCarbon windows, calibration adjustments, and native BoAmps validation |
| `storage.py` | Atomic artifacts, stable identities, locks, and attempt histories |
| `analysis.py` | Null-preserving pandas summary CLI |
| `analysis_notebook.py` | Marimo/Altair run-level tables and figures |
| `qev_setup/README.md` | Decision-model provenance and reference-runtime acceptance gates |
| `requirements.in` | Direct client/analysis dependencies |
| `requirements.lock` | Hash-pinned transitive client/analysis environment |
| `setup/fetch_inputs.sh` | Hash-checked TweetEval and BoAmps schema fetch; no model weights |
| `setup/inputs.sha256` | Expected SHA-256 for every downloaded input/schema file |
| `tests/` | Offline synthetic/unit tests; not benchmark runs |

## Reproducible client and analysis environment

The supported client interpreter is **Python 3.12.13**. The lock file was
generated with `uv pip compile --generate-hashes`.

After the coordinator grants installation permission, create a clean client
environment from the lock:

```bash
uv venv --python 3.12.13 .venv
uv pip sync --python .venv/bin/python requirements.lock
.venv/bin/python -c 'import sys, pandas, altair, marimo; print(sys.version); print(pandas.__version__, altair.__version__, marimo.__version__)'
```

Use `.venv/bin/python` and `.venv/bin/marimo` directly; shell activation is not
required. Do not install vLLM, QEV, PyTorch, or checkpoint-specific constraints
into this client lock. Their separate engine environments are described below
and in `qev_setup/README.md`.

### Pinned dataset and BoAmps schemas

`setup/fetch_inputs.sh` downloads only the TweetEval sentiment files and five
BoAmps schema files, verifies their SHA-256 values, and performs no model
operations. It still requires download permission:

```bash
bash setup/fetch_inputs.sh
sha256sum -c setup/inputs.sha256
```

TweetEval is pinned to revision
`4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66`; expected train/validation/test
sizes are 45,615/2,000/12,284. BoAmps schemas are pinned to
`af1680c56ce2cefd5777fcac1c57591b685d6be5`. The runner verifies input sizes and
hashes again offline. Keep the raw input files and checksum manifest with the
experiment provenance.

### Engine environments and model snapshots

- **vLLM:** use the separately pinned Python environment with **vLLM 0.19.0**
  and the configured local/Hub revision. The primary condition is BF16,
  deterministic decoding, thinking disabled, and prefix caching disabled.
- **QEV decision models:** use each checkpoint's own installer and isolated
  reference environment, not the client environment or a generative Qwen
  loader. The reference card uses OneJev commit
  `81ce62f1597c91e46767d4d02d6ac2e18534fe94` plus the checkpoint's pinned
  `serve.patch`. Configure absolute local snapshot, patched OneJev, and Python
  paths in `experiment.json`. See `qev_setup/README.md` for exact revisions,
  image/dependency constraints, attention/readout checks, and smoke gates.
- **Classical models:** SVM, multinomial NB, and from-scratch fastText run in the
  client environment. They fit on the training split only; validation is not
  used for tuning in the current configuration.

Model downloads and engine installation are separate coordinator-owned steps.
The configuration's local snapshot paths and exact revisions are part of the
experimental condition, not conveniences that can be changed to make a run fit.

## Configuration

`experiment.json` defines three phases: `gpu_pilot`, `cpu_pilot`, and `full`.
Select one phase per command. The validated programmatic interface is:

```python
from configuration import configure
phase = configure("gpu_pilot", "experiment.json")
```

Important configuration semantics:

- `sample_size` and the panel seed determine a fixed set of zero-based TweetEval
  test IDs; warmup rows come from validation with the derived seed.
- `passes` is configured **per model and width**, for example
  `{"svm": {"1": 200, "16": 1200}}`. JSON width keys are strings. Pass counts
  are fixed before execution and must not be tuned afterward to improve a
  result.
- Width 1 is the common primary comparison. Width 16 means static independent
  groups for classical pipelines and closed-loop bursts of independent requests
  for vLLM. QEV supports only one independent tweet per choice request.
- Repetitions are run-level observations, not pools of independent tweets.
- `energy` pins the ISO3 country code, sampling interval, and included GPU IDs.
  Device-idle, loaded-idle, and settling durations are explicit settings.

The planned sample/pass settings and phase progression are specified in
[DESIGN.md](DESIGN.md). Treat `experiment.json` as authoritative for resolved
values and preserve it with the artifacts.

## Runner commands

All commands below read `experiment.json` and write under
`artifacts/<phase>/` by default. These are subcommand options: `plan`, `run`,
and `verify` accept `--config PATH`; all four accept `--out PATH`. For example:

```bash
.venv/bin/python ./benchmark.py plan --phase full --config experiment.json --out artifacts/full
```

`status` reads the existing `plan.json` and therefore does not need `--config`.

**To verify the completed laptop benchmark**, use the exact measured profile,
not the initial defaults (see [RESULTS.md](RESULTS.md)):

```bash
.venv/bin/python benchmark.py verify --phase full \
  --config profiles/laptop-full-isolated-02.json --out artifacts/full_isolated_02
```

The corrected CPU pilot uses `profiles/laptop-cpu-isolated-02.json` and
`artifacts/cpu_pilot_isolated_02`. Both used the isolated sampler at 0.2 s.
Earlier in-process pilot artifacts are retained as engineering evidence, not
combined with the final results. Start a fresh artifact root if intentionally
collecting new repetitions rather than resuming the stored plan.

### 1. Plan and preflight

Planning resolves configuration and dataset metadata without launching models:

```bash
.venv/bin/python ./benchmark.py plan --phase gpu_pilot
.venv/bin/python ./benchmark.py plan --phase cpu_pilot
.venv/bin/python ./benchmark.py plan --phase full
```

Review selected snapshots/revisions, GPU preflight evidence, supported widths,
calibration jobs, pass counts, output paths, and any recorded skips before a
runtime handoff. Planning is not evidence that a model loads or produces valid
outputs.

### 2. Run in the required order

After explicit handoff and a quiet-host check:

```bash
.venv/bin/python ./benchmark.py run --phase gpu_pilot
.venv/bin/python ./benchmark.py run --phase cpu_pilot
.venv/bin/python ./benchmark.py run --phase full
```

Do not advance to the next phase until the previous phase's acceptance gates in
`DESIGN.md` pass. Useful bounded/selected invocations are:

```bash
# At most N newly attempted planned runs, including calibration runs.
.venv/bin/python ./benchmark.py run --phase full --limit-new N

# One configured model ID or one workload kind.
.venv/bin/python ./benchmark.py run --phase full --model MODEL_ID
.venv/bin/python ./benchmark.py run --phase full --kind device_idle
.venv/bin/python ./benchmark.py run --phase full --kind training
.venv/bin/python ./benchmark.py run --phase full --kind loaded_idle
.venv/bin/python ./benchmark.py run --phase full --kind inference

# Request an orderly stop using a coordinator-chosen path.
.venv/bin/python ./benchmark.py run --phase full --stop-file .scratch/STOP
```

SIGINT/SIGTERM marks the active attempt incomplete. Do not kill unrelated
processes. Adapter lifecycle code terminates only its own server/worker process
group.

### 3. Resume without overwriting

Reissue the same `run` command to resume. `incomplete` and recovered stale
`running` states are retried automatically in fresh attempt directories;
`complete` states are never silently repeated. Failed or skipped states require
an explicit decision:

```bash
.venv/bin/python ./benchmark.py run --phase full --retry-failed
.venv/bin/python ./benchmark.py run --phase full --retry-skipped
```

A retry flag does not authorize changing hardware, precision, model identity,
or workload settings. If the configuration changes, create a distinct planned
condition rather than mutating completed evidence.

### 4. Inspect and verify

```bash
.venv/bin/python ./benchmark.py status --phase full
.venv/bin/python ./benchmark.py verify --phase full
```

`status` summarizes durable states. `verify` checks existing artifacts and does
not launch models. It is not a substitute for runtime attention/output/fit
acceptance gates.

## Artifact layout

Each phase is a separate analysis root:

```text
artifacts/<phase>/
├── plan.json
└── <run_id>/
    ├── state.json
    └── attempt-NNNN/
        ├── result.json
        ├── predictions.jsonl          # inference only
        └── energy/
            ├── emissions.csv
            ├── boamps_report_<uuid>.json
            ├── monitoring.json
            └── codecarbon.log
```

Engine-specific logs and resolved adapter metadata are retained beside these
files. `state.json` records the full trial, current status, active attempt, and
attempt history. The analysis loader reads only the active attempt's
`result.json`; inspect attempt history separately when auditing failures.
Native energy output remains in its original units and location.

## Analysis

### Pandas CLI

Print status and unaggregated trial summaries without writing files:

```bash
.venv/bin/python analysis.py artifacts/gpu_pilot
.venv/bin/python analysis.py artifacts/cpu_pilot
.venv/bin/python analysis.py artifacts/full
```

Write six tidy CSV tables to a separate output directory:

```bash
.venv/bin/python analysis.py artifacts/full --output-dir analysis_tables/full
```

The outputs are `runs.csv`, `trials.csv`, `class_quality.csv`,
`energy_components.csv`, `energy_intensity.csv`, and `performance.csv`. Missing
artifacts or fields become empty CSV cells, never zero. Each trial retains its
phase, kind, model, width, repetition, run identity, and serialized
trial/dataset/host/adapter context. Analyze phases separately unless a later
report deliberately joins them with explicit labels.

### Marimo notebook

Open the read-only analysis interface:

```bash
.venv/bin/marimo edit analysis_notebook.py
# or, for a served read-only session:
.venv/bin/marimo run analysis_notebook.py
```

Set the artifact root to one phase directory. Figures show individual measured
trials; quality includes successful inference only. Energy components and
latency retain kind/phase facets; per-tweet energy intensity and tweet
throughput are inference-only so training rows are not mislabelled as tweets.
CPU/GPU/RAM components, gross/device-idle-adjusted/loaded-idle-adjusted energy,
and p50/p95 latency are never collapsed into one measurement basis.

For an offline HTML export with no live notebook server, use an empty temporary
fixture root:

```bash
fixture="$(mktemp -d)"
SENTIMENT_BENCHMARK_ARTIFACT_ROOT="$fixture" \
  .venv/bin/marimo export html analysis_notebook.py \
  --no-include-code -o "$fixture/notebook.html"
```

The environment variable only changes the analysis notebook's default artifact
root. Exporting real artifacts is read-only but should still be done outside a
measurement window to avoid host background work.

## Rerun on a different host

The laptop run is one hardware profile, not the definitive matrix. On the
planned higher-memory host, keep its results in a new artifact root. Do not
resume the laptop's plan there: identical selections can produce identical
logical run IDs, and neither laptop idle baselines nor laptop GPU-fit decisions
transfer. For example, after staging the pinned runtime and snapshots:

```bash
.venv/bin/python benchmark.py plan --phase full \
  --config experiment.workstation.json --out artifacts/workstation/full
.venv/bin/python benchmark.py run --phase full \
  --config experiment.workstation.json --out artifacts/workstation/full
.venv/bin/python benchmark.py verify --phase full \
  --config experiment.workstation.json --out artifacts/workstation/full
```

Run the GPU and CPU pilots in their own workstation roots before the matrix.
Re-establish idle baselines, actual sampling adequacy, and model fit there.
This checkout targets one physical NVIDIA GPU at index 0; use that topology or
extend and verify GPU selection and the measurement boundary explicitly before
running on a multi-GPU host. An 80 GB single GPU is a sensible candidate for
the two unquantized 27B conditions under the current memory settings, not a fit
guarantee; check free VRAM, reserve, and engine startup on the new machine. Increase fixed pass counts if the faster hardware
makes windows too short; report submitted counts, actual samples, and the
sampling cadence. Preserve the same test IDs, checkpoint/engine versions,
precision, prompts, and widths for matched comparisons where possible. Record
any differences as separate conditions, including what hardware is metered, and
compare per host before making a cross-hardware interpretation. A model that
only runs on the new host does not provide a matched laptop comparison.

## Offline verification

Run the owned analysis tests without creating bytecode caches:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/test_analysis.py -q
```

Other `tests/test_*.py` modules are offline unit/synthetic tests; review their
current scope before invoking the whole suite. None of these tests grants
permission for model downloads, engine startup, or measurement.

## Reporting boundary

Git includes the code, pinned profiles, documentation, and derived analysis in
`results/`. Raw trial artifacts (`artifacts/`), local dataset copies, model
weights, environments, and notebook caches stay on the benchmark host. A fresh
clone can read the derived results, but needs the corresponding raw artifacts
before `benchmark.py verify` can reproduce the full-run audit.

Do not put pilot numbers or fabricated values into this README or `DESIGN.md`.
The coordinator owns the separate results report. It must distinguish complete
systems, phases, widths, repetitions, hardware/software conditions, measurement
bases, missing components, skips, and failed attempts; preserve run-level
variation; and include the limitations in `DESIGN.md`.
