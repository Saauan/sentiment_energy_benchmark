# Sentiment energy benchmark: agent guide

This repository implements a bespoke TweetEval sentiment benchmark. Do not add Inspect. Read `DESIGN.md` for the authoritative experimental protocol, `README.md` for setup and operation, and `RESULTS.md` for historical findings. A result describes a **complete system** (model, engine, interface, hardware, and workload), not a model architecture in isolation. Keep documentation and implementation consistent when changing behavior.

## Repository map

- `benchmark.py`: CLI and experiment orchestration (`plan`, `run`, `status`, `verify`).
- `configuration.py`, `experiment.json`, `profiles/`: configuration validation, active full phase, and historical profiles.
- `dataset.py`, `workload.py`: panel integrity/selection and workload construction.
- `adapters/`: classical, local vLLM, and local QEV system interfaces.
- `energy.py`, `meter_worker.py`, `hardware.py`: measurement, isolated metering, and hardware metadata.
- `storage.py`: trial/attempt persistence; `analysis.py` and `analysis_notebook.py`: run-level analysis.
- `setup/`, `qev_setup/`: pinned inputs and separate engine setup instructions; `schema/`: local BoAmps schemas.
- `tests/`: offline unit and protocol tests; `artifacts/` and `results/`: experiment evidence, not disposable scratch space.

## Experimental invariants

- The active `engine-session-v1` protocol uses one fresh engine per model and repetition, serving widths in seeded order; widths remain distinct conditions. Historical profiles and artifact roots are read-only verification inputs, not configurations to rerun or pool with the active protocol.
- Primary energy is gross CodeCarbon CPU + GPU + RAM joules per actual submitted tweet across all passes. First-pass predictions determine primary quality; the measured run, not a tweet or pass, is the energy replicate. Keep idle-adjusted energy separate from gross energy, and training/calibration separate from inference. CodeCarbon estimates are not wall-meter measurements.
- Width means different things across systems. QEV branch batching within a decision is not independent-tweet batching; packed QEV is a distinct shared-context condition. Never describe one as the other.
- Preserve the pinned models, engines, precision, prompts, dataset splits, and workload conditions. Do not silently quantize, offload, substitute models, change precision, or truncate inputs to fit. Preflight available VRAM before downloading or loading weights. `plan` does not establish runtime fit.
- Completed trials and native/raw reports are immutable. Use a fresh artifact root for a new measurement. Resuming an existing root uses its stored plan; retries belong in fresh `attempt-NNNN` directories, not overwritten evidence. Do not change a condition under the guise of a retry.

## Safe operations

- One owner at a time for GPU work, model downloads, installation/compilation, and **all** tracked windows, including CPU-only and idle calibration. Avoid background activity while measuring; do not launch or reap unrelated processes. Inspect the host and coordinate before any such operation.
- Follow the readiness rule in `README.md` and `DESIGN.md`: on a new host or after a relevant runtime/meter change, check only the affected setup and shortest planned window before measuring. Idle baselines remain part of measurement.
- Do not use sudo or commit unless separately authorized. Do not write to the Obsidian vault or Daily Notes. Obtain download permission before fetching inputs; do not submit artifacts upstream without separately reviewing hardware metadata, local identifiers, and dataset terms.
- The supported client is Python 3.12.13 with `requirements.lock`; GPU engines have separate pinned environments documented in `setup/gpu/README.md` and `qev_setup/README.md`. Do not install or download as a side effect of routine inspection or tests.

## Development and verification

- Prefer focused offline tests, e.g. `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/test_analysis.py -q`; select the tests relevant to the change. Tests and analysis do not authorize inference or downloads, and should not run concurrently with a measured window.
- `benchmark.py status` inspects a run without loading models; `verify` audits existing evidence without new measurements. See `README.md` for required arguments and examples. Treat `run` as a real measurement operation, not a smoke test.
- When modifying protocol-sensitive code, test the relevant configuration, workload, adapter, measurement, storage, and analysis behavior as applicable. Preserve provenance, missing values, and the separation of workload kinds and conditions; report any untested runtime or hardware-dependent behavior explicitly.
