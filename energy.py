"""Pinned CodeCarbon instrumentation and offline native-export gate.

The tracker runs in a separate process (``meter_worker.py``) so its periodic
sampler never shares a GIL or native scheduler with the measured workload; see
the module docstring there for the protocol, the evidence and the scope
argument. This module keeps the public surface: ``Meter(out, task, cfg)`` and
``capture(work) -> (result, wall_s, native)``.
"""
from __future__ import annotations

import csv
from datetime import datetime
from functools import lru_cache
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import subprocess
from time import perf_counter
from uuid import UUID

import meter_worker
from storage import atomic_json, content_hash, identity, load

SCHEMA_PIN = "af1680c56ce2cefd5777fcac1c57591b685d6be5"
ROOT = Path(__file__).resolve().parent
SCHEMA_DIR = ROOT / "schema" / SCHEMA_PIN
CODECARBON_PIN = "3.3.3"
J_PER_KWH = 3_600_000
# Bounded protocol deadlines. The measurement window itself is never waited on
# with a timeout: work() is the only unbounded part of capture().
DEFAULT_TIMEOUTS = {"prepare": 300.0, "begin": 120.0, "end": 120.0, "exit": 15.0}
TERMINATE_GRACE_S = 2.0
KILL_GRACE_S = 5.0


class MeterError(RuntimeError):
    """The meter worker failed: the window's numbers cannot be trusted."""


def task_context(kind, model, dataset, count, caption, data_quantity=None):
    stages = {"inference": "inference", "training": "training", "device_idle": "calibration", "loaded_idle": "calibration"}
    calibration = kind.endswith("idle")
    data = {"dataUsage": "input", "dataType": "other" if calibration else "text"}
    if calibration:
        data.update(source="other", dataQuantity=0)
    else:
        data["source"] = dataset["source_type"]
        if data_quantity is not None:
            data["dataQuantity"] = data_quantity
        if dataset.get("source_uri"):
            data["sourceUri"] = dataset["source_uri"]
    return {"taskStage": stages[kind], "taskFamily": "device idle calibration" if calibration else "text classification",
            "nbRequest": count, "algorithms": [{"algorithmName": model}], "dataset": [data],
            "taskDescription": caption}


@lru_cache(maxsize=1)
def validator():
    from jsonschema import Draft4Validator, FormatChecker
    from referencing import Registry, Resource
    paths = [SCHEMA_DIR / (name + "_schema.json") for name in ("report", "algorithm", "dataset", "measure", "hardware")]
    pins = {line.split()[1]: line.split()[0] for line in (ROOT / "setup/inputs.sha256").read_text().splitlines()}
    for path in paths:
        if content_hash(path) != pins[path.relative_to(ROOT).as_posix()]:
            raise ValueError(f"BoAmps schema hash mismatch: {path}")
    schemas = [load(path) for path in paths]
    # The pinned upstream algorithm schema declares `required: []`, disallowed by
    # the Draft-04 metaschema but semantically a no-op. Omit it only in memory.
    if schemas[1].get("required") == []:
        del schemas[1]["required"]
    for schema in schemas:
        Draft4Validator.check_schema(schema)
    registry = Registry().with_resources((schema["id"], Resource.from_contents(schema)) for schema in schemas)
    # No network retrieval, including references whose pinned schema $id uses /main/.
    return Draft4Validator(schemas[0], registry=registry, format_checker=FormatChecker())


def numeric(value, field, *, positive=False):
    if isinstance(value, bool):
        raise ValueError(f"unexpected boolean: {field}")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (positive and number == 0):
        raise ValueError(f"invalid numeric field: {field}={value}")
    return number


def validate_run(out, expected):
    out = Path(out)
    reports = list(out.glob("boamps_report_*.json"))
    if len(reports) != 1:
        raise ValueError(f"expected exactly one native report, got {len(reports)}")
    report = load(reports[0]); validator().validate(report)
    task = report["task"]
    for key in ("taskStage", "taskFamily", "taskDescription", "algorithms", "dataset", "nbRequest"):
        wanted = expected[key]
        # A missing zero is schema-legal for non-inference phases. Never assume for inference.
        observed = task.get(key, 0 if key == "nbRequest" and wanted == 0 else None)
        if observed != wanted or (key == "nbRequest" and type(observed) not in (int, float)):
            raise ValueError(f"BoAmps task {key} mismatch: {observed!r} != {wanted!r}")
    header = report["header"]
    if header.get("formatVersion") != "0.1":
        raise ValueError("unexpected native format version")
    datetime.fromisoformat(header["reportDatetime"])
    with (out / "emissions.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)) or len(rows) != 1 or None in rows[0]:
            raise ValueError("expected one unambiguous native CSV record")
    row = rows[0]
    rid = row["run_id"]
    uuid = UUID(rid)
    if rid != str(uuid) or uuid.version != 4 or rid != header["reportId"] or reports[0].name != f"boamps_report_{rid}.json":
        raise ValueError("native report/CSV ID mismatch")
    if row["tracking_mode"] != "machine" or row["codecarbon_version"] != CODECARBON_PIN:
        raise ValueError("unexpected tracking mode/version")
    measures = report["measures"]
    if len(measures) != 1 or measures[0]["measurementMethod"] != "codecarbon" or measures[0].get("version") != CODECARBON_PIN:
        raise ValueError("unexpected native measure/method/version")
    measure = measures[0]
    datetime.fromisoformat(measure["measurementDateTime"])
    duration = numeric(row["duration"], "duration", positive=True)
    total = numeric(row["energy_consumed"], "energy", positive=True)
    components = {key: numeric(row[key + "_energy"], key) for key in ("cpu", "gpu", "ram")}
    checks = [(measure["powerConsumption"], total, "powerConsumption"), (measure["measurementDuration"], duration, "duration"),
              (sum(components.values()), total, "sum of components")]
    if any(not math.isclose(numeric(a, key), b, abs_tol=1e-10, rel_tol=1e-8) for a, b, key in checks):
        raise ValueError("native duration or energy disagree across artifacts")
    metadata = load(out / "monitoring.json")
    if (metadata["tracking_mode"], metadata["codecarbon_version"], metadata["schema_pin"]) != ("machine", CODECARBON_PIN, SCHEMA_PIN):
        raise ValueError("invalid monitoring metadata")
    if metadata["expected_task"] != expected or metadata["pue"] != 1.0:
        raise ValueError("unexpected monitoring task/PUE")
    detected = {source["class"].lower() for source in metadata["sources"]}
    if any(value != 0 for key, value in components.items() if key not in detected):
        raise ValueError("CSV nonzero energy for a component missing from monitoring sources")
    return {"run_id": rid, "energy_kwh": total, "duration_s": duration, "schema_pin": SCHEMA_PIN,
            "source_units": {"energy": "kWh", "duration": "s"}, "count": expected["nbRequest"],
            **{key + "_energy_kwh": val if key in detected else None for key, val in components.items()},
            "source_signature": identity({k: metadata[k] for k in ("sources", "measure_power_secs", "pue", "tracking_mode")}),
            "sources": metadata["sources"], "native_report": reports[0].name,
            "scheduled_power_samples": metadata.get("timing", {}).get("scheduled_power_samples")}


class Meter:
    """Prepare the pinned tracker in a worker process; keep the timed window minimal.

    Construction spawns the worker and consumes its ``ready`` event, so hardware,
    geo, emissions and cloud initialisation all happen before the bright line.
    ``capture(work)`` opens the window, runs ``work`` in *this* process, closes the
    window and reaps the worker. Nothing is injected into ``work``.
    """
    def __init__(self, out, task, cfg, *, worker=None, timeouts=None):
        if version("codecarbon") != CODECARBON_PIN:
            raise RuntimeError(f"CodeCarbon {CODECARBON_PIN} required")
        self.out = Path(out)
        self.out.mkdir()
        self.task = task
        self.cfg = cfg
        self.timeouts = {**DEFAULT_TIMEOUTS, **(timeouts or {})}
        self.metadata = None
        self._process = None
        self._events = None
        self._closed = False
        self._used = False
        self._spawn(list(worker) if worker is not None else meter_worker.default_command())
        try:
            ready = self._await("ready", "prepare")
            self.metadata = ready["metadata"]
            self.metadata["meter"].update({
                "spawn_s": round(perf_counter() - self._spawned, 6),
                "worker_command": Path(self._command[0]).name,
                "window_owner": "parent",
            })
            atomic_json(self.out / "monitoring.json", self.metadata)
        except BaseException:
            self.close(force=True)
            raise

    # -- process -----------------------------------------------------------
    def _spawn(self, command):
        read_fd, write_fd = os.pipe()
        self._command = command
        log = (self.out / "worker.log").open("ab", buffering=0)
        try:
            self._process = subprocess.Popen(
                [*command, "--events-fd", str(write_fd), "--out", str(self.out.resolve()),
                 "--config", json.dumps(self.cfg), "--task", json.dumps(self.task),
                 "--codecarbon-pin", CODECARBON_PIN, "--schema-pin", SCHEMA_PIN],
                stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                pass_fds=(write_fd,), close_fds=True, start_new_session=True)
        except BaseException:
            os.close(read_fd)
            raise
        finally:
            os.close(write_fd)
            log.close()
        self._events = meter_worker.ChannelReader(read_fd)
        self._spawned = perf_counter()

    def _send(self, command):
        process = self._process
        if process is None or process.stdin is None or process.poll() is not None:
            raise MeterError(f"meter worker is not running before {command!r}")
        try:
            process.stdin.write((json.dumps({"cmd": command}) + "\n").encode("utf-8"))
            process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            raise MeterError(f"meter worker pipe closed before {command!r}") from None

    def _await(self, wanted, stage):
        events = self._events
        if events is None:
            raise MeterError(f"meter worker channel closed before {wanted!r}")
        try:
            event = events.read(self.timeouts[stage])
        except EOFError:
            code = self._process.poll() if self._process is not None else None
            raise MeterError(f"meter worker exited during {stage} (returncode={code})") from None
        if event is None:
            raise MeterError(f"meter worker timed out after {self.timeouts[stage]}s waiting for {wanted!r} during {stage}")
        kind = event.get("event")
        if kind == "error":
            code = self._process.poll() if self._process is not None else None
            raise MeterError(f"meter worker failed during {event.get('stage')}: {event.get('error')} (returncode={code})")
        if kind != wanted:
            raise MeterError(f"unexpected meter worker event {kind!r} while waiting for {wanted!r} during {stage}")
        return event

    def _note(self, message):
        try:
            with (self.out / "worker.log").open("a", encoding="utf-8") as stream:
                stream.write(f"parent {message}\n")
        except OSError:
            pass

    def close(self, force=False):
        """Stop and reap our own worker; idempotent, bounded, never touches another pid."""
        if self._closed:
            return
        self._closed = True
        process, events = self._process, self._events
        self._process = self._events = None
        try:
            if process is None:
                return
            if process.poll() is None and not force:
                try:
                    process.stdin.write(b'{"cmd": "quit"}\n')
                    process.stdin.flush()
                except (BrokenPipeError, OSError, ValueError):
                    pass
            try:
                process.stdin.close()
            except (OSError, ValueError):
                pass
            if force:
                self._terminate(process)
            else:
                self._reap(process, self.timeouts["exit"])
        finally:
            if events is not None:
                events.close()

    def _reap(self, process, timeout):
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            pass
        self._note("terminating meter worker after quit/exit deadline")
        self._terminate(process)

    def _terminate(self, process):
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=TERMINATE_GRACE_S)
            return
        except subprocess.TimeoutExpired:
            pass
        self._note("killing meter worker after terminate grace")
        process.kill()
        process.wait(timeout=KILL_GRACE_S)

    # -- window ------------------------------------------------------------
    def capture(self, work):
        """Run ``work`` between window open and close; return (result, wall_s, native)."""
        if self._used:
            raise MeterError("a Meter measures exactly one window")
        self._used = True
        start_call = perf_counter()
        try:
            initial_samples, handshake = self._begin()
        except BaseException:
            self.close(force=True)
            raise
        start = perf_counter()
        pending = None
        try:
            result = work()
        except BaseException as exc:
            pending = exc
            raise
        finally:
            end = perf_counter()
            try:
                self._finish(start, end, start_call, initial_samples, handshake)
            except BaseException as exc:
                self.close(force=True)
                if pending is None:
                    raise
                # Keep the workload failure primary; the meter failure is on record.
                self._note(f"meter finalisation failed after workload error {pending!r}: {exc!r}")
        return result, end - start, validate_run(self.out, self.task)

    def _begin(self):
        sent = perf_counter()
        self._send("begin")
        event = self._await("started", "begin")
        acked = perf_counter()
        return event["samples"], {"begin_ack_s": acked - sent, "worker_start": event.get("worker_start"),
                                  "worker_start_s": event.get("start_s")}

    def _finish(self, start, end, start_call, initial_samples, handshake):
        sent = perf_counter()
        self._send("end")
        event = self._await("stopped", "end")
        stopped = perf_counter()
        lead = None
        if isinstance(handshake.get("worker_start"), (int, float)):
            lead = start - handshake["worker_start"]
            if not -1.0 <= lead <= 60.0:
                # Different clock bases: report the raw values, claim no lead.
                lead = None
        self.metadata["timing"] = {
            "window_s": end - start, "tracker_start_s": start - start_call,
            "tracker_stop_s": stopped - end, "clock": "perf_counter",
            "scheduled_power_samples": event["samples_before_stop"] - initial_samples,
            "total_power_samples": event["samples"],
            "meter": {"protocol": self.metadata["meter"]["protocol"],
                      "worker_window_s": event.get("worker_window_s"), "stop_s": event.get("stop_s"),
                      "begin_ack_s": handshake["begin_ack_s"], "end_ack_s": sent - end,
                      "stop_ack_s": stopped - sent, "worker_start_s": handshake.get("worker_start_s"),
                      "meter_lead_s": lead, "ticks": event.get("ticks")},
        }
        atomic_json(self.out / "monitoring.json", self.metadata)
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close(force=True)

    def __del__(self):
        try:
            self.close(force=True)
        except BaseException:  # noqa: BLE001 - interpreter shutdown
            pass


def idle_adjusted(gross, baseline, count):
    if not baseline or gross["source_signature"] != baseline["source_signature"]:
        return None
    # Signed results: do not mask negative values from unstable baselines/noise.
    return J_PER_KWH * (gross["energy_kwh"] - baseline["energy_kwh"] * gross["duration_s"] / baseline["duration_s"]) / count
