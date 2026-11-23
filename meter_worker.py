"""Out-of-process CodeCarbon meter worker for one measurement window.

Why a separate process
----------------------
CodeCarbon 3.3.3 in ``machine`` tracking mode samples the machine from a daemon
thread that shares its interpreter with the caller. In this benchmark the caller
runs the measured workload (CPU inference loops, adapter I/O, tokenisation) in
that same interpreter, so the sampler competes for the same GIL and the same
native scheduler. Observed failure: a 14.18 s inference window configured with
``measure_power_secs=0.1`` produced only two power samples, which integrated the
constant RAM power model over ~0.5 s instead of over the window.

The worker below runs the tracker (and only the tracker) in its own interpreter,
prepares it before the window, and opens/closes the window on commands from the
parent. The parent keeps the workload and owns the wall clock. Nothing is
injected into the workload: no sleep, no timestamp, no IPC call inside it.

Measurement semantics are preserved because all three CodeCarbon backends used
here read machine- or device-scoped counters rather than per-process quantities:

* CPU ``cpu_load`` -> ``psutil.cpu_times()`` machine deltas,
* RAM estimate -> ``psutil.virtual_memory().total`` constant-power model,
* GPU -> NVML device energy counter.

Protocol (``meter-worker/1``)
-----------------------------
Parent -> worker: newline-delimited JSON commands on stdin (``begin``, ``end``,
``quit``). Worker -> parent: newline-delimited JSON events on a dedicated pipe fd
passed with ``--events-fd``, so CodeCarbon console logging cannot corrupt the
control channel. ``ready`` carries the monitoring metadata that must exist
before the measurement window; ``started``/``stopped`` carry sample counts and
worker-side window facts.

Lifecycle logging goes to ``<out>/worker.log`` and to the pinned CodeCarbon log
``<out>/codecarbon.log`` (same file the in-process meter produced). Nothing extra
is logged inside the window beyond CodeCarbon's own messages.

Orphan avoidance: the worker owns no authority. It exits on ``quit``, on EOF of
its command pipe, and when its parent process disappears (ppid watchdog). If it
exits while a window is open, it closes the window and flushes its reports.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import select
import signal
import sys
from time import perf_counter, time
import traceback

PROTOCOL_VERSION = 1
MAX_EVENT_BYTES = 1 << 20
MAX_RECORDED_TICKS = 4096
POLL_SECONDS = 0.5


def default_command():
    """Command prefix the parent uses to start this worker."""
    return [sys.executable, str(Path(__file__).resolve())]


class Channel:
    """Newline-delimited JSON writer over a pipe fd (worker -> parent)."""

    def __init__(self, fd):
        self.fd = fd

    def send(self, message):
        payload = json.dumps(message, ensure_ascii=False, allow_nan=False, sort_keys=True).encode("utf-8") + b"\n"
        if len(payload) > MAX_EVENT_BYTES:
            raise ValueError(f"meter worker event too large: {len(payload)} bytes")
        view = memoryview(payload)
        while view:
            try:
                view = view[os.write(self.fd, view):]
            except InterruptedError:
                continue


class ChannelReader:
    """Reader for worker events with a deadline (parent side)."""

    def __init__(self, fd):
        self.fd = fd
        self._buffer = b""
        self.eof = False
        self.malformed = []

    def read(self, timeout):
        """Return the next event dict, or None when the deadline expires.

        Raises EOFError when the worker's channel is closed (worker gone).
        """
        from time import monotonic
        deadline = monotonic() + timeout
        while True:
            line = self._pop_line()
            if line is not None:
                try:
                    event = json.loads(line)
                except ValueError:
                    self.malformed.append(line[:200].decode("utf-8", "replace"))
                    continue
                if not isinstance(event, dict):
                    self.malformed.append(line[:200].decode("utf-8", "replace"))
                    continue
                return event
            if self.eof:
                raise EOFError("meter worker closed its event channel")
            remaining = deadline - monotonic()
            if remaining <= 0:
                return None
            ready, _, _ = select.select([self.fd], [], [], remaining)
            if not ready:
                return None
            chunk = os.read(self.fd, 65536)
            if not chunk:
                self.eof = True
                continue
            self._buffer += chunk

    def _pop_line(self):
        index = self._buffer.find(b"\n")
        if index < 0:
            return None
        line, self._buffer = self._buffer[:index], self._buffer[index + 1:]
        return line

    def close(self):
        try:
            os.close(self.fd)
        except OSError:
            pass


def record_ticks(tracker):
    """Wrap the pinned scheduler callback to record its monotonic cadence.

    Optional provenance only: when CodeCarbon ran its scheduled measurement
    callback and how long that callback took. It never touches the workload.
    Returns the mutable record, or None if the internal scheduler is not the
    expected shape (then no cadence evidence is claimed).
    """
    scheduler = getattr(tracker, "_scheduler", None)
    inner = getattr(scheduler, "function", None)
    if scheduler is None or not callable(inner):
        return None
    record = {"calls": 0, "gaps_s": [], "max_gap_s": None, "max_call_s": None, "truncated": False}

    def tick(*args, **kwargs):
        begin = perf_counter()
        try:
            return inner(*args, **kwargs)
        finally:
            now = perf_counter()
            record["calls"] += 1
            call_s = round(now - begin, 6)
            record["max_call_s"] = call_s if record["max_call_s"] is None else max(record["max_call_s"], call_s)
            previous = record.get("_last")
            if previous is not None:
                gap = round(now - previous, 6)
                if len(record["gaps_s"]) < MAX_RECORDED_TICKS:
                    record["gaps_s"].append(gap)
                    record["max_gap_s"] = gap if record["max_gap_s"] is None else max(record["max_gap_s"], gap)
                else:
                    record["truncated"] = True
            record["_last"] = now

    scheduler.function = tick
    return record


def describe_sources(tracker):
    """Monitoring records for the tracker's hardware backends (unchanged shape)."""
    records = []
    for source in tracker._hardware:
        name = type(source).__name__
        record = {"class": name, "description": source.description(), "mode": getattr(source, "_mode", None)}
        if name == "CPU":
            record["tdp_watts"] = getattr(source, "_tdp", None)
        elif name == "GPU":
            record["monitored_gpu_indices"] = list(source._get_gpu_ids())
            record["devices"] = [{"index": d.gpu_index, "backend": type(d).__name__} for d in source.devices.devices]
        elif name == "RAM":
            record["mode"] = "CodeCarbon RAM estimate" if source._force_ram_power is None else "forced RAM watts"
        records.append(record)
    return records


def prepare(cfg, task, tracker, schema_pin, codecarbon_pin):
    """Bring the tracker to the state where the window can start.

    Every private interface used here was already used by the in-process meter.
    All of it runs strictly before the window, so no lazy hardware/geo/emissions
    initialisation can land inside it.
    """
    tracker._ensure_hardware_ready()
    tracker._ensure_geo_metadata()
    tracker._ensure_emissions_engine()
    tracker._ensure_cloud_conf()
    return {
        "tracking_mode": tracker._tracking_mode, "codecarbon_version": codecarbon_pin,
        "schema_pin": schema_pin, "pue": tracker._pue, "measure_power_secs": tracker._measure_power_secs,
        "gpu_ids_configured": tracker._gpu_ids, "detected_hardware": tracker.get_detected_hardware(),
        "country_iso_code": cfg["country_iso_code"], "sources": describe_sources(tracker),
        "expected_task": task, "boundary": "estimated CodeCarbon component energy, not a wall meter",
    }


def build_tracker(cfg, task, out, codecarbon_pin, *, boamps_output=None):
    """Set up the pinned tracker, optionally supplying our BoAmps handler."""
    if codecarbon_pin and version("codecarbon") != codecarbon_pin:
        raise RuntimeError(f"CodeCarbon {codecarbon_pin} required")
    import logging

    from codecarbon import OfflineEmissionsTracker
    from codecarbon.external.logger import logger
    from codecarbon.output_methods.base_output import OutputMethod

    handler = logging.FileHandler(Path(out) / "codecarbon.log", encoding="utf-8")
    logger.addHandler(handler)
    try:
        outputs = [OutputMethod.CSV, OutputMethod.BOAMPS]
        kwargs = {}
        if boamps_output is not None:
            outputs.remove(OutputMethod.BOAMPS)
            kwargs["output_handlers"] = [boamps_output]
        return OfflineEmissionsTracker(
            country_iso_code=cfg["country_iso_code"], output_dir=str(out), output_file="emissions.csv",
            output_methods=outputs, tracking_mode="machine",
            gpu_ids=cfg.get("gpu_ids"), measure_power_secs=cfg["sample_secs"],
            metadata={"boamps": {"task": task}}, pue=1.0, allow_multiple_runs=True, log_level="info",
            on_csv_write="append", experiment_name="sentiment-energy-benchmark", **kwargs)
    except BaseException:
        logger.removeHandler(handler)
        handler.close()
        raise


class Worker:
    """Owns one CodeCarbon tracker for exactly one measurement window."""

    def __init__(self, out, cfg, task, channel, tracker_factory=None, schema_pin=None, codecarbon_pin=None,
                 *, export_tracking_modes=False):
        self.out = Path(out)
        self.cfg = cfg
        self.task = task
        self.channel = channel
        self.tracker_factory = tracker_factory
        self.schema_pin = schema_pin
        self.codecarbon_pin = codecarbon_pin
        self.export_tracking_modes = export_tracking_modes
        self.tracker = None
        self.ticks = None
        self.metadata = None
        self.parent_pid = os.getppid()
        self.signalled = None

    def log(self, message):
        try:
            with (self.out / "worker.log").open("a", encoding="utf-8") as stream:
                stream.write(f"{time():.3f} pid={os.getpid()} {message}\n")
        except OSError:
            pass

    def build(self):
        output = None
        if self.tracker_factory is None:
            if self.export_tracking_modes:
                from boamps_export import TrackingBoAmpsOutput

                output = TrackingBoAmpsOutput.from_dict({"task": self.task}, output_dir=str(self.out))
            self.tracker = build_tracker(self.cfg, self.task, self.out, self.codecarbon_pin, boamps_output=output)
        else:
            self.tracker = self.tracker_factory(self.out, self.task, self.cfg)
        self.metadata = prepare(self.cfg, self.task, self.tracker, self.schema_pin, self.codecarbon_pin)
        if output is not None:
            output.sources = self.metadata["sources"]
        self.ticks = record_ticks(self.tracker)
        self.metadata["meter"] = {
            "protocol": f"meter-worker/{PROTOCOL_VERSION}", "mode": "separate_process",
            "worker_pid": os.getpid(), "parent_pid": self.parent_pid,
            "python": sys.version.split()[0], "executable": sys.executable,
            "codecarbon_version": self.codecarbon_pin, "clock": "perf_counter/CLOCK_MONOTONIC",
            "backends": [record["class"] for record in self.metadata["sources"]],
            "scopes": {"CPU": "machine cpu_times load model",
                       "RAM": "machine total-memory power model (constant watts)",
                       "GPU": "NVML device energy counter"},
            "window": "parent perf_counter around work; worker sample counts around begin/end",
            "overhead": "the worker's own sampling load is inside the machine-scope estimates it reports",
            "per_tick_monotonic": self.ticks is not None,
        }

    def run(self):
        self.log(f"worker start parent={self.parent_pid}")
        self._install_signals()
        try:
            self.build()
        except BaseException as exc:  # noqa: BLE001 - reported to the parent, never raised to it
            self._error("prepare", exc)
            return 1
        try:
            self.channel.send({"v": PROTOCOL_VERSION, "event": "ready", "metadata": self.metadata})
        except BaseException as exc:  # noqa: BLE001
            self.log(f"cannot announce readiness: {exc!r}")
            return 1
        self.log("worker ready")
        window = None
        code = 0
        try:
            for line in self._commands():
                if line is None:
                    if self.signalled is not None or not self._parent_alive():
                        self.log(f"worker stopping: signal={self.signalled} parent_alive={self._parent_alive()}")
                        break
                    continue
                command = line.get("cmd") if isinstance(line, dict) else None
                if command == "begin" and window is None:
                    window = self._begin()
                elif command == "end" and window is not None:
                    self._end(window)
                    window = None
                elif command == "quit":
                    break
                else:
                    self._error("command", ValueError(f"unexpected command in state {window is not None}: {command!r}"))
                    code = 1
                    break
        except BaseException as exc:  # noqa: BLE001 - broken channel, worker bug
            self.log(f"worker command loop failed: {exc!r}")
            code = 1
        finally:
            self._restore_signals()
            if window is not None:
                # Parent vanished or the channel broke mid-window: close the
                # window so the attempt keeps a complete, honest sample record.
                try:
                    self._end(window)
                except BaseException as exc:  # noqa: BLE001
                    self._error("end", exc)
                    code = 1
        self.log(f"worker exit code={code}")
        return code

    # -- commands ----------------------------------------------------------
    def _commands(self):
        """Yield command dicts; yields None every poll tick for liveness checks."""
        buffer = b""
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], POLL_SECONDS)
            if not ready:
                yield None
                continue
            chunk = os.read(sys.stdin.fileno(), 65536)
            if not chunk:
                if buffer.strip():
                    yield self._decode(buffer.strip())
                yield None
                return
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if line.strip():
                    yield self._decode(line)

    def _decode(self, raw):
        try:
            return json.loads(raw)
        except ValueError:
            self._error("command", ValueError(f"malformed command: {raw[:120]!r}"))
            return {"cmd": "__malformed__"}

    def _begin(self):
        started = perf_counter()
        try:
            self.tracker.start()
            if self.tracker._start_time is None:
                raise RuntimeError("tracker.start() did not open a window")
            samples = self.tracker._power_measurement_count
            self.channel.send({"v": PROTOCOL_VERSION, "event": "started", "samples": samples,
                               "worker_start": perf_counter(), "start_s": perf_counter() - started})
        except BaseException as exc:  # noqa: BLE001
            self._error("begin", exc)
            raise
        self.log(f"window begin samples={samples} start_s={perf_counter() - started:.3f}")
        return {"start": self.tracker._start_time}

    def _end(self, window):
        # Read the counter at the instant the parent's window ended, before
        # stop() adds its final "stale" measurement.
        samples_before_stop = self.tracker._power_measurement_count
        stopped = perf_counter()
        self.tracker.stop()
        after = perf_counter()
        samples = self.tracker._power_measurement_count
        self.log(f"window end samples_before_stop={samples_before_stop} samples={samples} stop_s={after - stopped:.3f}")
        self.channel.send({
            "v": PROTOCOL_VERSION, "event": "stopped", "samples_before_stop": samples_before_stop,
            "samples": samples, "stop_s": after - stopped, "worker_stop": stopped,
            "worker_window_s": after - window["start"], "ticks": summarize_ticks(self.ticks),
        })

    def _error(self, stage, exc):
        self.log(f"worker error stage={stage}: {exc!r}\n{traceback.format_exc()}")
        try:
            self.channel.send({"v": PROTOCOL_VERSION, "event": "error", "stage": stage,
                               "error": f"{type(exc).__name__}: {exc}"})
        except BaseException:  # noqa: BLE001 - the channel itself is gone
            pass

    # -- process discipline ------------------------------------------------
    def _install_signals(self):
        def handler(signum, _frame):
            # Terminal SIGINT is the parent's business; the worker keeps the
            # window consistent and lets the parent close it.
            self.signalled = signum

        for name in ("SIGINT", "SIGTERM", "SIGHUP"):
            number = getattr(signal, name, None)
            if number is not None:
                try:
                    signal.signal(number, handler)
                except (ValueError, OSError):
                    pass

    def _restore_signals(self):
        for name in ("SIGINT", "SIGTERM", "SIGHUP"):
            number = getattr(signal, name, None)
            if number is not None:
                try:
                    signal.signal(number, signal.SIG_DFL)
                except (ValueError, OSError):
                    pass

    def _parent_alive(self):
        return os.getppid() == self.parent_pid


def summarize_ticks(record):
    if not record:
        return None
    gaps = list(record["gaps_s"])
    summary = {"calls": record["calls"], "max_gap_s": record["max_gap_s"], "max_call_s": record["max_call_s"],
               "gaps_s": gaps, "truncated": record["truncated"]}
    if gaps:
        summary["mean_gap_s"] = round(sum(gaps) / len(gaps), 6)
    return summary


def parse(argv):
    parser = argparse.ArgumentParser(prog="meter_worker", description=__doc__.splitlines()[0])
    parser.add_argument("--events-fd", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--codecarbon-pin")
    parser.add_argument("--schema-pin")
    parser.add_argument("--export-tracking-modes", action="store_true")
    return parser.parse_args(argv)


def main(argv=None, tracker_factory=None):
    args = parse(sys.argv[1:] if argv is None else argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    worker = Worker(out, json.loads(args.config), json.loads(args.task), Channel(args.events_fd),
                    tracker_factory=tracker_factory, schema_pin=args.schema_pin,
                    codecarbon_pin=args.codecarbon_pin, export_tracking_modes=args.export_tracking_modes)
    return worker.run()


if __name__ == "__main__":
    sys.exit(main())
