"""Observe the actual host and refuse infeasible full-precision GPU runs before load."""
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from adapters.base import Unavailable

MIB = 1024 ** 2


def command(*args):
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=5, check=True)
        return proc.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def sysread(path):
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def snapshot():
    fields = ("index", "uuid", "name", "driver_version", "memory.total", "memory.free", "pstate",
              "temperature.gpu", "power.draw", "power.limit", "clocks.current.graphics", "clocks.current.memory", "utilization.gpu")
    text = command("nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits")
    gpus = []
    if text:
        for line in text.splitlines():
            gpus.append(dict(zip(fields, (v.strip() for v in line.split(",")), strict=True)))
    cpu = next((x.partition(":")[2].strip() for x in (sysread("/proc/cpuinfo") or "").splitlines()
                if x.startswith("model name")), platform.processor())
    mem = (sysread("/proc/meminfo") or "").splitlines()
    return {
        "platform": platform.platform(), "cpu": cpu, "cpu_logical_count": os.cpu_count(),
        "cpu_affinity_count": len(os.sched_getaffinity(0)), "gpu": gpus, "python": platform.python_version(),
        "gpu_compute_apps": command("nvidia-smi", "--query-compute-apps=pid,name,used_gpu_memory", "--format=csv,noheader,nounits"),
        "host_id": platform.node(), "memory": {x.partition(":")[0]: x.partition(":")[2].strip() for x in mem[:3]},
        "loadavg": list(os.getloadavg()), "ac_online": sysread("/sys/class/power_supply/AC/online"),
        "cpu_governor": sysread("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "cpu_energy_perf_preference": sysread("/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference"),
        "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES")},
        "package_versions": {key: version(key) for key in ("codecarbon", "jsonschema")},
        "interpreter": sys.executable,
    }


def preflight(spec, host):
    """Return evidence and an optional skip reason; never downloads or creates CUDA context."""
    if spec.get("device", "cpu") != "cuda":
        return {"device": "cpu", "gpu_unused_but_metered": bool(host["gpu"])}, None
    config = spec["engine_safety"]
    gpus = host["gpu"]
    weights = config["weight_shards_bytes"]
    need_weights = sum(weights.values())
    reserve = int(config["reserve_mib"]) * MIB
    device = int(config.get("gpu_index", 0))
    gpu = next((device_row for device_row in gpus if int(device_row["index"]) == device), None)
    evidence = {"device": "cuda", "quantization": None, "precision": "bfloat16", "weight_shards_bytes": weights,
                "weights_bytes": need_weights, "reserve_bytes": reserve, "required_bytes": need_weights + reserve,
                "gpu_index": device, "gpu_observed": gpu, "weight_revision": spec["revision"]}
    if not gpu:
        return evidence, f"GPU index {device} unavailable to nvidia-smi"
    try:
        total, free = (int(gpu[k]) * MIB for k in ("memory.total", "memory.free"))
    except ValueError:
        return evidence, "GPU total/free VRAM not measurable"
    if need_weights > total:
        return evidence, f"weight bytes alone exceed GPU total VRAM ({need_weights/MIB:.0f} vs {total/MIB:.0f} MiB); no download/load"
    if need_weights + reserve > free:
        return evidence, f"insufficient currently free VRAM for weights + configured reserve ({(need_weights+reserve)/MIB:.0f} vs {free/MIB:.0f} MiB)"
    if len(gpus) != 1 or device != 0 or os.environ.get("CUDA_VISIBLE_DEVICES") not in {None, "0", gpu["uuid"]}:
        return evidence, "only the single visible GPU index 0 is supported by this protocol"
    if host["gpu_compute_apps"]:
        return evidence, "other active GPU compute app; refuse to load"
    snapshot_path = Path(spec.get("local_snapshot") or "/nonexistent")
    if not (snapshot_path / "config.json").is_file() or any(
            not (path := snapshot_path / name).is_file() or path.stat().st_size != size for name, size in weights.items()):
        return evidence, f"unavailable or size-mismatched pinned local snapshot: {snapshot_path}"
    config_file = snapshot_path / "config.json"
    try:
        checkpoint = json.loads(config_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return evidence, f"cannot read pinned checkpoint config: {exc}"
    if not isinstance(checkpoint, dict):
        return evidence, "checkpoint config is not a JSON object"
    checkpoint_configs = [checkpoint] + [v for k, v in checkpoint.items() if k.endswith("_config") and isinstance(v, dict)]
    if any(cfg.get("quantization_config") or cfg.get("compression_config") for cfg in checkpoint_configs):
        return evidence, "checkpoint declares quantization or compression; primary BF16 condition forbids it"
    evidence["local_snapshot"] = str(snapshot_path)
    return evidence, None


def looks_like_oom(exc, attempt):
    signature = (str(exc) + " " + type(exc).__name__).lower()
    for path in Path(attempt).glob("*.log"):
        # The owned, bounded stderr logs can reveal OOM below the serving supervisor.
        with path.open("rb") as log:
            log.seek(max(0, path.stat().st_size - 65536))
            signature += log.read().decode(errors="replace").lower()
    return any(marker in signature for marker in ("cuda out of memory", "cudaerrormemoryallocation", "outofmemoryerror",
                                                   "not enough free memory", "no available memory for the cache blocks",
                                                   "free memory on device", "cublas_status_alloc_failed"))
