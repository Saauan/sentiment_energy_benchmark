"""Offline tests: no GPU contexts or CodeCarbon hardware probes."""
from copy import deepcopy
import csv
from pathlib import Path
from uuid import uuid4

from jsonschema import ValidationError
import pytest

from dataset import LABELS, TWEETEVAL_REV, configure
from energy import SCHEMA_PIN, idle_adjusted, task_context, validate_run
from hardware import MIB, preflight
from storage import Run, atomic_json, identity, load, measurement_lock
from workload import audit, persist, run_workload


def fixture(out, kind="inference", count=4):
    uid = str(uuid4())
    dt = "2026-11-23T12:11:23+00:00"
    task = task_context(kind, "test fixture", {"source_type": "public"}, count, "test receipt", 2)
    report = {"header": {"formatVersion": "0.1", "reportId": uid, "reportDatetime": dt},
              "task": task, "measures": [{"measurementMethod": "codecarbon", "version": "3.3.3",
                                            "powerConsumption": 0.0003, "measurementDateTime": dt,
                                            "measurementDuration": 8.0}],
              "infrastructure": {"infraType": "onPremise", "components": [{"componentType": "cpu", "nbComponent": 1}]}}
    report_path = out / f"boamps_report_{uid}.json"
    atomic_json(report_path, report)
    row = {"duration": 8, "energy_consumed": .0003, "cpu_energy": .0001, "gpu_energy": 0,
           "ram_energy": .0002, "run_id": uid, "tracking_mode": "machine", "codecarbon_version": "3.3.3"}
    with (out / "emissions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
    atomic_json(out / "monitoring.json", {"expected_task": task, "tracking_mode": "machine", "sources": [{"class": "CPU"}, {"class": "RAM"}],
                                          "schema_pin": SCHEMA_PIN, "codecarbon_version": "3.3.3", "pue": 1.0, "measure_power_secs": 1})
    return task, report_path


@pytest.mark.parametrize("kind,count", [("inference", 4), ("device_idle", 0), ("training", 0)])
def test_native_units_and_missing_components(tmp_path, kind, count):
    task, _ = fixture(tmp_path, kind=kind, count=count)
    summary = validate_run(tmp_path, task)
    assert summary["energy_kwh"] == .0003
    assert summary["duration_s"] == 8
    assert summary["gpu_energy_kwh"] is None
    assert summary["count"] == count


@pytest.mark.parametrize("violation,expected", [("schema", ValidationError), ("count", ValueError), ("energy", ValueError),
                                                ("id", ValueError), ("duplicate", ValueError), ("dataset", ValueError)])
def test_report_gate_fails_closed(tmp_path, violation, expected):
    task, report_path = fixture(tmp_path)
    report = load(report_path)
    if violation == "schema":
        report["task"]["dataset"][0]["dataType"] = "an_invalid_schema_enum"
    elif violation == "count":
        report["task"]["nbRequest"] += 1
    elif violation == "energy":
        report["measures"][0]["powerConsumption"] *= 0.5
    elif violation == "id":
        report["header"]["reportId"] = str(uuid4())
    elif violation == "duplicate":
        atomic_json(tmp_path / "boamps_report_duplicate.json", report)
    elif violation == "dataset":
        report["task"]["dataset"][0]["source"] = "private"
    atomic_json(report_path, report)
    with pytest.raises(expected):
        validate_run(tmp_path, task)


def test_signed_calibration_and_backend_matching():
    gross = {"energy_kwh": .0001, "duration_s": 20, "source_signature": "same"}
    baseline = {"energy_kwh": .00009, "duration_s": 10, "source_signature": "same"}
    assert idle_adjusted(gross, baseline, 8) == pytest.approx(-36)
    assert idle_adjusted(gross, {**baseline, "source_signature": "other"}, 8) is None


def test_interrupted_attempt_is_identifiable_and_never_overwritten(tmp_path):
    trial = {"run_id": "fixture", "phase": "test"}
    run = Run(tmp_path, trial)
    first = run.start()
    atomic_json(first / "partial.json", {"this": 12})
    run = Run(tmp_path, trial)
    run.recover()
    assert run.state["status"] == "incomplete"
    second = run.start()
    assert second != first
    atomic_json(second / "result.json", {"finished": True})
    run.finish("complete")
    assert [entry["status"] for entry in run.state["history"]] == ["incomplete", "complete"]
    assert load(first / "partial.json") == {"this": 12}
    with pytest.raises(ValueError):
        Run(tmp_path, trial).start()


def test_lock_rejects_second_writer(tmp_path):
    with measurement_lock(tmp_path / "lock"), pytest.raises(RuntimeError):
        with measurement_lock(tmp_path / "lock"):
            raise AssertionError("lock is not exclusive")


def test_pinned_panel_is_deterministic_and_split_correct():
    spec = {"adapter": "tweeteval", "path": "data/tweeteval", "revision": TWEETEVAL_REV}
    if not Path("data/tweeteval/mapping.txt").is_file():
        pytest.skip("local TweetEval data not staged; run setup/fetch_inputs.sh")
    first, train, val, panel, warm = configure(spec, sample_size=17, warmup_rows=8, seed=1729)
    second = configure(spec, sample_size=17, warmup_rows=8, seed=1729)[0]
    assert len(train) == 45615 and len(val) == 2000
    assert first == second and identity(panel) == first["test_panel_sha256"]
    assert warm == [val[i] for i in first["warmup_val_ids"]]


def test_preflight_skips_unfit_and_checks_file_size(tmp_path):
    gpu = {"index": "0", "uuid": "gpu0", "memory.total": "12000", "memory.free": "11000"}
    host = {"gpu": [gpu], "gpu_compute_apps": ""}
    config = {"weight_shards_bytes": {"weights.safetensors": MIB * 15000}, "reserve_mib": 1000, "gpu_index": 0}
    spec = {"device": "cuda", "revision": "pinned", "local_snapshot": str(tmp_path), "engine_safety": config}
    assert "alone exceed GPU" in preflight(spec, host)[1]
    config["weight_shards_bytes"]["weights.safetensors"] = 17
    assert "size-mismatched" in preflight(spec, host)[1]
    (tmp_path / "weights.safetensors").write_bytes(b"a" * 17)
    (tmp_path / "config.json").write_text("{}")
    assert preflight(spec, host)[1] is None
    (tmp_path / "config.json").write_text('{"text_config":{"quantization_config":{"quant_method":"awq"}}}')
    assert "forbids" in preflight(spec, host)[1]


def test_exact_quality_from_first_saved_pass_missing_counts_and_partial_group(tmp_path):
    class Model:
        def predict(self, texts):
            return [{"label": None if t == "neutral" else t, "completed": t != "neutral", "raw": t} for t in texts]

    panel = [(i, name, i) for i, name in enumerate(LABELS)]
    rows = run_workload(Model(), panel, 2, 3)
    file = tmp_path / "predictions.jsonl"
    persist(file, rows)
    args = {"test_ids": [0, 1, 2], "gold_by_id": dict(enumerate(LABELS)), "batch_size": 2,
            "passes": 3, "adapter_kind": "nb", "window_s": 3}
    counts, quality, perf = audit(file, **args)
    assert counts["submitted"] == 9 and counts["valid"] == counts["completed"] == counts["correct"] == 6
    assert counts["groups"] == 6 and counts["final_group_width"] == 1
    assert quality["accuracy"] == quality["macro_recall"] == quality["macro_f1"] == pytest.approx(2 / 3)
    assert quality["basis_n"] == 3 and quality["repeat_disagreements_from_first"] == 0
    assert perf["throughput_tweets_s"] == 2
    altered = deepcopy(rows)
    altered[4]["gold"] = "positive"
    persist(file, altered)
    with pytest.raises(ValueError, match="panel/group mismatch"):
        audit(file, **args)
