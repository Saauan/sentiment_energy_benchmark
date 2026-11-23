"""Closed-loop inference and quality calculated exclusively from persisted predictions."""
from collections import Counter
import json
import math
from pathlib import Path
from time import perf_counter

from dataset import LABELS
from storage import atomic_bytes, canonical, content_hash


def groups(rows, batch_size):
    for i in range(0, len(rows), batch_size):
        yield rows[i:i + batch_size]


def predict_group(adapter, rows):
    start = perf_counter()
    result = adapter.predict([r[1] for r in rows])
    elapsed = perf_counter() - start
    if not isinstance(result, list) or len(result) != len(rows):
        raise ValueError("adapter response cardinality mismatch: explicit missing markers required")
    records = []
    for (row_id, _, gid), value in zip(rows, result, strict=True):
        if not isinstance(value, dict) or not {"label", "raw"} <= set(value):
            raise ValueError("adapter response must provide label and raw")
        label = value["label"]
        if label not in (*LABELS, None):
            raise ValueError(f"adapter response has unparsed label: {label!r}")
        completed = value.get("completed", True)
        if type(completed) is not bool or (not completed and label is not None):
            raise ValueError("invalid completion/label flags")
        latency = value.get("latency_s", elapsed)
        if isinstance(latency, bool) or not math.isfinite(latency) or latency < 0:
            raise ValueError("invalid adapter latency")
        # Fallback latency is completion of its full static group, not model-only time.
        records.append({"id": row_id, "gold": LABELS[gid], "predicted": label, "raw": value["raw"],
                        "completed": completed, "latency_s": latency, "latency_scope": "adapter request"
                        if "latency_s" in value else "closed-loop group completion", "group_s": elapsed})
    return records


def run_workload(adapter, panel, batch_size, passes):
    result = []
    for pass_idx in range(passes):
        for group_idx, rows in enumerate(groups(panel, batch_size)):
            for row in predict_group(adapter, rows):
                row.update(pass_index=pass_idx, group_index=group_idx, group_width=len(rows))
                result.append(row)
    return result


def persist(path, rows):
    atomic_bytes(path, b"".join(canonical(row) + b"\n" for row in rows))
    return content_hash(path)


def _percentile(values, frac):
    vals = sorted(values)
    pos = (len(vals) - 1) * frac
    lo = int(pos)
    return vals[lo] + (vals[math.ceil(pos)] - vals[lo]) * (pos - lo)


def _divide(num, denom):
    return num / denom if denom else 0.0


def _quality(rows):
    per_class = {}
    for label in LABELS:
        gold = sum(row["gold"] == label for row in rows)
        pred = sum(row["predicted"] == label for row in rows)
        tp = sum(row["gold"] == row["predicted"] == label for row in rows)
        precision, recall = _divide(tp, pred), _divide(tp, gold)
        per_class[label] = {"precision": precision, "recall": recall, "f1": _divide(2 * precision * recall, precision + recall),
                            "support": gold, "predicted": pred, "correct": tp}
    return {"per_class": per_class, "macro_recall": sum(p["recall"] for p in per_class.values()) / 3,
            "macro_f1": sum(p["f1"] for p in per_class.values()) / 3,
            "accuracy": sum(row["predicted"] == row["gold"] for row in rows) / len(rows),
            "coverage": sum(row["predicted"] in LABELS for row in rows) / len(rows)}


def audit(path, *, test_ids, gold_by_id, batch_size, passes, adapter_kind, window_s):
    rows = [json.loads(line) for line in Path(path).read_bytes().splitlines()]
    n = len(test_ids)
    if len(rows) != n * passes or n == 0 or window_s <= 0:
        raise ValueError("saved prediction count/window mismatch")
    group_latencies = []
    for i, row in enumerate(rows):
        pos = i % n
        group = pos // batch_size
        tid = test_ids[pos]
        if (row["pass_index"], row["id"], row["gold"], row["group_index"], row["group_width"]) != (
                i // n, tid, gold_by_id[tid], group, min(batch_size, n - group * batch_size)):
            raise ValueError(f"saved prediction panel/group mismatch at row {i}")
        if row["predicted"] not in (*LABELS, None) or type(row["completed"]) is not bool or (
                not row["completed"] and row["predicted"] is not None):
            raise ValueError("saved prediction state is invalid")
        if pos % batch_size == 0:
            group_latencies.append(row["group_s"])
    valid = sum(r["predicted"] in LABELS for r in rows)
    correct = sum(r["predicted"] == r["gold"] for r in rows)
    completed = sum(r["completed"] for r in rows)
    counts = {"submitted": len(rows), "completed": completed, "valid": valid, "correct": correct,
              "pass_count": passes, "panel_size": n, "groups": len(group_latencies), "retry_requests": 0,
              "http_request_count": len(rows) if adapter_kind == "vllm" else 0,
              "qev_choice_count": len(rows) if adapter_kind == "qev" else 0,
              "final_group_width": n % batch_size or min(batch_size, n)}
    quality = _quality(rows[:n])
    quality.update(basis_pass=0, basis_n=n, repeat_disagreements_from_first=sum(
        row["predicted"] != rows[i % n]["predicted"] for i, row in enumerate(rows[n:], start=n)))
    latencies = [r["latency_s"] for r in rows if r["completed"]]
    performance = {"window_s": window_s, "throughput_tweets_s": completed / window_s,
                   "valid_throughput_tweets_s": valid / window_s,
                   "latency_p50_s": _percentile(latencies, .50) if latencies else None,
                   "latency_p95_s": _percentile(latencies, .95) if latencies else None,
                   "latency_scopes": dict(Counter(r["latency_scope"] for r in rows)),
                   "group_latency_p50_s": _percentile(group_latencies, .50),
                   "group_latency_p95_s": _percentile(group_latencies, .95)}
    return counts, quality, performance
