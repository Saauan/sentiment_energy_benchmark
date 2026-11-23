"""Offline dataset adapters. Rows are (zero-based split index, text, integer label)."""
from pathlib import Path
from random import Random

from storage import content_hash, identity

ROOT = Path(__file__).resolve().parent
TWEETEVAL_REV = "4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66"
TWEETEVAL_URI = f"https://github.com/cardiffnlp/tweeteval/tree/{TWEETEVAL_REV}/datasets/sentiment"
LABELS = ("negative", "neutral", "positive")


def _line_pairs(root, **_spec):
    rows, files = {}, {}
    for split in ("train", "val", "test"):
        ptext = root / f"{split}_text.txt"
        pgold = root / f"{split}_labels.txt"
        texts = ptext.read_text(encoding="utf-8").split("\n")
        labels = pgold.read_text(encoding="utf-8").split("\n")
        if texts[-1] == "":
            texts.pop()
        if labels[-1] == "":
            labels.pop()
        if not texts or len(labels) != len(texts) or set(labels) - {"0", "1", "2"}:
            raise ValueError(f"malformed input split: {split}")
        rows[split] = [(i, t, int(y)) for i, (t, y) in enumerate(zip(texts, labels, strict=True))]
        files.update({p.name: content_hash(p) for p in (ptext, pgold)})
    return rows, files


def _tweeteval(root, **spec):
    if spec.get("revision") != TWEETEVAL_REV:
        raise ValueError(f"TweetEval adapter pinned to {TWEETEVAL_REV}")
    rows, files = _line_pairs(root)
    files["mapping.txt"] = content_hash(root / "mapping.txt")
    expected = {line.split()[1].rsplit("/", 1)[-1]: line.split()[0]
                for line in (ROOT / "setup/inputs.sha256").read_text().splitlines() if "data/tweeteval/" in line}
    if files != expected or dict(map(lambda kv: (kv[0], len(kv[1])), rows.items())) != {"train": 45615, "val": 2000, "test": 12284}:
        raise ValueError("TweetEval input size or hash mismatch; run setup/fetch_inputs.sh")
    return rows, files


ADAPTERS = {"tweeteval": _tweeteval, "line_pairs": _line_pairs}


def configure(spec, *, sample_size, warmup_rows, seed):
    adapter = spec["adapter"]
    root = ROOT / spec["path"]
    rows, files = ADAPTERS[adapter](root, revision=spec.get("revision"))
    if not (1 <= sample_size <= len(rows["test"])) or not (0 <= warmup_rows <= len(rows["val"])):
        raise ValueError("sample/warmup size exceeds dataset or is out of range")
    test_ids = sorted(Random(seed).sample(range(len(rows["test"])), sample_size))
    val_ids = sorted(Random(seed + 1).sample(range(len(rows["val"])), warmup_rows))
    panel = [rows["test"][i] for i in test_ids]
    warmup = [rows["val"][i] for i in val_ids]
    return {
        "name": spec.get("name", adapter), "adapter": adapter, "revision": spec.get("revision"),
        "source_uri": TWEETEVAL_URI if adapter == "tweeteval" else spec.get("source_uri"),
        "source_type": "public" if adapter == "tweeteval" else spec.get("source_type", "other"),
        "files_sha256": files, "split_sizes": {key: len(value) for key, value in rows.items()},
        "test_ids": test_ids, "warmup_val_ids": val_ids, "label_order": list(LABELS),
        "test_panel_sha256": identity(panel), "panel_seed": seed,
        "test_support": {label: sum(row[2] == i for row in panel) for i, label in enumerate(LABELS)},
    }, rows["train"], rows["val"], panel, warmup
