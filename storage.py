"""Stable identifiers and durable, append-only attempt histories (Linux/local FS)."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile


def now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def canonical(data):
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def identity(data):
    return hashlib.sha256(canonical(data)).hexdigest()


def content_hash(path):
    with Path(path).open("rb") as src:
        return hashlib.file_digest(src, "sha256").hexdigest()


def load(path):
    def bad_constant(value):
        raise ValueError(f"non-finite JSON constant {value}")

    def unique(pairs):
        out = {}
        for key, val in pairs:
            if key in out:
                raise ValueError(f"duplicate JSON key: {key}")
            out[key] = val
        return out

    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=bad_constant, object_pairs_hook=unique)


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_bytes(path, data):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.tmp.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        sync_dir(path.parent)
    finally:
        Path(tmp).unlink(missing_ok=True)


def atomic_json(path, data):
    atomic_bytes(path, (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())


def sync_tree(root):
    """Flush native outputs before publishing a completed state."""
    for directory, _, files in os.walk(root, topdown=False):
        for name in files:
            path = Path(directory, name)
            if not path.is_symlink():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
        sync_dir(directory)


@contextmanager
def measurement_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"another benchmark holds {path}") from None
        yield


class Run:
    """The sole writer must hold the project measurement lock while using this API."""

    def __init__(self, out, trial):
        self.trial = trial
        self.path = Path(out, trial["run_id"])
        self.file = self.path / "state.json"
        self.state = load(self.file) if self.file.exists() else {
            "run_id": trial["run_id"], "trial": trial, "status": "pending", "active_attempt": None,
            "history": [],
        }
        if self.state["trial"] != trial or self.state["run_id"] != trial["run_id"]:
            raise ValueError(f"trial mismatch: {self.file}")

    def recover(self):
        if self.state["status"] == "running":
            self.finish("incomplete", "previous process left a running checkpoint")

    def start(self):
        if self.state["status"] not in {"pending", "incomplete", "failed", "skipped"}:
            raise ValueError(f"not retryable: {self.state['status']}")
        self.path.mkdir(exist_ok=True)
        sync_dir(self.path.parent)
        n = len(self.state["history"]) + 1
        while True:
            name = f"attempt-{n:04d}"
            attempt = self.path / name
            try:
                attempt.mkdir()
                break
            except FileExistsError:
                # A hard kill after mkdir but before writing state can leave an orphan dir.
                n += 1
        self.state["active_attempt"] = name
        self.state["status"] = "running"
        self.state["reason"] = None
        self.state["history"].append({"attempt": name, "status": "running", "started_at": now()})
        atomic_json(self.file, self.state)
        return attempt

    def finish(self, status, reason=None):
        if self.state["status"] != "running" or status not in {"complete", "incomplete", "failed", "skipped"}:
            raise ValueError(f"invalid transition to {status}")
        entry = self.state["history"][-1]
        if status == "complete":
            attempt = self.path / entry["attempt"]
            if not (attempt / "result.json").is_file():
                raise RuntimeError("result required before completion")
            sync_tree(attempt)
        entry.update(status=status, reason=reason, finished_at=now())
        self.state.update(status=status, reason=reason)
        atomic_json(self.file, self.state)
