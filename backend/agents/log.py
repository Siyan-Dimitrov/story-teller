"""JSONL append/read helpers with per-path locking.

Slice 2 keeps agent state in append-only JSONL files under projects/_agents/.
At 21 projects this stays trivial to grep and trivial to wipe. SQLite arrives
in slice 3 if the queue ever needs cross-process readers.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

from .. import config

log = logging.getLogger(__name__)


AGENTS_DIR = config.PROJECTS_DIR / "_agents"
COSTS_PATH = AGENTS_DIR / "costs.jsonl"
RUNS_PATH = AGENTS_DIR / "agent_runs.jsonl"
BUDGET_POLICIES_PATH = AGENTS_DIR / "budget_policies.json"


_LOCKS: dict[Path, threading.Lock] = {}
_DICT_LOCK = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    key = path.resolve()
    with _DICT_LOCK:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[key] = lock
        return lock


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append(path: Path, row: dict) -> None:
    """Append a single JSON object as one line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, default=str, separators=(",", ":")) + "\n"
    with _lock_for(path):
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
        _truncate_if_huge(path)


def read_all(path: Path) -> list[dict]:
    """Read every line, skipping malformed rows with a warning."""
    if not path.exists():
        return []
    out: list[dict] = []
    with _lock_for(path):
        for raw in path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError as e:
                log.warning(f"Skipping malformed JSONL line in {path.name}: {e}")
    return out


def _truncate_if_huge(path: Path, max_mb: int = 10) -> None:
    """Rotate to <name>.archive.jsonl when file exceeds max_mb. Caller holds the lock."""
    try:
        size_mb = path.stat().st_size / (1024 * 1024)
    except OSError:
        return
    if size_mb < max_mb:
        return
    archive = path.with_suffix(path.suffix + ".archive.jsonl")
    try:
        if archive.exists():
            archive.unlink()
        path.rename(archive)
        log.info(f"Rotated {path.name} -> {archive.name} ({size_mb:.1f}MB)")
    except OSError as e:
        log.warning(f"Could not rotate {path}: {e}")
