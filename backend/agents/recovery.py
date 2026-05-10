"""Restart-safe recovery for in-flight Producer runs.

On backend startup we scan agent_runs.jsonl, reduce to per-project last phase,
and re-spawn Producer for any project whose latest phase is start/step_complete/paused
(i.e., not done/failed). Per-project state.step on disk tells the producer
which steps to skip.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Optional

from . import log as agent_log
from . import producer as agents_producer
from .. import project_store as store

log = logging.getLogger(__name__)


# A project is "live" if its last logged phase is one of these.
LIVE_PHASES = {"start", "step_complete", "paused"}
TERMINAL_PHASES = {"done", "failed"}


def _last_phase_per_project() -> dict[tuple[str, str], dict]:
    """Reduce agent_runs.jsonl to {(group_id, project_id): last_row}."""
    rows = agent_log.read_all(agent_log.RUNS_PATH)
    latest: dict[tuple[str, str], dict] = {}
    for row in rows:
        gid, pid = row.get("group_id"), row.get("project_id")
        if not gid or not pid:
            continue
        latest[(gid, pid)] = row
    return latest


def _live_projects_by_group() -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for (gid, pid), row in _last_phase_per_project().items():
        if row.get("phase") in LIVE_PHASES:
            grouped[gid].append(pid)
    return grouped


async def resume_unfinished(*, dry_run: bool = False) -> list[str]:
    """Resume any in-flight Producer runs.

    Returns list of "<group_id>/<project_id>" identifiers that were resumed.
    With dry_run=True, returns the same list without actually launching.
    """
    grouped = _live_projects_by_group()
    if not grouped:
        return []

    resumed: list[str] = []
    for gid, pids in grouped.items():
        # Filter to projects that still need work — skip if state.step == 'assembled'
        # AND latest phase wasn't 'done' (i.e., assemble done but publish never recorded).
        actionable: list[str] = []
        for pid in pids:
            try:
                state = store.load_state(pid)
            except FileNotFoundError:
                log.info(f"Recovery: {pid} no longer exists; skipping")
                continue
            actionable.append(pid)
        if not actionable:
            continue

        ident = f"{gid}/{','.join(actionable)}"
        resumed.append(ident)
        if dry_run:
            continue

        log.info(f"Recovery: resuming Producer for group {gid} ({len(actionable)} projects)")
        # Fire-and-forget; producer manages its own state.
        asyncio.create_task(
            agents_producer.run_producer_pipeline(gid, actionable, cfg={})
        )
    return resumed
