"""Producer agent — runs a book group end-to-end with critic-gated regenerate.

Press-Go autonomy. Calls the same module functions the legacy per-step
endpoints call, wrapped in retry + critic + budget logic. Writes through to
the existing _batch_progress dict so the BatchProgress UI keeps working;
extra fields are read by the new AgentsDashboard.

Pause/cancel reuses batch._batch_paused. Restart-resume is handled by
recovery.resume_unfinished() reading agent_runs.jsonl on startup.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from pathlib import Path
from typing import Optional

from .. import config, project_store as store
from .. import script_gen, voice_gen, image_gen, image_styles
from ..video_assembly import assemble_video
from ..models import DEFAULT_VOICE_INSTRUCT
from ..batch import _batch_progress, _batch_paused, _batch_run_config
from .base import AgentVerdict
from .budget import (
    BudgetExceeded,
    check_budget,
    ensure_budget_or_raise,
    get_policy,
    set_run_context,
    reset_run_context,
)
from .critic import critique_script
from .publisher import Publisher
from . import log as agent_log

log = logging.getLogger(__name__)


MAX_SCRIPT_RETRIES = 2
MAX_TRANSIENT_RETRIES = 3
RETRY_BACKOFFS = (2.0, 5.0, 15.0)
PRODUCER_AGENT = "producer"


class ProducerError(RuntimeError):
    """Hard failure that should mark the project failed."""


# ── Progress writes ──────────────────────────────────────────────

def _ensure_group_progress(group_id: str, project_ids: list[str], cap_cents: int) -> None:
    """Create or refresh _batch_progress[group_id] with producer-specific fields."""
    chapters = []
    already_completed = 0
    for pid in project_ids:
        try:
            state = store.load_state(pid)
        except FileNotFoundError:
            state = {}
        is_done = state.get("step") == "assembled"
        chapters.append({
            "project_id": pid,
            "chapter_index": state.get("chapter_index", 0),
            "title": state.get("title", ""),
            "status": "completed" if is_done else "pending",
            "current_step": None,
            "failed_step": None,
            "error": None,
            "critic_attempts": 0,
            "critic_verdict": None,
            "retry_count": 0,
        })
        if is_done:
            already_completed += 1

    _batch_progress[group_id] = {
        "group_id": group_id,
        "total": len(project_ids),
        "completed": already_completed,
        "failed": 0,
        "current_chapter": None,
        "current_step": None,
        "chapters": chapters,
        "finished": False,
        "paused": False,
        "source": "producer",
        "cost_cents": 0,
        "cap_cents": cap_cents,
    }


def _index_of(group_id: str, project_id: str) -> int:
    progress = _batch_progress.get(group_id) or {}
    for i, ch in enumerate(progress.get("chapters") or []):
        if ch.get("project_id") == project_id:
            return i
    return -1


def _emit(
    group_id: str,
    project_id: str,
    *,
    phase: str,
    step: Optional[str] = None,
    attempt: Optional[int] = None,
    critic_severity: Optional[str] = None,
    retry_count: Optional[int] = None,
    error: Optional[str] = None,
) -> None:
    row = {
        "ts": agent_log.now_iso(),
        "group_id": group_id,
        "project_id": project_id,
        "agent": PRODUCER_AGENT,
        "phase": phase,
    }
    if step is not None: row["step"] = step
    if attempt is not None: row["attempt"] = attempt
    if critic_severity is not None: row["critic_severity"] = critic_severity
    if retry_count is not None: row["retry_count"] = retry_count
    if error is not None: row["error"] = error[:500]
    agent_log.append(agent_log.RUNS_PATH, row)


def _refresh_cost(group_id: str) -> None:
    progress = _batch_progress.get(group_id)
    if not progress:
        return
    try:
        progress["cost_cents"] = check_budget(group_id).used_cents
    except Exception as e:
        log.warning(f"refresh_cost: {e}")


def _paused(group_id: str) -> bool:
    return bool(_batch_paused.get(group_id, False))


def _set_chapter(group_id: str, project_id: str, **fields) -> None:
    progress = _batch_progress.get(group_id)
    if not progress:
        return
    idx = _index_of(group_id, project_id)
    if idx < 0:
        return
    progress["chapters"][idx].update(fields)


# ── Retry helper ─────────────────────────────────────────────────

async def _with_retry(coro_factory, *, what: str, retries: int = MAX_TRANSIENT_RETRIES):
    """Run an async callable, retrying on transient failures with backoff."""
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            return await coro_factory()
        except (BudgetExceeded, ProducerError):
            raise
        except Exception as e:
            last_exc = e
            if attempt < retries - 1:
                backoff = RETRY_BACKOFFS[min(attempt, len(RETRY_BACKOFFS) - 1)]
                log.warning(f"{what} failed (attempt {attempt + 1}/{retries}): {e}; retrying in {backoff}s")
                await asyncio.sleep(backoff)
            else:
                log.error(f"{what} exhausted retries: {e}")
    raise ProducerError(f"{what} failed after {retries} attempts: {last_exc}")


# ── Per-step runners ────────────────────────────────────────────

async def _run_script_with_critic(project_id: str, group_id: str, cfg: dict) -> dict:
    """Generate script, run critic, regenerate up to MAX_SCRIPT_RETRIES with feedback."""
    state = store.load_state(project_id)
    pdir = store.project_dir(project_id)
    base_custom_prompt = state.get("custom_prompt", "")
    addendum = (cfg.get("script_prompt_addendum") or "").strip()
    if addendum:
        # Prefix so the addendum lands at the top of the user message — closest
        # to the system prompt, gets best LLM attention.
        base_custom_prompt = f"{addendum}\n\n{base_custom_prompt}".strip()
    skill_tone = cfg.get("tone")
    feedback_chunks: list[str] = []

    last_verdict: Optional[AgentVerdict] = None
    for attempt in range(MAX_SCRIPT_RETRIES + 1):
        _set_chapter(group_id, project_id, current_step="script", critic_attempts=attempt)
        store.update_state(project_id, step="generating_script", error=None)

        custom_prompt = base_custom_prompt
        if feedback_chunks:
            custom_prompt = (custom_prompt + "\n\n" + "\n\n".join(feedback_chunks)).strip()

        script = await _with_retry(
            lambda: script_gen.generate_script(
                source_tale=state.get("source_tale", ""),
                custom_prompt=custom_prompt,
                target_minutes=state.get("target_minutes", 5.0),
                ollama_model=state.get("ollama_model"),
                tone=skill_tone or state.get("tone", ""),
            ),
            what=f"script_gen({project_id} attempt {attempt + 1})",
        )
        store.save_json(project_id, "script.json", script)
        store.update_state(
            project_id,
            step="scripted",
            title=script.get("title", state.get("title", "")),
        )

        verdict = await critique_script(
            script=script,
            target_minutes=state.get("target_minutes", 5.0),
            tone=state.get("tone", ""),
            ollama_model=state.get("ollama_model"),
            skip_llm=cfg.get("critic_skip_llm", False),
        )
        last_verdict = verdict
        _set_chapter(
            group_id, project_id,
            critic_verdict=verdict.to_dict(),
            critic_attempts=attempt + 1,
        )
        _emit(
            group_id, project_id,
            phase="step_complete", step="script",
            attempt=attempt + 1, critic_severity=verdict.severity,
        )

        if verdict.accept:
            log.info(f"Producer {project_id}: script accepted on attempt {attempt + 1}")
            return script

        if attempt >= MAX_SCRIPT_RETRIES:
            if verdict.severity == "fatal":
                raise ProducerError(
                    f"Script critic returned fatal severity after {attempt + 1} attempts: {verdict.feedback}"
                )
            log.warning(
                f"Producer {project_id}: shipping script despite {verdict.severity} severity "
                f"after {attempt + 1} attempts"
            )
            return script

        feedback_chunks.append(
            f"Previous attempt issues (DO NOT repeat):\n{verdict.feedback}"
        )

    return script  # unreachable


async def _run_voice(project_id: str, group_id: str, cfg: dict) -> None:
    profile_id = cfg.get("voice_profile_id") or ""
    if not profile_id:
        raise ProducerError("voice_profile_id required for producer voice step")
    state = store.load_state(project_id)
    script = store.load_json(project_id, "script.json")
    pdir = store.project_dir(project_id)

    _set_chapter(group_id, project_id, current_step="voice")
    store.update_state(
        project_id,
        step="generating_voice",
        error=None,
        voice_profile_id=profile_id,
        voice_language=cfg.get("voice_language", "en"),
    )

    scenes = await _with_retry(
        lambda: voice_gen.generate_all_scenes(
            scenes=script["scenes"],
            profile_id=profile_id,
            language=cfg.get("voice_language", "en"),
            project_dir=pdir,
            instruct=cfg.get("voice_instruct", DEFAULT_VOICE_INSTRUCT),
        ),
        what=f"voice_gen({project_id})",
    )
    script["scenes"] = scenes
    store.save_json(project_id, "script.json", script)
    store.update_state(project_id, step="voiced")
    _emit(group_id, project_id, phase="step_complete", step="voice")


async def _run_images(project_id: str, group_id: str, cfg: dict) -> None:
    state = store.load_state(project_id)
    script = store.load_json(project_id, "script.json")
    pdir = store.project_dir(project_id)
    backend = cfg.get("image_backend") or state.get("image_backend") or "replicate"

    _set_chapter(group_id, project_id, current_step="images")
    store.update_state(
        project_id,
        step="generating_images",
        error=None,
        image_backend=backend,
    )

    style_prompt = cfg.get("style_prompt") or image_styles.DEFAULT_STYLE_PROMPT
    scenes = await _with_retry(
        lambda: image_gen.generate_all_scenes(
            scenes=script["scenes"],
            project_dir=pdir,
            backend=backend,
            style_prompt=style_prompt,
            lora_keys=cfg.get("lora_keys"),
            character_consistency=bool(cfg.get("character_consistency", False)),
            project_seed=store.get_project_seed(project_id),
        ),
        what=f"image_gen({project_id})",
    )
    script["scenes"] = scenes
    store.save_json(project_id, "script.json", script)
    store.update_state(project_id, step="illustrated")
    _refresh_cost(group_id)
    _emit(group_id, project_id, phase="step_complete", step="images")


async def _run_assemble(project_id: str, group_id: str) -> None:
    state = store.load_state(project_id)
    script = store.load_json(project_id, "script.json")
    pdir = store.project_dir(project_id)

    _set_chapter(group_id, project_id, current_step="assemble")
    store.update_state(project_id, step="assembling", error=None)

    # assemble_video is sync; run in executor so we don't block the loop.
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        lambda: assemble_video(
            scenes=script["scenes"],
            project_dir=pdir,
            project_id=project_id,
            music_track=state.get("music_track"),
            music_volume=state.get("music_volume"),
        ),
    )
    store.update_state(project_id, step="assembled")
    _emit(group_id, project_id, phase="step_complete", step="assemble")


async def _run_publish(project_id: str, group_id: str) -> None:
    _set_chapter(group_id, project_id, current_step="publish")
    state = store.load_state(project_id)
    await Publisher().publish(project_id, ollama_model=state.get("ollama_model"))
    _emit(group_id, project_id, phase="step_complete", step="publish")


# ── Per-project orchestrator ────────────────────────────────────

async def run_project(project_id: str, group_id: str, cfg: dict) -> None:
    """Walk one project script → critic → voice → budget → images → assemble → publish."""
    token = set_run_context({
        "project_id": project_id,
        "group_id": group_id,
        "agent": PRODUCER_AGENT,
    })
    _emit(group_id, project_id, phase="start")
    try:
        # script + critic
        await _run_script_with_critic(project_id, group_id, cfg)
        if _paused(group_id):
            _emit(group_id, project_id, phase="paused")
            return

        # voice
        await _run_voice(project_id, group_id, cfg)
        if _paused(group_id):
            _emit(group_id, project_id, phase="paused")
            return

        # budget gate before paid image calls
        ensure_budget_or_raise(group_id)

        # images
        await _run_images(project_id, group_id, cfg)
        if _paused(group_id):
            _emit(group_id, project_id, phase="paused")
            return

        # assemble
        await _run_assemble(project_id, group_id)
        if _paused(group_id):
            _emit(group_id, project_id, phase="paused")
            return

        # publish (no upload — just metadata file)
        await _run_publish(project_id, group_id)

        _set_chapter(group_id, project_id, status="completed", current_step=None)
        _emit(group_id, project_id, phase="done")

    except BudgetExceeded as e:
        log.warning(f"Producer {project_id}: budget exceeded — {e}")
        store.update_state(project_id, error=f"budget_cap_reached: {e}")
        _set_chapter(
            group_id, project_id,
            status="failed", error=str(e),
            failed_step="budget", current_step=None,
        )
        _emit(group_id, project_id, phase="failed", error=str(e))
        raise
    except Exception as e:
        tb = traceback.format_exc()
        log.error(f"Producer {project_id} failed: {tb}")
        store.update_state(project_id, error=f"{e}\n{tb}")
        _set_chapter(
            group_id, project_id,
            status="failed", error=str(e), current_step=None,
        )
        _emit(group_id, project_id, phase="failed", error=str(e))
        raise
    finally:
        reset_run_context(token)


# ── Outer batch wrapper ─────────────────────────────────────────

async def run_producer_pipeline(
    group_id: str,
    project_ids: list[str],
    cfg: Optional[dict] = None,
) -> None:
    """Run a list of projects sequentially. Continues on per-project failure."""
    cfg = cfg or {}
    cap_cents = int(get_policy(group_id)["cap_cents"])

    # Race guard against legacy batch flow (see slice 2 plan, risk #5)
    existing = _batch_progress.get(group_id)
    if existing and not existing.get("finished") and existing.get("source") == "legacy":
        raise ProducerError(
            f"Group {group_id} already running under legacy batch flow; refusing to overlap."
        )

    _ensure_group_progress(group_id, project_ids, cap_cents)
    _batch_paused[group_id] = False
    _batch_run_config[group_id] = {"source": "producer", **cfg, "project_ids": project_ids}

    progress = _batch_progress[group_id]
    for i, pid in enumerate(project_ids):
        if progress["chapters"][i]["status"] == "completed":
            log.info(f"Producer {group_id}: chapter {i} ({pid}) already complete; skipping")
            continue
        if _paused(group_id):
            progress["paused"] = True
            log.info(f"Producer {group_id}: paused before chapter {i}")
            return
        progress["current_chapter"] = i
        progress["chapters"][i]["status"] = "running"
        try:
            await run_project(pid, group_id, cfg)
            progress["completed"] += 1
        except Exception as e:
            progress["failed"] += 1
            log.error(f"Producer {group_id}: chapter {i} ({pid}) failed: {e}")
        finally:
            _refresh_cost(group_id)

    progress["current_chapter"] = None
    progress["current_step"] = None
    progress["paused"] = False
    progress["finished"] = True
    log.info(
        f"Producer {group_id} finished: {progress['completed']}/{progress['total']} done, "
        f"{progress['failed']} failed, {progress['cost_cents']}¢ used of {cap_cents}¢ cap"
    )
