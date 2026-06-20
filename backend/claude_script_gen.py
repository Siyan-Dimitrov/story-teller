"""Claude-powered screenplay generation.

Three-pass pipeline (writer → critic → reviser) driven by the Claude Agent
SDK. Authenticates against the user's existing Claude Code OAuth credentials
(``~/.claude/.credentials.json``), so no Anthropic API key is required when
the user is signed into Claude Code.

Exposes ``generate_script(...)`` with the same signature shape as
``script_gen.generate_script`` so ``main.py`` can route either backend.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from . import config
from .grimm_tales import get_tale
from .script_gen import normalize_scenes, _repair_truncated_json, _extract_llm_content

log = logging.getLogger(__name__)


class ClaudeAuthError(RuntimeError):
    """Raised when the Claude Agent SDK cannot authenticate."""


class ClaudeBackendError(RuntimeError):
    """Raised when the Claude backend fails for a non-auth reason."""


@dataclass(frozen=True)
class RoleSpec:
    """Which provider+model runs a given pass of the pipeline."""
    provider: str  # "claude" | "ollama"
    model: str

    def label(self) -> str:
        return f"{self.provider}:{self.model}"


def _parse_role(value: str | None, fallback_claude_model: str) -> RoleSpec:
    """Parse a role spec.

    Accepts:
      - empty / None         → claude with ``fallback_claude_model``
      - "claude:opus-4-7"    → explicit Claude
      - "ollama:kimi-k2.5:cloud" → explicit Ollama (model may itself contain colons)
      - "claude-sonnet-4-5"  → bare model name, provider inferred as claude
      - "kimi-k2.5:cloud"    → bare model name with a colon — inferred as ollama
    """
    if not value:
        return RoleSpec("claude", fallback_claude_model)
    v = value.strip()
    if v.startswith("claude:"):
        return RoleSpec("claude", v[len("claude:"):].strip())
    if v.startswith("ollama:"):
        return RoleSpec("ollama", v[len("ollama:"):].strip())
    if v.lower().startswith("claude-"):
        return RoleSpec("claude", v)
    # A bare model name with a colon almost always means an Ollama tag
    # (e.g. "kimi-k2.5:cloud", "llama3.2:3b"). Fall through to Claude only
    # for colon-free names that don't match the "claude-*" pattern.
    if ":" in v:
        return RoleSpec("ollama", v)
    return RoleSpec("claude", v)


_PROMPT_CACHE: dict[str, str] = {}


def _read_prompt(name: str) -> str:
    """Read a prompt file from ``config.CLAUDE_PROMPTS_DIR`` with caching."""
    if name in _PROMPT_CACHE:
        return _PROMPT_CACHE[name]
    path: Path = config.CLAUDE_PROMPTS_DIR / name
    if not path.exists():
        raise ClaudeBackendError(f"Missing prompt file: {path}")
    text = path.read_text(encoding="utf-8")
    _PROMPT_CACHE[name] = text
    return text


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return text


def _extract_first_json_object(text: str) -> str:
    """Slice out the first top-level ``{...}`` block, in case the model
    surrounds it with stray prose. We still fail fast on real garbage."""
    start = text.find("{")
    if start < 0:
        return text
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def _parse_json(raw: str) -> dict[str, Any]:
    """Parse model output as JSON. Tolerates code fences and a trailing
    truncation. Raises ``ClaudeBackendError`` on unrecoverable garbage."""
    text = _strip_code_fences(raw)
    text = _extract_first_json_object(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        log.warning("Claude JSON parse failed (%s) — attempting truncation repair", e)
        try:
            return json.loads(_repair_truncated_json(text))
        except json.JSONDecodeError as e2:
            preview = text[:400].replace("\n", " ")
            raise ClaudeBackendError(
                f"Claude returned malformed JSON: {e2}. Preview: {preview!r}"
            ) from e2


def _validate_script(script: dict[str, Any]) -> list[str]:
    """Return a list of human-readable validation errors. Empty list = valid."""
    errors: list[str] = []
    if not isinstance(script, dict):
        return ["root is not an object"]
    if not isinstance(script.get("title", ""), str):
        errors.append("title must be a string")
    if not isinstance(script.get("synopsis", ""), str):
        errors.append("synopsis must be a string")
    scenes = script.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        errors.append("scenes must be a non-empty array")
        return errors
    for i, scene in enumerate(scenes):
        if not isinstance(scene, dict):
            errors.append(f"scene[{i}] is not an object")
            continue
        if not isinstance(scene.get("narration", ""), str) or not scene.get("narration", "").strip():
            errors.append(f"scene[{i}].narration must be a non-empty string")
        prompts = scene.get("image_prompts")
        single = scene.get("image_prompt")
        if isinstance(prompts, list) and prompts:
            if not all(isinstance(p, str) and p.strip() for p in prompts):
                errors.append(f"scene[{i}].image_prompts entries must be non-empty strings")
        elif isinstance(single, str) and single.strip():
            pass  # legacy single-prompt shape — normalize_scenes will lift it
        else:
            errors.append(f"scene[{i}] must have image_prompts (array) or image_prompt (string)")
    return errors


async def _run_claude(
    *, system_prompt: str, user_prompt: str, model: str, pass_name: str,
) -> tuple[str, float]:
    """Run one Claude pass via the Agent SDK. Returns (text, cost_usd).

    On Windows the FastAPI process runs under ``WindowsSelectorEventLoopPolicy``
    (see ``backend/main.py``), and that loop cannot spawn subprocesses — but
    the SDK shells out to ``claude.exe``. We bounce the SDK call into a
    worker thread with its own ``ProactorEventLoop`` so the subprocess
    transport works without changing the parent loop policy.
    """
    try:
        from claude_agent_sdk import (
            ClaudeAgentOptions,
            query,
            AssistantMessage,
            ResultMessage,
            TextBlock,
        )
    except ImportError as e:
        raise ClaudeBackendError(
            "claude-agent-sdk is not installed. Run `pip install -r requirements.txt`."
        ) from e

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        model=model,
        max_turns=1,
        allowed_tools=[],
        disallowed_tools=[
            "Read", "Write", "Edit", "Bash", "Glob", "Grep",
            "WebFetch", "WebSearch", "TaskCreate", "TaskUpdate", "TaskList",
            "NotebookEdit",
        ],
        permission_mode="bypassPermissions",
        setting_sources=[],
    )

    async def _do_call() -> tuple[str, float]:
        chunks: list[str] = []
        cost_usd = 0.0
        async for message in query(prompt=user_prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        chunks.append(block.text)
            elif isinstance(message, ResultMessage):
                cost_usd = float(getattr(message, "total_cost_usd", 0.0) or 0.0)
        return "".join(chunks).strip(), cost_usd

    def _thread_run() -> tuple[str, float]:
        if sys.platform == "win32":
            loop = asyncio.ProactorEventLoop()
        else:
            loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                asyncio.wait_for(_do_call(), timeout=config.CLAUDE_TIMEOUT_SECONDS)
            )
        finally:
            loop.close()

    try:
        text, cost_usd = await asyncio.to_thread(_thread_run)
    except asyncio.TimeoutError as e:
        raise ClaudeBackendError(
            f"Claude {pass_name} pass timed out after {config.CLAUDE_TIMEOUT_SECONDS:.0f}s"
        ) from e
    except Exception as e:
        msg = str(e).lower()
        if "credential" in msg or "unauthorized" in msg or "auth" in msg or "login" in msg:
            raise ClaudeAuthError(
                "Claude Agent SDK could not authenticate. Run `claude login` once "
                "to sign in with your Claude Code subscription, then retry."
            ) from e
        raise ClaudeBackendError(f"Claude {pass_name} pass failed: {e}") from e

    if not text:
        raise ClaudeBackendError(f"Claude {pass_name} pass returned no text content")
    return text, cost_usd


async def _run_ollama(
    *, system_prompt: str, user_prompt: str, model: str, pass_name: str,
) -> tuple[str, float]:
    """Run one Ollama pass via /api/chat. Returns (text, 0.0).

    Cost is always 0 — Ollama runs locally (or against the user's own
    Ollama-compatible cloud endpoint). Reuses the same response-extraction
    logic as ``script_gen.py`` to handle thinking-model variants.
    """
    base_url = config.OLLAMA_URL
    try:
        async with httpx.AsyncClient(timeout=config.CLAUDE_TIMEOUT_SECONDS) as client:
            resp = await client.post(
                f"{base_url}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "options": {
                        "temperature": config.LLM_TEMPERATURE,
                        "num_predict": config.LLM_MAX_TOKENS,
                    },
                },
            )
    except httpx.TimeoutException as e:
        raise ClaudeBackendError(
            f"Ollama {pass_name} pass timed out after {config.CLAUDE_TIMEOUT_SECONDS:.0f}s"
        ) from e
    except httpx.HTTPError as e:
        raise ClaudeBackendError(
            f"Ollama {pass_name} pass failed: {e} (check OLLAMA_URL={base_url})"
        ) from e

    if resp.status_code != 200:
        raise ClaudeBackendError(
            f"Ollama {pass_name} pass returned {resp.status_code}: {resp.text[:300]}"
        )
    text = _extract_llm_content(resp.json()).strip()
    if not text:
        raise ClaudeBackendError(f"Ollama {pass_name} pass returned no content")
    return text, 0.0


async def _run_pass(
    *, role: RoleSpec, system_prompt: str, user_prompt: str, pass_name: str,
) -> tuple[str, float]:
    """Dispatch one pass to the configured provider."""
    log.info("Pass %s: %s", pass_name, role.label())
    if role.provider == "claude":
        return await _run_claude(
            system_prompt=system_prompt, user_prompt=user_prompt,
            model=role.model, pass_name=pass_name,
        )
    if role.provider == "ollama":
        return await _run_ollama(
            system_prompt=system_prompt, user_prompt=user_prompt,
            model=role.model, pass_name=pass_name,
        )
    raise ClaudeBackendError(f"Unknown provider for {pass_name}: {role.provider!r}")


def _build_writer_user_prompt(
    *,
    source_tale: str,
    custom_prompt: str,
    target_minutes: float,
    tone: str,
) -> str:
    parts: list[str] = []
    if source_tale:
        tale = get_tale(source_tale)
        if tale:
            parts.append("Adapt this story into your narrator voice:")
            parts.append(f"Title: {tale['title']}")
            parts.append(f"Origin: {tale['origin']}")
            parts.append(f"Synopsis:\n{tale['synopsis']}")
        else:
            parts.append(f"Adapt this story: {source_tale}")
    if tone:
        parts.append(f"\nAdaptation tone: {tone}. Infuse the story with this tone throughout.")
    if custom_prompt:
        if len(custom_prompt) > 160_000:
            # Claude Sonnet's context comfortably exceeds typical chapter
            # length; this guard only protects against accidental whole-book
            # paste. Real long-form handling lives in a future streaming path.
            log.warning("Custom prompt is %d chars — truncating to 160000", len(custom_prompt))
            custom_prompt = custom_prompt[:160_000] + "\n[... truncated ...]"
        parts.append(f"\nSource material / additional direction:\n{custom_prompt}")
    scene_count = max(5, int(target_minutes * 1.5))
    parts.append(f"\nTarget length: approximately {target_minutes} minutes when narrated aloud.")
    parts.append(f"Aim for roughly {scene_count} scenes.")
    parts.append("\nReturn the screenplay JSON object now.")
    return "\n".join(parts)


def _build_critic_user_prompt(*, source_summary: str, draft: dict[str, Any]) -> str:
    return (
        "Source material the writer was given:\n"
        f"{source_summary}\n\n"
        "Writer's draft (JSON):\n"
        f"{json.dumps(draft, ensure_ascii=False, indent=2)}\n\n"
        "Return your critique JSON now."
    )


def _build_reviser_user_prompt(
    *,
    source_summary: str,
    draft: dict[str, Any],
    critique: dict[str, Any],
) -> str:
    return (
        "Source material:\n"
        f"{source_summary}\n\n"
        "Your draft (JSON):\n"
        f"{json.dumps(draft, ensure_ascii=False, indent=2)}\n\n"
        "Editor's critique (JSON):\n"
        f"{json.dumps(critique, ensure_ascii=False, indent=2)}\n\n"
        "Return the revised screenplay JSON now."
    )


def _summarize_source(
    *,
    source_tale: str,
    custom_prompt: str,
    tone: str,
    target_minutes: float,
) -> str:
    """Compact context for critic/reviser passes. Avoids re-pasting a whole
    chapter twice through the prompt while still anchoring revisions."""
    parts: list[str] = [f"Target length: {target_minutes} minutes."]
    if tone:
        parts.append(f"Tone: {tone}.")
    if source_tale:
        tale = get_tale(source_tale)
        if tale:
            parts.append(f"Title: {tale['title']} ({tale['origin']})")
            parts.append(f"Synopsis: {tale['synopsis']}")
        else:
            parts.append(f"Source: {source_tale}")
    if custom_prompt:
        snippet = custom_prompt if len(custom_prompt) <= 8_000 else custom_prompt[:8_000] + " [...]"
        parts.append(f"Source / direction:\n{snippet}")
    return "\n".join(parts)


def _resolve_roles(
    *,
    claude_model: str | None,
    writer_model: str | None,
    critic_model: str | None,
    reviser_model: str | None,
) -> tuple[RoleSpec, RoleSpec, RoleSpec]:
    """Resolve the three role specs in this priority order, per role:

    1. explicit per-role argument (from API request / project state)
    2. matching env-var override (PIPELINE_<ROLE>_MODEL)
    3. the project's ``claude_model`` (or env CLAUDE_MODEL) used for all roles
    """
    fallback = (claude_model or config.CLAUDE_MODEL).strip()
    writer = _parse_role(writer_model or config.PIPELINE_WRITER_MODEL or None, fallback)
    critic = _parse_role(critic_model or config.PIPELINE_CRITIC_MODEL or None, fallback)
    reviser = _parse_role(reviser_model or config.PIPELINE_REVISER_MODEL or None, fallback)
    return writer, critic, reviser


async def generate_script(
    source_tale: str = "",
    custom_prompt: str = "",
    target_minutes: float = 5.0,
    claude_model: str | None = None,
    pipeline_writer_model: str | None = None,
    pipeline_critic_model: str | None = None,
    pipeline_reviser_model: str | None = None,
    tone: str = "",
    max_revisions: int | None = None,
    **_ignored,  # absorb ollama_* args so main.py can call either backend uniformly
) -> dict[str, Any]:
    """Generate a screenplay using the three-pass writer/critic/reviser pipeline.

    Each pass can run on a different provider+model. By default all three use
    the project's ``claude_model``; pass ``pipeline_*_model`` to override.

    The return shape matches ``script_gen.generate_script`` so downstream
    pipeline stages (voice, images, assembly) need no changes.
    """
    writer_role, critic_role, reviser_role = _resolve_roles(
        claude_model=claude_model,
        writer_model=pipeline_writer_model,
        critic_model=pipeline_critic_model,
        reviser_model=pipeline_reviser_model,
    )
    revisions = config.CLAUDE_MAX_REVISIONS if max_revisions is None else max_revisions
    log.info(
        "Screenplay pipeline: writer=%s critic=%s reviser=%s target=%smin revisions=%d",
        writer_role.label(), critic_role.label(), reviser_role.label(),
        target_minutes, revisions,
    )

    writer_system = _read_prompt("screenwriter_system.md")
    writer_user = _build_writer_user_prompt(
        source_tale=source_tale,
        custom_prompt=custom_prompt,
        target_minutes=target_minutes,
        tone=tone,
    )

    # ── Pass 1: writer ────────────────────────────────────────
    draft_raw, draft_cost = await _run_pass(
        role=writer_role,
        system_prompt=writer_system,
        user_prompt=writer_user,
        pass_name="writer",
    )
    draft = _parse_json(draft_raw)
    draft_errors = _validate_script(draft)
    if draft_errors:
        log.warning("Writer draft validation errors: %s — requesting repair", draft_errors)
        repair_prompt = (
            writer_user
            + "\n\nYour previous output had these schema problems:\n"
            + "\n".join(f"- {e}" for e in draft_errors)
            + "\nReturn a corrected JSON object."
        )
        draft_raw, repair_cost = await _run_pass(
            role=writer_role,
            system_prompt=writer_system,
            user_prompt=repair_prompt,
            pass_name="writer-repair",
        )
        draft_cost += repair_cost
        draft = _parse_json(draft_raw)
        draft_errors = _validate_script(draft)
        if draft_errors:
            raise ClaudeBackendError(
                "Writer output failed schema validation twice: " + "; ".join(draft_errors)
            )

    total_cost = draft_cost
    final = draft

    # ── Passes 2 & 3: critic + reviser ────────────────────────
    if revisions > 0:
        source_summary = _summarize_source(
            source_tale=source_tale,
            custom_prompt=custom_prompt,
            tone=tone,
            target_minutes=target_minutes,
        )
        critic_raw, critic_cost = await _run_pass(
            role=critic_role,
            system_prompt=_read_prompt("screenplay_critic_system.md"),
            user_prompt=_build_critic_user_prompt(source_summary=source_summary, draft=draft),
            pass_name="critic",
        )
        total_cost += critic_cost
        try:
            critique = _parse_json(critic_raw)
        except ClaudeBackendError as e:
            log.warning("Critic JSON unparseable (%s) — shipping draft", e)
            critique = {"overall": "ship", "issues": []}

        verdict = str(critique.get("overall", "ship")).lower()
        issues = critique.get("issues") or []
        if verdict == "revise" and issues:
            log.info("Critic flagged %d issues — running reviser", len(issues))
            revised_raw, revised_cost = await _run_pass(
                role=reviser_role,
                system_prompt=_read_prompt("screenwriter_reviser_system.md"),
                user_prompt=_build_reviser_user_prompt(
                    source_summary=source_summary, draft=draft, critique=critique,
                ),
                pass_name="reviser",
            )
            total_cost += revised_cost
            revised = _parse_json(revised_raw)
            revised_errors = _validate_script(revised)
            if revised_errors:
                log.warning(
                    "Reviser output failed validation (%s) — keeping original draft",
                    revised_errors,
                )
            else:
                final = revised
        else:
            log.info("Critic verdict: ship")

    final = normalize_scenes(final)
    final.setdefault("_claude_cost_usd", round(total_cost, 4))
    final.setdefault("_pipeline_models", {
        "writer": writer_role.label(),
        "critic": critic_role.label(),
        "reviser": reviser_role.label(),
    })
    log.info("Screenplay pipeline complete — notional cost ≈ $%.4f", total_cost)
    return final
