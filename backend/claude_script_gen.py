"""Claude-powered screenplay generation.

Three-pass pipeline (writer → critic → reviser) driven by the Claude Agent
SDK via :mod:`backend.llm`. Authenticates against the user's existing Claude
Code OAuth credentials (``~/.claude/.credentials.json``), so no Anthropic API
key is required when the user is signed into Claude Code.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from . import config, llm
from .llm import ClaudeAuthError, ClaudeBackendError, parse_json  # noqa: F401 — re-exported
from .grimm_tales import get_tale
from .script_gen import normalize_scenes

log = logging.getLogger(__name__)


def _resolve_model(value: str | None, fallback: str) -> str:
    """Resolve a per-role model override to a bare Claude model name.

    Tolerates the legacy ``claude:<model>`` prefix from before master became
    Claude-only; anything else is used verbatim.
    """
    if not value:
        return fallback
    v = value.strip()
    if v.startswith("claude:"):
        v = v[len("claude:"):].strip()
    return v or fallback


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


def _word_budget(target_minutes: float, voice_language: str) -> int:
    """Total English narration words that speak in ~target_minutes of audio."""
    factor = config.NARRATION_LANGUAGE_FACTORS.get(voice_language, 1.0)
    return int(round(target_minutes * config.NARRATION_WPM / factor / 10) * 10)


def _budget_lines(target_minutes: float, voice_language: str) -> list[str]:
    budget = _word_budget(target_minutes, voice_language)
    scene_count = max(3, round(target_minutes * 1.5))
    lines = [
        f"\nTarget length: {target_minutes} minutes of narrated audio.",
        f"HARD NARRATION BUDGET: about {budget} words of narration TOTAL across "
        "all scenes (stay within ±10%). This is calibrated to the narrator's "
        "measured speaking rate — every extra word pushes the video past its "
        "target length.",
    ]
    if config.NARRATION_LANGUAGE_FACTORS.get(voice_language, 1.0) != 1.0:
        lines.append(
            "The budget is already shortened because the narration will be "
            f"translated to another language ({voice_language}) that takes "
            "longer to speak — do not compensate."
        )
    lines.append(
        f"Aim for roughly {scene_count} scenes (~{budget // scene_count} words each)."
    )
    return lines


def _build_writer_user_prompt(
    *,
    source_tale: str,
    custom_prompt: str,
    target_minutes: float,
    tone: str,
    voice_language: str = "en",
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
            # Claude's context comfortably exceeds typical chapter length;
            # this guard only protects against accidental whole-book paste.
            log.warning("Custom prompt is %d chars — truncating to 160000", len(custom_prompt))
            custom_prompt = custom_prompt[:160_000] + "\n[... truncated ...]"
        parts.append(f"\nSource material / additional direction:\n{custom_prompt}")
    parts.extend(_budget_lines(target_minutes, voice_language))
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
    voice_language: str = "en",
) -> str:
    """Compact context for critic/reviser passes. Avoids re-pasting a whole
    chapter twice through the prompt while still anchoring revisions."""
    parts: list[str] = [
        f"Target length: {target_minutes} minutes. HARD narration budget: "
        f"~{_word_budget(target_minutes, voice_language)} words total (±10%) — "
        "flag the draft if it exceeds this; revisions must not add net length."
    ]
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


def _resolve_models(
    *,
    claude_model: str | None,
    writer_model: str | None,
    critic_model: str | None,
    reviser_model: str | None,
) -> tuple[str, str, str]:
    """Resolve the three role models in this priority order, per role:

    1. explicit per-role argument (from API request / project state)
    2. matching env-var override (PIPELINE_<ROLE>_MODEL)
    3. the project's ``claude_model`` (or env CLAUDE_MODEL) used for all roles
    """
    fallback = (claude_model or config.CLAUDE_MODEL).strip()
    writer = _resolve_model(writer_model or config.PIPELINE_WRITER_MODEL or None, fallback)
    critic = _resolve_model(critic_model or config.PIPELINE_CRITIC_MODEL or None, fallback)
    reviser = _resolve_model(reviser_model or config.PIPELINE_REVISER_MODEL or None, fallback)
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
    voice_language: str = "en",
    **_ignored,  # absorb legacy args (e.g. ollama_model) from old project state
) -> dict[str, Any]:
    """Generate a screenplay using the three-pass writer/critic/reviser pipeline.

    Each pass can run on a different Claude model. By default all three use
    the project's ``claude_model``; pass ``pipeline_*_model`` to override.
    """
    writer_model, critic_model, reviser_model = _resolve_models(
        claude_model=claude_model,
        writer_model=pipeline_writer_model,
        critic_model=pipeline_critic_model,
        reviser_model=pipeline_reviser_model,
    )
    revisions = config.CLAUDE_MAX_REVISIONS if max_revisions is None else max_revisions
    log.info(
        "Screenplay pipeline: writer=%s critic=%s reviser=%s target=%smin revisions=%d",
        writer_model, critic_model, reviser_model, target_minutes, revisions,
    )

    writer_system = _read_prompt("screenwriter_system.md")
    writer_user = _build_writer_user_prompt(
        source_tale=source_tale,
        custom_prompt=custom_prompt,
        target_minutes=target_minutes,
        tone=tone,
        voice_language=voice_language,
    )

    # ── Pass 1: writer ────────────────────────────────────────
    draft_raw, draft_cost = await llm.complete_with_cost(
        writer_system, writer_user, model=writer_model, pass_name="writer",
    )
    draft = parse_json(draft_raw)
    draft_errors = _validate_script(draft)
    if draft_errors:
        log.warning("Writer draft validation errors: %s — requesting repair", draft_errors)
        repair_prompt = (
            writer_user
            + "\n\nYour previous output had these schema problems:\n"
            + "\n".join(f"- {e}" for e in draft_errors)
            + "\nReturn a corrected JSON object."
        )
        draft_raw, repair_cost = await llm.complete_with_cost(
            writer_system, repair_prompt, model=writer_model, pass_name="writer-repair",
        )
        draft_cost += repair_cost
        draft = parse_json(draft_raw)
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
            voice_language=voice_language,
        )
        critic_raw, critic_cost = await llm.complete_with_cost(
            _read_prompt("screenplay_critic_system.md"),
            _build_critic_user_prompt(source_summary=source_summary, draft=draft),
            model=critic_model,
            pass_name="critic",
        )
        total_cost += critic_cost
        try:
            critique = parse_json(critic_raw)
        except ClaudeBackendError as e:
            log.warning("Critic JSON unparseable (%s) — shipping draft", e)
            critique = {"overall": "ship", "issues": []}

        verdict = str(critique.get("overall", "ship")).lower()
        issues = critique.get("issues") or []
        if verdict == "revise" and issues:
            log.info("Critic flagged %d issues — running reviser", len(issues))
            revised_raw, revised_cost = await llm.complete_with_cost(
                _read_prompt("screenwriter_reviser_system.md"),
                _build_reviser_user_prompt(
                    source_summary=source_summary, draft=draft, critique=critique,
                ),
                model=reviser_model,
                pass_name="reviser",
            )
            total_cost += revised_cost
            revised = parse_json(revised_raw)
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
        "writer": writer_model,
        "critic": critic_model,
        "reviser": reviser_model,
    })
    log.info("Screenplay pipeline complete — notional cost ≈ $%.4f", total_cost)
    return final
