"""ScriptCritic — reviews a generated script and returns a structured verdict.

Programmatic checks catch the cheap, deterministic bugs the content audit
surfaced (truncation, empty prompts, scene drop-out). One LLM pass evaluates
narrative completeness — whether the last scene actually resolves anything.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from .base import (
    AgentVerdict,
    Issue,
    LLMCallError,
    call_llm_json,
    worst_severity,
)

log = logging.getLogger(__name__)


# A scene narration is "incomplete" if it ends without terminal punctuation
# or trails off on a connector word. Keeps the regex narrow on purpose — false
# positives turn into noisy regenerate loops.
_INCOMPLETE_END = re.compile(
    r"(?:\.\.\.|…|,|;|:|\b(?:and|but|or|the|a|an|of|to|in|on|with|that|which|while|as|when)|[A-Za-z])\s*$"
)
_TERMINAL_PUNCT = re.compile(r"[.!?\"\'\)\]]\s*$")


_PROMPT_INCOMPLETE_TAIL = re.compile(
    r"\b(?:as|with|of|the|a|an|and|but|or|to|in|on|like|that|which|while|when|by|for|from|into|onto)\s*$",
    re.IGNORECASE,
)


def _looks_truncated(text: str) -> bool:
    """Best-effort check: text doesn't end on terminal punctuation or trails on a connector word."""
    text = (text or "").rstrip().rstrip(",:;")
    if not text:
        return True
    if _TERMINAL_PUNCT.search(text):
        return False
    return bool(_PROMPT_INCOMPLETE_TAIL.search(text)) or not text[-1].isalnum() and text[-1] not in "\"')]"


def _check_truncation(scenes: list[dict]) -> list[Issue]:
    """Detect truncation in the last scene's narration AND image prompts.

    The LLM hits its token limit late, so corruption usually shows up in the
    final scene's last image prompt before it shows up in narration.
    """
    if not scenes:
        return []
    issues: list[Issue] = []
    last = scenes[-1]
    last_idx = last.get("index", len(scenes) - 1)

    narration = (last.get("narration") or "").rstrip()
    if not narration:
        issues.append(Issue(
            kind="truncation",
            severity="fatal",
            description="Final scene has empty narration.",
            scene_index=last_idx,
            suggested_fix="Re-run script generation with fewer scenes or shorter target_minutes.",
        ))
    elif _looks_truncated(narration):
        issues.append(Issue(
            kind="truncation",
            severity="fatal",
            description=f"Final scene narration appears truncated. Last 80 chars: {narration[-80:]!r}",
            scene_index=last_idx,
            suggested_fix="Re-run script generation with fewer scenes — the LLM hit the token limit before resolving the story.",
        ))

    # Walk all image prompts; truncation in the last few scenes is the key signal.
    for s in scenes:
        prompts = list(s.get("image_prompts") or [])
        single = (s.get("image_prompt") or "").strip()
        if single and not prompts:
            prompts = [single]
        for prompt_idx, p in enumerate(prompts):
            if _looks_truncated(p):
                issues.append(Issue(
                    kind="truncated_image_prompt",
                    severity="major",
                    description=f"image_prompts[{prompt_idx}] appears truncated. Tail: {p[-80:]!r}",
                    scene_index=s.get("index"),
                    suggested_fix="Regenerate this scene's image prompts; the LLM cut off mid-sentence.",
                ))
    return issues


def _check_image_prompts(scenes: list[dict]) -> list[Issue]:
    """Every scene must have at least one image prompt."""
    issues: list[Issue] = []
    for s in scenes:
        prompts = s.get("image_prompts") or []
        single = (s.get("image_prompt") or "").strip()
        if not prompts and not single:
            issues.append(Issue(
                kind="empty_image_prompts",
                severity="major",
                description="Scene has no image_prompts and no image_prompt.",
                scene_index=s.get("index"),
                suggested_fix="Re-run script generation; this scene's prompts were dropped (likely truncation).",
            ))
    return issues


def _check_scene_count(scenes: list[dict], target_minutes: float) -> list[Issue]:
    """Sanity-check scene count against requested duration. ~30s per scene is the soft floor."""
    if target_minutes <= 0:
        return []
    expected_min = max(2, int(target_minutes * 60 / 90))   # >=1 scene per 90s
    expected_max = max(expected_min + 1, int(target_minutes * 60 / 15))  # <=1 scene per 15s
    if len(scenes) < expected_min:
        return [Issue(
            kind="too_few_scenes",
            severity="minor",
            description=f"{len(scenes)} scenes for a {target_minutes}-minute target — expected at least {expected_min}.",
            suggested_fix="Likely fine, but pacing may be slow.",
        )]
    if len(scenes) > expected_max:
        return [Issue(
            kind="too_many_scenes",
            severity="minor",
            description=f"{len(scenes)} scenes for a {target_minutes}-minute target — expected at most {expected_max}.",
            suggested_fix="Each scene will be very short; consider merging.",
        )]
    return []


_NARRATIVE_SYSTEM = """You are a story editor reviewing a short narrated video script.
Reply with strict JSON only — no prose, no markdown fences.

Schema:
{
  "ends_resolved": true|false,        // does the last scene resolve the central conflict?
  "voice_drift": "ok"|"too_literary"|"too_dry",
  "tone_match": true|false,           // does the script match the requested tone?
  "notes": "<one sentence summary of any structural issue, or empty string>"
}"""


def _build_narrative_user(*, title: str, tone: str, scenes: list[dict]) -> str:
    """Compact the script down to what the editor needs — title, tone, opening, last 3 scenes."""
    summary_scenes: list[str] = []
    for idx, s in enumerate(scenes):
        if idx == 0 or idx >= len(scenes) - 3:
            summary_scenes.append(f"Scene {s.get('index', idx)}: {s.get('narration', '')[:400]}")
    body = "\n\n".join(summary_scenes)
    return (
        f"Title: {title}\n"
        f"Requested tone: {tone or 'unspecified'}\n"
        f"Total scenes: {len(scenes)}\n\n"
        f"Opening + last three scenes:\n\n{body}"
    )


async def _check_narrative(
    *,
    title: str,
    tone: str,
    scenes: list[dict],
    model: Optional[str],
) -> list[Issue]:
    """LLM pass — verdict on resolution, voice, tone."""
    if not scenes:
        return []
    try:
        verdict = await call_llm_json(
            system=_NARRATIVE_SYSTEM,
            user=_build_narrative_user(title=title, tone=tone, scenes=scenes),
            model=model,
            temperature=0.1,
        )
    except LLMCallError as e:
        log.warning(f"Narrative critic LLM call failed; skipping: {e}")
        return []

    issues: list[Issue] = []
    if not verdict.get("ends_resolved", True):
        issues.append(Issue(
            kind="unresolved_ending",
            severity="major",
            description=verdict.get("notes") or "Last scene does not resolve the central conflict.",
            scene_index=scenes[-1].get("index"),
            suggested_fix="Regenerate with explicit instruction to resolve the conflict in the final scene.",
        ))
    drift = verdict.get("voice_drift", "ok")
    if drift in ("too_literary", "too_dry"):
        issues.append(Issue(
            kind="voice_drift",
            severity="minor",
            description=f"Narration voice reads as {drift.replace('_', ' ')} rather than spoken story.",
            suggested_fix="Add second-person address, contractions, and rhetorical questions.",
        ))
    if not verdict.get("tone_match", True):
        issues.append(Issue(
            kind="tone_mismatch",
            severity="minor",
            description=verdict.get("notes") or f"Script does not match requested tone: {tone!r}.",
            suggested_fix=f"Regenerate with stricter adherence to the {tone!r} tone.",
        ))
    return issues


def _build_feedback(issues: list[Issue]) -> str:
    """Compose a single feedback string a writer agent can act on."""
    if not issues:
        return ""
    lines = []
    for i in issues:
        prefix = f"Scene {i.scene_index}: " if i.scene_index is not None else ""
        lines.append(f"- [{i.severity.upper()}] {prefix}{i.description}"
                     + (f" Fix: {i.suggested_fix}" if i.suggested_fix else ""))
    return "Issues to address on the next attempt:\n" + "\n".join(lines)


class ScriptCritic:
    """Verdict-only agent — does not mutate state. Caller decides what to do."""

    name = "script_critic"

    async def critique(
        self,
        *,
        script: dict,
        target_minutes: float = 5.0,
        tone: str = "",
        ollama_model: Optional[str] = None,
        skip_llm: bool = False,
    ) -> AgentVerdict:
        scenes = script.get("scenes") or []
        title = script.get("title") or ""
        effective_tone = tone or script.get("tone") or ""

        issues: list[Issue] = []
        issues.extend(_check_truncation(scenes))
        issues.extend(_check_image_prompts(scenes))
        issues.extend(_check_scene_count(scenes, target_minutes))

        if not skip_llm and scenes:
            issues.extend(await _check_narrative(
                title=title,
                tone=effective_tone,
                scenes=scenes,
                model=ollama_model,
            ))

        severity = worst_severity(issues)
        accept = severity in ("ok", "minor")
        return AgentVerdict(
            agent=self.name,
            accept=accept,
            severity=severity,
            issues=issues,
            feedback=_build_feedback(issues),
            metadata={
                "scene_count": len(scenes),
                "target_minutes": target_minutes,
            },
        )


async def critique_script(
    *,
    script: dict,
    target_minutes: float = 5.0,
    tone: str = "",
    ollama_model: Optional[str] = None,
    skip_llm: bool = False,
) -> AgentVerdict:
    """Convenience function — same as ScriptCritic().critique(...)."""
    return await ScriptCritic().critique(
        script=script,
        target_minutes=target_minutes,
        tone=tone,
        ollama_model=ollama_model,
        skip_llm=skip_llm,
    )
