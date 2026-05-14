"""ImagePromptCritic — reviews and rewrites image prompts before they hit the
image backend.

The script critic only catches truncation and empties on image prompts. It does
not look at quality: a prompt can be syntactically fine but vague ("a dark
forest, mysterious") and still produce flat, generic art.

This agent reads each scene's narration alongside its image_prompts and asks
Kimi whether each prompt has the visual specificity needed to render a strong
frame. Weak prompts get rewritten in place; strong ones are left alone.

Runs one LLM call per scene in parallel (asyncio.gather). Failures are
non-fatal — a scene whose critic call errored simply keeps its original prompts.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

from .base import LLMCallError, call_llm_json

log = logging.getLogger(__name__)


_SYSTEM_PROMPT = """You are an art director reviewing image prompts for a dark fairy-tale video.

For each prompt, decide whether it is specific enough to produce a strong, distinctive frame, or whether it is generic and needs a rewrite.

A STRONG prompt names:
- The subject and their pose / action (not just "a figure")
- Setting specifics (architecture, vegetation, weather, time of day)
- Lighting direction and quality (e.g. "low side-light from a single candle", "moonlight through bare branches")
- Composition cues (framing, foreground/background, scale)

A WEAK prompt is vague ("a dark forest, mysterious"), generic ("gothic atmosphere"), or just restates the narration without visual detail.

When rewriting:
- Preserve every named character and their canonical traits (use their names exactly as given).
- Preserve the scene's key story beat — do not invent new events.
- Do not append art-style boilerplate ("dark fairy tale illustration", "gothic storybook art", "atmospheric", etc). The image backend adds style separately.
- Keep symbolic / non-graphic framing for disturbing beats (the script critic already enforced this — don't undo it).
- Keep length similar to the original — concise, comma-separated, no prose.

Reply with strict JSON only — no prose, no markdown fences.

Schema:
{
  "prompts": [
    {
      "index": 0,                          // matches the input prompt index
      "decision": "keep" | "rewrite",
      "prompt": "<rewritten prompt if decision=rewrite, else omit>",
      "why": "<one short phrase explaining the decision>"
    },
    ...
  ]
}"""


def _build_user_prompt(*, narration: str, prompts: list[str], scene_index: int, tone: str) -> str:
    """Compose the per-scene user message. Narration is truncated to keep the call cheap."""
    narration_excerpt = (narration or "").strip()
    if len(narration_excerpt) > 600:
        narration_excerpt = narration_excerpt[:600].rsplit(" ", 1)[0] + "…"
    prompts_block = "\n".join(f"[{i}] {p}" for i, p in enumerate(prompts))
    return (
        f"Scene index: {scene_index}\n"
        f"Tone: {tone or 'unspecified'}\n\n"
        f"Narration:\n{narration_excerpt}\n\n"
        f"Image prompts to review:\n{prompts_block}"
    )


@dataclass
class PromptRewrite:
    scene_index: int
    prompt_index: int
    original: str
    rewritten: str
    why: str


@dataclass
class ImagePromptCritiqueResult:
    rewrites: list[PromptRewrite] = field(default_factory=list)
    scenes_reviewed: int = 0
    scenes_failed: int = 0

    @property
    def rewrite_count(self) -> int:
        return len(self.rewrites)


async def _critique_one_scene(
    *,
    scene: dict,
    scene_index: int,
    tone: str,
    model: Optional[str],
) -> list[PromptRewrite]:
    prompts = list(scene.get("image_prompts") or [])
    if not prompts:
        single = (scene.get("image_prompt") or "").strip()
        if single:
            prompts = [single]
    if not prompts:
        return []

    narration = scene.get("narration") or ""
    try:
        verdict = await call_llm_json(
            system=_SYSTEM_PROMPT,
            user=_build_user_prompt(
                narration=narration,
                prompts=prompts,
                scene_index=scene_index,
                tone=tone,
            ),
            model=model,
            temperature=0.2,
        )
    except LLMCallError as e:
        log.warning(f"ImagePromptCritic scene {scene_index} LLM error: {e}")
        raise

    rewrites: list[PromptRewrite] = []
    for entry in verdict.get("prompts") or []:
        try:
            idx = int(entry.get("index"))
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= len(prompts):
            continue
        if entry.get("decision") != "rewrite":
            continue
        new_prompt = (entry.get("prompt") or "").strip()
        if not new_prompt:
            continue
        why = (entry.get("why") or "").strip()
        rewrites.append(PromptRewrite(
            scene_index=scene_index,
            prompt_index=idx,
            original=prompts[idx],
            rewritten=new_prompt,
            why=why,
        ))
    return rewrites


def _apply_rewrites(script: dict, rewrites: list[PromptRewrite]) -> None:
    """Apply rewrites in place. Keeps image_prompt[0] aligned with image_prompts[0]."""
    by_scene: dict[int, list[PromptRewrite]] = {}
    for r in rewrites:
        by_scene.setdefault(r.scene_index, []).append(r)

    for scene in script.get("scenes") or []:
        idx = scene.get("index")
        scene_rewrites = by_scene.get(idx) or []
        if not scene_rewrites:
            continue
        prompts = list(scene.get("image_prompts") or [])
        if not prompts:
            single = (scene.get("image_prompt") or "").strip()
            if single:
                prompts = [single]
        if not prompts:
            continue
        for r in scene_rewrites:
            if 0 <= r.prompt_index < len(prompts):
                prompts[r.prompt_index] = r.rewritten
        scene["image_prompts"] = prompts
        scene["image_prompt"] = prompts[0]


async def critique_image_prompts(
    *,
    script: dict,
    tone: str = "",
    ollama_model: Optional[str] = None,
) -> ImagePromptCritiqueResult:
    """Review every scene's image_prompts. Returns a result describing any rewrites.

    The script is mutated in place — callers can persist it via store.save_json.
    """
    scenes = script.get("scenes") or []
    if not scenes:
        return ImagePromptCritiqueResult()

    effective_tone = tone or script.get("tone") or ""

    tasks = [
        _critique_one_scene(
            scene=s,
            scene_index=s.get("index", i),
            tone=effective_tone,
            model=ollama_model,
        )
        for i, s in enumerate(scenes)
    ]
    settled = await asyncio.gather(*tasks, return_exceptions=True)

    result = ImagePromptCritiqueResult(scenes_reviewed=len(scenes))
    for outcome in settled:
        if isinstance(outcome, Exception):
            result.scenes_failed += 1
            continue
        result.rewrites.extend(outcome)

    if result.rewrites:
        _apply_rewrites(script, result.rewrites)
        for r in result.rewrites:
            log.info(
                f"ImagePromptCritic rewrote scene {r.scene_index}/prompt {r.prompt_index} "
                f"({r.why!r}): {r.original[:80]!r} → {r.rewritten[:80]!r}"
            )
    log.info(
        f"ImagePromptCritic: reviewed {result.scenes_reviewed} scenes, "
        f"rewrote {result.rewrite_count} prompts, {result.scenes_failed} scenes errored"
    )
    return result
