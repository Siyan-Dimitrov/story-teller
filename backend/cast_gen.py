"""Cast-bible extraction for character consistency.

The screenwriter is asked to emit a ``cast`` array and per-scene ``characters``
tags directly (see ``prompts/screenwriter_system.md``). But older scripts or a
model that simply forgot won't have one. ``ensure_cast`` backfills a cast bible
from an existing script via the LLM so the reference-image consistency path
works regardless of how the script was made.

The derived cast feeds ``image_gen``: one canonical portrait is rendered per
member and passed back as a reference image into every scene that member
appears in.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from . import llm
from .script_gen import _normalize_cast, normalize_scenes

log = logging.getLogger(__name__)


_CAST_SYSTEM = (
    "You are a script analyst. You read a finished narrated screenplay and "
    "extract its CHARACTER BIBLE for an image pipeline that keeps each "
    "character's appearance consistent across scenes. Respond with valid JSON "
    "only — no markdown, no commentary."
)


def _build_cast_user_prompt(script: dict[str, Any]) -> str:
    title = script.get("title", "")
    synopsis = script.get("synopsis", "")
    visual_style = script.get("visual_style", "")
    lines: list[str] = []
    for sc in script.get("scenes", []):
        idx = sc.get("index", 0)
        narration = (sc.get("narration", "") or "").strip()
        prompts = sc.get("image_prompts") or ([sc.get("image_prompt")] if sc.get("image_prompt") else [])
        prompt_blob = " | ".join(p for p in prompts if p)
        lines.append(f"[scene {idx}] {narration}\n  images: {prompt_blob}")
    scenes_blob = "\n".join(lines)
    return (
        f"Title: {title}\n"
        f"Synopsis: {synopsis}\n"
        f"Visual style: {visual_style}\n\n"
        "Scenes (narration + image prompts):\n"
        f"{scenes_blob}\n\n"
        "Identify every recurring character (anyone appearing in more than one "
        "scene, plus any single-scene character who is the visual focus of that "
        "scene). For each, write an IMMUTABLE appearance description (apparent "
        "age — say 'adult'/'elderly'/'young child' explicitly — build, hair, "
        "face, clothing, palette, one signature detail; appearance only, no "
        "plot) and a reference_prompt that renders them ALONE on a plain "
        "neutral background, neutral pose, even lighting, no other people — "
        "exactly one figure shown once, a single view, never a character "
        "sheet or multiple poses.\n\n"
        "Then tag each scene with the ids of the characters that visibly appear "
        "in it. Keep the cast small (1-6). Return EXACTLY this JSON:\n"
        "{\n"
        '  "cast": [\n'
        '    {"id": "slug", "name": "Name", "role": "protagonist|antagonist|supporting|minor",\n'
        '     "description": "appearance only", "reference_prompt": "solo portrait prompt"}\n'
        "  ],\n"
        '  "scene_characters": {"0": ["slug"], "1": [], "2": ["slug", "other"]}\n'
        "}"
    )


async def generate_cast(script: dict[str, Any], **_ignored) -> dict[str, Any]:
    """Derive a cast bible from a finished script. Returns the updated script
    (with ``cast`` populated and each scene's ``characters`` tagged).

    Falls back to leaving the script unchanged (empty cast) if the LLM call or
    parse fails — consistency simply degrades to no-reference generation rather
    than breaking the pipeline.
    """
    try:
        raw = await llm.complete(
            _CAST_SYSTEM,
            _build_cast_user_prompt(script),
            pass_name="cast bible",
        )
        data = llm.parse_json(raw)
    except Exception as e:  # noqa: BLE001
        log.warning("Cast extraction failed (%s) — leaving cast empty", e)
        script.setdefault("cast", [])
        return script

    script["cast"] = _normalize_cast(data.get("cast"))
    valid_ids = {c["id"] for c in script["cast"]}

    # Apply scene_characters mapping (keys may be str or int).
    mapping = data.get("scene_characters") or {}
    if isinstance(mapping, dict):
        norm_map: dict[int, list[str]] = {}
        for k, v in mapping.items():
            try:
                ki = int(k)
            except (TypeError, ValueError):
                continue
            if isinstance(v, list):
                norm_map[ki] = [c for c in v if isinstance(c, str) and c in valid_ids]
        for sc in script.get("scenes", []):
            idx = sc.get("index", 0)
            if idx in norm_map:
                sc["characters"] = norm_map[idx]

    # Re-run normalization so scene["characters"] is filtered to valid ids and
    # any structural defaults are filled.
    return normalize_scenes(script)


async def ensure_cast(
    script: dict[str, Any],
    overwrite: bool = False,
    **_ignored,
) -> dict[str, Any]:
    """Ensure ``script['cast']`` is populated, deriving it if needed."""
    existing = script.get("cast")
    if existing and not overwrite:
        return script
    log.info("Backfilling cast bible from script via LLM")
    return await generate_cast(script)
