"""Shorts director — picks the best self-contained scenes to cut as vertical
shorts and writes a scroll-stopping hook line for each.

Primary path: an LLM scores every scene for "standalone short potential"
(dramatic, self-contained, hooky) and returns the top N with a hook line.
Fallback: a heuristic on mood intensity + how well the scene's narration
length fits the 8-58s short window.
"""

from __future__ import annotations

import json
import logging

import httpx

from . import config

log = logging.getLogger(__name__)


# Rough "drama" weighting for the screenwriter's mood vocabulary — higher means
# more likely to grab a scrolling viewer.
_MOOD_WEIGHT = {
    "horrifying": 5, "tense": 5, "ominous": 4, "triumphant": 4,
    "dark": 3, "melancholy": 3, "whimsical": 2, "peaceful": 1, "neutral": 1,
}

_SYSTEM = (
    "You are a short-form video editor. From a narrated story's scenes you pick "
    "the few that work best as standalone vertical shorts — moments that are "
    "self-contained, visually dramatic, and hook a scrolling viewer in the "
    "first second. Respond with valid JSON only."
)


def _fits_short(audio_duration: float | None) -> float:
    """0..1 score for how well a scene's spoken length fits a short."""
    if not audio_duration:
        return 0.4  # unknown — mild prior
    lo = config.SHORT_MIN_DURATION
    hi = config.SHORT_MAX_DURATION - config.SHORT_TAIL_DURATION
    if lo <= audio_duration <= hi:
        return 1.0
    if audio_duration < lo:
        return max(0.0, audio_duration / lo)
    return max(0.2, hi / audio_duration)  # too long → trimmable but penalised


def _heuristic(scenes: list[dict], count: int) -> list[dict]:
    scored = []
    for sc in scenes:
        mood = (sc.get("mood") or "neutral").lower()
        score = _MOOD_WEIGHT.get(mood, 1) + 4.0 * _fits_short(sc.get("audio_duration"))
        scored.append((score, sc))
    scored.sort(key=lambda x: (-x[0], x[1].get("index", 0)))
    out = []
    for score, sc in scored[:count]:
        narration = (sc.get("narration") or "").strip()
        hook = narration.split(".")[0][:80].strip() if narration else ""
        out.append({
            "scene_index": sc.get("index", 0),
            "hook": hook,
            "reason": f"mood={sc.get('mood')}, fits short window",
            "score": round(float(score), 2),
        })
    return out


def _build_user_prompt(script: dict, count: int) -> str:
    lines = []
    for sc in script.get("scenes", []):
        dur = sc.get("audio_duration")
        dur_s = f"{dur:.0f}s" if dur else "?"
        lines.append(
            f"[{sc.get('index')}] mood={sc.get('mood')} dur={dur_s}: "
            f"{(sc.get('narration') or '').strip()}"
        )
    blob = "\n".join(lines)
    return (
        f"Story: {script.get('title', '')}\n"
        f"Synopsis: {script.get('synopsis', '')}\n\n"
        f"Scenes:\n{blob}\n\n"
        f"Pick the {count} scenes that work best as standalone vertical shorts. "
        "Prefer scenes that are self-contained (don't require prior context), "
        "emotionally charged, and visually striking. For each, write a HOOK: a "
        "punchy curiosity-gap line of at most 12 words for the top of the short "
        "(no spoilers, no hashtags). Return EXACTLY this JSON:\n"
        '{"shorts": [{"scene_index": 0, "hook": "...", "reason": "...", "score": 1-10}]}'
    )


def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(ln for ln in text.split("\n") if not ln.strip().startswith("```")).strip()
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start >= 0 and end > start else text


async def suggest_shorts(
    script: dict,
    ollama_model: str | None = None,
    count: int | None = None,
) -> list[dict]:
    """Return a ranked list of {scene_index, hook, reason, score}."""
    scenes = script.get("scenes") or []
    if not scenes:
        return []
    n = count or config.SHORTS_PER_PROJECT
    n = max(1, min(n, len(scenes)))

    model = ollama_model or config.OLLAMA_MODEL
    try:
        async with httpx.AsyncClient(timeout=config.LLM_TIMEOUT_SECONDS) as client:
            resp = await client.post(
                f"{config.OLLAMA_URL}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": _SYSTEM},
                        {"role": "user", "content": _build_user_prompt(script, n)},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.5, "num_predict": 2000},
                },
            )
        resp.raise_for_status()
        from .script_gen import _extract_llm_content
        data = json.loads(_extract_json(_extract_llm_content(resp.json())))
        picks = data.get("shorts") or []
        valid_idx = {sc.get("index") for sc in scenes}
        out = []
        seen = set()
        for p in picks:
            idx = p.get("scene_index")
            if idx in valid_idx and idx not in seen:
                seen.add(idx)
                out.append({
                    "scene_index": idx,
                    "hook": (p.get("hook") or "").strip(),
                    "reason": (p.get("reason") or "").strip(),
                    "score": p.get("score", 0),
                })
        if out:
            return out[:n]
        log.warning("Shorts director returned no valid picks — using heuristic")
    except Exception as e:  # noqa: BLE001
        log.warning("Shorts director LLM failed (%s) — using heuristic", e)

    return _heuristic(scenes, n)
