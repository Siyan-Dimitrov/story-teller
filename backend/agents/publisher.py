"""Publisher agent — prepares YouTube metadata; does NOT upload.

Replaces the single-title sludge from export.generate_youtube_metadata with:
  - 3 title candidates in distinct molds (question / IP-piggyback / mystery)
  - hook-first description (first 125 chars must hook)
  - ~12 tags
  - 3-5 word thumbnail caption
  - chapter marks derived from scene audio_duration

Output goes to projects/<id>/youtube_metadata.json. The user reviews and
uploads manually — no API calls beyond the LLM.
"""

from __future__ import annotations

import logging
from typing import Optional

from .. import project_store as store
from .base import call_llm_json, LLMCallError

log = logging.getLogger(__name__)


_SYSTEM_PROMPT = """You are a YouTube metadata strategist for narrated dark-fairy-tale videos. The channel adapts public-domain stories (Brothers Grimm, Project Gutenberg) into atmospheric narrated videos with AI-generated images.

Reply with strict JSON only. No prose, no markdown fences.

Schema:
{
  "title_candidates": [
    {"mold": "question",     "title": "..."},
    {"mold": "ip_piggyback", "title": "..."},
    {"mold": "mystery",      "title": "..."}
  ],
  "selected_index": 0,
  "selected_rationale": "<one sentence on why this title wins for CTR>",
  "description": "<800-1200 chars; the FIRST 125 characters MUST hook the viewer with a concrete fact, question, or open loop>",
  "tags": ["<lowercase, comma-free, 10-14 tags blending specific story themes and broad channel terms>"],
  "thumbnail_caption": "<3-5 word punchy phrase for thumbnail overlay>",
  "chapter_marks": [
    {"time_seconds": 0, "title": "Cold open"},
    ...
  ]
}

Rules:
- "question" mold: the title IS a question (Did/Why/What/How…).
- "ip_piggyback" mold: references a known IP the source resembles ("the original Sleeping Beauty", "before Disney").
- "mystery" mold: states a specific intriguing fact and stops, leaving the resolution offscreen.
- BAN the pattern "<Story Name>: A Dark Fairy Tale of X and Y" — every output uses it; it dies on thumbnails.
- Description first sentence must reference a concrete detail from the story, not a generic mood.
- Chapter marks: 4-8 marks. The first must be at time_seconds 0. Marks should align with major narrative beats."""


def _build_user_prompt(
    *,
    title: str,
    tone: str,
    book_title: str,
    synopsis: str,
    scenes: list[dict],
) -> str:
    lines = [
        f"Title: {title}",
        f"Tone: {tone or 'unspecified'}",
        f"Source book: {book_title or 'unknown'}",
        f"Synopsis: {synopsis or '(no synopsis)'}",
        f"Total scenes: {len(scenes)}",
        "",
        "Scenes (index | mood | duration_s | narration excerpt):",
    ]
    cumulative = 0.0
    for s in scenes:
        dur = float(s.get("audio_duration") or s.get("duration_hint") or 10.0)
        narr = (s.get("narration") or "").strip().replace("\n", " ")[:140]
        lines.append(
            f"  {s.get('index', '?'):>2} | t={cumulative:>6.1f}s "
            f"| mood={s.get('mood', '?'):<12} | {narr!r}"
        )
        cumulative += dur
    lines.append("")
    lines.append(f"Total duration ≈ {cumulative:.0f} seconds.")
    return "\n".join(lines)


_DEFAULT_TAGS = [
    "dark fairy tale",
    "narrated story",
    "gothic fairy tale",
    "audiobook",
    "fairy tales for adults",
    "bedtime stories",
    "folklore",
    "story narration",
]


def _validate_and_clamp(payload: dict, scenes: list[dict]) -> dict:
    """Defensive normalisation against LLM JSON drift."""
    out: dict = {}

    candidates = payload.get("title_candidates") or []
    fixed: list[dict] = []
    for mold in ("question", "ip_piggyback", "mystery"):
        match = next((c for c in candidates if (c or {}).get("mold") == mold), None)
        if match and isinstance(match.get("title"), str):
            fixed.append({"mold": mold, "title": match["title"].strip()[:100]})
        else:
            fixed.append({"mold": mold, "title": ""})
    out["title_candidates"] = fixed

    sel = payload.get("selected_index", 0)
    try:
        sel = int(sel)
    except (TypeError, ValueError):
        sel = 0
    if not 0 <= sel < 3 or not fixed[sel]["title"]:
        sel = next((i for i, c in enumerate(fixed) if c["title"]), 0)
    out["selected_index"] = sel
    out["selected_title"] = fixed[sel]["title"] if fixed[sel]["title"] else None
    out["selected_rationale"] = (payload.get("selected_rationale") or "").strip()[:300]

    description = (payload.get("description") or "").strip()
    out["description"] = description[:1200]
    out["description_hook"] = description[:125]

    tags = payload.get("tags") or []
    cleaned_tags: list[str] = []
    seen = set()
    for t in tags:
        if not isinstance(t, str):
            continue
        t = t.strip().lower().strip(",.;:")
        if not t or t in seen:
            continue
        seen.add(t)
        cleaned_tags.append(t[:40])
        if len(cleaned_tags) >= 14:
            break
    if len(cleaned_tags) < 8:
        for t in _DEFAULT_TAGS:
            if t in seen:
                continue
            cleaned_tags.append(t)
            seen.add(t)
            if len(cleaned_tags) >= 8:
                break
    out["tags"] = cleaned_tags

    thumb = (payload.get("thumbnail_caption") or "").strip()
    words = thumb.split()
    if len(words) > 5:
        thumb = " ".join(words[:5])
    out["thumbnail_caption"] = thumb[:60]

    total_s = sum(
        float(s.get("audio_duration") or s.get("duration_hint") or 10.0) for s in scenes
    )
    raw_marks = payload.get("chapter_marks") or []
    marks: list[dict] = []
    for m in raw_marks:
        if not isinstance(m, dict):
            continue
        try:
            t = int(float(m.get("time_seconds", -1)))
        except (TypeError, ValueError):
            continue
        title_str = (m.get("title") or "").strip()
        if not title_str or t < 0 or t > total_s + 1:
            continue
        marks.append({"time_seconds": t, "title": title_str[:60]})
    marks.sort(key=lambda m: m["time_seconds"])
    if not marks or marks[0]["time_seconds"] != 0:
        marks.insert(0, {"time_seconds": 0, "title": "Cold open"})
    seen_t = set()
    deduped: list[dict] = []
    for m in marks:
        if m["time_seconds"] in seen_t:
            continue
        seen_t.add(m["time_seconds"])
        deduped.append(m)
    out["chapter_marks"] = deduped[:8]

    return out


class Publisher:
    """Generates YouTube metadata. publish() is idempotent — safe to rerun."""

    name = "publisher"

    async def publish(
        self,
        project_id: str,
        *,
        ollama_model: Optional[str] = None,
    ) -> dict:
        state = store.load_state(project_id)
        script = store.load_json(project_id, "script.json")
        if not script:
            raise RuntimeError(f"Project {project_id} has no script.json — run script step first.")

        scenes = script.get("scenes") or []
        prompt = _build_user_prompt(
            title=script.get("title") or state.get("title", ""),
            tone=state.get("tone", ""),
            book_title=state.get("book_title", ""),
            synopsis=script.get("synopsis", ""),
            scenes=scenes,
        )

        try:
            raw_payload = await call_llm_json(
                system=_SYSTEM_PROMPT,
                user=prompt,
                model=ollama_model or state.get("ollama_model"),
                temperature=0.4,
                timeout=180.0,
            )
        except LLMCallError as e:
            log.error(f"Publisher LLM call failed for {project_id}: {e}")
            raise

        payload = _validate_and_clamp(raw_payload, scenes)
        store.save_json(project_id, "youtube_metadata.json", payload)
        log.info(
            f"Publisher wrote youtube_metadata.json for {project_id}: "
            f"title={payload.get('selected_title')!r}, tags={len(payload['tags'])}, marks={len(payload['chapter_marks'])}"
        )
        return payload


async def publish_metadata(project_id: str, *, ollama_model: Optional[str] = None) -> dict:
    return await Publisher().publish(project_id, ollama_model=ollama_model)
