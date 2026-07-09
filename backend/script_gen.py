"""Script/scene shared utilities and story search.

The actual screenplay generation lives in :mod:`backend.claude_script_gen`
(writer → critic → reviser on Claude). This module keeps the shared scene
normalization helpers and the Claude-backed story search.
"""

import logging
import re

from . import config, llm

log = logging.getLogger(__name__)

# Upper bound on images per scene, enforced in code because the prompt-side
# rule ("one per ~20 words, max 10") is only advisory to the LLM.
MAX_IMAGE_PROMPTS_PER_SCENE = 10


def normalize_scenes(script: dict) -> dict:
    """Backfill indices, default mood/duration, and reconcile image_prompt(s).

    Shared by all script backends so downstream code (voice, images,
    assembly) sees a uniform shape regardless of the LLM that generated the
    script.
    """
    # Per-story art direction (the "feel" derived from the story itself). Kept
    # at the top level so image generation can use it as the style for every
    # scene. Trim, and drop it entirely if the model left it blank so callers
    # cleanly fall back to the user's selected style.
    vs = script.get("visual_style")
    if isinstance(vs, str) and vs.strip():
        script["visual_style"] = vs.strip()
    else:
        script.pop("visual_style", None)

    # Normalize the cast bible (character consistency). Tolerate models that
    # omit it entirely — downstream code only uses it when the user enables
    # character consistency, and cast_gen can backfill it on demand.
    script["cast"] = _normalize_cast(script.get("cast"))
    valid_ids = {c["id"] for c in script["cast"]}

    for i, scene in enumerate(script.get("scenes", [])):
        scene["index"] = i
        scene.setdefault("mood", "neutral")
        scene.setdefault("duration_hint", 15.0)
        if "image_prompts" not in scene or not scene["image_prompts"]:
            single = scene.get("image_prompt", "")
            scene["image_prompts"] = [single] if single else []
        # Hard cap regardless of what the LLM emitted — every prompt is a paid
        # image, and slots shorter than ~5s read as a strobing slideshow.
        if len(scene["image_prompts"]) > MAX_IMAGE_PROMPTS_PER_SCENE:
            scene["image_prompts"] = scene["image_prompts"][:MAX_IMAGE_PROMPTS_PER_SCENE]
        if scene["image_prompts"]:
            scene["image_prompt"] = scene["image_prompts"][0]
        # Keep only character ids that exist in the cast; drop hallucinated tags.
        chars = scene.get("characters") or []
        if isinstance(chars, list) and valid_ids:
            scene["characters"] = [c for c in chars if isinstance(c, str) and c in valid_ids]
        else:
            scene["characters"] = []
    return script


def _slugify(value: str) -> str:
    """Lowercase, hyphenated, alnum-only slug for a cast id."""
    slug = re.sub(r"[^a-z0-9]+", "-", (value or "").lower()).strip("-")
    return slug or "character"


def _normalize_cast(cast) -> list[dict]:
    """Coerce a model-supplied ``cast`` into a clean list of unique members.

    Drops entries without a usable name/description, fills a stable ``id``
    slug, and de-duplicates ids so scene tags resolve unambiguously.
    """
    if not isinstance(cast, list):
        return []
    out: list[dict] = []
    seen: set[str] = set()
    for entry in cast:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip()
        desc = str(entry.get("description", "")).strip()
        if not name and not desc:
            continue
        base_id = _slugify(str(entry.get("id", "")).strip() or name or desc[:20])
        cid = base_id
        n = 2
        while cid in seen:
            cid = f"{base_id}-{n}"
            n += 1
        seen.add(cid)
        out.append({
            "id": cid,
            "name": name or cid.replace("-", " ").title(),
            "role": str(entry.get("role", "")).strip(),
            "description": desc,
            "reference_prompt": str(entry.get("reference_prompt", "")).strip(),
            "reference_image_path": entry.get("reference_image_path"),
        })
    return out


SEARCH_PROMPT = """You are a literary expert. Given a search query, suggest well-known short stories, fairy tales, fables, and folk tales that match.

Respond ONLY with valid JSON (no markdown fences). Use this exact structure:
{
  "results": [
    {
      "title": "The story title",
      "author": "Author name or 'Traditional'",
      "origin": "e.g. German folklore, French fairy tale, Greek mythology",
      "synopsis": "A 3-5 sentence synopsis of the full story",
      "themes": ["theme1", "theme2"],
      "tone_suggestion": "dark"
    }
  ]
}

Guidelines:
- Return well-known, public domain stories that people would recognize
- Include a mix: fairy tales, fables, myths, classic short stories
- Synopsis should be detailed enough to adapt into a video script
- tone_suggestion should be the most natural adaptation tone: dark, humorous, gothic, whimsical, romantic, or tragic
- Prioritize stories with strong visual potential and dramatic arcs
"""


async def search_stories(
    query: str = "",
    count: int = 6,
    **_ignored,  # absorb legacy args (e.g. ollama_model) from old callers
) -> list[dict]:
    """Use Claude to suggest well-known stories matching a query."""
    user_prompt = f"Suggest {count} well-known short stories, fairy tales, or folk tales"
    if query:
        user_prompt += f" matching this theme or query: {query}"
    user_prompt += ".\nReturn diverse results from different cultures and time periods."

    log.info(f"Searching stories: query={query!r}")

    raw = await llm.complete(
        SEARCH_PROMPT,
        user_prompt,
        model=config.CLAUDE_FAST_MODEL,
        pass_name="story-search",
        timeout=config.CLAUDE_FAST_TIMEOUT_SECONDS,
    )
    parsed = llm.parse_json(raw)
    return parsed.get("results", [])
