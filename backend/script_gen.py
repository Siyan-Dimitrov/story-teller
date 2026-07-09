"""Script generation via Ollama LLM."""

import json
import logging
import re
import httpx

from . import config
from .grimm_tales import get_tale

log = logging.getLogger(__name__)

# Upper bound on images per scene, enforced in code because the prompt-side
# rule ("one per ~20 words, max 10") is only advisory to the LLM.
MAX_IMAGE_PROMPTS_PER_SCENE = 10


def normalize_scenes(script: dict) -> dict:
    """Backfill indices, default mood/duration, and reconcile image_prompt(s).

    Shared by both the Ollama and Claude script backends so downstream code
    (voice, images, assembly) sees a uniform shape regardless of the LLM
    that generated the script.
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


def _repair_truncated_json(raw: str) -> str:
    """Attempt to repair JSON truncated by token limits.

    Closes any open strings, arrays, and objects so json.loads can parse
    whatever complete scenes the LLM managed to emit.
    """
    # Remove any trailing partial escape sequence
    raw = re.sub(r'\\$', '', raw.rstrip())

    in_string = False
    escape = False
    stack: list[str] = []  # tracks open { and [

    for ch in raw:
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ('{', '['):
            stack.append(ch)
        elif ch == '}' and stack and stack[-1] == '{':
            stack.pop()
        elif ch == ']' and stack and stack[-1] == '[':
            stack.pop()

    # Close the open string if we're inside one
    if in_string:
        raw += '"'

    # Close open containers in reverse order
    for opener in reversed(stack):
        raw += ']' if opener == '[' else '}'

    return raw


def _extract_llm_content(data: dict) -> str:
    """Extract text content from Ollama response, handling thinking models.

    Some models (e.g. kimi-k2.5) put their response in the 'thinking' field
    with an empty 'content' field. This checks both.
    """
    msg = data.get("message", {})
    content = (msg.get("content") or "").strip()
    if content:
        return content
    # Thinking models: extract JSON from the thinking field
    thinking = (msg.get("thinking") or "").strip()
    if thinking:
        # Try to find JSON object in the thinking text
        start = thinking.find("{")
        end = thinking.rfind("}") + 1
        if start >= 0 and end > start:
            return thinking[start:end]
    return content


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
    ollama_model: str | None = None,
    ollama_base_url: str | None = None,
) -> list[dict]:
    """Use the LLM to suggest well-known stories matching a query."""
    model = ollama_model or config.OLLAMA_MODEL
    base_url = ollama_base_url or config.OLLAMA_URL

    user_prompt = f"Suggest {count} well-known short stories, fairy tales, or folk tales"
    if query:
        user_prompt += f" matching this theme or query: {query}"
    user_prompt += ".\nReturn diverse results from different cultures and time periods."

    log.info(f"Searching stories: query={query!r}, model={model}")

    async with httpx.AsyncClient(timeout=config.LLM_TIMEOUT_SECONDS) as client:
        resp = await client.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SEARCH_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.7, "num_predict": 8000},
            },
        )
        if resp.status_code != 200:
            body = resp.text
            log.error(f"Ollama error {resp.status_code}: {body}")
            raise RuntimeError(f"Ollama returned {resp.status_code}: {body}")

    data = resp.json()
    content = _extract_llm_content(data)

    # Strip markdown fences if present
    if content.startswith("```"):
        lines = content.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines)

    parsed = json.loads(content)
    return parsed.get("results", [])

SYSTEM_PROMPT = """You are a master storyteller who writes dark fairy tales for adults.
Your stories are atmospheric, gothic, and gripping — like a campfire tale that keeps listeners riveted.
You write in a conversational narrator voice: vivid, dramatic, with dark humor.

When given a fairy tale source, you retell it faithfully to the dark original but in your own compelling voice.
You break the story into SCENES, each a self-contained visual moment.

Respond ONLY with valid JSON (no markdown fences). Use this exact structure:
{
  "title": "The story title",
  "synopsis": "A 2-3 sentence synopsis",
  "scenes": [
    {
      "narration": "The narrator's text for this scene (2-4 paragraphs, spoken aloud)",
      "image_prompts": [
        "First image prompt — must depict a specific moment from the narration",
        "Second image prompt — must depict a different specific moment from the narration",
        "... one prompt per ~20 words of narration (minimum 3, maximum 10 per scene)"
      ],
      "mood": "one word mood: dark, tense, whimsical, melancholy, horrifying, peaceful, ominous, triumphant",
      "duration_hint": 15.0
    }
  ]
}

Guidelines:
- Each scene's narration should be 60-120 words for short videos (3-5 min total), 100-200 words for longer ones
- duration_hint is approximate seconds — will be overridden by actual voice audio length
- Each scene needs one image_prompt per roughly 20 words of narration — minimum 3, maximum 10 (a 100-word scene gets 5, a 200-word scene gets 10). Each image is on screen ~7 seconds, so too few prompts makes the video feel static.
- CRITICAL — image prompts must be grounded in the narration text:
  1. Read the narration you wrote for the scene
  2. Identify its most visually striking moments or images — one per prompt, in story order (beginning → middle → end of the scene's beat)
  3. Write each prompt as a literal depiction of that moment: name the specific characters, objects, setting, and action happening
  4. Do NOT write generic prompts like "a dark forest" — instead write "the woodcutter's adult daughter kneeling beside a broken juniper branch, a crimson ribbon in her hands, moonlight through bare trees"
- Each prompt must include: WHO (specific character/creature), WHAT (specific action), WHERE (specific setting detail), and WHEN/LIGHTING if relevant
- Include continuity anchors for recurring characters: apparent age, clothing, hair, posture, and one memorable identifying feature
- Include explicit camera/framing language in every prompt, such as "wide establishing shot", "low-angle medium shot", "over-the-shoulder shot", "close-up", or "silhouette against the doorway"
- Vary composition across a scene's prompts — mix distinct framings (e.g. wide establishing shot, medium action shot, close-up emotional/detail shot, alternate angle such as over-the-shoulder or low-angle). Never use the same framing twice in a row.
- Do NOT append generic art-style boilerplate to image_prompts. The selected image backend adds style separately via style_prompt.
- Do NOT request captions, title cards, typography, subtitles, logos, or text inside the image unless the story explicitly requires a visible sign or written object
- For image_prompts, adapt disturbing story beats as symbolic, non-graphic folklore imagery. Avoid gore, visible injury, sexual content, intimate contact, restraint, torture, explicit burning, cannibalism, and children in danger.
- When a recurring character is grown, say "adult" in the image prompt so image backends do not infer a minor.
- Keep each image_prompt concise: one vivid sentence, 35-70 words, concrete nouns and actions only
- Aim for the number of scenes that fits the target length (roughly 1 scene per 30-60 seconds)
- The narration should be vivid and engaging when read aloud — this is a voiceover script
- Never break the fourth wall or reference that this is a video/script
"""


async def generate_script(
    source_tale: str = "",
    custom_prompt: str = "",
    target_minutes: float = 5.0,
    ollama_model: str | None = None,
    ollama_base_url: str | None = None,
    tone: str = "",
) -> dict:
    model = ollama_model or config.OLLAMA_MODEL
    base_url = ollama_base_url or config.OLLAMA_URL

    # Build the user prompt
    parts = []
    if source_tale:
        tale = get_tale(source_tale)
        if tale:
            parts.append(f"Retell this dark fairy tale in your narrator voice:\n\n")
            parts.append(f"Title: {tale['title']}\n")
            parts.append(f"Origin: {tale['origin']}\n")
            parts.append(f"Synopsis:\n{tale['synopsis']}\n")
        else:
            parts.append(f"Write a dark fairy tale based on: {source_tale}\n")

    if tone:
        parts.append(f"\nAdaptation tone: {tone}. Infuse the story with this tone throughout.\n")

    if custom_prompt:
        # Warn if text is very long (might exceed LLM context)
        if len(custom_prompt) > 8000:
            log.warning(f"Custom prompt is {len(custom_prompt):,} chars - may exceed LLM context. Consider shorter chapters.")
        # Truncate extremely long text to prevent API errors
        max_prompt_len = 12000
        if len(custom_prompt) > max_prompt_len:
            parts.append(f"\nSource material (truncated from {len(custom_prompt):,} chars):\n{custom_prompt[:max_prompt_len]}...\n")
        else:
            parts.append(f"\nAdditional direction: {custom_prompt}\n")

    scene_count = max(5, int(target_minutes * 1.5))
    parts.append(f"\nTarget length: approximately {target_minutes} minutes when narrated aloud.")
    parts.append(f"\nAim for roughly {scene_count} scenes.")

    user_prompt = "\n".join(parts)

    log.info(f"Generating script with model={model}, target={target_minutes}min")

    async with httpx.AsyncClient(timeout=config.LLM_TIMEOUT_SECONDS) as client:
        resp = await client.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": config.LLM_TEMPERATURE,
                    "num_predict": config.LLM_MAX_TOKENS,
                },
            },
        )
        if resp.status_code != 200:
            body = resp.text
            log.error(f"Ollama error {resp.status_code}: {body}")
            raise RuntimeError(f"Ollama returned {resp.status_code}: {body}")

    data = resp.json()
    content = _extract_llm_content(data)

    # Check if response was truncated by token limit
    done_reason = data.get("done_reason", "")
    if done_reason == "length":
        log.warning("LLM response was truncated (hit token limit). Will attempt JSON repair.")

    # Strip markdown fences if present
    if content.startswith("```"):
        lines = content.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines)

    try:
        script = json.loads(content)
    except json.JSONDecodeError:
        log.warning("JSON parse failed — attempting to repair truncated response")
        repaired = _repair_truncated_json(content)
        script = json.loads(repaired)

    return normalize_scenes(script)
