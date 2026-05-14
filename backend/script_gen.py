"""Script generation via Ollama (Kimi)."""

import asyncio
import json
import logging
import re
import httpx

from . import config
from .grimm_tales import get_tale
from .agents.budget import cost_logged

log = logging.getLogger(__name__)


# Source-text handling
LONG_SOURCE_WARN_CHARS = 8000      # warn the user
LONG_SOURCE_SUMMARIZE_CHARS = 12000  # above this, run a summarize pre-pass
SUMMARIZE_TARGET_CHARS = 3500
TRUNCATION_RETRY_FACTOR = 0.7      # shrink target_minutes by 30% on truncation
MAX_TRUNCATION_RETRIES = 2


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
    content = _strip_markdown_fences(content) if content else ""

    if not content.strip():
        log.warning(f"search_stories: model returned empty content (model={model}); returning []")
        return []
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        log.warning(f"search_stories: model returned non-JSON ({e}); returning []. head={content[:200]!r}")
        return []
    return parsed.get("results", []) if isinstance(parsed, dict) else []

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
        "Second image prompt — must depict a different specific moment from the narration"
      ],
      "mood": "one word mood: dark, tense, whimsical, melancholy, horrifying, peaceful, ominous, triumphant",
      "duration_hint": 15.0
    }
  ]
}

Guidelines:
- Each scene's narration should be 60-120 words for short videos (3-5 min total), 100-200 words for longer ones
- duration_hint is approximate seconds — will be overridden by actual voice audio length
- Each scene needs exactly 2 image_prompts
- CRITICAL — image prompts must be grounded in the narration text:
  1. Read the narration you wrote for the scene
  2. Identify the two most visually striking moments or images described in it
  3. Write each prompt as a literal depiction of that moment: name the specific characters, objects, setting, and action happening
  4. Do NOT write generic prompts like "a dark forest" — instead write "the woodcutter's adult daughter kneeling beside a broken juniper branch, a crimson ribbon in her hands, moonlight through bare trees"
- Each prompt must include: WHO (specific character/creature), WHAT (specific action), WHERE (specific setting detail), and WHEN/LIGHTING if relevant
- Include continuity anchors for recurring characters: apparent age, clothing, hair, posture, and one memorable identifying feature
- Include explicit camera/framing language in every prompt, such as "wide establishing shot", "low-angle medium shot", "over-the-shoulder shot", "close-up", or "silhouette against the doorway"
- Vary composition between the two prompts: usually one wider environmental shot and one tighter emotional/action shot
- Do NOT append generic art-style boilerplate to image_prompts. The selected image backend adds style separately via style_prompt.
- Do NOT request captions, title cards, typography, subtitles, logos, or text inside the image unless the story explicitly requires a visible sign or written object
- For image_prompts, adapt disturbing story beats as symbolic, non-graphic folklore imagery. Avoid gore, visible injury, sexual content, intimate contact, restraint, torture, explicit burning, cannibalism, and children in danger.
- When a recurring character is grown, say "adult" in the image prompt so image backends do not infer a minor.
- Keep each image_prompt concise: one vivid sentence, 35-70 words, concrete nouns and actions only
- Aim for the number of scenes that fits the target length (roughly 1 scene per 30-60 seconds)
- The narration should be vivid and engaging when read aloud — this is a voiceover script
- Never break the fourth wall or reference that this is a video/script
"""


SUMMARIZE_SYSTEM_PROMPT = """You are a faithful summarizer of fairy-tale source texts. Compress the input into a dense plot synopsis preserving every named character, every key event in order, every setting change, every supernatural element, and the ending. No commentary. No interpretation. No omitted acts. Aim for about 3000-4000 characters."""


def _strip_markdown_fences(content: str) -> str:
    if content.startswith("```"):
        lines = content.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines)
    return content


def _normalize_script(script: dict, effective_target_minutes: float | None = None) -> dict:
    for i, scene in enumerate(script.get("scenes", [])):
        scene["index"] = i
        scene.setdefault("mood", "neutral")
        scene.setdefault("duration_hint", 15.0)
        if "image_prompts" not in scene or not scene["image_prompts"]:
            single = scene.get("image_prompt", "")
            scene["image_prompts"] = [single] if single else []
        if scene["image_prompts"]:
            scene["image_prompt"] = scene["image_prompts"][0]
    if effective_target_minutes is not None:
        # The actual minutes the LLM was asked to deliver, after any truncation
        # retries shrunk the target. The Critic should evaluate against this
        # — not against the original `state.target_minutes` — otherwise it
        # will reject every shrunk script as "too short" and loop forever.
        script["effective_target_minutes"] = round(effective_target_minutes, 1)
    return script


async def _summarize_long_source(text: str, ollama_model: str | None, base_url: str | None) -> str:
    """Compress a long chapter via a cheap Ollama call. Falls back to the original
    text if the summary is empty/too-short — better to send a long prompt than
    nothing."""
    model = ollama_model or config.OLLAMA_MODEL
    url = base_url or config.OLLAMA_URL
    log.info(f"Summarizing long source ({len(text):,} chars) via {model}")
    async with httpx.AsyncClient(timeout=config.LLM_TIMEOUT_SECONDS) as client:
        resp = await client.post(
            f"{url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SUMMARIZE_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 4096},
            },
        )
        resp.raise_for_status()
    data = resp.json()
    summary = _extract_llm_content(data).strip()
    if len(summary) < 200:
        # Capture the response keys so we can tell whether the model returned an
        # error envelope, a structured-output skip, or just an empty response.
        msg = data.get("message", {})
        log.warning(
            f"Summarize returned only {len(summary)} chars (model={model}); "
            f"falling back to original {len(text):,}-char source. "
            f"done_reason={data.get('done_reason')!r}, msg_keys={list(msg.keys())}, "
            f"content_len={len(msg.get('content') or '')}, thinking_len={len(msg.get('thinking') or '')}"
        )
        return text
    log.info(f"Summary length: {len(summary):,} chars (from {len(text):,})")
    return summary


def _build_user_prompt(
    source_tale: str,
    custom_prompt: str,
    target_minutes: float,
    tone: str,
) -> str:
    parts = []
    if source_tale:
        tale = get_tale(source_tale)
        if tale:
            parts.append("Retell this dark fairy tale in your narrator voice:\n")
            parts.append(f"Title: {tale['title']}\n")
            parts.append(f"Origin: {tale['origin']}\n")
            parts.append(f"Synopsis:\n{tale['synopsis']}\n")
        else:
            parts.append(f"Write a dark fairy tale based on: {source_tale}\n")
    if tone:
        parts.append(f"\nAdaptation tone: {tone}. Infuse the story with this tone throughout.\n")
    if custom_prompt:
        parts.append(f"\nAdditional direction: {custom_prompt}\n")
    scene_count = max(5, int(target_minutes * 1.5))
    parts.append(f"\nTarget length: approximately {target_minutes} minutes when narrated aloud.")
    parts.append(f"\nAim for roughly {scene_count} scenes.")
    return "\n".join(parts)


async def _call_ollama_for_script(user_prompt: str, model: str, base_url: str) -> tuple[str, str]:
    """One Ollama chat call. Returns (content, done_reason)."""
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
    return _extract_llm_content(data), data.get("done_reason", "")


def _parse_script_content(content: str, allow_repair: bool) -> dict:
    """Parse LLM content into script dict. allow_repair=False asks the caller to retry instead of patching."""
    content = _strip_markdown_fences(content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        if not allow_repair:
            raise
        log.warning("JSON parse failed — attempting to repair truncated response")
        return json.loads(_repair_truncated_json(content))


@cost_logged("ollama", "OLLAMA_MODEL")
async def generate_script(
    source_tale: str = "",
    custom_prompt: str = "",
    target_minutes: float = 5.0,
    ollama_model: str | None = None,
    ollama_base_url: str | None = None,
    tone: str = "",
) -> dict:
    if custom_prompt:
        if len(custom_prompt) >= LONG_SOURCE_WARN_CHARS:
            log.warning(
                f"Custom prompt is {len(custom_prompt):,} chars — long sources increase truncation risk"
            )
        if len(custom_prompt) >= LONG_SOURCE_SUMMARIZE_CHARS:
            try:
                custom_prompt = await _summarize_long_source(custom_prompt, ollama_model, ollama_base_url)
            except Exception as e:
                log.warning(f"Summarize pre-pass failed ({e}); proceeding with full text")

    return await _generate_script_with_retry(
        source_tale=source_tale,
        custom_prompt=custom_prompt,
        target_minutes=target_minutes,
        tone=tone,
        ollama_model=ollama_model,
        ollama_base_url=ollama_base_url,
        retry_depth=0,
    )


async def _generate_script_with_retry(
    *,
    source_tale: str,
    custom_prompt: str,
    target_minutes: float,
    tone: str,
    ollama_model: str | None,
    ollama_base_url: str | None,
    retry_depth: int,
) -> dict:
    user_prompt = _build_user_prompt(source_tale, custom_prompt, target_minutes, tone)

    model = ollama_model or config.OLLAMA_MODEL
    base_url = ollama_base_url or config.OLLAMA_URL
    log.info(f"Generating script via ollama model={model}, target={target_minutes:.1f}min")
    content, done_reason = await _call_ollama_for_script(user_prompt, model, base_url)

    # Step 2A: detect length-truncation BEFORE attempting repair, retry smaller.
    if done_reason == "length" and retry_depth < MAX_TRUNCATION_RETRIES:
        shrunk = round(target_minutes * TRUNCATION_RETRY_FACTOR, 1)
        log.warning(
            f"Ollama hit token limit (done_reason=length) at target={target_minutes:.1f}min — "
            f"retrying with target={shrunk:.1f}min (attempt {retry_depth + 2}/{MAX_TRUNCATION_RETRIES + 1})"
        )
        return await _generate_script_with_retry(
            source_tale=source_tale, custom_prompt=custom_prompt,
            target_minutes=shrunk, tone=tone,
            ollama_model=ollama_model, ollama_base_url=ollama_base_url,
            retry_depth=retry_depth + 1,
        )
    if done_reason == "length":
        log.warning("Ollama still truncated after retries; falling back to JSON repair")

    return _normalize_script(_parse_script_content(content, allow_repair=True), target_minutes)
