"""Script generation via Ollama LLM."""

import json
import logging
import httpx

from . import config
from .grimm_tales import get_tale

log = logging.getLogger(__name__)


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
                "options": {"temperature": 0.7, "num_predict": 4000},
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
        "First visual moment: detailed description for AI image generation with style cues",
        "Second visual moment: a different angle, close-up, or progression of the action",
        "Third visual moment: another key visual beat from this scene",
        "Fourth visual moment: dramatic reaction, landscape, or detail shot"
      ],
      "mood": "one word mood: dark, tense, whimsical, melancholy, horrifying, peaceful, ominous, triumphant",
      "duration_hint": 15.0
    }
  ]
}

Guidelines:
- Each scene's narration should be 60-120 words for short videos (3-5 min total), 100-200 words for longer ones
- duration_hint is approximate seconds — will be overridden by actual voice audio length
- Each scene needs exactly 4 image_prompts — each captures a different visual moment within the scene
- Think of image_prompts like storyboard panels: wide establishing shot, character focus, action beat, emotional close-up
- Every image_prompt should include style cues: "dark fairy tale illustration, gothic storybook art, atmospheric, detailed, moody lighting"
- Vary the composition across prompts: wide shots, close-ups, overhead views, detail shots
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

    # Strip markdown fences if present
    if content.startswith("```"):
        lines = content.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines)

    script = json.loads(content)

    # Normalize scene indices and image prompts
    for i, scene in enumerate(script.get("scenes", [])):
        scene["index"] = i
        scene.setdefault("mood", "neutral")
        scene.setdefault("duration_hint", 15.0)

        # Normalize image_prompts: support both old single and new multi format
        if "image_prompts" not in scene or not scene["image_prompts"]:
            # Backward compat: wrap single image_prompt into a list
            single = scene.get("image_prompt", "")
            scene["image_prompts"] = [single] if single else []

        # Set image_prompt to first prompt for backward compat
        if scene["image_prompts"]:
            scene["image_prompt"] = scene["image_prompts"][0]

    return script
