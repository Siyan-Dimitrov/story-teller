"""Script generation via Ollama LLM."""

import json
import logging
import httpx

from . import config
from .grimm_tales import get_tale

log = logging.getLogger(__name__)

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
      "image_prompt": "A detailed visual description for AI image generation: dark fairy tale illustration style, gothic, atmospheric, specific scene details, mood, lighting",
      "mood": "one word mood: dark, tense, whimsical, melancholy, horrifying, peaceful, ominous, triumphant",
      "duration_hint": 15.0
    }
  ]
}

Guidelines:
- Each scene's narration should be 60-120 words for short videos (3-5 min total), 100-200 words for longer ones
- duration_hint is approximate seconds — will be overridden by actual voice audio length
- image_prompt should always include style cues: "dark fairy tale illustration, gothic storybook art, atmospheric, detailed, moody lighting"
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
        resp.raise_for_status()

    data = resp.json()
    content = data["message"]["content"].strip()

    # Strip markdown fences if present
    if content.startswith("```"):
        lines = content.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines)

    script = json.loads(content)

    # Normalize scene indices
    for i, scene in enumerate(script.get("scenes", [])):
        scene["index"] = i
        scene.setdefault("mood", "neutral")
        scene.setdefault("duration_hint", 15.0)

    return script
