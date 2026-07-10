"""Narration localization — translate scenes for non-English voice-over.

The screenwriter always writes English. When a project's voice language is
not English, this pass translates each scene's narration sentence-by-sentence
(1:1 alignment) so the English sentences can be timed as subtitles against
the localized audio. Results are stored on the scene:

    scene["localized"] = {
        "lang": "ja",
        "text": "...",              # what TTS speaks
        "sentences": ["...", ...],   # localized, 1:1 with sentences_en
        "sentences_en": ["...", ...],
    }
"""

from __future__ import annotations

import logging
import re

from . import config, llm

log = logging.getLogger(__name__)

LANGUAGE_NAMES = {
    "de": "German", "fr": "French", "es": "Spanish", "it": "Italian",
    "pt": "Portuguese", "ru": "Russian", "ja": "Japanese", "ko": "Korean",
    "zh": "Chinese",
}

_SYSTEM = (
    "You are a literary translator for narrated story videos. You translate "
    "English narration into the target language with natural, dramatic spoken "
    "prosody — not word-for-word, but faithful in meaning and register. "
    "CRITICAL: you translate sentence by sentence, producing EXACTLY one "
    "target sentence per source sentence, in the same order, so the two lists "
    "stay aligned for subtitling. For Japanese, write rare or ambiguous name "
    "kanji in kana so text-to-speech reads them correctly. "
    "Respond with valid JSON only."
)


def split_sentences(text: str) -> list[str]:
    """Split English narration into sentences for subtitle alignment."""
    parts = re.split(r'(?:(?<=[.!?])|(?<=[.!?]["\'”’]))\s+', text.strip())
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if out and len(p) < 20:
            out[-1] = out[-1] + " " + p
        else:
            out.append(p)
    return out or [text.strip()]


def _build_user_prompt(scenes_sentences: dict[int, list[str]], language: str) -> str:
    lang_name = LANGUAGE_NAMES.get(language, language)
    blocks = []
    for idx, sents in scenes_sentences.items():
        numbered = "\n".join(f"  {i}: {s}" for i, s in enumerate(sents))
        blocks.append(f"Scene {idx}:\n{numbered}")
    return (
        f"Translate this story narration to {lang_name}. One translated "
        "sentence per numbered source sentence, same order. Return EXACTLY "
        "this JSON:\n"
        '{"scenes": [{"scene_index": 0, "sentences": ["...", "..."]}]}\n\n'
        + "\n\n".join(blocks)
    )


async def localize_scenes(scenes: list[dict], language: str) -> list[dict]:
    """Translate narration for all scenes that need it. Mutates and returns scenes.

    Scenes already localized to ``language`` are left untouched, so re-runs
    only translate what changed.
    """
    if language == "en" or language not in LANGUAGE_NAMES:
        return scenes

    todo: dict[int, list[str]] = {}
    for sc in scenes:
        loc = sc.get("localized") or {}
        if loc.get("lang") == language and loc.get("text"):
            continue
        todo[sc["index"]] = split_sentences(sc.get("narration") or "")
    if not todo:
        return scenes

    log.info(f"Localizing {len(todo)} scene(s) to {LANGUAGE_NAMES[language]}")
    raw = await llm.complete(
        _SYSTEM,
        _build_user_prompt(todo, language),
        model=config.CLAUDE_FAST_MODEL,
        pass_name="localize",
        timeout=config.CLAUDE_FAST_TIMEOUT_SECONDS,
    )
    data = llm.parse_json(raw)
    translations = {
        item.get("scene_index"): item.get("sentences") or []
        for item in data.get("scenes") or []
    }

    joiner = "" if language in ("ja", "zh") else " "
    for sc in scenes:
        idx = sc["index"]
        if idx not in todo:
            continue
        src = todo[idx]
        out = [s.strip() for s in translations.get(idx, []) if s and s.strip()]
        if len(out) != len(src):
            log.warning(
                f"Scene {idx}: translation returned {len(out)} sentences for "
                f"{len(src)} source sentences — subtitle timing will be scene-level"
            )
            if not out:
                raise RuntimeError(f"Localization failed for scene {idx}")
        sc["localized"] = {
            "lang": language,
            "text": joiner.join(out),
            "sentences": out,
            "sentences_en": src,
        }
    return scenes
