"""Voice director — picks the narrator voice for a story and an emotion
preset for each scene (MiniMax voice backend).

Primary path: an LLM reads the story's title, synopsis, tone and every scene's
mood + narration, picks ONE voice from the curated catalog for the whole story
(one narrator throughout — consistency matters more than per-scene fit) and
assigns each scene one of MiniMax's emotion presets.
Fallback: the configured default voice + a mood→emotion map.
"""

from __future__ import annotations

import logging

from . import config, llm

log = logging.getLogger(__name__)

# Curated MiniMax system voices the director may pick from. Full list:
# https://platform.minimax.io/docs/faq/system-voice-id
VOICE_CATALOG: dict[str, str] = {
    "English_Deep-VoicedGentleman": "deep male, dark gravitas — grim/gothic tales",
    "English_CaptivatingStoryteller": "classic male storyteller, warm but dramatic",
    "English_expressive_narrator": "expressive male audiobook narrator, wide dynamic range",
    "English_ManWithDeepVoice": "very deep male, ominous register — horror",
    "English_WiseScholar": "measured older male, scholarly — myths and chronicles",
    "English_PatientMan": "calm, unhurried male — gentle or melancholy stories",
    "English_Wiselady": "wise older female, composed — folk tales, fables",
    "English_SentimentalLady": "soft emotional female — tragic or tender stories",
    "English_SereneWoman": "calm serene female — dreamlike or peaceful stories",
    "English_Graceful_Lady": "elegant female, refined — romantic or courtly tales",
}

# MiniMax speech-2.8-hd emotion presets.
EMOTIONS = {
    "auto", "happy", "sad", "angry", "fearful",
    "disgusted", "surprised", "calm", "fluent", "neutral",
}

# Fallback map from the screenwriter's mood vocabulary to an emotion preset.
_MOOD_EMOTION = {
    "horrifying": "fearful",
    "tense": "fearful",
    "ominous": "calm",
    "dark": "calm",
    "melancholy": "sad",
    "whimsical": "happy",
    "triumphant": "happy",
    "peaceful": "calm",
    "neutral": "auto",
}

_SYSTEM = (
    "You are a voice director for narrated story videos. Given a story and its "
    "scenes, you pick the single narrator voice that best fits the story's "
    "nature (one voice for the whole story — never per scene) and an emotion "
    "preset for each scene's delivery. Respond with valid JSON only."
)


def _fallback(scenes: list[dict], voice_id: str | None) -> dict:
    emotions = {
        sc.get("index", i): _MOOD_EMOTION.get((sc.get("mood") or "").lower(), "auto")
        for i, sc in enumerate(scenes)
    }
    return {
        "voice_id": voice_id or config.MINIMAX_DEFAULT_VOICE,
        "reason": "fallback: default voice + mood-based emotions",
        "emotions": emotions,
    }


def _build_user_prompt(script_meta: dict, scenes: list[dict], pick_voice: bool) -> str:
    lines = []
    for sc in scenes:
        lines.append(
            f"[{sc.get('index')}] mood={sc.get('mood')}: "
            f"{(sc.get('narration') or '').strip()[:220]}"
        )
    catalog = "\n".join(f"- {vid}: {desc}" for vid, desc in VOICE_CATALOG.items())
    voice_part = (
        f"Voices to choose from:\n{catalog}\n\n"
        "Pick the ONE voice whose character best fits this story's nature "
        "(genre, darkness, gender of perspective, era).\n\n"
        if pick_voice
        else ""
    )
    return (
        f"Story: {script_meta.get('title', '')}\n"
        f"Tone: {script_meta.get('tone', '') or 'unspecified'}\n"
        f"Synopsis: {script_meta.get('synopsis', '')}\n\n"
        f"Scenes:\n" + "\n".join(lines) + "\n\n"
        + voice_part
        + "For EACH scene pick one emotion preset from: "
        "auto, happy, sad, angry, fearful, disgusted, surprised, calm, fluent, neutral. "
        "Match the scene's narrative beat, not just its mood tag — use 'calm' for "
        "dread/ominous beats, 'fearful' for terror, 'sad' for grief, 'auto' when "
        "nothing clearly dominates. Return EXACTLY this JSON:\n"
        '{"voice_id": "...", "reason": "one sentence", '
        '"scene_emotions": [{"scene_index": 0, "emotion": "calm"}]}'
    )


async def direct(
    script_meta: dict,
    scenes: list[dict],
    voice_id: str | None = None,
) -> dict:
    """Return {"voice_id", "reason", "emotions": {scene_index: emotion}}.

    If ``voice_id`` is given (user pinned a voice) the LLM only assigns
    per-scene emotions and the pinned voice is kept.
    """
    if not scenes:
        return _fallback(scenes, voice_id)

    try:
        raw = await llm.complete(
            _SYSTEM,
            _build_user_prompt(script_meta, scenes, pick_voice=voice_id is None),
            model=config.CLAUDE_FAST_MODEL,
            pass_name="voice director",
            timeout=config.CLAUDE_FAST_TIMEOUT_SECONDS,
        )
        data = llm.parse_json(raw)

        chosen = voice_id or (data.get("voice_id") or "").strip()
        if chosen not in VOICE_CATALOG and chosen != voice_id:
            log.warning("Voice director picked unknown voice %r — using default", chosen)
            chosen = config.MINIMAX_DEFAULT_VOICE

        valid_idx = {sc.get("index") for sc in scenes}
        emotions: dict[int, str] = {}
        for item in data.get("scene_emotions") or []:
            idx = item.get("scene_index")
            emo = (item.get("emotion") or "").strip().lower()
            if idx in valid_idx and emo in EMOTIONS:
                emotions[idx] = emo
        # Any scene the LLM missed falls back to its mood mapping
        for sc in scenes:
            idx = sc.get("index")
            if idx not in emotions:
                emotions[idx] = _MOOD_EMOTION.get((sc.get("mood") or "").lower(), "auto")

        result = {
            "voice_id": chosen,
            "reason": (data.get("reason") or "").strip(),
            "emotions": emotions,
        }
        log.info(
            "Voice director: %s (%s)", chosen, result["reason"] or "no reason given"
        )
        return result
    except Exception as e:  # noqa: BLE001
        log.warning("Voice director LLM failed (%s) — using fallback", e)
        return _fallback(scenes, voice_id)


def list_profiles() -> list[dict]:
    """Voice catalog as profile entries for the UI dropdown."""
    auto = {
        "id": "auto",
        "name": "Auto — director picks per story",
        "language": "en",
    }
    return [auto] + [
        {"id": vid, "name": f"{vid.removeprefix('English_')} — {desc}", "language": "en"}
        for vid, desc in VOICE_CATALOG.items()
    ]
