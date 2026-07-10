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

# Curated MiniMax system voices the director may pick from, per language.
# Full list: https://platform.minimax.io/docs/faq/system-voice-id
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

VOICE_CATALOG_JA: dict[str, str] = {
    "Japanese_IntellectualSenior": "older male, measured and scholarly — classic kataribe narrator",
    "Japanese_DominantMan": "deep authoritative male — heavy, menacing register",
    "Japanese_SeriousCommander": "stern mature male — clipped dramatic delivery",
    "Japanese_GentleButler": "refined older male — quiet dread, understated",
    "Japanese_ColdQueen": "mature female, cold and imperious — dark fairy tales",
    "Japanese_CalmLady": "mature female, even and low-key — neutral dramatic narrator",
}

CATALOGS: dict[str, dict[str, str]] = {"en": VOICE_CATALOG, "ja": VOICE_CATALOG_JA}

# Wider rosters for CASTING character dialogue (narrator catalogs above are
# curated for narration fit; casting needs range: young/old, hero/villain).
CAST_ROSTER_JA: dict[str, str] = {
    **VOICE_CATALOG_JA,
    "Japanese_DecisivePrincess": "young female, commanding and sharp",
    "Japanese_LoyalKnight": "young adult male, earnest hero",
    "Japanese_KindLady": "warm adult female",
    "Japanese_OptimisticYouth": "bright young male",
    "Japanese_GenerousIzakayaOwner": "hearty middle-aged male, jovial",
    "Japanese_SportyStudent": "energetic teen male",
    "Japanese_InnocentBoy": "young boy",
    "Japanese_GracefulMaiden": "gentle young female",
    "Japanese_DependableWoman": "steady adult female",
    "Japanese_DominantMan": "deep authoritative male — villains, tyrants",
}
CAST_ROSTERS: dict[str, dict[str, str]] = {"en": VOICE_CATALOG, "ja": CAST_ROSTER_JA}

DEFAULT_VOICES = {"ja": "Japanese_IntellectualSenior"}


def _catalog(language: str) -> dict[str, str]:
    # Languages without a dedicated catalog use the English voices — MiniMax
    # system voices are multilingual and language_boost steers pronunciation.
    return CATALOGS.get(language, VOICE_CATALOG)


def _default_voice(language: str) -> str:
    return DEFAULT_VOICES.get(language, config.MINIMAX_DEFAULT_VOICE)

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


def _fallback(scenes: list[dict], voice_id: str | None, language: str = "en") -> dict:
    emotions = {
        sc.get("index", i): _MOOD_EMOTION.get((sc.get("mood") or "").lower(), "auto")
        for i, sc in enumerate(scenes)
    }
    return {
        "voice_id": voice_id or _default_voice(language),
        "reason": "fallback: default voice + mood-based emotions",
        "emotions": emotions,
    }


def _build_user_prompt(
    script_meta: dict, scenes: list[dict], pick_voice: bool, language: str = "en"
) -> str:
    lines = []
    for sc in scenes:
        lines.append(
            f"[{sc.get('index')}] mood={sc.get('mood')}: "
            f"{(sc.get('narration') or '').strip()[:220]}"
        )
    catalog = "\n".join(f"- {vid}: {desc}" for vid, desc in _catalog(language).items())
    voice_part = (
        f"Voices to choose from:\n{catalog}\n\n"
        "Pick the ONE voice whose character best fits this story's nature "
        "(genre, darkness, gender of perspective, era).\n\n"
        if pick_voice
        else ""
    )
    style = script_meta.get("narration_style") or ""
    style_line = f"Narration style: {style}\n" if style else ""
    return (
        f"Story: {script_meta.get('title', '')}\n"
        f"Tone: {script_meta.get('tone', '') or 'unspecified'}\n"
        + style_line
        + f"Synopsis: {script_meta.get('synopsis', '')}\n\n"
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
    language: str = "en",
) -> dict:
    """Return {"voice_id", "reason", "emotions": {scene_index: emotion}}.

    If ``voice_id`` is given (user pinned a voice) the LLM only assigns
    per-scene emotions and the pinned voice is kept. ``language`` selects
    which voice catalog the director picks from.
    """
    if not scenes:
        return _fallback(scenes, voice_id, language)

    try:
        raw = await llm.complete(
            _SYSTEM,
            _build_user_prompt(
                script_meta, scenes, pick_voice=voice_id is None, language=language
            ),
            model=config.CLAUDE_FAST_MODEL,
            pass_name="voice director",
            timeout=config.CLAUDE_FAST_TIMEOUT_SECONDS,
        )
        data = llm.parse_json(raw)

        chosen = voice_id or (data.get("voice_id") or "").strip()
        if chosen not in _catalog(language) and chosen != voice_id:
            log.warning("Voice director picked unknown voice %r — using default", chosen)
            chosen = _default_voice(language)

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
        return _fallback(scenes, voice_id, language)


_CAST_SYSTEM = (
    "You are a voice casting director for an animated story. Given the cast "
    "of characters and the available voices, assign each character the voice "
    "that best fits their age, gender and role — plus one narrator voice. "
    "Prefer distinct voices for characters who share scenes. "
    "Respond with valid JSON only."
)


async def cast_voices(
    script_meta: dict,
    cast: list[dict],
    language: str = "en",
    narrator_voice: str | None = None,
) -> dict[str, str]:
    """Assign a voice per cast id plus 'narrator'. Falls back to round-robin."""
    roster = CAST_ROSTERS.get(language, CAST_ROSTERS["en"])
    narrator_default = narrator_voice or _default_voice(language)

    def _fallback_cast() -> dict[str, str]:
        voices = {"narrator": narrator_default}
        pool = [v for v in roster if v != narrator_default]
        for i, member in enumerate(cast):
            if member.get("id"):
                voices[member["id"]] = pool[i % len(pool)]
        return voices

    if not cast:
        return {"narrator": narrator_default}

    cast_desc = "\n".join(
        f"- {c.get('id')}: {c.get('name', '')} — {c.get('role', '')}. {c.get('description', '')}"
        for c in cast if c.get("id")
    )
    roster_desc = "\n".join(f"- {vid}: {desc}" for vid, desc in roster.items())
    narrator_part = (
        f'Narrator voice is already fixed: use "{narrator_default}" for "narrator".'
        if narrator_voice
        else "Also pick the best narrator voice."
    )
    user = (
        f"Story: {script_meta.get('title', '')}\n"
        f"Tone: {script_meta.get('tone', '')}\n\n"
        f"Cast:\n{cast_desc}\n\nAvailable voices:\n{roster_desc}\n\n"
        f"{narrator_part} Return EXACTLY this JSON:\n"
        '{"voices": {"narrator": "...", "<cast_id>": "..."}}'
    )
    try:
        raw = await llm.complete(
            _CAST_SYSTEM, user,
            model=config.CLAUDE_FAST_MODEL,
            pass_name="voice casting",
            timeout=config.CLAUDE_FAST_TIMEOUT_SECONDS,
        )
        picked = (llm.parse_json(raw).get("voices") or {})
        voices = {"narrator": narrator_default}
        if not narrator_voice and picked.get("narrator") in roster:
            voices["narrator"] = picked["narrator"]
        valid_ids = {c.get("id") for c in cast if c.get("id")}
        fallback = _fallback_cast()
        for cid in valid_ids:
            v = picked.get(cid)
            voices[cid] = v if v in roster else fallback[cid]
        log.info("Voice casting: %s", voices)
        return voices
    except Exception as e:  # noqa: BLE001
        log.warning("Voice casting LLM failed (%s) — using round-robin", e)
        return _fallback_cast()


def list_profiles() -> list[dict]:
    """Voice catalogs as profile entries for the UI dropdown."""
    auto = {
        "id": "auto",
        "name": "Auto — director picks per story",
        "language": "*",
    }
    entries = [auto]
    for lang, catalog in CATALOGS.items():
        prefix = "English_" if lang == "en" else "Japanese_"
        entries += [
            {"id": vid, "name": f"{vid.removeprefix(prefix)} — {desc}", "language": lang}
            for vid, desc in catalog.items()
        ]
    return entries
