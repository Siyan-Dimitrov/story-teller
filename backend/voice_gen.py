"""Voice generation via MiniMax Speech on Replicate.

Each scene's narration goes out as a single TTS request (the model accepts up
to 10,000 characters), so no sentence chunking or chunk-boundary cleanup is
needed. The narrator voice and per-scene emotion preset come from the LLM
voice director (see voice_director.py).
"""

import asyncio
import logging
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

from . import config, voice_director

log = logging.getLogger(__name__)

# Trailing silence appended to each scene's audio (seconds)
SCENE_TRAILING_SILENCE = 0.70

# Map the UI's 2-letter language codes to MiniMax language_boost values.
MINIMAX_LANGUAGE_BOOST = {
    "en": "English", "de": "German", "fr": "French", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "ru": "Russian", "ja": "Japanese",
    "ko": "Korean", "zh": "Chinese",
}


def _clean_for_minimax(text: str) -> str:
    return text.replace("…", "...").replace(" ", " ").strip()


def _append_trailing_silence(wav_path: Path, silence_seconds: float) -> float:
    """Append trailing silence to a WAV file. Returns new total duration."""
    data, sr = sf.read(str(wav_path))
    silence_samples = int(silence_seconds * sr)
    if data.ndim == 1:
        silence = np.zeros(silence_samples, dtype=data.dtype)
    else:
        silence = np.zeros((silence_samples, data.shape[1]), dtype=data.dtype)
    combined = np.concatenate([data, silence])
    sf.write(str(wav_path), combined, sr, format="WAV")
    return len(combined) / sr


async def _generate_minimax_scene(
    text: str,
    voice_id: str,
    emotion: str,
    language: str,
) -> bytes:
    """Generate one scene's speech via MiniMax on Replicate. Returns WAV bytes."""
    import replicate as _replicate

    inp = {
        "text": _clean_for_minimax(text),
        "voice_id": voice_id,
        "emotion": emotion if emotion in voice_director.EMOTIONS else "auto",
        "speed": config.MINIMAX_SPEED,
        "audio_format": "wav",
        "sample_rate": config.MINIMAX_SAMPLE_RATE,
        "language_boost": MINIMAX_LANGUAGE_BOOST.get(language, "Automatic"),
    }

    last_error: Exception | None = None
    for attempt in range(config.MINIMAX_MAX_RETRIES + 1):
        try:
            loop = asyncio.get_event_loop()
            output = await loop.run_in_executor(
                None, lambda: _replicate.run(config.MINIMAX_TTS_MODEL, input=inp)
            )
            url = str(output[0]) if isinstance(output, list) else str(output)
            async with httpx.AsyncClient(timeout=config.VOICE_TIMEOUT_SECONDS) as client:
                resp = await client.get(url, follow_redirects=True)
                resp.raise_for_status()
                return resp.content
        except Exception as e:  # noqa: BLE001
            last_error = e
            err = str(e).lower()
            retryable = any(kw in err for kw in [
                "throttl", "rate", "429", "too many", "overloaded",
                "503", "502", "timeout", "timed out",
            ])
            if retryable and attempt < config.MINIMAX_MAX_RETRIES:
                wait = 5.0 * (attempt + 1)
                log.warning(f"MiniMax TTS transient error ({e}) — retrying in {wait}s")
                await asyncio.sleep(wait)
            else:
                raise
    raise RuntimeError(f"MiniMax TTS failed: {last_error}")


async def generate_all_scenes(
    scenes: list[dict],
    profile_id: str,
    language: str,
    project_dir: Path,
    script_meta: dict | None = None,
) -> list[dict]:
    """Generate voice for all scenes.

    profile_id "auto" (or empty) lets the voice director pick the narrator
    voice; a concrete voice_id pins it and the director only assigns
    per-scene emotions. Returns updated scenes with audio_path,
    audio_duration, voice_id and emotion.
    """
    audio_dir = project_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    pinned = profile_id if profile_id and profile_id != "auto" else None
    direction = await voice_director.direct(
        script_meta or {}, scenes, voice_id=pinned, language=language
    )
    voice_id = direction["voice_id"]
    emotions = direction["emotions"]
    log.info(f"MiniMax voice: {voice_id} ({len(scenes)} scenes)")

    for i, scene in enumerate(scenes):
        idx = scene["index"]
        emotion = emotions.get(idx, "auto")
        output_path = audio_dir / f"scene_{idx:04d}.wav"
        # Localized narration (set by localize.localize_scenes) takes over
        # when it matches the requested language; English subs come later.
        loc = scene.get("localized") or {}
        text = loc.get("text") if loc.get("lang") == language else scene["narration"]
        try:
            wav_bytes = await _generate_minimax_scene(
                text, voice_id, emotion, language
            )
            output_path.write_bytes(wav_bytes)
            duration = _append_trailing_silence(output_path, SCENE_TRAILING_SILENCE)
            scene["audio_path"] = str(output_path.relative_to(project_dir))
            scene["audio_duration"] = duration
            scene["voice_id"] = voice_id
            scene["emotion"] = emotion
            scene.pop("voice_error", None)
            log.info(f"Scene {idx}: {duration:.1f}s (emotion={emotion})")
        except Exception as e:  # noqa: BLE001
            log.error(f"MiniMax voice generation failed for scene {idx}: {e}")
            scene["audio_path"] = None
            scene["audio_duration"] = scene.get("duration_hint", 10.0)
            scene["voice_error"] = str(e)

        if i < len(scenes) - 1:
            await asyncio.sleep(config.MINIMAX_DELAY_SECONDS)

    return scenes
