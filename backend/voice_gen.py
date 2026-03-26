"""Voice generation via VoiceBox API with sentence-level chunking.

Splits narration into individual sentences before TTS to avoid Qwen3-TTS
quality degradation on longer text (trailing off, accelerating speech rate,
ignored punctuation). Chunks are generated separately and concatenated with
silence gaps for natural pacing.
"""

import asyncio
import io
import logging
import re
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

from . import config

log = logging.getLogger(__name__)

# Silence inserted between sentences (seconds)
SENTENCE_GAP_SECONDS = 0.20
# Trailing silence appended to each scene's audio (seconds)
SCENE_TRAILING_SILENCE = 0.70


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences at natural boundaries (.!?).

    Merges very short trailing fragments with the previous sentence.
    Ensures every sentence ends with punctuation for proper TTS pacing.
    """
    # Normalize em-dashes to commas (Qwen TTS handles commas better)
    text = text.replace("\u2014", ", ").replace("--", ", ")

    # Split on sentence-ending punctuation (including inside closing quotes)
    parts = re.split(r'(?:(?<=[.!?])|(?<=[.!?]["\'\u201d\u2019]))\s+', text.strip())

    # Merge very short fragments (< 30 chars) with previous sentence
    sentences: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if sentences and len(part) < 30:
            sentences[-1] = sentences[-1] + " " + part
        else:
            sentences.append(part)

    # Ensure every sentence ends with punctuation (handle closing quotes)
    result: list[str] = []
    for s in sentences:
        s = s.strip()
        if s and not re.search(r'[.!?]["\'\u201d\u2019)]*$', s):
            s += "."
        if s:
            result.append(s)

    return result if result else [text]


async def _generate_single(
    text: str,
    profile_id: str,
    language: str,
    instruct: str,
    client: httpx.AsyncClient,
) -> tuple[bytes, float]:
    """Generate speech for a single text chunk. Returns (wav_bytes, duration)."""
    body: dict = {
        "profile_id": profile_id,
        "text": text,
        "language": language,
    }
    if instruct:
        body["instruct"] = instruct

    resp = await client.post(f"{config.VOICEBOX_URL}/generate", json=body)
    if resp.status_code != 200:
        raise RuntimeError(f"VoiceBox returned {resp.status_code}: {resp.text}")

    gen = resp.json()
    generation_id = gen["id"]
    dur = gen.get("duration", 0.0)

    audio_resp = await client.get(f"{config.VOICEBOX_URL}/audio/{generation_id}")
    if audio_resp.status_code != 200:
        raise RuntimeError(
            f"VoiceBox audio download failed {audio_resp.status_code}: {audio_resp.text}"
        )

    return audio_resp.content, dur


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


def _concatenate_wav_chunks(
    chunks: list[bytes], gap_seconds: float = SENTENCE_GAP_SECONDS
) -> tuple[bytes, float]:
    """Concatenate WAV byte chunks with silence gaps between them.

    Returns (combined_wav_bytes, total_duration_seconds).
    """
    arrays: list[np.ndarray] = []
    sample_rate: int | None = None

    for chunk_bytes in chunks:
        data, sr = sf.read(io.BytesIO(chunk_bytes))
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            log.warning(f"Sample rate mismatch: {sr} vs {sample_rate}")
        arrays.append(data)

    if not arrays or sample_rate is None:
        return b"", 0.0

    # Create silence gap matching the audio shape
    silence_samples = int(gap_seconds * sample_rate)
    if arrays[0].ndim == 1:
        silence = np.zeros(silence_samples, dtype=arrays[0].dtype)
    else:
        silence = np.zeros((silence_samples, arrays[0].shape[1]), dtype=arrays[0].dtype)

    # Interleave audio chunks with silence
    combined_parts: list[np.ndarray] = []
    for i, arr in enumerate(arrays):
        combined_parts.append(arr)
        if i < len(arrays) - 1:
            combined_parts.append(silence)

    combined = np.concatenate(combined_parts)
    total_duration = len(combined) / sample_rate

    buf = io.BytesIO()
    sf.write(buf, combined, sample_rate, format="WAV")
    buf.seek(0)

    return buf.read(), total_duration


async def list_profiles() -> list[dict]:
    """Get available voice profiles from VoiceBox."""
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{config.VOICEBOX_URL}/profiles")
        if resp.status_code != 200:
            log.error(f"VoiceBox profiles error {resp.status_code}: {resp.text}")
            raise RuntimeError(f"VoiceBox returned {resp.status_code}: {resp.text}")
    profiles = resp.json()
    return [
        {"id": p["id"], "name": p["name"], "language": p.get("language", "en")}
        for p in profiles
    ]


async def generate_voice(
    text: str,
    profile_id: str,
    language: str,
    output_path: Path,
    instruct: str = "",
    client: httpx.AsyncClient | None = None,
) -> float:
    """Generate speech for text with sentence-level chunking.

    Splits text into sentences, generates each separately via VoiceBox,
    then concatenates with silence gaps. This avoids Qwen TTS quality
    degradation on longer text (trailing off, accelerating speech rate).
    """
    sentences = _split_into_sentences(text)
    log.info(
        f"Generating voice for {len(text)} chars ({len(sentences)} sentence(s)), "
        f"profile={profile_id}"
    )

    async def _do(c: httpx.AsyncClient) -> float:
        if len(sentences) == 1:
            # Single sentence - no chunking overhead
            wav_bytes, dur = await _generate_single(
                sentences[0], profile_id, language, instruct, c
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(wav_bytes)
            return dur

        # Multiple sentences - generate each, then concatenate
        wav_chunks: list[bytes] = []
        for i, sentence in enumerate(sentences):
            log.debug(f"  Chunk {i + 1}/{len(sentences)}: {sentence[:60]}...")
            wav_bytes, _ = await _generate_single(
                sentence, profile_id, language, instruct, c
            )
            wav_chunks.append(wav_bytes)
            # Brief pause between API calls
            if i < len(sentences) - 1:
                await asyncio.sleep(0.3)

        combined_bytes, total_dur = _concatenate_wav_chunks(wav_chunks)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(combined_bytes)
        return total_dur

    if client:
        duration = await _do(client)
    else:
        async with httpx.AsyncClient(timeout=config.VOICE_TIMEOUT_SECONDS) as c:
            duration = await _do(c)

    log.info(f"Voice saved to {output_path}, duration={duration:.1f}s")
    return duration


async def generate_all_scenes(
    scenes: list[dict],
    profile_id: str,
    language: str,
    project_dir: Path,
    instruct: str = "",
) -> list[dict]:
    """Generate voice for all scenes with sentence-level chunking.

    Returns updated scenes with audio_path and audio_duration.
    """
    audio_dir = project_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    async with httpx.AsyncClient(timeout=config.VOICE_TIMEOUT_SECONDS) as client:
        for i, scene in enumerate(scenes):
            idx = scene["index"]
            output_path = audio_dir / f"scene_{idx:04d}.wav"
            try:
                duration = await generate_voice(
                    text=scene["narration"],
                    profile_id=profile_id,
                    language=language,
                    output_path=output_path,
                    instruct=instruct,
                    client=client,
                )
                # Append trailing silence for natural pause between scenes
                duration = _append_trailing_silence(output_path, SCENE_TRAILING_SILENCE)
                scene["audio_path"] = str(output_path.relative_to(project_dir))
                scene["audio_duration"] = duration
                scene.pop("voice_error", None)
            except Exception as e:
                log.error(f"Voice generation failed for scene {idx}: {e}")
                scene["audio_path"] = None
                scene["audio_duration"] = scene.get("duration_hint", 10.0)
                scene["voice_error"] = str(e)

            # Brief pause between scenes to avoid overwhelming VoiceBox's DB pool
            if i < len(scenes) - 1:
                await asyncio.sleep(1.0)

    return scenes
