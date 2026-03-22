"""Voice generation via VoiceBox API."""

import logging
from pathlib import Path
import httpx

from . import config

log = logging.getLogger(__name__)


async def list_profiles() -> list[dict]:
    """Get available voice profiles from VoiceBox."""
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{config.VOICEBOX_URL}/profiles")
        if resp.status_code != 200:
            log.error(f"VoiceBox profiles error {resp.status_code}: {resp.text}")
            raise RuntimeError(f"VoiceBox returned {resp.status_code}: {resp.text}")
    profiles = resp.json()
    return [{"id": p["id"], "name": p["name"], "language": p.get("language", "en")}
            for p in profiles]


async def generate_voice(
    text: str,
    profile_id: str,
    language: str,
    output_path: Path,
    instruct: str = "",
) -> float:
    """Generate speech for text, save to output_path. Returns duration in seconds."""
    log.info(f"Generating voice for {len(text)} chars, profile={profile_id}")

    async with httpx.AsyncClient(timeout=config.VOICE_TIMEOUT_SECONDS) as client:
        # Generate speech
        body: dict = {
            "profile_id": profile_id,
            "text": text,
            "language": language,
        }
        if instruct:
            body["instruct"] = instruct

        resp = await client.post(
            f"{config.VOICEBOX_URL}/generate",
            json=body,
        )
        if resp.status_code != 200:
            error_body = resp.text
            log.error(f"VoiceBox generate error {resp.status_code}: {error_body}")
            raise RuntimeError(f"VoiceBox returned {resp.status_code}: {error_body}")
        gen = resp.json()
        generation_id = gen["id"]
        duration = gen.get("duration", 0.0)

        # Download audio file
        audio_resp = await client.get(f"{config.VOICEBOX_URL}/audio/{generation_id}")
        if audio_resp.status_code != 200:
            error_body = audio_resp.text
            log.error(f"VoiceBox audio download error {audio_resp.status_code}: {error_body}")
            raise RuntimeError(f"VoiceBox audio download failed {audio_resp.status_code}: {error_body}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(audio_resp.content)

    log.info(f"Voice saved to {output_path}, duration={duration:.1f}s")
    return duration


async def generate_all_scenes(
    scenes: list[dict],
    profile_id: str,
    language: str,
    project_dir: Path,
    instruct: str = "",
) -> list[dict]:
    """Generate voice for all scenes. Returns updated scenes with audio_path and audio_duration."""
    audio_dir = project_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    for scene in scenes:
        idx = scene["index"]
        output_path = audio_dir / f"scene_{idx:04d}.wav"
        try:
            duration = await generate_voice(
                text=scene["narration"],
                profile_id=profile_id,
                language=language,
                output_path=output_path,
                instruct=instruct,
            )
            scene["audio_path"] = str(output_path.relative_to(project_dir))
            scene["audio_duration"] = duration
        except Exception as e:
            log.error(f"Voice generation failed for scene {idx}: {e}")
            scene["audio_path"] = None
            scene["audio_duration"] = scene.get("duration_hint", 10.0)
            scene["voice_error"] = str(e)

    return scenes
