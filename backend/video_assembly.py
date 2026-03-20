"""Video assembly using MoviePy — images + audio with Ken Burns effects."""

import logging
import random
from pathlib import Path

from moviepy import (
    ImageClip,
    AudioFileClip,
    CompositeVideoClip,
    concatenate_videoclips,
)

from . import config

log = logging.getLogger(__name__)


def _ken_burns_clip(
    image_path: str,
    duration: float,
    effect: str = "zoom_in",
    target_size: tuple[int, int] = (config.VIDEO_WIDTH, config.VIDEO_HEIGHT),
) -> CompositeVideoClip:
    """Create an image clip with Ken Burns effect (zoom/pan)."""
    w, h = target_size
    # Load image larger than target to allow for zoom/pan
    scale = 1.3
    clip = (
        ImageClip(image_path)
        .resized((int(w * scale), int(h * scale)))
        .with_duration(duration)
    )

    iw, ih = int(w * scale), int(h * scale)

    if effect == "zoom_in":
        # Start wide, end close
        def make_frame_pos(t):
            progress = t / duration if duration > 0 else 0
            zoom = 1.0 + progress * 0.15
            cw = int(w / zoom)
            ch = int(h / zoom)
            x = (iw - cw) // 2
            y = (ih - ch) // 2
            return (x, y)

        clip = clip.cropped(
            x1=lambda t: (iw - int(w / (1.0 + (t / duration if duration > 0 else 0) * 0.15))) // 2,
            y1=lambda t: (ih - int(h / (1.0 + (t / duration if duration > 0 else 0) * 0.15))) // 2,
            width=lambda t: int(w / (1.0 + (t / duration if duration > 0 else 0) * 0.15)),
            height=lambda t: int(h / (1.0 + (t / duration if duration > 0 else 0) * 0.15)),
        ).resized(target_size)

    elif effect == "zoom_out":
        clip = clip.cropped(
            x1=lambda t: (iw - int(w / (1.15 - (t / duration if duration > 0 else 0) * 0.15))) // 2,
            y1=lambda t: (ih - int(h / (1.15 - (t / duration if duration > 0 else 0) * 0.15))) // 2,
            width=lambda t: int(w / (1.15 - (t / duration if duration > 0 else 0) * 0.15)),
            height=lambda t: int(h / (1.15 - (t / duration if duration > 0 else 0) * 0.15)),
        ).resized(target_size)

    elif effect == "pan_left":
        max_pan = iw - w
        clip = clip.cropped(
            x1=lambda t: int(max_pan * (1 - t / duration)) if duration > 0 else 0,
            y1=(ih - h) // 2,
            width=w,
            height=h,
        )

    elif effect == "pan_right":
        max_pan = iw - w
        clip = clip.cropped(
            x1=lambda t: int(max_pan * (t / duration)) if duration > 0 else 0,
            y1=(ih - h) // 2,
            width=w,
            height=h,
        )

    else:
        # Static center crop
        clip = clip.cropped(
            x1=(iw - w) // 2,
            y1=(ih - h) // 2,
            width=w,
            height=h,
        )

    return clip


def assemble_video(
    scenes: list[dict],
    project_dir: Path,
    output_filename: str = "final.mp4",
    crossfade: float = config.CROSSFADE_DURATION,
) -> Path:
    """Assemble final video from scenes with images and audio."""
    output_path = project_dir / output_filename
    clips = []

    kb_effects = config.KB_DIRECTIONS.copy()

    for scene in scenes:
        image_path = scene.get("image_path")
        audio_path = scene.get("audio_path")

        if not image_path:
            log.warning(f"Scene {scene.get('index', '?')} has no image, skipping")
            continue

        # Resolve paths relative to project dir
        abs_image = project_dir / image_path
        abs_audio = project_dir / audio_path if audio_path else None

        if not abs_image.exists():
            log.warning(f"Image not found: {abs_image}, skipping")
            continue

        # Determine duration from audio or hint
        if abs_audio and abs_audio.exists():
            audio_clip = AudioFileClip(str(abs_audio))
            duration = audio_clip.duration
        else:
            duration = scene.get("audio_duration", scene.get("duration_hint", 10.0))
            audio_clip = None

        # Ken Burns effect — cycle through effects or use scene's setting
        effect = scene.get("kb_effect", random.choice(kb_effects))

        video_clip = _ken_burns_clip(
            image_path=str(abs_image),
            duration=duration,
            effect=effect,
        )

        if audio_clip:
            video_clip = video_clip.with_audio(audio_clip)

        clips.append(video_clip)

    if not clips:
        raise RuntimeError("No valid scenes to assemble")

    # Concatenate with crossfade
    if crossfade > 0 and len(clips) > 1:
        final = concatenate_videoclips(clips, method="compose", padding=-crossfade)
    else:
        final = concatenate_videoclips(clips, method="compose")

    final = final.with_fps(config.VIDEO_FPS)

    log.info(f"Rendering video to {output_path} ({final.duration:.1f}s)")
    final.write_videofile(
        str(output_path),
        codec="libx264",
        audio_codec="aac",
        fps=config.VIDEO_FPS,
        preset="medium",
        threads=4,
        logger=None,
    )

    # Capture duration before cleanup
    total_duration = round(final.duration, 2)

    # Clean up
    final.close()
    for c in clips:
        c.close()

    log.info(f"Video assembled: {output_path} ({total_duration}s)")
    return output_path, total_duration
