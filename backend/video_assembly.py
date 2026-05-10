"""Video assembly using MoviePy — images + audio with Ken Burns effects."""

import logging
import random
import subprocess
import threading
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from proglog import ProgressBarLogger

from moviepy import (
    VideoClip,
    AudioFileClip,
    CompositeAudioClip,
    concatenate_videoclips,
)
from moviepy.audio.fx import MultiplyVolume, AudioFadeIn, AudioFadeOut, AudioLoop

from . import config

log = logging.getLogger(__name__)

# ── Assembly progress tracking ───────────────────────────────

_assembly_lock = threading.Lock()
_assembly_tasks: dict[str, dict] = {}


class _AssemblyProgressLogger(ProgressBarLogger):
    """Captures MoviePy encoding progress for the frontend."""

    def __init__(self, project_id: str):
        super().__init__()
        self.project_id = project_id

    def callback(self, **changes):
        pass

    def bars_callback(self, bar, attr, value, old_value=None):
        if attr == "index":
            total = self.bars[bar].get("total", 0)
            if total > 0:
                with _assembly_lock:
                    state = _assembly_tasks.get(self.project_id)
                    if state:
                        state["progress"] = round(value / total, 3)
                        state["phase"] = "encoding"
                        if state.get("cancel"):
                            raise RuntimeError("Assembly cancelled by user")


def get_assembly_progress(project_id: str) -> dict:
    with _assembly_lock:
        state = _assembly_tasks.get(project_id)
        if not state:
            return {"active": False, "progress": 0, "phase": "idle", "error": None}
        return {
            "active": state.get("active", False),
            "progress": state["progress"],
            "phase": state["phase"],
            "error": state.get("error"),
        }


def cancel_assembly(project_id: str) -> bool:
    with _assembly_lock:
        state = _assembly_tasks.get(project_id)
        if state and state.get("active"):
            state["cancel"] = True
            return True
        return False


def _init_progress(project_id: str):
    with _assembly_lock:
        _assembly_tasks[project_id] = {
            "active": True,
            "progress": 0,
            "phase": "preparing",
            "cancel": False,
            "error": None,
        }


def _update_progress(project_id: str, **kw):
    with _assembly_lock:
        state = _assembly_tasks.get(project_id)
        if state:
            state.update(kw)
            if state.get("cancel"):
                raise RuntimeError("Assembly cancelled by user")


def _finish_progress(project_id: str, error: str | None = None):
    with _assembly_lock:
        state = _assembly_tasks.get(project_id)
        if state:
            state["active"] = False
            if error:
                state["phase"] = "error"
                state["error"] = error
            else:
                state["phase"] = "done"
                state["progress"] = 1.0


# ── Ken Burns ────────────────────────────────────────────────

def _ken_burns_clip(
    image_path: str,
    duration: float,
    effect: str = "zoom_in",
    target_size: tuple[int, int] = (config.VIDEO_WIDTH, config.VIDEO_HEIGHT),
) -> VideoClip:
    """Create a clip with Ken Burns effect using per-frame cropping.

    MoviePy v2 doesn't support lambdas in cropped(), so we build
    a VideoClip with a custom make_frame that crops the source image
    per-frame using numpy slicing.
    """
    w, h = target_size
    scale = 1.3
    iw, ih = int(w * scale), int(h * scale)

    # Load and resize the source image once as a numpy array
    pil_img = PILImage.open(image_path).convert("RGB").resize((iw, ih), PILImage.LANCZOS)
    src = np.array(pil_img)

    def _crop_and_resize(src_arr, x1, y1, cw, ch):
        """Crop a region from src and resize to target."""
        x1 = max(0, min(x1, iw - cw))
        y1 = max(0, min(y1, ih - ch))
        cropped = src_arr[y1:y1 + ch, x1:x1 + cw]
        if cw == w and ch == h:
            return cropped
        # BILINEAR is ~4x faster than LANCZOS, imperceptible for Ken Burns
        pil = PILImage.fromarray(cropped).resize((w, h), PILImage.BILINEAR)
        return np.array(pil)

    # Cache: consecutive frames often share the same integer crop dims
    _cache = {}

    def _cached_crop(src_arr, x1, y1, cw, ch):
        key = (x1, y1, cw, ch)
        if key not in _cache:
            _cache.clear()  # only keep one entry to limit memory
            _cache[key] = _crop_and_resize(src_arr, x1, y1, cw, ch)
        return _cache[key]

    if effect == "zoom_in":
        def make_frame(t):
            p = t / duration if duration > 0 else 0
            zoom = 1.0 + p * 0.15
            cw, ch = int(w / zoom), int(h / zoom)
            x1, y1 = (iw - cw) // 2, (ih - ch) // 2
            return _cached_crop(src, x1, y1, cw, ch)

    elif effect == "zoom_out":
        def make_frame(t):
            p = t / duration if duration > 0 else 0
            zoom = 1.15 - p * 0.15
            cw, ch = int(w / zoom), int(h / zoom)
            x1, y1 = (iw - cw) // 2, (ih - ch) // 2
            return _cached_crop(src, x1, y1, cw, ch)

    elif effect == "pan_left":
        max_pan = iw - w
        cy = (ih - h) // 2

        def make_frame(t):
            p = t / duration if duration > 0 else 0
            x1 = int(max_pan * (1 - p))
            return _crop_and_resize(src, x1, cy, w, h)

    elif effect == "pan_right":
        max_pan = iw - w
        cy = (ih - h) // 2

        def make_frame(t):
            p = t / duration if duration > 0 else 0
            x1 = int(max_pan * p)
            return _crop_and_resize(src, x1, cy, w, h)

    else:
        # Static center crop
        x1, y1 = (iw - w) // 2, (ih - h) // 2
        static_frame = _crop_and_resize(src, x1, y1, w, h)

        def make_frame(t):
            return static_frame

    return VideoClip(make_frame, duration=duration).with_fps(config.VIDEO_FPS)


# ── Assembly ─────────────────────────────────────────────────

def _resolve_music_path(music_track: str | None) -> Path | None:
    """Resolve a user-supplied music_track to an absolute path.

    Accepts an absolute path, a filename inside data/music/, an http(s) URL
    (which will be downloaded into the music cache), or None.
    """
    if not music_track:
        return None
    if music_track.startswith(("http://", "https://")):
        from . import music_search
        return music_search.download_music_to_cache(music_track)
    candidate = Path(music_track)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    in_music_dir = config.MUSIC_DIR / music_track
    if in_music_dir.exists():
        return in_music_dir
    log.warning(f"Music track not found: {music_track}")
    return None


def _loudnorm_track(src: Path) -> Path:
    """Return a loudness-normalized copy of src (cached on disk).

    Different tracks ship at wildly different mastering levels, so a single master
    volume multiplier sounds inconsistent scene-to-scene. We pre-process each track
    through ffmpeg's EBU R128 loudnorm filter so every bed hits the same perceived
    loudness target (-23 LUFS) — then the user's volume slider lands predictably.

    Cached copies live in data/music/.loudnorm/ and are keyed by source mtime so
    edits/replacements invalidate automatically. Falls back to the raw file if
    ffmpeg is unavailable or the pass fails — we don't want to block assembly.
    """
    cache_dir = config.MUSIC_DIR / ".loudnorm"
    cache_dir.mkdir(exist_ok=True)

    try:
        mtime = int(src.stat().st_mtime)
    except OSError:
        return src

    cache_path = cache_dir / f"{src.stem}_{mtime}.mp3"
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path

    try:
        result = subprocess.run(
            [
                config.FFMPEG_PATH, "-y",
                "-i", str(src),
                "-filter:a", "loudnorm=I=-23:TP=-2:LRA=7",
                "-c:a", "libmp3lame",
                "-b:a", "192k",
                "-ar", "44100",
                str(cache_path),
            ],
            capture_output=True,
            timeout=120,
        )
        if result.returncode != 0 or not cache_path.exists() or cache_path.stat().st_size == 0:
            log.warning(
                f"loudnorm failed for {src.name} (rc={result.returncode}), using raw track. "
                f"stderr: {result.stderr.decode('utf-8', errors='replace')[-400:]}"
            )
            if cache_path.exists():
                cache_path.unlink(missing_ok=True)
            return src
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        log.warning(f"loudnorm unavailable/timed out for {src.name}: {e}. Using raw track.")
        return src

    log.info(f"loudnorm: cached normalized copy of {src.name} → {cache_path.name}")
    return cache_path


def _build_music_bed(music_path: Path, total_duration: float, volume: float) -> AudioFileClip | None:
    """Load music, loop to cover total_duration, fade, scale volume."""
    normalized_path = _loudnorm_track(music_path)
    try:
        bed = AudioFileClip(str(normalized_path))
    except Exception as e:
        log.error(f"Failed to load music {normalized_path}: {e}")
        return None

    if bed.duration < total_duration:
        bed = bed.with_effects([AudioLoop(duration=total_duration)])
    else:
        bed = bed.with_duration(total_duration)

    fade = min(config.MUSIC_FADE_SECONDS, total_duration / 4)
    bed = bed.with_effects([
        MultiplyVolume(max(0.0, min(1.0, volume))),
        AudioFadeIn(fade),
        AudioFadeOut(fade),
    ])
    log.info(
        f"Music bed: {music_path.name} @ vol={volume:.2f}, "
        f"{total_duration:.1f}s with {fade:.1f}s fades"
    )
    return bed


def assemble_video(
    scenes: list[dict],
    project_dir: Path,
    output_filename: str = "final.mp4",
    crossfade: float = config.CROSSFADE_DURATION,
    project_id: str | None = None,
    music_track: str | None = None,
    music_volume: float | None = None,
) -> tuple[Path, float]:
    """Assemble final video from scenes with images and audio.

    music_track: filename inside data/music/ or an absolute path. Optional.
    music_volume: 0.0-1.0, defaults to config.MUSIC_DEFAULT_VOLUME.
    """
    output_path = project_dir / output_filename
    clips = []

    if project_id:
        _init_progress(project_id)

    try:
        kb_effects = config.KB_DIRECTIONS.copy()

        for i, scene in enumerate(scenes):
            audio_path = scene.get("audio_path")

            # Collect image paths — prefer image_paths list, fall back to single image_path
            image_paths = scene.get("image_paths") or []
            if not image_paths:
                single = scene.get("image_path")
                image_paths = [single] if single else []

            # Resolve and filter to existing images
            abs_images = []
            for ip in image_paths:
                abs_img = project_dir / ip
                if abs_img.exists():
                    abs_images.append(abs_img)
                else:
                    log.warning(f"Image not found: {abs_img}, skipping")

            if not abs_images:
                log.warning(f"Scene {scene.get('index', '?')} has no images, skipping")
                continue

            # Resolve audio
            abs_audio = project_dir / audio_path if audio_path else None

            if project_id:
                _update_progress(
                    project_id,
                    phase=f"preparing scene {i + 1}/{len(scenes)}",
                )

            # Determine total scene duration from audio or hint
            if abs_audio and abs_audio.exists():
                audio_clip = AudioFileClip(str(abs_audio))
                scene_duration = audio_clip.duration
            else:
                scene_duration = scene.get("audio_duration", scene.get("duration_hint", 10.0))
                audio_clip = None

            # Split duration across images
            num_images = len(abs_images)
            per_image_duration = scene_duration / num_images

            scene_clips = []
            for img_idx, abs_img in enumerate(abs_images):
                if num_images > 1:
                    effect = kb_effects[img_idx % len(kb_effects)]
                else:
                    effect = scene.get("kb_effect", random.choice(kb_effects))

                clip = _ken_burns_clip(
                    image_path=str(abs_img),
                    duration=per_image_duration,
                    effect=effect,
                )
                scene_clips.append(clip)

            # Concatenate sub-clips for this scene
            if len(scene_clips) == 1:
                scene_video = scene_clips[0]
            else:
                scene_video = concatenate_videoclips(scene_clips, method="compose")

            if audio_clip:
                # Check for per-scene music override
                scene_music_track = scene.get("music_track")
                if scene_music_track:
                    scene_music_path = _resolve_music_path(scene_music_track)
                    if scene_music_path is not None:
                        scene_vol = scene.get("music_volume")
                        vol = scene_vol if scene_vol is not None else (music_volume if music_volume is not None else config.MUSIC_DEFAULT_VOLUME)
                        bed = _build_music_bed(scene_music_path, scene_duration, vol)
                        if bed is not None:
                            scene_audio = CompositeAudioClip([audio_clip, bed])
                            scene_video = scene_video.with_audio(scene_audio)
                        else:
                            scene_video = scene_video.with_audio(audio_clip)
                    else:
                        scene_video = scene_video.with_audio(audio_clip)
                else:
                    scene_video = scene_video.with_audio(audio_clip)
            elif scene.get("music_track"):
                # Scene has music but no voice audio — just use the music bed
                scene_music_path = _resolve_music_path(scene.get("music_track"))
                if scene_music_path is not None:
                    scene_vol = scene.get("music_volume")
                    vol = scene_vol if scene_vol is not None else (music_volume if music_volume is not None else config.MUSIC_DEFAULT_VOLUME)
                    bed = _build_music_bed(scene_music_path, scene_duration, vol)
                    if bed is not None:
                        scene_video = scene_video.with_audio(bed)

            clips.append(scene_video)

        if not clips:
            raise RuntimeError("No valid scenes to assemble")

        if project_id:
            _update_progress(project_id, phase="encoding", progress=0)

        # Concatenate with crossfade
        if crossfade > 0 and len(clips) > 1:
            final = concatenate_videoclips(clips, method="compose", padding=-crossfade)
        else:
            final = concatenate_videoclips(clips, method="compose")

        final = final.with_fps(config.VIDEO_FPS)

        # Mix in global background music bed only if no per-scene music was used
        any_scene_music = any(s.get("music_track") for s in scenes)
        music_path = _resolve_music_path(music_track)
        if music_path is not None and not any_scene_music:
            vol = music_volume if music_volume is not None else config.MUSIC_DEFAULT_VOLUME
            bed = _build_music_bed(music_path, final.duration, vol)
            if bed is not None:
                if final.audio is not None:
                    composite = CompositeAudioClip([final.audio, bed])
                else:
                    composite = bed
                final = final.with_audio(composite)

        # Build logger for progress tracking
        progress_logger = None
        if project_id:
            progress_logger = _AssemblyProgressLogger(project_id)

        log.info(f"Rendering video to {output_path} ({final.duration:.1f}s)")
        final.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            fps=config.VIDEO_FPS,
            preset="fast",
            threads=0,  # auto-detect all CPU cores
            logger=progress_logger or "bar",
        )

        # Capture duration before cleanup
        total_duration = round(final.duration, 2)

        # Clean up
        final.close()
        for c in clips:
            c.close()

        log.info(f"Video assembled: {output_path} ({total_duration}s)")

        if project_id:
            _finish_progress(project_id)

        return output_path, total_duration

    except Exception as e:
        if project_id:
            _finish_progress(project_id, error=str(e))
        raise
