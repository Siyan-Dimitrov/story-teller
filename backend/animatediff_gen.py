"""Image-to-video clip generation via Replicate.

Replaces the legacy ComfyUI/AnimateDiff path. Takes an existing still image
and asks a cloud I2V model (Wan 2.6 I2V Flash by default) to add subtle
motion — hair sway, cloth ripple, fire flicker, character breathing — while
preserving the input image's style and composition.

The output is rendered to MP4 by the model, then extracted to a directory of
PNG frames so `video_assembly._animatediff_clip` can consume it unchanged.
"""

import asyncio
import base64
import logging
import subprocess
import time
import uuid
from io import BytesIO
from pathlib import Path

import httpx

from . import config

log = logging.getLogger(__name__)

# ── Preset → motion-prompt mapping ──────────────────────────
# The LLM classifier in animation.py picks one of these preset names per
# image. Each preset translates to a short motion description appended to
# the scene's visual prompt before being sent to the I2V model.
I2V_PRESETS = {
    "animatediff_subtle": {
        "motion": "very gentle motion, subtle breathing, faint hair sway, soft cloth ripple, atmospheric haze drifting",
        "description": "Subtle ambient motion — for calm, intimate moments",
    },
    "animatediff_moderate": {
        "motion": "moderate motion, gestures, walking, flowing hair and clothing, candle flame flicker",
        "description": "Moderate motion — for active character scenes",
    },
    "animatediff_dramatic": {
        "motion": "strong motion, dramatic action, magic effects, swirling smoke and embers, fast flowing elements",
        "description": "Dramatic motion — for climactic or magical scenes",
    },
}

VALID_ANIMATEDIFF_MOTIONS = set(I2V_PRESETS.keys())

NEGATIVE_PROMPT = (
    "blurry, low quality, jpeg artifacts, deformed, ugly, "
    "static, no motion, frozen still image, watermark, text"
)


# ── Availability check ──────────────────────────────────────

_i2v_available: bool | None = None  # None = not yet tested


async def check_animatediff_available() -> bool:
    """Check whether I2V is configured. Kept as `check_animatediff_available`
    so existing callers in `animation.py` don't change.
    """
    global _i2v_available

    if _i2v_available is not None:
        return _i2v_available

    if not config.I2V_ENABLED:
        log.info("[I2V] Disabled via config.I2V_ENABLED")
        _i2v_available = False
        return False

    if not config.REPLICATE_API_TOKEN:
        log.warning("[I2V] REPLICATE_API_TOKEN not set — I2V unavailable")
        _i2v_available = False
        return False

    log.info(f"[I2V] Available via Replicate model: {config.REPLICATE_I2V_MODEL}")
    _i2v_available = True
    return True


def reset_availability():
    global _i2v_available
    _i2v_available = None


# ── Frame extraction ────────────────────────────────────────

def _extract_frames_from_mp4(mp4_path: Path, clip_dir: Path, fps: int) -> int:
    """Use ffmpeg to extract PNG frames at `fps` from an MP4 into clip_dir.

    Returns the number of frames written. Returns 0 on failure.
    """
    clip_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(clip_dir / "frame_%04d.png")

    try:
        result = subprocess.run(
            [
                config.FFMPEG_PATH, "-y",
                "-i", str(mp4_path),
                "-vf", f"fps={fps}",
                "-q:v", "2",
                pattern,
            ],
            capture_output=True,
            timeout=120,
        )
        if result.returncode != 0:
            log.error(
                f"[I2V] ffmpeg frame extraction failed (rc={result.returncode}): "
                f"{result.stderr.decode('utf-8', errors='replace')[-400:]}"
            )
            return 0
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        log.error(f"[I2V] ffmpeg unavailable/timed out: {e}")
        return 0

    frames = sorted(clip_dir.glob("frame_*.png"))
    return len(frames)


def _image_to_data_url(image_path: Path, max_side: int = 1280, max_bytes: int = 900_000) -> str:
    """Encode an image as a compact JPEG ``data:`` URL for the I2V `image` input.

    Replicate only accepts inline data URIs up to ~1 MB; a larger one makes the
    model container fail to fetch the image (the "HTTP Error. Checking again"
    loop) and return E006 "input was invalid". A raw 16:9 PNG is ~2.5 MB as a
    data URL, so we downscale the long edge to ``max_side`` and re-encode to
    JPEG, dropping quality until the payload fits under ``max_bytes``. JPEG also
    sidesteps Wan's rejection of PNGs with an alpha channel.
    """
    from io import BytesIO
    from PIL import Image as _PILImage

    img = _PILImage.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), _PILImage.LANCZOS)

    for quality in (90, 85, 80, 72, 65, 55):
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        data = buf.getvalue()
        if len(data) <= max_bytes:
            break

    b64 = base64.b64encode(data).decode("ascii")
    log.info(
        "[I2V] encoded %s -> JPEG data URL (%d px long edge, q=%d, %.0f KB)",
        image_path.name, max(img.size), quality, len(data) / 1024,
    )
    return f"data:image/jpeg;base64,{b64}"


def _extract_video_url(output) -> str | None:
    """Pull a usable video URL out of whatever shape ``replicate.run`` returns.

    replicate>=1.0 returns ``FileOutput`` objects (``str(x) == x.url``), but a
    model may return a scalar, a list, a dict (``{"video": ...}``), or an
    iterator. This normalises all of them to a single URL string.
    """
    # FileOutput and most scalars expose an explicit .url attribute.
    url = getattr(output, "url", None)
    if isinstance(url, str) and url:
        return url
    if isinstance(output, str):
        return output or None
    if isinstance(output, dict):
        for key in ("video", "output", "mp4", "url", "file"):
            if output.get(key):
                return _extract_video_url(output[key])
        return None
    if isinstance(output, (list, tuple)):
        return _extract_video_url(output[0]) if output else None
    # Last resort: an iterator/generator → take the first item.
    try:
        first = next(iter(output))
        if first is not output:
            return _extract_video_url(first)
    except (TypeError, StopIteration):
        pass
    s = str(output)
    return s if s.startswith(("http://", "https://", "data:")) else None


def _describe_exception(e: Exception) -> str:
    """Build a detailed, single-string description of a failed I2V call.

    Surfaces the server-side Replicate prediction error/logs (the real reason a
    generation failed) and any HTTP response status, which the bare exception
    message hides.
    """
    detail = f"{type(e).__name__}: {e}"

    # replicate.exceptions.ModelError carries the failed prediction, whose
    # `.error` and `.logs` hold the real server-side reason.
    prediction = getattr(e, "prediction", None)
    if prediction is not None:
        pstatus = getattr(prediction, "status", None)
        perr = getattr(prediction, "error", None)
        plogs = getattr(prediction, "logs", None)
        detail += f" | prediction.status={pstatus} error={perr!r}"
        if plogs:
            detail += f"\n--- replicate logs (tail) ---\n{str(plogs)[-1500:]}"

    resp = getattr(e, "response", None)
    if resp is not None:
        detail += f" | http_status={getattr(resp, 'status_code', '?')}"

    return detail


# ── Core generation function ────────────────────────────────

def _build_i2v_input(model: str, image_data_url: str, prompt: str, seed: int) -> dict:
    """Build the model-specific Replicate input dict for the configured I2V model.

    Different I2V models use different parameter names and enums, so the shape
    is selected by model family. ``image_data_url`` is a compact JPEG ``data:``
    URL (see ``_image_to_data_url``).
    """
    m = model.lower()
    duration = config.I2V_DURATION_SECONDS

    if "kling" in m:
        # kwaivgi/kling-v2.1: requires `start_image` + `prompt`. There is NO
        # seed / cfg_scale / aspect_ratio / resolution param — quality is set
        # via `mode` (standard=720p, pro=1080p) and duration must be 5 or 10.
        return {
            "start_image": image_data_url,
            "prompt": prompt,
            "negative_prompt": NEGATIVE_PROMPT,
            "duration": 10 if duration >= 10 else 5,
            "mode": "pro" if config.I2V_RESOLUTION == "1080p" else "standard",
        }

    # Default: wan-video/wan2.x family.
    return {
        "image": image_data_url,
        "prompt": prompt,
        "negative_prompt": NEGATIVE_PROMPT,
        "duration": duration,
        "resolution": config.I2V_RESOLUTION,
        "seed": seed,
        # The model defaults audio_enabled=true, whose post-generation audio
        # step has been observed to fail (E006). We add our own audio later.
        "audio_enabled": config.I2V_AUDIO_ENABLED,
        "enable_prompt_expansion": config.I2V_PROMPT_EXPANSION,
    }


def _build_motion_prompt(scene_prompt: str, motion_preset: str, style_prompt: str) -> str:
    """Build the I2V text prompt: style + scene + motion description."""
    preset = I2V_PRESETS.get(motion_preset, I2V_PRESETS["animatediff_subtle"])
    motion_desc = preset["motion"]
    parts = [p.strip() for p in (style_prompt, scene_prompt, motion_desc) if p and p.strip()]
    return ", ".join(parts)


async def generate_animatediff_clip(
    image_path: Path,
    prompt: str,
    output_dir: Path,
    scene_index: int,
    img_index: int,
    motion_preset: str = "animatediff_subtle",
    style_prompt: str = "dark fairy tale, gothic storybook art, atmospheric, moody",
) -> Path | None:
    """Generate a motion clip from a still image via Replicate I2V.

    Kept under its old name so `animation.py` and `video_assembly.py` don't
    need to be rewired. Returns the path to a directory of PNG frames, or
    None on failure (which the caller treats as a depthflow fallback).
    """
    import replicate as _replicate

    if not config.REPLICATE_API_TOKEN:
        log.error("[I2V] REPLICATE_API_TOKEN missing — cannot run I2V")
        return None

    motion_prompt = _build_motion_prompt(prompt, motion_preset, style_prompt)

    clip_dir = output_dir / f"animatediff_s{scene_index:04d}_i{img_index}"
    clip_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        f"[I2V] Scene {scene_index} img {img_index}: model={config.REPLICATE_I2V_MODEL}, "
        f"preset={motion_preset}, duration={config.I2V_DURATION_SECONDS}s, "
        f"resolution={config.I2V_RESOLUTION}"
    )

    try:
        seed = int(time.time() * 1000) % (2**32) + scene_index * 100 + img_index

        # Build a compact JPEG data URL: it carries an explicit MIME type (Wan
        # rejects uploaded files whose URL has no extension) AND stays under
        # Replicate's ~1 MB inline-data-URI limit (an oversized one causes the
        # container's image fetch to fail with E006 "input was invalid").
        data_url = _image_to_data_url(image_path)

        inp = _build_i2v_input(config.REPLICATE_I2V_MODEL, data_url, motion_prompt, seed)

        loop = asyncio.get_event_loop()
        output = await loop.run_in_executor(
            None,
            lambda: _replicate.run(config.REPLICATE_I2V_MODEL, input=inp),
        )

        video_url = _extract_video_url(output)
        if not video_url:
            raise RuntimeError(
                "could not extract a video URL from replicate output "
                f"(type={type(output).__name__}, repr={output!r}"[:300] + ")"
            )
        log.info(f"[I2V] Scene {scene_index} img {img_index}: downloading {video_url}")

        async with httpx.AsyncClient(timeout=config.I2V_TIMEOUT_SECONDS) as client:
            resp = await client.get(video_url)
            resp.raise_for_status()
            mp4_path = clip_dir / "source.mp4"
            mp4_path.write_bytes(resp.content)

        frame_count = _extract_frames_from_mp4(mp4_path, clip_dir, config.I2V_OUTPUT_FPS)
        if frame_count == 0:
            size = mp4_path.stat().st_size if mp4_path.exists() else 0
            log.error(
                "[I2V] Scene %s img %s: ffmpeg extracted 0 frames from %d-byte mp4 "
                "(kept %s for inspection)",
                scene_index, img_index, size, mp4_path.name,
            )
            generate_animatediff_clip.last_error = (
                f"ffmpeg extracted 0 frames from {size}-byte mp4"
            )
            return None

        # Delete the source mp4 to save disk only when configured to; otherwise
        # keep it so successful clips can still be inspected/debugged.
        if config.I2V_DELETE_SOURCE_MP4:
            try:
                mp4_path.unlink()
            except OSError:
                pass

        log.info(
            f"[I2V] Scene {scene_index} img {img_index}: {frame_count} frames saved to {clip_dir.name}"
        )
        generate_animatediff_clip.last_error = None
        return clip_dir

    except Exception as e:
        detail = _describe_exception(e)
        log.error(
            "[I2V] Scene %s img %s FAILED: %s",
            scene_index, img_index, detail, exc_info=True,
        )
        # Breadcrumb so the caller can surface why the clip fell back to parallax.
        generate_animatediff_clip.last_error = detail
        return None


async def generate_all_animatediff_clips(
    scenes: list[dict],
    project_dir: Path,
    style_prompt: str = "dark fairy tale, gothic storybook art, atmospheric, moody",
    progress_cb=None,
) -> list[dict]:
    """Generate I2V clips for all images classified as 'animatediff'.

    Updates scenes in place with `animatediff_clip_paths`. Falls back to
    depthflow for any image whose generation fails or that exceeds the
    per-project cap.
    """
    animatediff_dir = project_dir / "animatediff_clips"
    animatediff_dir.mkdir(exist_ok=True)

    targets: list[tuple[int, int, str, str]] = []  # (scene_idx, img_idx, rel_path, motion_preset)
    for si, scene in enumerate(scenes):
        anim_types = scene.get("animation_types") or []
        motion_presets = scene.get("motion_presets") or []
        image_paths = scene.get("image_paths") or []
        scene.setdefault("animatediff_clip_paths", [None] * len(image_paths))

        for img_idx, rel_path in enumerate(image_paths):
            anim_type = anim_types[img_idx] if img_idx < len(anim_types) else "depthflow"
            if anim_type != "animatediff":
                continue
            preset = (
                motion_presets[img_idx] if img_idx < len(motion_presets) else "animatediff_subtle"
            )
            targets.append((si, img_idx, rel_path, preset))

    if not targets:
        log.info("[I2V] No images classified as animatediff, skipping")
        return scenes

    # Apply budget cap: keep first N targets, downgrade the rest to depthflow.
    cap = config.I2V_MAX_CLIPS_PER_PROJECT
    if len(targets) > cap:
        log.warning(
            f"[I2V] {len(targets)} clips would exceed cap of {cap} per project — "
            f"downgrading the extras to depthflow"
        )
        for si, img_idx, _rel, _preset in targets[cap:]:
            anim_types = scenes[si].get("animation_types") or []
            motion_presets = scenes[si].get("motion_presets") or []
            if img_idx < len(anim_types):
                anim_types[img_idx] = "depthflow"
            if img_idx < len(motion_presets):
                motion_presets[img_idx] = "dolly_forward"
        targets = targets[:cap]

    total = len(targets)
    log.info(f"[I2V] Generating {total} clips (model={config.REPLICATE_I2V_MODEL})")
    done = 0

    for si, img_idx, rel_path, preset in targets:
        scene = scenes[si]
        idx = scene.get("index", si)
        abs_path = project_dir / rel_path

        if not abs_path.exists():
            log.warning(f"[I2V] Image not found: {abs_path}")
            done += 1
            continue

        if progress_cb:
            progress_cb(
                phase=f"I2V clip {done + 1}/{total}",
                progress=done / max(total, 1),
            )

        scene_prompt = (
            scene.get("image_prompts", [scene.get("image_prompt", "")])[img_idx]
            if img_idx < len(scene.get("image_prompts", []))
            else scene.get("image_prompt", "")
        )

        clip_dir = await generate_animatediff_clip(
            image_path=abs_path,
            prompt=scene_prompt,
            output_dir=animatediff_dir,
            scene_index=idx,
            img_index=img_idx,
            motion_preset=preset,
            style_prompt=style_prompt,
        )

        if clip_dir:
            rel_clip = str(clip_dir.relative_to(project_dir))
            while len(scene["animatediff_clip_paths"]) <= img_idx:
                scene["animatediff_clip_paths"].append(None)
            scene["animatediff_clip_paths"][img_idx] = rel_clip
        else:
            err = getattr(generate_animatediff_clip, "last_error", None) or "unknown error"
            log.warning(
                "[I2V] Scene %s img %s: generation failed (%s) — "
                "falling back to depth parallax", idx, img_idx, err,
            )
            # Surface to state JSON so the UI can show why I2V didn't run.
            errs = scene.setdefault("animatediff_errors", [None] * len(image_paths))
            while len(errs) <= img_idx:
                errs.append(None)
            errs[img_idx] = err
            anim_types = scene.get("animation_types") or []
            motion_presets = scene.get("motion_presets") or []
            if img_idx < len(anim_types):
                anim_types[img_idx] = "depthflow"
            if img_idx < len(motion_presets):
                motion_presets[img_idx] = "dolly_forward"

        done += 1

        # Throttle between Replicate calls to respect rate limits.
        if done < total:
            await asyncio.sleep(config.REPLICATE_DELAY_SECONDS)

    return scenes
