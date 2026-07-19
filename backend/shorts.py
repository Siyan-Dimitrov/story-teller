"""Vertical 9:16 Shorts renderer for Story Teller.

A short is a self-contained mini-scene from a story, reframed to portrait
(1080x1920) with:
  • the scene's existing still images, cover-cropped to vertical with a slow
    Ken Burns move (so we reuse what's already rendered — no new image spend);
  • a soft cinematic vignette;
  • burned-in captions, one band per spoken sentence (85% of Shorts views are
    muted, so the text carries the story);
  • an optional headline/hook card up top;
  • an end card driving viewers to the full video.

Audio is the scene's existing narration .wav — no extra TTS calls.

This module is self-contained (its own font/overlay helpers) so it doesn't
couple to the long-form video_assembly internals.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont

from moviepy import (
    VideoClip,
    VideoFileClip,
    AudioFileClip,
    CompositeVideoClip,
    concatenate_videoclips,
)

from . import captions, config

log = logging.getLogger(__name__)

SW = config.SHORT_WIDTH
SH = config.SHORT_HEIGHT
FPS = config.SHORT_FPS

try:
    from .voice_gen import SCENE_TRAILING_SILENCE as _TRAILING_SILENCE
except Exception:  # noqa: BLE001
    _TRAILING_SILENCE = 0.7

# Cinematic default palette (story tone, not the punchy facts-yellow).
ACCENT = "#E8C26A"      # warm candlelight gold
TINT = "#0B0B10"        # near-black blue tint for the vignette


# ── small drawing helpers ───────────────────────────────────────

def _load_font(size: int, *, bold: bool = True) -> ImageFont.FreeTypeFont:
    candidates = (
        ["georgiab.ttf", "Georgia Bold.ttf", "arialbd.ttf", "Arial Bold.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"]
        if bold else
        ["georgia.ttf", "arial.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"]
    )
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        test = (current + " " + word).strip()
        if draw.textbbox((0, 0), test, font=font)[2] > max_width and current:
            lines.append(current)
            current = word
        else:
            current = test
    if current:
        lines.append(current)
    return lines


def _ease_out_cubic(t: float) -> float:
    return 1.0 - (1.0 - t) ** 3


# ── vignette overlay ────────────────────────────────────────────

def _vignette_overlay(tint_hex: str = TINT, tint_alpha: int = 40,
                      vignette_alpha: int = 120) -> np.ndarray:
    overlay = Image.new("RGBA", (SW, SH), (0, 0, 0, 0))
    overlay = Image.alpha_composite(
        overlay, Image.new("RGBA", (SW, SH), ImageColor.getrgb(tint_hex) + (tint_alpha,))
    )
    mask = Image.new("L", (SW, SH), 0)
    md = ImageDraw.Draw(mask)
    cx, cy = SW // 2, SH // 2
    max_r = int((cx ** 2 + cy ** 2) ** 0.5)
    rings = 24
    for i in range(rings):
        r = int(max_r * (i + 1) / rings)
        a = int(vignette_alpha * (i / rings) ** 2)
        md.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=255 - a)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=24))
    vignette = Image.new("RGBA", (SW, SH), (0, 0, 0, vignette_alpha))
    vignette.putalpha(Image.eval(mask, lambda v: 255 - v))
    overlay = Image.alpha_composite(overlay, vignette)
    return np.array(overlay)


def _static_overlay_clip(overlay_rgba: np.ndarray, duration: float) -> VideoClip:
    """A full-frame RGBA overlay as its own clip (composited on top)."""
    rgb = overlay_rgba[:, :, :3].copy()
    alpha = (overlay_rgba[:, :, 3] / 255.0).astype(np.float64)

    clip = VideoClip(lambda _t: rgb, duration=duration).with_fps(FPS)
    mask = VideoClip(lambda _t: alpha, duration=duration, is_mask=True).with_fps(FPS)
    return clip.with_mask(mask)


# ── image bed (vertical Ken Burns) ──────────────────────────────

def _ken_burns_vertical(image_path: Path, duration: float, effect: str) -> VideoClip:
    """Cover-crop an image to 9:16 and apply a slow zoom/pan."""
    scale = 1.18
    iw, ih = int(SW * scale), int(SH * scale)

    pil = Image.open(str(image_path)).convert("RGB")
    # Cover-resize to (iw, ih): scale so it covers, then center-crop.
    src_w, src_h = pil.size
    cover = max(iw / src_w, ih / src_h)
    rw, rh = int(round(src_w * cover)), int(round(src_h * cover))
    pil = pil.resize((rw, rh), Image.LANCZOS)
    left = (rw - iw) // 2
    top = (rh - ih) // 2
    pil = pil.crop((left, top, left + iw, top + ih))
    src = np.array(pil)

    def crop_resize(x1, y1, cw, ch):
        x1 = max(0, min(x1, iw - cw))
        y1 = max(0, min(y1, ih - ch))
        region = src[y1:y1 + ch, x1:x1 + cw]
        if (cw, ch) == (SW, SH):
            return region
        return np.array(Image.fromarray(region).resize((SW, SH), Image.BILINEAR))

    if effect == "zoom_out":
        def make_frame(t):
            p = t / duration if duration > 0 else 0
            zoom = 1.12 - p * 0.12
            cw, ch = int(SW / zoom), int(SH / zoom)
            return crop_resize((iw - cw) // 2, (ih - ch) // 2, cw, ch)
    elif effect == "pan_up":
        max_pan = ih - SH
        def make_frame(t):
            p = t / duration if duration > 0 else 0
            return crop_resize((iw - SW) // 2, int(max_pan * (1 - p)), SW, SH)
    elif effect == "pan_down":
        max_pan = ih - SH
        def make_frame(t):
            p = t / duration if duration > 0 else 0
            return crop_resize((iw - SW) // 2, int(max_pan * p), SW, SH)
    else:  # zoom_in (default)
        def make_frame(t):
            p = t / duration if duration > 0 else 0
            zoom = 1.0 + p * 0.12
            cw, ch = int(SW / zoom), int(SH / zoom)
            return crop_resize((iw - cw) // 2, (ih - ch) // 2, cw, ch)

    return VideoClip(make_frame, duration=duration).with_fps(FPS)


def _build_image_bed(image_paths: list[Path], total_dur: float) -> VideoClip:
    """Sequence the scene's images across total_dur with alternating moves."""
    valid = [p for p in image_paths if p and Path(p).exists()]
    if not valid:
        # Solid dark frame fallback so a short still renders.
        frame = np.zeros((SH, SW, 3), dtype=np.uint8)
        return VideoClip(lambda _t: frame, duration=total_dur).with_fps(FPS)

    per = max(1.5, total_dur / len(valid))
    effects = ["zoom_in", "pan_up", "zoom_out", "pan_down"]
    clips = []
    acc = 0.0
    for i, p in enumerate(valid):
        d = min(per, total_dur - acc)
        if d <= 0.05:
            break
        clips.append(_ken_burns_vertical(p, d, effects[i % len(effects)]))
        acc += d
    bed = concatenate_videoclips(clips) if len(clips) > 1 else clips[0]
    # Guard exact duration.
    if abs(bed.duration - total_dur) > 0.05:
        bed = bed.with_duration(total_dur)
    return bed


# ── text cards ──────────────────────────────────────────────────

def _headline_clip(text: str, duration: float) -> VideoClip:
    img = Image.new("RGBA", (SW, SH), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = _load_font(70, bold=True)
    lines = _wrap_text(draw, text, font, int(SW * 0.84))[:4]
    line_h = int(70 * 1.2)
    block_h = line_h * len(lines)
    body_w = max((draw.textbbox((0, 0), ln, font=font)[2] for ln in lines), default=0)

    pad_x, pad_y = 46, 44
    panel_w = min(SW - 60, body_w + pad_x * 2)
    panel_h = block_h + pad_y * 2
    panel_x = (SW - panel_w) // 2
    panel_y = int(SH * 0.20) - panel_h // 2
    draw.rounded_rectangle(
        [(panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h)],
        radius=28, fill=(8, 8, 12, 190),
    )
    accent = ImageColor.getrgb(ACCENT) + (255,)
    draw.rounded_rectangle(
        [(panel_x + 22, panel_y + 16), (panel_x + panel_w - 22, panel_y + 22)],
        radius=3, fill=accent,
    )
    cur_y = panel_y + pad_y + 12
    for ln in lines:
        bbox = draw.textbbox((0, 0), ln, font=font)
        x = panel_x + (panel_w - (bbox[2] - bbox[0])) // 2
        draw.text((x + 3, cur_y + 3), ln, font=font, fill=(0, 0, 0, 210))
        draw.text((x, cur_y), ln, font=font, fill=(245, 240, 230, 255))
        cur_y += line_h
    return _static_overlay_clip(np.array(img), duration)


def _load_caption_font(size: int) -> ImageFont.FreeTypeFont:
    """Heavy condensed sans for karaoke captions (Impact-style). The serif
    Georgia stays on the hook/CTA cards; captions need punch and legibility."""
    for c in ("impact.ttf", "ariblk.ttf", "arialbd.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(c, size)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def _karaoke_chunk_layers(chunk: dict) -> list[VideoClip]:
    """One overlay clip per word-state of a caption chunk: the whole chunk text
    is visible for the chunk's interval, with the currently spoken word in the
    accent color. Word states tile the interval exactly (no gaps/overlap)."""
    display = [w.upper() for w in chunk["words"]] if config.SHORT_CAPTION_UPPERCASE \
        else list(chunk["words"])
    stroke = config.SHORT_CAPTION_STROKE
    size = config.SHORT_CAPTION_FONT_SIZE
    max_w = int(SW * 0.90)

    probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    text_line = " ".join(display)
    font = _load_caption_font(size)
    while size > 30 and probe.textbbox((0, 0), text_line, font=font,
                                       stroke_width=stroke)[2] > max_w:
        size -= 4
        font = _load_caption_font(size)

    space_w = probe.textlength(" ", font=font)
    widths = [probe.textlength(d, font=font) for d in display]
    total_w = sum(widths) + space_w * (len(display) - 1)
    x0 = (SW - total_w) / 2
    y = int(SH * config.SHORT_CAPTION_Y) - size // 2
    active = ImageColor.getrgb(config.SHORT_CAPTION_ACTIVE_COLOR) + (255,)

    bounds = list(chunk["starts"]) + [chunk["end"]]
    clips: list[VideoClip] = []
    for ai in range(len(display)):
        img = Image.new("RGBA", (SW, SH), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        x = x0
        for wi, txt in enumerate(display):
            fill = active if wi == ai else (255, 255, 255, 255)
            d.text((x, y), txt, font=font, fill=fill,
                   stroke_width=stroke, stroke_fill=(0, 0, 0, 235))
            x += widths[wi] + space_w
        start, end = bounds[ai], bounds[ai + 1]
        dur = max(0.06, end - start)
        clips.append(_static_overlay_clip(np.array(img), dur).with_start(start))
    return clips


def _build_caption_layers(scene: dict, project_dir: Path, body_dur: float) -> list[VideoClip]:
    """Karaoke captions for a scene: word timings (whisper-aligned when
    available, estimated otherwise) grouped into few-word chunks with the
    spoken word highlighted."""
    narration = (scene.get("narration") or "").strip()
    if not narration or body_dur <= 0:
        return []
    audio_rel = scene.get("audio_path")
    audio_path = (project_dir / audio_rel) if audio_rel else None
    speech_dur = max(1.0, body_dur - _TRAILING_SILENCE)

    words = captions.get_word_timings(narration, audio_path, speech_dur)
    if not words:
        return []
    layers: list[VideoClip] = []
    for chunk in captions.build_chunks(words):
        if chunk["start"] >= body_dur - 0.1:
            break
        if chunk["end"] > body_dur:
            chunk["end"] = body_dur
            chunk["starts"] = [min(s, body_dur - 0.06) for s in chunk["starts"]]
        layers.extend(_karaoke_chunk_layers(chunk))
    return layers


def _cta_clip(text: str, duration: float) -> VideoClip:
    img = Image.new("RGBA", (SW, SH), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = _load_font(52, bold=True)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y = 40, 24
    panel_w = tw + pad_x * 2
    panel_h = th + pad_y * 2 + 6
    panel_x = (SW - panel_w) // 2
    panel_y = int(SH * 0.5) - panel_h // 2
    accent = ImageColor.getrgb(ACCENT) + (255,)
    draw.rounded_rectangle(
        [(panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h)],
        radius=panel_h // 2, fill=accent,
    )
    tx = panel_x + pad_x
    ty = panel_y + pad_y - bbox[1]
    draw.text((tx, ty), text, font=font, fill=(15, 15, 20, 255))

    base = np.array(img)
    rgb = base[:, :, :3].copy()
    alpha = (base[:, :, 3] / 255.0).astype(np.float64)

    def make_mask(t):
        if t < 0.5:
            return alpha * _ease_out_cubic(t / 0.5)
        return alpha

    clip = VideoClip(lambda _t: rgb, duration=duration).with_fps(FPS)
    mask = VideoClip(make_mask, duration=duration, is_mask=True).with_fps(FPS)
    return clip.with_mask(mask)


# ── clip-from-final (cheap: reuse the finished video) ──────────

def compute_scene_time_ranges(
    scenes: list[dict],
    crossfade: float = config.CROSSFADE_DURATION,
) -> dict[int, tuple[float, float]]:
    """Map scene index -> (start, end) seconds in the assembled final.mp4.

    Mirrors ``video_assembly.assemble_video``: only scenes that have at least
    one image are placed on the timeline, each spans its narration duration,
    and consecutive scenes overlap by ``crossfade`` (concatenated with
    padding=-crossfade).
    """
    ranges: dict[int, tuple[float, float]] = {}
    cursor = 0.0
    for sc in scenes:
        imgs = sc.get("image_paths") or ([sc.get("image_path")] if sc.get("image_path") else [])
        if not [p for p in imgs if p]:
            continue  # skipped by the assembler too
        dur = sc.get("audio_duration") or sc.get("duration_hint") or 10.0
        start = cursor
        end = start + dur
        ranges[sc.get("index")] = (start, end)
        cursor = end - crossfade
    return ranges


def _ffmpeg_reframe(final_path: Path, start: float, duration: float,
                    out_path: Path, mode: str) -> None:
    """Cut [start, start+duration] from final_path and reframe to 9:16."""
    if mode == "crop":
        vf = f"[0:v]scale={SW}:{SH}:force_original_aspect_ratio=increase,crop={SW}:{SH},setsar=1[v]"
    else:  # "fit" — whole frame over a blurred fill
        vf = (
            f"[0:v]split=2[bg][fg];"
            f"[bg]scale={SW}:{SH}:force_original_aspect_ratio=increase,"
            f"crop={SW}:{SH},boxblur=30:2[bgb];"
            f"[fg]scale={SW}:-2:force_original_aspect_ratio=decrease[fgs];"
            f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,setsar=1[v]"
        )
    cmd = [
        config.FFMPEG_PATH, "-y",
        "-ss", f"{start:.3f}", "-i", str(final_path), "-t", f"{duration:.3f}",
        "-filter_complex", vf, "-map", "[v]", "-map", "0:a?",
        "-r", str(FPS), "-c:v", "libx264", "-preset", "veryfast",
        "-c:a", "aac", "-movflags", "+faststart", str(out_path),
    ]
    log.info("ffmpeg reframe: cut %.2f-%.2fs of %s", start, start + duration, final_path.name)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg reframe failed: {proc.stderr[-500:]}")


def render_short_from_final(
    scene: dict,
    scenes: list[dict],
    project_dir: Path,
    output_path: Path,
    hook: str | None = None,
    cta_text: str = "Watch the full story",
    reframe: str | None = None,
) -> tuple[Path, float]:
    """Cut this scene's segment out of final.mp4, reframe to 9:16, and overlay
    captions + hook + CTA. Keeps the music and motion already in the video.
    """
    final_path = project_dir / "final.mp4"
    if not final_path.exists():
        raise FileNotFoundError("final.mp4 not found — assemble the video first")

    ranges = compute_scene_time_ranges(scenes)
    idx = scene.get("index")
    if idx not in ranges:
        raise RuntimeError(f"Scene {idx} is not on the final timeline (no images?)")
    start, end = ranges[idx]
    duration = max(0.5, end - start)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".reframe.mp4")

    base = None
    layers: list = []
    final_clip = None
    try:
        _ffmpeg_reframe(final_path, start, duration, tmp, reframe or config.SHORT_REFRAME)

        base = VideoFileClip(str(tmp))
        dur = base.duration
        layers.append(base)

        headline = (hook or "").strip()
        if headline:
            layers.append(_headline_clip(headline, min(dur, max(2.5, dur * 0.4))))
        layers.extend(_build_caption_layers(scene, project_dir, dur))
        if cta_text:
            cta_start = max(0.0, dur - config.SHORT_CTA_LEAD)
            layers.append(_cta_clip(cta_text, dur - cta_start).with_start(cta_start))

        composite = CompositeVideoClip(layers, size=(SW, SH)).with_duration(dur)
        if base.audio is not None:
            composite = composite.with_audio(base.audio)
        final_clip = composite.with_fps(FPS)

        log.info("Rendering short (from final) for scene %s (%.1fs) -> %s",
                 idx, dur, output_path.name)
        final_clip.write_videofile(
            str(output_path),
            codec=config.SHORT_VIDEO_CODEC,
            audio_codec=config.SHORT_AUDIO_CODEC,
            fps=FPS, preset="veryfast", threads=0, logger=None,
        )
        return output_path, round(dur, 2)
    finally:
        for c in [final_clip, *layers]:
            try:
                if c is not None:
                    c.close()
            except Exception:  # noqa: BLE001
                pass
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:  # noqa: BLE001
            pass


# ── dispatcher ──────────────────────────────────────────────────

def render_short(
    scene: dict,
    project_dir: Path,
    output_path: Path,
    hook: str | None = None,
    cta_text: str = "Watch the full story",
    scenes: list[dict] | None = None,
    source: str | None = None,
) -> tuple[Path, float]:
    """Render one vertical short.

    ``source`` (default ``config.SHORT_SOURCE``): "final" cuts from final.mp4
    (cheap, keeps music + motion); "portrait" renders from the scene's native
    9:16 Nano Banana frames (``portrait_image_paths``, generated by the
    caller); anything else re-renders from the scene's stills. When portrait
    frames are missing, the final-cut path (blur-fill) is preferred over
    hard-cropping the landscape stills."""
    want = source or config.SHORT_SOURCE
    final_exists = (project_dir / "final.mp4").exists()
    has_portrait = bool(scene.get("portrait_image_paths"))
    try_final = want == "final" or (want == "portrait" and not has_portrait)
    if try_final and final_exists and scenes is not None:
        try:
            return render_short_from_final(
                scene, scenes, project_dir, output_path, hook=hook, cta_text=cta_text,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("Clip-from-final failed (%s) — falling back to stills", e)
    return render_short_from_stills(
        scene, project_dir, output_path, hook=hook, cta_text=cta_text,
    )


# ── stills renderer (fallback / no final.mp4) ──────────────────

def render_short_from_stills(
    scene: dict,
    project_dir: Path,
    output_path: Path,
    hook: str | None = None,
    cta_text: str = "Watch the full story",
) -> tuple[Path, float]:
    """Render one vertical short from a scene's images. Returns (path, duration).

    Requires the scene to have a generated ``audio_path`` and ``image_paths``.
    Native 9:16 frames (``portrait_image_paths``) are preferred when present —
    the cover-crop then keeps the full composition instead of cutting the sides.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio_rel = scene.get("audio_path")
    if not audio_rel:
        raise RuntimeError(f"Scene {scene.get('index')} has no narration audio")
    audio_path = project_dir / audio_rel
    if not audio_path.exists():
        raise RuntimeError(f"Narration audio missing on disk: {audio_path}")

    image_rels = scene.get("portrait_image_paths") or scene.get("image_paths") or []
    image_paths = [project_dir / p for p in image_rels]

    source_audio = []
    layers: list = []
    final = None
    try:
        audio = AudioFileClip(str(audio_path))
        source_audio.append(audio)
        body_dur = audio.duration
        clip_dur = max(config.SHORT_MIN_DURATION, body_dur + config.SHORT_TAIL_DURATION)
        if clip_dur > config.SHORT_MAX_DURATION:
            log.warning(
                "Scene %s narration (%.1fs) exceeds max short length; trimming",
                scene.get("index"), body_dur,
            )
            body_dur = config.SHORT_MAX_DURATION - config.SHORT_TAIL_DURATION
            clip_dur = config.SHORT_MAX_DURATION
            audio = audio.subclipped(0, body_dur)

        # Visual bed runs the whole duration (body + tail).
        bed = _build_image_bed(image_paths, clip_dur)
        layers.append(bed)
        layers.append(_static_overlay_clip(_vignette_overlay(), clip_dur))

        # Headline shows for the first part of the body.
        headline = (hook or "").strip()
        if headline:
            head_dur = min(body_dur, max(2.5, body_dur * 0.4))
            layers.append(_headline_clip(headline, head_dur))

        # Captions across the body.
        layers.extend(_build_caption_layers(scene, project_dir, body_dur))

        # CTA card over the closing seconds (through the silent tail).
        if cta_text:
            cta_start = max(0.0, clip_dur - config.SHORT_CTA_LEAD)
            layers.append(_cta_clip(cta_text, clip_dur - cta_start).with_start(cta_start))

        composite = CompositeVideoClip(layers, size=(SW, SH)).with_duration(clip_dur)
        composite = composite.with_audio(audio)
        final = composite.with_fps(FPS)

        log.info("Rendering short for scene %s (%.1fs) -> %s",
                 scene.get("index"), clip_dur, output_path.name)
        final.write_videofile(
            str(output_path),
            codec=config.SHORT_VIDEO_CODEC,
            audio_codec=config.SHORT_AUDIO_CODEC,
            fps=FPS,
            preset="veryfast",
            threads=0,
            logger=None,
        )
        return output_path, round(clip_dur, 2)
    finally:
        for c in [final, *layers, *source_audio]:
            try:
                if c is not None:
                    c.close()
            except Exception:  # noqa: BLE001
                pass
