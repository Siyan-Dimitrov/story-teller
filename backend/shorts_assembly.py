"""Vertical 1080x1920 Shorts renderer for Story Teller (v2).

v1 critique findings that drove v2:
  - Headline pinned for the full body_dur stole contrast from captions and
    killed the image. v2: headline pops, holds 2s, then fades out.
  - Single landscape still + slow Ken Burns for 25s read as a held breath.
    v2: cuts between 2-4 distinct images every ~5s with a quick crossfade.
  - "Full story ▲" tail with up-arrow doesn't map to any real YouTube
    affordance. v2: plain "Subscribe" pill, no arrow.

Inputs to assemble_short:
  - image_paths: list of stills (the director picks 2-3 from neighboring
    candidate scenes for visual variety).
  - audio_path: a single hook narration WAV.
  - headline: 4-6 word concrete-noun headline (transient — gone after 2.7s).
  - narration: text used to derive burned-in captions.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont

from moviepy import (
    VideoClip,
    AudioFileClip,
    CompositeVideoClip,
    concatenate_videoclips,
)
from moviepy.audio.fx import MultiplyVolume

from . import config

log = logging.getLogger(__name__)


SHORT_WIDTH = 1080
SHORT_HEIGHT = 1920
SHORT_FPS = 30
SHORT_MIN_DURATION = 8.0
SHORT_MAX_DURATION = 58.0
SHORT_TAIL_DURATION = 2.5

# Transient headline timing (the v1 fix). The headline announces the hook
# in the first beat, then GETS OUT OF THE WAY so the image + captions can do
# their job.
HEADLINE_POP_DUR = 0.32
HEADLINE_HOLD_DUR = 2.0
HEADLINE_FADE_DUR = 0.4
HEADLINE_TOTAL_DUR = HEADLINE_POP_DUR + HEADLINE_HOLD_DUR + HEADLINE_FADE_DUR

# Image bed timing — cuts every BED_IMAGE_DUR seconds with a quick crossfade.
# 5s feels alive on mobile without seizure-cutting. Each image gets a light
# Ken Burns inside its segment to keep micro-motion alive on muted views.
BED_IMAGE_DUR = 5.0
BED_CROSSFADE_DUR = 0.35

HEADLINE_ANCHOR_Y_PCT = 0.32     # panel CENTRE — sits in upper third
CAPTION_ANCHOR_Y_PCT = 0.62      # captions sit mid-screen (image breathes below)


# ── Font + text helpers ─────────────────────────────────────────

def _load_font(size: int, *, bold: bool = True) -> ImageFont.ImageFont:
    candidates: list[str] = []
    if bold:
        candidates.extend([
            "arialbd.ttf",
            "Arial Bold.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/segoeuib.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ])
    else:
        candidates.extend([
            "arial.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ])
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        test = (current + " " + word).strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] > max_width and current:
            lines.append(current)
            current = word
        else:
            current = test
    if current:
        lines.append(current)
    return lines


def _ease_out_cubic(t: float) -> float:
    return 1.0 - (1.0 - t) ** 3


_SENT_SPLIT_RE = re.compile(r'(?:(?<=[.!?])|(?<=[.!?]["\'”’]))\s+')


def _split_sentences(text: str) -> list[str]:
    parts = _SENT_SPLIT_RE.split((text or "").strip())
    return [p.strip() for p in parts if p and p.strip()]


# ── Per-image Ken Burns inside the bed segment ──────────────────

def _kb_cover_clip(
    image_path: Path,
    duration: float,
    *,
    target_size: tuple[int, int] = (SHORT_WIDTH, SHORT_HEIGHT),
    zoom_from: float = 1.04,
    zoom_to: float = 1.14,
    drift_x: int = 0,
    drift_y: int = 0,
) -> VideoClip:
    """Cover-crop the image to portrait and pan/zoom slowly across the segment.

    Each image segment is short (~5s), so the Ken Burns range is tighter than
    v1 (1.04→1.14 instead of 1.05→1.18) — too much motion at 5s starts to feel
    twitchy. drift_x/drift_y allow tiny horizontal/vertical pan offsets so
    consecutive segments don't all look like centered zooms.
    """
    target_w, target_h = target_size
    pil = Image.open(image_path).convert("RGB")
    src_w, src_h = pil.size
    cover_scale = max(target_w / src_w, target_h / src_h) * zoom_to
    work_w = max(target_w, int(round(src_w * cover_scale)))
    work_h = max(target_h, int(round(src_h * cover_scale)))
    pil = pil.resize((work_w, work_h), Image.LANCZOS)
    work = np.array(pil)

    def make_frame(t):
        p = (t / duration) if duration > 0 else 0.0
        zoom = zoom_from + (zoom_to - zoom_from) * p
        cw = int(target_w / zoom * zoom_to)
        ch = int(target_h / zoom * zoom_to)
        cw = max(target_w, min(work_w, cw))
        ch = max(target_h, min(work_h, ch))
        x1 = (work_w - cw) // 2 + int(drift_x * (p - 0.5) * 2)
        y1 = (work_h - ch) // 2 + int(drift_y * (p - 0.5) * 2)
        x1 = max(0, min(work_w - cw, x1))
        y1 = max(0, min(work_h - ch, y1))
        crop = work[y1:y1 + ch, x1:x1 + cw]
        if (cw, ch) == (target_w, target_h):
            return crop
        return np.array(
            Image.fromarray(crop).resize((target_w, target_h), Image.BILINEAR)
        )

    return VideoClip(make_frame, duration=duration).with_fps(SHORT_FPS)


def _build_image_bed(image_paths: list[Path], total_dur: float) -> VideoClip:
    """Multi-image bed: cut between N images every BED_IMAGE_DUR with crossfade.

    If only one image is supplied, returns a single Ken Burns clip covering
    total_dur. Otherwise distributes total_dur across the available images,
    each with its own micro Ken Burns and a small crossfade overlap.
    """
    if len(image_paths) == 1:
        return _kb_cover_clip(image_paths[0], total_dur)

    # Slightly overlap segments so concatenate can crossfade.
    n = min(len(image_paths), max(2, int(round(total_dur / BED_IMAGE_DUR))))
    image_paths = image_paths[:n]
    seg_dur = total_dur / n + BED_CROSSFADE_DUR * (n - 1) / n
    # Alternate the drift direction so cuts don't all push the same way.
    drifts = [(0, 0), (40, 0), (-40, 0), (0, 30), (0, -30)]
    segments: list[VideoClip] = []
    for i, img in enumerate(image_paths):
        dx, dy = drifts[i % len(drifts)]
        seg = _kb_cover_clip(img, seg_dur, drift_x=dx, drift_y=dy)
        if i > 0:
            seg = seg.with_start(0).with_effects([]) if False else seg
        segments.append(seg)

    bed = concatenate_videoclips(
        segments,
        method="compose",
        padding=-BED_CROSSFADE_DUR,
    ).with_duration(total_dur).with_fps(SHORT_FPS)
    return bed


# ── Vignette + tint overlay ─────────────────────────────────────

def _build_vignette_overlay(
    size: tuple[int, int],
    tint_hex: str,
    *,
    tint_alpha: int = 35,
    vignette_alpha: int = 120,
) -> np.ndarray:
    w, h = size
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))

    tint_rgb = ImageColor.getrgb(tint_hex)
    fill = Image.new("RGBA", size, tint_rgb + (tint_alpha,))
    overlay = Image.alpha_composite(overlay, fill)

    mask = Image.new("L", size, 0)
    md = ImageDraw.Draw(mask)
    cx, cy = w // 2, h // 2
    max_r = int(((cx ** 2) + (cy ** 2)) ** 0.5)
    rings = 24
    for i in range(rings):
        r = int(max_r * (i + 1) / rings)
        a = int(vignette_alpha * (i / rings) ** 2)
        md.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=255 - a)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=24))
    vignette = Image.new("RGBA", size, (0, 0, 0, vignette_alpha))
    inv = Image.eval(mask, lambda v: 255 - v)
    vignette.putalpha(inv)
    overlay = Image.alpha_composite(overlay, vignette)

    return np.array(overlay)


def _apply_overlay_to_clip(footage_clip: VideoClip, overlay_rgba: np.ndarray) -> VideoClip:
    overlay_rgb = overlay_rgba[:, :, :3].astype(np.float32)
    overlay_alpha = (overlay_rgba[:, :, 3:4] / 255.0).astype(np.float32)

    def transform(get_frame, t):
        frame = get_frame(t).astype(np.float32)
        out = frame * (1.0 - overlay_alpha) + overlay_rgb * overlay_alpha
        return np.clip(out, 0, 255).astype(np.uint8)

    return footage_clip.transform(transform)


# ── Transient headline (the v1 fix) ─────────────────────────────

def _create_headline_frame(text: str, accent_hex: str = "#FFC857") -> np.ndarray:
    w, h = SHORT_WIDTH, SHORT_HEIGHT
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Bigger, fewer lines — v2 headlines are 4-6 words max, so we can go
    # bigger and constrain to 2 lines for a more poster-like read.
    text_font_size = 92
    font = _load_font(text_font_size, bold=True)
    accent = ImageColor.getrgb(accent_hex) + (255,)

    body_max_w = int(w * 0.86)
    lines = _wrap_text(draw, text, font, body_max_w)[:2]
    line_h = int(text_font_size * 1.15)
    body_h = line_h * len(lines)
    body_w = max(
        (draw.textbbox((0, 0), line, font=font)[2] for line in lines),
        default=0,
    )

    pad_x, pad_y = 56, 48
    panel_w = min(w - 60, body_w + pad_x * 2)
    panel_h = body_h + pad_y * 2
    panel_x = (w - panel_w) // 2
    panel_y = int(h * HEADLINE_ANCHOR_Y_PCT) - panel_h // 2

    draw.rounded_rectangle(
        [(panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h)],
        radius=36,
        fill=(8, 8, 12, 210),
    )
    # Accent stripe top of the panel
    draw.rounded_rectangle(
        [(panel_x + 28, panel_y + 18),
         (panel_x + panel_w - 28, panel_y + 28)],
        radius=4,
        fill=accent,
    )

    cur_y = panel_y + pad_y + 14
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        lw = bbox[2] - bbox[0]
        x = panel_x + (panel_w - lw) // 2
        draw.text((x + 3, cur_y + 3), line, font=font, fill=(0, 0, 0, 220))
        draw.text((x, cur_y), line, font=font, fill=(255, 255, 255, 255))
        cur_y += line_h

    return np.array(img)


def _create_headline_clip(text: str, body_dur: float, accent_hex: str) -> VideoClip:
    """v2: headline shows briefly then VANISHES.

    Timeline:
       0.00s ┄ 0.32s : scale-pop-in + alpha-ramp
       0.32s ┄ 2.32s : hold static at full alpha
       2.32s ┄ 2.72s : alpha fade-out
       2.72s ┄  body_dur : nothing (mask returns 0)

    Clip duration is body_dur so it remains in the composite, but it costs
    nothing after the fade — make_mask returns a zero-alpha array.
    """
    base_rgba = _create_headline_frame(text, accent_hex)
    base_rgb = base_rgba[:, :, :3].copy()
    base_alpha = (base_rgba[:, :, 3] / 255.0).astype(np.float64)
    zero_alpha = np.zeros_like(base_alpha)
    base_pil = Image.fromarray(base_rgba, mode="RGBA")
    size = (SHORT_WIDTH, SHORT_HEIGHT)

    def _scaled_rgba(scale: float) -> np.ndarray:
        if scale >= 0.999:
            return base_rgba
        new_w = max(1, int(size[0] * scale))
        new_h = max(1, int(size[1] * scale))
        scaled = base_pil.resize((new_w, new_h), Image.LANCZOS)
        canvas = Image.new("RGBA", size, (0, 0, 0, 0))
        canvas.paste(scaled, ((size[0] - new_w) // 2, (size[1] - new_h) // 2), scaled)
        return np.array(canvas)

    pop_end = HEADLINE_POP_DUR
    hold_end = pop_end + HEADLINE_HOLD_DUR
    fade_end = hold_end + HEADLINE_FADE_DUR

    def make_frame(t):
        if t < pop_end:
            scale = 0.92 + 0.08 * _ease_out_cubic(t / pop_end)
            return _scaled_rgba(scale)[:, :, :3]
        return base_rgb

    def make_mask(t):
        if t < pop_end:
            scale = 0.92 + 0.08 * _ease_out_cubic(t / pop_end)
            scaled = _scaled_rgba(scale)
            alpha = (scaled[:, :, 3] / 255.0).astype(np.float64)
            return alpha * _ease_out_cubic(t / pop_end)
        if t < hold_end:
            return base_alpha
        if t < fade_end:
            k = 1.0 - (t - hold_end) / HEADLINE_FADE_DUR
            return base_alpha * max(0.0, k)
        return zero_alpha

    clip = VideoClip(make_frame, duration=body_dur).with_fps(SHORT_FPS)
    mask = VideoClip(make_mask, duration=body_dur, is_mask=True).with_fps(SHORT_FPS)
    return clip.with_mask(mask)


# ── Burned-in captions ──────────────────────────────────────────

def _caption_clip(text: str, duration: float, accent_hex: str) -> VideoClip:
    w, h = SHORT_WIDTH, SHORT_HEIGHT
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font_size = 68
    font = _load_font(font_size, bold=True)
    accent = ImageColor.getrgb(accent_hex) + (255,)

    body_max_w = int(w * 0.84)
    lines = _wrap_text(draw, text, font, body_max_w)[:4]
    line_h = int(font_size * 1.20)
    block_h = line_h * len(lines)
    anchor_y = int(h * CAPTION_ANCHOR_Y_PCT)
    cur_y = anchor_y - block_h // 2

    stroke = 7
    stroke_color = (0, 0, 0, 245)
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        lw = bbox[2] - bbox[0]
        x = (w - lw) // 2
        for dx in (-stroke, 0, stroke):
            for dy in (-stroke, 0, stroke):
                if dx == 0 and dy == 0:
                    continue
                draw.text((x + dx, cur_y + dy), line, font=font, fill=stroke_color)
        fill = accent if i == 0 else (255, 255, 255, 255)
        draw.text((x, cur_y), line, font=font, fill=fill)
        cur_y += line_h

    base_rgba = np.array(img)
    base_rgb = base_rgba[:, :, :3].copy()
    base_alpha = (base_rgba[:, :, 3] / 255.0).astype(np.float64)

    def make_frame(_t):
        return base_rgb

    def make_mask(t):
        if t < 0.18:
            return base_alpha * (t / 0.18)
        if t > duration - 0.18:
            return base_alpha * max(0.0, (duration - t) / 0.18)
        return base_alpha

    clip = VideoClip(make_frame, duration=duration).with_fps(SHORT_FPS)
    mask = VideoClip(make_mask, duration=duration, is_mask=True).with_fps(SHORT_FPS)
    return clip.with_mask(mask)


def _build_caption_layers(
    narration: str,
    body_dur: float,
    accent_hex: str,
) -> list[VideoClip]:
    """Time-distribute sentence captions across body_dur.

    Note: in v2 we start captions AFTER the headline has popped (HEADLINE_POP_DUR)
    so they don't compete for attention during the first beat. The headline
    fades out by t ≈ 2.7s; captions start at t=0 with the first one anchored
    on the first sentence regardless. They'll briefly co-exist with the
    headline for ~2s by design — the headline is at 32% Y, captions at 62% Y,
    no vertical overlap.
    """
    if not narration or body_dur <= 0:
        return []

    sentences = _split_sentences(narration)
    if not sentences:
        return []

    total_chars = sum(len(s) for s in sentences)
    layers: list[VideoClip] = []
    cursor = 0.0
    min_chunk = 1.4
    for s in sentences:
        if cursor >= body_dur - 0.05:
            break
        share = (len(s) / total_chars) if total_chars else (1.0 / len(sentences))
        dur = max(min_chunk, body_dur * share)
        remaining = body_dur - cursor
        if dur > remaining:
            dur = remaining
            if dur < 0.4:
                break
        layers.append(_caption_clip(s, dur, accent_hex).with_start(cursor))
        cursor += dur
    return layers


# ── Subscribe tail ──────────────────────────────────────────────

def _create_subscribe_tail(
    duration: float,
    accent_hex: str,
    label: str = "Subscribe",
) -> VideoClip:
    """v2: plain 'Subscribe' pill, no up-arrow.

    v1 used 'Full story ▲' which suggested a swipe-up affordance that
    doesn't exist on YouTube Shorts. v2 just says Subscribe; the channel
    handles the conversion via the avatar tap.
    """
    w, h = SHORT_WIDTH, SHORT_HEIGHT
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = _load_font(56, bold=True)
    accent = ImageColor.getrgb(accent_hex) + (255,)

    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    pad_x, pad_y = 36, 24
    panel_w = tw + pad_x * 2
    panel_h = th + pad_y * 2 + 6
    panel_x = (w - panel_w) // 2  # centered — feels more like a CTA card than a corner pill
    panel_y = int(h * 0.42) - panel_h // 2

    draw.rounded_rectangle(
        [(panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h)],
        radius=panel_h // 2,
        fill=accent,
    )
    text_x = panel_x + pad_x
    text_y = panel_y + pad_y - bbox[1]
    draw.text((text_x + 2, text_y + 2), label, font=font, fill=(0, 0, 0, 200))
    draw.text((text_x, text_y), label, font=font, fill=(20, 20, 25, 255))

    base_rgba = np.array(img)
    base_rgb = base_rgba[:, :, :3].copy()
    base_alpha = (base_rgba[:, :, 3] / 255.0).astype(np.float64)

    def make_frame(_t):
        return base_rgb

    def make_mask(t):
        if t < 0.6:
            return base_alpha * _ease_out_cubic(t / 0.6)
        return base_alpha

    clip = VideoClip(make_frame, duration=duration).with_fps(SHORT_FPS)
    mask = VideoClip(make_mask, duration=duration, is_mask=True).with_fps(SHORT_FPS)
    return clip.with_mask(mask)


# ── Public assembler ────────────────────────────────────────────

def assemble_short(
    *,
    image_paths: list[Path],
    audio_path: Path,
    headline: str,
    narration: str,
    output_path: Path,
    accent_hex: str = "#FFC857",
    tint_hex: str = "#0D0D0D",
    voice_volume: Optional[float] = None,
) -> tuple[Path, float]:
    """Render one Short and return (path, duration_seconds).

    Args:
        image_paths: 1-N stills used as the visual bed. Multiple stills are
            cut between every BED_IMAGE_DUR seconds with a quick crossfade.
            Each gets a per-segment Ken Burns to keep micro-motion alive.
        audio_path: hook-narration WAV.
        headline: 4-6 word headline; shown only for ~2.7s at the start then
            fades out (v2 fix — v1 kept it pinned for 25s and killed the image).
        narration: full narration text used to derive burned-in captions.
        output_path: destination .mp4.
    """
    if not image_paths:
        raise ValueError("assemble_short: image_paths must contain at least one path")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_audio: list = []
    composite_layers: list = []
    final = None

    try:
        # ── Audio
        audio = AudioFileClip(str(audio_path))
        source_audio.append(audio)
        body_dur = audio.duration
        vol = config.VOICE_VOLUME if voice_volume is None else voice_volume
        if vol and vol != 1.0:
            audio = audio.with_effects([MultiplyVolume(vol)])
            source_audio.append(audio)

        clip_dur = max(SHORT_MIN_DURATION, body_dur + SHORT_TAIL_DURATION)
        if clip_dur > SHORT_MAX_DURATION:
            log.warning(
                f"Short narration ({body_dur:.1f}s) + tail ({SHORT_TAIL_DURATION}s) "
                f"exceeds {SHORT_MAX_DURATION}s; trimming narration."
            )
            body_dur = SHORT_MAX_DURATION - SHORT_TAIL_DURATION
            clip_dur = SHORT_MAX_DURATION
            audio = audio.subclipped(0, body_dur)
            source_audio.append(audio)

        # ── Visual bed: multi-image cuts (or single KB if only one image)
        bed = _build_image_bed(image_paths, clip_dur)
        overlay = _build_vignette_overlay(
            (SHORT_WIDTH, SHORT_HEIGHT),
            tint_hex,
            tint_alpha=35,
            vignette_alpha=120,
        )
        bed = _apply_overlay_to_clip(bed, overlay)
        composite_layers.append(bed)

        # ── Transient headline (gone after ~2.7s)
        if headline.strip():
            head_clip = _create_headline_clip(headline.strip(), body_dur, accent_hex)
            composite_layers.append(head_clip)

        # ── Burned-in captions
        composite_layers.extend(
            _build_caption_layers(narration, body_dur, accent_hex)
        )

        # ── Subscribe tail
        tail = _create_subscribe_tail(SHORT_TAIL_DURATION, accent_hex).with_start(body_dur)
        composite_layers.append(tail)

        composite = CompositeVideoClip(
            composite_layers,
            size=(SHORT_WIDTH, SHORT_HEIGHT),
        ).with_duration(clip_dur)
        composite = composite.with_audio(audio)
        final = composite.with_fps(SHORT_FPS)

        log.info(
            f"Rendering short ({clip_dur:.1f}s, "
            f"{len(image_paths)} bed image(s)) → {output_path.name}"
        )
        final.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            fps=SHORT_FPS,
            preset="veryfast",
            threads=0,
            logger=None,
        )
        return output_path, round(clip_dur, 2)

    finally:
        if final is not None:
            try:
                final.close()
            except Exception:
                pass
        for c in composite_layers:
            try:
                c.close()
            except Exception:
                pass
        for c in source_audio:
            try:
                c.close()
            except Exception:
                pass
