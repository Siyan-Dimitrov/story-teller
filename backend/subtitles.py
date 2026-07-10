"""English subtitles for non-English narration, burned into final.mp4.

Timing: each scene's window on the assembled timeline comes from
shorts.compute_scene_time_ranges (mirrors the assembler). Within a scene,
sentence boundaries are placed proportionally by localized-sentence length —
TTS speaking rate is near-uniform per character, so this tracks the audio
closely without needing word-level alignment.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from . import config
from .shorts import compute_scene_time_ranges
from .voice_gen import SCENE_TRAILING_SILENCE

log = logging.getLogger(__name__)

# Split subtitle cues longer than this into two lines (~standard 42 cpl).
MAX_CUE_CHARS = 84
MIN_CUE_SECONDS = 1.0

SUB_STYLE = (
    "FontName=Arial,FontSize=16,PrimaryColour=&H00FFFFFF,"
    "OutlineColour=&H00000000,Outline=2,Shadow=0,MarginV=36"
)


def _fmt_ts(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    h, rem = divmod(ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _wrap(text: str) -> str:
    if len(text) <= MAX_CUE_CHARS // 2:
        return text
    words = text.split()
    mid, best, acc = len(text) // 2, 0, 0
    for i, w in enumerate(words[:-1]):
        acc += len(w) + 1
        if abs(acc - mid) < abs(best - mid):
            best = acc
        if acc >= mid:
            return " ".join(words[: i + 1]) + "\n" + " ".join(words[i + 1:])
    return text


def build_srt(scenes: list[dict]) -> str | None:
    """Build SRT content from localized scenes. None if nothing to subtitle."""
    ranges = compute_scene_time_ranges(scenes)
    cues: list[tuple[float, float, str]] = []

    for sc in scenes:
        loc = sc.get("localized") or {}
        if not loc.get("lang") or loc.get("lang") == "en":
            continue
        window = ranges.get(sc.get("index"))
        if not window:
            continue
        start, end = window
        speech_end = max(start + 0.5, end - SCENE_TRAILING_SILENCE)

        sents_en = loc.get("sentences_en") or [sc.get("narration") or ""]
        sents_loc = loc.get("sentences") or []
        if len(sents_loc) != len(sents_en):
            sents_loc = sents_en  # fall back to weighting by the English text

        weights = [max(1, len(s)) for s in sents_loc]
        total = sum(weights)
        cursor = start
        for sent_en, w in zip(sents_en, weights):
            dur = (speech_end - start) * (w / total)
            cues.append((cursor, cursor + max(dur, MIN_CUE_SECONDS), sent_en))
            cursor += dur

    if not cues:
        return None

    lines: list[str] = []
    for i, (t0, t1, text) in enumerate(cues):
        if i + 1 < len(cues):
            t1 = min(t1, cues[i + 1][0])  # a MIN_CUE_SECONDS floor can overlap the next cue
        lines.append(str(i + 1))
        lines.append(f"{_fmt_ts(t0)} --> {_fmt_ts(t1)}")
        lines.append(_wrap(text))
        lines.append("")
    return "\n".join(lines)


def burn_subtitles(video_path: Path, scenes: list[dict]) -> bool:
    """Burn English subs into video_path in place if any scene is localized.

    Returns True if subtitles were burned.
    """
    srt = build_srt(scenes)
    if not srt:
        return False

    workdir = video_path.parent
    srt_path = workdir / "subs.srt"
    srt_path.write_text(srt, encoding="utf-8")
    tmp_path = workdir / f"{video_path.stem}_subbed.mp4"

    # Run from the project dir with relative paths — the subtitles filter
    # chokes on Windows drive-colon paths.
    cmd = [
        config.FFMPEG_PATH, "-y",
        "-i", video_path.name,
        "-vf", f"subtitles={srt_path.name}:force_style='{SUB_STYLE}'",
        "-c:v", "libx264", "-preset", "fast", "-c:a", "copy",
        "-movflags", "+faststart", tmp_path.name,
    ]
    log.info("Burning English subtitles into %s", video_path.name)
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(workdir))
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg subtitle burn failed: {proc.stderr[-500:]}")
    os.replace(tmp_path, video_path)
    return True
