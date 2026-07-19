"""Word-timed captions for Shorts.

Produces per-word timings for a scene's narration and groups them into short
karaoke chunks (a few words each) so burned-in captions can highlight the word
being spoken. Two timing engines:

  • "whisper"  — forced alignment: faster-whisper transcribes the narration WAV
    with word timestamps, and those timings are mapped back onto the script's
    own words (the displayed text is always the script text, never the
    transcription). Results are cached in a sidecar JSON next to the WAV.
  • "estimate" — dependency-free fallback: length-proportional allocation with
    an extra pause weight after sentence-final punctuation, approximating the
    narrator's sentence gaps.

The estimate path is used automatically whenever whisper or its model is
unavailable, or the alignment looks degenerate (e.g. localized narration whose
audio doesn't match the caption text).
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import re
import threading
import unicodedata
from pathlib import Path

from . import config

log = logging.getLogger(__name__)

# Sentence-final punctuation (possibly followed by closing quotes/brackets).
_SENT_END = re.compile(r"[.!?…]+[\"'”’)\]]*$")

_MIN_WORD_WEIGHT = 3.0
# Extra weight on a sentence-final word to model the narrator's pause after it.
_PAUSE_WEIGHT = 5.0
# A caption chunk stays up at most this long after its last word ends.
_CHUNK_HOLD = 0.6


def _normalize(tok: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", tok.lower()) if c.isalnum()
    )


def tokenize(narration: str) -> list[str]:
    """Whitespace tokens with pure-punctuation tokens glued onto the previous
    word, so an em-dash never becomes a caption 'word' of its own."""
    out: list[str] = []
    for tok in narration.split():
        if _normalize(tok):
            out.append(tok)
        elif out:
            out[-1] += tok
    return out


# ── estimate engine ─────────────────────────────────────────────

def _estimate_word_timings(tokens: list[str], speech_dur: float) -> list[dict]:
    weights = []
    for tok in tokens:
        w = max(float(len(_normalize(tok))), _MIN_WORD_WEIGHT)
        if _SENT_END.search(tok):
            w += _PAUSE_WEIGHT
        weights.append(w)
    total = sum(weights) or 1.0
    words: list[dict] = []
    t = 0.0
    for tok, w in zip(tokens, weights):
        d = speech_dur * (w / total)
        words.append({"word": tok, "start": round(t, 3), "end": round(t + d, 3)})
        t += d
    return words


# ── whisper engine ──────────────────────────────────────────────

_model = None
_model_lock = threading.Lock()


def _get_model():
    global _model
    with _model_lock:
        if _model is None:
            from faster_whisper import WhisperModel

            log.info(
                "Loading faster-whisper model %r (first use; downloads on first run)",
                config.SHORT_CAPTION_WHISPER_MODEL,
            )
            _model = WhisperModel(
                config.SHORT_CAPTION_WHISPER_MODEL, device="cpu", compute_type="int8"
            )
        return _model


def _whisper_words(audio_path: Path) -> list[dict]:
    model = _get_model()
    segments, _info = model.transcribe(
        str(audio_path), word_timestamps=True, vad_filter=True
    )
    out: list[dict] = []
    for seg in segments:
        for w in seg.words or []:
            token = (w.word or "").strip()
            if token:
                out.append({"word": token, "start": float(w.start), "end": float(w.end)})
    return out


def _align_to_script(
    tokens: list[str], whisper_words: list[dict], speech_dur: float
) -> list[dict] | None:
    """Map whisper word timings onto the script tokens. Matched tokens take the
    transcribed word's timing; gaps are interpolated by character weight between
    anchors. Returns None when too little matches to trust (caller falls back)."""
    if not tokens or not whisper_words:
        return None
    a = [_normalize(t) for t in tokens]
    b = [_normalize(w["word"]) for w in whisper_words]
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)

    starts: list[float | None] = [None] * len(tokens)
    ends: list[float | None] = [None] * len(tokens)
    matched = 0
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            i, j = blk.a + k, blk.b + k
            starts[i] = whisper_words[j]["start"]
            ends[i] = whisper_words[j]["end"]
            matched += 1
    if matched / len(tokens) < 0.5:
        return None

    # Fill unmatched runs by spreading the anchor-to-anchor interval across the
    # run's tokens proportionally to their length.
    def _weight(tok: str) -> float:
        return max(float(len(_normalize(tok))), _MIN_WORD_WEIGHT)

    i = 0
    while i < len(tokens):
        if starts[i] is not None:
            i += 1
            continue
        run_start = i
        while i < len(tokens) and starts[i] is None:
            i += 1
        run_end = i  # exclusive
        left = ends[run_start - 1] if run_start > 0 else 0.0
        right = starts[run_end] if run_end < len(tokens) else max(
            speech_dur, (ends[run_start - 1] or 0.0) + 0.3
        )
        span = max(0.12, float(right) - float(left))
        ws = [_weight(tokens[k]) for k in range(run_start, run_end)]
        total = sum(ws) or 1.0
        t = float(left)
        for k, w in zip(range(run_start, run_end), ws):
            d = span * (w / total)
            starts[k] = t
            ends[k] = t + d
            t += d

    # Enforce monotonic, positive-duration words.
    words: list[dict] = []
    cursor = 0.0
    for tok, s, e in zip(tokens, starts, ends):
        s = max(float(s), cursor)
        e = max(float(e), s + 0.06)
        words.append({"word": tok, "start": round(s, 3), "end": round(e, 3)})
        cursor = s
    return words


# ── sidecar cache ───────────────────────────────────────────────

def _cache_path(audio_path: Path) -> Path:
    return audio_path.with_suffix(".words.json")


def _narration_key(narration: str) -> str:
    return hashlib.sha1(narration.encode("utf-8")).hexdigest()


def _load_cache(audio_path: Path, narration: str) -> list[dict] | None:
    p = _cache_path(audio_path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("narration_sha1") == _narration_key(narration):
            return data.get("words") or None
    except Exception:  # noqa: BLE001
        pass
    return None


def _save_cache(audio_path: Path, narration: str, words: list[dict]) -> None:
    try:
        _cache_path(audio_path).write_text(
            json.dumps(
                {"narration_sha1": _narration_key(narration), "engine": "whisper",
                 "words": words},
                indent=1,
            ),
            encoding="utf-8",
        )
    except Exception as e:  # noqa: BLE001
        log.warning("Could not write caption timing cache: %s", e)


# ── public API ──────────────────────────────────────────────────

def get_word_timings(
    narration: str, audio_path: Path | None, speech_dur: float
) -> list[dict]:
    """Per-word [{word, start, end}] for a narration, best engine available."""
    tokens = tokenize(narration)
    if not tokens or speech_dur <= 0:
        return []
    if (
        config.SHORT_CAPTION_ALIGNMENT == "whisper"
        and audio_path is not None
        and Path(audio_path).exists()
    ):
        cached = _load_cache(Path(audio_path), narration)
        if cached:
            return cached
        try:
            aligned = _align_to_script(tokens, _whisper_words(Path(audio_path)), speech_dur)
            if aligned:
                _save_cache(Path(audio_path), narration, aligned)
                return aligned
            log.warning(
                "Whisper alignment didn't match caption text for %s — using estimate",
                audio_path,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("Whisper alignment unavailable (%s) — using estimated timings", e)
    return _estimate_word_timings(tokens, speech_dur)


def _make_chunk(cur: list[dict]) -> dict:
    return {
        "words": [w["word"] for w in cur],
        "starts": [w["start"] for w in cur],
        "start": cur[0]["start"],
        "end": cur[-1]["end"],
    }


def build_chunks(
    words: list[dict],
    max_words: int | None = None,
    max_chars: int | None = None,
) -> list[dict]:
    """Group word timings into caption chunks: break at sentence ends, at
    ``max_words``, or when the joined text exceeds ``max_chars``. Each chunk
    holds until the next one starts (capped) so captions don't flicker off
    during short pauses."""
    max_words = max_words or config.SHORT_CAPTION_MAX_WORDS
    max_chars = max_chars or config.SHORT_CAPTION_MAX_CHARS

    chunks: list[dict] = []
    cur: list[dict] = []
    for w in words:
        if cur:
            joined = len(" ".join(x["word"] for x in cur)) + 1 + len(w["word"])
            if len(cur) >= max_words or joined > max_chars:
                chunks.append(_make_chunk(cur))
                cur = []
        cur.append(w)
        if _SENT_END.search(w["word"]):
            chunks.append(_make_chunk(cur))
            cur = []
    if cur:
        chunks.append(_make_chunk(cur))

    for i, c in enumerate(chunks):
        limit = chunks[i + 1]["start"] if i + 1 < len(chunks) else c["end"] + _CHUNK_HOLD
        c["end"] = max(c["end"], min(limit, c["end"] + _CHUNK_HOLD))
    return chunks
