"""A/B test: MiniMax Speech 2.8-HD vs Chatterbox (both via Replicate) vs VoiceBox baseline.

Renders the same 3 scenes from The Six Swans on each provider, one request per
scene (no sentence chunking — both models handle full paragraphs), then
concatenates each provider's scenes with the same 0.7s inter-scene gap the
assembler uses. Listen to the *_concat.wav files in test_voice/ab/ and compare:
  - voice consistency across scene boundaries (the key criterion)
  - prosody / dark-storytelling delivery
  - clicks/thumps at joins
  - pacing (words per minute) vs VoiceBox's ~32% undershoot

Requires REPLICATE_API_TOKEN in the environment. Cost: well under $0.50 total.

Usage:  venv/Scripts/python test_tts_ab.py [--provider minimax|chatterbox|all]
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import replicate
import soundfile as sf

PROJECT = Path("projects/88f5421b4769")  # The Six Swans, 24 scenes
SCENE_INDICES = [0, 1, 2]
OUT_DIR = Path("test_voice/ab")
SCENE_GAP_SECONDS = 0.7

MINIMAX_MODEL = "minimax/speech-2.8-hd"
MINIMAX_VOICE = os.getenv("MINIMAX_VOICE", "English_Deep-VoicedGentleman")
CHATTERBOX_MODEL = "resemble-ai/chatterbox"


def clean_text(text: str) -> str:
    """Same punctuation normalization voice_gen.py applies for VoiceBox."""
    return (
        text.replace("…", ", ")
        .replace("...", ", ")
        .replace(" ", " ")
        .strip()
    )


def load_scenes() -> list[str]:
    script = json.loads((PROJECT / "script.json").read_text(encoding="utf-8"))
    return [clean_text(script["scenes"][i]["narration"]) for i in SCENE_INDICES]


def run_replicate(model: str, inp: dict, out_path: Path) -> float:
    """Run a Replicate TTS model, save output audio, return elapsed seconds."""
    t0 = time.time()
    output = replicate.run(model, input=inp)
    url = str(output[0]) if isinstance(output, list) else str(output)
    import httpx

    resp = httpx.get(url, timeout=120, follow_redirects=True)
    resp.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(resp.content)
    return time.time() - t0


def gen_minimax(text: str, out_path: Path) -> float:
    return run_replicate(
        MINIMAX_MODEL,
        {
            "text": text,
            "voice_id": MINIMAX_VOICE,
            "emotion": "auto",
            "audio_format": "wav",
            "sample_rate": 44100,
            "speed": 1.0,
        },
        out_path,
    )


def gen_chatterbox(text: str, out_path: Path) -> float:
    return run_replicate(
        CHATTERBOX_MODEL,
        {
            "prompt": text,
            "seed": 42,  # fixed seed for cross-scene voice stability
            "exaggeration": 0.6,
            "cfg_weight": 0.5,
            "temperature": 0.8,
        },
        out_path,
    )


def concat_wavs(paths: list[Path], out_path: Path) -> float:
    """Concatenate audio files with silence gaps. Returns total duration."""
    arrays, sr0 = [], None
    for p in paths:
        data, sr = sf.read(str(p))
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr0 is None:
            sr0 = sr
        elif sr != sr0:
            print(f"  ! sample rate mismatch {sr} vs {sr0} in {p.name}")
        arrays.append(data.astype(np.float32))
    gap = np.zeros(int(SCENE_GAP_SECONDS * sr0), dtype=np.float32)
    parts = []
    for i, a in enumerate(arrays):
        parts.append(a)
        if i < len(arrays) - 1:
            parts.append(gap)
    combined = np.concatenate(parts)
    sf.write(str(out_path), combined, sr0, format="WAV")
    return len(combined) / sr0


def report(name: str, scene_paths: list[Path], texts: list[str]) -> None:
    total_words = sum(len(t.split()) for t in texts)
    concat_path = OUT_DIR / f"{name}_concat.wav"
    dur = concat_wavs(scene_paths, concat_path)
    wpm = total_words / (dur / 60)
    print(f"{name:11s} concat={concat_path}  duration={dur:.1f}s  {wpm:.0f} wpm")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="all", choices=["minimax", "chatterbox", "all"])
    args = ap.parse_args()

    if not os.getenv("REPLICATE_API_TOKEN"):
        sys.exit("REPLICATE_API_TOKEN not set")

    texts = load_scenes()
    total_chars = sum(len(t) for t in texts)
    print(f"{len(texts)} scenes, {total_chars} chars total\n")

    providers = {
        "minimax": gen_minimax,
        "chatterbox": gen_chatterbox,
    }
    if args.provider != "all":
        providers = {args.provider: providers[args.provider]}

    results: dict[str, list[Path]] = {}
    for name, gen in providers.items():
        print(f"--- {name} ---")
        paths = []
        for i, text in zip(SCENE_INDICES, texts):
            out = OUT_DIR / name / f"scene_{i:04d}.wav"
            elapsed = gen(text, out)
            d, sr = sf.read(str(out))
            print(f"  scene {i}: {len(d)/sr:.1f}s audio in {elapsed:.1f}s ({sr} Hz)")
            paths.append(out)
        results[name] = paths

    print("\n=== Concatenated results (listen to these) ===")
    for name, paths in results.items():
        report(name, paths, texts)

    # VoiceBox baseline from the existing project audio
    baseline = [PROJECT / "audio" / f"scene_{i:04d}.wav" for i in SCENE_INDICES]
    if all(p.exists() for p in baseline):
        report("voicebox", baseline, texts)


if __name__ == "__main__":
    main()
