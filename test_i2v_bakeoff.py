"""I2V model bake-off for full-motion anime shots (feature/anime).

Renders the same two anime stills — a dialogue close-up (facial acting) and
an action shot (body + background motion) — on five Replicate I2V models.
Compare test_output/i2v_bakeoff/<model>__<shot>.mp4 for: character motion
quality, cel-shade style preservation, artifacting, and cost/speed.

Approx published pricing per ~5s clip (verify against invoice):
  kling-v2.1 standard 720p   ~$0.25   (current pipeline baseline)
  seedance-1-lite 720p       ~$0.09
  wan-2.2-i2v-fast 720p      ~$0.05
  hailuo-2.3-fast 768p (6s)  ~$0.15
  pixverse-v5 720p           ~$0.30

Usage: venv/Scripts/python test_i2v_bakeoff.py [--models kling,seedance,...]
"""

import argparse
import sys
import time
from pathlib import Path

import httpx
import replicate

sys.path.insert(0, str(Path(__file__).parent))
from backend.animatediff_gen import _image_to_data_url  # noqa: E402

OUT_DIR = Path("test_output/i2v_bakeoff")

SHOTS = {
    "dialogue": {
        "image": Path("projects/a782473584ef/images/scene_0003_img_1.png"),
        "prompt": (
            "The young emperor's face trembles with shock, eyes widening, lips "
            "parting as he speaks a desperate line; candle flames flicker; the "
            "dark orbs drift slowly around the throne. 2D anime motion, cel "
            "shaded, subtle camera push-in."
        ),
    },
    "action": {
        "image": Path("projects/a782473584ef/images/scene_0002_img_1.png"),
        "prompt": (
            "The merchant lord shouts and thrusts his pointing arm forward, "
            "robes and sleeves swinging; broken brass automatons spark and "
            "twitch in the background. Dynamic 2D anime action motion, cel "
            "shaded, slight camera shake."
        ),
    },
}

MODELS = {
    "kling": ("kwaivgi/kling-v2.1", lambda img, p: {
        "start_image": img, "prompt": p, "duration": 5, "mode": "standard"}),
    "seedance": ("bytedance/seedance-1-lite", lambda img, p: {
        "image": img, "prompt": p, "duration": 5, "resolution": "720p"}),
    "wan": ("wan-video/wan-2.2-i2v-fast", lambda img, p: {
        "image": img, "prompt": p, "resolution": "720p"}),
    "hailuo": ("minimax/hailuo-2.3-fast", lambda img, p: {
        "first_frame_image": img, "prompt": p, "duration": 6, "resolution": "768p"}),
    "pixverse": ("pixverse/pixverse-v5", lambda img, p: {
        "image": img, "prompt": p, "duration": 5, "quality": "720p"}),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    args = ap.parse_args()
    picks = [m.strip() for m in args.models.split(",") if m.strip() in MODELS]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for shot_name, shot in SHOTS.items():
        img_url = _image_to_data_url(shot["image"])
        for key in picks:
            slug, build = MODELS[key]
            out_path = OUT_DIR / f"{key}__{shot_name}.mp4"
            print(f"--- {key} / {shot_name} ({slug})")
            t0 = time.time()
            try:
                output = replicate.run(slug, input=build(img_url, shot["prompt"]))
                url = str(output[0]) if isinstance(output, list) else str(output)
                resp = httpx.get(url, timeout=300, follow_redirects=True)
                resp.raise_for_status()
                out_path.write_bytes(resp.content)
                elapsed = time.time() - t0
                results.append((key, shot_name, elapsed, len(resp.content) // 1024))
                print(f"    ok: {elapsed:.0f}s gen, {len(resp.content)//1024} KB -> {out_path.name}")
            except Exception as e:  # noqa: BLE001
                elapsed = time.time() - t0
                results.append((key, shot_name, elapsed, None))
                print(f"    FAILED after {elapsed:.0f}s: {str(e)[:300]}")

    print("\n=== Summary ===")
    for key, shot, elapsed, kb in results:
        status = f"{kb} KB" if kb else "FAILED"
        print(f"{key:9s} {shot:9s} {elapsed:5.0f}s  {status}")


if __name__ == "__main__":
    main()
