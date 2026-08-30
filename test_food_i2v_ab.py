"""Food-reel I2V bake-off: which video model animates photoreal food stills best?

Generates 5 recipe "beats" (flourless potato buns) as 9:16 Nano Banana stills,
then animates every still with each candidate image-to-video model on
Replicate, and writes an HTML gallery for side-by-side review.

Run:
    venv\\Scripts\\python.exe test_food_i2v_ab.py
    venv\\Scripts\\python.exe test_food_i2v_ab.py --models seedance-2.0,veo-3.1-fast
    venv\\Scripts\\python.exe test_food_i2v_ab.py --stills test_output/food_i2v_ab/<ts>/stills

Outputs land in test_output/food_i2v_ab/<timestamp>/:
    stills/beat_N.png            the source frames
    clips/<model>/beat_N.mp4     one clip per (model, beat)
    results.json                 prediction ids, predict_time, errors
    index.html                   gallery: rows = beats, columns = models

Cost (approx, all 7 models x 5 beats): ~$30 on Replicate.
"""

from __future__ import annotations

import argparse
import asyncio
import html
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
log = logging.getLogger("food_i2v_ab")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend import config  # noqa: E402
from backend.image_gen import generate_image_nano_banana  # noqa: E402

STYLE = (
    "Photoreal food photography, bright soft natural window light, shallow depth "
    "of field, white ceramic bowl on a pale marble counter with a beige linen "
    "cloth, a sprig of parsley, vertical 9:16 phone composition, no text, no "
    "watermark, no captions"
)

# (still prompt, I2V motion prompt) per recipe beat — mirrors the reference reel.
BEATS: list[tuple[str, str]] = [
    (
        "Close-up: a woman's hand grating a peeled potato on a steel box grater "
        "into a white ceramic bowl, a small pile of fine potato shreds already in the bowl",
        "The hand slides the potato down the grater, fresh shreds fall into the bowl, "
        "slow gentle camera push-in, steady handheld feel",
    ),
    (
        "Close-up overhead: two bright egg yolks in a white ceramic bowl, a metal spoon "
        "held above adding a dollop of thick Greek yogurt onto the yolks",
        "The spoon tilts and the yogurt slides off onto the yolks, the yolks wobble "
        "slightly, subtle handheld camera movement",
    ),
    (
        "Close-up: a fork whisking egg yolks, yogurt and grated potato into a thick "
        "golden batter in a white ceramic bowl, a hand holding the bowl steady",
        "The whisk stirs in small circles, the batter folds over itself and thickens, "
        "camera holds steady with a very slight drift",
    ),
    (
        "Close-up: two hands shaping a round ball of pale dough on a parchment-lined "
        "baking tray, three more dough balls beside it, sesame seeds scattered on top",
        "The hands press and roll the dough ball smooth, then fingertips sprinkle sesame "
        "seeds that fall onto the buns, gentle camera tilt down",
    ),
    (
        "Close-up: a hand holding up one golden-brown sesame bun above a plate of "
        "freshly baked buns, glossy crust, soft focus background, faint steam",
        "The hand gently squeezes the bun and it springs back showing it is soft and "
        "fluffy, slow rotation toward the camera, faint steam rising",
    ),
]

MOTION_SUFFIX = (
    "Photoreal, natural light, realistic hands with five fingers, "
    "no text, no captions, no watermark."
)
NEGATIVE = (
    "text, captions, watermark, logo, extra fingers, deformed hands, "
    "morphing, blur, cartoon, illustration"
)


def _seedance(slug: str, resolution: str = "720p"):
    def build(img, prompt: str) -> dict:
        return {
            "image": img, "prompt": prompt, "duration": 5, "resolution": resolution,
            "aspect_ratio": "9:16", "generate_audio": False,
        }
    return slug, build


MODELS: dict[str, tuple] = {
    "seedance-2.0": _seedance("bytedance/seedance-2.0"),
    "seedance-2.0-fast": _seedance("bytedance/seedance-2.0-fast"),
    "seedance-2.5": _seedance("bytedance/seedance-2.5"),
    "seedance-2.5-480p": _seedance("bytedance/seedance-2.5", "480p"),
    "happyhorse-1.0": ("alibaba/happyhorse-1.0", lambda img, p: {
        "image": img, "prompt": p, "duration": 5, "resolution": "720p",
    }),
    "veo-3.1-fast": ("google/veo-3.1-fast", lambda img, p: {
        "image": img, "prompt": p, "duration": 6, "resolution": "720p",
        "aspect_ratio": "9:16", "generate_audio": False, "negative_prompt": NEGATIVE,
    }),
    "kling-v3": ("kwaivgi/kling-v3-video", lambda img, p: {
        "start_image": img, "prompt": p, "duration": 5, "mode": "standard",
        "generate_audio": False, "negative_prompt": NEGATIVE,
    }),
    "hailuo-2.3": ("minimax/hailuo-2.3", lambda img, p: {
        "first_frame_image": img, "prompt": p, "duration": 6, "resolution": "768p",
    }),
}


async def make_stills(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"beat_{i + 1}.png" for i in range(len(BEATS))]
    # Beat 1 first, then the rest anchored on it so props/lighting match.
    await generate_image_nano_banana(BEATS[0][0], STYLE, paths[0], aspect_ratio="9:16")
    await asyncio.gather(*[
        generate_image_nano_banana(
            BEATS[i][0], STYLE, paths[i], aspect_ratio="9:16",
            reference_images=[paths[0]], style_reference_only=True,
        )
        for i in range(1, len(BEATS))
    ])
    return paths


def _first_url(output) -> str | None:
    if isinstance(output, str):
        return output
    if isinstance(output, list) and output:
        return _first_url(output[0])
    if isinstance(output, dict):
        for k in ("video", "output", "url"):
            if output.get(k):
                return _first_url(output[k])
    return None


def run_one(model_key: str, beat_idx: int, still: Path, out_path: Path) -> dict:
    import replicate

    slug, build = MODELS[model_key]
    prompt = f"{BEATS[beat_idx][1]}. {MOTION_SUFFIX}"
    rec: dict = {"model": model_key, "slug": slug, "beat": beat_idx + 1, "prompt": prompt}
    t0 = time.time()
    try:
        for attempt in range(10):
            try:
                with open(still, "rb") as fh:
                    pred = replicate.predictions.create(model=slug, input=build(fh, prompt))
                break
            except replicate.exceptions.ReplicateError as e:
                # Low-credit accounts get a burst limit of 5 creations; back off and retry.
                if "429" not in str(e) or attempt == 9:
                    raise
                log.warning("[%s beat %d] 429 throttled, retrying in 8s", model_key, beat_idx + 1)
                time.sleep(8)
        rec["prediction_id"] = pred.id
        log.info("[%s beat %d] prediction %s started", model_key, beat_idx + 1, pred.id)
        pred.wait()
        rec["status"] = pred.status
        rec["predict_time"] = (pred.metrics or {}).get("predict_time")
        if pred.status != "succeeded":
            rec["error"] = str(pred.error or pred.status)
            log.error("[%s beat %d] %s: %s", model_key, beat_idx + 1, pred.status, rec["error"])
            return rec
        url = _first_url(pred.output)
        if not url:
            rec["error"] = f"no video url in output: {pred.output!r}"
            return rec
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with httpx.stream("GET", url, follow_redirects=True, timeout=300) as r:
            r.raise_for_status()
            with open(out_path, "wb") as fh:
                for chunk in r.iter_bytes():
                    fh.write(chunk)
        rec["path"] = out_path.relative_to(out_path.parents[2]).as_posix()
        log.info("[%s beat %d] done in %.0fs (predict %.0fs)", model_key, beat_idx + 1,
                 time.time() - t0, rec["predict_time"] or 0)
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
        log.error("[%s beat %d] failed: %s", model_key, beat_idx + 1, rec["error"])
    rec["wall_time"] = round(time.time() - t0, 1)
    return rec


def write_gallery(run_dir: Path, stills: list[Path], model_keys: list[str], results: list[dict]) -> Path:
    by_key = {(r["model"], r["beat"]): r for r in results}
    cols = "".join(f"<th>{html.escape(m)}</th>" for m in model_keys)
    rows = []
    for i, still in enumerate(stills):
        cells = [
            f'<td><img src="stills/{still.name}">'
            f'<div class="cap">{html.escape(BEATS[i][1])}</div></td>'
        ]
        for m in model_keys:
            r = by_key.get((m, i + 1), {})
            if r.get("path"):
                pt = r.get("predict_time")
                meta = f"{pt:.0f}s" if pt else ""
                cells.append(
                    f'<td><video src="{r["path"]}" controls muted loop playsinline '
                    f'preload="metadata"></video><div class="cap">{meta}</div></td>'
                )
            else:
                cells.append(f'<td class="err">{html.escape(r.get("error", "not run"))}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    doc = f"""<!doctype html><meta charset="utf-8"><title>Food I2V bake-off</title>
<style>
body{{font-family:system-ui;background:#111;color:#eee;margin:16px}}
table{{border-collapse:collapse}} th,td{{padding:6px;vertical-align:top;border:1px solid #333}}
th{{position:sticky;top:0;background:#222}}
img,video{{width:220px;aspect-ratio:9/16;object-fit:cover;background:#000;display:block}}
.cap{{width:220px;font-size:11px;color:#aaa;margin-top:4px}} .err{{color:#f66;font-size:12px;max-width:220px}}
button{{margin-bottom:12px}}
</style>
<h2>Food I2V bake-off — {html.escape(run_dir.name)}</h2>
<button onclick="document.querySelectorAll('video').forEach(v=>v.play())">Play all</button>
<button onclick="document.querySelectorAll('video').forEach(v=>v.pause())">Pause all</button>
<table><tr><th>still</th>{cols}</tr>{''.join(rows)}</table>
"""
    p = run_dir / "index.html"
    p.write_text(doc, encoding="utf-8")
    return p


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS),
                    help="comma-separated subset of: " + ", ".join(MODELS))
    ap.add_argument("--stills", type=Path, default=None, help="reuse stills from this directory")
    ap.add_argument("--beats", default=None, help="comma-separated beat numbers to run (default all)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="reuse this run dir: keep existing clips, only generate missing ones")
    args = ap.parse_args()

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in model_keys if m not in MODELS]
    if unknown:
        sys.exit(f"unknown models: {unknown}")
    if not config.REPLICATE_API_TOKEN:
        sys.exit("REPLICATE_API_TOKEN not set")

    prior: list[dict] = []
    if args.run_dir:
        run_dir = args.run_dir
        results_file = run_dir / "results.json"
        if results_file.exists():
            prior = [r for r in json.loads(results_file.read_text(encoding="utf-8"))
                     if r.get("path") and (run_dir / r["path"]).exists()]
        stills = sorted((run_dir / "stills").glob("beat_*.png"))
    elif args.stills:
        run_dir = (config.BASE_DIR / "test_output" / "food_i2v_ab"
                   / datetime.now().strftime("%Y%m%d_%H%M%S"))
        (run_dir / "stills").mkdir(parents=True)
        stills = []
        for s in sorted(args.stills.glob("beat_*.png")):
            dst = run_dir / "stills" / s.name
            dst.write_bytes(s.read_bytes())
            stills.append(dst)
    else:
        run_dir = (config.BASE_DIR / "test_output" / "food_i2v_ab"
                   / datetime.now().strftime("%Y%m%d_%H%M%S"))
        stills = await make_stills(run_dir / "stills")
    log.info("%d stills ready in %s", len(stills), run_dir / "stills")

    sem = asyncio.Semaphore(args.concurrency)

    async def job(m: str, i: int) -> dict:
        async with sem:
            return await asyncio.to_thread(
                run_one, m, i, stills[i], run_dir / "clips" / m / f"beat_{i + 1}.mp4"
            )

    have = {(r["model"], r["beat"]) for r in prior}
    beats = ([int(b) for b in args.beats.split(",")] if args.beats
             else list(range(1, len(stills) + 1)))
    todo = [(m, b - 1) for m in model_keys for b in beats if (m, b) not in have]
    log.info("%d clips already present, %d to generate", len(have), len(todo))
    results = prior + await asyncio.gather(*[job(m, i) for m, i in todo])
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    columns = [m for m in MODELS if any(r["model"] == m for r in results)]
    gallery = write_gallery(run_dir, stills, columns, results)

    ok = sum(1 for r in results if r.get("path"))
    log.info("done: %d/%d clips succeeded", ok, len(results))
    for m in model_keys:
        rs = [r for r in results if r["model"] == m]
        fails = [r for r in rs if not r.get("path")]
        times = [r["predict_time"] for r in rs if r.get("predict_time")]
        avg = sum(times) / len(times) if times else 0
        log.info("  %-18s ok=%d fail=%d avg_predict=%.0fs", m, len(rs) - len(fails), len(fails), avg)
    print(f"\nGALLERY: {gallery}")


if __name__ == "__main__":
    asyncio.run(main())
