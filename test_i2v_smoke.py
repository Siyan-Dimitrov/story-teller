"""I2V smoke test — calls generate_animatediff_clip on a handful of
existing chapter images so you can eyeball what Wan 2.6 I2V Flash produces
on your actual Flux-LoRA artwork.

Run:
    set REPLICATE_API_TOKEN=r8_...      (or sourced from .env via config.py)
    py test_i2v_smoke.py

Outputs land in test_output/i2v_smoke/<timestamp>/, one subdir per clip:
    scene_NN_preset/
        frame_0001.png ... frame_NNNN.png    (extracted frames)
        preview.mp4                            (frames re-assembled at 16fps for easy viewing)
        prompt.txt                             (the exact motion prompt sent)

Cost: ~3 clips × $0.25 ≈ $0.75 on wan-video/wan2.6-i2v-flash @ 720p / 5s.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("test_i2v_smoke")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend import config  # noqa: E402
from backend.animatediff_gen import generate_animatediff_clip, I2V_PRESETS  # noqa: E402

CHAPTER_PROJECT = PROJECT_ROOT / "projects" / "111a81b57579"  # Yamata-no-Orochi chapter

# (scene_index, image_index, motion_preset) — chosen to span the three preset bands
SAMPLES: list[tuple[int, int, str]] = [
    (1, 0, "animatediff_subtle"),     # serpent coiled in misty valleys — slow breathing, mist drift
    (9, 0, "animatediff_moderate"),   # princess grinding cockle-shell, sparks flying — hands & sparks
    (13, 0, "animatediff_dramatic"),  # burning moor with circling flames — strong fire motion
]


def _frames_to_mp4(clip_dir: Path, out_mp4: Path, fps: int) -> bool:
    pattern = str(clip_dir / "frame_%04d.png")
    try:
        result = subprocess.run(
            [
                config.FFMPEG_PATH, "-y",
                "-framerate", str(fps),
                "-i", pattern,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "18",
                str(out_mp4),
            ],
            capture_output=True,
            timeout=60,
        )
        if result.returncode != 0:
            log.warning(
                f"ffmpeg preview assembly failed (rc={result.returncode}): "
                f"{result.stderr.decode('utf-8', errors='replace')[-300:]}"
            )
            return False
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        log.warning(f"ffmpeg unavailable for preview: {e}")
        return False


async def main() -> int:
    if not config.REPLICATE_API_TOKEN:
        log.error("REPLICATE_API_TOKEN is not set — aborting.")
        return 1

    script_path = CHAPTER_PROJECT / "script.json"
    if not script_path.exists():
        log.error(f"Source project not found: {script_path}")
        return 1

    with script_path.open(encoding="utf-8") as f:
        script = json.load(f)
    scenes = script["scenes"]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = PROJECT_ROOT / "test_output" / "i2v_smoke" / ts
    out_root.mkdir(parents=True, exist_ok=True)

    log.info(f"Output dir: {out_root}")
    log.info(f"Model: {config.REPLICATE_I2V_MODEL}")
    log.info(f"Duration: {config.I2V_DURATION_SECONDS}s @ {config.I2V_RESOLUTION}")

    results: list[dict] = []
    for scene_idx, img_idx, preset in SAMPLES:
        scene = scenes[scene_idx]
        prompts = scene.get("image_prompts") or [scene.get("image_prompt", "")]
        scene_prompt = prompts[img_idx] if img_idx < len(prompts) else prompts[0]

        image_paths = scene.get("image_paths") or []
        rel = image_paths[img_idx] if img_idx < len(image_paths) else None
        if not rel:
            log.error(f"Scene {scene_idx} img {img_idx} has no image_path in script.json")
            continue
        src_image = CHAPTER_PROJECT / rel
        if not src_image.exists():
            log.error(f"Source image missing: {src_image}")
            continue

        sub_out = out_root / f"scene_{scene_idx:02d}_{preset.replace('animatediff_', '')}"
        sub_out.mkdir(parents=True, exist_ok=True)

        # Copy source image alongside the output for easy before/after comparison.
        (sub_out / "_source.png").write_bytes(src_image.read_bytes())

        log.info(
            f"=== Scene {scene_idx} img {img_idx} | preset={preset} ==="
            f"\n    prompt: {scene_prompt[:140]}..."
        )

        clip_dir = await generate_animatediff_clip(
            image_path=src_image,
            prompt=scene_prompt,
            output_dir=sub_out.parent,
            scene_index=scene_idx,
            img_index=img_idx,
            motion_preset=preset,
            style_prompt="dark fairy tale, gothic storybook art, atmospheric, moody",
        )

        if clip_dir is None:
            log.error(f"Scene {scene_idx}: clip generation FAILED")
            results.append({"scene": scene_idx, "preset": preset, "status": "failed"})
            continue

        # generate_animatediff_clip writes to output_dir / f"animatediff_s{scene:04d}_i{img}",
        # not to our `sub_out`. Move frames in so the folder layout is intuitive for the user.
        frames = sorted(clip_dir.glob("frame_*.png"))
        for fr in frames:
            target = sub_out / fr.name
            fr.rename(target)
        # Clean up empty source dir
        try:
            clip_dir.rmdir()
        except OSError:
            pass

        # Record the prompt that was sent for reference.
        motion_desc = I2V_PRESETS[preset]["motion"]
        (sub_out / "prompt.txt").write_text(
            f"preset: {preset}\n"
            f"motion description: {motion_desc}\n"
            f"\n"
            f"scene prompt: {scene_prompt}\n",
            encoding="utf-8",
        )

        # Reassemble frames into a viewable MP4.
        preview_mp4 = sub_out / "preview.mp4"
        if _frames_to_mp4(sub_out, preview_mp4, fps=config.I2V_OUTPUT_FPS):
            log.info(f"    preview: {preview_mp4.relative_to(PROJECT_ROOT)}")

        results.append({
            "scene": scene_idx,
            "preset": preset,
            "status": "ok",
            "frames": len(frames),
            "dir": str(sub_out.relative_to(PROJECT_ROOT)),
        })

    log.info("=" * 60)
    log.info("Summary:")
    for r in results:
        log.info(f"  {r}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
