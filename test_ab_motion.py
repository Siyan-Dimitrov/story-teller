"""A/B test: old ping-pong vs new forward-slowdown for I2V scene clips.

Generates ONE real I2V clip from an existing image, then renders the same
10s slot two ways and prints the source-frame timeline of each so the
difference (reverse/loop vs forward+slow) is explicit.

Run:  venv\Scripts\python.exe test_ab_motion.py
"""

import asyncio
import math
from pathlib import Path

from backend import config, animatediff_gen, video_assembly as va
from moviepy import ImageSequenceClip

IMG = config.PROJECTS_DIR / "024c18d63b97" / "images" / "scene_0003_img_0.png"
OUT = config.BASE_DIR / "test_output" / "ab"
OUT.mkdir(parents=True, exist_ok=True)
SLOT = 10.0  # simulate a 10s per-image slot (e.g. a 30s scene / 3 images)


def old_pingpong_clip(clip_dir: str, duration: float, target_size):
    """The PREVIOUS behaviour, reproduced for comparison."""
    frame_paths = [str(f) for f in sorted(Path(clip_dir).glob("frame_*.png"))]
    ad_fps = config.ANIMATEDIFF_DEFAULT_FPS
    pingpong = frame_paths + frame_paths[-2:0:-1]            # forward + reverse
    total = int(duration * ad_fps)
    repeated = []
    while len(repeated) < total:
        repeated.extend(pingpong)
    repeated = repeated[:total]
    clip = ImageSequenceClip(repeated, fps=ad_fps)
    if tuple(clip.size) != tuple(target_size):
        clip = clip.resized(new_size=target_size)
    if abs(clip.duration - duration) > 0.1:
        clip = clip.with_speed_scaled(clip.duration / duration)
    return clip.with_duration(duration).with_fps(config.VIDEO_FPS)


def frame_index_timeline(clip_dir: str, duration: float, method: str):
    """Which source frame index is shown at each second (0..duration)."""
    frames = sorted(Path(clip_dir).glob("frame_*.png"))
    n = len(frames)
    ad_fps = config.ANIMATEDIFF_DEFAULT_FPS
    if method == "old":
        pingpong = list(range(n)) + list(range(n - 2, 0, -1))
        total = int(duration * ad_fps)
        seq = (pingpong * (total // len(pingpong) + 1))[:total]
        # speed-scaled to exactly `duration`, so sample by fraction
        return [seq[min(len(seq) - 1, int((t / duration) * (len(seq) - 1)))]
                for t in range(int(duration) + 1)]
    else:
        slowdown = min(duration / (n / ad_fps), config.I2V_MAX_SLOWDOWN)
        playback_fps = ad_fps / slowdown
        total = max(1, math.ceil(duration * playback_fps))
        seq = (list(range(n)) * (total // n + 1))[:total]
        return [seq[min(len(seq) - 1, int(t * playback_fps))]
                for t in range(int(duration) + 1)]


async def main():
    print(f"Generating one I2V clip from {IMG.name} "
          f"(model={config.REPLICATE_I2V_MODEL}, {config.I2V_DURATION_SECONDS}s)...")
    clip_dir = await animatediff_gen.generate_animatediff_clip(
        image_path=IMG,
        prompt="an antique grandfather clock standing still in a dim workshop, faint dust drifting",
        output_dir=OUT,
        scene_index=3,
        img_index=0,
        style_prompt="storybook ink-and-watercolor, warm candlelight",
    )
    if not clip_dir:
        print("I2V generation FAILED")
        return
    frames = sorted(Path(clip_dir).glob("frame_*.png"))
    native = len(frames) / config.ANIMATEDIFF_DEFAULT_FPS
    print(f"clip: {len(frames)} frames @ {config.ANIMATEDIFF_DEFAULT_FPS}fps = {native:.1f}s native\n")

    size = (config.VIDEO_WIDTH, config.VIDEO_HEIGHT)

    new_clip = va._animatediff_clip(str(clip_dir), SLOT, size)
    new_clip.write_videofile(str(OUT / "new_forward.mp4"), codec="libx264",
                             fps=config.VIDEO_FPS, logger=None)
    new_clip.close()

    old_clip = old_pingpong_clip(str(clip_dir), SLOT, size)
    old_clip.write_videofile(str(OUT / "old_pingpong.mp4"), codec="libx264",
                             fps=config.VIDEO_FPS, logger=None)
    old_clip.close()

    print(f"\nSource-frame index shown at each second (slot={SLOT:.0f}s, native={native:.1f}s):")
    print("  t(s):  " + " ".join(f"{t:>3}" for t in range(int(SLOT) + 1)))
    print("  OLD :  " + " ".join(f"{i:>3}" for i in frame_index_timeline(str(clip_dir), SLOT, "old")))
    print("  NEW :  " + " ".join(f"{i:>3}" for i in frame_index_timeline(str(clip_dir), SLOT, "new")))
    print("\n(OLD rises then FALLS = rewind, then repeats. NEW rises monotonically "
          "= forward, slowed; small restart only for the tail.)")
    print("\nVideos:", OUT / "old_pingpong.mp4", "|", OUT / "new_forward.mp4")


if __name__ == "__main__":
    asyncio.run(main())
