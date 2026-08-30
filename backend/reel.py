"""Recipe reels: short vertical food videos built from a known recipe.

A reel is a project of kind "reel". It reuses the story pipeline where the
shapes match and adds one dedicated build job:

  1. ``generate_script`` — Claude writes 5–7 narrated beats (hook → steps →
     reveal + CTA) from the recipe text only.
  2. voice — the normal voice step narrates each beat (MiniMax).
  3. ``start_build`` — one native 9:16 Nano Banana still per beat (beat 1
     anchors the look of the rest), one Seedance I2V clip per still, then
     ``shorts.render_reel`` cuts the clips to the narration with karaoke
     captions, the hook headline and the CTA card.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import threading
import traceback
from pathlib import Path
from typing import Callable

from . import config, image_gen, llm, shorts
from .animatediff_gen import generate_animatediff_clip

log = logging.getLogger(__name__)

DEFAULT_STYLE = (
    "Photoreal food photography, bright soft natural window light, shallow depth "
    "of field, clean light kitchen, vertical 9:16 phone composition, no text, "
    "no watermark, no captions"
)
MOTION_SUFFIX = (
    "Photoreal, natural light, realistic hands with five fingers, "
    "no text, no captions, no watermark."
)
# Each still after the first is an edit of the previous clip's last frame, so
# the reel reads as one continuous take where every step adds to the picture.
CONTINUITY_DIRECTIVE = (
    " The reference image is the previous shot of this exact scene. Keep the "
    "same camera angle, framing, kitchen, counter, bowl, tray, utensils, "
    "lighting and hands. Change ONLY what this prompt describes — the food's "
    "new state and the action — so the image reads as the next moment of the "
    "same continuous take."
)


async def generate_script(
    recipe_text: str,
    target_seconds: float,
    claude_model: str | None = None,
) -> dict:
    """Write the reel script from the recipe. Returns a script.json-shaped dict."""
    recipe_text = (recipe_text or "").strip()
    if not recipe_text:
        raise ValueError("A recipe reel needs the recipe text (ingredients + steps)")
    system = (config.CLAUDE_PROMPTS_DIR / "reel_writer_system.md").read_text(encoding="utf-8")
    words = int(target_seconds * config.REEL_NARRATION_WPM / 60)
    user = (
        f"Target: about {target_seconds:.0f} seconds of narration ≈ {words} words "
        f"total across all beats.\n\nRECIPE:\n{recipe_text}"
    )
    raw = await llm.complete(
        system, user, model=claude_model or config.CLAUDE_MODEL, pass_name="reel_writer"
    )
    script = llm.parse_json(raw)
    scenes = script.get("scenes") or []
    if not scenes:
        raise ValueError("Reel writer returned no scenes")
    for i, sc in enumerate(scenes):
        sc["index"] = i
        sc["image_prompt"] = (sc.get("image_prompt") or "").strip()
        sc["image_prompts"] = [sc["image_prompt"]]
        sc["motion_prompt"] = (sc.get("motion_prompt") or "").strip()
        sc.setdefault("mood", "upbeat")
        n_words = len((sc.get("narration") or "").split())
        sc["duration_hint"] = round(n_words / config.REEL_NARRATION_WPM * 60, 1)
    script["visual_style"] = (script.get("visual_style") or "").strip() or DEFAULT_STYLE
    script["target_seconds"] = target_seconds
    return script


# ── build job (stills → clips → render) ─────────────────────

_lock = threading.Lock()
_tasks: dict[str, dict] = {}


def get_progress(project_id: str) -> dict:
    with _lock:
        st = _tasks.get(project_id)
        if not st:
            return {"active": False, "stage": "", "done": 0, "total": 0, "error": None}
        return dict(st)


def _set(project_id: str, **fields) -> None:
    with _lock:
        st = _tasks.get(project_id)
        if st:
            st.update(fields)


def start_build(
    project_id: str,
    script: dict,
    project_dir: Path,
    on_done: Callable[[dict, dict | None, str | None], None],
) -> bool:
    """Run stills → clips → render on a daemon thread.

    ``on_done(script, reel, error)`` is called at the end with the updated
    script (clip paths recorded per scene). Returns False if a build is
    already active for this project.
    """
    with _lock:
        if _tasks.get(project_id, {}).get("active"):
            return False
        _tasks[project_id] = {
            "active": True, "stage": "beats", "done": 0,
            "total": 2 * len(script.get("scenes") or []), "error": None,
        }

    def _run():
        try:
            reel = asyncio.run(_build(project_id, script, project_dir))
            on_done(script, reel, None)
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            log.error("Reel build failed: %s", tb)
            _set(project_id, error=str(e))
            on_done(script, None, f"{e}\n{tb}")
        finally:
            _set(project_id, active=False)

    threading.Thread(target=_run, daemon=True).start()
    return True


async def _build(project_id: str, script: dict, project_dir: Path) -> dict:
    scenes = script["scenes"]
    style = script.get("visual_style") or DEFAULT_STYLE
    images_dir = project_dir / "images"
    clips_dir = project_dir / "clips"

    # 1. Beats, strictly in order: still → clip → next still. Every still after
    #    the first is an edit of the previous clip's LAST FRAME, so each step
    #    visibly adds to the same picture. A beat is only regenerated when its
    #    prompt changed — and then every beat after it, since they chain.
    _set(project_id, stage="beats", done=0, total=2 * len(scenes))
    prev_frame: Path | None = None
    dirty = False
    for i, sc in enumerate(scenes):
        still = images_dir / f"scene_{i:04d}_img_0.png"
        if dirty or not still.exists() or sc.get("reel_still_prompt") != sc["image_prompt"]:
            await image_gen.generate_image_nano_banana(
                sc["image_prompt"], style, still, aspect_ratio="9:16",
                reference_images=[prev_frame] if prev_frame else None,
                reference_directive=CONTINUITY_DIRECTIVE if prev_frame else None,
            )
            sc["reel_still_prompt"] = sc["image_prompt"]
            dirty = True
        sc["image_path"] = still.relative_to(project_dir).as_posix()
        sc["image_paths"] = [sc["image_path"]]
        _set(project_id, done=2 * i + 1)

        clip_dir = clips_dir / f"animatediff_s{i:04d}_i0"
        mp4 = clip_dir / "source.mp4"
        prompt = f"{sc.get('motion_prompt') or sc['image_prompt']}. {MOTION_SUFFIX}"
        if dirty or not mp4.exists() or sc.get("reel_clip_prompt") != prompt:
            await generate_animatediff_clip(
                image_path=still, prompt=prompt,
                output_dir=clips_dir, scene_index=i, img_index=0,
                motion_preset=None, style_prompt="",
                model=config.REEL_I2V_MODEL, resolution=config.REEL_I2V_RESOLUTION,
            )
            sc["reel_clip_prompt"] = prompt
            dirty = True
        frames = sorted(clip_dir.glob("frame_*.png")) if clip_dir.exists() else []
        if mp4.exists() and frames:
            sc["reel_clip_path"] = mp4.relative_to(project_dir).as_posix()
            sc.pop("reel_clip_error", None)
            prev_frame = frames[-1]
        else:
            sc["reel_clip_error"] = (
                getattr(generate_animatediff_clip, "last_error", None) or "I2V failed"
            )
            log.warning("Reel beat %d has no clip (%s) — using the still",
                        i, sc["reel_clip_error"])
            prev_frame = still
        _set(project_id, done=2 * i + 2)

    # 2. Render.
    _set(project_id, stage="render")
    path, duration = await asyncio.to_thread(
        shorts.render_reel, script, project_dir, project_dir / "reel.mp4"
    )
    try:
        export_dir = config.OUTPUT_DIR / project_id
        export_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, export_dir / "reel.mp4")
    except Exception as ce:  # noqa: BLE001
        log.warning("Reel output-folder copy failed: %s", ce)
    return {"path": path.relative_to(project_dir).as_posix(), "duration": duration}
