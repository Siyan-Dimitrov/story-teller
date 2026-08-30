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
CLIP_CONCURRENCY = 3  # Replicate throttles creation bursts on low-credit accounts


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
    words = int(target_seconds * config.NARRATION_WPM / 60)
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
        sc["duration_hint"] = round(n_words / config.NARRATION_WPM * 60, 1)
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
            "active": True, "stage": "stills", "done": 0,
            "total": len(script.get("scenes") or []), "error": None,
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

    # 1. Stills — beat 1 first, the rest anchored on it so props/light match.
    #    A still is only re-rendered when its prompt changed (re-runs are free).
    _set(project_id, stage="stills", done=0, total=len(scenes))
    anchor: Path | None = None
    changed: set[int] = set()
    for i, sc in enumerate(scenes):
        out = images_dir / f"scene_{i:04d}_img_0.png"
        if not out.exists() or sc.get("reel_still_prompt") != sc["image_prompt"]:
            await image_gen.generate_image_nano_banana(
                sc["image_prompt"], style, out, aspect_ratio="9:16",
                reference_images=[anchor] if anchor else None,
                style_reference_only=anchor is not None,
            )
            sc["reel_still_prompt"] = sc["image_prompt"]
            changed.add(i)
        sc["image_path"] = out.relative_to(project_dir).as_posix()
        sc["image_paths"] = [sc["image_path"]]
        anchor = anchor or out
        _set(project_id, done=i + 1)

    # 2. Clips — Seedance, a few at a time.
    _set(project_id, stage="clips", done=0, total=len(scenes))
    sem = asyncio.Semaphore(CLIP_CONCURRENCY)
    finished = 0

    async def _clip(i: int, sc: dict) -> None:
        nonlocal finished
        async with sem:
            mp4 = clips_dir / f"animatediff_s{i:04d}_i0" / "source.mp4"
            prompt = f"{sc.get('motion_prompt') or sc['image_prompt']}. {MOTION_SUFFIX}"
            if i in changed or not mp4.exists() or sc.get("reel_clip_prompt") != prompt:
                await generate_animatediff_clip(
                    image_path=project_dir / sc["image_path"], prompt=prompt,
                    output_dir=clips_dir, scene_index=i, img_index=0,
                    motion_preset=None, style_prompt="",
                    model=config.REEL_I2V_MODEL, resolution=config.REEL_I2V_RESOLUTION,
                )
                sc["reel_clip_prompt"] = prompt
            if mp4.exists():
                sc["reel_clip_path"] = mp4.relative_to(project_dir).as_posix()
                sc.pop("reel_clip_error", None)
            else:
                sc["reel_clip_error"] = (
                    getattr(generate_animatediff_clip, "last_error", None) or "I2V failed"
                )
                log.warning("Reel beat %d has no clip (%s) — using the still",
                            i, sc["reel_clip_error"])
            finished += 1
            _set(project_id, done=finished)

    await asyncio.gather(*[_clip(i, sc) for i, sc in enumerate(scenes)])

    # 3. Render.
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
