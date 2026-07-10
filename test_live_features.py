"""Live end-to-end test for the two new features (no server needed).

1. Shorts: cut one scene out of an existing final.mp4 and reframe to 9:16.
2. Nano Banana: render a character portrait, then a scene image that
   references that portrait (character consistency).

Run:  venv\Scripts\python.exe test_live_features.py
"""

import asyncio
import json
import shutil
from pathlib import Path

from backend import config, image_gen, shorts

PROJECT = "02d6d1ae00b8"  # The Curse of Yig
PDIR = config.PROJECTS_DIR / PROJECT
OUT = config.BASE_DIR / "test_output"
OUT.mkdir(exist_ok=True)


def test_short_from_final(script: dict) -> None:
    print("\n=== TEST 1: short cut from final.mp4 ===")
    scenes = script["scenes"]
    # Pick a mid scene with images and a sane length.
    scene = next(s for s in scenes if s["index"] == 2)
    ranges = shorts.compute_scene_time_ranges(scenes)
    print("scene 2 timeline range:", ranges.get(2))
    out = OUT / "short_scene2.mp4"
    path, dur = shorts.render_short_from_final(
        scene=scene,
        scenes=scenes,
        project_dir=PDIR,
        output_path=out,
        hook="What waited beneath the asylum?",
    )
    print(f"OK -> {path} ({dur}s, {path.stat().st_size // 1024} KB)")


async def test_nano_banana(script: dict) -> None:
    print("\n=== TEST 2: Nano Banana image + character reference ===")
    nb_dir = OUT / "nano_banana"
    if nb_dir.exists():
        shutil.rmtree(nb_dir)
    nb_dir.mkdir(parents=True)

    style = (script.get("visual_style") or "").strip() or \
        "1920s pulp-horror illustration, muted sepia palette, eerie lamplight, grainy texture"
    print("style:", style[:90], "...")

    # A tiny hand-made cast member so the test is deterministic (no Ollama).
    cast = [{
        "id": "narrator",
        "name": "The Investigator",
        "role": "protagonist",
        "description": "adult man, late 30s, gaunt face, dark short hair, weary eyes, "
                       "brown 1920s travelling coat and waistcoat",
        "reference_prompt": "An adult man in his late thirties, gaunt weathered face, "
                            "short dark hair, weary eyes, wearing a brown 1920s travelling "
                            "coat and waistcoat, standing alone against a plain neutral grey "
                            "background, even soft lighting, neutral expression, waist-up portrait",
        "reference_image_path": None,
    }]

    print("-> rendering character portrait...")
    cast = await image_gen.generate_character_references(
        cast=cast,
        project_dir=nb_dir,
        backend="nano_banana",
        style_prompt=style,
        project_seed=12345,
    )
    portrait = cast[0].get("reference_image_path")
    print("portrait:", portrait, "| error:", cast[0].get("reference_image_error"))
    if not portrait:
        print("FAILED: no portrait produced")
        return
    ppath = nb_dir / portrait
    print(f"   portrait file: {ppath} ({ppath.stat().st_size // 1024} KB)")

    # Now a scene image that should reuse the character's look.
    scene_prompt = (
        "The investigator descends a narrow stone basement stair by lamplight, "
        "one hand on the damp wall, looking down into darkness; low-angle medium shot"
    )
    print("-> rendering scene image with the portrait as reference...")
    scene_out = nb_dir / "scene_with_ref.png"
    await image_gen.generate_image_nano_banana(
        prompt=scene_prompt,
        style_prompt=style,
        output_path=scene_out,
        reference_images=[ppath],
    )
    print(f"OK -> {scene_out} ({scene_out.stat().st_size // 1024} KB)")


def main() -> None:
    script = json.load(open(PDIR / "script.json", encoding="utf-8"))
    print("project:", script.get("title"), "| scenes:", len(script["scenes"]))
    test_short_from_final(script)
    asyncio.run(test_nano_banana(script))
    print("\nAll done. Artifacts in:", OUT)


if __name__ == "__main__":
    main()
