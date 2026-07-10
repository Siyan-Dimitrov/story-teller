"""Single-LoRA Replicate test — invokes the same image_gen.generate_all_scenes
function the Producer pipeline calls, with backend='replicate' and
lora_keys=['dark_gothic'].

Run:
    set REPLICATE_API_TOKEN=r8_...     (or sourced from .env via config.py)
    py test_replicate_single_lora.py

Outputs land in test_output/replicate_single_lora/<timestamp>/images/.
The output dir is preserved so you can eyeball the results.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("test_replicate_single_lora")


from backend import image_gen, image_styles  # noqa: E402
from backend import project_store  # noqa: E402


GROVE_OF_BRIARS = (
    "wide establishing shot of an ancient briar-choked grove at dusk, twisted thorn "
    "branches arching overhead like a cathedral nave, low side-light from a single "
    "lantern hung on an iron hook, mist pooling in the moss below, no figures, "
    "painterly composition with deep shadow in the foreground"
)
SISTER_AT_THE_WELL = (
    "medium shot of an adult woman in a long woolen dress kneeling beside a stone "
    "well, her hands resting on the rim, head bowed, moonlight from above carving her "
    "silhouette, ivy climbing the well stones, autumn leaves on the ground, "
    "no other figures, symbolic stillness"
)
RAVEN_ON_A_CROWN = (
    "close shot of a tarnished silver crown resting on a dark velvet cushion in an "
    "empty hall, a single raven perched on its uppermost point, candlelight from "
    "off-frame catching the metal, dust motes in the air, shallow depth of field, "
    "no human figures"
)
THE_LONG_HALL = (
    "wide interior shot of a long gothic hall with vaulted ceilings, a single line "
    "of black-clad mourners walking away from camera toward a distant lit doorway, "
    "stained glass windows casting cold blue light across the stone floor, "
    "symbolic procession, painterly chiaroscuro, no faces visible"
)

TEST_PROMPTS = [GROVE_OF_BRIARS, SISTER_AT_THE_WELL, RAVEN_ON_A_CROWN, THE_LONG_HALL]


def _build_test_scenes() -> list[dict]:
    return [
        {"index": idx, "image_prompts": [prompt]}
        for idx, prompt in enumerate(TEST_PROMPTS)
    ]


def _resolve_style_prompt() -> str:
    style = image_styles.get_style("gothic_folklore")
    if style is None:
        return image_styles.DEFAULT_STYLE_PROMPT
    return style.prompt


async def main() -> int:
    token = os.environ.get("REPLICATE_API_TOKEN", "").strip()
    if not token:
        log.error(
            "REPLICATE_API_TOKEN is not set. Set it in your environment or in the "
            ".env file backend/config.py reads from. Aborting."
        )
        return 2

    masked = token[:6] + "..." + token[-4:] if len(token) > 12 else "***"
    log.info(f"REPLICATE_API_TOKEN present (length={len(token)}, head/tail={masked})")

    # Isolated output dir — does not touch the real projects/ tree.
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_root = Path("test_output") / "replicate_single_lora" / ts
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "images").mkdir(exist_ok=True)
    log.info(f"Output directory: {out_root.resolve()}")

    scenes = _build_test_scenes()
    style_prompt = _resolve_style_prompt()
    lora_keys = ["dark_gothic"]

    project_seed = project_store.seed_from_project_id(f"test_{ts}")
    log.info(
        f"Invoking image_gen.generate_all_scenes — backend=replicate, "
        f"scenes={len(scenes)}, lora_keys={lora_keys}, project_seed={project_seed}"
    )
    log.info(f"Style prompt: {style_prompt}")
    log.info(f"Single LoRA URL: {os.environ.get('FLUX_LORA_DARK_GOTHIC') or '(default from config.py)'}")

    t0 = time.monotonic()
    try:
        updated_scenes = await image_gen.generate_all_scenes(
            scenes=scenes,
            project_dir=out_root,
            backend="replicate",
            style_prompt=style_prompt,
            lora_keys=lora_keys,
            character_consistency=False,
            project_seed=project_seed,
        )
    except Exception as e:
        log.exception(f"generate_all_scenes raised: {e}")
        return 1
    dt = time.monotonic() - t0

    successes: list[tuple[int, Path]] = []
    failures: list[tuple[int, str]] = []
    for s in updated_scenes:
        idx = s.get("index")
        paths = s.get("image_paths") or []
        if paths:
            for p in paths:
                full = out_root / p
                successes.append((idx, full))
        errors = s.get("image_errors") or []
        if errors:
            for err in errors:
                failures.append((idx, str(err)))

    print()
    print("=" * 60)
    print(f"  Wall time: {dt:.1f}s ({dt/len(scenes):.1f}s avg per prompt)")
    print(f"  Scenes:    {len(scenes)} sent, {len(successes)} images written")
    print(f"  Output:    {out_root.resolve()}")
    print("=" * 60)
    if successes:
        print("  Successes:")
        for idx, p in successes:
            size = p.stat().st_size if p.exists() else 0
            print(f"    scene {idx}: {p.relative_to(out_root)}  ({size:,} bytes)")
    if failures:
        print("  Failures / errors:")
        for idx, err in failures:
            print(f"    scene {idx}: {err[:200]}")
    print()
    return 0 if successes and not failures else (0 if successes else 1)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
