"""GPT Image comparison for the LoRA samples.

For each LoRA folder under test_output/lora_samples/<latest>/, generate one
GPT Image rendering of the same figure-at-the-well scene used in the Replicate
sample. Passes the LoRA key to generate_image_gpt_image so the LoRA's
descriptive text lands as "Visual style references" in the GPT prompt — same
behavior the Producer pipeline uses when image_backend='gpt_image'.

Run:
    py test_gpt_image_lora_compare.py [<run_timestamp>]

If no timestamp arg, uses the most recent run under test_output/lora_samples/.

Outputs land next to the existing Replicate PNGs:
    test_output/lora_samples/<ts>/<lora>/images/scene_0001_img_0_gpt.png
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("test_gpt_image_lora_compare")


from backend import image_gen  # noqa: E402
from backend import project_store  # noqa: E402
from backend import config as backend_config  # noqa: E402


# Use the same figure prompt the Replicate samples used so the comparison is
# apples-to-apples. (The environment-only prompt is fine too, but the figure
# scene exercises both character and atmosphere — harder for both backends.)
FIGURE_PROMPT = (
    "medium shot of an adult woman in a long woolen dress kneeling beside a stone "
    "well, her hands resting on the rim, head bowed, moonlight from above carving her "
    "silhouette, ivy climbing the well stones, autumn leaves on the ground, "
    "no other figures, symbolic stillness"
)

NEUTRAL_STYLE_PROMPT = (
    "dark folklore illustration, cinematic composition, atmospheric lighting, "
    "non-photorealistic, painterly, no text or watermarks"
)


def _resolve_run_dir(arg: str | None) -> Path:
    samples_root = Path("test_output") / "lora_samples"
    if not samples_root.exists():
        raise SystemExit(f"No samples root at {samples_root.resolve()}")
    if arg:
        candidate = samples_root / arg
        if not candidate.exists():
            raise SystemExit(f"Run dir not found: {candidate.resolve()}")
        return candidate
    runs = sorted([p for p in samples_root.iterdir() if p.is_dir()])
    if not runs:
        raise SystemExit(f"No runs under {samples_root.resolve()}")
    return runs[-1]


async def _render_one(*, lora_key: str, output_path: Path) -> tuple[bool, str]:
    seed = project_store.seed_from_project_id(f"gpt_compare_{lora_key}")
    try:
        await image_gen.generate_image_gpt_image(
            prompt=FIGURE_PROMPT,
            style_prompt=NEUTRAL_STYLE_PROMPT,
            output_path=output_path,
            seed=seed,
            lora_keys=[lora_key],
        )
        return True, ""
    except image_gen.OpenAIImageAccessError as e:
        return False, f"AUTH: {e}"
    except image_gen.OpenAIImageSafetyError as e:
        return False, f"SAFETY: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


async def main(argv: list[str]) -> int:
    if not backend_config.OPENAI_API_KEY:
        log.error("OPENAI_API_KEY not set in .env. Aborting.")
        return 2

    run_dir = _resolve_run_dir(argv[1] if len(argv) > 1 else None)
    log.info(f"Run dir: {run_dir.resolve()}")

    lora_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir()])
    if not lora_dirs:
        log.error(f"No LoRA folders under {run_dir}")
        return 1
    lora_keys = [p.name for p in lora_dirs if p.name in image_gen.AVAILABLE_LORAS]
    log.info(f"GPT Image rendering {len(lora_keys)} LoRA descriptions: {lora_keys}")
    log.info(f"Model: {backend_config.OPENAI_IMAGE_MODEL}, quality={backend_config.OPENAI_IMAGE_QUALITY}, "
             f"size={backend_config.OPENAI_IMAGE_SIZE}")

    delay = backend_config.OPENAI_IMAGE_DELAY_SECONDS
    t0 = time.monotonic()
    results: list[tuple[str, bool, str]] = []
    for i, lora_key in enumerate(lora_keys):
        if i > 0 and delay > 0:
            log.info(f"Inter-call throttle: sleeping {delay:.1f}s")
            await asyncio.sleep(delay)
        out_dir = run_dir / lora_key / "images"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / "scene_0001_img_0_gpt.png"
        log.info(f"[{lora_key}] rendering → {output_path}")
        call_t0 = time.monotonic()
        ok, err = await _render_one(lora_key=lora_key, output_path=output_path)
        call_dt = time.monotonic() - call_t0
        log.info(f"[{lora_key}] {'OK' if ok else 'FAIL'} in {call_dt:.1f}s" + (f" — {err}" if err else ""))
        results.append((lora_key, ok, err))

    total_dt = time.monotonic() - t0
    print()
    print("=" * 72)
    print(f"  Wall time:  {total_dt:.1f}s ({total_dt/60:.1f} min)")
    print(f"  Run dir:    {run_dir.resolve()}")
    print("=" * 72)
    succ = 0
    for lora_key, ok, err in results:
        marker = "OK  " if ok else "FAIL"
        line = f"  [{marker}] {lora_key:<25}"
        if err:
            line += f"  {err[:160]}"
        print(line)
        if ok:
            succ += 1
    print("-" * 72)
    print(f"  Totals: {succ}/{len(results)} GPT Image renders succeeded")
    print()
    print("Side-by-side comparison:")
    print(f"  Replicate:  {run_dir.resolve()}\\<lora>\\images\\scene_0001_img_0.png")
    print(f"  GPT Image:  {run_dir.resolve()}\\<lora>\\images\\scene_0001_img_0_gpt.png")
    print()
    return 0 if succ == len(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv)))
