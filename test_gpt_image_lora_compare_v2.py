"""GPT Image comparison v2 — dominant per-LoRA style.

v1 (test_gpt_image_lora_compare.py) passed every render the same neutral
style_prompt and appended each LoRA's description as a trailing
"Visual style references:" suffix. The result: all 9 renders looked nearly
identical because the constant scaffold (preamble + neutral style + closing
instruction) dwarfed the ~1-sentence variable suffix.

v2 makes the LoRA description the dominant style instruction by passing it AS
the style_prompt, with lora_keys=None so no trailing reference is appended.
Same scene, same scaffold; only the style line varies — and it varies as much
as the LoRA descriptions themselves vary.

Run:
    py test_gpt_image_lora_compare_v2.py [<run_timestamp>]

Outputs land beside v1 renders as scene_0001_img_0_gpt2.png inside each
LoRA folder, so all three versions sit side-by-side:
    scene_0001_img_0.png       — Replicate + LoRA weights
    scene_0001_img_0_gpt.png   — GPT Image + neutral style + LoRA trailing ref (v1)
    scene_0001_img_0_gpt2.png  — GPT Image + LoRA description as dominant style (v2)
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
log = logging.getLogger("test_gpt_image_lora_compare_v2")


from backend import image_gen  # noqa: E402
from backend import project_store  # noqa: E402
from backend import config as backend_config  # noqa: E402


FIGURE_PROMPT = (
    "medium shot of an adult woman in a long woolen dress kneeling beside a stone "
    "well, her hands resting on the rim, head bowed, moonlight from above carving her "
    "silhouette, ivy climbing the well stones, autumn leaves on the ground, "
    "no other figures, symbolic stillness"
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


async def _render_one(*, lora_key: str, output_path: Path) -> tuple[bool, str, str]:
    """Render with the LoRA description elevated to dominant style.

    Returns (ok, error_message, style_prompt_used)."""
    description = image_gen.AVAILABLE_LORAS.get(lora_key, {}).get("description", "")
    if not description:
        return False, f"No description for LoRA {lora_key}", ""

    seed = project_store.seed_from_project_id(f"gpt_compare_v2_{lora_key}")
    try:
        await image_gen.generate_image_gpt_image(
            prompt=FIGURE_PROMPT,
            style_prompt=description,  # ← dominant style, not a trailing reference
            output_path=output_path,
            seed=seed,
            lora_keys=None,            # ← prevents the "Visual style references:" appendage
        )
        return True, "", description
    except image_gen.OpenAIImageAccessError as e:
        return False, f"AUTH: {e}", description
    except image_gen.OpenAIImageSafetyError as e:
        return False, f"SAFETY: {e}", description
    except Exception as e:
        return False, f"{type(e).__name__}: {e}", description


async def main(argv: list[str]) -> int:
    if not backend_config.OPENAI_API_KEY:
        log.error("OPENAI_API_KEY not set in .env. Aborting.")
        return 2

    run_dir = _resolve_run_dir(argv[1] if len(argv) > 1 else None)
    log.info(f"Run dir: {run_dir.resolve()}")

    lora_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir()])
    lora_keys = [p.name for p in lora_dirs if p.name in image_gen.AVAILABLE_LORAS]
    log.info(f"GPT Image v2 rendering {len(lora_keys)} LoRAs with their description as dominant style")
    log.info(f"Model: {backend_config.OPENAI_IMAGE_MODEL}, quality={backend_config.OPENAI_IMAGE_QUALITY}, "
             f"size={backend_config.OPENAI_IMAGE_SIZE}")

    delay = backend_config.OPENAI_IMAGE_DELAY_SECONDS
    t0 = time.monotonic()
    results: list[tuple[str, bool, str, str]] = []
    rendered_this_run = 0
    for lora_key in lora_keys:
        out_dir = run_dir / lora_key / "images"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / "scene_0001_img_0_gpt2.png"
        if output_path.exists() and output_path.stat().st_size > 0:
            log.info(f"[{lora_key}] SKIP — already rendered ({output_path.stat().st_size:,} bytes)")
            description = image_gen.AVAILABLE_LORAS.get(lora_key, {}).get("description", "")
            results.append((lora_key, True, "(skipped — already on disk)", description))
            continue
        if rendered_this_run > 0 and delay > 0:
            log.info(f"Inter-call throttle: sleeping {delay:.1f}s")
            await asyncio.sleep(delay)
        log.info(f"[{lora_key}] rendering → {output_path}")
        call_t0 = time.monotonic()
        try:
            ok, err, style_used = await _render_one(lora_key=lora_key, output_path=output_path)
        except Exception as e:
            log.exception(f"[{lora_key}] unhandled exception in _render_one: {e}")
            ok, err, style_used = False, f"unhandled {type(e).__name__}: {e}", ""
        rendered_this_run += 1
        call_dt = time.monotonic() - call_t0
        log.info(
            f"[{lora_key}] {'OK' if ok else 'FAIL'} in {call_dt:.1f}s — style={style_used[:80]!r}"
            + (f" — {err}" if err else "")
        )
        results.append((lora_key, ok, err, style_used))

    total_dt = time.monotonic() - t0
    print()
    print("=" * 80)
    print(f"  Wall time:  {total_dt:.1f}s ({total_dt/60:.1f} min)")
    print(f"  Run dir:    {run_dir.resolve()}")
    print("=" * 80)
    succ = 0
    for lora_key, ok, err, style_used in results:
        marker = "OK  " if ok else "FAIL"
        print(f"  [{marker}] {lora_key:<25} style: {style_used[:65]!r}...")
        if err:
            print(f"           error: {err[:160]}")
        if ok:
            succ += 1
    print("-" * 80)
    print(f"  Totals: {succ}/{len(results)} v2 renders succeeded")
    print()
    print("Three-way side-by-side per LoRA:")
    print(f"  {run_dir.resolve()}\\<lora>\\images\\")
    print("    scene_0001_img_0.png       (Replicate + LoRA weights)")
    print("    scene_0001_img_0_gpt.png   (GPT v1 — neutral style, LoRA as trailing ref)")
    print("    scene_0001_img_0_gpt2.png  (GPT v2 — LoRA description as dominant style)")
    print()
    return 0 if succ == len(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv)))
