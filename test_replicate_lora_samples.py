"""Sample every available LoRA via the production pipeline path.

For each LoRA in image_gen.AVAILABLE_LORAS, runs image_gen.generate_all_scenes
with backend='replicate' on two contrasting prompts (one environment-only, one
with a single figure) and saves the outputs to a per-LoRA subdirectory.

Run:
    py test_replicate_lora_samples.py

Outputs:
    test_output/lora_samples/<timestamp>/<lora_key>/images/scene_*.png
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("test_replicate_lora_samples")


from backend import image_gen, image_styles  # noqa: E402
from backend import project_store  # noqa: E402
from backend import config as backend_config  # noqa: E402


# Two contrasting prompts that exercise different LoRA strengths:
#   • Environment-only: tests atmosphere, lighting, composition.
#   • Single figure:    tests character rendering / stylization.

ENVIRONMENT_PROMPT = (
    "wide establishing shot of an ancient briar-choked grove at dusk, twisted thorn "
    "branches arching overhead like a cathedral nave, low side-light from a single "
    "lantern hung on an iron hook, mist pooling in the moss below, no figures, "
    "painterly composition with deep shadow in the foreground"
)
FIGURE_PROMPT = (
    "medium shot of an adult woman in a long woolen dress kneeling beside a stone "
    "well, her hands resting on the rim, head bowed, moonlight from above carving her "
    "silhouette, ivy climbing the well stones, autumn leaves on the ground, "
    "no other figures, symbolic stillness"
)


def _build_scenes() -> list[dict]:
    return [
        {"index": 0, "image_prompts": [ENVIRONMENT_PROMPT]},
        {"index": 1, "image_prompts": [FIGURE_PROMPT]},
    ]


# Generic neutral style prompt — keep it simple so the LoRA itself is the
# differentiator across samples, not a heavy style stack.
NEUTRAL_STYLE_PROMPT = (
    "dark folklore illustration, cinematic composition, atmospheric lighting, "
    "non-photorealistic, painterly, no text or watermarks"
)


async def _sample_one_lora(*, lora_key: str, out_root: Path) -> tuple[int, list[str]]:
    """Run two scenes through the Replicate path with a single LoRA. Returns
    (success_count, list of errors)."""
    lora_dir = out_root / lora_key
    lora_dir.mkdir(parents=True, exist_ok=True)
    (lora_dir / "images").mkdir(exist_ok=True)

    seed = project_store.seed_from_project_id(f"lora_test_{lora_key}")
    scenes = _build_scenes()

    log.info(f"[{lora_key}] starting — seed={seed}")
    try:
        updated = await image_gen.generate_all_scenes(
            scenes=scenes,
            project_dir=lora_dir,
            backend="replicate",
            style_prompt=NEUTRAL_STYLE_PROMPT,
            lora_keys=[lora_key],
            character_consistency=False,
            project_seed=seed,
        )
    except Exception as e:
        log.exception(f"[{lora_key}] generate_all_scenes raised: {e}")
        return 0, [str(e)]

    successes = 0
    errors: list[str] = []
    for s in updated:
        paths = s.get("image_paths") or []
        successes += sum(1 for _ in paths)
        for err in (s.get("image_errors") or []):
            errors.append(str(err))
    return successes, errors


async def main() -> int:
    token = os.environ.get("REPLICATE_API_TOKEN", "").strip()
    if not token:
        log.error("REPLICATE_API_TOKEN is not set. Aborting.")
        return 2

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_root = Path("test_output") / "lora_samples" / ts
    out_root.mkdir(parents=True, exist_ok=True)
    log.info(f"Output root: {out_root.resolve()}")

    lora_keys = list(image_gen.AVAILABLE_LORAS.keys())
    log.info(f"Sampling {len(lora_keys)} LoRAs × 2 prompts = {len(lora_keys) * 2} images")
    log.info(f"LoRAs: {lora_keys}")

    delay_between_loras = backend_config.REPLICATE_DELAY_SECONDS
    t0 = time.monotonic()

    summary: list[dict] = []
    for i, lora_key in enumerate(lora_keys):
        # Throttle between LoRAs — generate_all_scenes only throttles internally,
        # not across calls. Without this we'd burst Replicate at every boundary.
        if i > 0:
            log.info(f"Inter-LoRA throttle: sleeping {delay_between_loras:.1f}s")
            await asyncio.sleep(delay_between_loras)

        lora_t0 = time.monotonic()
        successes, errors = await _sample_one_lora(lora_key=lora_key, out_root=out_root)
        lora_dt = time.monotonic() - lora_t0
        summary.append({
            "lora_key": lora_key,
            "successes": successes,
            "errors": errors,
            "elapsed_s": round(lora_dt, 1),
        })
        log.info(
            f"[{lora_key}] done in {lora_dt:.1f}s — {successes}/2 images, {len(errors)} errors"
        )

    total_dt = time.monotonic() - t0

    print()
    print("=" * 72)
    print(f"  Wall time:  {total_dt:.1f}s ({total_dt/60:.1f} min)")
    print(f"  Output:     {out_root.resolve()}")
    print("=" * 72)
    total_ok = 0
    total_err = 0
    for row in summary:
        marker = "OK " if row["successes"] == 2 and not row["errors"] else "WARN" if row["successes"] >= 1 else "FAIL"
        print(
            f"  [{marker}] {row['lora_key']:<25} {row['successes']}/2 images  "
            f"({row['elapsed_s']:>5.1f}s)"
            + (f"  errors={row['errors']}" if row['errors'] else "")
        )
        total_ok += row["successes"]
        total_err += len(row["errors"])
    print("-" * 72)
    print(f"  Totals: {total_ok}/{len(lora_keys) * 2} images, {total_err} errors")
    print()
    print("Each LoRA's images live in:")
    print(f"  {out_root.resolve()}\\<lora_key>\\images\\")
    print()
    return 0 if total_ok == len(lora_keys) * 2 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
