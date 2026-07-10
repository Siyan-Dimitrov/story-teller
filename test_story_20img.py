"""Generate a tiny story (~21 images) end-to-end with Nano Banana + cast
consistency, saved as a real, browsable project.

Run:  venv\Scripts\python.exe test_story_20img.py
"""

import asyncio
import json

from backend import config, image_gen, script_gen
from backend import project_store as store

# Speed up the throttle for this test run (default 11s is tuned for tiny-credit
# Flux accounts; Nano Banana + retry/backoff handles a faster cadence fine).
config.REPLICATE_DELAY_SECONDS = 2.0

STYLE = ("storybook ink-and-watercolor illustration, warm candlelit palette of amber "
         "and deep teal, soft grain, gentle rim light, cosy and slightly melancholic")

CAST = [
    {
        "id": "elias",
        "name": "Elias the Clockmaker",
        "role": "protagonist",
        "description": ("elderly man around seventy, silver hair tied back, deep kind "
                        "wrinkles, round brass spectacles, long charcoal-grey work apron "
                        "over a cream linen shirt, ink-stained fingers"),
        "reference_prompt": ("An elderly man around seventy with silver hair tied back, deep "
                             "kind wrinkles, round brass spectacles, wearing a long charcoal-grey "
                             "work apron over a cream linen shirt, standing alone against a plain "
                             "neutral background, even soft light, neutral expression, waist-up portrait"),
    },
    {
        "id": "ada",
        "name": "Ada",
        "role": "supporting",
        "description": ("young woman in her early twenties, auburn hair in a loose braid, "
                        "freckles, forest-green dress with a high collar, a small brass key "
                        "on a ribbon around her neck"),
        "reference_prompt": ("A young woman in her early twenties with auburn hair in a loose "
                             "braid, freckles, wearing a forest-green dress with a high collar and "
                             "a small brass key on a ribbon around her neck, standing alone against "
                             "a plain neutral background, even soft light, neutral expression, waist-up portrait"),
    },
]

SCENES = [
    {
        "narration": "At the end of Lantern Lane stood a clock shop that never quite slept.",
        "characters": ["elias"],
        "image_prompts": [
            "wide establishing shot of a narrow clock shop at dusk on a cobbled lane, dozens of clock faces glowing in the window, warm amber light spilling onto wet stones, blue evening sky above",
            "medium shot of Elias the elderly clockmaker in his charcoal apron and brass spectacles, standing in the doorway holding a lantern, looking down the empty lane at dusk",
            "close-up on Elias's ink-stained hands winding a small pocket watch by candlelight, brass gears catching the warm glow",
        ],
    },
    {
        "narration": "Each evening his daughter Ada came to learn the language of gears.",
        "characters": ["ada"],
        "image_prompts": [
            "medium shot of Ada, a young woman with an auburn braid and forest-green dress, pushing open the shop door, evening light behind her, a brass key on a ribbon at her throat",
            "over-the-shoulder shot of Ada leaning over a workbench cluttered with tiny gears and tools, lamplight on her freckled face",
            "close-up on the small brass key resting against Ada's green collar as she bends to her work",
        ],
    },
    {
        "narration": "Together they tended a hundred small machines, and one very old one.",
        "characters": ["elias", "ada"],
        "image_prompts": [
            "two-shot of Elias and Ada side by side at the long workbench, both bent over a half-built clock, warm lamp between them, shelves of clocks behind",
            "medium shot of Elias pointing to a gear while Ada watches, his brass spectacles reflecting the lamplight, her braid falling forward",
            "low-angle shot looking up at a towering antique grandfather clock in the corner, Elias and Ada small at its base in candlelight",
        ],
    },
    {
        "narration": "But on the longest night, the great clock fell silent.",
        "characters": ["elias"],
        "image_prompts": [
            "wide shot of the tall antique clock standing dark and still, its pendulum frozen, Elias staring up at it with a lantern, long shadows across the shop floor",
            "close-up on the stopped pendulum and a cracked brass gear inside the open clock case, cold blue moonlight mixing with warm lamp glow",
            "medium shot of Elias resting a worried hand on the silent clock, his face lit from below by the lantern",
        ],
    },
    {
        "narration": "It was Ada who found the hidden door behind the dial.",
        "characters": ["ada"],
        "image_prompts": [
            "medium shot of Ada lifting the clock's painted dial to reveal a tiny hidden compartment, her eyes wide with discovery, lamplight on her face",
            "close-up on Ada's hand drawing out a folded brass mechanism no bigger than a moth from the compartment",
            "over-the-shoulder shot of Ada holding the brass key from her necklace up to a matching keyhole inside the clock",
        ],
    },
    {
        "narration": "Side by side, they wound the heart of the old machine back to life.",
        "characters": ["elias", "ada"],
        "image_prompts": [
            "two-shot of Elias and Ada together turning a large brass winding key inside the open clock, warm light flaring from within, both faces lit with effort and hope",
            "close-up on four hands — old and young, ink-stained and freckled — gripping the same brass key",
            "medium shot of the pendulum beginning to swing again, Elias and Ada watching it, relief on their faces in the golden light",
        ],
    },
    {
        "narration": "And at dawn the great clock chimed, as if it had only been waiting for them.",
        "characters": ["elias", "ada"],
        "image_prompts": [
            "wide shot of the clock shop at dawn, pale gold light through the window, the tall clock proud and ticking, Elias and Ada standing before it",
            "medium two-shot of Elias and Ada smiling at each other beside the chiming clock, soft sunrise light on their faces",
        ],
    },
]


async def main() -> None:
    project_id, pdir = store.create_project()
    print("Created project:", project_id)

    script = {
        "title": "The Clockmaker's Daughter",
        "synopsis": "An old clockmaker and his daughter bring a silent antique clock back to life.",
        "visual_style": STYLE,
        "cast": CAST,
        "scenes": SCENES,
        "target_minutes": 2.0,
        "tone": "warm, gentle, melancholic",
    }
    script = script_gen.normalize_scenes(script)
    store.save_json(project_id, "script.json", script)
    store.update_state(project_id, title=script["title"], image_backend="nano_banana", step="scripted")

    seed = store.get_project_seed(project_id)
    n_imgs = sum(len(s["image_prompts"]) for s in script["scenes"])
    print(f"Story: {script['title']} | {len(script['scenes'])} scenes | {n_imgs} images | cast={len(CAST)}")

    print("\n-> Generating character portraits...")
    cast = await image_gen.generate_character_references(
        cast=script["cast"], project_dir=pdir, backend="nano_banana",
        style_prompt=STYLE, project_seed=seed,
    )
    script["cast"] = cast
    store.save_json(project_id, "script.json", script)
    for m in cast:
        print(f"   {m['id']}: {m.get('reference_image_path')} (err={m.get('reference_image_error')})")

    print("\n-> Generating all scene images with character references...")
    scenes = await image_gen.generate_all_scenes(
        scenes=script["scenes"], project_dir=pdir, backend="nano_banana",
        style_prompt=STYLE, character_consistency=True, project_seed=seed, cast=cast,
    )
    script["scenes"] = scenes
    store.save_json(project_id, "script.json", script)
    store.update_state(project_id, step="illustrated")

    ok = sum(len(s.get("image_paths") or []) for s in scenes)
    print(f"\nDone. {ok}/{n_imgs} images generated.")
    print("Project id:", project_id, "->", pdir)


if __name__ == "__main__":
    asyncio.run(main())
