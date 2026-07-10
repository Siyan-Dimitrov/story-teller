"""Smoke test for the Claude three-pass screenwriter.

Runs writer → critic → reviser on a tiny source so we can eyeball the
output and confirm subscription auth works.
"""

import asyncio
import json
import logging
import sys

# Force UTF-8 stdout so unicode rules (U+2500 etc) don't crash on Windows cp1252.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

from backend import claude_script_gen


SOURCE = (
    "A miller dies leaving only a cat to his youngest son. The cat asks for "
    "a pair of fine boots and a sack, and proceeds to trick the king, a "
    "fearsome ogre, and an entire countryside into believing his ragged "
    "master is the great Marquis of Carabas. By the end the boy has won a "
    "castle and a princess, and the cat lives in idleness as a great lord."
)


async def main() -> int:
    print("=" * 60)
    print("Running Claude screenwriter: writer + critic + reviser")
    print("=" * 60)
    script = await claude_script_gen.generate_script(
        custom_prompt=SOURCE,
        target_minutes=2.0,
        tone="dark whimsical",
        pipeline_writer_model="claude:claude-sonnet-4-5",
        pipeline_critic_model="claude:claude-opus-4-7",
        pipeline_reviser_model="claude:claude-sonnet-4-5",
    )

    # Save the result first so any pretty-print bug can't lose it.
    with open("test_claude_screenwriter_output.json", "w", encoding="utf-8") as f:
        json.dump(script, f, indent=2, ensure_ascii=False)

    scenes = script.get("scenes", [])
    bar = "-" * 60
    print()
    print(bar)
    print(f"Title:    {script.get('title')}")
    print(f"Synopsis: {script.get('synopsis')}")
    print(f"Scenes:   {len(scenes)}")
    print(f"Cost:     ${script.get('_claude_cost_usd', 0):.4f}")
    print(bar)
    for s in scenes:
        print(f"\nScene {s['index']} [{s['mood']}, hint {s['duration_hint']}s]")
        narration = s["narration"]
        preview = narration if len(narration) <= 320 else narration[:320] + " ..."
        print(f"  Narration: {preview}")
        for j, p in enumerate(s.get("image_prompts", [])):
            print(f"  Img[{j}]:    {p}")
    print()
    print("Full JSON saved to test_claude_screenwriter_output.json")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
