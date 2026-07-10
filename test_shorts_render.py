"""One-shot smoke test for shorts_assembly.

Reuses scene 0 of an existing assembled project as a stand-in for hook audio
+ scene image so we can verify the visual renderer end-to-end without
calling the LLM or VoiceBox. Real director runs would generate fresh
narration; this test just exercises the assembly pipeline.
"""

import sys
from pathlib import Path

from backend import shorts_assembly

PROJECT_ID = "111a81b57579"
PROJECT_DIR = Path("C:/Dev/story_teller/projects") / PROJECT_ID
IMAGE = PROJECT_DIR / "images" / "scene_0000_img_0.png"
AUDIO = PROJECT_DIR / "audio" / "scene_0000.wav"
OUT = PROJECT_DIR / "shorts" / "smoke_test.mp4"

if not IMAGE.exists():
    sys.exit(f"Image missing: {IMAGE}")
if not AUDIO.exists():
    sys.exit(f"Audio missing: {AUDIO}")

HEADLINE = "What waited in the dragon's hollow?"
# Use the existing scene narration as caption source so we can eyeball
# whether the burned-in captions render correctly.
NARRATION = (
    "A king's daughter was promised to whoever could slay the seven-headed dragon. "
    "Many tried. None returned. One day a young hunter rode toward the mountain "
    "with nothing but a hound at his heel and an old blade at his side."
)

print(f"Rendering smoke-test short to {OUT}")
out_path, dur = shorts_assembly.assemble_short(
    image_path=IMAGE,
    audio_path=AUDIO,
    headline=HEADLINE,
    narration=NARRATION,
    output_path=OUT,
)
print(f"Done: {out_path} ({dur:.2f}s)")
