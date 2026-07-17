"""Export assembled project to a clean output folder with YouTube metadata."""

import logging
import re
import shutil
from pathlib import Path

from . import config, llm

log = logging.getLogger(__name__)


def slugify(text: str, max_len: int = 60) -> str:
    """Convert text to a filesystem-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:max_len]


YOUTUBE_META_SYSTEM = "You are a YouTube SEO expert specializing in storytelling and animation channels."

YOUTUBE_META_PROMPT = """Given the following story details, generate optimized YouTube metadata.

Story title: {title}
Synopsis: {synopsis}
Tone: {tone}
Themes: {themes}
Number of scenes: {scene_count}
Book/Series: {book_title}

Respond ONLY with plain text in this exact format (no markdown fences, no JSON):

TITLE: [An engaging, clickable YouTube title under 100 characters. Include the story name and a hook.]

DESCRIPTION:
[A 3-5 paragraph YouTube description. First paragraph should hook viewers in 2 sentences. Include the story summary, mention the dark/gothic style, and add a call to action. End with relevant keywords naturally woven into sentences.]

TAGS:
[Comma-separated list of 20-30 YouTube tags, mixing broad and specific. Include: story name, genre tags, mood tags, related stories, style tags like "dark fairy tale", "gothic animation", "storytelling", "narrated story"]

HASHTAGS:
[5-8 hashtags for the YouTube description, e.g. #DarkFairyTales #GothicStorytelling]

CATEGORY: [One of: Entertainment, Film & Animation, Education]

Guidelines:
- Title should create curiosity without clickbait
- If the story is part of a book or series, reference the book name in the title and description
- Include the book/series name as a tag if applicable
- Description first 2 lines appear in search results — make them count
- Tags should include long-tail keywords for discoverability
- Include tags for related/similar stories viewers might search for
- Add seasonal or trending tags if the story themes align
"""


async def generate_youtube_metadata(
    title: str,
    synopsis: str,
    tone: str,
    themes: list[str],
    scene_count: int,
    book_title: str = "",
    **_ignored,
) -> str:
    """Generate YouTube-optimized metadata using the LLM."""
    prompt = YOUTUBE_META_PROMPT.format(
        title=title,
        synopsis=synopsis,
        tone=tone or "dark, gothic",
        themes=", ".join(themes) if themes else "fairy tale, dark fantasy",
        scene_count=scene_count,
        book_title=book_title or "N/A (standalone story)",
    )

    log.info(f"Generating YouTube metadata for: {title}")

    content = await llm.complete(
        YOUTUBE_META_SYSTEM,
        prompt,
        pass_name="youtube metadata",
    )

    # Strip markdown fences if present (the response is plain text, not JSON)
    return llm.strip_code_fences(content)


def shorts_promo_section(title: str, shorts: list[dict] | None) -> str:
    """Publishing companion for the Shorts: the Related-video-link reminder and
    the pinned-comment text. Links in Shorts comments/descriptions are not
    clickable on YouTube, so the Related video link is the only tap-through."""
    pinned = f'Full story on my channel: "{title}"' if title else "Full story on my channel."
    lines = [
        "",
        "=" * 60,
        "SHORTS PROMOTION",
        "",
        "After publishing each Short, open it in YouTube Studio (desktop)",
        "and set this video as its RELATED VIDEO — that is the only",
        "clickable path from a Short. Links in Shorts comments and",
        "descriptions are NOT clickable.",
        "",
        f"Pinned comment for every Short: {pinned}",
    ]
    if shorts:
        lines.append("")
        lines.append("Rendered shorts (stagger over the first week, evenings):")
        for i, sh in enumerate(shorts, start=1):
            name = Path(sh.get("path") or "").name
            hook = (sh.get("hook") or "").strip()
            entry = f"  {i}. scene {sh.get('scene_index', 0) + 1} — {name}"
            if hook:
                entry += f'  (hook: "{hook}")'
            lines.append(entry)
    return "\n".join(lines) + "\n"


def export_project(
    project_dir: Path,
    title: str,
    project_id: str,
    metadata_text: str | None = None,
) -> Path:
    """Copy final video, images, script, and metadata to output folder."""
    slug = slugify(title) if title else project_id
    output_dir = config.OUTPUT_DIR / slug

    # If folder already exists, add project_id suffix to avoid overwrite
    if output_dir.exists():
        output_dir = config.OUTPUT_DIR / f"{slug}--{project_id[:8]}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy final video
    video = project_dir / "final.mp4"
    if video.exists():
        shutil.copy2(video, output_dir / "final.mp4")

    # Copy all scene images
    images_src = project_dir / "images"
    if images_src.exists():
        images_dst = output_dir / "images"
        if images_dst.exists():
            shutil.rmtree(images_dst)
        shutil.copytree(images_src, images_dst)

    # Copy all audio files
    audio_src = project_dir / "audio"
    if audio_src.exists():
        audio_dst = output_dir / "audio"
        if audio_dst.exists():
            shutil.rmtree(audio_dst)
        shutil.copytree(audio_src, audio_dst)

    # Copy script
    script_file = project_dir / "script.json"
    if script_file.exists():
        shutil.copy2(script_file, output_dir / "script.json")

    # Write YouTube metadata
    if metadata_text:
        (output_dir / "youtube_metadata.txt").write_text(metadata_text, encoding="utf-8")

    log.info(f"Project exported to {output_dir}")
    return output_dir
