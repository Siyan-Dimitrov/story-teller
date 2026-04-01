"""Batch chapter analysis and sequential pipeline runner."""

import json
import logging
import re
import uuid
from typing import Optional

import httpx

from . import config
from . import project_store as store
from . import script_gen, voice_gen, image_gen
from .video_assembly import assemble_video
from .models import DEFAULT_VOICE_INSTRUCT

log = logging.getLogger(__name__)

# ── In-memory batch progress tracking ────────────────────────

_batch_progress: dict[str, dict] = {}
_batch_paused: dict[str, bool] = {}
_batch_run_config: dict[str, dict] = {}


def get_batch_progress(group_id: str) -> dict:
    """Get current batch progress. Returns stored progress or reconstructs from disk."""
    if group_id in _batch_progress:
        return _batch_progress[group_id]
    # Reconstruct from project states on disk
    return _reconstruct_progress(group_id)


def _reconstruct_progress(group_id: str) -> dict:
    """Rebuild progress from project state files for a given group."""
    projects = store.list_projects()
    group_projects = [
        p for p in projects
        if p.get("book_group_id") == group_id
    ]
    group_projects.sort(key=lambda p: p.get("chapter_index", 0))

    chapters = []
    completed = 0
    failed = 0
    for p in group_projects:
        step = p.get("step", "created")
        error = p.get("error")
        if step == "assembled":
            status = "completed"
            completed += 1
        elif error:
            status = "failed"
            failed += 1
        else:
            status = "pending"
        chapters.append({
            "project_id": p["project_id"],
            "chapter_index": p.get("chapter_index", 0),
            "title": p.get("title", ""),
            "status": status,
            "current_step": None,
            "failed_step": step if error else None,
            "error": error,
        })

    return {
        "group_id": group_id,
        "total": len(chapters),
        "completed": completed,
        "failed": failed,
        "current_chapter": None,
        "current_step": None,
        "chapters": chapters,
        "finished": True,
        "paused": False,
    }


# ── Chapter analysis: two-stage (regex split → LLM classify) ─

# Patterns that indicate chapter boundaries, ordered by specificity
_CHAPTER_PATTERNS = [
    # "CHAPTER I", "Chapter 1", "CHAPTER THE FIRST", etc.
    re.compile(r"^\s*(CHAPTER|Chapter)\s+[\dIVXLCDMivxlcdm]+\.?(?:\s*[.:\-—]\s*.*)?$", re.MULTILINE),
    # "CHAPTER I. Title Text" or "CHAPTER 1 — Title"
    re.compile(r"^\s*(CHAPTER|Chapter)\s+\w+.*$", re.MULTILINE),
    # "Part I", "PART ONE", "Part 1"
    re.compile(r"^\s*(PART|Part)\s+[\dIVXLCDMivxlcdm]+\.?\s*$", re.MULTILINE),
    # Bare Roman numerals on their own line: "I.", "II.", "III.", "IV." (at least "I.")
    re.compile(r"^\s*[IVXLCDM]{1,6}\.?\s*$", re.MULTILINE),
    # Story collections: all-caps title lines surrounded by blank lines
    re.compile(r"\n\n\s*([A-Z][A-Z\s\-':,]{4,}[A-Z])\s*\n\n", re.MULTILINE),
]

# Minimum chapter length in characters — skip tiny fragments
_MIN_CHAPTER_CHARS = 500

# How many chars of each chapter to send to the LLM for classification
_CLASSIFY_EXCERPT_CHARS = 2000

CHAPTER_CLASSIFY_PROMPT = """You are a literary analyst. Given the TITLE LINE and OPENING EXCERPT of one chapter from a book, return metadata about it.

Return ONLY valid JSON (no markdown fences):
{
  "title": "A clean chapter title (e.g. 'The Juniper Tree', not 'CHAPTER IV')",
  "suggested_tone": "dark, whimsical, tragic, gothic, humorous, romantic, etc.",
  "summary": "1-2 sentence summary of what this chapter/story is about based on the excerpt"
}

Guidelines:
- If the heading is just "CHAPTER IV" with no subtitle, infer a title from the content
- suggested_tone should reflect the mood of the excerpt
- The summary should capture the key premise or opening situation
"""


def _split_chapters_by_pattern(text: str) -> list[dict]:
    """Stage 1: Use regex to find chapter boundaries in the raw text.

    Returns a list of {"heading": str, "start": int, "end": int} dicts.
    """
    for pattern in _CHAPTER_PATTERNS:
        matches = list(pattern.finditer(text))
        if len(matches) >= 2:
            chapters = []
            for i, m in enumerate(matches):
                start = m.start()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                body = text[start:end].strip()
                if len(body) < _MIN_CHAPTER_CHARS:
                    continue
                chapters.append({
                    "heading": m.group(0).strip(),
                    "start": start,
                    "end": end,
                })
            if len(chapters) >= 2:
                log.info(f"Chapter split: pattern {pattern.pattern!r} found {len(chapters)} chapters")
                return chapters
    return []


def _split_chapters_by_blanks(text: str) -> list[dict]:
    """Fallback: split on large gaps of whitespace (3+ blank lines)."""
    parts = re.split(r"\n\s*\n\s*\n\s*\n", text)
    chapters = []
    offset = 0
    for part in parts:
        part_stripped = part.strip()
        if len(part_stripped) >= _MIN_CHAPTER_CHARS:
            start = text.find(part_stripped[:80], offset)
            if start < 0:
                start = offset
            end = start + len(part_stripped)
            # Use the first line as heading
            first_line = part_stripped.split("\n", 1)[0].strip()
            chapters.append({
                "heading": first_line[:120],
                "start": start,
                "end": end,
            })
            offset = end
    return chapters


async def _classify_chapter(
    heading: str,
    excerpt: str,
    book_title: str,
    model: str,
    base_url: str,
) -> dict:
    """Stage 2: Send a short excerpt to the LLM to get title, tone, and summary."""
    user_prompt = f"Book: {book_title}\nHeading: {heading}\n\nExcerpt:\n{excerpt}"

    try:
        async with httpx.AsyncClient(timeout=config.LLM_TIMEOUT_SECONDS) as client:
            resp = await client.post(
                f"{base_url}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": CHAPTER_CLASSIFY_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 1000,
                    },
                },
            )
        if resp.status_code != 200:
            log.warning(f"LLM classify failed ({resp.status_code}), using defaults")
            return {}

        data = resp.json()
        content = script_gen._extract_llm_content(data)
        if not content.strip():
            return {}

        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines)

        return json.loads(content)
    except Exception as exc:
        log.warning(f"LLM classify error for '{heading[:50]}': {exc}")
        return {}


async def analyze_chapters(
    text: str,
    book_title: str = "",
    ollama_model: str | None = None,
) -> dict:
    """Two-stage chapter analysis: regex split then LLM classify.

    Stage 1 — scan the raw text for chapter headings using regex patterns.
              No LLM call, works on any size book instantly.
    Stage 2 — for each detected chapter, send only the heading + first ~2000
              chars to the LLM for title/tone/summary classification.
              Small, fast, reliable calls.
    """
    model = ollama_model or config.OLLAMA_MODEL
    base_url = config.OLLAMA_URL

    log.info(f"Analyzing chapters: title={book_title!r}, model={model}, text_len={len(text)}")

    # Stage 1: regex-based chapter splitting
    raw_chapters = _split_chapters_by_pattern(text)
    if not raw_chapters:
        log.info("No chapter pattern matched, trying blank-line split")
        raw_chapters = _split_chapters_by_blanks(text)

    if not raw_chapters:
        # Last resort: treat the entire text as one chapter
        log.warning("Could not detect chapters, treating entire text as one chapter")
        raw_chapters = [{"heading": book_title or "Full Text", "start": 0, "end": len(text)}]

    # Stage 2: LLM classification of each chapter excerpt
    chapters = []
    for i, ch in enumerate(raw_chapters):
        chapter_text = text[ch["start"]:ch["end"]].strip()
        excerpt = chapter_text[:_CLASSIFY_EXCERPT_CHARS]

        meta = await _classify_chapter(
            heading=ch["heading"],
            excerpt=excerpt,
            book_title=book_title,
            model=model,
            base_url=base_url,
        )

        char_count = len(chapter_text)
        # Clean up heading for fallback title
        fallback_title = ch["heading"][:80].strip()
        if not fallback_title or fallback_title.upper() == fallback_title and len(fallback_title) < 15:
            fallback_title = f"Chapter {i + 1}"

        chapters.append({
            "title": meta.get("title", fallback_title),
            "text": chapter_text,
            "suggested_tone": meta.get("suggested_tone", "dark"),
            "summary": meta.get("summary", ""),
            "estimated_duration": estimate_duration(char_count),
            "char_count": char_count,
        })
        log.info(f"Chapter {i + 1}/{len(raw_chapters)}: '{chapters[-1]['title']}' ({char_count} chars)")

    result_title = book_title or "Unknown"
    log.info(f"Chapter analysis complete: {len(chapters)} chapters detected in '{result_title}'")

    return {
        "book_title": result_title,
        "chapters": chapters,
    }


def estimate_duration(char_count: int) -> float:
    """Estimate narration duration in minutes from character count.

    Calculates based on BATCH_NARRATION_RATE (characters per minute).
    Minimum duration is 1 minute to avoid impossibly short targets.
    No maximum cap — long chapters can have extended durations.
    """
    minutes = char_count / config.BATCH_NARRATION_RATE
    return max(1.0, round(minutes, 1))


# ── Batch project creation ───────────────────────────────────

def create_batch_projects(
    book_title: str,
    chapters: list[dict],
    ollama_model: str = "kimi-k2.5:cloud",
    voice_profile_id: str | None = None,
    voice_language: str = "en",
    image_backend: str = "comfyui",
) -> tuple[str, list[str]]:
    """Create one project per chapter, linked by a book_group_id."""
    book_group_id = uuid.uuid4().hex[:12]
    project_ids = []

    for i, ch in enumerate(chapters):
        pid, pdir = store.create_project()

        chapter_title = f"{book_title} — {ch['title']}" if book_title else ch["title"]

        full_text = ch.get("text", "")
        store.update_state(
            pid,
            title=chapter_title,
            book_group_id=book_group_id,
            book_title=book_title,
            chapter_index=i,
            ollama_model=ollama_model,
            target_minutes=ch.get("estimated_duration", 5.0),
            tone=ch.get("suggested_tone", "dark"),
            custom_prompt=full_text,
            image_backend=image_backend,
        )
        # Also save full text to a file for reference/debugging
        if full_text:
            (pdir / "source_text.txt").write_text(full_text, encoding="utf-8")
            log.info(f"Chapter {i}: saved {len(full_text):,} chars to source_text.txt")
        if voice_profile_id:
            store.update_state(pid, voice_profile_id=voice_profile_id, voice_language=voice_language)

        project_ids.append(pid)
        log.info(f"Created chapter project {pid}: '{chapter_title}' (ch {i})")

    log.info(f"Batch created: group={book_group_id}, {len(project_ids)} chapters from '{book_title}'")
    return book_group_id, project_ids


# ── Sequential batch pipeline runner ─────────────────────────

STEP_ORDER = ["script", "voice", "images", "qc", "animate", "assemble"]


async def run_batch_pipeline(
    group_id: str,
    project_ids: list[str],
    steps: list[str],
    voice_profile_id: str = "",
    voice_language: str = "en",
    voice_instruct: str = DEFAULT_VOICE_INSTRUCT,
    image_backend: str = "comfyui",
    style_prompt: str = "dark fairy tale illustration, gothic storybook art",
    lora_keys: list[str] | None = None,
):
    """Process chapters sequentially through the pipeline."""
    # Initialize progress — skip already-completed chapters
    chapters_progress = []
    already_completed = 0
    for pid in project_ids:
        state = store.load_state(pid)
        is_done = state.get("step") == "assembled"
        chapters_progress.append({
            "project_id": pid,
            "chapter_index": state.get("chapter_index", 0),
            "title": state.get("title", ""),
            "status": "completed" if is_done else "pending",
            "current_step": None,
            "failed_step": None,
            "error": None,
        })
        if is_done:
            already_completed += 1

    _batch_progress[group_id] = {
        "group_id": group_id,
        "total": len(project_ids),
        "completed": already_completed,
        "failed": 0,
        "current_chapter": None,
        "current_step": None,
        "chapters": chapters_progress,
        "finished": False,
        "paused": False,
    }

    # Store run config for pause/resume
    _batch_paused[group_id] = False
    _batch_run_config[group_id] = {
        "steps": steps,
        "project_ids": project_ids,
        "voice_profile_id": voice_profile_id,
        "voice_language": voice_language,
        "voice_instruct": voice_instruct,
        "image_backend": image_backend,
        "style_prompt": style_prompt,
        "lora_keys": lora_keys,
    }

    for i, pid in enumerate(project_ids):
        progress = _batch_progress[group_id]

        # Skip already-completed chapters
        if progress["chapters"][i]["status"] == "completed":
            log.info(f"Batch {group_id}: chapter {i} ({pid}) already completed, skipping")
            continue

        # Check for pause request
        if _batch_paused.get(group_id, False):
            progress["current_chapter"] = None
            progress["current_step"] = None
            progress["paused"] = True
            log.info(f"Batch {group_id}: paused before chapter {i}")
            return

        progress["current_chapter"] = i
        progress["chapters"][i]["status"] = "running"

        try:
            await _run_chapter_pipeline(
                pid, i, group_id, steps,
                voice_profile_id=voice_profile_id,
                voice_language=voice_language,
                voice_instruct=voice_instruct,
                image_backend=image_backend,
                style_prompt=style_prompt,
                lora_keys=lora_keys,
            )
            progress["completed"] += 1
            progress["chapters"][i]["status"] = "completed"
            log.info(f"Batch {group_id}: chapter {i} ({pid}) completed")
        except Exception as e:
            log.error(f"Batch {group_id}: chapter {i} ({pid}) failed: {e}")
            progress["failed"] += 1
            progress["chapters"][i]["status"] = "failed"
            progress["chapters"][i]["error"] = str(e)
            # Continue to next chapter

        progress["chapters"][i]["current_step"] = None

    # Mark batch as finished
    progress = _batch_progress[group_id]
    progress["current_chapter"] = None
    progress["current_step"] = None
    progress["paused"] = False
    progress["finished"] = True
    log.info(
        f"Batch {group_id} finished: {progress['completed']}/{progress['total']} completed, "
        f"{progress['failed']} failed"
    )


async def _run_chapter_pipeline(
    project_id: str,
    chapter_index: int,
    group_id: str,
    steps: list[str],
    voice_profile_id: str = "",
    voice_language: str = "en",
    voice_instruct: str = DEFAULT_VOICE_INSTRUCT,
    image_backend: str = "comfyui",
    style_prompt: str = "dark fairy tale illustration, gothic storybook art",
    lora_keys: list[str] | None = None,
):
    """Run pipeline steps for a single chapter project."""
    state = store.load_state(project_id)
    pdir = store.project_dir(project_id)
    progress = _batch_progress[group_id]

    def _update_step(step_name: str):
        progress["current_step"] = step_name
        progress["chapters"][chapter_index]["current_step"] = step_name

    # Step 1: Script generation
    if "script" in steps:
        _update_step("script")
        store.update_state(project_id, step="generating_script", error=None)
        script = await script_gen.generate_script(
            source_tale=state.get("source_tale", ""),
            custom_prompt=state.get("custom_prompt", ""),
            target_minutes=state.get("target_minutes", 5.0),
            ollama_model=state.get("ollama_model"),
            tone=state.get("tone", ""),
        )
        store.save_json(project_id, "script.json", script)
        store.update_state(project_id, step="scripted", title=script.get("title", state.get("title", "")))

    # Load script for subsequent steps
    script = store.load_json(project_id, "script.json")
    if not script:
        raise RuntimeError(f"No script found for project {project_id}")

    # Step 2: Voice generation
    if "voice" in steps:
        if not voice_profile_id:
            raise RuntimeError("Voice profile ID required for voice generation")
        _update_step("voice")
        store.update_state(project_id, step="generating_voice", error=None,
                          voice_profile_id=voice_profile_id, voice_language=voice_language)
        scenes = await voice_gen.generate_all_scenes(
            scenes=script["scenes"],
            profile_id=voice_profile_id,
            language=voice_language,
            project_dir=pdir,
            instruct=voice_instruct,
        )
        script["scenes"] = scenes
        store.save_json(project_id, "script.json", script)
        store.update_state(project_id, step="voiced")

    # Step 3: Image generation
    if "images" in steps:
        _update_step("images")
        store.update_state(project_id, step="generating_images", error=None, image_backend=image_backend)
        scenes = await image_gen.generate_all_scenes(
            scenes=script["scenes"],
            project_dir=pdir,
            backend=image_backend,
            style_prompt=style_prompt,
            lora_keys=lora_keys,
        )
        script["scenes"] = scenes
        store.save_json(project_id, "script.json", script)
        store.update_state(project_id, step="illustrated")

    # Step 4: QC (optional)
    if "qc" in steps:
        _update_step("qc")
        from .image_qc import run_qc_for_project
        store.update_state(project_id, step="qc_running", error=None)
        scenes = await run_qc_for_project(
            scenes=script["scenes"],
            project_dir=pdir,
            vision_model=config.OLLAMA_VISION_MODEL,
            style_prompt=style_prompt,
            pass_threshold=config.QC_PASS_THRESHOLD,
            project_id=project_id,
        )
        script["scenes"] = scenes
        store.save_json(project_id, "script.json", script)
        store.update_state(project_id, step="qc_passed")

    # Step 5: Animate (optional)
    if "animate" in steps:
        _update_step("animate")
        from .animation import prepare_animations
        store.update_state(project_id, step="animating", error=None)
        scenes = await prepare_animations(
            scenes=script["scenes"],
            project_dir=pdir,
            ollama_model=state.get("ollama_model"),
            project_id=project_id,
        )
        script["scenes"] = scenes
        store.save_json(project_id, "script.json", script)
        store.update_state(project_id, step="animated")

    # Step 6: Assemble video
    if "assemble" in steps:
        _update_step("assemble")
        store.update_state(project_id, step="assembling", error=None)
        output, duration = assemble_video(
            scenes=script["scenes"],
            project_dir=pdir,
            project_id=project_id,
        )
        store.update_state(project_id, step="assembled")

        # Export
        try:
            from .export import export_project, generate_youtube_metadata
            title = script.get("title", state.get("title", ""))
            synopsis = script.get("synopsis", "")
            tone = state.get("tone", "dark")
            themes = list({s.get("mood", "") for s in script.get("scenes", []) if s.get("mood")})

            metadata = await generate_youtube_metadata(
                title=title,
                synopsis=synopsis,
                tone=tone,
                themes=themes,
                scene_count=len(script.get("scenes", [])),
                ollama_model=state.get("ollama_model"),
                book_title=state.get("book_title", ""),
            )
            out_dir = export_project(
                project_dir=pdir,
                title=title,
                project_id=project_id,
                metadata_text=metadata,
            )
            store.update_state(project_id, output_dir=str(out_dir))
        except Exception as ex:
            log.error(f"Export failed for {project_id} (video OK): {ex}")


# ── Pause / resume helpers ─────────────────────────────────

def pause_batch(group_id: str) -> bool:
    """Request the batch pipeline to pause after the current chapter."""
    if group_id not in _batch_progress:
        return False
    if _batch_progress[group_id].get("finished", True):
        return False
    _batch_paused[group_id] = True
    return True


def resume_batch(group_id: str) -> dict | None:
    """Clear pause flag and return stored config so the endpoint can re-launch."""
    _batch_paused[group_id] = False
    progress = _batch_progress.get(group_id)
    if progress:
        progress["paused"] = False
        progress["finished"] = False
    return _batch_run_config.get(group_id)
