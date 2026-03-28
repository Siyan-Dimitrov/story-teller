"""Batch chapter analysis and sequential pipeline runner."""

import json
import logging
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
    }


# ── Chapter analysis via LLM ────────────────────────────────

CHAPTER_ANALYSIS_PROMPT = """You are a literary analyst. Given a book's full text, identify all chapters, parts, stories, or tales within it.

Return ONLY valid JSON (no markdown fences). Use this exact structure:
{
  "book_title": "The detected or confirmed book title",
  "chapters": [
    {
      "title": "Chapter/story title",
      "start_marker": "First unique phrase of this chapter (10-20 words, must appear exactly in the text)",
      "end_marker": "Last unique phrase of this chapter (10-20 words, must appear exactly in the text) or __END__ if last chapter",
      "suggested_tone": "dark, whimsical, tragic, gothic, humorous, romantic, etc.",
      "summary": "1-2 sentence summary of what happens in this chapter"
    }
  ]
}

Guidelines:
- Detect chapters by headings like "CHAPTER I", "Part One", "I.", story titles, or narrative breaks
- For story collections (e.g. "Ghost Stories"), each individual story/tale is a separate chapter
- The start_marker must be a unique phrase that appears EXACTLY in the source text — use the first sentence or first distinctive phrase of the chapter
- The end_marker must be a unique phrase near the END of the chapter — use the last sentence or last distinctive phrase. Use __END__ only for the very last chapter
- suggested_tone should reflect the mood of that specific chapter
- Order chapters as they appear in the text
- Do NOT include prefaces, introductions, appendices, or table of contents as chapters
"""


async def analyze_chapters(
    text: str,
    book_title: str = "",
    ollama_model: str | None = None,
) -> dict:
    """Send book text to LLM and get chapter breakdown with boundaries."""
    model = ollama_model or config.OLLAMA_MODEL
    base_url = config.OLLAMA_URL

    user_prompt = f"Analyze this book and identify all chapters/stories/tales.\n"
    if book_title:
        user_prompt += f"Book title: {book_title}\n"
    user_prompt += f"\nFull text ({len(text)} characters):\n\n{text}"

    log.info(f"Analyzing chapters: title={book_title!r}, model={model}, text_len={len(text)}")

    async with httpx.AsyncClient(timeout=config.CHAPTER_ANALYSIS_TIMEOUT_SECONDS) as client:
        resp = await client.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": CHAPTER_ANALYSIS_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Low temp for structured analysis
                    "num_predict": config.CHAPTER_ANALYSIS_MAX_TOKENS,
                },
            },
        )
        if resp.status_code != 200:
            body = resp.text
            log.error(f"Ollama error {resp.status_code}: {body}")
            raise RuntimeError(f"Ollama returned {resp.status_code}: {body}")

    data = resp.json()
    content = script_gen._extract_llm_content(data)

    # Strip markdown fences if present
    if content.startswith("```"):
        lines = content.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines)

    parsed = json.loads(content)

    # Extract actual chapter text using markers
    chapters = []
    raw_chapters = parsed.get("chapters", [])

    for i, ch in enumerate(raw_chapters):
        start_marker = ch.get("start_marker", "")
        end_marker = ch.get("end_marker", "")

        # Find chapter boundaries in the source text
        chapter_text = _extract_chapter_text(text, start_marker, end_marker, raw_chapters, i)

        char_count = len(chapter_text)
        estimated_duration = estimate_duration(char_count)

        chapters.append({
            "title": ch.get("title", f"Chapter {i + 1}"),
            "text": chapter_text,
            "suggested_tone": ch.get("suggested_tone", "dark"),
            "estimated_duration": estimated_duration,
            "char_count": char_count,
        })

    result_title = parsed.get("book_title", book_title or "Unknown")
    log.info(f"Chapter analysis complete: {len(chapters)} chapters detected in '{result_title}'")

    return {
        "book_title": result_title,
        "chapters": chapters,
    }


def _extract_chapter_text(
    full_text: str,
    start_marker: str,
    end_marker: str,
    all_chapters: list[dict],
    chapter_index: int,
) -> str:
    """Extract chapter text between start and end markers."""
    # Find start position
    start_pos = 0
    if start_marker:
        idx = full_text.find(start_marker)
        if idx >= 0:
            start_pos = idx
        else:
            # Try case-insensitive partial match
            lower_text = full_text.lower()
            lower_marker = start_marker.lower()
            idx = lower_text.find(lower_marker)
            if idx >= 0:
                start_pos = idx

    # Find end position
    end_pos = len(full_text)
    if end_marker and end_marker != "__END__":
        idx = full_text.find(end_marker, start_pos)
        if idx >= 0:
            end_pos = idx + len(end_marker)
        else:
            # Try case-insensitive
            lower_text = full_text.lower()
            lower_marker = end_marker.lower()
            idx = lower_text.find(lower_marker, start_pos)
            if idx >= 0:
                end_pos = idx + len(end_marker)
            else:
                # Fallback: use next chapter's start marker
                if chapter_index + 1 < len(all_chapters):
                    next_start = all_chapters[chapter_index + 1].get("start_marker", "")
                    if next_start:
                        idx = full_text.find(next_start, start_pos + 1)
                        if idx >= 0:
                            end_pos = idx

    return full_text[start_pos:end_pos].strip()


def estimate_duration(char_count: int) -> float:
    """Estimate narration duration in minutes from character count."""
    minutes = char_count / config.BATCH_NARRATION_RATE
    return max(1.0, min(15.0, round(minutes, 1)))


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

        store.update_state(
            pid,
            title=chapter_title,
            book_group_id=book_group_id,
            chapter_index=i,
            ollama_model=ollama_model,
            target_minutes=ch.get("estimated_duration", 5.0),
            tone=ch.get("suggested_tone", "dark"),
            custom_prompt=ch.get("text", ""),
            image_backend=image_backend,
        )
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
    }

    for i, pid in enumerate(project_ids):
        progress = _batch_progress[group_id]

        # Skip already-completed chapters
        if progress["chapters"][i]["status"] == "completed":
            log.info(f"Batch {group_id}: chapter {i} ({pid}) already completed, skipping")
            continue

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
