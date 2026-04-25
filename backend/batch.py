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

# Patterns that indicate chapter boundaries, ordered by specificity.
# All patterns are tested against \n-normalised text.
_CHAPTER_PATTERNS = [
    # "CHAPTER I. Title Text" or "CHAPTER 1 — Title" (keyword + number + title on SAME line)
    re.compile(r"^\s*(?:CHAPTER|BOOK)\s+(?:\d+|[IVXLCDM]+)[.:\-—]+[ \t]+\S.+$", re.MULTILINE | re.IGNORECASE),
    # "CHAPTER I", "Chapter 1", "CHAPTER THE FIRST" (keyword + number, standalone)
    re.compile(r"^\s*(?:CHAPTER|BOOK)\s+(?:\d+|[IVXLCDM]+)\.?\s*$", re.MULTILINE | re.IGNORECASE),
    # "Part I", "PART ONE", "Part 1"
    re.compile(r"^\s*PART\s+(?:\d+|[IVXLCDM]+)\.?\s*$", re.MULTILINE | re.IGNORECASE),
    # Roman numerals (≥2 chars to avoid "I", "V", "C", "D", "M") with title on same line
    re.compile(r"^\s*[IVXLCDM]{2,}[.:\-—]+[ \t]+\S.+$", re.MULTILINE),
    # Bare roman numerals ≥2 chars: "II.", "XIV"
    re.compile(r"^\s*[IVXLCDM]{2,}\.?\s*$", re.MULTILINE),
    # Numeric with title on same line: "1. Chapter Title", "2 - Next Chapter"
    re.compile(r"^\s*\d{1,3}[.:\-—]+[ \t]+\S.+$", re.MULTILINE),
    # Story collections: all-caps title lines (≥6 chars) surrounded by blank lines
    re.compile(r"\n\n\s*([A-Z][A-Z\s\-':,]{4,}[A-Z])\s*\n\n"),
    # Section breaks: "***", "---", "___"
    re.compile(r"^\s*[*_\-]{3,}\s*$", re.MULTILINE),
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


def _strip_toc_matches(matches: list) -> list:
    """Remove a dense cluster of regex matches at the start (table of contents).

    TOC entries sit very close together (< 200 chars apart).  Body chapters
    have thousands of chars between headings.  We walk forward from the start
    and drop matches until we hit a real gap.
    """
    if len(matches) < 6:
        return matches

    toc_end = 0
    for i in range(1, len(matches)):
        gap = matches[i].start() - matches[i - 1].start()
        if gap > 500:
            break
        toc_end = i

    if toc_end >= 5:
        log.debug("Stripped %d TOC-cluster matches at start of text", toc_end + 1)
        return matches[toc_end + 1:]
    return matches


def _score_split(chapters: list[dict], text_len: int) -> float:
    """Score how reasonable a chapter split is.  Higher = better, 0 = invalid."""
    if len(chapters) < 2 or text_len == 0:
        return 0.0
    sizes = [ch["end"] - ch["start"] for ch in chapters]
    median = sorted(sizes)[len(sizes) // 2]
    # Reject if median chapter is tiny (likely false-positive matches)
    if median < 800:
        return 0.0
    # Reject if too many chapters for the text
    if len(chapters) > max(300, text_len // 400):
        return 0.0
    mean = sum(sizes) / len(sizes)
    coverage = sum(sizes) / text_len
    # Consistency: coefficient of variation (lower = more uniform chapters)
    std = (sum((s - mean) ** 2 for s in sizes) / len(sizes)) ** 0.5
    cv = std / mean if mean > 0 else 0
    consistency = max(0.1, 1.0 - cv * 0.3)
    # Prefer reasonable chapter counts, gently penalise extremes
    if 5 <= len(chapters) <= 150:
        count_factor = 1.0
    elif 2 <= len(chapters) < 5 or 150 < len(chapters) <= 250:
        count_factor = 0.7
    else:
        count_factor = 0.3
    return coverage * consistency * count_factor


def _split_chapters_by_pattern(text: str) -> list[dict]:
    """Stage 1: Use regex to find chapter boundaries in the raw text.

    Tries every pattern and keeps the one that produces the best-scoring
    split, avoiding the old first-match-wins pitfall.

    Returns a list of {"heading": str, "start": int, "end": int} dicts.
    """
    # Normalise line endings — Gutenberg texts use \r\n which breaks $ anchors
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    best_chapters: list[dict] = []
    best_score = 0.0
    best_pattern = ""

    for pattern in _CHAPTER_PATTERNS:
        matches = list(pattern.finditer(text))
        if len(matches) < 2:
            continue

        # Strip dense TOC cluster at the start before building chapters
        matches = _strip_toc_matches(matches)
        if len(matches) < 2:
            continue

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

        score = _score_split(chapters, len(text))
        log.debug(
            "Pattern %r: %d matches → %d chapters, score=%.3f",
            pattern.pattern, len(matches), len(chapters), score,
        )
        if score > best_score:
            best_score = score
            best_chapters = chapters
            best_pattern = pattern.pattern

    if best_chapters:
        log.info(
            "Chapter split: chose %r with %d chapters (score=%.3f)",
            best_pattern, len(best_chapters), best_score,
        )
    else:
        log.info("No chapter pattern produced a valid split in %s chars of text", f"{len(text):,}")

    return best_chapters


def _split_chapters_by_blanks(text: str) -> list[dict]:
    """Fallback: split on large gaps of whitespace (3+ blank lines) or section breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Try section breaks first (common in Project Gutenberg)
    section_breaks = list(re.finditer(r"\n\s*[*_=\-#~]{3,}\s*\n", text))
    if len(section_breaks) >= 2:
        chapters = []
        for i, m in enumerate(section_breaks):
            start = m.end()  # Start after the break
            end = section_breaks[i + 1].start() if i + 1 < len(section_breaks) else len(text)
            body = text[start:end].strip()
            if len(body) >= _MIN_CHAPTER_CHARS:
                # Use first line as heading
                first_line = body.split("\n", 1)[0].strip()[:120]
                chapters.append({
                    "heading": first_line or f"Section {i + 1}",
                    "start": start,
                    "end": end,
                })
        if len(chapters) >= 2:
            log.info(f"Chapter split: found {len(chapters)} chapters by section breaks")
            return chapters

    # Fall back to paragraph gaps
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
    if chapters:
        log.info(f"Chapter split: found {len(chapters)} chapters by paragraph gaps")
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
                        "num_predict": 4000,
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

    # Normalise line endings once so start/end offsets are consistent
    text = text.replace("\r\n", "\n").replace("\r", "\n")
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


# ── Text splitting helper ───────────────────────────────────


def _split_text_into_parts(text: str, n: int) -> list[str]:
    """Split *text* into *n* roughly equal parts at paragraph boundaries.

    Never cuts mid-sentence — finds the nearest double-newline to each
    target split point.  Returns exactly *n* non-empty strings.
    """
    if n <= 1:
        return [text]

    text = text.strip()
    target_size = len(text) / n
    parts: list[str] = []
    start = 0

    for part_idx in range(n - 1):
        ideal = int(target_size * (part_idx + 1))
        # Search for nearest paragraph break (double newline) around the ideal point
        best = -1
        best_dist = len(text)
        search_start = max(start + 200, ideal - 2000)
        search_end = min(len(text) - 200, ideal + 2000)
        pos = search_start
        while pos < search_end:
            pos = text.find("\n\n", pos)
            if pos == -1 or pos >= search_end:
                break
            dist = abs(pos - ideal)
            if dist < best_dist:
                best = pos
                best_dist = dist
            pos += 1

        if best == -1:
            # No paragraph break found — split at ideal point
            best = ideal

        parts.append(text[start:best].strip())
        start = best

    parts.append(text[start:].strip())

    # Filter out empty parts (shouldn't happen, but be safe)
    parts = [p for p in parts if p]
    return parts


# ── Batch project creation ───────────────────────────────────

def create_batch_projects(
    book_title: str,
    chapters: list[dict],
    ollama_model: str = "kimi-k2.5:cloud",
    voice_profile_id: str | None = None,
    voice_language: str = "en",
    image_backend: str = "comfyui",
) -> tuple[str, list[str]]:
    """Create one project per chapter (or per part), linked by a book_group_id.

    If a chapter has ``parts`` > 1, its text is split into that many
    sub-parts, each becoming its own project.
    """
    book_group_id = uuid.uuid4().hex[:12]
    project_ids = []
    project_index = 0

    for ch in chapters:
        num_parts = max(1, int(ch.get("parts", 1)))
        full_text = ch.get("text", "")
        base_title = ch.get("title", "")
        tone = ch.get("suggested_tone", "dark")
        total_duration = ch.get("estimated_duration", 5.0)

        if num_parts > 1 and full_text:
            part_texts = _split_text_into_parts(full_text, num_parts)
        else:
            part_texts = [full_text]

        for part_idx, part_text in enumerate(part_texts):
            pid, pdir = store.create_project()

            if len(part_texts) > 1:
                chapter_title = f"{book_title} — {base_title} — Part {part_idx + 1}/{len(part_texts)}" if book_title else f"{base_title} — Part {part_idx + 1}/{len(part_texts)}"
                part_duration = max(1.0, round(total_duration / len(part_texts), 1))
            else:
                chapter_title = f"{book_title} — {base_title}" if book_title else base_title
                part_duration = total_duration

            store.update_state(
                pid,
                title=chapter_title,
                book_group_id=book_group_id,
                book_title=book_title,
                chapter_index=project_index,
                ollama_model=ollama_model,
                target_minutes=part_duration,
                tone=tone,
                custom_prompt=part_text,
                image_backend=image_backend,
            )
            if part_text:
                (pdir / "source_text.txt").write_text(part_text, encoding="utf-8")
                log.info(f"Project {project_index}: saved {len(part_text):,} chars to source_text.txt")
            if voice_profile_id:
                store.update_state(pid, voice_profile_id=voice_profile_id, voice_language=voice_language)

            project_ids.append(pid)
            log.info(f"Created project {pid}: '{chapter_title}' (idx {project_index})")
            project_index += 1

    log.info(f"Batch created: group={book_group_id}, {len(project_ids)} projects from '{book_title}'")
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
    character_consistency: bool = False,
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
        "character_consistency": character_consistency,
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
                character_consistency=character_consistency,
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
    character_consistency: bool = False,
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
            character_consistency=character_consistency,
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
