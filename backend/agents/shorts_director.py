"""ShortsDirector — produce vertical 9:16 hook shorts from a finished story.

Rewritten after the v1 critique: v1 hooks opened with riddle-questions,
ran 25-30s, kept a static headline pinned over a single Ken-Burns'd still
for the whole clip, and were forbidden from naming the antagonist or the
transgression — i.e. the actually-hookable moments.

v2 rules:
  1) Pick a hook scene from the FIRST 60% of the script (anti-resolution clamp).
  2) Write a 30-50 word narration (12-18s spoken) that opens with a CONCRETE
     present-tense image, NOT a rhetorical question. Reveal the antagonist
     and the transgression. Forbid only the resolution.
  3) Produce a 4-6 word headline anchored on a concrete noun + transgressive
     verb. No metaphors, no abstractions.
  4) Director also collects 2-3 ADDITIONAL bed images from candidate scenes
     (chosen scene + neighbors) so the renderer can cut between them.

Multi-shorts: a single LLM call returns N hooks on distinct scenes, then we
voice + render sequentially.

Output: projects/{id}/shorts/short_NN.mp4
State: project state.shorts_paths = ["shorts/short_01.mp4", ...]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .. import project_store as store
from .. import shorts_assembly, voice_gen
from ..models import DEFAULT_VOICE_INSTRUCT
from .base import call_llm_json, LLMCallError

log = logging.getLogger(__name__)


# Keep the hook to the first 60% of scenes — anything later risks spoiling
# the climax. 0.6 lands at scene 6 of 10, scene 12 of 20, etc.
HOOK_SCENE_CUTOFF_PCT = 0.6

# v2 word target: 30-50 words = 12-18s spoken at storyteller pace. The v1
# 55-85 range was audiobook-length and produced 27s of narration — too long
# for a scroll-stop format where the promise has to land in <15s.
HOOK_TARGET_WORDS_MIN = 30
HOOK_TARGET_WORDS_MAX = 50

# Hard cap on shorts per call.
MAX_SHORTS_PER_CALL = 3

# How many DIFFERENT images to gather for the visual bed of a single short.
# 3 = chosen scene + one before + one after, cuts every ~5s. Looks alive
# vs a single still Ken-Burns'd for 15s straight.
BED_IMAGES_PER_SHORT = 3


_SYSTEM_PROMPT_SINGLE = """You are a YouTube Shorts hook writer for narrated dark-fairy-tale videos. Your output becomes a 15-25 second vertical clip that has to stop a thumb mid-scroll on a muted phone.

Reply with strict JSON only. No prose, no markdown fences.

Schema:
{
  "selected_scene_index": <int>,
  "selected_scene_rationale": "<one sentence on why this scene is the strongest visual hook>",
  "headline": "<4-6 word headline: a concrete noun + a transgressive verb. NO metaphors. NO abstractions.>",
  "narration": "<30-50 word fresh hook narration. Opens with a CONCRETE present-tense image, NOT a question. Reveals the antagonist and the transgression. Ends mid-action, before the rescue.>"
}

HEADLINE rules:
- 4-6 words, all caps not required.
- Must contain a concrete noun (mother, shirts, nettles, blade, cradle, well) and a verb of action or transgression (sews, drowns, devours, sells, burns, steals).
- BAN metaphors and adjectives-as-nouns ("the breathing", "the listening", "the watching").
- BAN literary sibilance for its own sake. Plain is better than poetic.
- GOOD: "Mother Sews Skin Into Shirts" / "She Drowns Six Sons At Dawn" / "Witch Cooks Children In Honey"
- BAD: "She Sewed Shirts That Breathed" / "What Waited In The Woods" / "The Mother's Curse"

NARRATION rules:
- 30-50 words. Aim for 40. Counts as 12-18 seconds spoken.
- First sentence MUST be a present-tense image with a subject and a menacing verb. NO rhetorical questions, NO "What if…", NO "Have you ever…", NO "Imagine…".
- BAN question-mark openers entirely. The first sentence does not contain a "?".
- Name the antagonist. Name the transgression. Show the moment of harm.
- Forbidden: how the curse breaks, who rescues whom, what the ending is. Show the wound, hide the bandage.
- Forbidden: meta-references ("in this video", "watch the full story", "subscribe").
- Forbidden: vague mood words ("ominous", "haunting", "eerie", "atmospheric"). Use specific visual nouns instead.

EXAMPLE (Hansel & Gretel):
{
  "selected_scene_index": 3,
  "selected_scene_rationale": "The witch's first interaction with the children — she's already poisoning them with kindness.",
  "headline": "Witch Fattens Children For Her Oven",
  "narration": "The old woman cuts thick slices of honey-cake for the boy. She lifts a brass key and unlocks the iron cage by the hearth. He laughs as she pushes him inside, still chewing. The girl is set to scrubbing the long table where the knives are laid out, one for each finger."
}

Notice: opens with an image (the woman cutting cake), names the antagonist (witch implied by cage + knives), names the transgression (caging the child), shows the moment of harm — and stops before the rescue."""


def _system_prompt_multi(count: int) -> str:
    return f"""You are a YouTube Shorts hook writer for narrated dark-fairy-tale videos. Each output becomes a 15-25 second vertical clip that has to stop a thumb mid-scroll on a muted phone. Write {count} DIFFERENT hooks for A/B testing — each on a different scene.

Reply with strict JSON only. No prose, no markdown fences.

Schema:
{{
  "shorts": [
    {{
      "selected_scene_index": <int>,
      "selected_scene_rationale": "<one sentence on why this scene is the strongest visual hook>",
      "headline": "<4-6 word headline: a concrete noun + a transgressive verb. NO metaphors. NO abstractions.>",
      "narration": "<30-50 word hook. Opens with a CONCRETE present-tense image, NOT a question. Reveals the antagonist and the transgression. Ends mid-action.>"
    }}
    // exactly {count} entries
  ]
}}

ACROSS the {count} hooks:
- Each MUST pick a DIFFERENT "selected_scene_index". No repeats.
- Each MUST open on a DIFFERENT concrete image (not all on the antagonist; not all on the child; vary sight vs sound vs touch).
- Headlines must be visibly distinct from one another.

PER HOOK:

HEADLINE rules:
- 4-6 words. Concrete noun + transgressive verb. NO metaphors, NO abstractions.
- GOOD: "Mother Sews Skin Into Shirts" / "She Drowns Six Sons At Dawn"
- BAD: "She Sewed Shirts That Breathed" / "What Waited In The Woods"

NARRATION rules:
- 30-50 words. Aim for 40. ~12-18 seconds spoken.
- First sentence: concrete present-tense image with a menacing verb. NO rhetorical questions. NO "?" in the first sentence.
- Name the antagonist. Name the transgression. Show the moment of harm.
- Forbidden: the resolution (how the curse breaks, who rescues, the ending), meta-references, vague mood words.

EXAMPLE entry (Hansel & Gretel):
{{
  "selected_scene_index": 3,
  "selected_scene_rationale": "The witch's first interaction — kindness as poison.",
  "headline": "Witch Fattens Children For Her Oven",
  "narration": "The old woman cuts thick slices of honey-cake for the boy. She lifts a brass key and unlocks the iron cage by the hearth. He laughs as she pushes him inside, still chewing. The girl is set to scrubbing the long table where the knives are laid out, one for each finger."
}}"""


# Backwards-compat alias.
_SYSTEM_PROMPT = _SYSTEM_PROMPT_SINGLE


def _build_user_prompt(
    *,
    title: str,
    synopsis: str,
    tone: str,
    candidate_scenes: list[dict],
    count: int = 1,
) -> str:
    lines = [
        f"Title: {title}",
        f"Tone: {tone or 'unspecified'}",
        f"Synopsis: {synopsis or '(no synopsis)'}",
        "",
        "Candidate scenes (only pick from these — they are the first 60% of the story):",
    ]
    for s in candidate_scenes:
        narr = (s.get("narration") or "").strip().replace("\n", " ")
        if len(narr) > 320:
            narr = narr[:317] + "…"
        img_prompt = (s.get("image_prompt") or "").strip().replace("\n", " ")[:180]
        lines.append(
            f"  scene {s.get('index', '?')}: mood={s.get('mood', '?')}\n"
            f"    image: {img_prompt!r}\n"
            f"    narration: {narr!r}"
        )
    lines.append("")
    if count > 1:
        lines.append(
            f"Write {count} hooks. Each 30-50 words. Each opens on a concrete present-tense "
            "image (no questions). Each names the antagonist and the transgression. "
            "End mid-action, before any rescue."
        )
    else:
        lines.append(
            "Write a 30-50 word hook for the selected scene. Open on a concrete "
            "present-tense image — no rhetorical questions. Name the antagonist "
            "and the transgression. End mid-action, before any rescue."
        )
    return "\n".join(lines)


_FORBIDDEN_OPENERS = (
    "what ", "what's ", "what if",
    "have you", "imagine ", "did you", "do you ",
    "why does", "why do ", "why is", "why was",
    "when ", "where ",
)


def _validate_hook_payload(payload: dict, candidates: list[dict]) -> dict:
    """Clamp / sanity-check a single hook payload before we spend voice/render cents.

    Beyond v1's bounds-checking, v2 enforces:
      - First sentence must not be a question (no "?" in it).
      - First sentence must not start with a known rhetorical opener.
    If the LLM violates these, we don't try to repair the text — we raise so
    the caller can either fall back gracefully (multi-shorts dedup) or
    surface the failure (single-short).
    """
    valid_indices = {s.get("index") for s in candidates}

    idx = payload.get("selected_scene_index")
    try:
        idx = int(idx)
    except (TypeError, ValueError):
        idx = -1
    if idx not in valid_indices:
        idx = candidates[0].get("index") if candidates else 0
        log.warning(f"ShortsDirector: LLM picked invalid scene; defaulting to {idx}")

    headline = (payload.get("headline") or "").strip().rstrip(".")
    if not headline:
        raise ValueError("hook headline missing")
    if len(headline) > 70:
        headline = headline[:67] + "…"

    narration = (payload.get("narration") or "").strip()
    word_count = len(narration.split())
    if word_count < HOOK_TARGET_WORDS_MIN // 2:
        raise ValueError(f"hook narration too short ({word_count} words)")

    # Enforce: first sentence is not a question.
    first_sentence = ""
    for marker in (".", "!", "?"):
        if marker in narration:
            idx_m = narration.index(marker)
            first_sentence = narration[:idx_m + 1].strip()
            break
    if not first_sentence:
        first_sentence = narration
    if "?" in first_sentence:
        raise ValueError(
            f"hook narration opens with a question: {first_sentence[:80]!r}"
        )
    fl = first_sentence.lower()
    for opener in _FORBIDDEN_OPENERS:
        if fl.startswith(opener):
            raise ValueError(
                f"hook narration starts with banned opener {opener!r}: "
                f"{first_sentence[:80]!r}"
            )

    # Soft upper bound: trim to ~MAX words on sentence boundary if overlong.
    if word_count > HOOK_TARGET_WORDS_MAX * 1.5:
        sents = narration.replace("!", ".").replace("?", ".").split(".")
        kept: list[str] = []
        running = 0
        for s in sents:
            ws = len(s.split())
            if running + ws > HOOK_TARGET_WORDS_MAX:
                break
            kept.append(s.strip())
            running += ws
        narration = ". ".join(s for s in kept if s) + "."

    return {
        "selected_scene_index": idx,
        "headline": headline,
        "narration": narration,
        "rationale": (payload.get("selected_scene_rationale") or "").strip()[:300],
    }


def _validate_multi_payload(raw: dict, candidates: list[dict], requested: int) -> list[dict]:
    """Parse the multi-short LLM response. Dedupes scene_index, drops invalid entries."""
    items = raw.get("shorts")
    if not isinstance(items, list) or not items:
        # Recover if the model collapsed into the single-object schema.
        if "selected_scene_index" in raw or "narration" in raw:
            items = [raw]
        else:
            raise ValueError("LLM response missing 'shorts' array")

    hooks: list[dict] = []
    seen_indices: set[int] = set()
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            log.warning(f"ShortsDirector: dropping non-dict hook entry at position {i}")
            continue
        try:
            hook = _validate_hook_payload(item, candidates)
        except ValueError as e:
            log.warning(f"ShortsDirector: dropping invalid hook at position {i}: {e}")
            continue
        if hook["selected_scene_index"] in seen_indices:
            log.warning(
                f"ShortsDirector: dropping duplicate scene_index "
                f"{hook['selected_scene_index']} at position {i}"
            )
            continue
        seen_indices.add(hook["selected_scene_index"])
        hooks.append(hook)
        if len(hooks) >= requested:
            break

    if not hooks:
        raise ValueError("LLM produced no usable hooks after dedup/validation")
    return hooks


def _candidate_scenes(scenes: list[dict]) -> list[dict]:
    """First HOOK_SCENE_CUTOFF_PCT of scenes that have at least one image on disk."""
    if not scenes:
        return []
    cutoff = max(1, int(round(len(scenes) * HOOK_SCENE_CUTOFF_PCT)))
    candidates = scenes[:cutoff]
    return [s for s in candidates if (s.get("image_path") or s.get("image_paths"))]


def _scene_by_index(scenes: list[dict], index: int) -> Optional[dict]:
    for s in scenes:
        if s.get("index") == index:
            return s
    return None


def _resolve_image(project_dir: Path, scene: dict) -> Optional[Path]:
    candidates: list[str] = []
    if scene.get("image_path"):
        candidates.append(scene["image_path"])
    candidates.extend(scene.get("image_paths") or [])
    for rel in candidates:
        p = project_dir / rel
        if p.exists():
            return p
    return None


def _collect_bed_images(
    *,
    project_dir: Path,
    scenes: list[dict],
    candidates: list[dict],
    chosen_idx: int,
    n: int = BED_IMAGES_PER_SHORT,
) -> list[Path]:
    """Pick n bed images for the short, walking outward from the chosen scene.

    Strategy:
      1) If the chosen scene has multiple image_paths, use those first.
      2) Then add one image from the nearest candidate scene before, then
         after, then two-before, two-after, etc. — staying within candidates
         (the anti-spoiler clamp).
      3) Stop when we have n distinct images OR run out of candidates.

    Returns at least 1 image; falls back to a single chosen-scene image if
    no neighbors are available.
    """
    bed: list[Path] = []
    seen: set[Path] = set()

    chosen_scene = _scene_by_index(scenes, chosen_idx)
    if chosen_scene:
        # 1) Multiple images on the chosen scene
        rels = []
        if chosen_scene.get("image_paths"):
            rels.extend(chosen_scene["image_paths"])
        elif chosen_scene.get("image_path"):
            rels.append(chosen_scene["image_path"])
        for rel in rels:
            p = project_dir / rel
            if p.exists() and p not in seen:
                bed.append(p)
                seen.add(p)
                if len(bed) >= n:
                    return bed

    # 2) Walk outward through candidates
    cand_indices = [c.get("index") for c in candidates]
    if chosen_idx not in cand_indices:
        # Fall back: just use whatever images are available from candidates
        for c in candidates:
            img = _resolve_image(project_dir, c)
            if img and img not in seen:
                bed.append(img)
                seen.add(img)
                if len(bed) >= n:
                    return bed
        return bed or []

    chosen_pos = cand_indices.index(chosen_idx)
    # Interleave: -1, +1, -2, +2, ...
    max_offset = max(chosen_pos, len(cand_indices) - chosen_pos - 1)
    for offset in range(1, max_offset + 1):
        for direction in (-1, 1):
            pos = chosen_pos + direction * offset
            if 0 <= pos < len(cand_indices):
                cand = candidates[pos]
                img = _resolve_image(project_dir, cand)
                if img and img not in seen:
                    bed.append(img)
                    seen.add(img)
                    if len(bed) >= n:
                        return bed
    return bed


async def _voice_and_render_one(
    *,
    project_id: str,
    project_dir: Path,
    shorts_dir: Path,
    scenes: list[dict],
    candidates: list[dict],
    hook: dict,
    profile_id: str,
    voice_language: str,
    voice_instruct: str,
    audio_filename: str,
) -> dict:
    """Voice and render one hook. Returns the per-short result dict."""
    selected = _scene_by_index(scenes, hook["selected_scene_index"])
    if not selected:
        selected = candidates[0]
        hook["selected_scene_index"] = selected.get("index")
        log.warning(
            f"ShortsDirector {project_id}: selected scene missing; "
            f"falling back to {selected.get('index')}"
        )

    bed_images = _collect_bed_images(
        project_dir=project_dir,
        scenes=scenes,
        candidates=candidates,
        chosen_idx=selected.get("index"),
    )
    if not bed_images:
        raise RuntimeError(
            f"Project {project_id}: no bed images available for scene "
            f"{selected.get('index')}."
        )
    log.info(
        f"ShortsDirector {project_id}: bed has {len(bed_images)} image(s) for "
        f"scene #{selected.get('index')}"
    )

    audio_path = shorts_dir / audio_filename
    await voice_gen.generate_voice(
        text=hook["narration"],
        profile_id=profile_id,
        language=voice_language,
        output_path=audio_path,
        instruct=voice_instruct,
    )

    existing = sorted(shorts_dir.glob("short_*.mp4"))
    next_n = len(existing) + 1
    output_path = shorts_dir / f"short_{next_n:02d}.mp4"

    rendered_path, duration = shorts_assembly.assemble_short(
        image_paths=bed_images,
        audio_path=audio_path,
        headline=hook["headline"],
        narration=hook["narration"],
        output_path=output_path,
    )

    rel_path = str(rendered_path.relative_to(project_dir)).replace("\\", "/")
    return {
        "short_path": rel_path,
        "duration": duration,
        "headline": hook["headline"],
        "narration": hook["narration"],
        "selected_scene_index": hook["selected_scene_index"],
        "rationale": hook["rationale"],
        "bed_image_count": len(bed_images),
    }


async def generate_short(
    project_id: str,
    *,
    count: int = 1,
    ollama_model: Optional[str] = None,
    voice_profile_id: Optional[str] = None,
    voice_language: Optional[str] = None,
    voice_instruct: Optional[str] = None,
) -> dict:
    """Generate one or more Shorts for the given project.

    Return shape:
        count == 1: a single result dict (backwards-compatible)
        count >  1: {"shorts": [...], "count": N, "requested": R}
    """
    try:
        count = int(count)
    except (TypeError, ValueError):
        count = 1
    requested = count
    if count < 1:
        count = 1
    if count > MAX_SHORTS_PER_CALL:
        log.info(
            f"ShortsDirector {project_id}: count={requested} clamped to "
            f"{MAX_SHORTS_PER_CALL}"
        )
        count = MAX_SHORTS_PER_CALL

    state = store.load_state(project_id)
    script = store.load_json(project_id, "script.json")
    if not script:
        raise RuntimeError(f"Project {project_id} has no script.json — cannot make short.")

    scenes = script.get("scenes") or []
    if not scenes:
        raise RuntimeError(f"Project {project_id} script has no scenes.")

    candidates = _candidate_scenes(scenes)
    if not candidates:
        raise RuntimeError(
            f"Project {project_id}: no scenes in the first {int(HOOK_SCENE_CUTOFF_PCT * 100)}% "
            "have a generated image; cannot pick a hook scene."
        )

    if count > len(candidates):
        log.info(
            f"ShortsDirector {project_id}: only {len(candidates)} candidate scenes; "
            f"clamping count {count} -> {len(candidates)}"
        )
        count = len(candidates)

    profile_id = voice_profile_id or state.get("voice_profile_id") or ""
    if not profile_id:
        raise RuntimeError(
            f"Project {project_id} has no voice_profile_id set; cannot generate hook voice."
        )

    resolved_voice_language = voice_language or state.get("voice_language", "en")
    resolved_voice_instruct = voice_instruct or DEFAULT_VOICE_INSTRUCT

    user_prompt = _build_user_prompt(
        title=script.get("title") or state.get("title", ""),
        synopsis=script.get("synopsis", ""),
        tone=state.get("tone", ""),
        candidate_scenes=candidates,
        count=count,
    )

    system_prompt = (
        _system_prompt_multi(count) if count > 1 else _SYSTEM_PROMPT_SINGLE
    )

    try:
        raw = await call_llm_json(
            system=system_prompt,
            user=user_prompt,
            model=ollama_model or state.get("ollama_model"),
            temperature=0.65,
            timeout=180.0,
        )
    except LLMCallError as e:
        log.error(f"ShortsDirector LLM call failed for {project_id}: {e}")
        raise

    if count > 1:
        hooks = _validate_multi_payload(raw, candidates, requested=count)
    else:
        hooks = [_validate_hook_payload(raw, candidates)]

    log.info(
        f"ShortsDirector {project_id}: {len(hooks)} hook(s) ready — "
        f"scenes={[h['selected_scene_index'] for h in hooks]}"
    )

    project_dir = store.project_dir(project_id)
    shorts_dir = project_dir / "shorts"
    shorts_dir.mkdir(exist_ok=True)

    results: list[dict] = []
    for i, hook in enumerate(hooks, start=1):
        audio_filename = f"hook_{i:02d}.wav" if len(hooks) > 1 else "hook.wav"
        try:
            res = await _voice_and_render_one(
                project_id=project_id,
                project_dir=project_dir,
                shorts_dir=shorts_dir,
                scenes=scenes,
                candidates=candidates,
                hook=hook,
                profile_id=profile_id,
                voice_language=resolved_voice_language,
                voice_instruct=resolved_voice_instruct,
                audio_filename=audio_filename,
            )
        except Exception as e:
            log.exception(
                f"ShortsDirector {project_id}: hook {i}/{len(hooks)} "
                f"(scene {hook.get('selected_scene_index')}) failed: {e}"
            )
            continue

        rel_path = res["short_path"]
        prior = state.get("shorts_paths") or []
        prior.append(rel_path)
        store.update_state(project_id, shorts_paths=prior)
        state["shorts_paths"] = prior

        results.append(res)

    if not results:
        raise RuntimeError(
            f"Project {project_id}: all {len(hooks)} short(s) failed during voice/render."
        )

    if requested <= 1:
        return results[0]

    return {
        "shorts": results,
        "count": len(results),
        "requested": requested,
    }
