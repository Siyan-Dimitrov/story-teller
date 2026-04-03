"""Image quality control via Ollama vision model.

After image generation, each image is sent to a vision-capable LLM
which scores it on prompt adherence, artistic quality, technical quality,
and style consistency. Users review results and select which images to
regenerate before proceeding to animation.
"""

import base64
import json
import logging
import threading
import time
from pathlib import Path

import httpx

from . import config
from . import image_gen

log = logging.getLogger(__name__)

# ── QC progress tracking ────────────────────────────────────

_qc_lock = threading.Lock()
_qc_tasks: dict[str, dict] = {}


def get_qc_progress(project_id: str) -> dict:
    with _qc_lock:
        state = _qc_tasks.get(project_id)
        if not state:
            return {"active": False, "progress": 0, "phase": "idle", "error": None}
        return {
            "active": state.get("active", False),
            "progress": state["progress"],
            "phase": state["phase"],
            "error": state.get("error"),
        }


def _init_qc_progress(project_id: str):
    with _qc_lock:
        _qc_tasks[project_id] = {
            "active": True,
            "progress": 0,
            "phase": "starting",
            "error": None,
        }


def _update_qc_progress(project_id: str, **kwargs):
    with _qc_lock:
        if project_id in _qc_tasks:
            _qc_tasks[project_id].update(kwargs)


def _finish_qc_progress(project_id: str, error: str | None = None):
    with _qc_lock:
        if project_id in _qc_tasks:
            _qc_tasks[project_id]["active"] = False
            _qc_tasks[project_id]["phase"] = "error" if error else "done"
            _qc_tasks[project_id]["progress"] = 0 if error else 1.0
            if error:
                _qc_tasks[project_id]["error"] = error


# ── Vision evaluation ───────────────────────────────────────

QC_SYSTEM_PROMPT = """You are an image quality reviewer for a dark fairy tale video project.
You evaluate AI-generated illustrations against their original prompt.

Score each image on these criteria (1-5 scale):
1. prompt_adherence: Does the image match what was described?
2. artistic_quality: Good composition, color palette, detail level?
3. technical_quality: Free of artifacts, blur, deformities, unwanted text/watermarks?
4. style_consistency: Does it feel like a dark fairy tale illustration — gothic, atmospheric, moody?

Respond ONLY with valid JSON (no markdown fences):
{"prompt_adherence": 4, "artistic_quality": 3, "technical_quality": 5, "style_consistency": 4, "reasoning": "Brief 1-2 sentence explanation"}"""


async def evaluate_image(
    image_path: Path,
    prompt: str,
    style_prompt: str,
    vision_model: str,
    ollama_url: str | None = None,
) -> dict:
    """Send an image to Ollama vision model for quality evaluation.

    Returns dict with scores, average_score, and reasoning.
    """
    base_url = ollama_url or config.OLLAMA_URL

    # Read and base64 encode the image
    image_bytes = image_path.read_bytes()
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    user_prompt = (
        f"Evaluate this AI-generated image.\n"
        f"It was generated from this prompt: \"{prompt}\"\n"
        f"Target style: {style_prompt}\n\n"
        f"Score it on all 4 criteria (1-5) and provide brief reasoning."
    )

    try:
        async with httpx.AsyncClient(timeout=config.QC_TIMEOUT_SECONDS) as client:
            resp = await client.post(
                f"{base_url}/api/chat",
                json={
                    "model": vision_model,
                    "messages": [
                        {"role": "system", "content": QC_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": user_prompt,
                            "images": [b64_image],
                        },
                    ],
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 4000},
                },
            )

            if resp.status_code != 200:
                error_text = resp.text[:500]
                log.error(f"[QC] Vision model error: HTTP {resp.status_code}: {error_text}")
                return _error_result(f"Vision model HTTP {resp.status_code}: {error_text}")

            data = resp.json()
            content = _extract_content(data)

            if not content:
                log.warning("[QC] Empty response from vision model")
                return _error_result("Empty response from vision model")

            # Strip markdown fences
            if content.startswith("```"):
                lines = content.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                content = "\n".join(lines)

            # Extract JSON
            start = content.find("{")
            end = content.rfind("}") + 1
            if start < 0 or end <= start:
                log.warning(f"[QC] No JSON in response: {content[:200]}")
                return _error_result("No JSON in vision model response")

            scores = json.loads(content[start:end])

            # Validate and compute average
            criteria = ["prompt_adherence", "artistic_quality", "technical_quality", "style_consistency"]
            for c in criteria:
                val = scores.get(c)
                if not isinstance(val, (int, float)) or val < 1 or val > 5:
                    scores[c] = 3  # default if missing/invalid

            avg = sum(scores[c] for c in criteria) / len(criteria)

            return {
                "scores": {c: scores[c] for c in criteria},
                "average_score": round(avg, 2),
                "reasoning": scores.get("reasoning", ""),
                "error": None,
            }

    except json.JSONDecodeError as e:
        log.warning(f"[QC] JSON parse error: {e}")
        return _error_result(f"JSON parse error: {e}")
    except httpx.ConnectError:
        log.error(f"[QC] Cannot connect to Ollama at {base_url}")
        return _error_result("Cannot connect to Ollama")
    except (httpx.TimeoutException, httpx.ReadTimeout) as e:
        log.error(f"[QC] Timeout calling Ollama vision model: {type(e).__name__}")
        return _error_result(f"Vision model timed out ({type(e).__name__})")
    except Exception as e:
        msg = str(e) or f"{type(e).__name__} (no details)"
        log.error(f"[QC] Error: {type(e).__name__}: {msg}")
        return _error_result(msg)


def _extract_content(data: dict) -> str:
    """Extract text from Ollama response, handling thinking models."""
    msg = data.get("message", {})
    content = (msg.get("content") or "").strip()
    if content:
        return content
    thinking = (msg.get("thinking") or "").strip()
    if thinking:
        start = thinking.find("{")
        end = thinking.rfind("}") + 1
        if start >= 0 and end > start:
            return thinking[start:end]
    return content


def _error_result(reason: str) -> dict:
    """Return an error QC result when evaluation fails."""
    return {
        "scores": {},
        "average_score": 0,
        "reasoning": "",
        "error": reason or "Unknown error",
    }


# ── Core QC pipeline (evaluate only, no regeneration) ─────

async def run_qc_for_project(
    scenes: list[dict],
    project_dir: Path,
    vision_model: str,
    style_prompt: str,
    pass_threshold: float,
    project_id: str | None = None,
    targets: list[dict] | None = None,
) -> list[dict]:
    """Evaluate images across scenes (no regeneration).

    If targets is provided, only evaluate those specific images.
    Otherwise evaluate all images.
    Scores each image and marks pass/fail based on threshold.
    Users review results and choose which to regenerate separately.
    """
    if project_id:
        _init_qc_progress(project_id)

    # Build target set for selective QC
    target_set: set[tuple[int, int]] | None = None
    if targets:
        target_set = {(t["scene_index"], t["image_index"]) for t in targets}

    # Count total images to evaluate
    total_images = 0
    for si, scene in enumerate(scenes):
        paths = scene.get("image_paths") or []
        for ii in range(len(paths)):
            if target_set is None or (si, ii) in target_set:
                total_images += 1

    if total_images == 0:
        log.info("[QC] No images to evaluate")
        if project_id:
            _finish_qc_progress(project_id)
        return scenes

    log.info(f"[QC] Evaluating {total_images} images with model={vision_model}, threshold={pass_threshold}")

    done = 0
    total_passed = 0
    total_failed = 0
    total_errors = 0

    try:
        for si, scene in enumerate(scenes):
            idx = scene.get("index", si)
            image_paths = scene.get("image_paths") or []
            prompts = scene.get("image_prompts") or []
            # Preserve existing QC results for images we're not re-evaluating
            existing_qc = {r["image_index"]: r for r in (scene.get("qc_results") or []) if isinstance(r, dict)}
            qc_results = []

            for img_idx, rel_path in enumerate(image_paths):
                # Skip images not in targets (if selective)
                if target_set is not None and (si, img_idx) not in target_set:
                    # Keep existing result if any
                    if img_idx in existing_qc:
                        qc_results.append(existing_qc[img_idx])
                    continue

                abs_path = project_dir / rel_path
                prompt = prompts[img_idx] if img_idx < len(prompts) else scene.get("image_prompt", "")

                if project_id:
                    _update_qc_progress(
                        project_id,
                        phase=f"Evaluating scene {idx + 1} image {img_idx + 1}",
                        progress=done / total_images,
                    )

                if not abs_path.exists():
                    log.warning(f"[QC] Image not found: {abs_path}")
                    qc_results.append({
                        "image_index": img_idx,
                        "passed": True,
                        "scores": {},
                        "average_score": 0,
                        "reasoning": "Image file not found — skipped",
                        "error": "File not found",
                        "attempts": 0,
                    })
                    done += 1
                    continue

                result = await evaluate_image(
                    image_path=abs_path,
                    prompt=prompt,
                    style_prompt=style_prompt,
                    vision_model=vision_model,
                )

                has_error = result.get("error") is not None
                passed = not has_error and result["average_score"] >= pass_threshold

                if has_error:
                    total_errors += 1
                    log.warning(f"[QC] Scene {idx} img {img_idx}: ERROR — {result['error']}")
                elif passed:
                    total_passed += 1
                    log.info(f"[QC] Scene {idx} img {img_idx}: PASS (score={result['average_score']})")
                else:
                    total_failed += 1
                    log.info(f"[QC] Scene {idx} img {img_idx}: FAIL (score={result['average_score']})")

                old_attempts = existing_qc.get(img_idx, {}).get("attempts", 0)
                qc_results.append({
                    "image_index": img_idx,
                    "passed": passed,
                    "scores": result.get("scores", {}),
                    "average_score": result.get("average_score", 0),
                    "reasoning": result.get("reasoning", ""),
                    "error": result.get("error"),
                    "attempts": old_attempts + 1,
                })

                done += 1

            scene["qc_results"] = qc_results
            scene["qc_passed"] = all(r.get("passed", False) for r in qc_results if r)

        log.info(f"[QC] Complete: {total_passed} passed, {total_failed} failed, {total_errors} errors out of {total_images}")

        if project_id:
            _finish_qc_progress(project_id)

        return scenes

    except Exception as e:
        if project_id:
            _finish_qc_progress(project_id, error=str(e))
        raise


# ── Regenerate selected images ────────────────────────────

async def regenerate_and_evaluate(
    scenes: list[dict],
    project_dir: Path,
    targets: list[dict],
    vision_model: str,
    style_prompt: str,
    image_backend: str,
    lora_keys: list[str] | None,
    pass_threshold: float,
    project_id: str | None = None,
) -> list[dict]:
    """Regenerate specific images and re-evaluate them.

    targets: list of {"scene_index": int, "image_index": int}
    """
    if project_id:
        _init_qc_progress(project_id)

    total = len(targets)
    if total == 0:
        if project_id:
            _finish_qc_progress(project_id)
        return scenes

    log.info(f"[QC] Regenerating {total} selected images")

    try:
        for i, target in enumerate(targets):
            si = target["scene_index"]
            ii = target["image_index"]

            if si >= len(scenes):
                continue

            scene = scenes[si]
            image_paths = scene.get("image_paths") or []
            prompts = scene.get("image_prompts") or []

            if ii >= len(image_paths):
                continue

            abs_path = project_dir / image_paths[ii]
            prompt = prompts[ii] if ii < len(prompts) else scene.get("image_prompt", "")

            if project_id:
                _update_qc_progress(
                    project_id,
                    phase=f"Regenerating {i + 1}/{total}: scene {si + 1} img {ii + 1}",
                    progress=i / total,
                )

            # Regenerate with a new seed
            new_seed = int(time.time() * 1000) % (2**32) + si * 1000 + ii * 42 + i * 7919
            try:
                if image_backend == "comfyui":
                    await image_gen.generate_image_comfyui(
                        prompt=prompt,
                        style_prompt=style_prompt,
                        output_path=abs_path,
                        seed=new_seed,
                        lora_keys=lora_keys,
                    )
                elif image_backend == "replicate":
                    await image_gen.generate_image_replicate(
                        prompt=prompt,
                        style_prompt=style_prompt,
                        output_path=abs_path,
                        seed=new_seed,
                        lora_keys=lora_keys,
                    )
                else:
                    await image_gen.generate_image_ollama(
                        prompt=prompt,
                        style_prompt=style_prompt,
                        output_path=abs_path,
                    )
                log.info(f"[QC] Regenerated scene {si} img {ii}")
            except Exception as e:
                log.error(f"[QC] Regeneration failed for scene {si} img {ii}: {e}")
                continue

            # Re-evaluate
            if project_id:
                _update_qc_progress(
                    project_id,
                    phase=f"Re-evaluating {i + 1}/{total}: scene {si + 1} img {ii + 1}",
                    progress=(i + 0.5) / total,
                )

            result = await evaluate_image(
                image_path=abs_path,
                prompt=prompt,
                style_prompt=style_prompt,
                vision_model=vision_model,
            )

            has_error = result.get("error") is not None
            passed = not has_error and result["average_score"] >= pass_threshold

            # Update QC results
            qc_results = scene.get("qc_results") or []
            while len(qc_results) <= ii:
                qc_results.append({})

            old_attempts = qc_results[ii].get("attempts", 0) if ii < len(qc_results) and isinstance(qc_results[ii], dict) else 0

            qc_results[ii] = {
                "image_index": ii,
                "passed": passed,
                "scores": result.get("scores", {}),
                "average_score": result.get("average_score", 0),
                "reasoning": result.get("reasoning", ""),
                "error": result.get("error"),
                "attempts": old_attempts + 1,
            }
            scene["qc_results"] = qc_results
            scene["qc_passed"] = all(
                r.get("passed", False) for r in qc_results if isinstance(r, dict) and r
            )

        if project_id:
            _finish_qc_progress(project_id)

        return scenes

    except Exception as e:
        if project_id:
            _finish_qc_progress(project_id, error=str(e))
        raise


async def evaluate_single_image(
    scene: dict,
    image_index: int,
    project_dir: Path,
    vision_model: str,
    style_prompt: str,
) -> dict:
    """Evaluate a single image (for manual retry from frontend)."""
    image_paths = scene.get("image_paths") or []
    prompts = scene.get("image_prompts") or []

    if image_index >= len(image_paths):
        return {"error": "Image index out of range"}

    abs_path = project_dir / image_paths[image_index]
    prompt = prompts[image_index] if image_index < len(prompts) else scene.get("image_prompt", "")

    result = await evaluate_image(
        image_path=abs_path,
        prompt=prompt,
        style_prompt=style_prompt,
        vision_model=vision_model,
    )

    return result
