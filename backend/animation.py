"""Image animation — LLM classification and depth-based parallax preparation."""

import json
import logging
import threading
import numpy as np
from pathlib import Path
from PIL import Image as PILImage, ImageFilter
import httpx

from . import config

log = logging.getLogger(__name__)

try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
    log.info(
        "OpenCV not available — using numpy fallback for parallax "
        "(install opencv-python-headless for better quality)"
    )

# ── Animation progress tracking ──────────────────────────────

_anim_lock = threading.Lock()
_anim_tasks: dict[str, dict] = {}


def get_animation_progress(project_id: str) -> dict:
    with _anim_lock:
        state = _anim_tasks.get(project_id)
        if not state:
            return {"active": False, "progress": 0, "phase": "idle", "error": None}
        return {
            "active": state.get("active", False),
            "progress": state["progress"],
            "phase": state["phase"],
            "error": state.get("error"),
        }


def _init_anim_progress(project_id: str):
    with _anim_lock:
        _anim_tasks[project_id] = {
            "active": True,
            "progress": 0,
            "phase": "classifying",
            "error": None,
        }


def _update_anim_progress(project_id: str, **kw):
    with _anim_lock:
        state = _anim_tasks.get(project_id)
        if state:
            state.update(kw)


def _finish_anim_progress(project_id: str, error: str | None = None):
    with _anim_lock:
        state = _anim_tasks.get(project_id)
        if state:
            state["active"] = False
            if error:
                state["phase"] = "error"
                state["error"] = error
            else:
                state["phase"] = "done"
                state["progress"] = 1.0


# ── LLM Classification ──────────────────────────────────────

CLASSIFY_PROMPT = """You are classifying images for animation in a dark fairy tale video.

For each image prompt below, choose:
1. Animation type: "depthflow", "portrait", or "animatediff"
2. A motion preset name

Use "animatediff" for images with a character performing an action, characters with flowing hair/clothing, magic/particle effects, fire, water, wind, or any scene where actual pixel motion would look impressive. This uses AI video generation for realistic character animation.

Use "portrait" ONLY for images that are primarily a close-up of a single character's face/head (head-and-shoulders framing where the face is the dominant element).

Use "depthflow" for landscapes, wide establishing shots with no main character, static architecture, or scenes where subtle camera movement is more appropriate than character motion.

Motion presets for animatediff:
- animatediff_subtle: gentle motion — hair swaying, cloth rippling, subtle breathing (good for calm, intimate character moments)
- animatediff_moderate: moderate motion — walking, gesturing, flowing elements (good for active character scenes)
- animatediff_dramatic: strong motion — action, magic effects, dramatic movement (good for climactic or magical scenes)

Motion presets for depthflow:
- dolly_forward: camera slowly pushes toward subject (good for tense, ominous, dramatic reveal)
- dolly_backward: camera slowly pulls away (good for endings, reveals of scale)
- pan_left: camera slides left with depth parallax (good for establishing shots, journeys)
- pan_right: camera slides right with depth parallax (good for transitions, discoveries)
- orbital_left: slight arc around subject (good for dramatic, powerful moments)
- orbital_right: slight arc around subject (good for mystery, suspense)
- gentle_rise: camera slowly rises upward (good for awe, peaceful, melancholy)
- gentle_float: dreamy floating motion (good for magical, surreal, peaceful moments)

Motion presets for portrait:
- portrait_breathe: very subtle zoom pulse like breathing (good for calm, intimate)
- portrait_reveal: slow dolly forward to reveal face details (good for dramatic, intense)
- portrait_drift: gentle lateral drift with depth separation (good for contemplative, mysterious)

Choose motions that enhance the scene's mood. Vary your choices — don't repeat the same motion for every image. Prefer animatediff for character-focused scenes where motion adds life.

Respond ONLY with valid JSON (no markdown fences):
{
  "classifications": [
    {"index": 0, "type": "depthflow", "motion": "dolly_forward"},
    {"index": 1, "type": "animatediff", "motion": "animatediff_moderate"},
    {"index": 2, "type": "portrait", "motion": "portrait_breathe"}
  ]
}"""

VALID_DEPTHFLOW_MOTIONS = {
    "dolly_forward",
    "dolly_backward",
    "pan_left",
    "pan_right",
    "orbital_left",
    "orbital_right",
    "gentle_rise",
    "gentle_float",
}
VALID_PORTRAIT_MOTIONS = {"portrait_breathe", "portrait_reveal", "portrait_drift"}
VALID_ANIMATEDIFF_MOTIONS = {"animatediff_subtle", "animatediff_moderate", "animatediff_dramatic"}
ALL_VALID_MOTIONS = VALID_DEPTHFLOW_MOTIONS | VALID_PORTRAIT_MOTIONS | VALID_ANIMATEDIFF_MOTIONS


async def _classify_batch(
    entries: list[str],
    start_index: int,
    model: str,
) -> dict[int, dict]:
    """Classify a batch of image entries via LLM. Returns {global_index: classification}."""
    user_prompt = (
        f"Classify these {len(entries)} image prompts for animation:\n\n"
        + "\n\n".join(entries)
    )

    try:
        async with httpx.AsyncClient(timeout=config.LLM_TIMEOUT_SECONDS) as client:
            resp = await client.post(
                f"{config.OLLAMA_URL}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": CLASSIFY_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 8000},
                },
            )
            resp.raise_for_status()

        data = resp.json()
        # Handle thinking models (e.g. kimi-k2.5) where content is in 'thinking' field
        msg = data.get("message", {})
        content = (msg.get("content") or "").strip()
        if not content:
            thinking = (msg.get("thinking") or "").strip()
            if thinking:
                # Extract JSON from thinking text
                json_start = thinking.find("{")
                json_end = thinking.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    content = thinking[json_start:json_end]
                    log.info(f"[Animation] Batch {start_index}: extracted JSON from thinking field")
        log.info(
            f"[Animation] Batch {start_index}-{start_index + len(entries) - 1}: "
            f"LLM response ({len(content)} chars): {content[:300]}"
        )

        if not content:
            log.warning(f"[Animation] Batch {start_index}: empty LLM response")
            return {}

        # Strip markdown fences if present
        if "```" in content:
            lines = content.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines).strip()

        # Extract JSON object if there's surrounding text
        if not content.startswith("{"):
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                content = content[start:end]

        result = json.loads(content)
        batch_cls = {}
        for c in result.get("classifications", []):
            # Map batch-local index to global index
            local_idx = c["index"]
            global_idx = start_index + local_idx
            batch_cls[global_idx] = c
            batch_cls[global_idx]["index"] = global_idx
        return batch_cls

    except httpx.ConnectError:
        log.error(f"[Animation] Cannot connect to Ollama at {config.OLLAMA_URL}")
        return {}
    except httpx.TimeoutException:
        log.error(f"[Animation] Batch {start_index}: timed out after {config.LLM_TIMEOUT_SECONDS}s")
        return {}
    except json.JSONDecodeError as e:
        log.error(f"[Animation] Batch {start_index}: JSON parse error: {e}")
        log.error(f"[Animation] Raw content: {content[:500]}")
        return {}
    except Exception as e:
        log.error(f"[Animation] Batch {start_index}: {type(e).__name__}: {e}")
        return {}


_CLASSIFY_BATCH_SIZE = 8  # images per LLM call — keeps prompt small for cloud models


async def classify_scene_animations(
    scenes: list[dict],
    ollama_model: str | None = None,
) -> list[dict]:
    """Use LLM to classify each image for animation type and motion preset."""
    model = ollama_model or config.OLLAMA_MODEL

    # Build list of (mood, prompt) tuples and scene mapping
    image_data: list[tuple[str, str]] = []  # (mood, prompt)
    image_map: list[tuple[int, int]] = []  # (scene_list_index, img_idx)

    for si, scene in enumerate(scenes):
        prompts = scene.get("image_prompts") or []
        if not prompts:
            single = scene.get("image_prompt", "")
            prompts = [single] if single else []
        mood = scene.get("mood", "neutral")

        for img_idx, prompt in enumerate(prompts):
            image_data.append((mood, prompt))
            image_map.append((si, img_idx))

    if not image_data:
        return scenes

    log.info(f"[Animation] Classifying {len(image_data)} images via {model} "
             f"in batches of {_CLASSIFY_BATCH_SIZE}...")

    # Classify in batches to avoid empty responses from cloud models
    classifications: dict[int, dict] = {}
    for batch_start in range(0, len(image_data), _CLASSIFY_BATCH_SIZE):
        batch = image_data[batch_start:batch_start + _CLASSIFY_BATCH_SIZE]
        # Build entries with batch-local indices (0..N) so LLM output matches
        batch_entries = [
            f"Image {i} (mood: {mood}): {prompt}"
            for i, (mood, prompt) in enumerate(batch)
        ]

        batch_result = await _classify_batch(batch_entries, batch_start, model)
        # Retry once if empty (cloud models sometimes return empty)
        if not batch_result:
            log.info(f"[Animation] Batch {batch_start}: retrying...")
            batch_result = await _classify_batch(batch_entries, batch_start, model)
        classifications.update(batch_result)
        log.info(f"[Animation] Batch {batch_start}: got {len(batch_result)} classifications")

    log.info(f"[Animation] Total: {len(classifications)}/{len(image_data)} classified by LLM")

    # Log classification distribution
    if classifications:
        type_counts: dict[str, int] = {}
        motion_counts: dict[str, int] = {}
        for c in classifications.values():
            t = c.get("type", "?")
            m = c.get("motion", "?")
            type_counts[t] = type_counts.get(t, 0) + 1
            motion_counts[m] = motion_counts.get(m, 0) + 1
        log.info(f"[Animation] Types: {type_counts}, Motions: {motion_counts}")

    # Rotating fallback presets when LLM gives no result for an image
    _FALLBACK_PRESETS = [
        "dolly_forward", "pan_left", "gentle_rise", "orbital_right",
        "pan_right", "gentle_float", "dolly_backward", "orbital_left",
    ]

    # Initialize animation fields on all scenes
    for scene in scenes:
        num_imgs = len(scene.get("image_paths") or [])
        scene.setdefault("animation_types", ["depthflow"] * num_imgs)
        scene.setdefault("motion_presets", ["dolly_forward"] * num_imgs)

    # Map LLM results back to scenes
    for global_idx, (si, img_idx) in enumerate(image_map):
        scene = scenes[si]
        cls = classifications.get(global_idx, {})
        fallback_motion = _FALLBACK_PRESETS[global_idx % len(_FALLBACK_PRESETS)]
        anim_type = cls.get("type", "depthflow")
        motion = cls.get("motion", fallback_motion)

        # Validate type
        if anim_type not in ("depthflow", "portrait", "animatediff"):
            anim_type = "depthflow"

        # Validate motion against type
        if anim_type == "animatediff" and motion not in VALID_ANIMATEDIFF_MOTIONS:
            motion = "animatediff_subtle"
        elif anim_type == "portrait" and motion not in VALID_PORTRAIT_MOTIONS:
            motion = "portrait_breathe"
        elif anim_type == "depthflow" and motion not in VALID_DEPTHFLOW_MOTIONS:
            motion = fallback_motion

        # Ensure lists are right length
        while len(scene["animation_types"]) <= img_idx:
            scene["animation_types"].append("depthflow")
            scene["motion_presets"].append("dolly_forward")

        scene["animation_types"][img_idx] = anim_type
        scene["motion_presets"][img_idx] = motion

    return scenes


# ── Depth Estimation ─────────────────────────────────────────


def estimate_depth_gradient(
    image_path: str, target_w: int, target_h: int
) -> np.ndarray:
    """Estimate depth using image analysis heuristics (no ML dependencies).

    Returns depth map normalized to [0, 1] where 0=far, 1=near.
    Uses vertical gradient, edge strength, local contrast, and center bias.
    """
    pil_img = PILImage.open(image_path).convert("RGB")

    # Work at reduced resolution for speed
    work_w = min(target_w, 640)
    work_h = int(work_w * target_h / target_w)
    img = pil_img.resize((work_w, work_h), PILImage.BILINEAR)
    arr = np.array(img).astype(np.float32)

    # 1. Vertical gradient — bottom is near, top is far (natural perspective)
    y_grad = np.linspace(0, 1, work_h)[:, np.newaxis]
    y_grad = np.broadcast_to(y_grad, (work_h, work_w)).copy()

    # 2. Edge strength — sharp edges suggest foreground
    gray = np.mean(arr, axis=2)
    dx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    dy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    edges = np.sqrt(dx**2 + dy**2)
    edges = edges / (edges.max() + 1e-8)
    # Smooth edges
    blur_r = max(3, work_w // 40)
    edge_pil = PILImage.fromarray((edges * 255).astype(np.uint8))
    edge_pil = edge_pil.filter(ImageFilter.GaussianBlur(radius=blur_r))
    edges = np.array(edge_pil).astype(np.float32) / 255.0

    # 3. Local contrast — higher contrast = closer
    gray_pil = PILImage.fromarray(gray.astype(np.uint8))
    blurred = np.array(
        gray_pil.filter(ImageFilter.GaussianBlur(radius=max(5, work_w // 20)))
    ).astype(np.float32)
    contrast = np.abs(gray - blurred)
    contrast = contrast / (contrast.max() + 1e-8)
    contrast_pil = PILImage.fromarray((contrast * 255).astype(np.uint8))
    contrast_pil = contrast_pil.filter(
        ImageFilter.GaussianBlur(radius=max(3, work_w // 40))
    )
    contrast = np.array(contrast_pil).astype(np.float32) / 255.0

    # 4. Center bias — subjects tend to be centered
    cx, cy = work_w / 2.0, work_h / 2.0
    xs = np.arange(work_w, dtype=np.float32)[np.newaxis, :]
    ys = np.arange(work_h, dtype=np.float32)[:, np.newaxis]
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    max_dist = np.sqrt(cx**2 + cy**2)
    center_bias = 1.0 - (dist / max_dist)
    center_bias = np.power(center_bias, 0.5)

    # Combine channels
    depth = (
        0.35 * y_grad
        + 0.25 * edges
        + 0.20 * contrast
        + 0.20 * center_bias
    )

    # Normalize
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    # Heavy smooth for natural parallax transitions
    depth_pil = PILImage.fromarray((depth * 255).astype(np.uint8))
    depth_pil = depth_pil.filter(
        ImageFilter.GaussianBlur(radius=max(5, work_w // 25))
    )

    # Resize to target
    depth_pil = depth_pil.resize((target_w, target_h), PILImage.BILINEAR)
    depth = np.array(depth_pil).astype(np.float32) / 255.0

    return depth


async def estimate_depth_comfyui(
    image_path: str, target_w: int, target_h: int
) -> np.ndarray | None:
    """Estimate depth using ComfyUI's MiDaS preprocessor.

    Returns depth map or None if ComfyUI/MiDaS unavailable.
    """
    import uuid as _uuid
    from io import BytesIO

    try:
        filename = Path(image_path).name
        client_id = _uuid.uuid4().hex[:8]
        log.info(f"[ComfyUI depth] Starting for {filename} (target {target_w}x{target_h})")

        async with httpx.AsyncClient(timeout=config.IMAGE_TIMEOUT_SECONDS) as client:
            # Upload image to ComfyUI input directory
            log.debug(f"[ComfyUI depth] Uploading {filename} to ComfyUI...")
            with open(image_path, "rb") as f:
                upload_resp = await client.post(
                    f"{config.COMFYUI_URL}/upload/image",
                    files={"image": (filename, f, "image/png")},
                    data={"overwrite": "true"},
                )
            if upload_resp.status_code != 200:
                log.warning(
                    f"[ComfyUI depth] Upload failed: HTTP {upload_resp.status_code} — {upload_resp.text[:300]}"
                )
                return None

            uploaded = upload_resp.json()
            uploaded_name = uploaded.get("name", filename)
            log.debug(f"[ComfyUI depth] Uploaded as: {uploaded_name}")

            # Build MiDaS depth workflow
            workflow = {
                "1": {
                    "class_type": "LoadImage",
                    "inputs": {"image": uploaded_name},
                },
                "2": {
                    "class_type": "MiDaS-DepthMapPreprocessor",
                    "inputs": {
                        "image": ["1", 0],
                        "a": 6.283185307179586,
                        "bg_threshold": 0.1,
                        "resolution": min(target_w, 1024),
                    },
                },
                "3": {
                    "class_type": "SaveImage",
                    "inputs": {
                        "images": ["2", 0],
                        "filename_prefix": "st_depth",
                    },
                },
            }

            # Queue prompt
            log.debug(f"[ComfyUI depth] Submitting workflow to {config.COMFYUI_URL}/prompt")
            resp = await client.post(
                f"{config.COMFYUI_URL}/prompt",
                json={"prompt": workflow, "client_id": client_id},
            )
            if resp.status_code != 200:
                log.warning(
                    f"[ComfyUI depth] Prompt rejected: HTTP {resp.status_code} — {resp.text[:500]}"
                )
                return None

            resp_data = resp.json()

            # ComfyUI returns 200 with error body for invalid/missing nodes
            if "error" in resp_data or "node_errors" in resp_data:
                err_type = resp_data.get("error", {}).get("type", "unknown")
                err_msg = resp_data.get("error", {}).get("message", "unknown error")
                node_errors = resp_data.get("node_errors", {})
                log.warning(
                    f"[ComfyUI depth] Workflow error — type: {err_type}, "
                    f"message: {err_msg}, node_errors: {node_errors}"
                )
                if "does not exist" in err_msg:
                    log.warning(
                        "[ComfyUI depth] MiDaS node not installed. "
                        "Install via ComfyUI-Manager: search 'controlnet aux' or "
                        "'comfyui_controlnet_aux'. Falling back to gradient depth."
                    )
                return None

            prompt_id = resp_data["prompt_id"]
            log.info(f"[ComfyUI depth] Queued prompt {prompt_id}, polling for result...")

            # Poll for completion
            import asyncio

            for poll_i in range(120):
                hist_resp = await client.get(
                    f"{config.COMFYUI_URL}/history/{prompt_id}"
                )
                if hist_resp.status_code != 200:
                    log.warning(
                        f"[ComfyUI depth] History poll failed: HTTP {hist_resp.status_code}"
                    )
                    await asyncio.sleep(1.0)
                    continue

                history = hist_resp.json()

                if prompt_id in history:
                    status_info = history[prompt_id].get("status", {})
                    if status_info.get("status_str") == "error":
                        msgs = status_info.get("messages", [])
                        log.warning(f"[ComfyUI depth] Execution error: {msgs}")
                        return None

                    outputs = history[prompt_id].get("outputs", {})
                    for _node_id, node_out in outputs.items():
                        if "images" in node_out:
                            img_info = node_out["images"][0]
                            log.debug(
                                f"[ComfyUI depth] Got result: {img_info['filename']}"
                            )
                            params = {
                                "filename": img_info["filename"],
                                "subfolder": img_info.get("subfolder", ""),
                                "type": img_info.get("type", "output"),
                            }
                            img_resp = await client.get(
                                f"{config.COMFYUI_URL}/view", params=params
                            )
                            if img_resp.status_code != 200:
                                log.warning(
                                    f"[ComfyUI depth] Failed to download result: "
                                    f"HTTP {img_resp.status_code}"
                                )
                                return None

                            depth_pil = PILImage.open(
                                BytesIO(img_resp.content)
                            ).convert("L")
                            depth_pil = depth_pil.resize(
                                (target_w, target_h), PILImage.BILINEAR
                            )
                            log.info(f"[ComfyUI depth] Success for {filename}")
                            return np.array(depth_pil).astype(np.float32) / 255.0

                    log.warning(
                        f"[ComfyUI depth] Prompt finished but no image output. "
                        f"Outputs: {list(outputs.keys())}"
                    )
                    return None

                await asyncio.sleep(1.0)

            log.warning(f"[ComfyUI depth] Timed out after 120s for {filename}")
            return None

    except httpx.ConnectError:
        log.warning(f"[ComfyUI depth] Cannot connect to {config.COMFYUI_URL} — is ComfyUI running?")
        return None
    except httpx.TimeoutException:
        log.warning(f"[ComfyUI depth] Request timed out ({config.IMAGE_TIMEOUT_SECONDS}s)")
        return None
    except Exception as e:
        log.warning(f"[ComfyUI depth] Unexpected error: {type(e).__name__}: {e}")
        return None


_comfyui_depth_available: bool | None = None  # None = not yet tested


async def estimate_depth(
    image_path: str,
    target_w: int = config.IMAGE_WIDTH,
    target_h: int = config.IMAGE_HEIGHT,
) -> tuple[np.ndarray, str]:
    """Estimate depth map, trying ComfyUI first, then gradient fallback.

    Returns (depth_array, method_used) where method is 'comfyui' or 'gradient'.
    Only attempts ComfyUI once — if it fails, all subsequent calls skip straight to gradient.
    """
    global _comfyui_depth_available

    if config.DEPTH_METHOD in ("comfyui", "auto") and _comfyui_depth_available is not False:
        depth = await estimate_depth_comfyui(image_path, target_w, target_h)
        if depth is not None:
            _comfyui_depth_available = True
            return depth, "comfyui"
        # First failure — log once and skip ComfyUI for all remaining images
        if _comfyui_depth_available is None:
            _comfyui_depth_available = False
            log.warning(
                "[ComfyUI depth] MiDaS unavailable — using gradient for all images. "
                "Install comfyui_controlnet_aux in ComfyUI-Manager to enable MiDaS depth."
            )

    depth = estimate_depth_gradient(image_path, target_w, target_h)
    if _comfyui_depth_available is None:
        _comfyui_depth_available = False
    return depth, "gradient"


# ── Main API ─────────────────────────────────────────────────


async def prepare_animations(
    scenes: list[dict],
    project_dir: Path,
    ollama_model: str | None = None,
    project_id: str | None = None,
) -> list[dict]:
    """Classify images and generate depth maps + AnimateDiff clips.

    Updates scenes in-place with animation_types, motion_presets,
    depth maps, and AnimateDiff clip paths.
    """
    from .animatediff_gen import (
        check_animatediff_available,
        generate_all_animatediff_clips,
    )

    if project_id:
        _init_anim_progress(project_id)

    try:
        # Step 1: Check if AnimateDiff is available
        ad_available = False
        if config.ANIMATEDIFF_ENABLED:
            ad_available = await check_animatediff_available()
            log.info(f"[Animation] AnimateDiff available: {ad_available}")

        # Step 2: LLM classification
        if project_id:
            _update_anim_progress(project_id, phase="classifying images", progress=0.05)

        scenes = await classify_scene_animations(scenes, ollama_model)
        log.info("Animation classification complete")

        # If AnimateDiff is not available, downgrade animatediff → depthflow
        if not ad_available:
            for scene in scenes:
                anim_types = scene.get("animation_types") or []
                motion_presets = scene.get("motion_presets") or []
                for i, t in enumerate(anim_types):
                    if t == "animatediff":
                        anim_types[i] = "depthflow"
                        if i < len(motion_presets):
                            motion_presets[i] = "dolly_forward"
                scene["animation_types"] = anim_types
                scene["motion_presets"] = motion_presets
            log.info("[Animation] AnimateDiff unavailable — all animatediff images downgraded to depthflow")

        # Step 3: Generate depth maps (for depthflow and portrait images)
        depth_dir = project_dir / "depth_maps"
        depth_dir.mkdir(exist_ok=True)

        # Count images needing depth maps (skip animatediff)
        total_depth = 0
        for s in scenes:
            anim_types = s.get("animation_types") or []
            for i, _ in enumerate(s.get("image_paths") or []):
                t = anim_types[i] if i < len(anim_types) else "depthflow"
                if t != "animatediff":
                    total_depth += 1

        done_depth = 0

        for scene in scenes:
            idx = scene.get("index", 0)
            image_paths = scene.get("image_paths") or []
            anim_types = scene.get("animation_types") or []
            scene["depth_map_paths"] = []

            for img_idx, rel_path in enumerate(image_paths):
                anim_type = anim_types[img_idx] if img_idx < len(anim_types) else "depthflow"

                # Skip depth map for animatediff images
                if anim_type == "animatediff":
                    scene["depth_map_paths"].append(None)
                    continue

                abs_path = project_dir / rel_path
                if not abs_path.exists():
                    log.warning(f"Image not found for depth: {abs_path}")
                    scene["depth_map_paths"].append(None)
                    done_depth += 1
                    continue

                # Progress: depth maps take 0.1-0.6 of total progress
                if project_id:
                    _update_anim_progress(
                        project_id,
                        phase=f"depth map {done_depth + 1}/{total_depth}",
                        progress=0.1 + 0.5 * (done_depth / max(total_depth, 1)),
                    )

                depth, method = await estimate_depth(
                    str(abs_path), config.VIDEO_WIDTH, config.VIDEO_HEIGHT
                )

                depth_filename = f"scene_{idx:04d}_img_{img_idx}_depth.png"
                depth_path = depth_dir / depth_filename
                depth_pil = PILImage.fromarray((depth * 255).astype(np.uint8))
                depth_pil.save(str(depth_path), "PNG")

                rel_depth = str(depth_path.relative_to(project_dir))
                scene["depth_map_paths"].append(rel_depth)
                done_depth += 1

                log.info(f"Depth map saved: {depth_filename} (method: {method})")

        # Step 4: Generate AnimateDiff clips (if any images classified as animatediff)
        if ad_available:
            ad_count = sum(
                1 for s in scenes
                for t in (s.get("animation_types") or [])
                if t == "animatediff"
            )
            if ad_count > 0:
                log.info(f"[Animation] Generating {ad_count} AnimateDiff clips...")

                def _ad_progress(phase: str, progress: float):
                    if project_id:
                        # AnimateDiff takes 0.6-0.95 of total progress
                        _update_anim_progress(
                            project_id,
                            phase=phase,
                            progress=0.6 + 0.35 * progress,
                        )

                scenes = await generate_all_animatediff_clips(
                    scenes=scenes,
                    project_dir=project_dir,
                    progress_cb=_ad_progress,
                )

        if project_id:
            _finish_anim_progress(project_id)

        return scenes

    except Exception as e:
        if project_id:
            _finish_anim_progress(project_id, error=str(e))
        raise
