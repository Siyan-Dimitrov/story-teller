"""AnimateDiff video clip generation via ComfyUI API.

Takes an existing still image and generates a short animated clip using
SD1.5 + AnimateDiff v3 motion module through ComfyUI's API. The source image
is used as img2img input with low denoise to preserve the SDXL art style
while adding character motion.
"""

import asyncio
import logging
import time
import uuid
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image as PILImage

from . import config

log = logging.getLogger(__name__)

# ── Preset parameters ───────────────────────────────────────

ANIMATEDIFF_PRESETS = {
    "animatediff_subtle": {
        "denoise": 0.40,
        "num_frames": 16,
        "description": "Gentle motion — hair, cloth, subtle breathing",
    },
    "animatediff_moderate": {
        "denoise": 0.50,
        "num_frames": 16,
        "description": "Moderate motion — gestures, walking, flowing elements",
    },
    "animatediff_dramatic": {
        "denoise": 0.60,
        "num_frames": 24,
        "description": "Strong motion — action, magic effects, dramatic movement",
    },
}

VALID_ANIMATEDIFF_MOTIONS = set(ANIMATEDIFF_PRESETS.keys())

# ── Availability check ──────────────────────────────────────

_animatediff_available: bool | None = None  # None = not yet tested


async def check_animatediff_available() -> bool:
    """Check if AnimateDiff nodes and required models are available in ComfyUI."""
    global _animatediff_available

    if _animatediff_available is not None:
        return _animatediff_available

    # Check ComfyUI is running
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{config.COMFYUI_URL}/object_info/ADE_AnimateDiffLoaderGen1")
            if resp.status_code != 200:
                log.warning("[AnimateDiff] ADE_AnimateDiffLoaderGen1 node not found in ComfyUI")
                _animatediff_available = False
                return False
    except Exception as e:
        log.warning(f"[AnimateDiff] Cannot reach ComfyUI: {e}")
        _animatediff_available = False
        return False

    # Check SD1.5 checkpoint exists
    sd15_path = Path(config.ANIMATEDIFF_SD15_CHECKPOINT)
    if not sd15_path.is_absolute():
        # Check ComfyUI models directory
        comfyui_dir = Path(config.COMFYUI_URL.replace("http://", "").replace("https://", ""))
        # We can't check files on the ComfyUI host easily, so we'll trust the config
        pass

    log.info("[AnimateDiff] Nodes available in ComfyUI")
    _animatediff_available = True
    return True


def reset_availability():
    """Reset the availability cache (e.g., after installing nodes)."""
    global _animatediff_available
    _animatediff_available = None


# ── ComfyUI workflow builder ────────────────────────────────

def _build_animatediff_workflow(
    uploaded_image_name: str,
    prompt_text: str,
    negative_text: str,
    num_frames: int = 16,
    denoise: float = 0.45,
    seed: int = 0,
    fps: float = 8.0,
) -> dict:
    """Build a ComfyUI workflow for AnimateDiff img2img.

    Workflow graph:
      1  -> LoadImage (the existing scene image)
      2  -> ImageScale (downscale to AnimateDiff resolution)
      3  -> CheckpointLoaderSimple (SD1.5) -> MODEL, CLIP, VAE
      4  -> ADE_AnimateDiffLoaderGen1 (takes MODEL, applies motion module) -> MODEL
      5  -> VAEEncode (encode init image to latent)
      6  -> RepeatLatentBatch (repeat for num_frames)
      7  -> CLIPTextEncode (positive)
      8  -> CLIPTextEncode (negative)
      9  -> KSampler (img2img with AnimateDiff model)
      10 -> VAEDecode (decode video latents to frames)
      11 -> SaveImage (save individual frames)
    """
    ad_width = config.ANIMATEDIFF_WIDTH
    ad_height = config.ANIMATEDIFF_HEIGHT

    workflow = {
        # Load the source image
        "1": {
            "class_type": "LoadImage",
            "inputs": {"image": uploaded_image_name},
        },
        # Downscale to AnimateDiff resolution
        "2": {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["1", 0],
                "upscale_method": "lanczos",
                "width": ad_width,
                "height": ad_height,
                "crop": "center",
            },
        },
        # SD1.5 checkpoint
        "3": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": config.ANIMATEDIFF_SD15_CHECKPOINT},
        },
        # AnimateDiff: takes MODEL from checkpoint, applies motion module
        "4": {
            "class_type": "ADE_AnimateDiffLoaderGen1",
            "inputs": {
                "model": ["3", 0],
                "model_name": config.ANIMATEDIFF_MOTION_MODULE,
                "beta_schedule": "autoselect",
            },
        },
        # VAE Encode init image (single frame)
        "5": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["2", 0],
                "vae": ["3", 2],
            },
        },
        # Repeat latent for num_frames
        "6": {
            "class_type": "RepeatLatentBatch",
            "inputs": {
                "samples": ["5", 0],
                "amount": num_frames,
            },
        },
        # CLIP text encode - positive
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt_text,
                "clip": ["3", 1],
            },
        },
        # CLIP text encode - negative
        "8": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative_text,
                "clip": ["3", 1],
            },
        },
        # KSampler - img2img with AnimateDiff model
        "9": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": 20,
                "cfg": 7.0,
                "sampler_name": "euler_ancestral",
                "scheduler": "normal",
                "denoise": denoise,
                "model": ["4", 0],
                "positive": ["7", 0],
                "negative": ["8", 0],
                "latent_image": ["6", 0],
            },
        },
        # VAE Decode
        "10": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["9", 0],
                "vae": ["3", 2],
            },
        },
        # Save frames as images
        "11": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["10", 0],
                "filename_prefix": f"st_animdiff_{seed}",
            },
        },
    }

    return workflow


NEGATIVE_PROMPT = (
    "blurry, low quality, text, watermark, signature, jpeg artifacts, "
    "deformed, ugly, static, no motion, frozen, still image"
)


# ── Core generation function ────────────────────────────────

async def generate_animatediff_clip(
    image_path: Path,
    prompt: str,
    output_dir: Path,
    scene_index: int,
    img_index: int,
    motion_preset: str = "animatediff_subtle",
    style_prompt: str = "dark fairy tale, gothic storybook art, atmospheric, moody",
) -> Path | None:
    """Generate an AnimateDiff video clip from an existing still image.

    Returns path to a directory of output frames, or None on failure.
    """
    preset = ANIMATEDIFF_PRESETS.get(motion_preset, ANIMATEDIFF_PRESETS["animatediff_subtle"])
    denoise = preset["denoise"]
    num_frames = preset["num_frames"]

    seed = int(time.time() * 1000) % (2**32) + scene_index * 100 + img_index
    full_prompt = f"{style_prompt}, {prompt}, animated, motion" if style_prompt else prompt

    # Create output directory for this clip's frames
    clip_dir = output_dir / f"animatediff_s{scene_index:04d}_i{img_index}"
    clip_dir.mkdir(parents=True, exist_ok=True)

    filename = Path(image_path).name
    client_id = uuid.uuid4().hex[:8]

    log.info(
        f"[AnimateDiff] Generating clip for scene {scene_index} img {img_index}: "
        f"preset={motion_preset}, denoise={denoise}, frames={num_frames}"
    )

    try:
        async with httpx.AsyncClient(timeout=config.ANIMATEDIFF_TIMEOUT_SECONDS) as client:
            # Step 1: Upload image to ComfyUI
            with open(image_path, "rb") as f:
                upload_resp = await client.post(
                    f"{config.COMFYUI_URL}/upload/image",
                    files={"image": (filename, f, "image/png")},
                    data={"overwrite": "true"},
                )
            if upload_resp.status_code != 200:
                log.error(f"[AnimateDiff] Upload failed: HTTP {upload_resp.status_code}")
                return None

            uploaded = upload_resp.json()
            uploaded_name = uploaded.get("name", filename)

            # Step 2: Build and queue workflow
            workflow = _build_animatediff_workflow(
                uploaded_image_name=uploaded_name,
                prompt_text=full_prompt,
                negative_text=NEGATIVE_PROMPT,
                num_frames=num_frames,
                denoise=denoise,
                seed=seed,
            )

            resp = await client.post(
                f"{config.COMFYUI_URL}/prompt",
                json={"prompt": workflow, "client_id": client_id},
            )
            if resp.status_code != 200:
                body = resp.text[:1000]
                log.error(f"[AnimateDiff] Prompt rejected: HTTP {resp.status_code} — {body}")
                return None
            resp_data = resp.json()

            # Check for workflow errors (node_errors={} is normal/success)
            node_errors = resp_data.get("node_errors", {})
            if "error" in resp_data or node_errors:
                err_msg = resp_data.get("error", {}).get("message", "unknown")
                log.error(f"[AnimateDiff] Workflow error: {err_msg}, nodes: {node_errors}")
                return None

            prompt_id = resp_data["prompt_id"]
            log.info(f"[AnimateDiff] Queued prompt {prompt_id}, polling...")

            # Step 3: Poll for completion (AnimateDiff is slow — up to 5 minutes)
            for poll_i in range(int(config.ANIMATEDIFF_TIMEOUT_SECONDS)):
                hist_resp = await client.get(f"{config.COMFYUI_URL}/history/{prompt_id}")
                if hist_resp.status_code != 200:
                    await asyncio.sleep(2.0)
                    continue

                history = hist_resp.json()
                if prompt_id not in history:
                    await asyncio.sleep(2.0)
                    continue

                # Check for execution error
                status_info = history[prompt_id].get("status", {})
                if status_info.get("status_str") == "error":
                    msgs = status_info.get("messages", [])
                    log.error(f"[AnimateDiff] Execution error: {msgs}")
                    return None

                outputs = history[prompt_id].get("outputs", {})
                for node_id, node_out in outputs.items():
                    if "images" in node_out:
                        # Download all frames
                        images_info = node_out["images"]
                        log.info(f"[AnimateDiff] Got {len(images_info)} frames, downloading...")

                        for frame_i, img_info in enumerate(images_info):
                            params = {
                                "filename": img_info["filename"],
                                "subfolder": img_info.get("subfolder", ""),
                                "type": img_info.get("type", "output"),
                            }
                            img_resp = await client.get(
                                f"{config.COMFYUI_URL}/view", params=params
                            )
                            if img_resp.status_code != 200:
                                log.warning(f"[AnimateDiff] Failed to download frame {frame_i}")
                                continue

                            # Save frame and upscale to target resolution
                            frame_pil = PILImage.open(BytesIO(img_resp.content)).convert("RGB")
                            if frame_pil.size != (config.VIDEO_WIDTH, config.VIDEO_HEIGHT):
                                frame_pil = frame_pil.resize(
                                    (config.VIDEO_WIDTH, config.VIDEO_HEIGHT),
                                    PILImage.LANCZOS,
                                )
                            frame_path = clip_dir / f"frame_{frame_i:04d}.png"
                            frame_pil.save(str(frame_path), "PNG")

                        log.info(
                            f"[AnimateDiff] Saved {len(images_info)} frames to {clip_dir}"
                        )
                        return clip_dir

                log.warning(f"[AnimateDiff] No image output in history for {prompt_id}")
                return None

            log.error(f"[AnimateDiff] Timed out waiting for {prompt_id}")
            return None

    except httpx.ConnectError:
        log.error(f"[AnimateDiff] Cannot connect to ComfyUI at {config.COMFYUI_URL}")
        return None
    except Exception as e:
        log.error(f"[AnimateDiff] Error: {type(e).__name__}: {e}")
        return None


async def generate_all_animatediff_clips(
    scenes: list[dict],
    project_dir: Path,
    style_prompt: str = "dark fairy tale, gothic storybook art, atmospheric, moody",
    progress_cb=None,
) -> list[dict]:
    """Generate AnimateDiff clips for all images classified as 'animatediff'.

    Updates scenes in-place with animatediff_clip_paths.
    """
    animatediff_dir = project_dir / "animatediff_clips"
    animatediff_dir.mkdir(exist_ok=True)

    # Count total animatediff images for progress
    total_ad = 0
    for scene in scenes:
        anim_types = scene.get("animation_types") or []
        for t in anim_types:
            if t == "animatediff":
                total_ad += 1

    if total_ad == 0:
        log.info("[AnimateDiff] No images classified as animatediff, skipping")
        return scenes

    log.info(f"[AnimateDiff] Generating {total_ad} clips...")
    done_ad = 0

    for scene in scenes:
        idx = scene.get("index", 0)
        image_paths = scene.get("image_paths") or []
        anim_types = scene.get("animation_types") or []
        motion_presets = scene.get("motion_presets") or []
        scene.setdefault("animatediff_clip_paths", [None] * len(image_paths))

        for img_idx, rel_path in enumerate(image_paths):
            anim_type = anim_types[img_idx] if img_idx < len(anim_types) else "depthflow"
            if anim_type != "animatediff":
                continue

            preset = motion_presets[img_idx] if img_idx < len(motion_presets) else "animatediff_subtle"
            abs_path = project_dir / rel_path

            if not abs_path.exists():
                log.warning(f"[AnimateDiff] Image not found: {abs_path}")
                continue

            if progress_cb:
                progress_cb(
                    phase=f"AnimateDiff clip {done_ad + 1}/{total_ad}",
                    progress=done_ad / max(total_ad, 1),
                )

            clip_dir = await generate_animatediff_clip(
                image_path=abs_path,
                prompt=scene.get("image_prompts", [scene.get("image_prompt", "")])[img_idx]
                    if img_idx < len(scene.get("image_prompts", [])) else scene.get("image_prompt", ""),
                output_dir=animatediff_dir,
                scene_index=idx,
                img_index=img_idx,
                motion_preset=preset,
                style_prompt=style_prompt,
            )

            if clip_dir:
                rel_clip = str(clip_dir.relative_to(project_dir))
                while len(scene["animatediff_clip_paths"]) <= img_idx:
                    scene["animatediff_clip_paths"].append(None)
                scene["animatediff_clip_paths"][img_idx] = rel_clip
                log.info(f"[AnimateDiff] Scene {idx} img {img_idx}: clip saved to {rel_clip}")
            else:
                log.warning(
                    f"[AnimateDiff] Scene {idx} img {img_idx}: generation failed, "
                    f"will fall back to depth parallax"
                )
                # Revert to depthflow so assembly uses parallax instead
                if img_idx < len(scene["animation_types"]):
                    scene["animation_types"][img_idx] = "depthflow"

            done_ad += 1

    return scenes
