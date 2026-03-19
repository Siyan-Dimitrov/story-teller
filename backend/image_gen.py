"""Image generation via ComfyUI API or Ollama."""

import io
import json
import time
import uuid
import logging
from pathlib import Path
import httpx

from . import config

log = logging.getLogger(__name__)

# ComfyUI workflow template for SDXL text-to-image
# This is a minimal SDXL workflow — can be extended with LoRAs
COMFYUI_WORKFLOW = {
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 0,
            "steps": 25,
            "cfg": 7.0,
            "sampler_name": "euler_ancestral",
            "scheduler": "normal",
            "denoise": 1.0,
            "model": ["4", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["5", 0],
        },
    },
    "4": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {
            "ckpt_name": "sd_xl_base_1.0.safetensors",
        },
    },
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {
            "width": config.IMAGE_WIDTH,
            "height": config.IMAGE_HEIGHT,
            "batch_size": 1,
        },
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "",  # Filled with image_prompt
            "clip": ["4", 1],
        },
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "blurry, low quality, text, watermark, signature, jpeg artifacts, deformed, ugly, modern, photograph, realistic photo",
            "clip": ["4", 1],
        },
    },
    "8": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["3", 0],
            "vae": ["4", 2],
        },
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {
            "filename_prefix": "storyteller",
            "images": ["8", 0],
        },
    },
}


async def generate_image_comfyui(
    prompt: str,
    style_prompt: str,
    output_path: Path,
    seed: int | None = None,
) -> Path:
    """Generate an image using ComfyUI's SDXL workflow."""
    import copy

    workflow = copy.deepcopy(COMFYUI_WORKFLOW)

    # Set the prompt
    full_prompt = f"{style_prompt}, {prompt}" if style_prompt else prompt
    workflow["6"]["inputs"]["text"] = full_prompt

    # Set seed
    if seed is None:
        seed = int(time.time() * 1000) % (2**32)
    workflow["3"]["inputs"]["seed"] = seed

    # Unique client ID for tracking
    client_id = uuid.uuid4().hex[:8]

    async with httpx.AsyncClient(timeout=config.IMAGE_TIMEOUT_SECONDS) as client:
        # Queue the prompt
        resp = await client.post(
            f"{config.COMFYUI_URL}/prompt",
            json={"prompt": workflow, "client_id": client_id},
        )
        resp.raise_for_status()
        prompt_data = resp.json()
        prompt_id = prompt_data["prompt_id"]

        log.info(f"ComfyUI prompt queued: {prompt_id}")

        # Poll for completion
        while True:
            hist_resp = await client.get(f"{config.COMFYUI_URL}/history/{prompt_id}")
            hist_resp.raise_for_status()
            history = hist_resp.json()

            if prompt_id in history:
                outputs = history[prompt_id].get("outputs", {})
                # Find the SaveImage node output
                for node_id, node_out in outputs.items():
                    if "images" in node_out:
                        img_info = node_out["images"][0]
                        filename = img_info["filename"]
                        subfolder = img_info.get("subfolder", "")
                        img_type = img_info.get("type", "output")

                        # Download the image
                        params = {"filename": filename, "subfolder": subfolder, "type": img_type}
                        img_resp = await client.get(
                            f"{config.COMFYUI_URL}/view",
                            params=params,
                        )
                        img_resp.raise_for_status()

                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        output_path.write_bytes(img_resp.content)
                        log.info(f"Image saved to {output_path}")
                        return output_path

                raise RuntimeError(f"No image output found in ComfyUI response")

            await _async_sleep(1.0)


async def generate_image_ollama(
    prompt: str,
    style_prompt: str,
    output_path: Path,
) -> Path:
    """Generate an image using an Ollama vision/generation model (placeholder).

    Note: Ollama doesn't natively support image generation yet.
    This would need to route through a model that supports it,
    or use a different cloud API. For now, generates a placeholder.
    """
    from PIL import Image, ImageDraw, ImageFont

    log.warning("Ollama image generation not yet supported — creating text placeholder")

    img = Image.new("RGB", (config.IMAGE_WIDTH, config.IMAGE_HEIGHT), color=(20, 15, 30))
    draw = ImageDraw.Draw(img)

    # Draw the prompt as text
    try:
        font = ImageFont.truetype("arial", 28)
    except OSError:
        font = ImageFont.load_default()

    # Word wrap
    words = prompt.split()
    lines = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        if len(test) > 60:
            lines.append(current)
            current = word
        else:
            current = test
    if current:
        lines.append(current)

    y = config.IMAGE_HEIGHT // 2 - len(lines) * 20
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        x = (config.IMAGE_WIDTH - w) // 2
        draw.text((x, y), line, fill=(180, 160, 200), font=font)
        y += 40

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path), "PNG")
    log.info(f"Placeholder image saved to {output_path}")
    return output_path


async def generate_all_scenes(
    scenes: list[dict],
    project_dir: Path,
    backend: str = "comfyui",
    style_prompt: str = "dark fairy tale illustration, gothic storybook art, atmospheric, detailed, moody lighting",
) -> list[dict]:
    """Generate images for all scenes. Returns updated scenes with image_path."""
    images_dir = project_dir / "images"
    images_dir.mkdir(exist_ok=True)

    gen_fn = generate_image_comfyui if backend == "comfyui" else generate_image_ollama

    for scene in scenes:
        idx = scene["index"]
        output_path = images_dir / f"scene_{idx:04d}.png"
        try:
            if backend == "comfyui":
                await gen_fn(
                    prompt=scene["image_prompt"],
                    style_prompt=style_prompt,
                    output_path=output_path,
                    seed=idx * 42,
                )
            else:
                await gen_fn(
                    prompt=scene["image_prompt"],
                    style_prompt=style_prompt,
                    output_path=output_path,
                )
            scene["image_path"] = str(output_path.relative_to(project_dir))
        except Exception as e:
            log.error(f"Image generation failed for scene {idx}: {e}")
            # Fallback to text placeholder
            try:
                await generate_image_ollama(
                    prompt=scene["image_prompt"],
                    style_prompt="",
                    output_path=output_path,
                )
                scene["image_path"] = str(output_path.relative_to(project_dir))
            except Exception as e2:
                log.error(f"Placeholder also failed for scene {idx}: {e2}")
                scene["image_path"] = None
                scene["image_error"] = str(e)

    return scenes


async def _async_sleep(seconds: float):
    import asyncio
    await asyncio.sleep(seconds)
