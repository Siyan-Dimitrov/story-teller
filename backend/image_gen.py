"""Image generation via ComfyUI API or Ollama."""

import time
import uuid
import logging
from pathlib import Path
import httpx

from . import config

log = logging.getLogger(__name__)

# ── Available LoRAs ─────────────────────────────────────────────
# Each entry: filename, trigger words to prepend, strength_model, strength_clip
AVAILABLE_LORAS = {
    "tim_burton": {
        "file": "Tim_Burton_Painting_Style_SDXL.safetensors",
        "trigger": "Tim Burton Style",
        "strength_model": 0.7,
        "strength_clip": 0.7,
    },
    "storybook": {
        "file": "StorybookRedmondV2.safetensors",
        "trigger": "KidsRedmAF",
        "strength_model": 0.6,
        "strength_clip": 0.6,
    },
    "dark_gothic": {
        "file": "dark_gothic_fantasy_xl.safetensors",
        "trigger": "dark gothic fantasy",
        "strength_model": 0.65,
        "strength_clip": 0.65,
    },
    "mark_ryden": {
        "file": "Mark_Ryden_Style.safetensors",
        "trigger": "Mark Ryden Style",
        "strength_model": 0.7,
        "strength_clip": 0.7,
    },
    "dave_mckean": {
        "file": "Dave_McKean_Style.safetensors",
        "trigger": "Dave McKean Style",
        "strength_model": 0.65,
        "strength_clip": 0.65,
    },
}

# Default LoRA combination — Tim Burton + dark gothic gives the closest
# match to the abitfrank channel aesthetic
DEFAULT_LORAS = ["tim_burton", "dark_gothic"]


def _build_workflow(
    prompt_text: str,
    negative_text: str,
    lora_keys: list[str] | None = None,
    seed: int = 0,
) -> dict:
    """Build a ComfyUI workflow dict with optional LoRA chain.

    Nodes:
      4  -> CheckpointLoaderSimple
      10, 11, ... -> LoraLoader chain (one per LoRA)
      6  -> CLIPTextEncode (positive)
      7  -> CLIPTextEncode (negative)
      5  -> EmptyLatentImage
      3  -> KSampler
      8  -> VAEDecode
      9  -> SaveImage
    """
    if lora_keys is None:
        lora_keys = DEFAULT_LORAS

    workflow: dict = {}

    # Checkpoint loader — always node "4"
    workflow["4"] = {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"},
    }

    # Build LoRA chain: each LoraLoader takes model+clip from previous
    # First LoRA connects to checkpoint ("4"), subsequent connect to previous LoRA node
    lora_nodes = []
    valid_loras = [k for k in lora_keys if k in AVAILABLE_LORAS]

    prev_model_ref = ["4", 0]  # model output
    prev_clip_ref = ["4", 1]   # clip output

    for i, key in enumerate(valid_loras):
        lora = AVAILABLE_LORAS[key]
        node_id = str(10 + i)
        workflow[node_id] = {
            "class_type": "LoraLoader",
            "inputs": {
                "lora_name": lora["file"],
                "strength_model": lora["strength_model"],
                "strength_clip": lora["strength_clip"],
                "model": prev_model_ref,
                "clip": prev_clip_ref,
            },
        }
        prev_model_ref = [node_id, 0]
        prev_clip_ref = [node_id, 1]
        lora_nodes.append(key)

    # Collect trigger words from active LoRAs
    triggers = ", ".join(AVAILABLE_LORAS[k]["trigger"] for k in valid_loras)
    full_prompt = f"{triggers}, {prompt_text}" if triggers else prompt_text

    # CLIP text encoders — connect to last LoRA (or checkpoint if no LoRAs)
    workflow["6"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": full_prompt, "clip": prev_clip_ref},
    }
    workflow["7"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": negative_text, "clip": prev_clip_ref},
    }

    # Empty latent
    workflow["5"] = {
        "class_type": "EmptyLatentImage",
        "inputs": {
            "width": config.IMAGE_WIDTH,
            "height": config.IMAGE_HEIGHT,
            "batch_size": 1,
        },
    }

    # KSampler — model from last LoRA (or checkpoint)
    workflow["3"] = {
        "class_type": "KSampler",
        "inputs": {
            "seed": seed,
            "steps": 25,
            "cfg": 7.0,
            "sampler_name": "euler_ancestral",
            "scheduler": "normal",
            "denoise": 1.0,
            "model": prev_model_ref,
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["5", 0],
        },
    }

    # VAE Decode — vae always from checkpoint node "4" output slot 2
    workflow["8"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
    }

    # Save image
    workflow["9"] = {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "storyteller", "images": ["8", 0]},
    }

    return workflow


NEGATIVE_PROMPT = (
    "blurry, low quality, text, watermark, signature, jpeg artifacts, "
    "deformed, ugly, modern, photograph, realistic photo, 3d render"
)


async def generate_image_comfyui(
    prompt: str,
    style_prompt: str,
    output_path: Path,
    seed: int | None = None,
    lora_keys: list[str] | None = None,
) -> Path:
    """Generate an image using ComfyUI's SDXL workflow with LoRA support."""
    if seed is None:
        seed = int(time.time() * 1000) % (2**32)

    # Combine style prompt with scene prompt
    full_prompt = f"{style_prompt}, {prompt}" if style_prompt else prompt

    workflow = _build_workflow(
        prompt_text=full_prompt,
        negative_text=NEGATIVE_PROMPT,
        lora_keys=lora_keys,
        seed=seed,
    )

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
    lora_keys: list[str] | None = None,
) -> list[dict]:
    """Generate images for all scenes. Returns updated scenes with image_path."""
    images_dir = project_dir / "images"
    images_dir.mkdir(exist_ok=True)

    for scene in scenes:
        idx = scene["index"]
        output_path = images_dir / f"scene_{idx:04d}.png"
        try:
            if backend == "comfyui":
                await generate_image_comfyui(
                    prompt=scene["image_prompt"],
                    style_prompt=style_prompt,
                    output_path=output_path,
                    seed=idx * 42,
                    lora_keys=lora_keys,
                )
            else:
                await generate_image_ollama(
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
