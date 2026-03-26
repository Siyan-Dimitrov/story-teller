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
# For ComfyUI: Uses local .safetensors files
# For Replicate: Uses flux_lora_key to look up URL in config.FLUX_LORA_URLS
AVAILABLE_LORAS = {
    "tim_burton": {
        "file": "Tim_Burton_Painting_Style_SDXL.safetensors",  # ComfyUI local (SDXL)
        "trigger": "vicgoth",  # FLUX trigger for Keltezaa/victorian-gothic-horror
        "strength_model": 1.0,
        "strength_clip": 0.7,
        "flux_lora_key": "tim_burton",  # Key in config.FLUX_LORA_URLS
    },
    "storybook": {
        "file": "StorybookRedmondV2.safetensors",  # ComfyUI local (SDXL)
        "trigger": "d00dlet00n",  # FLUX trigger for renderartist/doodletoonflux
        "strength_model": 0.8,
        "strength_clip": 0.6,
        "flux_lora_key": "storybook",
    },
    "dark_gothic": {
        "file": "dark_gothic_fantasy_xl.safetensors",  # ComfyUI local (SDXL)
        "trigger": "",  # FLUX: nerijs/dark-fantasy-illustration-flux (no trigger needed)
        "strength_model": 1.2,
        "strength_clip": 0.65,
        "flux_lora_key": "dark_gothic",
    },
    "dark_fantasy": {
        "file": "dark_fantasy_flux.safetensors",  # ComfyUI local
        "trigger": "",  # FLUX: Shakker-Labs Dark Fantasy (no trigger needed)
        "strength_model": 0.7,
        "strength_clip": 0.75,
        "flux_lora_key": "dark_fantasy",
    },
    "mark_ryden": {
        "file": "Mark_Ryden_Style.safetensors",  # ComfyUI local (SDXL)
        "trigger": "evangsurreal",  # FLUX trigger for brushpenbob/Flux-surrealism
        "strength_model": 0.8,
        "strength_clip": 0.7,
        "flux_lora_key": "mark_ryden",
    },
    "dave_mckean": {
        "file": "Dave_McKean_Style.safetensors",  # ComfyUI local (SDXL)
        "trigger": "w3irdth1ngs",  # FLUX trigger for renderartist/weirdthingsflux
        "strength_model": 0.8,
        "strength_clip": 0.65,
        "flux_lora_key": "weird_surreal",  # Uses alternatives dict
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

    # Collect trigger words from active LoRAs (skip empty triggers)
    triggers = ", ".join(t for k in valid_loras if (t := AVAILABLE_LORAS[k]["trigger"]))
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
            "steps": 40,
            "cfg": 7.5,
            "sampler_name": "dpmpp_2m",
            "scheduler": "karras",
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


def _flux_aspect_ratio() -> str:
    """Map IMAGE_WIDTH x IMAGE_HEIGHT to the closest Flux-supported aspect ratio."""
    ratio = config.IMAGE_WIDTH / config.IMAGE_HEIGHT
    supported = [
        ("1:1", 1.0), ("16:9", 16 / 9), ("21:9", 21 / 9), ("3:2", 3 / 2),
        ("2:3", 2 / 3), ("4:5", 4 / 5), ("5:4", 5 / 4), ("3:4", 3 / 4),
        ("4:3", 4 / 3), ("9:16", 9 / 16), ("9:21", 9 / 21),
    ]
    return min(supported, key=lambda x: abs(x[1] - ratio))[0]


def _build_flux_lora_input(
    prompt: str,
    style_prompt: str,
    seed: int,
    lora_keys: list[str] | None = None,
) -> dict:
    """Build Replicate FLUX LoRA input parameters.

    Supports up to 2 LoRAs via lora_weights and extra_lora parameters.
    """
    # Collect trigger words from active LoRAs
    triggers = []
    lora_urls = []
    lora_scales = []

    if lora_keys:
        for key in lora_keys:
            if key in AVAILABLE_LORAS:
                lora_info = AVAILABLE_LORAS[key]
                if lora_info["trigger"]:  # Only add non-empty triggers
                    triggers.append(lora_info["trigger"])
                flux_key = lora_info.get("flux_lora_key")
                if flux_key:
                    # Check primary URLs first, then alternatives
                    lora_url = config.FLUX_LORA_URLS.get(flux_key) or config.FLUX_LORA_ALTERNATIVES.get(flux_key)
                    if lora_url:
                        lora_urls.append(lora_url)
                        lora_scales.append(lora_info.get("strength_model", 0.7))

    # Combine trigger words with style and scene prompt
    trigger_str = ", ".join(triggers) if triggers else ""
    if trigger_str and style_prompt:
        full_prompt = f"{trigger_str}, {style_prompt}, {prompt}"
    elif trigger_str:
        full_prompt = f"{trigger_str}, {prompt}"
    elif style_prompt:
        full_prompt = f"{style_prompt}, {prompt}"
    else:
        full_prompt = prompt

    inp = {
        "prompt": full_prompt,
        "aspect_ratio": _flux_aspect_ratio(),
        "megapixels": "1",
        "seed": seed,
        "num_outputs": 1,
        "output_format": "png",
    }

    # Add LoRA weights if available
    if len(lora_urls) >= 1:
        inp["lora_weights"] = lora_urls[0]
        inp["lora_scale"] = lora_scales[0]
        log.info(f"Primary LoRA: {lora_urls[0]} (scale={lora_scales[0]})")

    if len(lora_urls) >= 2:
        inp["extra_lora"] = lora_urls[1]
        inp["extra_lora_scale"] = lora_scales[1]
        log.info(f"Secondary LoRA: {lora_urls[1]} (scale={lora_scales[1]})")

    # Add API tokens if using CivitAI URLs
    if any("civitai.com" in url for url in lora_urls):
        if config.CIVITAI_API_TOKEN:
            inp["civitai_api_token"] = config.CIVITAI_API_TOKEN
            log.debug("Added CivitAI API token")

    return inp


async def generate_image_replicate(
    prompt: str,
    style_prompt: str,
    output_path: Path,
    seed: int | None = None,
    lora_keys: list[str] | None = None,
) -> Path:
    """Generate an image using Replicate's FLUX model with LoRA support.

    Supports loading LoRA weights from HuggingFace or CivitAI URLs.
    Includes retry with exponential backoff for rate-limit (429) and transient errors.
    """
    import asyncio
    import replicate as _replicate

    if seed is None:
        seed = int(time.time() * 1000) % (2**32)

    model = config.REPLICATE_MODEL
    inp = _build_flux_lora_input(prompt, style_prompt, seed, lora_keys)

    # Add inference steps based on model variant
    if "schnell" in model:
        inp["num_inference_steps"] = 4
        inp["go_fast"] = True  # Use fp8 quantization for speed
    else:
        inp["num_inference_steps"] = 28
        inp["go_fast"] = False  # Use bf16 for better LoRA fidelity

    max_retries = config.REPLICATE_MAX_RETRIES
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            log.info(f"Replicate [{model}]: generating (seed={seed}, attempt {attempt + 1}/{max_retries + 1})")

            loop = asyncio.get_event_loop()
            output = await loop.run_in_executor(None, lambda: _replicate.run(model, input=inp))

            # output is a list of FileOutput objects (URL-like)
            image_url = str(output[0]) if isinstance(output, list) else str(output)

            async with httpx.AsyncClient(timeout=config.REPLICATE_TIMEOUT_SECONDS) as client:
                resp = await client.get(image_url)
                resp.raise_for_status()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(resp.content)

            log.info(f"Replicate image saved to {output_path}")
            return output_path

        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            is_retryable = any(kw in err_str for kw in [
                "throttl", "rate", "429", "too many", "overloaded",
                "503", "502", "timeout", "timed out",
            ])

            if not is_retryable or attempt >= max_retries:
                log.error(f"Replicate generation failed (not retryable or out of retries): {e}")
                raise

            # Exponential backoff: 3s, 6s, 12s ...
            wait = 3.0 * (2 ** attempt)
            log.warning(f"Replicate rate-limited/transient error, retrying in {wait:.0f}s: {e}")
            await asyncio.sleep(wait)

    raise last_error  # shouldn't reach here, but just in case


async def generate_all_scenes(
    scenes: list[dict],
    project_dir: Path,
    backend: str = "comfyui",
    style_prompt: str = "dark fairy tale illustration, gothic storybook art, atmospheric, detailed, moody lighting",
    lora_keys: list[str] | None = None,
) -> list[dict]:
    """Generate images for all scenes. Returns updated scenes with image_paths."""
    images_dir = project_dir / "images"
    images_dir.mkdir(exist_ok=True)

    import asyncio

    total_prompts = sum(
        len(s.get("image_prompts") or []) or (1 if s.get("image_prompt") else 0)
        for s in scenes
    )
    generated = 0

    for scene in scenes:
        idx = scene["index"]
        # Use image_prompts list if available, fall back to single image_prompt
        prompts = scene.get("image_prompts") or []
        if not prompts:
            single = scene.get("image_prompt", "")
            prompts = [single] if single else []

        scene["image_paths"] = []
        scene.pop("image_error", None)  # clear previous errors

        for img_idx, prompt in enumerate(prompts):
            output_path = images_dir / f"scene_{idx:04d}_img_{img_idx}.png"

            # Throttle Replicate calls to avoid rate-limiting
            # Rate limit: 6 requests/min with <$5 credit = 1 request every 10s minimum
            if backend == "replicate" and generated > 0:
                delay = config.REPLICATE_DELAY_SECONDS
                log.info(f"Throttling: waiting {delay:.1f}s before next Replicate call (rate limit: 6 req/min with <$5 credit)")
                await asyncio.sleep(delay)

            try:
                if backend == "comfyui":
                    await generate_image_comfyui(
                        prompt=prompt,
                        style_prompt=style_prompt,
                        output_path=output_path,
                        seed=idx * 1000 + img_idx * 42,
                        lora_keys=lora_keys,
                    )
                elif backend == "replicate":
                    await generate_image_replicate(
                        prompt=prompt,
                        style_prompt=style_prompt,
                        output_path=output_path,
                        seed=idx * 1000 + img_idx * 42,
                        lora_keys=lora_keys,
                    )
                else:
                    await generate_image_ollama(
                        prompt=prompt,
                        style_prompt=style_prompt,
                        output_path=output_path,
                    )
                rel = str(output_path.relative_to(project_dir))
                scene["image_paths"].append(rel)
                generated += 1
                log.info(f"Scene {idx} image {img_idx + 1}/{len(prompts)} done ({generated}/{total_prompts} total)")
            except Exception as e:
                log.error(f"Image generation failed for scene {idx} img {img_idx}: {e}")
                # Fallback to text placeholder
                try:
                    await generate_image_ollama(
                        prompt=prompt,
                        style_prompt="",
                        output_path=output_path,
                    )
                    rel = str(output_path.relative_to(project_dir))
                    scene["image_paths"].append(rel)
                except Exception as e2:
                    log.error(f"Placeholder also failed for scene {idx} img {img_idx}: {e2}")
                    scene["image_error"] = str(e)
                generated += 1

        # Backward compat: set image_path to first image
        scene["image_path"] = scene["image_paths"][0] if scene["image_paths"] else None

    return scenes


async def _async_sleep(seconds: float):
    import asyncio
    await asyncio.sleep(seconds)
