"""Image generation via ComfyUI, Replicate, OpenAI GPT Image, or Ollama."""

import base64
import re
import time
import uuid
import logging
from pathlib import Path
import httpx

from . import config, image_styles
from . import project_store as store

log = logging.getLogger(__name__)


class OpenAIImageAccessError(RuntimeError):
    """Raised when OpenAI rejects the image request for auth or org access."""


class OpenAIImageSafetyError(RuntimeError):
    """Raised when OpenAI rejects the image request for safety reasons."""

    def __init__(self, message: str, request_id: str | None = None):
        super().__init__(message)
        self.request_id = request_id

# ── Available LoRAs ─────────────────────────────────────────────
# Each entry: filename, trigger words to prepend, strength_model, strength_clip
# For ComfyUI: Uses local .safetensors files
# For Replicate: Uses flux_lora_key to look up URL in config.FLUX_LORA_URLS
AVAILABLE_LORAS = {
    "tim_burton": {
        "file": "Tim_Burton_Painting_Style_SDXL.safetensors",
        "trigger": "vicgoth",
        "strength_model": 1.0,
        "strength_clip": 0.7,
        "flux_lora_key": "tim_burton",
        "description": "Sepia-toned Victorian horror with aged, haunting aesthetic and dramatic contrasts",
    },
    "storybook": {
        "file": "StorybookRedmondV2.safetensors",
        "trigger": "d00dlet00n",
        "strength_model": 0.8,
        "strength_clip": 0.6,
        "flux_lora_key": "storybook",
        "description": "Hand-drawn storybook illustration with marker and pencil textures",
    },
    "dark_gothic": {
        "file": "dark_gothic_fantasy_xl.safetensors",
        "trigger": "",
        "strength_model": 1.2,
        "strength_clip": 0.65,
        "flux_lora_key": "dark_gothic",
        "description": "Rich, painterly dark fantasy with deep shadows and atmospheric lighting",
    },
    "mark_ryden": {
        "file": "Mark_Ryden_Style.safetensors",
        "trigger": "evangsurreal",
        "strength_model": 0.8,
        "strength_clip": 0.7,
        "flux_lora_key": "mark_ryden",
        "description": "Dreamlike surrealist art with otherworldly atmosphere and sci-fi undertones",
    },
    "painterly_illustration": {
        "file": "",
        "trigger": "artistic style blends reality and illustration elements",
        "strength_model": 0.8,
        "strength_clip": 0.7,
        "flux_lora_key": "painterly_illustration",
        "description": "Illustrated characters with realistic backgrounds, blending painterly and photographic styles",
    },
    "golden_atmosphere": {
        "file": "",
        "trigger": "Golden Dust",
        "strength_model": 0.8,
        "strength_clip": 0.7,
        "flux_lora_key": "golden_atmosphere",
        "description": "Warm golden hour tones with luminous dust particles and sun-drenched light",
    },
    "ghibli_whimsical": {
        "file": "",
        "trigger": "GHIBSKY style",
        "strength_model": 0.8,
        "strength_clip": 0.7,
        "flux_lora_key": "ghibli_whimsical",
        "description": "Studio Ghibli meets Makoto Shinkai — warm, lush landscapes with hand-painted feel",
    },
    "children_sketch": {
        "file": "",
        "trigger": "sketched style",
        "strength_model": 0.8,
        "strength_clip": 0.7,
        "flux_lora_key": "children_sketch",
        "description": "Simple hand-drawn children's illustration with soft pastel colors and gentle linework",
    },
    "concept_art": {
        "file": "",
        "trigger": "mj painterly",
        "strength_model": 0.8,
        "strength_clip": 0.7,
        "flux_lora_key": "concept_art",
        "description": "Cinematic concept art with dramatic composition and rich painterly detail",
    },
    "sketch_paint": {
        "file": "",
        "trigger": "sk3tchpa1nt",
        "strength_model": 0.8,
        "strength_clip": 0.7,
        "flux_lora_key": "sketch_paint",
        "description": "Blend of sketch linework and painterly color washes, mixing drawing and painting",
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


def _build_kontext_input(
    prompt: str,
    style_prompt: str,
    seed: int,
    lora_keys: list[str] | None = None,
    reference_image: "Path | None" = None,
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
                    lora_url = config.FLUX_LORA_URLS.get(flux_key)
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
        "guidance_scale": 3.5,
    }

    if reference_image is not None:
        inp["image"] = reference_image

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
    reference_image: "Path | None" = None,
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
    inp = _build_kontext_input(prompt, style_prompt, seed, lora_keys, reference_image=reference_image)

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


GPT_IMAGE_STYLE_BOILERPLATE = [
    "dark fairy tale illustration",
    "gothic storybook art",
    "atmospheric",
    "detailed",
    "moody lighting",
    "dramatic shadows",
    "rich colors",
]

GPT_IMAGE_SAFETY_REPLACEMENTS = [
    (r"\bTim Burton(?: inspired| style)?\b", "whimsical gothic silhouettes"),
    (r"\bStudio Ghibli(?: style)?\b", "warm hand-painted animation aesthetic"),
    (r"\bYoung Talia\b", "adult Talia"),
    (r"\bTalia's\b", "adult Talia's"),
    (r"\bhis daughter\b", "his adult daughter"),
    (r"\blittle girl\b", "young princess"),
    (r"\bparanoid intensity\b", "grave concern"),
    (r"\bcollapsing backward\b", "sinking backward in a dramatic faint"),
    (r"\bvisible splinter of flax under her fingernail\b", "tiny golden flax fiber near her hand"),
    (r"\bsplinter of flax under her fingernail\b", "tiny golden flax fiber near her hand"),
    (r"\bfingernail\b", "hand"),
    (r"\bblood on her hands\b", "dark red ribbon around her hands"),
    (r"\bblood\b", "deep red color"),
    (r"\bsevered\b", "broken"),
    (r"\bcorpse\b", "motionless figure"),
    (r"\bbody lying\b", "resting figure"),
    (r"\bdead\b", "silent"),
    (r"\bdeath\b", "sleep"),
    (r"\bpregnant beneath her velvet gown\b", "wearing a full velvet gown"),
    (r"\bpregnant\b", "wearing a full velvet gown"),
    (r"\brounded belly under the rich velvet fabric\b", "rich velvet gown arranged in soft folds"),
    (r"\bbelly\b", "gown"),
    (r"\bhands resting passively on it\b", "hands resting calmly on the velvet fabric"),
    (r"\bdisheveled\b", "rumpled"),
    (r"\bnewborn baby nursing from Talia's breast while she sleeps\b", "newborn baby resting safely beside adult Talia while she sleeps"),
    (r"\bnewborn baby nursing from adult Talia's breast while she sleeps\b", "newborn baby resting safely beside adult Talia while she sleeps"),
    (r"\bnewborn baby nursing from adult Talia's blanket while she sleeps\b", "newborn baby resting safely beside adult Talia while she sleeps"),
    (r"\bnursing from Talia's breast\b", "resting safely beside adult Talia"),
    (r"\bnursing infants\b", "newborn twins"),
    (r"\bbreast\b", "blanket"),
    (r"\bsucking on Talia's finger\b", "gently holding adult Talia's hand"),
    (r"\bBaby Moon sucking on adult Talia's finger\b", "Baby Moon gently holding adult Talia's hand"),
    (r"\btiny flax splinter\b", "tiny golden flax fiber"),
    (r"\binfant's tongue\b", "fold of golden cloth"),
    (r"\binfant's small hand against her skin\b", "infant's small hand on a blanket"),
    (r"\btongue\b", "golden cloth"),
    (r"\bmouth\b", "expression"),
    (r"\bscream of horror\b", "startled gasp"),
    (r"\bhorror\b", "shock"),
    (r"\bnewborn twins crying\b", "newborn twins resting safely"),
    (r"\bdrag a screaming Talia away\b", "escort adult Talia away in a tense court scene"),
    (r"\bCook holding the twins as soldiers escort adult Talia away\b", "Cook keeping the twins safely aside while guards escort adult Talia through the room"),
    (r"\bshe reaches for them desperately\b", "she looks back with worry"),
    (r"\bscreaming\b", "distressed"),
    (r"\bguards holding her arms\b", "guards standing nearby"),
    (r"\bbound with ropes\b", "surrounded by guards"),
    (r"\biron chains\b", "shadowed court garments"),
    (r"\bchains\b", "dark ribbons"),
    (r"\bcarving knives\b", "copper utensils"),
    (r"\bhunting knife\b", "small ceremonial blade"),
    (r"\bsmall ceremonial blade\b", "small ceremonial tool"),
    (r"\bcutting the ropes binding\b", "loosening the ceremonial ribbons around"),
    (r"\baround Talia to a wooden stake\b", "near adult Talia beside a wooden post"),
    (r"\baround adult Talia to a wooden stake\b", "near adult Talia beside a wooden post"),
    (r"\bloosening the ceremonial ribbons around adult Talia to a wooden wooden post\b", "loosening ceremonial ribbons near adult Talia beside a wooden post"),
    (r"\baround adult Talia to a wooden wooden post\b", "near adult Talia beside a wooden post"),
    (r"\bwooden pyre\b", "ceremonial wooden platform"),
    (r"\bpyre\b", "ceremonial platform"),
    (r"\bflames nearby catching the edge of her velvet dress\b", "firelight glowing at a safe distance"),
    (r"\bflames beginning around her feet\b", "firelight glowing on the stones nearby"),
    (r"\bhair catching fire\b", "hair lit by orange firelight"),
    (r"\bwatching the burning\b", "watching the firelit courtyard"),
    (r"\badult Talia watching the firelit courtyard, holding both twins close to her chest\b", "adult Talia standing safely away from the firelit courtyard with both twins nearby"),
    (r"\bholding both twins close to her chest\b", "with both twins nearby at a safe distance"),
    (r"\bburning\b", "firelit courtyard scene"),
    (r"\bthe flames firelit courtyard scene brightly\b", "warm firelight glowing"),
    (r"\bmeat on his plate\b", "covered dish on his plate"),
    (r"\btied to the wooden stake\b", "standing before a wooden post"),
    (r"\btied to\b", "standing beside"),
    (r"\bstake\b", "wooden post"),
    (r"\bcannibalism\b", "cruel banquet intrigue"),
    (r"\bcannibal\b", "cruel banquet"),
    (r"\broasted meat\b", "covered banquet dish"),
    (r"\bsmall children are concealed\b", "children are hidden safely"),
]


def _normalize_prompt_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "")
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*,+", ", ", text)
    return text.strip(" ,")


def _strip_gpt_image_style_boilerplate(text: str) -> str:
    cleaned = text or ""
    for phrase in GPT_IMAGE_STYLE_BOILERPLATE:
        cleaned = re.sub(rf"(?:,\s*)?\b{re.escape(phrase)}\b", "", cleaned, flags=re.IGNORECASE)
    return _normalize_prompt_text(cleaned)


def _make_gpt_image_safe_text(text: str) -> str:
    safe_text = _strip_gpt_image_style_boilerplate(text)
    for pattern, replacement in GPT_IMAGE_SAFETY_REPLACEMENTS:
        safe_text = re.sub(pattern, replacement, safe_text, flags=re.IGNORECASE)

    if re.search(r"\bTalia\b", safe_text, flags=re.IGNORECASE) and not re.search(
        r"\badult Talia\b|\bLord Talia\b|\bnewborn\b|\binfant\b|\bbaby\b",
        safe_text,
        flags=re.IGNORECASE,
    ):
        safe_text = re.sub(r"\bTalia\b", "adult Talia", safe_text, count=1, flags=re.IGNORECASE)

    safe_text = re.sub(r"\bLord adult Talia\b", "Lord Talia", safe_text, flags=re.IGNORECASE)
    return _normalize_prompt_text(safe_text)


def _build_gpt_image_prompt(
    prompt: str,
    style_prompt: str,
    lora_keys: list[str] | None = None,
    safety_mode: bool = False,
) -> str:
    style_parts = []
    if style_prompt:
        style_parts.append(style_prompt)

    # GPT Image cannot load LoRA weights, but preserving the selected style
    # descriptions keeps batch presets meaningful when this backend is selected.
    if lora_keys:
        descriptions = [
            AVAILABLE_LORAS[key]["description"]
            for key in lora_keys
            if key in AVAILABLE_LORAS and AVAILABLE_LORAS[key].get("description")
        ]
        if descriptions:
            style_parts.append("Visual style references: " + "; ".join(descriptions))

    style_text = ", ".join(style_parts) or "cinematic storybook illustration"
    scene_text = _strip_gpt_image_style_boilerplate(prompt)

    if safety_mode:
        scene_text = _make_gpt_image_safe_text(scene_text)
        style_text = _make_gpt_image_safe_text(style_text)
        return (
            "Create a cinematic 16:9 storybook illustration for an adult folklore video.\n"
            f"Scene: {scene_text}\n"
            f"Style: {style_text}\n"
            "Keep the visual symbolic and non-graphic: no gore, no visible injury, "
            "no sexual content, no intimate contact, no restraint, no torture, "
            "no explicit burning, and no minors in danger. Use atmospheric lighting, "
            "strong composition, rich detail, and no captions, watermarks, UI elements, "
            "logos, or unintended text."
        )

    style_text = _strip_gpt_image_style_boilerplate(style_text)

    return (
        "Create a cinematic 16:9 illustration for an adult folklore story video.\n"
        f"Scene: {scene_text}\n"
        f"Style: {style_text}\n"
        "Use atmospheric lighting, strong composition, rich detail, and no captions, "
        "watermarks, UI elements, logos, or unintended text."
    )


def _extract_openai_error(resp: httpx.Response) -> tuple[str, dict, str | None]:
    detail = resp.text
    error_obj: dict = {}
    request_id = resp.headers.get("x-request-id") or resp.headers.get("openai-request-id")
    try:
        parsed = resp.json()
        if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
            error_obj = parsed["error"]
            detail = error_obj.get("message") or detail
    except Exception:
        pass
    return detail, error_obj, request_id


def _is_openai_safety_error(status_code: int, detail: str, error_obj: dict) -> bool:
    fields = [detail]
    for key in ("message", "code", "type", "param"):
        value = error_obj.get(key)
        if value:
            fields.append(str(value))
    haystack = " ".join(fields).lower()
    markers = (
        "safety",
        "content_policy",
        "content policy",
        "moderation",
        "unsafe",
        "disallowed",
        "rejected",
    )
    return status_code == 400 and any(marker in haystack for marker in markers)


def _raise_openai_image_error(resp: httpx.Response) -> None:
    detail, error_obj, request_id = _extract_openai_error(resp)
    request_text = f" Request ID: {request_id}." if request_id else ""

    if resp.status_code in (401, 403):
        raise OpenAIImageAccessError(
            f"OpenAI image generation failed ({resp.status_code}): {detail}{request_text}"
        )

    if _is_openai_safety_error(resp.status_code, detail, error_obj):
        raise OpenAIImageSafetyError(
            "OpenAI rejected this image prompt for safety reasons."
            f"{request_text} Try a symbolic, non-graphic version of the scene.",
            request_id=request_id,
        )

    raise RuntimeError(f"OpenAI image generation failed ({resp.status_code}): {detail}{request_text}")


async def _request_gpt_image(client: httpx.AsyncClient, payload: dict) -> bytes:
    resp = await client.post(
        f"{config.OPENAI_IMAGE_BASE_URL}/images/generations",
        json=payload,
    )

    if resp.status_code >= 400:
        _raise_openai_image_error(resp)

    data = resp.json()
    images = data.get("data") or []
    if not images:
        raise RuntimeError("OpenAI image generation returned no image data")

    image_info = images[0]
    b64_json = image_info.get("b64_json")
    if b64_json:
        return base64.b64decode(b64_json)
    if image_info.get("url"):
        image_resp = await client.get(image_info["url"], timeout=60)
        image_resp.raise_for_status()
        return image_resp.content
    raise RuntimeError("OpenAI image generation response had no b64_json or url")


async def generate_image_gpt_image(
    prompt: str,
    style_prompt: str,
    output_path: Path,
    seed: int | None = None,
    lora_keys: list[str] | None = None,
) -> Path:
    """Generate an image using OpenAI GPT Image 2 through the Images API."""
    if not config.OPENAI_API_KEY:
        raise RuntimeError("OpenAI image backend requires OPENAI_API_KEY in the repo .env file.")

    headers = {
        "Authorization": f"Bearer {config.OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    if config.OPENAI_ORG_ID:
        headers["OpenAI-Organization"] = config.OPENAI_ORG_ID

    payload = {
        "model": config.OPENAI_IMAGE_MODEL,
        "prompt": _build_gpt_image_prompt(prompt, style_prompt, lora_keys),
        "n": 1,
        "size": config.OPENAI_IMAGE_SIZE,
        "quality": config.OPENAI_IMAGE_QUALITY,
        "output_format": config.OPENAI_IMAGE_FORMAT,
    }
    if config.OPENAI_IMAGE_BACKGROUND:
        payload["background"] = config.OPENAI_IMAGE_BACKGROUND

    log.info(
        "OpenAI image [%s]: generating size=%s quality=%s seed=%s",
        config.OPENAI_IMAGE_MODEL,
        config.OPENAI_IMAGE_SIZE,
        config.OPENAI_IMAGE_QUALITY,
        seed,
    )

    async with httpx.AsyncClient(
        timeout=config.OPENAI_IMAGE_TIMEOUT_SECONDS,
        headers=headers,
    ) as client:
        try:
            image_bytes = await _request_gpt_image(client, payload)
        except OpenAIImageSafetyError as first_error:
            log.warning(
                "OpenAI safety rejected image prompt; retrying with sanitized prompt%s",
                f" (request_id={first_error.request_id})" if first_error.request_id else "",
            )
            safe_payload = dict(payload)
            safe_payload["prompt"] = _build_gpt_image_prompt(
                prompt,
                style_prompt,
                lora_keys,
                safety_mode=True,
            )
            try:
                image_bytes = await _request_gpt_image(client, safe_payload)
            except OpenAIImageSafetyError as retry_error:
                raise OpenAIImageSafetyError(
                    "OpenAI rejected this image prompt after a safe retry."
                    " Rewrite the scene as symbolic, non-graphic folklore imagery and retry.",
                    request_id=retry_error.request_id or first_error.request_id,
                ) from retry_error

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(image_bytes)
    log.info(f"OpenAI image saved to {output_path}")
    return output_path


def _record_scene_image_error(
    scene: dict,
    image_index: int,
    message: str,
    *,
    safety: bool = False,
) -> None:
    image_errors = scene.setdefault("image_errors", [])
    while len(image_errors) <= image_index:
        image_errors.append(None)
    image_errors[image_index] = message

    active_errors = [err for err in image_errors if err]
    if len(active_errors) == 1:
        scene["image_error"] = active_errors[0]
    elif active_errors:
        scene["image_error"] = f"{len(active_errors)} image prompts failed. First error: {active_errors[0]}"

    if safety:
        scene["image_safety_error"] = True


def _finalize_scene_image_errors(scene: dict) -> None:
    image_errors = scene.get("image_errors") or []
    if not any(image_errors):
        scene.pop("image_errors", None)
        scene.pop("image_error", None)
        scene.pop("image_safety_error", None)


async def generate_scene_images(
    scene: dict,
    project_dir: Path,
    backend: str = "comfyui",
    style_prompt: str = image_styles.DEFAULT_STYLE_PROMPT,
    lora_keys: list[str] | None = None,
    reference_image: Path | None = None,
    project_seed: int | None = None,
) -> dict:
    """Generate images for a single scene. Returns updated scene with image_paths."""
    images_dir = project_dir / "images"
    images_dir.mkdir(exist_ok=True)

    idx = scene["index"]
    prompts = scene.get("image_prompts") or []
    if not prompts:
        single = scene.get("image_prompt", "")
        prompts = [single] if single else []

    scene["image_paths"] = []
    scene["image_errors"] = []
    scene.pop("image_error", None)
    scene.pop("image_safety_error", None)

    for img_idx, prompt in enumerate(prompts):
        output_path = images_dir / f"scene_{idx:04d}_img_{img_idx}.png"
        image_seed = store.derive_image_seed(project_seed, idx, img_idx)

        if backend == "gpt_image" and img_idx > 0 and config.OPENAI_IMAGE_DELAY_SECONDS > 0:
            delay = config.OPENAI_IMAGE_DELAY_SECONDS
            log.info(f"Throttling: waiting {delay:.1f}s before next OpenAI image call")
            await _async_sleep(delay)

        try:
            if backend == "comfyui":
                await generate_image_comfyui(
                    prompt=prompt,
                    style_prompt=style_prompt,
                    output_path=output_path,
                    seed=image_seed,
                    lora_keys=lora_keys,
                )
            elif backend == "replicate":
                await generate_image_replicate(
                    prompt=prompt,
                    style_prompt=style_prompt,
                    output_path=output_path,
                    seed=image_seed,
                    lora_keys=lora_keys,
                    reference_image=reference_image,
                )
            elif backend == "gpt_image":
                await generate_image_gpt_image(
                    prompt=prompt,
                    style_prompt=style_prompt,
                    output_path=output_path,
                    seed=image_seed,
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
            log.info(f"Scene {idx} image {img_idx + 1}/{len(prompts)} done")
        except OpenAIImageAccessError as e:
            log.error(f"OpenAI image access failed for scene {idx} img {img_idx}: {e}")
            scene["image_error"] = str(e)
            raise
        except OpenAIImageSafetyError as e:
            log.error(f"OpenAI image safety rejection for scene {idx} img {img_idx}: {e}")
            _record_scene_image_error(scene, img_idx, str(e), safety=True)
        except Exception as e:
            log.error(f"Image generation failed for scene {idx} img {img_idx}: {e}")
            try:
                await generate_image_ollama(
                    prompt=prompt,
                    style_prompt="",
                    output_path=output_path,
                )
                rel = str(output_path.relative_to(project_dir))
                scene["image_paths"].append(rel)
                _record_scene_image_error(
                    scene,
                    img_idx,
                    f"Generated placeholder after image backend failed: {e}",
                )
            except Exception as e2:
                log.error(f"Placeholder also failed for scene {idx} img {img_idx}: {e2}")
                _record_scene_image_error(scene, img_idx, str(e))

    scene["image_path"] = scene["image_paths"][0] if scene["image_paths"] else None
    _finalize_scene_image_errors(scene)
    return scene


async def generate_all_scenes(
    scenes: list[dict],
    project_dir: Path,
    backend: str = "comfyui",
    style_prompt: str = image_styles.DEFAULT_STYLE_PROMPT,
    lora_keys: list[str] | None = None,
    character_consistency: bool = False,
    project_seed: int | None = None,
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
    reference_image_path: Path | None = None

    for scene in scenes:
        idx = scene["index"]
        # Use image_prompts list if available, fall back to single image_prompt
        prompts = scene.get("image_prompts") or []
        if not prompts:
            single = scene.get("image_prompt", "")
            prompts = [single] if single else []

        scene["image_paths"] = []
        scene["image_errors"] = []
        scene.pop("image_error", None)  # clear previous errors
        scene.pop("image_safety_error", None)

        for img_idx, prompt in enumerate(prompts):
            output_path = images_dir / f"scene_{idx:04d}_img_{img_idx}.png"
            image_seed = store.derive_image_seed(project_seed, idx, img_idx)

            # Throttle cloud image calls to avoid common low-tier rate limits.
            if backend == "replicate" and generated > 0:
                delay = config.REPLICATE_DELAY_SECONDS
                log.info(f"Throttling: waiting {delay:.1f}s before next Replicate call")
                await asyncio.sleep(delay)
            elif backend == "gpt_image" and generated > 0 and config.OPENAI_IMAGE_DELAY_SECONDS > 0:
                delay = config.OPENAI_IMAGE_DELAY_SECONDS
                log.info(f"Throttling: waiting {delay:.1f}s before next OpenAI image call")
                await asyncio.sleep(delay)

            try:
                if backend == "comfyui":
                    await generate_image_comfyui(
                        prompt=prompt,
                        style_prompt=style_prompt,
                        output_path=output_path,
                        seed=image_seed,
                        lora_keys=lora_keys,
                    )
                elif backend == "replicate":
                    await generate_image_replicate(
                        prompt=prompt,
                        style_prompt=style_prompt,
                        output_path=output_path,
                        seed=image_seed,
                        lora_keys=lora_keys,
                        reference_image=reference_image_path,
                    )
                elif backend == "gpt_image":
                    await generate_image_gpt_image(
                        prompt=prompt,
                        style_prompt=style_prompt,
                        output_path=output_path,
                        seed=image_seed,
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
                # Capture first image as reference for character consistency
                if character_consistency and backend == "replicate" and reference_image_path is None:
                    reference_image_path = output_path
                    log.info(f"Character consistency: using {output_path} as reference image")
                generated += 1
                log.info(f"Scene {idx} image {img_idx + 1}/{len(prompts)} done ({generated}/{total_prompts} total)")
            except OpenAIImageAccessError as e:
                log.error(f"OpenAI image access failed for scene {idx} img {img_idx}: {e}")
                scene["image_error"] = str(e)
                raise
            except OpenAIImageSafetyError as e:
                log.error(f"OpenAI image safety rejection for scene {idx} img {img_idx}: {e}")
                _record_scene_image_error(scene, img_idx, str(e), safety=True)
                generated += 1
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
                    _record_scene_image_error(
                        scene,
                        img_idx,
                        f"Generated placeholder after image backend failed: {e}",
                    )
                except Exception as e2:
                    log.error(f"Placeholder also failed for scene {idx} img {img_idx}: {e2}")
                    _record_scene_image_error(scene, img_idx, str(e))
                generated += 1

        # Backward compat: set image_path to first image
        scene["image_path"] = scene["image_paths"][0] if scene["image_paths"] else None
        _finalize_scene_image_errors(scene)

    return scenes


async def _async_sleep(seconds: float):
    import asyncio
    await asyncio.sleep(seconds)
