"""Story Teller configuration."""

import shutil
import os
from pathlib import Path

# ── Directories ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECTS_DIR = BASE_DIR / "projects"
OUTPUT_DIR = BASE_DIR / "output"
DATA_DIR = BASE_DIR / "data"
TALES_DIR = DATA_DIR / "tales"
MUSIC_DIR = DATA_DIR / "music"

PROJECTS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
TALES_DIR.mkdir(exist_ok=True)
MUSIC_DIR.mkdir(exist_ok=True)

# ── External services ────────────────────────────────────────
VOICEBOX_URL = os.getenv("VOICEBOX_URL", "http://localhost:17493")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "kimi-k2.5:cloud")
COMFYUI_URL = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")

# ── Stock media + music APIs ─────────────────────────────────
# Shared with yt_facts_video_gen — keys live in start_full.bat
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "").strip()
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY", "").strip()
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "").strip()

# Jamendo: free royalty-free instrumental music (https://devportal.jamendo.com)
JAMENDO_CLIENT_ID = os.getenv("JAMENDO_CLIENT_ID", "").strip()
JAMENDO_URL = "https://api.jamendo.com/v3.0"

# ── Replicate (cloud image generation) ──────────────────────
# Strip whitespace — a trailing space (common from `set KEY=value ` in .bat files
# or copy-paste) makes httpx reject the "Authorization: Bearer ..." header with
# "Illegal header value". Write the cleaned value back so the replicate SDK,
# which reads os.environ directly, also sees the normalized token.
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "").strip()
if REPLICATE_API_TOKEN:
    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
# Use LoRA-enabled models for style control
REPLICATE_MODEL = os.getenv("REPLICATE_MODEL", "black-forest-labs/flux-dev-lora")  # or flux-schnell-lora
REPLICATE_TIMEOUT_SECONDS = float(os.getenv("REPLICATE_TIMEOUT_SECONDS", "120.0"))
REPLICATE_DELAY_SECONDS = float(os.getenv("REPLICATE_DELAY_SECONDS", "11.0"))  # delay between API calls (6/min rate limit with <$5 credit)
REPLICATE_MAX_RETRIES = int(os.getenv("REPLICATE_MAX_RETRIES", "3"))  # retries on rate-limit / transient errors

# FLUX LoRA URLs (public HuggingFace safetensors URLs)
# These are loaded dynamically via Replicate's lora_weights parameter
# All URLs verified working (HTTP 200, no auth required) as of 2026-04-23
FLUX_LORA_URLS = {
    # Victorian Gothic Horror — sepia-toned, aged, haunting aesthetic (trigger: "vicgoth")
    "tim_burton": os.getenv(
        "FLUX_LORA_TIM_BURTON",
        "https://huggingface.co/Keltezaa/victorian-gothic-horror/resolve/main/victoriangothic_v50_rank64_bf16-step01500.safetensors"
    ),
    # Dark Fantasy Illustration — dark fantasy retro illustrations (no trigger word, strength 1.2)
    "dark_gothic": os.getenv(
        "FLUX_LORA_DARK_GOTHIC",
        "https://huggingface.co/nerijs/dark-fantasy-illustration-flux/resolve/main/darkfantasy_illustration_v2.safetensors"
    ),
    # Shakker-Labs Dark Fantasy — fantasy creatures, metallic textures, magical light
    "dark_fantasy": os.getenv(
        "FLUX_LORA_DARK_FANTASY",
        "https://huggingface.co/Shakker-Labs/FLUX.1-dev-LoRA-Dark-Fantasy/resolve/main/FLUX.1-dev-lora-Dark-Fantasy.safetensors"
    ),
    # Doodle Toon — whimsical storybook illustration (trigger: "d00dlet00n")
    "storybook": os.getenv(
        "FLUX_LORA_STORYBOOK",
        "https://huggingface.co/renderartist/doodletoonflux/resolve/main/d00dlet00n_Flux_v2_renderartist.safetensors"
    ),
    # Flux Surrealism — surrealist/dreamlike art (trigger: "evangsurreal")
    "mark_ryden": os.getenv(
        "FLUX_LORA_MARK_RYDEN",
        "https://huggingface.co/brushpenbob/Flux-surrealism/resolve/main/Flux_surrealism.safetensors"
    ),
    # Painterly Illustration — realistic/painterly blend (no trigger; use generic style prefix)
    "painterly_illustration": os.getenv(
        "FLUX_LORA_PAINTERLY",
        "https://huggingface.co/Shakker-Labs/FLUX.1-dev-LoRA-Vector-Journey/resolve/main/FLUX-dev-lora-Vector-Journey.safetensors"
    ),
    # Golden Hour atmosphere — warm tones, dust particles (trigger: "Golden Dust")
    "golden_atmosphere": os.getenv(
        "FLUX_LORA_GOLDEN",
        "https://huggingface.co/prithivMLmods/Golden-Dust-Flux-LoRA/resolve/main/Golden-Dust.safetensors"
    ),
    # Ghibsky — Ghibli + Shinkai landscapes (trigger: "GHIBSKY style")
    "ghibli_whimsical": os.getenv(
        "FLUX_LORA_GHIBLI",
        "https://huggingface.co/aleksa-codes/flux-ghibsky-illustration/resolve/main/lora.safetensors"
    ),
    # Children Simple Sketch — pastel hand-drawn (trigger: "sketched style")
    "children_sketch": os.getenv(
        "FLUX_LORA_CHILDREN_SKETCH",
        "https://huggingface.co/Shakker-Labs/FLUX.1-dev-LoRA-Children-Simple-Sketch/resolve/main/FLUX-dev-lora-children-simple-sketch.safetensors"
    ),
    # MJ Painterly — concept-art-style cinematic paintings (trigger: "mj painterly")
    "concept_art": os.getenv(
        "FLUX_LORA_CONCEPT_ART",
        "https://huggingface.co/Shakker-Labs/FLUX.1-dev-LoRA-MiaoKa-Yarn-World/resolve/main/FLUX-dev-lora-MiaoKa-Yarn-World.safetensors"
    ),
    # Sketch-Paint — linework + painted washes (trigger: "sk3tchpa1nt")
    "sketch_paint": os.getenv(
        "FLUX_LORA_SKETCH_PAINT",
        "https://huggingface.co/renderartist/weirdthingsflux/resolve/main/Weird_Things_Flux_v1_renderartist.safetensors"
    ),
}

# Alternative verified FLUX LoRA URLs (not wired into the main selector)
FLUX_LORA_ALTERNATIVES = {
    "dark_fantasy_alt": "https://huggingface.co/Omarito2412/Dark-Fantasy-Flux/resolve/main/dark_fantasy_flux.safetensors",
    "dark_creature": "https://huggingface.co/prithivMLmods/Dark-Thing-Flux-LoRA/resolve/main/Dark_Creature.safetensors",
    "weird_surreal": "https://huggingface.co/renderartist/weirdthingsflux/resolve/main/Weird_Things_Flux_v1_renderartist.safetensors",
}

# ── CivitAI (optional, for gated model URLs) ─────────────────
CIVITAI_API_TOKEN = os.getenv("CIVITAI_API_TOKEN", "")

# ── Gutenberg ───────────────────────────────────────────────
GUTENBERG_TIMEOUT_SECONDS = 30.0
GUTENBERG_TEXT_TIMEOUT_SECONDS = 30.0

# ── Video output ─────────────────────────────────────────────
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
VIDEO_FPS = 25
CROSSFADE_DURATION = 0.5  # seconds between scenes

# ── LLM ──────────────────────────────────────────────────────
LLM_TIMEOUT_SECONDS = 300.0
LLM_TEMPERATURE = 0.8
LLM_MAX_TOKENS = 16000

# ── Batch chapter analysis ───────────────────────────────────
BATCH_NARRATION_RATE = 800  # characters per minute for duration estimation

# ── Voice ────────────────────────────────────────────────────
VOICE_TIMEOUT_SECONDS = 180.0

# ── Images ───────────────────────────────────────────────────
IMAGE_TIMEOUT_SECONDS = 120.0
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

# ── AnimateDiff ─────────────────────────────────────────────
ANIMATEDIFF_ENABLED = True
ANIMATEDIFF_SD15_CHECKPOINT = "v1-5-pruned-emaonly.safetensors"
ANIMATEDIFF_MOTION_MODULE = "v3_sd15_mm.ckpt"
ANIMATEDIFF_WIDTH = 768   # must be divisible by 8
ANIMATEDIFF_HEIGHT = 512   # must be divisible by 8
ANIMATEDIFF_DEFAULT_FRAMES = 16
ANIMATEDIFF_DEFAULT_FPS = 8
ANIMATEDIFF_TIMEOUT_SECONDS = 300.0

# ── Image QC ────────────────────────────────────────────────
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llava:7b")
QC_PASS_THRESHOLD = float(os.getenv("QC_PASS_THRESHOLD", "3.0"))
QC_MAX_RETRIES = int(os.getenv("QC_MAX_RETRIES", "2"))
QC_TIMEOUT_SECONDS = float(os.getenv("QC_TIMEOUT_SECONDS", "300.0"))

# ── Background music ─────────────────────────────────────────
# Default volume for background music relative to voice (0.0-1.0)
MUSIC_DEFAULT_VOLUME = float(os.getenv("MUSIC_DEFAULT_VOLUME", "0.18"))
# Fade in/out seconds applied to the music track
MUSIC_FADE_SECONDS = float(os.getenv("MUSIC_FADE_SECONDS", "2.5"))
MUSIC_EXTENSIONS = (".mp3", ".wav", ".ogg", ".flac", ".m4a")

# ── Ken Burns defaults ───────────────────────────────────────
KB_ZOOM_RANGE = (1.0, 1.15)  # start/end zoom range
KB_DIRECTIONS = ["zoom_in", "zoom_out", "pan_left", "pan_right"]

# ── Animation / Depth Parallax ──────────────────────────────
PARALLAX_STRENGTH = 80.0  # max displacement in pixels at overscan resolution
DEPTH_METHOD = "gradient"  # "gradient" (fast, no deps), "comfyui" (MiDaS node required), "auto" (try comfyui, fallback gradient)

# ── FFmpeg ───────────────────────────────────────────────────
FFMPEG_PATH = shutil.which("ffmpeg") or "ffmpeg"

# Detect winget-installed FFmpeg on Windows
if os.name == "nt" and not shutil.which("ffmpeg"):
    _winget = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/WinGet/Links"
    if (_winget / "ffmpeg.exe").exists():
        os.environ["PATH"] = str(_winget) + os.pathsep + os.environ["PATH"]
        FFMPEG_PATH = str(_winget / "ffmpeg.exe")
