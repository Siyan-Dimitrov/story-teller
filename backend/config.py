"""Story Teller configuration."""

import shutil
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - allows startup before deps are installed
    def load_dotenv(path: Path) -> None:
        if not path.exists():
            return
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

# ── Directories ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

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
COMFYUI_URL = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")

# ── Voice backend (MiniMax Speech via Replicate) ─────────────
MINIMAX_TTS_MODEL = os.getenv("MINIMAX_TTS_MODEL", "minimax/speech-2.8-hd")
MINIMAX_DEFAULT_VOICE = os.getenv("MINIMAX_DEFAULT_VOICE", "English_Deep-VoicedGentleman")
MINIMAX_SPEED = float(os.getenv("MINIMAX_SPEED", "1.0"))  # 0.9–0.95 for slower storytelling pace
MINIMAX_SAMPLE_RATE = int(os.getenv("MINIMAX_SAMPLE_RATE", "44100"))
MINIMAX_MAX_RETRIES = int(os.getenv("MINIMAX_MAX_RETRIES", "3"))
MINIMAX_DELAY_SECONDS = float(os.getenv("MINIMAX_DELAY_SECONDS", "1.0"))  # between scene requests

# ── Stock media + music APIs ─────────────────────────────────
# Shared with yt_facts_video_gen — keys live in start_full.bat
# OpenAI GPT Image backend. The key is loaded from the repo-root .env file.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID", "").strip()
OPENAI_IMAGE_BASE_URL = os.getenv("OPENAI_IMAGE_BASE_URL", "https://api.openai.com/v1").rstrip("/")
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2").strip()
OPENAI_IMAGE_QUALITY = os.getenv("OPENAI_IMAGE_QUALITY", "medium").strip()
OPENAI_IMAGE_SIZE = os.getenv("OPENAI_IMAGE_SIZE", "2048x1152").strip()
OPENAI_IMAGE_FORMAT = os.getenv("OPENAI_IMAGE_FORMAT", "png").strip()
OPENAI_IMAGE_BACKGROUND = os.getenv("OPENAI_IMAGE_BACKGROUND", "auto").strip()
OPENAI_IMAGE_TIMEOUT_SECONDS = float(os.getenv("OPENAI_IMAGE_TIMEOUT_SECONDS", "240.0"))
OPENAI_IMAGE_DELAY_SECONDS = float(os.getenv("OPENAI_IMAGE_DELAY_SECONDS", "12.0"))

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

# ── Nano Banana (Google Gemini image, hosted on Replicate) ──
# Reference-image-driven generation for character consistency. Accepts a text
# prompt plus up to ~14 reference images (`image_input` array) and keeps the
# referenced subjects/style consistent across scenes — no LoRA training needed.
# Default to the cheaper "google/nano-banana" (Gemini Flash Image, ~$0.04/img)
# — it still takes reference images for character consistency, which is all we
# need for stylized frames that get animated afterwards. Set this to
# "google/nano-banana-pro" (~$0.13/img) only when you want maximum face fidelity.
REPLICATE_NANO_BANANA_MODEL = os.getenv("REPLICATE_NANO_BANANA_MODEL", "google/nano-banana")
# Output format Nano Banana returns (png keeps things lossless for downstream
# compositing/animation).
NANO_BANANA_OUTPUT_FORMAT = os.getenv("NANO_BANANA_OUTPUT_FORMAT", "png").strip()
# Aspect ratio for the one-off character reference "model sheet" portraits.
NANO_BANANA_CHAR_ASPECT_RATIO = os.getenv("NANO_BANANA_CHAR_ASPECT_RATIO", "3:4").strip()
# Max reference images attached to a single scene generation (model hard cap ~14;
# leave headroom for a style anchor on top of character portraits).
NANO_BANANA_MAX_REFS = int(os.getenv("NANO_BANANA_MAX_REFS", "6"))
# Input property name Replicate's Nano Banana model uses for the reference-image
# array. "image_input" is the documented key for google/nano-banana[-pro]; kept
# configurable in case a model revision renames it.
NANO_BANANA_IMAGE_PARAM = os.getenv("NANO_BANANA_IMAGE_PARAM", "image_input").strip()

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
IMAGE_CROSSFADE_DURATION = 0.4  # seconds between images within a scene

# ── Shorts (vertical 9:16 clips) ─────────────────────────────
# A short is a self-contained mini-scene reframed to portrait, with burned-in
# captions and an end card driving viewers to the full story.
SHORT_WIDTH = int(os.getenv("SHORT_WIDTH", "1080"))
SHORT_HEIGHT = int(os.getenv("SHORT_HEIGHT", "1920"))
SHORT_FPS = int(os.getenv("SHORT_FPS", "30"))
SHORT_MIN_DURATION = float(os.getenv("SHORT_MIN_DURATION", "8.0"))
SHORT_MAX_DURATION = float(os.getenv("SHORT_MAX_DURATION", "58.0"))   # YT cap 60s
SHORT_TAIL_DURATION = float(os.getenv("SHORT_TAIL_DURATION", "2.5"))  # end card
# How many shorts the director picks per project by default.
SHORTS_PER_PROJECT = int(os.getenv("SHORTS_PER_PROJECT", "3"))
# Video/audio codecs for short encodes (match the long-form pipeline).
SHORT_VIDEO_CODEC = os.getenv("SHORT_VIDEO_CODEC", "libx264")
SHORT_AUDIO_CODEC = os.getenv("SHORT_AUDIO_CODEC", "aac")
# How a 16:9 segment is reframed to 9:16 when cutting from the finished video:
#   "fit"  — whole frame visible, centered over a blurred fill (no content lost)
#   "crop" — cover-crop to full-bleed vertical (loses the sides)
SHORT_REFRAME = os.getenv("SHORT_REFRAME", "fit").strip()
# Source for shorts: "final" cuts the segment straight out of final.mp4 (keeps
# music + motion, cheap — just ffmpeg); "stills" re-renders from scene images.
SHORT_SOURCE = os.getenv("SHORT_SOURCE", "final").strip()

# ── Claude text generation ───────────────────────────────────
# All text generation (scripts, critic, classification, metadata) runs on
# Claude via the Agent SDK, which reads OAuth credentials from
# ~/.claude/.credentials.json (created by `claude login`) — no API key needed
# when the user is signed into Claude Code.
# Creative work: script writer/critic/reviser, cast bible, export metadata.
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-4-8").strip()
# Small/structured calls: image classification, chapter tagging, music
# suggestions, chapter splitting. Fast and cheap on subscription quota.
CLAUDE_FAST_MODEL = os.getenv("CLAUDE_FAST_MODEL", "claude-haiku-4-5").strip()
# Per-role model overrides for the script pipeline. Empty = CLAUDE_MODEL.
PIPELINE_WRITER_MODEL  = os.getenv("PIPELINE_WRITER_MODEL", "").strip()
PIPELINE_CRITIC_MODEL  = os.getenv("PIPELINE_CRITIC_MODEL", "").strip()
PIPELINE_REVISER_MODEL = os.getenv("PIPELINE_REVISER_MODEL", "").strip()
# 0 = writer only (single pass); 1 = writer + critic + reviser (recommended).
CLAUDE_MAX_REVISIONS = int(os.getenv("CLAUDE_MAX_REVISIONS", "1"))
CLAUDE_TIMEOUT_SECONDS = float(os.getenv("CLAUDE_TIMEOUT_SECONDS", "600.0"))
# Shorter timeout for small/structured calls (classification etc.).
CLAUDE_FAST_TIMEOUT_SECONDS = float(os.getenv("CLAUDE_FAST_TIMEOUT_SECONDS", "180.0"))
# Directory containing the writer/critic/reviser markdown system prompts.
CLAUDE_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

# ── Batch chapter analysis ───────────────────────────────────
BATCH_NARRATION_RATE = 800  # characters per minute for duration estimation

# ── Voice ────────────────────────────────────────────────────
VOICE_TIMEOUT_SECONDS = 180.0

# ── Images ───────────────────────────────────────────────────
IMAGE_TIMEOUT_SECONDS = 120.0
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

# ── Image-to-Video (Replicate) ──────────────────────────────
# Replaces the legacy local AnimateDiff/ComfyUI path. Scenes the LLM
# classifier labels as "animatediff" get a real motion clip generated
# by a cloud I2V model; everything else stays on the local depth-parallax
# render.
I2V_ENABLED = os.getenv("I2V_ENABLED", "1").strip() in ("1", "true", "True", "yes")
# kwaivgi/kling-v2.1 runs natively on Replicate (~$0.25/5s standard 720p,
# ~$0.49/5s pro 1080p) and is reliable. The previous default
# "wan-video/wan2.6-i2v-flash" proxies to Alibaba's backend and was failing
# deterministically with E006 in a post-generation upstream poll (2026-06).
# The input builder in animatediff_gen.py adapts per model family.
REPLICATE_I2V_MODEL = os.getenv("REPLICATE_I2V_MODEL", "kwaivgi/kling-v2.1")
I2V_DURATION_SECONDS = int(os.getenv("I2V_DURATION_SECONDS", "5"))  # kling: 5 or 10
# "720p"/"1080p" — for kling this maps to mode standard/pro; for wan it is the
# resolution param directly.
I2V_RESOLUTION = os.getenv("I2V_RESOLUTION", "720p")
I2V_OUTPUT_FPS = int(os.getenv("I2V_OUTPUT_FPS", "16"))  # frames extracted from MP4
# Budget guard — clip count cap per project to avoid runaway spend.
I2V_MAX_CLIPS_PER_PROJECT = int(os.getenv("I2V_MAX_CLIPS_PER_PROJECT", "36"))
# At most this many I2V clips per scene (the most dramatic classification wins);
# the other images in the scene use the free local depth-parallax motion.
I2V_MAX_CLIPS_PER_SCENE = int(os.getenv("I2V_MAX_CLIPS_PER_SCENE", "1"))
I2V_TIMEOUT_SECONDS = float(os.getenv("I2V_TIMEOUT_SECONDS", "600.0"))
# Wan 2.6 I2V generates an audio track by default (audio_enabled=true), which
# runs as a post-generation step over HTTP and has been observed to fail with
# E006 ("input was invalid"). We overlay our own narration/music downstream and
# only extract video frames, so audio is off by default. Prompt expansion (an
# LLM rewrite of the motion prompt) stays on — our prompts benefit from it and
# it runs before generation.
I2V_AUDIO_ENABLED = os.getenv("I2V_AUDIO_ENABLED", "0").strip() in ("1", "true", "True", "yes")
I2V_PROMPT_EXPANSION = os.getenv("I2V_PROMPT_EXPANSION", "1").strip() in ("1", "true", "True", "yes")
# Keep the downloaded source.mp4 on disk when frame extraction yields 0 frames,
# so failures can be inspected instead of vanishing. Set to "1" to always delete.
I2V_DELETE_SOURCE_MP4 = os.getenv("I2V_DELETE_SOURCE_MP4", "0").strip() in ("1", "true", "True", "yes")
# Frame loading still uses 8 fps for the assembly ping-pong cadence;
# `_animatediff_clip` in video_assembly.py reads this.
ANIMATEDIFF_DEFAULT_FPS = I2V_OUTPUT_FPS

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
