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

PROJECTS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
TALES_DIR.mkdir(exist_ok=True)

# ── External services ────────────────────────────────────────
VOICEBOX_URL = os.getenv("VOICEBOX_URL", "http://localhost:17493")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "kimi-k2.5:cloud")
COMFYUI_URL = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")

# ── Video output ─────────────────────────────────────────────
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
VIDEO_FPS = 25
CROSSFADE_DURATION = 0.5  # seconds between scenes

# ── LLM ──────────────────────────────────────────────────────
LLM_TIMEOUT_SECONDS = 300.0
LLM_TEMPERATURE = 0.8
LLM_MAX_TOKENS = 16000

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
