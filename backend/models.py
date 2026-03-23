"""Pydantic models for Story Teller API."""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional


# ── Scene ────────────────────────────────────────────────────

class Scene(BaseModel):
    index: int = 0
    narration: str = ""
    image_prompt: str = ""
    image_prompts: list[str] = Field(default_factory=list)  # Multiple prompts per scene
    mood: str = "neutral"
    duration_hint: float = 10.0
    # Populated after generation
    audio_path: Optional[str] = None
    audio_duration: Optional[float] = None
    image_path: Optional[str] = None  # First image (backward compat)
    image_paths: list[str] = Field(default_factory=list)  # All images for this scene
    kb_effect: str = "zoom_in"  # Ken Burns effect type (legacy fallback)
    # Animation fields (populated by /animate step)
    # QC fields (populated by /qc step)
    qc_results: list[dict] = Field(default_factory=list)  # per-image QC verdicts
    qc_passed: bool = False  # overall scene QC status
    # Animation fields (populated by /animate step)
    animation_types: list[str] = Field(default_factory=list)  # per-image: "depthflow", "portrait", or "animatediff"
    motion_presets: list[str] = Field(default_factory=list)  # per-image motion preset name
    depth_map_paths: list[str] = Field(default_factory=list)  # per-image depth map file paths
    animatediff_clip_paths: list[str] = Field(default_factory=list)  # per-image AnimateDiff output dirs


# ── Script ───────────────────────────────────────────────────

class Script(BaseModel):
    title: str = ""
    synopsis: str = ""
    scenes: list[Scene] = Field(default_factory=list)
    target_minutes: float = 5.0
    source_tale: str = ""
    tone: str = "dark, atmospheric, gothic"


# ── Project state ────────────────────────────────────────────

class ProjectState(BaseModel):
    project_id: str = ""
    step: str = "created"  # created | scripted | voiced | illustrated | animated | assembled
    error: Optional[str] = None
    title: str = ""
    source_tale: str = ""
    voice_profile_id: Optional[str] = None
    voice_language: str = "en"
    ollama_model: str = "kimi-k2.5:cloud"
    image_backend: str = "comfyui"  # comfyui | ollama
    target_minutes: float = 5.0
    created_at: str = ""


# ── API requests ─────────────────────────────────────────────

class CreateProjectRequest(BaseModel):
    source_tale: str = ""
    custom_prompt: str = ""
    target_minutes: float = 5.0
    ollama_model: str = "kimi-k2.5:cloud"
    tone: str = ""  # e.g. "dark", "humorous", "gothic noir"


class RunScriptRequest(BaseModel):
    ollama_model: Optional[str] = None
    target_minutes: Optional[float] = None
    custom_prompt: str = ""


class UpdateScriptRequest(BaseModel):
    title: str
    synopsis: str
    scenes: list[Scene]


DEFAULT_VOICE_INSTRUCT = (
    "Speak slowly and deliberately like a storyteller narrating a dark fairy tale. "
    "Use a calm, measured pace with dramatic pauses between sentences. "
    "Deep, atmospheric tone."
)


class RunVoiceRequest(BaseModel):
    profile_id: str
    language: str = "en"
    instruct: str = DEFAULT_VOICE_INSTRUCT


class RunImagesRequest(BaseModel):
    backend: str = "comfyui"  # comfyui | ollama
    style_prompt: str = "dark fairy tale illustration, gothic storybook art, atmospheric, detailed, moody lighting"
    lora_keys: Optional[list[str]] = None  # e.g. ["tim_burton", "dark_gothic"] — None uses defaults


class SearchStoriesRequest(BaseModel):
    query: str = ""  # e.g. "revenge", "transformation", "brothers grimm"
    count: int = 6


class StorySearchResult(BaseModel):
    title: str
    author: str
    origin: str  # e.g. "German folklore", "French fairy tale"
    synopsis: str
    themes: list[str] = Field(default_factory=list)
    tone_suggestion: str = "dark"


class RunQCRequest(BaseModel):
    vision_model: Optional[str] = None
    pass_threshold: float = 3.0
    style_prompt: str = "dark fairy tale illustration, gothic storybook art, atmospheric, detailed, moody lighting"
    targets: Optional[list["QCTarget"]] = None  # None = all images


class QCTarget(BaseModel):
    scene_index: int
    image_index: int


class RegenerateQCRequest(BaseModel):
    targets: list[QCTarget] = Field(default_factory=list)
    vision_model: Optional[str] = None
    pass_threshold: float = 3.0
    style_prompt: str = "dark fairy tale illustration, gothic storybook art, atmospheric, detailed, moody lighting"
    lora_keys: Optional[list[str]] = None


class RunAssembleRequest(BaseModel):
    pass


# ── API responses ────────────────────────────────────────────

class HealthStatus(BaseModel):
    ollama: bool = False
    voicebox: bool = False
    comfyui: bool = False
    ffmpeg: bool = False


class ProjectSummary(BaseModel):
    project_id: str
    title: str
    step: str
    source_tale: str
    created_at: str


class VoiceProfile(BaseModel):
    id: str
    name: str
    language: str
