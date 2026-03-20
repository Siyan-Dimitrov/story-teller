"""Pydantic models for Story Teller API."""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional


# ── Scene ────────────────────────────────────────────────────

class Scene(BaseModel):
    index: int = 0
    narration: str = ""
    image_prompt: str = ""
    mood: str = "neutral"
    duration_hint: float = 10.0
    # Populated after generation
    audio_path: Optional[str] = None
    audio_duration: Optional[float] = None
    image_path: Optional[str] = None
    kb_effect: str = "zoom_in"  # Ken Burns effect type


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
    step: str = "created"  # created | scripted | voiced | illustrated | assembled
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


class RunScriptRequest(BaseModel):
    ollama_model: Optional[str] = None
    target_minutes: Optional[float] = None
    custom_prompt: str = ""


class UpdateScriptRequest(BaseModel):
    title: str
    synopsis: str
    scenes: list[Scene]


class RunVoiceRequest(BaseModel):
    profile_id: str
    language: str = "en"


class RunImagesRequest(BaseModel):
    backend: str = "comfyui"  # comfyui | ollama
    style_prompt: str = "dark fairy tale illustration, gothic storybook art, atmospheric, detailed, moody lighting"
    lora_keys: Optional[list[str]] = None  # e.g. ["tim_burton", "dark_gothic"] — None uses defaults


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
