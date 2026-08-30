"""Pydantic models for Story Teller API."""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional


# ── Cast (character bible for consistency) ───────────────────

class CastMember(BaseModel):
    """A recurring character whose look is held consistent across scenes.

    The ``reference_image_path`` (a canonical portrait generated once) is fed
    back into every scene this character appears in, so reference-image models
    (Nano Banana / Flux Kontext) keep the character identical.
    """
    id: str = ""              # stable slug, e.g. "gretel"
    name: str = ""            # display name
    role: str = ""            # e.g. "protagonist", "antagonist", "minor"
    description: str = ""     # canonical appearance — age, hair, clothing, build, one signature detail
    reference_prompt: str = ""  # prompt used to render the canonical portrait
    reference_image_path: Optional[str] = None  # relative path once generated


# ── Scene ────────────────────────────────────────────────────

class Scene(BaseModel):
    index: int = 0
    narration: str = ""
    image_prompt: str = ""
    image_prompts: list[str] = Field(default_factory=list)  # Multiple prompts per scene
    motion_prompt: str = ""  # reels: the single I2V action for this beat
    characters: list[str] = Field(default_factory=list)  # cast ids appearing in this scene
    mood: str = "neutral"
    duration_hint: float = 10.0
    # Populated after generation
    audio_path: Optional[str] = None
    audio_duration: Optional[float] = None
    image_path: Optional[str] = None  # First image (backward compat)
    image_paths: list[str] = Field(default_factory=list)  # All images for this scene
    kb_effect: str = "zoom_in"  # Ken Burns effect type (legacy fallback)
    # Animation fields (populated by /animate step)
    animation_types: list[str] = Field(default_factory=list)  # per-image: "depthflow", "portrait", or "animatediff"
    motion_presets: list[str] = Field(default_factory=list)  # per-image motion preset name
    depth_map_paths: list[str] = Field(default_factory=list)  # per-image depth map file paths
    animatediff_clip_paths: list[str] = Field(default_factory=list)  # per-image animated clip output dirs
    animatediff_errors: list[Optional[str]] = Field(default_factory=list)  # per-image I2V failure reason (None if ok)
    # Per-scene music (optional override of global music track)
    music_track: Optional[str] = None  # filename in data/music/ or absolute path
    music_volume: Optional[float] = None  # 0.0-1.0 override


# ── Script ───────────────────────────────────────────────────

class Script(BaseModel):
    title: str = ""
    synopsis: str = ""
    visual_style: str = ""  # per-story art direction, derived from the story itself
    hook: str = ""  # reels: on-screen headline
    cta: str = ""  # reels: closing on-screen question
    cast: list[CastMember] = Field(default_factory=list)  # recurring characters
    scenes: list[Scene] = Field(default_factory=list)
    target_minutes: float = 5.0
    source_tale: str = ""
    tone: str = "dark, atmospheric, gothic"


# ── Project state ────────────────────────────────────────────

class ProjectState(BaseModel):
    project_id: str = ""
    step: str = "created"  # created | scripted | voiced | illustrated | animated | assembled
    kind: str = "story"  # story | reel (reel: scripted | voiced | building_reel | reel_assembled)
    target_seconds: Optional[float] = None  # reels only
    reel: Optional[dict] = None  # {"path", "duration"} once a reel is rendered
    error: Optional[str] = None
    title: str = ""
    source_tale: str = ""
    voice_profile_id: Optional[str] = None
    voice_language: str = "en"
    claude_model: Optional[str] = None  # base Claude model (used as default for all 3 roles)
    pipeline_writer_model: Optional[str] = None   # override claude_model for the writer pass
    pipeline_critic_model: Optional[str] = None   # override claude_model for the critic pass
    pipeline_reviser_model: Optional[str] = None  # override claude_model for the reviser pass
    image_backend: str = "comfyui"  # comfyui | replicate | gpt_image
    project_seed: Optional[int] = None
    target_minutes: float = 5.0
    suggested_length: Optional[str] = None  # e.g., "5 min", "short story", "flash fiction"
    created_at: str = ""
    # Batch chapter fields
    book_group_id: Optional[str] = None
    chapter_index: Optional[int] = None
    book_title: Optional[str] = None
    # Music preferences (persisted after assembly)
    music_track: Optional[str] = None
    music_volume: Optional[float] = None


# ── API requests ─────────────────────────────────────────────

class CreateProjectRequest(BaseModel):
    source_tale: str = ""
    custom_prompt: str = ""  # for kind="reel": the recipe text
    kind: str = "story"  # story | reel
    target_seconds: Optional[float] = None  # reels only
    target_minutes: float = 5.0
    claude_model: Optional[str] = None  # base Claude model (used as default for all 3 roles)
    pipeline_writer_model: Optional[str] = None
    pipeline_critic_model: Optional[str] = None
    pipeline_reviser_model: Optional[str] = None
    tone: str = ""  # e.g. "dark", "humorous", "gothic noir"


class RunScriptRequest(BaseModel):
    claude_model: Optional[str] = None
    pipeline_writer_model: Optional[str] = None
    pipeline_critic_model: Optional[str] = None
    pipeline_reviser_model: Optional[str] = None
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
    backend: str = "comfyui"  # comfyui | replicate | gpt_image
    style_id: Optional[str] = None
    custom_style_prompt: Optional[str] = None
    style_prompt: Optional[str] = None  # legacy clients can still send a full prompt
    lora_keys: Optional[list[str]] = None  # e.g. ["tim_burton", "dark_fantasy"] - None uses defaults for backend
    character_consistency: bool = False
    # For Replicate: Uses FLUX LoRA URLs from config.FLUX_LORA_URLS
    # For ComfyUI: Uses local .safetensors files from AVAILABLE_LORAS


class RegenerateSceneImagesRequest(BaseModel):
    backend: str = "comfyui"
    style_id: Optional[str] = None
    custom_style_prompt: Optional[str] = None
    style_prompt: Optional[str] = None
    lora_keys: Optional[list[str]] = None
    character_consistency: bool = False


class RepairImagesRequest(BaseModel):
    """Verify all scene images and regenerate only the failed/missing ones."""
    backend: str = "nano_banana"
    style_id: Optional[str] = None
    custom_style_prompt: Optional[str] = None
    style_prompt: Optional[str] = None
    lora_keys: Optional[list[str]] = None
    character_consistency: bool = True
    check_duplicates: bool = False  # opt-in Claude-vision duplicate screening


# ── Cast / character references ──────────────────────────────

class GenerateCastRequest(BaseModel):
    """(Re)derive the character bible from the current script via LLM."""
    overwrite: bool = False  # replace an existing cast instead of keeping it


class GenerateCharacterRefsRequest(BaseModel):
    """Render canonical portrait(s) for cast members so they can be reused as
    reference images across every scene."""
    backend: str = "nano_banana"  # nano_banana | gpt_image | replicate
    style_id: Optional[str] = None
    custom_style_prompt: Optional[str] = None
    style_prompt: Optional[str] = None
    cast_ids: Optional[list[str]] = None  # None = all cast members


class UpdateCastMemberRequest(BaseModel):
    """Manual edit of a single cast member's canonical description/prompt."""
    name: Optional[str] = None
    role: Optional[str] = None
    description: Optional[str] = None
    reference_prompt: Optional[str] = None


# ── Shorts ───────────────────────────────────────────────────

class SuggestShortsRequest(BaseModel):
    """Ask the LLM 'shorts director' to score scenes for standalone potential."""
    count: Optional[int] = None  # how many to suggest (default config.SHORTS_PER_PROJECT)


class RenderShortsRequest(BaseModel):
    """Render vertical shorts for the chosen scene indices.

    ``scene_indices`` None means auto-pick via the director.
    ``hooks`` optionally overrides the on-screen headline per scene index.
    ``source`` overrides config.SHORT_SOURCE: "final" cuts from final.mp4,
    "portrait" recomposes the stills as native 9:16 frames via Nano Banana.
    """
    scene_indices: Optional[list[int]] = None
    count: Optional[int] = None
    hooks: Optional[dict[int, str]] = None
    source: Optional[str] = None


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


# ── Gutenberg search ────────────────────────────────────────

class GutenbergAuthor(BaseModel):
    name: str = ""
    birth_year: Optional[int] = None
    death_year: Optional[int] = None


class GutenbergBookResult(BaseModel):
    gutenberg_id: int
    title: str = ""
    authors: list[GutenbergAuthor] = Field(default_factory=list)
    subjects: list[str] = Field(default_factory=list)
    bookshelves: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    download_count: int = 0
    text_url: Optional[str] = None


class GutenbergSearchRequest(BaseModel):
    query: str = ""
    topic: str = ""
    languages: str = ""  # e.g. "en", "de", "fr" — comma-separated for multiple
    page: int = 1


class GutenbergTextRequest(BaseModel):
    text_url: str
    max_chars: int = 2000  # 0 for full text


class UpdateSceneMusicRequest(BaseModel):
    music_track: Optional[str] = None  # filename in data/music/ or absolute path, null to clear
    music_volume: Optional[float] = None  # 0.0-1.0, null to use global default


class RunAssembleRequest(BaseModel):
    music_track: Optional[str] = None  # filename in data/music/ or absolute path
    music_volume: Optional[float] = None  # 0.0-1.0, defaults to config.MUSIC_DEFAULT_VOLUME


# ── API responses ────────────────────────────────────────────

class HealthStatus(BaseModel):
    claude: bool = False
    comfyui: bool = False
    replicate: bool = False
    openai: bool = False
    ffmpeg: bool = False


class ProjectSummary(BaseModel):
    project_id: str
    title: str
    step: str
    source_tale: str
    created_at: str
    book_group_id: Optional[str] = None
    chapter_index: Optional[int] = None
    tone: str = ""
    target_minutes: float = 5.0
    suggested_length: Optional[str] = None
    estimated_duration: float = 5.0  # Calculated from source text char count
    char_count: int = 0  # Source text character count
    kind: str = "story"


class BulkDeleteRequest(BaseModel):
    project_ids: list[str] = Field(default_factory=list)


class UpdateSettingsRequest(BaseModel):
    tone: Optional[str] = None
    target_minutes: Optional[float] = None
    suggested_length: Optional[str] = None
    music_track: Optional[str] = None
    music_volume: Optional[float] = None
    narration_style: Optional[str] = None  # e.g. "anime", or freeform direction


class SplitProjectRequest(BaseModel):
    parts: int = 1  # Number of parts to split the project into


class IntelligentSplitRequest(BaseModel):
    parts: int = 2  # Number of logical parts to split into


class TextPart(BaseModel):
    part_number: int
    title: str  # A descriptive title for this part
    summary: str  # Brief summary of what happens in this part
    split_after_text: str  # The exact text (last ~100 chars) where the split should happen
    char_count: int  # Approximate character count for this part


class IntelligentSplitResponse(BaseModel):
    parts: list[TextPart] = Field(default_factory=list)
    reasoning: str = ""  # Brief explanation of why these split points were chosen


class VoiceProfile(BaseModel):
    id: str
    name: str
    language: str


# ── Batch chapter analysis ──────────────────────────────────

class AnalyzeChaptersRequest(BaseModel):
    text: str
    book_title: str = ""


class AnalyzedChapter(BaseModel):
    title: str = ""
    text: str = ""
    suggested_tone: str = "dark"
    summary: str = ""
    estimated_duration: float = 5.0
    char_count: int = 0
    parts: int = 1  # Split chapter into N equal parts (1 = no split)


class AnalyzeChaptersResponse(BaseModel):
    book_title: str = ""
    chapters: list[AnalyzedChapter] = Field(default_factory=list)


class BatchCreateRequest(BaseModel):
    book_title: str = ""
    chapters: list[AnalyzedChapter] = Field(default_factory=list)
    voice_profile_id: Optional[str] = None
    voice_language: str = "en"
    image_backend: str = "comfyui"


class BatchCreateResponse(BaseModel):
    book_group_id: str
    project_ids: list[str] = Field(default_factory=list)


class BatchRunRequest(BaseModel):
    steps: list[str] = Field(default_factory=lambda: ["script", "voice", "images", "animate", "assemble"])
    project_ids: Optional[list[str]] = None  # if set, only run these chapters
    voice_profile_id: str = ""
    voice_language: str = "en"
    image_backend: str = "comfyui"
    style_id: Optional[str] = None
    custom_style_prompt: Optional[str] = None
    style_prompt: Optional[str] = None
    lora_keys: Optional[list[str]] = None
    character_consistency: bool = False


class ChapterProgress(BaseModel):
    project_id: str
    chapter_index: int = 0
    title: str = ""
    status: str = "pending"  # pending | running | completed | failed
    current_step: Optional[str] = None
    failed_step: Optional[str] = None
    error: Optional[str] = None


class BatchProgress(BaseModel):
    group_id: str
    total: int = 0
    completed: int = 0
    failed: int = 0
    current_chapter: Optional[int] = None
    current_step: Optional[str] = None
    chapters: list[ChapterProgress] = Field(default_factory=list)
    finished: bool = False
    paused: bool = False
