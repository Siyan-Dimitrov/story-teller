"""Skill bundles — reusable Style x Tone x Voice x Music presets.

Inspired by multica's skill system, but slimmed: a skill is a JSON file with
the fields the Producer normally takes per-run, applied as defaults that
explicit run-config overrides on a per-call basis.

Skills live in backend/agents/skills/<id>.json and are loaded once at import
time. Call reload_skills() during dev if you edit a file without restart.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


SKILLS_DIR = Path(__file__).resolve().parent / "skills"


@dataclass
class Skill:
    """A reusable preset bundle. Most fields are optional — anything left None
    means 'don't override the caller's value'."""

    id: str
    name: str
    description: str = ""
    tone: Optional[str] = None
    image_backend: Optional[str] = None
    style_id: Optional[str] = None
    style_prompt: Optional[str] = None
    lora_keys: Optional[list[str]] = None
    voice_profile_id: Optional[str] = None
    voice_instruct: Optional[str] = None
    music_query: Optional[str] = None
    script_prompt_addendum: Optional[str] = None
    script_backend: Optional[str] = None  # "ollama" (default) or "claude_code"
    target_minutes: Optional[float] = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


_SKILLS: dict[str, Skill] = {}


def _load_one(path: Path) -> Optional[Skill]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        log.warning(f"Skipping malformed skill {path.name}: {e}")
        return None
    skill_id = raw.get("id") or path.stem
    name = raw.get("name") or skill_id
    known = {f for f in Skill.__dataclass_fields__.keys() if f not in ("extra",)}
    extra = {k: v for k, v in raw.items() if k not in known}
    fields = {k: v for k, v in raw.items() if k in known and k != "id" and k != "name"}
    return Skill(id=skill_id, name=name, extra=extra, **fields)


def reload_skills() -> dict[str, Skill]:
    _SKILLS.clear()
    if not SKILLS_DIR.exists():
        return _SKILLS
    for path in sorted(SKILLS_DIR.glob("*.json")):
        skill = _load_one(path)
        if skill:
            _SKILLS[skill.id] = skill
    log.info(f"Loaded {len(_SKILLS)} skills: {sorted(_SKILLS)}")
    return _SKILLS


def list_skills() -> list[Skill]:
    if not _SKILLS:
        reload_skills()
    return sorted(_SKILLS.values(), key=lambda s: s.name.lower())


def get_skill(skill_id: str) -> Optional[Skill]:
    if not _SKILLS:
        reload_skills()
    return _SKILLS.get(skill_id)


# Fields the Producer's cfg accepts that a Skill can pre-fill.
_CFG_FIELDS = (
    "voice_profile_id",
    "voice_instruct",
    "image_backend",
    "style_id",
    "style_prompt",
    "lora_keys",
    "script_backend",
)


def apply_skill_to_cfg(skill: Skill, cfg: dict) -> dict:
    """Return a new cfg with skill fields applied as defaults.

    Explicit cfg values take precedence — a skill never overrides a value the
    caller deliberately passed. The script_prompt_addendum is preserved on cfg
    for the Producer to inject into custom_prompt at script-gen time.
    """
    merged = dict(cfg)
    for fld in _CFG_FIELDS:
        if merged.get(fld) in (None, "", []):
            value = getattr(skill, fld, None)
            if value not in (None, "", []):
                merged[fld] = value
    if skill.script_prompt_addendum and not merged.get("script_prompt_addendum"):
        merged["script_prompt_addendum"] = skill.script_prompt_addendum
    if skill.music_query and not merged.get("music_query"):
        merged["music_query"] = skill.music_query
    if skill.tone and not merged.get("tone"):
        merged["tone"] = skill.tone
    merged.setdefault("_applied_skill_id", skill.id)
    return merged
