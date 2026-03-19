"""File-based project storage for Story Teller."""

import json
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path

from .config import PROJECTS_DIR

log = logging.getLogger(__name__)


def create_project() -> tuple[str, Path]:
    project_id = uuid.uuid4().hex[:12]
    project_dir = PROJECTS_DIR / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "audio").mkdir(exist_ok=True)
    (project_dir / "images").mkdir(exist_ok=True)

    state = {
        "project_id": project_id,
        "step": "created",
        "error": None,
        "title": "",
        "source_tale": "",
        "voice_profile_id": None,
        "voice_language": "en",
        "ollama_model": "kimi-k2.5:cloud",
        "image_backend": "comfyui",
        "target_minutes": 5.0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(project_dir / "state.json", state)
    return project_id, project_dir


def project_dir(project_id: str) -> Path:
    return PROJECTS_DIR / project_id


def load_state(project_id: str) -> dict:
    path = PROJECTS_DIR / project_id / "state.json"
    if not path.exists():
        raise FileNotFoundError(f"Project {project_id} not found")
    return _read_json(path)


def update_state(project_id: str, **kwargs) -> dict:
    state = load_state(project_id)
    state.update(kwargs)
    _write_json(PROJECTS_DIR / project_id / "state.json", state)
    return state


def save_json(project_id: str, filename: str, data) -> Path:
    path = PROJECTS_DIR / project_id / filename
    _write_json(path, data)
    return path


def load_json(project_id: str, filename: str):
    path = PROJECTS_DIR / project_id / filename
    if not path.exists():
        return None
    return _read_json(path)


def list_projects() -> list[dict]:
    projects = []
    if not PROJECTS_DIR.exists():
        return projects
    for d in PROJECTS_DIR.iterdir():
        if d.is_dir() and (d / "state.json").exists():
            try:
                state = _read_json(d / "state.json")
                projects.append(state)
            except Exception:
                log.warning(f"Skipping corrupt project {d.name}")
    projects.sort(key=lambda p: p.get("created_at", ""), reverse=True)
    return projects


def _write_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))
