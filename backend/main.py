"""Story Teller — FastAPI backend."""

import asyncio
import logging
import shutil
import sys
import traceback
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import httpx

from . import config
from . import project_store as store
from . import script_gen, voice_gen, image_gen
from .video_assembly import assemble_video
from .grimm_tales import list_tales, get_tale
from .models import (
    CreateProjectRequest,
    RunScriptRequest,
    UpdateScriptRequest,
    RunVoiceRequest,
    RunImagesRequest,
    RunAssembleRequest,
    HealthStatus,
    ProjectSummary,
)

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="Story Teller", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5191", "http://127.0.0.1:5191"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ───────────────────────────────────────────────────

@app.get("/api/health")
async def health() -> HealthStatus:
    status = HealthStatus()
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            r = await client.get(f"{config.OLLAMA_URL}/api/tags")
            status.ollama = r.status_code == 200
        except Exception:
            pass
        try:
            r = await client.get(f"{config.VOICEBOX_URL}/health")
            status.voicebox = r.status_code == 200
        except Exception:
            pass
        try:
            r = await client.get(f"{config.COMFYUI_URL}/system_stats")
            status.comfyui = r.status_code == 200
        except Exception:
            pass
    status.ffmpeg = shutil.which("ffmpeg") is not None or Path(config.FFMPEG_PATH).exists()
    return status


# ── Tales catalog ────────────────────────────────────────────

@app.get("/api/tales")
async def get_tales():
    return list_tales()


@app.get("/api/tales/{tale_id}")
async def get_tale_detail(tale_id: str):
    tale = get_tale(tale_id)
    if not tale:
        raise HTTPException(404, "Tale not found")
    return tale


# ── Voice profiles (proxy to VoiceBox) ──────────────────────

@app.get("/api/profiles")
async def get_profiles():
    try:
        return await voice_gen.list_profiles()
    except Exception as e:
        log.error(f"Failed to fetch voice profiles: {e}")
        return []


# ── LoRAs ────────────────────────────────────────────────────

@app.get("/api/loras")
async def get_loras():
    """List available LoRA styles for image generation."""
    return {
        "available": {
            key: {"trigger": v["trigger"], "file": v["file"]}
            for key, v in image_gen.AVAILABLE_LORAS.items()
        },
        "defaults": image_gen.DEFAULT_LORAS,
    }


# ── Projects CRUD ────────────────────────────────────────────

@app.get("/api/projects")
async def list_projects():
    projects = store.list_projects()
    return [
        ProjectSummary(
            project_id=p["project_id"],
            title=p.get("title", ""),
            step=p.get("step", "created"),
            source_tale=p.get("source_tale", ""),
            created_at=p.get("created_at", ""),
        )
        for p in projects
    ]


@app.post("/api/projects")
async def create_project(req: CreateProjectRequest):
    pid, pdir = store.create_project()
    store.update_state(
        pid,
        source_tale=req.source_tale,
        ollama_model=req.ollama_model,
        target_minutes=req.target_minutes,
    )
    if req.source_tale:
        tale = get_tale(req.source_tale)
        if tale:
            store.update_state(pid, title=tale["title"])
    return store.load_state(pid)


@app.get("/api/projects/{project_id}")
async def get_project(project_id: str):
    try:
        state = store.load_state(project_id)
    except FileNotFoundError:
        raise HTTPException(404, "Project not found")

    # Attach sub-data if available
    script = store.load_json(project_id, "script.json")
    if script:
        state["script"] = script
    return state


@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    pdir = store.project_dir(project_id)
    if not pdir.exists():
        raise HTTPException(404, "Project not found")
    shutil.rmtree(pdir)
    return {"deleted": project_id}


# ── Stage 1: Script generation ───────────────────────────────

@app.post("/api/projects/{project_id}/script")
async def run_script(project_id: str, req: RunScriptRequest):
    state = store.load_state(project_id)
    store.update_state(project_id, step="generating_script", error=None)

    try:
        script = await script_gen.generate_script(
            source_tale=state.get("source_tale", ""),
            custom_prompt=req.custom_prompt,
            target_minutes=req.target_minutes or state.get("target_minutes", 5.0),
            ollama_model=req.ollama_model or state.get("ollama_model"),
        )
        store.save_json(project_id, "script.json", script)
        store.update_state(
            project_id,
            step="scripted",
            title=script.get("title", state.get("title", "")),
        )
        return script
    except Exception as e:
        tb = traceback.format_exc()
        log.error(f"Script generation failed: {tb}")
        store.update_state(project_id, step="created", error=f"{e}\n{tb}")
        raise HTTPException(500, str(e))


@app.put("/api/projects/{project_id}/script")
async def update_script(project_id: str, req: UpdateScriptRequest):
    store.load_state(project_id)  # Verify exists
    script = {
        "title": req.title,
        "synopsis": req.synopsis,
        "scenes": [s.model_dump() for s in req.scenes],
    }
    store.save_json(project_id, "script.json", script)
    store.update_state(project_id, step="scripted", title=req.title)
    return script


# ── Stage 2: Voice generation ────────────────────────────────

@app.post("/api/projects/{project_id}/voice")
async def run_voice(project_id: str, req: RunVoiceRequest):
    state = store.load_state(project_id)
    script = store.load_json(project_id, "script.json")
    if not script:
        raise HTTPException(400, "No script found — generate or upload a script first")

    store.update_state(
        project_id,
        step="generating_voice",
        error=None,
        voice_profile_id=req.profile_id,
        voice_language=req.language,
    )

    try:
        pdir = store.project_dir(project_id)
        scenes = await voice_gen.generate_all_scenes(
            scenes=script["scenes"],
            profile_id=req.profile_id,
            language=req.language,
            project_dir=pdir,
        )
        script["scenes"] = scenes
        store.save_json(project_id, "script.json", script)
        store.update_state(project_id, step="voiced")
        return {"scenes": scenes}
    except Exception as e:
        tb = traceback.format_exc()
        log.error(f"Voice generation failed: {tb}")
        store.update_state(project_id, step="scripted", error=f"{e}\n{tb}")
        raise HTTPException(500, str(e))


# ── Stage 3: Image generation ────────────────────────────────

@app.post("/api/projects/{project_id}/images")
async def run_images(project_id: str, req: RunImagesRequest):
    state = store.load_state(project_id)
    script = store.load_json(project_id, "script.json")
    if not script:
        raise HTTPException(400, "No script found")

    store.update_state(
        project_id,
        step="generating_images",
        error=None,
        image_backend=req.backend,
    )

    try:
        pdir = store.project_dir(project_id)
        scenes = await image_gen.generate_all_scenes(
            scenes=script["scenes"],
            project_dir=pdir,
            backend=req.backend,
            style_prompt=req.style_prompt,
            lora_keys=req.lora_keys,
        )
        script["scenes"] = scenes
        store.save_json(project_id, "script.json", script)
        store.update_state(project_id, step="illustrated")
        return {"scenes": scenes}
    except Exception as e:
        tb = traceback.format_exc()
        log.error(f"Image generation failed: {tb}")
        store.update_state(project_id, step="voiced", error=f"{e}\n{tb}")
        raise HTTPException(500, str(e))


# ── Stage 4: Video assembly ──────────────────────────────────

@app.post("/api/projects/{project_id}/assemble")
async def run_assemble(project_id: str, req: RunAssembleRequest):
    state = store.load_state(project_id)
    script = store.load_json(project_id, "script.json")
    if not script:
        raise HTTPException(400, "No script found")

    store.update_state(project_id, step="assembling", error=None)

    try:
        pdir = store.project_dir(project_id)
        output, duration = assemble_video(scenes=script["scenes"], project_dir=pdir)
        store.update_state(project_id, step="assembled")
        return {"video": str(output.relative_to(pdir)), "duration": duration}
    except Exception as e:
        tb = traceback.format_exc()
        log.error(f"Assembly failed: {tb}")
        store.update_state(project_id, step="illustrated", error=f"{e}\n{tb}")
        raise HTTPException(500, str(e))


# ── File serving ─────────────────────────────────────────────

@app.get("/api/projects/{project_id}/download")
async def download_video(project_id: str):
    pdir = store.project_dir(project_id)
    video = pdir / "final.mp4"
    if not video.exists():
        raise HTTPException(404, "Video not found — assemble first")
    return FileResponse(str(video), media_type="video/mp4", filename=f"{project_id}.mp4")


@app.get("/api/projects/{project_id}/artifacts/{filepath:path}")
async def get_artifact(project_id: str, filepath: str):
    pdir = store.project_dir(project_id)
    target = (pdir / filepath).resolve()
    # Security: ensure path is inside project dir
    if not str(target).startswith(str(pdir.resolve())):
        raise HTTPException(403, "Access denied")
    if not target.exists():
        raise HTTPException(404, "File not found")

    suffix = target.suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".mp4": "video/mp4",
    }
    return FileResponse(str(target), media_type=media_types.get(suffix, "application/octet-stream"))


# ── Run ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8102)
