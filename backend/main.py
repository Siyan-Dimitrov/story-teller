"""Story Teller — FastAPI backend."""

import asyncio
import json
import logging
import re
import shutil
import sys
import threading
import traceback
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import httpx

from . import config
from . import project_store as store
from . import script_gen, voice_gen, image_gen, gutenberg, batch, music_search
from .video_assembly import assemble_video, get_assembly_progress, cancel_assembly
from .animation import prepare_animations, get_animation_progress
from .grimm_tales import list_tales, get_tale
from .export import export_project, generate_youtube_metadata
from .models import (
    CreateProjectRequest,
    RunScriptRequest,
    UpdateScriptRequest,
    RunVoiceRequest,
    RunImagesRequest,
    RegenerateSceneImagesRequest,
    RunQCRequest,
    RegenerateQCRequest,
    RunAssembleRequest,
    SearchStoriesRequest,
    GutenbergSearchRequest,
    GutenbergTextRequest,
    HealthStatus,
    ProjectSummary,
    AnalyzeChaptersRequest,
    BatchCreateRequest,
    BatchRunRequest,
    UpdateSettingsRequest,
    BulkDeleteRequest,
    SplitProjectRequest,
    IntelligentSplitRequest,
    IntelligentSplitResponse,
    TextPart,
    UpdateSceneMusicRequest,
)
from .image_qc import run_qc_for_project, regenerate_and_evaluate, get_qc_progress, evaluate_single_image

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="Story Teller", version="0.1.0")
app.mount("/music", StaticFiles(directory=str(config.MUSIC_DIR)), name="music")
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
    status.replicate = bool(config.REPLICATE_API_TOKEN)
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


# ── Story search ─────────────────────────────────────────────

@app.post("/api/search-stories")
async def search_stories(req: SearchStoriesRequest):
    """Search for well-known stories using the LLM."""
    try:
        results = await script_gen.search_stories(
            query=req.query,
            count=req.count,
            ollama_model=req.ollama_model,
        )
        return {"results": results}
    except Exception as e:
        log.error(f"Story search failed: {e}")
        raise HTTPException(500, f"Story search failed: {e}")


# ── Gutenberg online search ──────────────────────────────────

@app.post("/api/gutenberg/search")
async def gutenberg_search(req: GutenbergSearchRequest):
    """Search Project Gutenberg for public domain books."""
    try:
        results = await gutenberg.search_gutenberg(
            query=req.query,
            page=req.page,
            topic=req.topic,
            languages=req.languages,
        )
        return results
    except Exception as e:
        log.error(f"Gutenberg search failed: {e}")
        raise HTTPException(502, f"Gutenberg search failed: {e}")


@app.post("/api/gutenberg/text")
async def gutenberg_text(req: GutenbergTextRequest):
    """Fetch plain text of a Gutenberg book."""
    try:
        result = await gutenberg.fetch_gutenberg_text(
            text_url=req.text_url,
            max_chars=req.max_chars,
        )
        return result
    except Exception as e:
        log.error(f"Gutenberg text fetch failed: {e}")
        raise HTTPException(502, f"Gutenberg text fetch failed: {e}")


# ── Batch chapter analysis ───────────────────────────────────

@app.post("/api/analyze-chapters")
async def analyze_chapters(req: AnalyzeChaptersRequest):
    """Analyze book text and detect chapters via LLM."""
    if not req.text.strip():
        raise HTTPException(400, "No text provided")
    try:
        result = await batch.analyze_chapters(
            text=req.text,
            book_title=req.book_title,
            ollama_model=req.ollama_model,
        )
        return result
    except Exception as e:
        log.error(f"Chapter analysis failed: {e}")
        raise HTTPException(500, f"Chapter analysis failed: {e}")


@app.post("/api/batch/create")
async def batch_create(req: BatchCreateRequest):
    """Create one project per chapter, linked by a book group ID."""
    if not req.chapters:
        raise HTTPException(400, "No chapters provided")
    try:
        group_id, project_ids = batch.create_batch_projects(
            book_title=req.book_title,
            chapters=[ch.model_dump() for ch in req.chapters],
            ollama_model=req.ollama_model,
            voice_profile_id=req.voice_profile_id,
            voice_language=req.voice_language,
            image_backend=req.image_backend,
        )
        return {"book_group_id": group_id, "project_ids": project_ids}
    except Exception as e:
        log.error(f"Batch creation failed: {e}")
        raise HTTPException(500, f"Batch creation failed: {e}")


@app.post("/api/batch/{group_id}/run")
async def batch_run(group_id: str, req: BatchRunRequest):
    """Start sequential pipeline processing for a batch group."""
    # Find all projects in this group
    projects = store.list_projects()
    group_projects = [
        p for p in projects
        if p.get("book_group_id") == group_id
    ]
    if not group_projects:
        raise HTTPException(404, f"No projects found for group {group_id}")

    group_projects.sort(key=lambda p: p.get("chapter_index", 0))

    # If specific chapters requested, filter to those
    if req.project_ids:
        allowed = set(req.project_ids)
        group_projects = [p for p in group_projects if p["project_id"] in allowed]
        if not group_projects:
            raise HTTPException(400, "None of the requested project_ids belong to this group")

    project_ids = [p["project_id"] for p in group_projects]

    # Check if already running
    existing = batch.get_batch_progress(group_id)
    if not existing.get("finished", True):
        return {"status": "already_running"}

    # Launch in background thread
    def _run():
        import asyncio as _asyncio
        loop = _asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                batch.run_batch_pipeline(
                    group_id=group_id,
                    project_ids=project_ids,
                    steps=req.steps,
                    voice_profile_id=req.voice_profile_id,
                    voice_language=req.voice_language,
                    voice_instruct=req.voice_instruct,
                    image_backend=req.image_backend,
                    style_prompt=req.style_prompt,
                    lora_keys=req.lora_keys,
                )
            )
        except Exception as e:
            log.error(f"Batch pipeline error: {e}")
        finally:
            loop.close()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return {"status": "running"}


@app.get("/api/batch/{group_id}/progress")
async def batch_progress(group_id: str):
    """Get batch processing progress."""
    return batch.get_batch_progress(group_id)


@app.post("/api/batch/{group_id}/pause")
async def batch_pause(group_id: str):
    """Pause a running batch pipeline after the current chapter completes."""
    ok = batch.pause_batch(group_id)
    if not ok:
        raise HTTPException(400, "Batch is not currently running")
    return {"status": "pausing"}


@app.post("/api/batch/{group_id}/resume")
async def batch_resume(group_id: str):
    """Resume a paused batch pipeline from where it left off."""
    run_config = batch.resume_batch(group_id)
    if not run_config:
        raise HTTPException(400, "No stored config found for this batch")

    project_ids = run_config["project_ids"]

    def _run():
        import asyncio as _asyncio
        loop = _asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                batch.run_batch_pipeline(
                    group_id=group_id,
                    project_ids=project_ids,
                    steps=run_config["steps"],
                    voice_profile_id=run_config["voice_profile_id"],
                    voice_language=run_config["voice_language"],
                    voice_instruct=run_config["voice_instruct"],
                    image_backend=run_config["image_backend"],
                    style_prompt=run_config["style_prompt"],
                    lora_keys=run_config["lora_keys"],
                )
            )
        except Exception as e:
            log.error(f"Batch resume error: {e}")
        finally:
            loop.close()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return {"status": "resumed"}


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
            key: {
                "trigger": v["trigger"],
                "file": v["file"],
                "has_flux": v.get("flux_lora_key") is not None
                    and (v["flux_lora_key"] in config.FLUX_LORA_URLS
                         or v["flux_lora_key"] in config.FLUX_LORA_ALTERNATIVES),
            }
            for key, v in image_gen.AVAILABLE_LORAS.items()
        },
        "defaults": image_gen.DEFAULT_LORAS,
    }


@app.get("/api/music")
async def list_music():
    """List available background music tracks in data/music/."""
    return {
        "available": music_search.find_local_music(),
        "default_volume": config.MUSIC_DEFAULT_VOLUME,
        "music_dir": str(config.MUSIC_DIR),
        "jamendo_enabled": bool(config.JAMENDO_CLIENT_ID),
    }


@app.get("/api/music/search")
async def search_music(query: str = "cinematic", limit: int = 8):
    """Search Jamendo for royalty-free instrumental music matching a mood/query."""
    if not config.JAMENDO_CLIENT_ID:
        raise HTTPException(
            status_code=503,
            detail="Jamendo not configured. Set JAMENDO_CLIENT_ID in the environment.",
        )
    tracks = await music_search.search_jamendo(query, limit=limit)
    return {"query": query, "results": tracks}


@app.post("/api/music/download")
async def download_music(url: str):
    """Download a remote music URL into data/music/ cache and return the local filename."""
    path = music_search.download_music_to_cache(url)
    if path is None:
        raise HTTPException(status_code=400, detail="Could not download music from that URL.")
    return {"name": path.name, "path": str(path), "size_bytes": path.stat().st_size}


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
            book_group_id=p.get("book_group_id"),
            chapter_index=p.get("chapter_index"),
            tone=p.get("tone", ""),
            target_minutes=p.get("target_minutes", 5.0),
            suggested_length=p.get("suggested_length"),
            **get_text_stats(p),
        )
        for p in projects
    ]


def get_text_stats(project: dict) -> dict:
    """Get character count and estimated duration from project source text."""
    # Try custom_prompt first, then source_tale lookup
    text = project.get("custom_prompt", "")
    if not text:
        source = project.get("source_tale", "")
        if source:
            from .grimm_tales import get_tale
            tale = get_tale(source)
            text = tale.get("full_text", "") if tale else ""
    # Also check for batch chapter source_text.txt
    if not text:
        try:
            pdir = store.project_dir(project["project_id"])
            source_file = pdir / "source_text.txt"
            if source_file.exists():
                text = source_file.read_text(encoding="utf-8")
        except Exception:
            pass
    char_count = len(text)
    minutes = char_count / config.BATCH_NARRATION_RATE
    return {
        "char_count": char_count,
        "estimated_duration": max(1.0, round(minutes, 1)),
    }


@app.post("/api/projects")
async def create_project(req: CreateProjectRequest):
    pid, pdir = store.create_project()
    store.update_state(
        pid,
        source_tale=req.source_tale,
        ollama_model=req.ollama_model,
        target_minutes=req.target_minutes,
        tone=req.tone,
        custom_prompt=req.custom_prompt,
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
    # Add calculated text stats
    stats = get_text_stats(state)
    state["char_count"] = stats["char_count"]
    state["estimated_duration"] = stats["estimated_duration"]
    return state


@app.post("/api/projects/bulk-delete")
async def bulk_delete_projects(req: BulkDeleteRequest):
    """Delete multiple projects at once."""
    deleted = []
    not_found = []
    for pid in req.project_ids:
        pdir = store.project_dir(pid)
        if pdir.exists():
            shutil.rmtree(pdir)
            deleted.append(pid)
        else:
            not_found.append(pid)
    return {"deleted": deleted, "not_found": not_found}


@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    pdir = store.project_dir(project_id)
    if not pdir.exists():
        raise HTTPException(404, "Project not found")
    shutil.rmtree(pdir)
    return {"deleted": project_id}


@app.delete("/api/book-group/{group_id}")
async def delete_book_group(group_id: str):
    """Delete all projects belonging to a book group."""
    all_projects = store.list_projects()
    group_projects = [p for p in all_projects if p.get("book_group_id") == group_id]
    if not group_projects:
        raise HTTPException(404, "No projects found for this book group")
    deleted = []
    for p in group_projects:
        pdir = store.project_dir(p["project_id"])
        if pdir.exists():
            shutil.rmtree(pdir)
            deleted.append(p["project_id"])
    return {"deleted": deleted, "group_id": group_id}


@app.post("/api/projects/{project_id}/duplicate")
async def duplicate_project(project_id: str):
    try:
        source_state = store.load_state(project_id)
    except FileNotFoundError:
        raise HTTPException(404, "Project not found")

    new_id, new_dir = store.create_project()
    # Copy settings from source
    copy_fields = [
        "source_tale", "tone", "target_minutes", "ollama_model",
        "voice_language", "image_backend", "suggested_length",
        "title", "music_track", "music_volume",
    ]
    updates = {k: source_state.get(k) for k in copy_fields if source_state.get(k) is not None}
    if updates:
        store.update_state(new_id, **updates)

    # Copy script if it exists
    script = store.load_json(project_id, "script.json")
    if script:
        store.save_json(new_id, "script.json", script)
        store.update_state(new_id, step="scripted")

    return store.load_state(new_id)


@app.put("/api/projects/{project_id}/settings")
async def update_settings(project_id: str, req: UpdateSettingsRequest):
    try:
        store.load_state(project_id)
    except FileNotFoundError:
        raise HTTPException(404, "Project not found")
    updates = {}
    if req.tone is not None:
        updates["tone"] = req.tone
    if req.target_minutes is not None:
        updates["target_minutes"] = req.target_minutes
    if req.suggested_length is not None:
        updates["suggested_length"] = req.suggested_length
    if req.music_track is not None:
        updates["music_track"] = req.music_track
    if req.music_volume is not None:
        updates["music_volume"] = req.music_volume
    if not updates:
        raise HTTPException(400, "No settings to update")
    state = store.update_state(project_id, **updates)
    return state


# ── Stage 1: Script generation ───────────────────────────────

@app.post("/api/projects/{project_id}/script")
async def run_script(project_id: str, req: RunScriptRequest):
    state = store.load_state(project_id)
    store.update_state(project_id, step="generating_script", error=None)

    try:
        script = await script_gen.generate_script(
            source_tale=state.get("source_tale", ""),
            custom_prompt=req.custom_prompt or state.get("custom_prompt", ""),
            target_minutes=req.target_minutes or state.get("target_minutes", 5.0),
            ollama_model=req.ollama_model or state.get("ollama_model"),
            tone=state.get("tone", ""),
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
    scenes = []
    for s in req.scenes:
        d = s.model_dump()
        # Normalize: ensure image_prompts is populated from image_prompt if empty
        if not d.get("image_prompts") and d.get("image_prompt"):
            d["image_prompts"] = [d["image_prompt"]]
        # Sync image_prompt backward-compat field with first prompt
        if d.get("image_prompts"):
            d["image_prompt"] = d["image_prompts"][0]
        scenes.append(d)
    script = {
        "title": req.title,
        "synopsis": req.synopsis,
        "scenes": scenes,
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
            instruct=req.instruct,
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


@app.post("/api/projects/{project_id}/images/{scene_index}")
async def regenerate_scene_images(project_id: str, scene_index: int, req: RegenerateSceneImagesRequest):
    state = store.load_state(project_id)
    script = store.load_json(project_id, "script.json")
    if not script:
        raise HTTPException(400, "No script found")

    scenes = script["scenes"]
    if scene_index < 0 or scene_index >= len(scenes):
        raise HTTPException(400, f"Invalid scene index {scene_index}")

    scene = scenes[scene_index]
    pdir = store.project_dir(project_id)

    # Determine reference image for character consistency
    reference_image = None
    if req.character_consistency and req.backend == "replicate":
        # Use first image of first scene as reference if available
        if scenes and scenes[0].get("image_paths"):
            ref_path = pdir / scenes[0]["image_paths"][0]
            if ref_path.exists():
                reference_image = ref_path

    try:
        updated_scene = await image_gen.generate_scene_images(
            scene=scene,
            project_dir=pdir,
            backend=req.backend,
            style_prompt=req.style_prompt,
            lora_keys=req.lora_keys,
            reference_image=reference_image,
        )
        scenes[scene_index] = updated_scene
        script["scenes"] = scenes
        store.save_json(project_id, "script.json", script)
        return {"scene": updated_scene}
    except Exception as e:
        tb = traceback.format_exc()
        log.error(f"Scene image regeneration failed: {tb}")
        raise HTTPException(500, str(e))


# ── Stage 3.5: Image QC ──────────────────────────────────

@app.post("/api/projects/{project_id}/qc")
async def run_qc(project_id: str, req: RunQCRequest):
    state = store.load_state(project_id)
    script = store.load_json(project_id, "script.json")
    if not script:
        raise HTTPException(400, "No script found")

    # Check if already running
    progress = get_qc_progress(project_id)
    if progress["active"]:
        return {"status": "already_running"}

    store.update_state(project_id, step="qc_running", error=None)
    pdir = store.project_dir(project_id)
    vision_model = req.vision_model or config.OLLAMA_VISION_MODEL

    def _run():
        import asyncio as _asyncio
        loop = _asyncio.new_event_loop()
        try:
            targets = [t.model_dump() for t in req.targets] if req.targets else None
            scenes = loop.run_until_complete(
                run_qc_for_project(
                    scenes=script["scenes"],
                    project_dir=pdir,
                    vision_model=vision_model,
                    style_prompt=req.style_prompt,
                    pass_threshold=req.pass_threshold,
                    project_id=project_id,
                    targets=targets,
                )
            )
            script["scenes"] = scenes
            store.save_json(project_id, "script.json", script)
            store.update_state(project_id, step="qc_passed")
        except Exception as e:
            tb = traceback.format_exc()
            log.error(f"QC failed: {tb}")
            store.update_state(project_id, step="illustrated", error=f"{e}\n{tb}")
        finally:
            loop.close()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return {"status": "qc_running"}


@app.get("/api/projects/{project_id}/qc-progress")
async def qc_progress(project_id: str):
    return get_qc_progress(project_id)


@app.post("/api/projects/{project_id}/qc-retry/{scene_index}/{image_index}")
async def qc_retry_image(project_id: str, scene_index: int, image_index: int):
    state = store.load_state(project_id)
    script = store.load_json(project_id, "script.json")
    if not script:
        raise HTTPException(400, "No script found")

    scenes = script["scenes"]
    if scene_index >= len(scenes):
        raise HTTPException(404, "Scene not found")

    pdir = store.project_dir(project_id)
    vision_model = config.OLLAMA_VISION_MODEL

    result = await evaluate_single_image(
        scene=scenes[scene_index],
        image_index=image_index,
        project_dir=pdir,
        vision_model=vision_model,
        style_prompt="dark fairy tale illustration, gothic storybook art, atmospheric, detailed, moody lighting",
    )

    # Update scene QC results
    scene = scenes[scene_index]
    qc_results = scene.get("qc_results", [])
    while len(qc_results) <= image_index:
        qc_results.append({})
    qc_results[image_index] = {
        "image_index": image_index,
        "passed": result.get("average_score", 0) >= config.QC_PASS_THRESHOLD,
        "scores": result.get("scores", {}),
        "average_score": result.get("average_score", 0),
        "reasoning": result.get("reasoning", ""),
        "attempts": 1,
    }
    scene["qc_results"] = qc_results
    scene["qc_passed"] = all(r.get("passed", False) for r in qc_results)
    store.save_json(project_id, "script.json", script)

    return result


@app.post("/api/projects/{project_id}/qc-regenerate")
async def qc_regenerate(project_id: str, req: RegenerateQCRequest):
    """Regenerate selected images and re-evaluate them."""
    state = store.load_state(project_id)
    script = store.load_json(project_id, "script.json")
    if not script:
        raise HTTPException(400, "No script found")

    if not req.targets:
        raise HTTPException(400, "No targets specified")

    progress = get_qc_progress(project_id)
    if progress["active"]:
        return {"status": "already_running"}

    store.update_state(project_id, step="qc_running", error=None)
    pdir = store.project_dir(project_id)
    vision_model = req.vision_model or config.OLLAMA_VISION_MODEL
    image_backend = state.get("image_backend", "comfyui")
    targets = [t.model_dump() for t in req.targets]

    def _run():
        import asyncio as _asyncio
        loop = _asyncio.new_event_loop()
        try:
            scenes = loop.run_until_complete(
                regenerate_and_evaluate(
                    scenes=script["scenes"],
                    project_dir=pdir,
                    targets=targets,
                    vision_model=vision_model,
                    style_prompt=req.style_prompt,
                    image_backend=image_backend,
                    lora_keys=req.lora_keys,
                    pass_threshold=req.pass_threshold,
                    project_id=project_id,
                )
            )
            script["scenes"] = scenes
            store.save_json(project_id, "script.json", script)
            store.update_state(project_id, step="qc_passed")
        except Exception as e:
            tb = traceback.format_exc()
            log.error(f"QC regeneration failed: {tb}")
            store.update_state(project_id, step="illustrated", error=f"{e}\n{tb}")
        finally:
            loop.close()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return {"status": "qc_running"}


# ── Stage 4: Animation preparation ───────────────────────

@app.post("/api/projects/{project_id}/animate")
async def run_animate(project_id: str):
    state = store.load_state(project_id)
    script = store.load_json(project_id, "script.json")
    if not script:
        raise HTTPException(400, "No script found")

    # Check if already animating
    progress = get_animation_progress(project_id)
    if progress["active"]:
        return {"status": "already_animating"}

    store.update_state(project_id, step="animating", error=None)
    pdir = store.project_dir(project_id)
    ollama_model = state.get("ollama_model")

    def _run():
        import asyncio as _asyncio
        loop = _asyncio.new_event_loop()
        try:
            scenes = loop.run_until_complete(
                prepare_animations(
                    scenes=script["scenes"],
                    project_dir=pdir,
                    ollama_model=ollama_model,
                    project_id=project_id,
                )
            )
            script["scenes"] = scenes
            store.save_json(project_id, "script.json", script)
            store.update_state(project_id, step="animated")
        except Exception as e:
            tb = traceback.format_exc()
            log.error(f"Animation prep failed: {tb}")
            fallback = "qc_passed" if state.get("step") in ("animating", "qc_passed") else "illustrated"
            store.update_state(project_id, step=fallback, error=f"{e}\n{tb}")
        finally:
            loop.close()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return {"status": "animating"}


@app.get("/api/projects/{project_id}/animation-progress")
async def animation_progress(project_id: str):
    return get_animation_progress(project_id)


# ── Stage 5: Video assembly ──────────────────────────────────

@app.post("/api/projects/{project_id}/assemble")
async def run_assemble(project_id: str, req: RunAssembleRequest):
    state = store.load_state(project_id)
    script = store.load_json(project_id, "script.json")
    if not script:
        raise HTTPException(400, "No script found")

    # Check if already assembling
    progress = get_assembly_progress(project_id)
    if progress["active"]:
        return {"status": "already_assembling"}

    store.update_state(project_id, step="assembling", error=None,
                        music_track=req.music_track, music_volume=req.music_volume)
    pdir = store.project_dir(project_id)

    def _run():
        try:
            output, duration = assemble_video(
                scenes=script["scenes"],
                project_dir=pdir,
                project_id=project_id,
                music_track=req.music_track,
                music_volume=req.music_volume,
            )
            store.update_state(project_id, step="assembled")

            # Export to output folder with YouTube metadata
            try:
                title = script.get("title", state.get("title", ""))
                synopsis = script.get("synopsis", "")
                tone = state.get("tone", "dark")
                themes = []
                for s in script.get("scenes", []):
                    mood = s.get("mood", "")
                    if mood and mood not in themes:
                        themes.append(mood)

                # Generate YouTube metadata via LLM (run async in new event loop)
                import asyncio as _asyncio
                loop = _asyncio.new_event_loop()
                metadata = loop.run_until_complete(
                    generate_youtube_metadata(
                        title=title,
                        synopsis=synopsis,
                        tone=tone,
                        themes=themes,
                        scene_count=len(script.get("scenes", [])),
                        ollama_model=state.get("ollama_model"),
                        book_title=state.get("book_title", ""),
                    )
                )
                loop.close()

                out_dir = export_project(
                    project_dir=pdir,
                    title=title,
                    project_id=project_id,
                    metadata_text=metadata,
                )
                store.update_state(project_id, output_dir=str(out_dir))
                log.info(f"Export complete: {out_dir}")
            except Exception as ex:
                log.error(f"Export/metadata failed (video still OK): {ex}")
        except Exception as e:
            tb = traceback.format_exc()
            log.error(f"Assembly failed: {tb}")
            # Fall back to animated if depth maps exist, else illustrated
            fallback = "animated" if (pdir / "depth_maps").exists() else "illustrated"
            store.update_state(project_id, step=fallback, error=f"{e}\n{tb}")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return {"status": "assembling"}


@app.get("/api/projects/{project_id}/assembly-progress")
async def assembly_progress(project_id: str):
    return get_assembly_progress(project_id)


@app.post("/api/projects/{project_id}/assembly-cancel")
async def assembly_cancel_endpoint(project_id: str):
    cancelled = cancel_assembly(project_id)
    if cancelled:
        fallback = "animated"  # preserve animation data on cancel
        store.update_state(project_id, step=fallback, error="Assembly cancelled")
    return {"cancelled": cancelled}


# ── Per-scene music suggestion ─────────────────────────────

SUGGEST_MUSIC_PROMPT = """You are a music supervisor for dark fairy tale videos.
Given scene descriptions, suggest a short 2-3 word instrumental music search query for each scene.

EXTREMELY IMPORTANT: Your ENTIRE response must be ONLY a valid JSON array. NO explanations before. NO explanations after. NO markdown code fences with ```. Just raw JSON starting with [ and ending with ].

JSON format: array of objects, one per scene, in order:
[
  {"scene_index": 0, "query": "dark orchestral", "reasoning": "..."},
  ...
]

Rules:
1. Each query should be 2-4 words, suitable for searching instrumental/background music
2. Queries should match the scene mood and atmosphere
3. Output ONLY the JSON array - nothing else
"""


# Mood-to-query fallback mapping
_MOOD_TO_QUERY = {
    "dark": "dark ambient",
    "gothic": "gothic orchestral",
    "tense": "suspense tension",
    "suspenseful": "suspense tension",
    "whimsical": "whimsical playful",
    "peaceful": "peaceful ambient",
    "sad": "melancholic piano",
    "melancholic": "melancholic strings",
    "epic": "epic cinematic",
    "dramatic": "dramatic orchestral",
    "mysterious": "mysterious atmospheric",
    "horror": "horror dark ambient",
    "romantic": "romantic strings",
    "joyful": "joyful uplifting",
    "calm": "calm ambient",
}


def _mood_to_query(mood: str) -> str:
    mood_lower = mood.lower()
    for key, query in _MOOD_TO_QUERY.items():
        if key in mood_lower:
            return query
    return "cinematic background"


@app.post("/api/projects/{project_id}/suggest-music")
async def suggest_music(project_id: str):
    """Suggest per-scene background music by querying Ollama and Jamendo."""
    state = store.load_state(project_id)
    script = store.load_json(project_id, "script.json")
    if not script:
        raise HTTPException(400, "No script found")

    scenes = script.get("scenes", [])
    if not scenes:
        raise HTTPException(400, "No scenes in script")

    ollama_model = state.get("ollama_model", config.OLLAMA_MODEL)
    base_url = config.OLLAMA_URL

    # Build prompt with scene summaries
    scene_descriptions = []
    for i, scene in enumerate(scenes):
        desc = f"Scene {i}: mood={scene.get('mood', 'neutral')}, narration_preview={scene.get('narration', '')[:120]!r}, image_prompt={scene.get('image_prompt', '')[:120]!r}"
        scene_descriptions.append(desc)

    user_message = "Suggest instrumental music queries for these scenes:\n\n" + "\n".join(scene_descriptions)

    queries = []
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{base_url}/api/chat",
                json={
                    "model": ollama_model,
                    "messages": [
                        {"role": "system", "content": SUGGEST_MUSIC_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.5, "num_predict": 2000},
                },
            )
        if resp.status_code == 200:
            data = resp.json()
            content = data.get("message", {}).get("content", "")
            if content:
                # Try to extract JSON array
                import json
                content = content.strip()
                # Remove markdown fences if present
                if content.startswith("```"):
                    content = content.split("\n", 1)[1]
                if content.endswith("```"):
                    content = content.rsplit("\n", 1)[0]
                if content.startswith("json"):
                    content = content[4:].strip()
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, list):
                        queries = parsed
                except json.JSONDecodeError:
                    log.warning(f"Could not parse LLM music suggestion response: {content[:200]}")
    except Exception as e:
        log.warning(f"LLM music suggestion failed: {e}")

    # Fallback to heuristic if LLM failed
    if not queries:
        queries = [
            {"scene_index": i, "query": _mood_to_query(scene.get("mood", "cinematic")), "reasoning": "Fallback from mood heuristic"}
            for i, scene in enumerate(scenes)
        ]

    # Search Jamendo for each unique query
    unique_queries = {q["query"] for q in queries if q.get("query")}
    query_results: dict[str, list[dict]] = {}
    for query in unique_queries:
        try:
            tracks = await music_search.search_jamendo(query, limit=3)
            query_results[query] = tracks
        except Exception as e:
            log.warning(f"Jamendo search failed for '{query}': {e}")
            query_results[query] = []

    # Build response
    result_scenes = []
    for q in queries:
        si = q.get("scene_index", 0)
        query = q.get("query", "cinematic background")
        tracks = query_results.get(query, [])
        result_scenes.append({
            "scene_index": si,
            "query": query,
            "reasoning": q.get("reasoning", ""),
            "tracks": tracks,
        })

    return {"scenes": result_scenes}


@app.put("/api/projects/{project_id}/scenes/{scene_index}/music")
async def update_scene_music(project_id: str, scene_index: int, req: UpdateSceneMusicRequest):
    """Update music track/volume for a single scene."""
    store.load_state(project_id)  # Verify exists
    script = store.load_json(project_id, "script.json")
    if not script:
        raise HTTPException(400, "No script found")

    scenes = script.get("scenes", [])
    if scene_index < 0 or scene_index >= len(scenes):
        raise HTTPException(400, f"Invalid scene index {scene_index}")

    scene = scenes[scene_index]
    if req.music_track is not None:
        if req.music_track == "":
            scene.pop("music_track", None)
        else:
            scene["music_track"] = req.music_track
    if req.music_volume is not None:
        scene["music_volume"] = req.music_volume

    script["scenes"] = scenes
    store.save_json(project_id, "script.json", script)
    return scene


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


# ── Project source text ──────────────────────────────────────

@app.get("/api/projects/{project_id}/source-text")
async def get_project_source_text(project_id: str):
    """Get the saved source text for a project."""
    try:
        state = store.load_state(project_id)
    except FileNotFoundError:
        raise HTTPException(404, "Project not found")

    pdir = store.project_dir(project_id)
    source_file = pdir / "source_text.txt"

    if not source_file.exists():
        # Return empty response if no source text saved
        return {"text": "", "char_count": 0, "project_id": project_id}

    text = source_file.read_text(encoding="utf-8")
    return {
        "text": text,
        "char_count": len(text),
        "project_id": project_id,
        "title": state.get("title", ""),
        "book_group_id": state.get("book_group_id"),
        "chapter_index": state.get("chapter_index"),
    }


@app.post("/api/projects/{project_id}/split")
async def split_project(project_id: str, req: SplitProjectRequest):
    """Split a project's source text into multiple parts, creating new projects.

    The original project is deleted after successful split.
    Returns the new project IDs created.
    """
    if req.parts < 2:
        raise HTTPException(400, "Parts must be at least 2")

    try:
        state = store.load_state(project_id)
    except FileNotFoundError:
        raise HTTPException(404, "Project not found")

    pdir = store.project_dir(project_id)
    source_file = pdir / "source_text.txt"

    if not source_file.exists():
        raise HTTPException(400, "Project has no source text to split")

    full_text = source_file.read_text(encoding="utf-8")
    if not full_text.strip():
        raise HTTPException(400, "Source text is empty")

    # Get project metadata for the new projects
    book_group_id = state.get("book_group_id")
    book_title = state.get("book_title", "")
    chapter_index = state.get("chapter_index", 0)
    base_title = state.get("title", "Untitled")
    ollama_model = state.get("ollama_model", config.OLLAMA_MODEL)
    image_backend = state.get("image_backend", "comfyui")
    tone = state.get("tone", "")
    voice_profile_id = state.get("voice_profile_id")
    voice_language = state.get("voice_language", "en")
    target_minutes = state.get("target_minutes", 5.0)

    # Calculate duration per part
    total_duration = target_minutes
    part_duration = max(1.0, round(total_duration / req.parts, 1))

    # Split the text
    part_texts = batch._split_text_into_parts(full_text, req.parts)

    if len(part_texts) < req.parts:
        raise HTTPException(400, f"Could not split text into {req.parts} meaningful parts")

    # Create new projects for each part
    new_project_ids = []
    for part_idx, part_text in enumerate(part_texts):
        new_pid, new_pdir = store.create_project()

        if req.parts > 1:
            part_title = f"{base_title} — Part {part_idx + 1}/{req.parts}"
        else:
            part_title = base_title

        store.update_state(
            new_pid,
            title=part_title,
            book_group_id=book_group_id,
            book_title=book_title,
            chapter_index=chapter_index + part_idx,
            ollama_model=ollama_model,
            target_minutes=part_duration,
            tone=tone,
            custom_prompt=part_text,
            image_backend=image_backend,
            voice_profile_id=voice_profile_id,
            voice_language=voice_language,
        )

        if part_text:
            (new_pdir / "source_text.txt").write_text(part_text, encoding="utf-8")

        new_project_ids.append(new_pid)
        log.info(f"Created split project {new_pid}: '{part_title}' from {project_id}")

    # Delete the original project after successful split
    shutil.rmtree(pdir)
    log.info(f"Deleted original project {project_id} after splitting into {len(new_project_ids)} parts")

    return {
        "original_project_id": project_id,
        "new_project_ids": new_project_ids,
        "parts": len(new_project_ids),
    }


# ── Helper for regular split (used as fallback) ─────────────────

async def _do_regular_split(project_id: str, full_text: str, state: dict, num_parts: int) -> dict:
    """Perform a regular character-based split. Used as fallback for intelligent split."""
    book_group_id = state.get("book_group_id")
    book_title = state.get("book_title", "")
    chapter_index = state.get("chapter_index", 0)
    base_title = state.get("title", "Untitled")
    ollama_model = state.get("ollama_model", config.OLLAMA_MODEL)
    image_backend = state.get("image_backend", "comfyui")
    tone = state.get("tone", "")
    voice_profile_id = state.get("voice_profile_id")
    voice_language = state.get("voice_language", "en")
    target_minutes = state.get("target_minutes", 5.0)

    # Calculate duration per part
    part_duration = max(1.0, round(target_minutes / num_parts, 1))

    # Split the text
    part_texts = batch._split_text_into_parts(full_text, num_parts)

    if len(part_texts) < num_parts:
        raise HTTPException(400, f"Could not split text into {num_parts} meaningful parts")

    # Create new projects for each part
    pdir = store.project_dir(project_id)
    new_project_ids = []
    split_details = []

    for part_idx, part_text in enumerate(part_texts):
        new_pid, new_pdir = store.create_project()
        part_title = f"Part {part_idx + 1}"

        if num_parts > 1:
            full_title = f"{base_title} — Part {part_idx + 1}/{num_parts}"
        else:
            full_title = base_title

        store.update_state(
            new_pid,
            title=full_title,
            book_group_id=book_group_id,
            book_title=book_title,
            chapter_index=chapter_index + part_idx,
            ollama_model=ollama_model,
            target_minutes=part_duration,
            tone=tone,
            custom_prompt=part_text,
            image_backend=image_backend,
            voice_profile_id=voice_profile_id,
            voice_language=voice_language,
        )

        if part_text:
            (new_pdir / "source_text.txt").write_text(part_text, encoding="utf-8")

        new_project_ids.append(new_pid)
        split_details.append({
            "title": part_title,
            "summary": f"Part {part_idx + 1} of {num_parts}",
            "char_count": len(part_text),
        })
        log.info(f"Created regular split project {new_pid}: '{full_title}' from {project_id}")

    # Delete the original project after successful split
    shutil.rmtree(pdir)
    log.info(f"Deleted original project {project_id} after regular split into {len(new_project_ids)} parts")

    return {
        "original_project_id": project_id,
        "new_project_ids": new_project_ids,
        "parts": len(new_project_ids),
        "split_details": split_details,
        "fallback": True,
        "message": "Intelligent split failed, used regular split instead",
    }


# ── Intelligent LLM-based splitting ───────────────────────────

INTELLIGENT_SPLIT_PROMPT = """You are a literary editor. Your task: split the given text into {num_parts} roughly equal parts.

EXTREMELY IMPORTANT: Your ENTIRE response must be ONLY a valid JSON object. NO explanations before. NO explanations after. NO markdown code fences with ```. Just raw JSON starting with {{ and ending with }}.

JSON format required:
- parts: array of {num_parts} objects, each with:
  - part_number: integer (1, 2, 3...)
  - title: string describing this section
  - summary: brief description of what happens
  - split_after_text: the EXACT last 50-100 characters from this part (copy verbatim from source text)
  - char_count: approximate character count
- reasoning: brief explanation of why you chose these split points

Rules:
1. Parts MUST be roughly equal in size. Each part should be between {min_pct}% and {max_pct}% of the total text. This is the most important rule.
2. Within the equal-size constraint, prefer to split at scene endings, time shifts, chapter breaks, or narrative pauses
3. split_after_text MUST be copied exactly from the source text
4. Output ONLY JSON - nothing else
"""


def _extract_json_from_llm_response(content: str) -> dict | None:
    """Extract JSON from LLM response, handling markdown fences and other common formats."""
    if not content or not content.strip():
        return None

    content = content.strip()

    # Try to find JSON between markdown fences with json tag
    json_match = re.search(r'```json\s*\n?(.*?)\n?```', content, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group(1).strip())
            log.info(f"Extracted JSON from ```json fence")
            return result
        except json.JSONDecodeError as e:
            log.warning(f"Failed to parse JSON from ```json fence: {e}")

    # Try to find JSON between plain markdown fences
    json_match = re.search(r'```\s*\n?(.*?)\n?```', content, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group(1).strip())
            log.info(f"Extracted JSON from ``` fence")
            return result
        except json.JSONDecodeError as e:
            log.warning(f"Failed to parse JSON from ``` fence: {e}")

    # Try to find JSON object with "parts" key - look for properly balanced braces
    # Try LAST match first (reasoning models output JSON at the end)
    all_matches = list(re.finditer(r'\{[\s\S]*?"parts"[\s\S]*?\}', content))
    for match in reversed(all_matches):
        try:
            json_str = match.group(0)
            # Try to find the outermost balanced braces
            brace_count = 0
            start = match.start()
            for i, char in enumerate(content[start:]):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = content[start:start+i+1]
                        break
            result = json.loads(json_str)
            log.info(f"Extracted JSON with 'parts' key (from {len(all_matches)} matches)")
            return result
        except json.JSONDecodeError:
            continue

    # Try to find any JSON object that looks like our response structure
    # Last resort: find content between first { and last }
    try:
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = content[start:end+1]
            result = json.loads(json_str)
            log.info(f"Extracted JSON from first '{' to last '}'")
            return result
    except json.JSONDecodeError:
        pass

    # Try the whole content as JSON
    try:
        result = json.loads(content)
        log.info(f"Parsed entire content as JSON")
        return result
    except json.JSONDecodeError:
        pass

    log.warning(f"All JSON extraction methods failed")
    return None


async def _find_split_points_with_llm(
    text: str,
    num_parts: int,
    model: str,
    base_url: str,
) -> list[dict]:
    """Use LLM to find logical split points in the text."""
    avg_pct = 100 / num_parts
    min_pct = max(14, int(avg_pct * 0.7))
    max_pct = min(86, int(avg_pct * 1.3))
    prompt = INTELLIGENT_SPLIT_PROMPT.format(num_parts=num_parts, min_pct=min_pct, max_pct=max_pct)

    # Send as much text as possible so the LLM can find balanced split points
    # (sending too little causes back-loaded splits since the LLM only sees the beginning)
    max_text = 60000
    text_for_llm = text[:max_text] if len(text) > max_text else text

    avg_chars = len(text) // num_parts
    user_message = (
        f"Please split this text into exactly {num_parts} roughly equal parts "
        f"(each part should be around {avg_chars:,} characters, between {min_pct}%-{max_pct}% of the total).\n\n"
        f"Text ({len(text)} chars total, {'showing first ' + str(max_text) + ' chars' if len(text) > max_text else 'full text'}):\n\n{text_for_llm}"
    )

    log.info(f"Calling LLM for intelligent split: model={model}, num_parts={num_parts}, text_len={len(text_for_llm)}")

    try:
        # First test if Ollama is reachable
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                health = await client.get(f"{base_url}/api/tags")
                log.info(f"Ollama health check: {health.status_code}")
                if health.status_code == 200:
                    models_data = health.json()
                    available_models = [m.get('name', m.get('model', '')) for m in models_data.get('models', [])]
                    log.info(f"Available Ollama models: {available_models}")
                    if model not in available_models and not any(m.startswith(model.split(':')[0]) for m in available_models):
                        log.error(f"Model '{model}' not found in available models: {available_models}")
            except Exception as e:
                log.error(f"Cannot connect to Ollama at {base_url}: {e}")
                return []

        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                f"{base_url}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": user_message},
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 8000,
                    },
                },
            )

        log.info(f"LLM response status: {resp.status_code}")

        if resp.status_code != 200:
            log.error(f"LLM split failed (HTTP {resp.status_code}), response: {resp.text[:500]}")
            try:
                error_data = resp.json()
                log.error(f"Ollama error details: {error_data}")
            except:
                pass
            return []

        data = resp.json()
        log.info(f"LLM response data keys: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}")

        # Try to get content directly from message.content (ollama standard)
        msg = data.get("message", {})
        content = msg.get("content", "") or ""

        # Also check for thinking field (some models use this)
        if not content:
            content = msg.get("thinking", "") or ""

        # Fallback to script_gen extractor
        if not content:
            content = script_gen._extract_llm_content(data)

        log.info(f"LLM content length: {len(content) if content else 0}")

        if not content or not content.strip():
            log.warning("LLM split returned empty content")
            return []

        # Log first 1000 chars of response for debugging
        log.info(f"LLM raw response preview: {content[:1000]}...")

        result = _extract_json_from_llm_response(content)
        if result is None:
            log.error(f"Could not extract JSON from LLM response")
            log.error(f"Raw content type: {type(content)}, length: {len(content)}")
            log.error(f"Full response:\n{'='*50}\n{content}\n{'='*50}")
            return []

        log.info(f"Extracted JSON keys: {list(result.keys()) if isinstance(result, dict) else 'not a dict'}")

        parts = result.get("parts", [])
        if not parts:
            log.warning(f"LLM response missing 'parts' array. Got keys: {list(result.keys())}")
            return []

        log.info(f"LLM found {len(parts)} split points")
        return parts

    except (httpx.TimeoutException, httpx.ReadTimeout):
        log.error("LLM split request timed out after 300s")
        return []
    except Exception as exc:
        log.error(f"LLM split error: {type(exc).__name__}: {exc}")
        import traceback
        log.error(traceback.format_exc())
        return []


def _find_split_position(text: str, split_after_text: str) -> int:
    """Find the position in text after the given split marker text."""
    if not split_after_text or len(split_after_text) < 10:
        return -1

    # Normalize whitespace in both texts for comparison
    def normalize_ws(s: str) -> str:
        return ' '.join(s.split())

    normalized_marker = normalize_ws(split_after_text)
    normalized_text = normalize_ws(text)

    # Try exact match first (case-sensitive) on normalized text
    pos = normalized_text.find(normalized_marker)
    if pos != -1:
        # Map back to original text position
        char_count = 0
        original_pos = 0
        for char in text:
            if char_count >= pos + len(normalized_marker):
                break
            if not char.isspace() or (char_count > 0 and normalized_text[char_count:char_count+1] == ' '):
                char_count += 1
            original_pos += 1
        return original_pos

    # Try case-insensitive on normalized
    pos = normalized_text.lower().find(normalized_marker.lower())
    if pos != -1:
        char_count = 0
        original_pos = 0
        for char in text:
            if char_count >= pos + len(normalized_marker):
                break
            if not char.isspace() or (char_count > 0 and normalized_text[char_count:char_count+1] == ' '):
                char_count += 1
            original_pos += 1
        return original_pos

    # Try matching just the last 30 chars of the marker
    short_marker = normalized_marker[-30:] if len(normalized_marker) > 30 else normalized_marker
    pos = normalized_text.find(short_marker)
    if pos != -1:
        char_count = 0
        original_pos = 0
        for char in text:
            if char_count >= pos + len(short_marker):
                break
            if not char.isspace() or (char_count > 0 and normalized_text[char_count:char_count+1] == ' '):
                char_count += 1
            original_pos += 1
        return original_pos

    return -1


def _find_paragraph_near(text: str, target_pos: int, search_range: int) -> int:
    """Find the nearest paragraph break (\\n\\n) to target_pos within search_range.
    Returns the position after the break, or -1 if none found."""
    text_len = len(text)
    best_para = -1
    best_dist = float('inf')
    for pos in range(max(0, target_pos - search_range), min(text_len - 1, target_pos + search_range)):
        if text[pos:pos+2] == "\n\n":
            dist = abs(pos - target_pos)
            if dist < best_dist:
                best_dist = dist
                best_para = pos
    return best_para + 2 if best_para != -1 else -1


def _split_text_intelligently(text: str, split_points: list[dict]) -> list[tuple[str, str, str]]:
    """Split text based on LLM-provided split points.

    Returns list of (title, summary, part_text) tuples.
    Enforces balance: no part can be smaller than 70% of the average part size.
    """
    num_parts = len(split_points)
    text_len = len(text)
    ideal_part_size = text_len // num_parts
    # Minimum part size: 70% of ideal (e.g. 2 parts → 35%-65% range)
    min_part_size = int(ideal_part_size * 0.70)

    parts = []
    start_pos = 0

    for i, point in enumerate(split_points):
        if i == len(split_points) - 1:
            part_text = text[start_pos:].strip()
            title = point.get("title", f"Part {i + 1}")
            summary = point.get("summary", "")
            parts.append((title, summary, part_text))
            break

        split_after = point.get("split_after_text", "")

        # Try to find LLM-suggested split position
        split_pos = _find_split_position(text[start_pos:], split_after) if split_after else -1
        if split_pos != -1:
            split_pos += start_pos

        # Balance check: reject the LLM position if it creates a part that's too small or too large
        remaining_text = text_len - start_pos
        remaining_parts = num_parts - i
        if split_pos != -1 and split_pos > start_pos:
            this_part_size = split_pos - start_pos
            leftover = remaining_text - this_part_size
            leftover_avg = leftover / (remaining_parts - 1) if remaining_parts > 1 else leftover
            if this_part_size < min_part_size:
                log.warning(f"LLM split at {split_pos} gives part {i+1} only {this_part_size} chars (min {min_part_size}), rebalancing")
                split_pos = -1
            elif leftover_avg < min_part_size:
                log.warning(f"LLM split at {split_pos} leaves too little for remaining parts ({leftover_avg:.0f} avg), rebalancing")
                split_pos = -1

        if split_pos == -1 or split_pos <= start_pos:
            # Fallback: target a balanced split
            target_pos = start_pos + (remaining_text // remaining_parts)
            search_range = int(ideal_part_size * 0.2)

            para_pos = _find_paragraph_near(text, target_pos, search_range)
            if para_pos != -1:
                split_pos = para_pos
            else:
                split_pos = target_pos
                next_period = text.find(". ", split_pos)
                if next_period != -1 and next_period < split_pos + 200:
                    split_pos = next_period + 2

        split_pos = max(split_pos, start_pos + min_part_size)
        split_pos = min(split_pos, text_len)

        part_text = text[start_pos:split_pos].strip()
        start_pos = split_pos

        title = point.get("title", f"Part {i + 1}")
        summary = point.get("summary", "")
        parts.append((title, summary, part_text))

    # Post-split balance check: log the distribution
    sizes = [len(p[2]) for p in parts]
    total = sum(sizes)
    pcts = [s / total * 100 for s in sizes]
    log.info(f"Split result: {num_parts} parts, sizes={sizes}, ratios={[f'{p:.1f}%' for p in pcts]}")

    return parts


@app.post("/api/projects/{project_id}/split-intelligent")
async def split_project_intelligent(project_id: str, req: IntelligentSplitRequest):
    """Split a project using LLM to find logical narrative break points.

    The original project is deleted after successful split.
    """
    if req.parts < 2:
        raise HTTPException(400, "Parts must be at least 2")

    try:
        state = store.load_state(project_id)
    except FileNotFoundError:
        raise HTTPException(404, "Project not found")

    pdir = store.project_dir(project_id)
    source_file = pdir / "source_text.txt"

    if not source_file.exists():
        raise HTTPException(400, "Project has no source text to split")

    full_text = source_file.read_text(encoding="utf-8")
    if not full_text.strip():
        raise HTTPException(400, "Source text is empty")

    # Get LLM model
    ollama_model = req.ollama_model or state.get("ollama_model", config.OLLAMA_MODEL)
    base_url = config.OLLAMA_URL

    # Ask LLM to find split points
    log.info(f"Requesting intelligent split for {project_id} into {req.parts} parts using {ollama_model}")
    log.info(f"Text length: {len(full_text)} chars, sending {min(len(full_text), 4000)} to LLM")

    split_points = await _find_split_points_with_llm(
        text=full_text,
        num_parts=req.parts,
        model=ollama_model,
        base_url=base_url,
    )

    if not split_points:
        log.error(f"Intelligent split failed for {project_id}: LLM returned no split points")
        raise HTTPException(500, f"LLM could not determine split points using model '{ollama_model}'. Check that Ollama is running, the model is available, and try again.")

    # Split text based on LLM suggestions
    parts = _split_text_intelligently(full_text, split_points)
    log.info(f"Split text into {len(parts)} parts")

    for i, (title, summary, text) in enumerate(parts):
        log.info(f"Part {i+1}: '{title}' ({len(text)} chars)")

    if len(parts) < 2:
        raise HTTPException(500, "Could not create meaningful split")

    # Get project metadata
    book_group_id = state.get("book_group_id")
    book_title = state.get("book_title", "")
    chapter_index = state.get("chapter_index", 0)
    base_title = state.get("title", "Untitled")
    image_backend = state.get("image_backend", "comfyui")
    tone = state.get("tone", "")
    voice_profile_id = state.get("voice_profile_id")
    voice_language = state.get("voice_language", "en")
    target_minutes = state.get("target_minutes", 5.0)

    # Calculate duration per part proportionally
    total_chars = len(full_text)

    # Create new projects
    new_project_ids = []
    for part_idx, (part_title, part_summary, part_text) in enumerate(parts):
        new_pid, new_pdir = store.create_project()

        # Calculate proportional duration
        part_duration = max(1.0, round(target_minutes * (len(part_text) / total_chars), 1))

        # Build full title with book info if available
        if book_title:
            full_title = f"{book_title} — {base_title} — {part_title}"
        else:
            full_title = f"{base_title} — {part_title}"

        store.update_state(
            new_pid,
            title=full_title,
            book_group_id=book_group_id,
            book_title=book_title,
            chapter_index=chapter_index + part_idx,
            ollama_model=ollama_model,
            target_minutes=part_duration,
            tone=tone,
            custom_prompt=part_text,
            image_backend=image_backend,
            voice_profile_id=voice_profile_id,
            voice_language=voice_language,
        )

        # Save source text
        (new_pdir / "source_text.txt").write_text(part_text, encoding="utf-8")

        # Save summary as metadata
        store.save_json(new_pid, "split_info.json", {
            "original_project_id": project_id,
            "part_number": part_idx + 1,
            "total_parts": len(parts),
            "title": part_title,
            "summary": part_summary,
            "char_count": len(part_text),
        })

        new_project_ids.append(new_pid)
        log.info(f"Created intelligent split project {new_pid}: '{full_title}' ({len(part_text)} chars)")

    # Delete original project
    shutil.rmtree(pdir)
    log.info(f"Deleted original project {project_id} after intelligent split into {len(new_project_ids)} parts")

    return {
        "original_project_id": project_id,
        "new_project_ids": new_project_ids,
        "parts": len(new_project_ids),
        "split_details": [
            {"title": p[0], "summary": p[1], "char_count": len(p[2])}
            for p in parts
        ],
    }


# ── Run ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8102)
