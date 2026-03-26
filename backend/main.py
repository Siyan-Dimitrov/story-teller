"""Story Teller — FastAPI backend."""

import asyncio
import logging
import shutil
import sys
import threading
import traceback
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import httpx

from . import config
from . import project_store as store
from . import script_gen, voice_gen, image_gen
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
    RunQCRequest,
    RegenerateQCRequest,
    RunAssembleRequest,
    SearchStoriesRequest,
    HealthStatus,
    ProjectSummary,
)
from .image_qc import run_qc_for_project, regenerate_and_evaluate, get_qc_progress, evaluate_single_image

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
        )
        return {"results": results}
    except Exception as e:
        log.error(f"Story search failed: {e}")
        raise HTTPException(500, f"Story search failed: {e}")


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

    store.update_state(project_id, step="assembling", error=None)
    pdir = store.project_dir(project_id)

    def _run():
        try:
            output, duration = assemble_video(
                scenes=script["scenes"],
                project_dir=pdir,
                project_id=project_id,
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
