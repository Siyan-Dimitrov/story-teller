# Story Teller — Next session plan

Branch: `master` (updated 2026-07-09)
Kimi/Ollama code preserved on branch: `open-source-models`

## What just landed (2026-07-09)

1. **Motion redesign** (commit `fddc7b5`): image count scales with narration
   (~1 prompt per 20 words, min 3, max 10); I2V rationed to 1 clip/scene
   (most dramatic wins, rest use free depth parallax); I2V plays once at
   native speed then freeze-zooms the last frame (never slowed/looped);
   real crossfades between images within a scene and between scenes.
2. **Claude-only text generation** (this commit): every text-LLM call goes
   through `backend/llm.py` → Claude Agent SDK → Claude Code subscription
   OAuth (no API key). Opus 4.8 (`CLAUDE_MODEL`) for writer/critic/reviser,
   cast bible, YouTube metadata; Haiku 4.5 (`CLAUDE_FAST_MODEL`) for image
   classification, chapter tagging, shorts selection, music suggestions,
   story search, chapter splitting. Ollama removed from backend + frontend.
   Verified: imports, frontend build, live Haiku smoke tests, code review.

Caveats carried forward:
- The Agent SDK subprocess hang risk (10-min timeout catches it). If long
  chapters hang repeatedly, switch to an Anthropic API key — the `llm.py`
  structure makes it a small change (SDK client swap in one file).
- `start_full.bat` (untracked) may still launch/configure Ollama — strip
  locally. Its SerpAPI key still needs rotating (see Security below).
- Old projects' `state.json` may contain `ollama_model`/`script_backend`
  keys — harmless, ignored.
- `image_gen.py` still has a placeholder "ollama" *image* backend (local
  text-placeholder images, no server) — unrelated to text migration.

## Step 1 — voice provider bake-off (DECISION PENDING)

User is unhappy with local VoiceBox (clicks, DC offsets, flat prosody).
Research done 2026-07-09 (full reports in that session's transcript; summary
in auto-memory `project_voice_provider.md`). Shortlist:

| Option | ~$/12-min video | Notes |
|---|---|---|
| **MiniMax Speech 2.8-HD via Replicate** | ~$0.60 | Top pick. Existing Replicate key/helpers; $3 one-time voice clone → stable voice ID; 44.1 kHz WAV; emotion + pause controls. |
| ElevenLabs `eleven_multilingual_v2` | ~$1.10–2.20 + $22/mo Creator | Request stitching (`previous_request_ids`) chains scene-to-scene prosody; stability ~0.7. Do NOT use v3 (no stitching, hallucination-prone). Creator tier is the commercial floor. |
| Hume Octave 2 | ~$0.84 ($70/mo Pro) | Best emotional acting; voice ID + continuation feature purpose-built for per-scene calls. Escalation if MiniMax isn't dramatic enough. |
| Chatterbox (local, MIT) | $0 | ~6 GB VRAM; `exaggeration` dial; devnen/Chatterbox-TTS-Server exposes an OpenAI-style REST API that slots in where VoiceBox is. |
| Fish Audio S2-Pro | ~$0.18 + $11/mo | Budget option. |

Avoid: OpenAI TTS (documented cross-request voice drift), PlayHT (dead),
F5-TTS/XTTS (non-commercial licences).

**Plan (~$5 total):** clone/design one dark-narrator voice; render the same
2–3 scenes on (1) MiniMax HD via Replicate, (2) Chatterbox local at
exaggeration ~0.6–0.7, (3) Hume free tier; A/B the concatenated results for
prosody and seam consistency. Then wire the winner as a `voice_backend`
alongside VoiceBox.

## Step 2 — validate the new pipeline end-to-end

Run one full chapter through Producer on master:
- Script via Opus 4.8 (watch for Agent SDK hangs; timeout fires at 10 min).
- Confirm variable image counts land (~1 per 20 words) and the critic
  doesn't fight the new rule.
- Confirm 1 I2V clip/scene + freeze-zoom tails + crossfades look right in
  the assembled video.
- Classification now runs on Haiku — check the type/motion distribution log.

## Step 3 — security cleanup (unchanged, one focused PR)

1. Path traversal in delete endpoints (`main.py`): validate project_id
   `^[a-f0-9]{12}$` + `is_relative_to(PROJECTS_DIR)` before rmtree.
2. Music absolute-path read (`video_assembly.py` `_resolve_music_path`):
   constrain to `MUSIC_DIR / Path(name).name`.
3. **Rotate the SerpAPI key** in `start_full.bat` — then the file can be
   committed.
(SSRF guards for gutenberg/music URLs landed in `54566ed`.)

## Step 4 — known-but-deferred bugs (still open)

- `update_state` non-atomic write + per-project lock (`project_store.py`).
- Loudnorm cache key truncates mtime to seconds (`video_assembly.py`).
- `voice_gen._split_into_sentences` returns `[""]` for empty input.
- Voice/duration mismatch: audio ~32% shorter than `duration_hint` —
  may be moot if the voice provider changes (Step 1).
- Render-time profiling: 47 min for an 8-min video; target <15 min
  (cProfile `assemble_video`; consider ffmpeg concat / parallel scenes).
- Chunk-then-stitch for >20-min chapters (Kimi truncation is gone with
  Claude, but very long chapters still deserve splitting — the intelligent
  split endpoint now runs on Haiku).

## Myths book group `0a86d60157a7` status (as of 2026-06-20)

- assembled: chapters 1, 2, 3, 19 · failed→scripted: 18 · created: 4-17, 20.
- Driving remaining chapters: ~90 min wall each; spend now = images + I2V
  (text is on subscription quota).

## How to start next session

1. Read this file.
2. Pick: voice bake-off (Step 1, needs user's ear) or pipeline validation
   (Step 2, autonomous).
3. If anything in the Claude migration regressed, the migration commit and
   the `open-source-models` branch make rollback/diff easy.
