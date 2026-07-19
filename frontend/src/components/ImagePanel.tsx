import { useState, useEffect, useRef } from 'react'
import { ImageIcon, Loader2, RefreshCw, RotateCcw, Users, Wrench, AlertTriangle } from 'lucide-react'
import type { ProjectState, ImageStyle, CastMember, ImagesProgress } from '../api'
import { api } from '../api'

interface Props {
  project: ProjectState
  onRefresh: () => void
  onNext: () => void
}

// Still-image backends offered in the UI. Nano Banana is the default (consistent
// + cheap); GPT Image kept as a fallback. ComfyUI/Replicate-Flux/Ollama were
// retired from the dropdown.
const IMAGE_BACKENDS = ['nano_banana', 'gpt_image'] as const
const DEFAULT_BACKEND = 'nano_banana'

export default function ImagePanel({ project, onRefresh, onNext }: Props) {
  const [backend, setBackend] = useState(
    (IMAGE_BACKENDS as readonly string[]).includes(project.image_backend)
      ? project.image_backend
      : DEFAULT_BACKEND
  )
  const [imageStyles, setImageStyles] = useState<ImageStyle[]>([])
  const [selectedStyleId, setSelectedStyleId] = useState('')
  const [customStylePrompt, setCustomStylePrompt] = useState('')
  const [characterConsistency, setCharacterConsistency] = useState(true)
  const [checkDuplicates, setCheckDuplicates] = useState(false)
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState<ImagesProgress | null>(null)
  const [runError, setRunError] = useState<string | null>(null)
  const [regeneratingScene, setRegeneratingScene] = useState<number | null>(null)
  const [regeneratingImage, setRegeneratingImage] = useState<string | null>(null) // `${scene}:${img}`
  const [cast, setCast] = useState<CastMember[]>(project.script?.cast || [])
  const [castBusy, setCastBusy] = useState<string | null>(null) // 'cast' | 'refs' | `ref:${id}`
  const [tick, setTick] = useState(0) // cache-buster for regenerated files
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const prevGeneratedRef = useRef(-1)

  useEffect(() => {
    setCast(project.script?.cast || [])
  }, [project.script?.cast])

  const supportsRefs = backend === 'nano_banana'

  useEffect(() => {
    api.imageStyles().then(data => {
      setImageStyles(data.styles)
      setSelectedStyleId(data.default_style_id || data.styles[0]?.id || '')
    }).catch(() => {})
  }, [])

  // Resume polling if a run is already active for this project (page reload,
  // navigating back mid-run, etc).
  useEffect(() => {
    api.imagesProgress(project.project_id).then(p => {
      if (p.active) {
        setRunning(true)
        setProgress(p)
        startPolling()
      }
    }).catch(() => {})
    return stopPolling
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project.project_id])

  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }

  const startPolling = () => {
    stopPolling()
    pollRef.current = setInterval(async () => {
      try {
        const p = await api.imagesProgress(project.project_id)
        setProgress(p)
        // Refresh the project (and bust the image cache) only when something
        // new landed on disk, so the grid fills in live without re-fetching
        // every image every poll.
        if (p.generated !== prevGeneratedRef.current) {
          prevGeneratedRef.current = p.generated
          setTick(t => t + 1)
          onRefresh()
        }
        if (!p.active) {
          stopPolling()
          setRunning(false)
          setRunError(p.error)
          setTick(t => t + 1)
          onRefresh()
        }
      } catch {
        // Backend might be restarting — keep polling.
      }
    }, 3000)
  }

  const selectedStyle = imageStyles.find(s => s.id === selectedStyleId)
  const customStyle = customStylePrompt.trim()
  const styleRequest = customStyle
    ? { custom_style_prompt: customStyle, style_prompt: customStyle }
    : selectedStyleId ? { style_id: selectedStyleId } : {}

  const requestBody = {
    backend,
    ...styleRequest,
    character_consistency: characterConsistency && supportsRefs,
  }

  const scenes = project.script?.scenes || []
  const hasGeneratedImages = scenes.some(s => (s.image_paths && s.image_paths.length > 0) || s.image_path)
  const hasImageResults = scenes.some(s => (s.image_paths && s.image_paths.length > 0) || s.image_path || s.image_error)
  const busy = running || regeneratingScene !== null || regeneratingImage !== null

  const handleGenerate = async () => {
    setRunError(null)
    setRunning(true)
    setProgress({ active: true, phase: 'starting', generated: 0, total: 0, error: null })
    prevGeneratedRef.current = -1
    try {
      await api.runImages(project.project_id, requestBody)
      startPolling()
    } catch (e) {
      setRunning(false)
      setRunError('Failed to start image generation: ' + (e as Error).message)
    }
  }

  const handleRepair = async () => {
    setRunError(null)
    setRunning(true)
    setProgress({ active: true, phase: 'scanning for failed images', generated: 0, total: 0, error: null })
    prevGeneratedRef.current = -1
    try {
      await api.repairImages(project.project_id, { ...requestBody, check_duplicates: checkDuplicates })
      startPolling()
    } catch (e) {
      setRunning(false)
      setRunError('Failed to start repair: ' + (e as Error).message)
    }
  }

  const handleRegenerateScene = async (sceneIndex: number) => {
    setRegeneratingScene(sceneIndex)
    try {
      await api.regenerateSceneImages(project.project_id, sceneIndex, requestBody)
      setTick(t => t + 1)
      onRefresh()
    } catch (e) {
      alert('Scene regeneration failed: ' + (e as Error).message)
    } finally {
      setRegeneratingScene(null)
    }
  }

  const handleRegenerateImage = async (sceneIndex: number, imageIndex: number) => {
    setRegeneratingImage(`${sceneIndex}:${imageIndex}`)
    try {
      await api.regenerateSingleImage(project.project_id, sceneIndex, imageIndex, requestBody)
      setTick(t => t + 1)
      onRefresh()
    } catch (e) {
      alert('Image regeneration failed: ' + (e as Error).message)
    } finally {
      setRegeneratingImage(null)
    }
  }

  const handleGenerateCast = async () => {
    setCastBusy('cast')
    try {
      const res = await api.generateCast(project.project_id, { overwrite: true })
      setCast(res.cast)
      onRefresh()
    } catch (e) {
      alert('Cast generation failed: ' + (e as Error).message)
    } finally {
      setCastBusy(null)
    }
  }

  const handleGenerateRefs = async (castIds?: string[]) => {
    setCastBusy(castIds && castIds.length === 1 ? `ref:${castIds[0]}` : 'refs')
    try {
      const res = await api.generateCharacterRefs(project.project_id, {
        backend: 'nano_banana',
        ...styleRequest,
        ...(castIds ? { cast_ids: castIds } : {}),
      })
      setCast(res.cast)
      setTick(t => t + 1)
      onRefresh()
    } catch (e) {
      alert('Character portrait generation failed: ' + (e as Error).message)
    } finally {
      setCastBusy(null)
    }
  }

  if (!project.script) {
    return (
      <div className="text-center py-16 text-[var(--text-muted)]">
        <ImageIcon size={40} className="mx-auto mb-3 opacity-30" />
        <p>Generate a script first.</p>
      </div>
    )
  }

  const pct = progress && progress.total > 0 ? Math.round((progress.generated / progress.total) * 100) : 0

  return (
    <div className="space-y-4">
      {/* Settings */}
      <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)]">
        <div className="space-y-3">
          <div className="flex items-end gap-4">
            <div className="w-40">
              <label className="block text-xs text-[var(--text-secondary)] mb-1.5">Backend</label>
              <select
                value={backend}
                onChange={e => setBackend(e.target.value)}
                className="w-full bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--border-focus)]"
              >
                <option value="nano_banana">Nano Banana (Consistent)</option>
                <option value="gpt_image">GPT Image 2 (OpenAI)</option>
              </select>
            </div>
            <div className="flex-1">
              <label className="block text-xs text-[var(--text-secondary)] mb-1.5">Image Style</label>
              <select
                value={selectedStyleId}
                onChange={e => setSelectedStyleId(e.target.value)}
                disabled={imageStyles.length === 0}
                className="w-full bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--border-focus)]"
              >
                {imageStyles.length === 0 && <option value="">Loading styles...</option>}
                {imageStyles.map(style => (
                  <option key={style.id} value={style.id}>{style.label}</option>
                ))}
              </select>
            </div>
          </div>
          {selectedStyle?.description && !customStyle && (
            <p className="text-xs text-[var(--text-muted)]">{selectedStyle.description}</p>
          )}
          <div>
            <label className="block text-xs text-[var(--text-secondary)] mb-1.5">Advanced Custom Style Override</label>
            <textarea
              value={customStylePrompt}
              onChange={e => setCustomStylePrompt(e.target.value)}
              placeholder="Optional custom style prompt. When set, it overrides the selected style."
              rows={2}
              className="w-full bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:outline-none focus:border-[var(--border-focus)] resize-none"
            />
          </div>
          {supportsRefs && (
            <label className="flex items-center gap-2 text-sm text-[var(--text-secondary)] cursor-pointer">
              <input
                type="checkbox"
                checked={characterConsistency}
                onChange={e => setCharacterConsistency(e.target.checked)}
                className="rounded border-[var(--border)]"
              />
              <span>Character Consistency</span>
              <span className="text-[10px] text-[var(--text-muted)]">
                Builds a cast bible + per-character reference portraits reused across every scene
              </span>
            </label>
          )}
          {backend === 'nano_banana' && characterConsistency && (
            <div className="rounded-lg border border-[var(--border)] bg-[var(--bg-tertiary)] p-3 space-y-3">
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-1.5 text-xs font-medium text-[var(--text-secondary)]">
                  <Users size={13} /> Cast & Character References
                </span>
                <div className="flex items-center gap-2">
                  <button
                    onClick={handleGenerateCast}
                    disabled={castBusy !== null || busy}
                    className="flex items-center gap-1.5 px-2.5 py-1 rounded text-xs border border-[var(--border)] text-[var(--text-secondary)] hover:text-[var(--accent)] disabled:opacity-50"
                  >
                    {castBusy === 'cast' ? <Loader2 size={11} className="animate-spin" /> : <RefreshCw size={11} />}
                    {cast.length > 0 ? 'Rebuild cast' : 'Build cast'}
                  </button>
                  <button
                    onClick={() => handleGenerateRefs()}
                    disabled={castBusy !== null || busy || cast.length === 0}
                    className="flex items-center gap-1.5 px-2.5 py-1 rounded text-xs bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white disabled:opacity-50"
                  >
                    {castBusy === 'refs' ? <Loader2 size={11} className="animate-spin" /> : <ImageIcon size={11} />}
                    Generate portraits
                  </button>
                </div>
              </div>
              {cast.length === 0 ? (
                <p className="text-[11px] text-[var(--text-muted)]">
                  No cast yet. Build the cast bible (auto-extracted from the script), generate the portraits, then run image generation — each scene reuses its characters' portraits for a consistent look. Running "Generate Images" does all of this automatically.
                </p>
              ) : (
                <div className="grid grid-cols-3 sm:grid-cols-4 gap-2">
                  {cast.map(member => (
                    <div
                      key={member.id}
                      className={`relative group rounded-lg overflow-hidden border bg-[var(--bg-secondary)] ${member.reference_image_error ? 'border-[var(--error)]' : 'border-[var(--border)]'}`}
                    >
                      {member.reference_image_path ? (
                        <img
                          src={api.artifactUrl(project.project_id, member.reference_image_path) + `?t=${tick}`}
                          alt={member.name}
                          className="w-full aspect-[3/4] object-cover"
                        />
                      ) : (
                        <div className="w-full aspect-[3/4] flex items-center justify-center text-[var(--text-muted)]">
                          <Users size={20} className="opacity-40" />
                        </div>
                      )}
                      <button
                        onClick={() => handleGenerateRefs([member.id])}
                        disabled={castBusy !== null || busy}
                        title="Regenerate this portrait"
                        className="absolute top-1 right-1 p-1 rounded bg-black/60 text-white opacity-0 group-hover:opacity-100 transition-opacity disabled:opacity-30"
                      >
                        {castBusy === `ref:${member.id}` ? <Loader2 size={11} className="animate-spin" /> : <RotateCcw size={11} />}
                      </button>
                      <div className="px-1.5 py-1">
                        <div className="text-[11px] font-medium text-[var(--text-primary)] truncate" title={member.name}>{member.name}</div>
                        {member.reference_image_error ? (
                          <div className="text-[9px] text-[var(--error)] truncate" title={member.reference_image_error}>portrait failed</div>
                        ) : (
                          member.role && <div className="text-[9px] text-[var(--text-muted)] capitalize truncate">{member.role}</div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
          {backend === 'gpt_image' && (
            <p className="text-xs text-[var(--text-muted)]">
              Uses <code>OPENAI_API_KEY</code> from the backend <code>.env</code> file. Style is taken from the selected style or custom prompt. No character-reference consistency on this backend.
            </p>
          )}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              {hasImageResults && (
                <>
                  <button
                    onClick={handleRepair}
                    disabled={busy}
                    title="Scan every scene and regenerate only failed or missing images"
                    className="flex items-center gap-2 px-3 py-2 rounded-lg border border-[var(--border)] text-[var(--text-secondary)] hover:text-[var(--accent)] text-sm transition-colors disabled:opacity-50"
                  >
                    <Wrench size={14} />
                    Repair missing
                  </button>
                  <label className="flex items-center gap-1.5 text-[11px] text-[var(--text-muted)] cursor-pointer">
                    <input
                      type="checkbox"
                      checked={checkDuplicates}
                      onChange={e => setCheckDuplicates(e.target.checked)}
                      className="rounded border-[var(--border)]"
                    />
                    also scan for duplicated people (slower)
                  </label>
                </>
              )}
            </div>
            <button
              onClick={handleGenerate}
              disabled={busy}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white text-sm font-medium transition-colors disabled:opacity-50"
            >
              {running ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
              {running ? 'Generating...' : hasGeneratedImages ? 'Regenerate All' : 'Generate Images'}
            </button>
          </div>

          {/* Live progress */}
          {running && progress && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-[var(--text-secondary)]">{progress.phase}</span>
                <span className="text-[var(--text-muted)] tabular-nums">
                  {progress.total > 0 ? `${progress.generated}/${progress.total} · ${pct}%` : ''}
                </span>
              </div>
              <div className="h-2 rounded-full bg-[var(--bg-tertiary)] overflow-hidden">
                <div
                  className="h-full rounded-full bg-[var(--accent)] transition-all duration-700 ease-out"
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>
          )}
          {runError && !running && (
            <div className="p-2 rounded-lg border border-[var(--error)]/30 bg-[var(--error)]/5 text-xs text-[var(--error)]">
              {runError}
            </div>
          )}
        </div>
      </div>

      {/* Image grid — one card per scene, every prompt slot visible */}
      {(hasImageResults || running) && (
        <div className="space-y-4">
          {scenes.map((scene, i) => {
            const prompts = (scene.image_prompts && scene.image_prompts.length > 0)
              ? scene.image_prompts
              : scene.image_prompt ? [scene.image_prompt] : []
            if (prompts.length === 0) return null

            const paths = (scene.image_paths && scene.image_paths.length > 0)
              ? scene.image_paths
              : scene.image_path ? [scene.image_path] : []
            const errors = scene.image_errors || []
            // Slots map to files by the deterministic filename, so a failed
            // slot in the middle can't shift later images.
            const pathFor = (j: number) =>
              paths.find(p => p.includes(`_img_${j}.`)) ?? (paths.length === prompts.length ? paths[j] : undefined)
            const doneCount = prompts.filter((_, j) => pathFor(j)).length

            return (
              <div key={i} className="rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] overflow-hidden">
                <div className="px-3 py-2 border-b border-[var(--border)] flex items-center justify-between">
                  <span className="text-xs font-medium text-[var(--accent)]">Scene {i + 1}</span>
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] text-[var(--text-muted)]">{doneCount}/{prompts.length} image{prompts.length !== 1 ? 's' : ''}</span>
                    <span className="text-[10px] text-[var(--text-muted)]">{scene.mood}</span>
                    <button
                      onClick={() => handleRegenerateScene(i)}
                      disabled={busy}
                      className="p-1 rounded text-[var(--text-muted)] hover:text-[var(--accent)] transition-colors disabled:opacity-40"
                      title="Regenerate this whole scene"
                    >
                      {regeneratingScene === i ? <Loader2 size={12} className="animate-spin" /> : <RotateCcw size={12} />}
                    </button>
                  </div>
                </div>
                <div className="grid grid-cols-4 gap-1 p-1">
                  {prompts.map((prompt, j) => {
                    const path = pathFor(j)
                    const err = errors[j]
                    const regenKey = `${i}:${j}`
                    const regenning = regeneratingImage === regenKey
                    return (
                      <div key={j} className={`relative group rounded overflow-hidden ${err ? 'ring-1 ring-[var(--error)]' : ''}`}>
                        {path ? (
                          <img
                            src={api.artifactUrl(project.project_id, path) + `?t=${tick}`}
                            alt={`Scene ${i + 1} - Image ${j + 1}`}
                            className="w-full aspect-video object-cover"
                          />
                        ) : (
                          <div className="w-full aspect-video flex flex-col items-center justify-center gap-1 bg-[var(--bg-tertiary)] text-[var(--text-muted)]">
                            {err ? <AlertTriangle size={14} className="text-[var(--error)]" /> : <ImageIcon size={14} className="opacity-40" />}
                            <span className="text-[9px]">{err ? 'failed' : running ? 'pending…' : 'not generated'}</span>
                          </div>
                        )}
                        <div className="absolute inset-0 bg-black/70 opacity-0 group-hover:opacity-100 transition-opacity p-1.5 flex flex-col">
                          <p className="text-[9px] text-white/80 leading-tight flex-1 overflow-auto">{prompt}</p>
                          {err && <p className="text-[9px] text-red-300 leading-tight max-h-8 overflow-auto">{err}</p>}
                          <div className="flex justify-end pt-1">
                            <button
                              onClick={() => handleRegenerateImage(i, j)}
                              disabled={busy}
                              title="Regenerate this image"
                              className="flex items-center gap-1 px-1.5 py-0.5 rounded bg-white/15 hover:bg-white/30 text-white text-[9px] disabled:opacity-40"
                            >
                              {regenning ? <Loader2 size={9} className="animate-spin" /> : <RotateCcw size={9} />}
                              redo
                            </button>
                          </div>
                        </div>
                      </div>
                    )
                  })}
                </div>
                {scene.image_error && (
                  <div className="px-3 pb-2 text-xs text-[var(--error)]">
                    {scene.image_error}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}

      {/* Next */}
      {hasGeneratedImages && !running && (
        <div className="flex justify-end">
          <button
            onClick={onNext}
            className="px-4 py-2 rounded-lg bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white text-sm font-medium transition-colors"
          >
            Next: Animate
          </button>
        </div>
      )}
    </div>
  )
}
