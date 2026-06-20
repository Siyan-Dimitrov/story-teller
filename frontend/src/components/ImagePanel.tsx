import { useState, useEffect } from 'react'
import { ImageIcon, Loader2, RefreshCw, RotateCcw, Users } from 'lucide-react'
import type { ProjectState, ImageStyle, CastMember } from '../api'
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
  const [generating, setGenerating] = useState(false)
  const [characterConsistency, setCharacterConsistency] = useState(false)
  const [regeneratingScene, setRegeneratingScene] = useState<number | null>(null)
  const [cast, setCast] = useState<CastMember[]>(project.script?.cast || [])
  const [castBusy, setCastBusy] = useState<'cast' | 'refs' | null>(null)

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

  const selectedStyle = imageStyles.find(s => s.id === selectedStyleId)
  const customStyle = customStylePrompt.trim()
  const styleRequest = customStyle
    ? { custom_style_prompt: customStyle, style_prompt: customStyle }
    : selectedStyleId ? { style_id: selectedStyleId } : {}

  const scenes = project.script?.scenes || []
  const hasGeneratedImages = scenes.some(s => (s.image_paths && s.image_paths.length > 0) || s.image_path)
  const hasImageResults = scenes.some(s => (s.image_paths && s.image_paths.length > 0) || s.image_path || s.image_error)

  const handleGenerate = async () => {
    setGenerating(true)
    try {
      await api.runImages(project.project_id, {
        backend,
        ...styleRequest,
        ...(characterConsistency && supportsRefs ? { character_consistency: true } : {}),
      })
      onRefresh()
    } catch (e) {
      alert('Image generation failed: ' + (e as Error).message)
      onRefresh()
    } finally {
      setGenerating(false)
    }
  }

  const handleRegenerateScene = async (sceneIndex: number) => {
    setRegeneratingScene(sceneIndex)
    try {
      await api.regenerateSceneImages(project.project_id, sceneIndex, {
        backend,
        ...styleRequest,
        character_consistency: characterConsistency && supportsRefs,
      })
      onRefresh()
    } catch (e) {
      alert('Scene regeneration failed: ' + (e as Error).message)
    } finally {
      setRegeneratingScene(null)
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

  const handleGenerateRefs = async () => {
    setCastBusy('refs')
    try {
      const res = await api.generateCharacterRefs(project.project_id, {
        backend: 'nano_banana',
        ...styleRequest,
      })
      setCast(res.cast)
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
                {backend === 'nano_banana'
                  ? 'Builds a cast bible + per-character reference portraits reused across every scene'
                  : 'First image used as visual reference for all subsequent images'}
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
                    disabled={castBusy !== null}
                    className="flex items-center gap-1.5 px-2.5 py-1 rounded text-xs border border-[var(--border)] text-[var(--text-secondary)] hover:text-[var(--accent)] disabled:opacity-50"
                  >
                    {castBusy === 'cast' ? <Loader2 size={11} className="animate-spin" /> : <RefreshCw size={11} />}
                    {cast.length > 0 ? 'Rebuild cast' : 'Build cast'}
                  </button>
                  <button
                    onClick={handleGenerateRefs}
                    disabled={castBusy !== null || cast.length === 0}
                    className="flex items-center gap-1.5 px-2.5 py-1 rounded text-xs bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white disabled:opacity-50"
                  >
                    {castBusy === 'refs' ? <Loader2 size={11} className="animate-spin" /> : <ImageIcon size={11} />}
                    Generate portraits
                  </button>
                </div>
              </div>
              {cast.length === 0 ? (
                <p className="text-[11px] text-[var(--text-muted)]">
                  No cast yet. Build the cast bible (auto-extracted from the script), generate the portraits, then run image generation — each scene reuses its characters' portraits for a consistent look.
                </p>
              ) : (
                <div className="grid grid-cols-3 sm:grid-cols-4 gap-2">
                  {cast.map(member => (
                    <div key={member.id} className="rounded-lg overflow-hidden border border-[var(--border)] bg-[var(--bg-secondary)]">
                      {member.reference_image_path ? (
                        <img
                          src={api.artifactUrl(project.project_id, member.reference_image_path)}
                          alt={member.name}
                          className="w-full aspect-[3/4] object-cover"
                        />
                      ) : (
                        <div className="w-full aspect-[3/4] flex items-center justify-center text-[var(--text-muted)]">
                          <Users size={20} className="opacity-40" />
                        </div>
                      )}
                      <div className="px-1.5 py-1">
                        <div className="text-[11px] font-medium text-[var(--text-primary)] truncate" title={member.name}>{member.name}</div>
                        {member.role && <div className="text-[9px] text-[var(--text-muted)] capitalize truncate">{member.role}</div>}
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
          <div className="flex justify-end">
            <button
              onClick={handleGenerate}
              disabled={generating}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white text-sm font-medium transition-colors disabled:opacity-50"
            >
              {generating ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
              {generating ? 'Generating...' : hasGeneratedImages ? 'Regenerate All' : 'Generate Images'}
            </button>
          </div>
        </div>
      </div>

      {/* Image grid */}
      {hasImageResults && (
        <div className="space-y-4">
          {scenes.map((scene, i) => {
            const paths = (scene.image_paths && scene.image_paths.length > 0)
              ? scene.image_paths
              : scene.image_path ? [scene.image_path] : []
            const prompts = (scene.image_prompts && scene.image_prompts.length > 0)
              ? scene.image_prompts
              : scene.image_prompt ? [scene.image_prompt] : []

            if (paths.length === 0 && !scene.image_error) return null

            return (
              <div key={i} className="rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] overflow-hidden">
                <div className="px-3 py-2 border-b border-[var(--border)] flex items-center justify-between">
                  <span className="text-xs font-medium text-[var(--accent)]">Scene {i + 1}</span>
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] text-[var(--text-muted)]">{paths.length} image{paths.length !== 1 ? 's' : ''}</span>
                    <span className="text-[10px] text-[var(--text-muted)]">{scene.mood}</span>
                    <button
                      onClick={() => handleRegenerateScene(i)}
                      disabled={regeneratingScene === i || generating}
                      className="p-1 rounded text-[var(--text-muted)] hover:text-[var(--accent)] transition-colors disabled:opacity-40"
                      title="Regenerate this scene"
                    >
                      {regeneratingScene === i ? <Loader2 size={12} className="animate-spin" /> : <RotateCcw size={12} />}
                    </button>
                  </div>
                </div>
                <div className="grid grid-cols-4 gap-1 p-1">
                  {paths.map((path, j) => (
                    <div key={j} className="relative group">
                      <img
                        src={api.artifactUrl(project.project_id, path)}
                        alt={`Scene ${i + 1} - Image ${j + 1}`}
                        className="w-full aspect-video object-cover rounded"
                      />
                      {prompts[j] && (
                        <div className="absolute inset-0 bg-black/70 opacity-0 group-hover:opacity-100 transition-opacity rounded p-1.5 overflow-auto">
                          <p className="text-[9px] text-white/80 leading-tight">{prompts[j]}</p>
                        </div>
                      )}
                    </div>
                  ))}
                  {scene.image_error && (
                    <div className="col-span-4 p-3 text-xs text-[var(--error)] text-center">
                      {scene.image_error}
                    </div>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* Next */}
      {hasGeneratedImages && (
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
