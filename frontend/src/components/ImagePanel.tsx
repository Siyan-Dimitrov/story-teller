import { useState, useEffect, useRef } from 'react'
import { ImageIcon, Loader2, RefreshCw, ChevronDown, RotateCcw } from 'lucide-react'
import type { ProjectState, LoraInfo } from '../api'
import { api } from '../api'

interface Props {
  project: ProjectState
  onRefresh: () => void
  onNext: () => void
}

const DEFAULT_STYLE = 'dark fairy tale illustration, gothic storybook art, atmospheric, detailed, moody lighting, Tim Burton inspired, rich colors, dramatic shadows'

function LoraDropdown({
  label,
  sublabel,
  value,
  onChange,
  entries,
  disabledKey,
}: {
  label: string
  sublabel?: string
  value: string
  onChange: (v: string) => void
  entries: [string, LoraInfo][]
  disabledKey?: string
}) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const selected = entries.find(([k]) => k === value)
  const displayName = selected ? selected[0].replace(/_/g, ' ') : 'None'

  return (
    <div className="flex-1" ref={ref}>
      <label className="block text-xs text-[var(--text-secondary)] mb-1.5">
        {label}
        {sublabel && <span className="ml-1 text-[var(--text-muted)]">{sublabel}</span>}
      </label>
      <div className="relative">
        <button
          type="button"
          onClick={() => setOpen(!open)}
          className="w-full flex items-center justify-between bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--border-focus)] text-left"
        >
          <span className="capitalize truncate">{displayName}</span>
          <ChevronDown size={14} className={`ml-2 transition-transform ${open ? 'rotate-180' : ''}`} />
        </button>
        {open && (
          <div className="absolute z-50 mt-1 w-full max-h-64 overflow-y-auto bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg shadow-lg">
            <button
              type="button"
              onClick={() => { onChange(''); setOpen(false) }}
              className={`w-full text-left px-3 py-2 text-sm hover:bg-[var(--bg-secondary)] ${!value ? 'text-[var(--accent)]' : 'text-[var(--text-primary)]'}`}
            >
              None
            </button>
            {entries.map(([key, lora]) => {
              const disabled = key === disabledKey
              return (
                <button
                  key={key}
                  type="button"
                  disabled={disabled}
                  onClick={() => { onChange(key); setOpen(false) }}
                  className={`w-full text-left px-3 py-2 hover:bg-[var(--bg-secondary)] ${disabled ? 'opacity-40 cursor-not-allowed' : ''} ${key === value ? 'text-[var(--accent)]' : 'text-[var(--text-primary)]'}`}
                >
                  <div className="text-sm capitalize">{key.replace(/_/g, ' ')}</div>
                  {lora.description && (
                    <div className="text-[10px] text-[var(--text-muted)] mt-0.5 leading-tight">{lora.description}</div>
                  )}
                </button>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}

export default function ImagePanel({ project, onRefresh, onNext }: Props) {
  const [backend, setBackend] = useState(project.image_backend || 'comfyui')
  const [stylePrompt, setStylePrompt] = useState(DEFAULT_STYLE)
  const [generating, setGenerating] = useState(false)
  const [availableLoras, setAvailableLoras] = useState<Record<string, LoraInfo>>({})
  const [primaryLora, setPrimaryLora] = useState('')
  const [secondaryLora, setSecondaryLora] = useState('')
  const [characterConsistency, setCharacterConsistency] = useState(false)
  const [regeneratingScene, setRegeneratingScene] = useState<number | null>(null)

  useEffect(() => {
    api.loras().then(data => {
      setAvailableLoras(data.available)
      if (data.defaults.length > 0) setPrimaryLora(data.defaults[0])
      if (data.defaults.length > 1) setSecondaryLora(data.defaults[1])
    }).catch(() => {})
  }, [])

  const scenes = project.script?.scenes || []
  const hasImages = scenes.some(s => (s.image_paths && s.image_paths.length > 0) || s.image_path)

  const handleGenerate = async () => {
    setGenerating(true)
    try {
      const lora_keys = [primaryLora, secondaryLora].filter(Boolean)
      await api.runImages(project.project_id, {
        backend,
        style_prompt: stylePrompt,
        lora_keys,
        ...(characterConsistency && backend === 'replicate' ? { character_consistency: true } : {}),
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
      const lora_keys = [primaryLora, secondaryLora].filter(Boolean)
      await api.regenerateSceneImages(project.project_id, sceneIndex, {
        backend,
        style_prompt: stylePrompt,
        lora_keys,
        character_consistency: characterConsistency && backend === 'replicate',
      })
      onRefresh()
    } catch (e) {
      alert('Scene regeneration failed: ' + (e as Error).message)
    } finally {
      setRegeneratingScene(null)
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
                <option value="comfyui">ComfyUI (Local)</option>
                <option value="replicate">Replicate Flux (Cloud)</option>
                <option value="gpt_image">GPT Image 2 (OpenAI)</option>
                <option value="ollama">Ollama (Placeholder)</option>
              </select>
            </div>
            <div className="flex-1">
              <label className="block text-xs text-[var(--text-secondary)] mb-1.5">Style Prompt (prepended to each scene)</label>
              <input
                type="text"
                value={stylePrompt}
                onChange={e => setStylePrompt(e.target.value)}
                className="w-full bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--border-focus)]"
              />
            </div>
          </div>
          {(backend === 'comfyui' || backend === 'replicate') && Object.keys(availableLoras).length > 0 && (() => {
            const loraEntries = Object.entries(availableLoras).filter(([, lora]) =>
              backend === 'replicate' ? lora.has_flux : true
            ) as [string, LoraInfo][]
            if (loraEntries.length === 0) return null
            return (
              <div className="flex items-end gap-4">
                <LoraDropdown
                  label="Primary LoRA Style"
                  sublabel={backend === 'replicate' ? '-- FLUX LoRA' : undefined}
                  value={primaryLora}
                  onChange={setPrimaryLora}
                  entries={loraEntries}
                  disabledKey={secondaryLora}
                />
                <LoraDropdown
                  label="Secondary LoRA Style"
                  sublabel="-- optional"
                  value={secondaryLora}
                  onChange={setSecondaryLora}
                  entries={loraEntries}
                  disabledKey={primaryLora}
                />
              </div>
            )
          })()}
          {backend === 'replicate' && (
            <label className="flex items-center gap-2 text-sm text-[var(--text-secondary)] cursor-pointer">
              <input
                type="checkbox"
                checked={characterConsistency}
                onChange={e => setCharacterConsistency(e.target.checked)}
                className="rounded border-[var(--border)]"
              />
              <span>Character Consistency</span>
              <span className="text-[10px] text-[var(--text-muted)]">First image used as visual reference for all subsequent images</span>
            </label>
          )}
          {backend === 'gpt_image' && (
            <p className="text-xs text-[var(--text-muted)]">
              Uses <code>OPENAI_API_KEY</code> from the backend <code>.env</code> file. LoRA selections are ignored; style is taken from the prompt.
            </p>
          )}
          <div className="flex justify-end">
            <button
              onClick={handleGenerate}
              disabled={generating}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white text-sm font-medium transition-colors disabled:opacity-50"
            >
              {generating ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
              {generating ? 'Generating...' : hasImages ? 'Regenerate All' : 'Generate Images'}
            </button>
          </div>
        </div>
      </div>

      {/* Image grid */}
      {hasImages && (
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
                  {scene.image_error && paths.length === 0 && (
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
      {hasImages && (
        <div className="flex justify-end">
          <button
            onClick={onNext}
            className="px-4 py-2 rounded-lg bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white text-sm font-medium transition-colors"
          >
            Next: Quality Check
          </button>
        </div>
      )}
    </div>
  )
}
