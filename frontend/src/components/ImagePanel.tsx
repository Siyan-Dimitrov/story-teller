import { useState, useEffect } from 'react'
import { ImageIcon, Loader2, RefreshCw } from 'lucide-react'
import type { ProjectState, LoraInfo } from '../api'
import { api } from '../api'

interface Props {
  project: ProjectState
  onRefresh: () => void
  onNext: () => void
}

const DEFAULT_STYLE = 'dark fairy tale illustration, gothic storybook art, atmospheric, detailed, moody lighting, Tim Burton inspired, rich colors, dramatic shadows'

export default function ImagePanel({ project, onRefresh, onNext }: Props) {
  const [backend, setBackend] = useState(project.image_backend || 'comfyui')
  const [stylePrompt, setStylePrompt] = useState(DEFAULT_STYLE)
  const [generating, setGenerating] = useState(false)
  const [availableLoras, setAvailableLoras] = useState<Record<string, LoraInfo>>({})
  const [selectedLoras, setSelectedLoras] = useState<string[]>([])

  useEffect(() => {
    api.loras().then(data => {
      setAvailableLoras(data.available)
      setSelectedLoras(data.defaults)
    }).catch(() => {})
  }, [])

  const scenes = project.script?.scenes || []
  const hasImages = scenes.some(s => (s.image_paths && s.image_paths.length > 0) || s.image_path)

  const handleGenerate = async () => {
    setGenerating(true)
    try {
      await api.runImages(project.project_id, { backend, style_prompt: stylePrompt, lora_keys: selectedLoras })
      onRefresh()
    } catch (e) {
      alert('Image generation failed: ' + (e as Error).message)
      onRefresh()
    } finally {
      setGenerating(false)
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
          {backend === 'comfyui' && Object.keys(availableLoras).length > 0 && (
            <div>
              <label className="block text-xs text-[var(--text-secondary)] mb-1.5">LoRA Styles (select one or more)</label>
              <div className="flex flex-wrap gap-2">
                {Object.entries(availableLoras).map(([key, lora]) => {
                  const active = selectedLoras.includes(key)
                  return (
                    <button
                      key={key}
                      onClick={() =>
                        setSelectedLoras(prev =>
                          active ? prev.filter(k => k !== key) : [...prev, key]
                        )
                      }
                      className={`px-3 py-1.5 rounded-lg text-xs border transition-colors ${
                        active
                          ? 'border-[var(--accent)] bg-[var(--accent)]/15 text-[var(--accent)]'
                          : 'border-[var(--border)] text-[var(--text-muted)] hover:border-[var(--border-focus)]'
                      }`}
                    >
                      {key.replace(/_/g, ' ')}
                      <span className="ml-1 opacity-60">({lora.trigger})</span>
                    </button>
                  )
                })}
              </div>
            </div>
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
