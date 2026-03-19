import { useState } from 'react'
import { ImageIcon, Loader2, RefreshCw } from 'lucide-react'
import type { ProjectState } from '../api'
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

  const scenes = project.script?.scenes || []
  const hasImages = scenes.some(s => s.image_path)

  const handleGenerate = async () => {
    setGenerating(true)
    try {
      await api.runImages(project.project_id, { backend, style_prompt: stylePrompt })
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
        <div className="grid grid-cols-3 gap-3">
          {scenes.map((scene, i) => (
            <div key={i} className="rounded-lg border border-[var(--border)] bg-[var(--bg-secondary)] overflow-hidden">
              {scene.image_path ? (
                <img
                  src={api.artifactUrl(project.project_id, scene.image_path)}
                  alt={`Scene ${i + 1}`}
                  className="w-full aspect-video object-cover"
                />
              ) : (
                <div className="w-full aspect-video bg-[var(--bg-tertiary)] flex items-center justify-center">
                  <span className="text-xs text-[var(--text-muted)]">
                    {scene.image_error ? 'Error' : 'No image'}
                  </span>
                </div>
              )}
              <div className="p-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium text-[var(--accent)]">Scene {i + 1}</span>
                  <span className="text-[10px] text-[var(--text-muted)]">{scene.mood}</span>
                </div>
                <p className="text-[10px] text-[var(--text-muted)] mt-1 line-clamp-2">{scene.image_prompt}</p>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Next */}
      {hasImages && (
        <div className="flex justify-end">
          <button
            onClick={onNext}
            className="px-4 py-2 rounded-lg bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white text-sm font-medium transition-colors"
          >
            Next: Video Assembly
          </button>
        </div>
      )}
    </div>
  )
}
