import { useState, useEffect, useRef, useCallback } from 'react'
import { Scissors, Loader2, Wand2, Download, Film, Check } from 'lucide-react'
import type { ProjectState, ShortSuggestion, ShortItem, ShortsProgress } from '../api'
import { api } from '../api'

interface Props {
  project: ProjectState
}

export default function ShortsPanel({ project }: Props) {
  const scenes = project.script?.scenes || []
  const hasAudio = scenes.some(s => s.audio_path)
  const hasImages = scenes.some(s => (s.image_paths && s.image_paths.length > 0) || s.image_path)

  const [suggestions, setSuggestions] = useState<ShortSuggestion[]>([])
  const [selected, setSelected] = useState<Set<number>>(new Set())
  const [hooks, setHooks] = useState<Record<number, string>>({})
  const [suggesting, setSuggesting] = useState(false)
  const [rendering, setRendering] = useState(false)
  const [progress, setProgress] = useState<ShortsProgress | null>(null)
  const [shorts, setShorts] = useState<ShortItem[]>([])
  const pollRef = useRef<number | null>(null)

  const refreshList = useCallback(async () => {
    try {
      const res = await api.listShorts(project.project_id)
      setShorts(res.shorts || [])
      setProgress(res.progress)
      return res.progress
    } catch {
      return null
    }
  }, [project.project_id])

  useEffect(() => {
    refreshList()
    return () => { if (pollRef.current) window.clearInterval(pollRef.current) }
  }, [refreshList])

  const startPolling = useCallback(() => {
    if (pollRef.current) window.clearInterval(pollRef.current)
    pollRef.current = window.setInterval(async () => {
      const p = await api.shortsProgress(project.project_id).catch(() => null)
      if (p) {
        setProgress(p)
        if (!p.active) {
          if (pollRef.current) window.clearInterval(pollRef.current)
          pollRef.current = null
          setRendering(false)
          refreshList()
        }
      }
    }, 2000)
  }, [project.project_id, refreshList])

  const handleSuggest = async () => {
    setSuggesting(true)
    try {
      const res = await api.suggestShorts(project.project_id, {})
      setSuggestions(res.suggestions)
      setSelected(new Set(res.suggestions.map(s => s.scene_index)))
      setHooks(Object.fromEntries(res.suggestions.map(s => [s.scene_index, s.hook])))
    } catch (e) {
      alert('Suggestion failed: ' + (e as Error).message)
    } finally {
      setSuggesting(false)
    }
  }

  const toggle = (idx: number) => {
    setSelected(prev => {
      const next = new Set(prev)
      if (next.has(idx)) next.delete(idx); else next.add(idx)
      return next
    })
  }

  const handleRender = async (auto: boolean) => {
    setRendering(true)
    try {
      const body = auto
        ? {}
        : { scene_indices: Array.from(selected).sort((a, b) => a - b), hooks }
      await api.renderShorts(project.project_id, body)
      startPolling()
    } catch (e) {
      alert('Render failed: ' + (e as Error).message)
      setRendering(false)
    }
  }

  if (!project.script) {
    return (
      <div className="text-center py-16 text-[var(--text-muted)]">
        <Scissors size={40} className="mx-auto mb-3 opacity-30" />
        <p>Generate a script first.</p>
      </div>
    )
  }

  if (!hasAudio || !hasImages) {
    return (
      <div className="text-center py-16 text-[var(--text-muted)]">
        <Scissors size={40} className="mx-auto mb-3 opacity-30" />
        <p>Shorts reuse each scene's narration and images.</p>
        <p className="text-sm mt-1">Generate voice and images first, then come back here.</p>
      </div>
    )
  }

  const isBusy = rendering || (progress?.active ?? false)

  return (
    <div className="space-y-4">
      <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-medium text-[var(--text-primary)]">Create Shorts</h3>
            <p className="text-xs text-[var(--text-muted)] mt-0.5">
              Cuts the best scenes out of your finished video and reframes them to vertical 9:16 with burned-in captions (keeps the music &amp; motion). An LLM picks the most self-contained, dramatic scenes.
            </p>
            {project.step !== 'assembled' && (
              <p className="text-[11px] text-[var(--text-muted)] mt-1 opacity-80">
                Tip: assemble the full video first so shorts include music &amp; motion. Until then they're rendered from the scene's stills.
              </p>
            )}
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={handleSuggest}
              disabled={suggesting || isBusy}
              className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm border border-[var(--border)] text-[var(--text-secondary)] hover:text-[var(--accent)] disabled:opacity-50"
            >
              {suggesting ? <Loader2 size={14} className="animate-spin" /> : <Wand2 size={14} />}
              Suggest scenes
            </button>
            <button
              onClick={() => handleRender(selected.size === 0)}
              disabled={isBusy}
              className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white disabled:opacity-50"
            >
              {isBusy ? <Loader2 size={14} className="animate-spin" /> : <Scissors size={14} />}
              {selected.size > 0 ? `Render ${selected.size} selected` : 'Auto-render top picks'}
            </button>
          </div>
        </div>

        {isBusy && progress && (
          <div className="text-xs text-[var(--accent)]">
            Rendering shorts… {progress.done}/{progress.total || '?'}
          </div>
        )}
        {progress?.error && (
          <div className="text-xs text-[var(--error)]">{progress.error}</div>
        )}

        {suggestions.length > 0 && (
          <div className="space-y-2">
            {suggestions.map(s => {
              const scene = scenes[s.scene_index]
              const isSel = selected.has(s.scene_index)
              return (
                <div
                  key={s.scene_index}
                  className={`flex gap-3 p-2 rounded-lg border ${isSel ? 'border-[var(--accent)]/50 bg-[var(--bg-tertiary)]' : 'border-[var(--border)]'}`}
                >
                  <button
                    onClick={() => toggle(s.scene_index)}
                    className={`shrink-0 w-5 h-5 rounded border flex items-center justify-center mt-0.5 ${isSel ? 'bg-[var(--accent)] border-[var(--accent)]' : 'border-[var(--border)]'}`}
                  >
                    {isSel && <Check size={12} className="text-white" />}
                  </button>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 text-xs text-[var(--text-muted)]">
                      <span className="text-[var(--accent)] font-medium">Scene {s.scene_index + 1}</span>
                      <span>·</span>
                      <span>{scene?.mood}</span>
                      {scene?.audio_duration ? <><span>·</span><span>{Math.round(scene.audio_duration)}s</span></> : null}
                      <span>·</span>
                      <span title={s.reason}>score {s.score}</span>
                    </div>
                    <input
                      value={hooks[s.scene_index] ?? ''}
                      onChange={e => setHooks(h => ({ ...h, [s.scene_index]: e.target.value }))}
                      placeholder="Hook line shown at the top of the short"
                      className="w-full mt-1 bg-[var(--bg-tertiary)] border border-[var(--border)] rounded px-2 py-1 text-xs text-[var(--text-primary)] focus:outline-none focus:border-[var(--border-focus)]"
                    />
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>

      {shorts.length > 0 && (
        <div>
          <h3 className="text-sm font-medium text-[var(--text-primary)] mb-2">Rendered shorts</h3>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
            {shorts.map((sh, i) => (
              <div key={i} className="rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] overflow-hidden">
                <video
                  src={api.artifactUrl(project.project_id, sh.path)}
                  controls
                  className="w-full aspect-[9/16] bg-black object-contain"
                />
                <div className="px-2 py-1.5 flex items-center justify-between">
                  <div className="min-w-0">
                    <div className="text-[11px] text-[var(--text-primary)]">Scene {sh.scene_index + 1}</div>
                    <div className="text-[9px] text-[var(--text-muted)]">{sh.duration}s</div>
                  </div>
                  <a
                    href={api.artifactUrl(project.project_id, sh.path)}
                    download
                    className="p-1.5 rounded text-[var(--text-muted)] hover:text-[var(--accent)]"
                    title="Download"
                  >
                    <Download size={14} />
                  </a>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {shorts.length === 0 && !isBusy && (
        <div className="text-center py-8 text-[var(--text-muted)] text-sm">
          <Film size={28} className="mx-auto mb-2 opacity-30" />
          No shorts yet. Suggest scenes or auto-render top picks above.
        </div>
      )}
    </div>
  )
}
