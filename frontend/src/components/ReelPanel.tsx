import { useState, useEffect, useRef, useCallback } from 'react'
import { Clapperboard, Loader2, Download } from 'lucide-react'
import type { ProjectState, ReelProgress } from '../api'
import { api } from '../api'

interface Props {
  project: ProjectState
  onRefresh: () => void
}

const STAGE_LABEL: Record<string, string> = {
  beats: 'Building beats in order (still → Seedance clip → next still)',
  render: 'Cutting to narration + captions',
}

export default function ReelPanel({ project, onRefresh }: Props) {
  const scenes = project.script?.scenes || []
  const voiced = scenes.length > 0 && scenes.every(s => s.audio_path)

  const [progress, setProgress] = useState<ReelProgress | null>(null)
  const [starting, setStarting] = useState(false)
  const pollRef = useRef<number | null>(null)

  const poll = useCallback(async () => {
    const p = await api.reelProgress(project.project_id).catch(() => null)
    if (!p) return
    setProgress(p)
    if (!p.active && pollRef.current) {
      window.clearInterval(pollRef.current)
      pollRef.current = null
      onRefresh()
    }
  }, [project.project_id, onRefresh])

  const startPolling = useCallback(() => {
    if (pollRef.current) window.clearInterval(pollRef.current)
    pollRef.current = window.setInterval(poll, 2000)
  }, [poll])

  useEffect(() => {
    poll().then(() => { if (project.step === 'building_reel') startPolling() })
    return () => { if (pollRef.current) window.clearInterval(pollRef.current) }
  }, [poll, startPolling, project.step])

  const handleBuild = async () => {
    setStarting(true)
    try {
      await api.buildReel(project.project_id)
      onRefresh()
      startPolling()
    } catch (e) {
      alert('Reel build failed to start: ' + (e as Error).message)
    } finally {
      setStarting(false)
    }
  }

  if (!voiced) {
    return (
      <div className="text-center py-16 text-[var(--text-muted)]">
        <Clapperboard size={40} className="mx-auto mb-3 opacity-30" />
        <p>The reel is cut to the narration.</p>
        <p className="text-sm mt-1">Generate the script and voice first, then come back here.</p>
      </div>
    )
  }

  const busy = starting || (progress?.active ?? false) || project.step === 'building_reel'
  const reel = project.reel || progress?.reel || null
  const hook = project.script?.hook
  const cta = project.script?.cta

  return (
    <div className="space-y-4">
      <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] space-y-3">
        <div className="flex items-center justify-between gap-4">
          <div>
            <h3 className="text-sm font-medium text-[var(--text-primary)]">Build reel</h3>
            <p className="text-xs text-[var(--text-muted)] mt-0.5">
              One continuous take: each beat's 9:16 still is an edit of the previous clip's last frame (Nano Banana), then a 5 s motion clip (Seedance 2.5 @ 480p),
              cut to the voiceover with one-word captions. About $0.55 and ~4 min per beat — {scenes.length} beats ≈ ${(scenes.length * 0.55).toFixed(2)}.
              Re-runs regenerate from the first beat whose prompt changed.
            </p>
          </div>
          <button
            onClick={handleBuild}
            disabled={busy}
            className="shrink-0 flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white disabled:opacity-50"
          >
            {busy ? <Loader2 size={14} className="animate-spin" /> : <Clapperboard size={14} />}
            {reel ? 'Rebuild reel' : 'Build reel'}
          </button>
        </div>

        {(hook || cta) && (
          <div className="text-xs text-[var(--text-muted)]">
            {hook && <span>Hook: <span className="text-[var(--text-secondary)]">{hook}</span></span>}
            {hook && cta && <span> · </span>}
            {cta && <span>CTA: <span className="text-[var(--text-secondary)]">{cta}</span></span>}
          </div>
        )}

        {busy && progress && (
          <div className="space-y-1.5">
            <div className="text-xs text-[var(--accent)]">
              {STAGE_LABEL[progress.stage] || 'Working'}…
              {progress.stage !== 'render' && progress.total > 0 && <> {progress.done}/{progress.total}</>}
            </div>
            <div className="h-1.5 rounded-full bg-[var(--bg-tertiary)] overflow-hidden">
              <div
                className="h-full rounded-full bg-[var(--accent)] transition-all duration-700"
                style={{ width: `${progress.stage === 'render' ? 95 : progress.total ? Math.round((progress.done / (progress.total + 1)) * 100) : 5}%` }}
              />
            </div>
          </div>
        )}
        {progress?.error && !busy && (
          <div className="text-xs text-[var(--error)] whitespace-pre-wrap">{progress.error}</div>
        )}
      </div>

      {reel && !busy && (
        <div className="flex gap-4 items-start">
          <video
            src={api.artifactUrl(project.project_id, reel.path)}
            controls
            className="w-64 aspect-[9/16] rounded-xl border border-[var(--border)] bg-black object-contain"
          />
          <div className="text-xs text-[var(--text-muted)] space-y-2">
            <div>{reel.duration}s · 1080×1920</div>
            <a
              href={api.artifactUrl(project.project_id, reel.path)}
              download
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-[var(--border)] text-[var(--text-secondary)] hover:text-[var(--accent)]"
            >
              <Download size={14} /> Download reel.mp4
            </a>
            {scenes.some(s => (s as { reel_clip_error?: string }).reel_clip_error) && (
              <div className="text-[var(--error)]">Some beats fell back to a still (motion clip failed) — rebuild to retry them.</div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
