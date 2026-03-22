import { useState, useEffect, useRef } from 'react'
import { Film, Download, Loader2, Play, X, FolderOpen } from 'lucide-react'
import type { ProjectState } from '../api'
import { api } from '../api'

interface Props {
  project: ProjectState
  onRefresh: () => void
}

export default function VideoPanel({ project, onRefresh }: Props) {
  const [assembling, setAssembling] = useState(false)
  const [progress, setProgress] = useState(0)
  const [phase, setPhase] = useState('')
  const [error, setError] = useState<string | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const isAssembled = project.step === 'assembled'
  const scenes = project.script?.scenes || []
  const hasImages = scenes.some(s => (s.image_paths && s.image_paths.length > 0) || s.image_path)
  const hasAudio = scenes.some(s => s.audio_path)

  // Poll assembly progress
  useEffect(() => {
    // Check if assembly is already in progress on mount
    if (project.step === 'assembling') {
      setAssembling(true)
      startPolling()
    }
    return () => stopPolling()
  }, [project.project_id])

  const startPolling = () => {
    stopPolling()
    pollRef.current = setInterval(async () => {
      try {
        const p = await api.assemblyProgress(project.project_id)
        setProgress(p.progress)
        setPhase(p.phase)

        if (!p.active) {
          stopPolling()
          setAssembling(false)
          if (p.error) {
            setError(p.error)
          } else if (p.phase === 'done') {
            setError(null)
          }
          onRefresh()
        }
      } catch {
        // Backend might be restarting, keep polling
      }
    }, 1500)
  }

  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }

  const handleAssemble = async () => {
    setAssembling(true)
    setProgress(0)
    setPhase('starting')
    setError(null)
    try {
      await api.runAssemble(project.project_id)
      startPolling()
    } catch (e) {
      setError('Failed to start assembly: ' + (e as Error).message)
      setAssembling(false)
    }
  }

  const handleCancel = async () => {
    try {
      await api.cancelAssembly(project.project_id)
      setPhase('cancelling')
    } catch {
      // ignore
    }
  }

  if (!hasImages && !hasAudio) {
    return (
      <div className="text-center py-16 text-[var(--text-muted)]">
        <Film size={40} className="mx-auto mb-3 opacity-30" />
        <p>Generate voice and images first before assembling the video.</p>
      </div>
    )
  }

  const pct = Math.round(progress * 100)

  return (
    <div className="space-y-4">
      {/* Assemble controls */}
      <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)]">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-medium">Video Assembly</h3>
            <p className="text-xs text-[var(--text-muted)] mt-0.5">
              Combines scene images with voice audio using depth parallax or Ken Burns effects
            </p>
          </div>
          {!assembling ? (
            <button
              onClick={handleAssemble}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white text-sm font-medium transition-colors"
            >
              <Film size={14} />
              {isAssembled ? 'Re-assemble' : 'Assemble Video'}
            </button>
          ) : (
            <button
              onClick={handleCancel}
              className="flex items-center gap-2 px-3 py-2 rounded-lg border border-[var(--error)]/50 text-sm text-[var(--error)] hover:bg-[var(--error)]/10 transition-colors"
            >
              <X size={14} />
              Cancel
            </button>
          )}
        </div>

        {/* Progress bar */}
        {assembling && (
          <div className="mt-4 space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="text-[var(--text-secondary)] capitalize">
                {phase === 'encoding'
                  ? `Encoding video... ${pct}%`
                  : phase === 'cancelling'
                    ? 'Cancelling...'
                    : phase || 'Starting...'}
              </span>
              <span className="text-[var(--text-muted)] tabular-nums">{pct}%</span>
            </div>
            <div className="h-2 rounded-full bg-[var(--bg-tertiary)] overflow-hidden">
              <div
                className="h-full rounded-full bg-[var(--accent)] transition-all duration-700 ease-out"
                style={{ width: `${pct}%` }}
              />
            </div>
          </div>
        )}

        {/* Error */}
        {error && !assembling && (
          <div className="mt-3 p-2 rounded-lg border border-[var(--error)]/30 bg-[var(--error)]/5 text-xs text-[var(--error)]">
            {error}
          </div>
        )}
      </div>

      {/* Status */}
      {!hasAudio && (
        <div className="p-3 rounded-lg border border-[var(--warning)]/30 bg-[var(--warning)]/5 text-xs text-[var(--warning)]">
          No audio generated yet — video will use duration hints for scene timing.
        </div>
      )}

      {/* Video preview */}
      {isAssembled && !assembling && (
        <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] overflow-hidden">
          <video
            src={api.downloadUrl(project.project_id)}
            controls
            className="w-full aspect-video bg-black"
          >
            Your browser does not support video playback.
          </video>
          <div className="p-4 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-sm text-[var(--success)]">
                <Play size={14} />
                <span>Video ready!</span>
              </div>
              <a
                href={api.downloadUrl(project.project_id)}
                download={`${project.title || 'story'}.mp4`}
                className="flex items-center gap-2 px-4 py-2 rounded-lg border border-[var(--accent)] text-sm text-[var(--accent)] hover:bg-[var(--accent)]/10 transition-colors"
              >
                <Download size={14} />
                Download MP4
              </a>
            </div>
            {project.output_dir && (
              <div className="flex items-center gap-2 p-2 rounded-lg bg-[var(--bg-tertiary)] text-xs text-[var(--text-secondary)]">
                <FolderOpen size={13} className="text-[var(--accent)] shrink-0" />
                <span className="truncate">Exported to: <span className="text-[var(--text-primary)] font-mono">{project.output_dir}</span></span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Scene breakdown */}
      {scenes.length > 0 && (
        <div>
          <h3 className="text-sm font-medium mb-2">Scene Breakdown</h3>
          <div className="space-y-1">
            {scenes.map((scene, i) => (
              <div
                key={i}
                className="flex items-center gap-3 p-2 rounded text-xs border border-[var(--border)] bg-[var(--bg-secondary)]"
              >
                <span className="text-[var(--accent)] font-medium w-14 shrink-0">Scene {i + 1}</span>
                <span className={`w-16 shrink-0 ${(scene.image_paths?.length || scene.image_path) ? 'text-[var(--success)]' : 'text-[var(--text-muted)]'}`}>
                  {scene.image_paths?.length ? `${scene.image_paths.length} imgs` : scene.image_path ? '1 img' : 'No image'}
                </span>
                <span className={`w-16 shrink-0 ${scene.audio_path ? 'text-[var(--success)]' : 'text-[var(--text-muted)]'}`}>
                  {scene.audio_path ? `${(scene.audio_duration || 0).toFixed(1)}s` : 'No audio'}
                </span>
                <span className="text-[var(--text-muted)] capitalize w-16 shrink-0">{(scene.kb_effect || 'auto').replace('_', ' ')}</span>
                <span className="text-[var(--text-muted)] truncate flex-1">{scene.mood}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
