import { useState, useEffect, useRef } from 'react'
import { Film, Download, Play, X, FolderOpen, Music } from 'lucide-react'
import type { ProjectState, MusicTrack } from '../api'
import { api } from '../api'

interface Props {
  project: ProjectState
  onRefresh: () => void
}

type MusicMode = 'auto' | 'local' | 'none'

function initialMode(value: string | null | undefined): MusicMode {
  if (!value || value === 'auto') return 'auto'
  if (value === 'none') return 'none'
  return 'local'
}

export default function VideoPanel({ project, onRefresh }: Props) {
  const [assembling, setAssembling] = useState(false)
  const [progress, setProgress] = useState(0)
  const [phase, setPhase] = useState('')
  const [error, setError] = useState<string | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const [musicMode, setMusicMode] = useState<MusicMode>(initialMode(project.background_music))
  const [localTracks, setLocalTracks] = useState<MusicTrack[]>([])
  const [localTrackPath, setLocalTrackPath] = useState<string>(
    project.background_music && project.background_music !== 'auto' && project.background_music !== 'none'
      ? project.background_music
      : '',
  )
  const [musicVolume, setMusicVolume] = useState<number>(project.music_volume ?? 0.15)

  useEffect(() => {
    api.listLocalMusic().then(setLocalTracks).catch(() => setLocalTracks([]))
  }, [])

  const persistMusic = async (patch: { background_music?: string | null; music_volume?: number }) => {
    try {
      await api.setProjectMusic(project.project_id, patch)
    } catch {
      // ignore; will be reapplied on next assemble call
    }
  }

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

  const resolveMusicPayload = (): { background_music: string | null; music_volume: number } => {
    if (musicMode === 'auto') return { background_music: 'auto', music_volume: musicVolume }
    if (musicMode === 'none') return { background_music: 'none', music_volume: musicVolume }
    return { background_music: localTrackPath || 'none', music_volume: musicVolume }
  }

  const handleAssemble = async () => {
    setAssembling(true)
    setProgress(0)
    setPhase('starting')
    setError(null)
    try {
      await api.runAssemble(project.project_id, resolveMusicPayload())
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

      {/* Background music controls */}
      <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)]">
        <div className="flex items-center gap-2 mb-3">
          <Music size={14} className="text-[var(--accent)]" />
          <h3 className="text-sm font-medium">Background Music</h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div>
            <label className="block text-xs text-[var(--text-muted)] mb-1">Source</label>
            <select
              value={musicMode}
              onChange={async (e) => {
                const mode = e.target.value as MusicMode
                setMusicMode(mode)
                const bg = mode === 'local' ? localTrackPath || 'none' : mode
                await persistMusic({ background_music: bg })
              }}
              className="w-full px-2 py-1.5 rounded-lg bg-[var(--bg-tertiary)] border border-[var(--border)] text-sm"
            >
              <option value="auto">Auto (tone-based)</option>
              <option value="local">Local track</option>
              <option value="none">None</option>
            </select>
          </div>
          {musicMode === 'local' && (
            <div>
              <label className="block text-xs text-[var(--text-muted)] mb-1">
                Track {localTracks.length === 0 && '(drop files into data/music/)'}
              </label>
              <select
                value={localTrackPath}
                onChange={async (e) => {
                  const path = e.target.value
                  setLocalTrackPath(path)
                  await persistMusic({ background_music: path || 'none' })
                }}
                className="w-full px-2 py-1.5 rounded-lg bg-[var(--bg-tertiary)] border border-[var(--border)] text-sm"
              >
                <option value="">— Select —</option>
                {localTracks.map(t => (
                  <option key={t.id} value={t.path || ''}>{t.title}</option>
                ))}
              </select>
            </div>
          )}
          <div className={musicMode === 'local' ? '' : 'md:col-span-2'}>
            <label className="flex items-center justify-between text-xs text-[var(--text-muted)] mb-1">
              <span>Volume</span>
              <span className="tabular-nums">{Math.round(musicVolume * 100)}%</span>
            </label>
            <input
              type="range"
              min={0}
              max={0.5}
              step={0.01}
              value={musicVolume}
              onChange={e => setMusicVolume(parseFloat(e.target.value))}
              onMouseUp={() => persistMusic({ music_volume: musicVolume })}
              onTouchEnd={() => persistMusic({ music_volume: musicVolume })}
              className="w-full accent-[var(--accent)]"
            />
          </div>
        </div>
        {project.selected_music && musicMode === 'auto' && (
          <div className="mt-3 text-xs text-[var(--text-secondary)]">
            ♪ {project.selected_music.title}
            {project.selected_music.artist ? ` — ${project.selected_music.artist}` : ''}
            {' '}({project.selected_music.source})
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
