import { useState, useEffect, useRef } from 'react'
import { Film, Download, Loader2, Play, Pause, X, FolderOpen, Music, Volume2, Search, Wand2, Trash2 } from 'lucide-react'
import type { ProjectState, MusicTrack } from '../api'
import { api } from '../api'

interface Props {
  project: ProjectState
  onRefresh: () => void
}

const MOOD_PRESETS = [
  { label: 'Cinematic', query: 'cinematic background' },
  { label: 'Dark / Gothic', query: 'dark gothic ambient' },
  { label: 'Orchestral', query: 'orchestral epic background' },
  { label: 'Suspense', query: 'suspense tension background' },
  { label: 'Fantasy', query: 'fantasy magical background' },
  { label: 'Peaceful', query: 'peaceful ambient background' },
  { label: 'Mysterious', query: 'mysterious atmospheric background' },
  { label: 'Horror', query: 'horror dark ambient background' },
  { label: 'Epic', query: 'epic dramatic background' },
]

type MusicSource = 'local' | 'jamendo'

export default function VideoPanel({ project, onRefresh }: Props) {
  const [assembling, setAssembling] = useState(false)
  const [progress, setProgress] = useState(0)
  const [phase, setPhase] = useState('')
  const [error, setError] = useState<string | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Music controls
  const [musicTracks, setMusicTracks] = useState<MusicTrack[]>([])
  const [selectedTrack, setSelectedTrack] = useState<string>('')
  const [musicVolume, setMusicVolume] = useState(0.18)
  const [musicLoading, setMusicLoading] = useState(false)

  // Jamendo search
  const [musicSource, setMusicSource] = useState<MusicSource>('jamendo')
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<MusicTrack[]>([])
  const [searching, setSearching] = useState(false)
  const [jamendoEnabled, setJamendoEnabled] = useState(false)
  const [downloadingTrack, setDownloadingTrack] = useState<string | null>(null)

  // Audio preview
  const [playingTrackId, setPlayingTrackId] = useState<string | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)

  // Per-scene music
  const [sceneSuggestions, setSceneSuggestions] = useState<Record<number, { query: string; reasoning: string; tracks: MusicTrack[]; assignedTrack?: string | null }>>({})
  const [suggestingAll, setSuggestingAll] = useState(false)

  const isAssembled = project.step === 'assembled'
  const scenes = project.script?.scenes || []
  const hasImages = scenes.some(s => (s.image_paths && s.image_paths.length > 0) || s.image_path)
  const hasAudio = scenes.some(s => s.audio_path)

  const saveTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Clean up audio on unmount
  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause()
        audioRef.current = null
      }
    }
  }, [])

  // Load available music on mount
  useEffect(() => {
    setMusicLoading(true)
    api.music()
      .then(data => {
        setMusicTracks(data.available)
        setJamendoEnabled(data.jamendo_enabled)

        // Restore persisted volume or use backend default
        const persistedVolume = project.music_volume
        setMusicVolume(typeof persistedVolume === 'number' ? persistedVolume : data.default_volume)

        // Try to restore persisted track
        const persistedTrack = project.music_track
        if (persistedTrack) {
          const matched = data.available.find(t =>
            t.name === persistedTrack || t.path === persistedTrack || t.id === persistedTrack
          )
          if (matched) {
            setSelectedTrack(matched.id)
            setMusicSource('local')
            return
          }
        }

        // Fallback defaults
        if (data.available.length > 0) {
          setSelectedTrack(data.available[0].id)
          setMusicSource('local')
        } else if (data.jamendo_enabled) {
          setMusicSource('jamendo')
        }
      })
      .catch(() => {})
      .finally(() => setMusicLoading(false))
  }, [project.project_id])

  // Auto-save music preferences when changed (debounced)
  useEffect(() => {
    if (saveTimeoutRef.current) clearTimeout(saveTimeoutRef.current)
    saveTimeoutRef.current = setTimeout(() => {
      const track = musicTracks.find(t => t.id === selectedTrack)
      const musicTrack = track ? (track.name || track.path || track.id) : null
      api.updateSettings(project.project_id, {
        music_track: musicTrack,
        music_volume: musicVolume,
      }).catch(() => {})
    }, 800)
    return () => {
      if (saveTimeoutRef.current) clearTimeout(saveTimeoutRef.current)
    }
  }, [selectedTrack, musicVolume, musicTracks, project.project_id])

  // Poll assembly progress
  useEffect(() => {
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

  const handleSearch = async (query: string) => {
    if (!query.trim() || !jamendoEnabled) return
    setSearching(true)
    setSearchQuery(query)
    try {
      const data = await api.musicSearch(query, 8)
      setSearchResults(data.results)
    } catch (e) {
      setError('Music search failed: ' + (e as Error).message)
    } finally {
      setSearching(false)
    }
  }

  const handleDownloadTrack = async (track: MusicTrack) => {
    if (!track.url) return
    setDownloadingTrack(track.id)
    try {
      const cached = await api.musicDownload(track.url)
      // Add the newly cached track to local list and select it
      const newTrack: MusicTrack = {
        ...track,
        id: `local_${cached.name.replace(/\.[^.]+$/, '')}`,
        name: cached.name,
        path: cached.path,
        source: 'local',
        size_bytes: cached.size_bytes,
      }
      setMusicTracks(prev => [...prev, newTrack])
      setSelectedTrack(newTrack.id)
      setMusicSource('local')
      setSearchResults([])
    } catch (e) {
      setError('Failed to download music track: ' + (e as Error).message)
    } finally {
      setDownloadingTrack(null)
    }
  }

  const handleAssemble = async () => {
    setAssembling(true)
    setProgress(0)
    setPhase('starting')
    setError(null)
    try {
      // If a Jamendo track is selected but not yet downloaded, download it first
      const track = searchResults.find(t => t.id === selectedTrack)
      if (musicSource === 'jamendo' && track && track.source === 'jamendo') {
        await handleDownloadTrack(track)
      }

      // Resolve selected track to a filename/path the backend understands
      const finalTrack = musicTracks.find(t => t.id === selectedTrack)
      const musicTrack = finalTrack ? (finalTrack.name || finalTrack.path || finalTrack.id) : undefined

      await api.runAssemble(project.project_id, {
        ...(musicTrack ? { music_track: musicTrack } : {}),
        music_volume: musicVolume,
      })
      startPolling()
    } catch (e) {
      setError('Failed to start assembly: ' + (e as Error).message)
      setAssembling(false)
    }
  }

  const togglePreview = (track: MusicTrack) => {
    const isPlaying = playingTrackId === track.id
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current = null
    }
    if (isPlaying) {
      setPlayingTrackId(null)
      return
    }
    const url = track.source === 'jamendo'
      ? track.url
      : `/music/${encodeURIComponent(track.name || '')}`
    if (!url) return
    const audio = new Audio(url)
    audio.onended = () => setPlayingTrackId(null)
    audio.onerror = () => setPlayingTrackId(null)
    audio.play().catch(() => {})
    audioRef.current = audio
    setPlayingTrackId(track.id)
  }

  const handleCancel = async () => {
    try {
      await api.cancelAssembly(project.project_id)
      setPhase('cancelling')
    } catch {
      // ignore
    }
  }

  const handleSuggestAll = async () => {
    setSuggestingAll(true)
    try {
      const data = await api.suggestMusic(project.project_id)
      const map: Record<number, { query: string; reasoning: string; tracks: MusicTrack[]; assignedTrack?: string | null }> = {}
      for (const s of data.scenes) {
        map[s.scene_index] = {
          query: s.query,
          reasoning: s.reasoning,
          tracks: s.tracks,
          assignedTrack: s.assigned_track ?? null,
        }
      }
      setSceneSuggestions(map)

      // The backend downloaded the top tracks into data/music/ and wrote them into
      // script.json as scene.music_track. Refresh the local music list so the newly
      // cached files appear as options, and reload the project so the dropdowns
      // pick up the auto-assigned values.
      try {
        const music = await api.music()
        setMusicTracks(music.available)
      } catch {}
      onRefresh()
    } catch (e) {
      setError('Music suggestion failed: ' + (e as Error).message)
    } finally {
      setSuggestingAll(false)
    }
  }

  const handleSetSceneMusic = async (sceneIndex: number, trackName: string | null) => {
    try {
      await api.updateSceneMusic(project.project_id, sceneIndex, {
        music_track: trackName,
        music_volume: null,
      })
      onRefresh()
    } catch (e) {
      setError('Failed to update scene music: ' + (e as Error).message)
    }
  }

  const handleClearAllSceneMusic = async () => {
    for (let i = 0; i < scenes.length; i++) {
      if (scenes[i].music_track) {
        await api.updateSceneMusic(project.project_id, i, { music_track: null, music_volume: null })
      }
    }
    setSceneSuggestions({})
    onRefresh()
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

        {/* Music controls */}
        {!assembling && (
          <div className="mt-4 pt-4 border-t border-[var(--border)] space-y-3">
            <div className="flex items-center gap-2">
              <Music size={14} className="text-[var(--accent)]" />
              <span className="text-xs font-medium text-[var(--text-secondary)]">Background Music</span>
            </div>

            {/* Source tabs */}
            <div className="flex gap-2">
              <button
                onClick={() => setMusicSource('local')}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                  musicSource === 'local'
                    ? 'bg-[var(--accent)] text-white'
                    : 'bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                }`}
              >
                Local ({musicTracks.length})
              </button>
              <button
                onClick={() => setMusicSource('jamendo')}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                  musicSource === 'jamendo'
                    ? 'bg-[var(--accent)] text-white'
                    : 'bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                }`}
              >
                Jamendo Search
              </button>
            </div>

            {musicSource === 'local' && (
              <div className="space-y-3">
                <div className="flex items-end gap-4">
                  <div className="flex-1">
                    <label className="block text-[10px] text-[var(--text-muted)] mb-1">Track</label>
                    <div className="flex items-center gap-2">
                    <select
                      value={selectedTrack}
                      onChange={e => setSelectedTrack(e.target.value)}
                      disabled={musicLoading}
                      className="flex-1 bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--border-focus)]"
                    >
                      {musicTracks.length === 0 && (
                        <option value="">No local music tracks</option>
                      )}
                      {musicTracks.map(t => (
                        <option key={t.id} value={t.id}>
                          {t.title}
                          {t.artist && t.artist !== 'Unknown' ? ` — ${t.artist}` : ''}
                        </option>
                      ))}
                    </select>
                    {(() => {
                      const track = musicTracks.find(t => t.id === selectedTrack)
                      if (!track) return null
                      return (
                        <button
                          onClick={() => togglePreview(track)}
                          className="p-2 rounded-lg border border-[var(--border)] hover:border-[var(--accent)] text-[var(--text-secondary)] hover:text-[var(--accent)] transition-colors shrink-0"
                          title="Preview track"
                        >
                          {playingTrackId === track.id ? <Pause size={14} /> : <Play size={14} />}
                        </button>
                      )
                    })()}
                    </div>
                  </div>
                  <div className="w-40">
                    <label className="block text-[10px] text-[var(--text-muted)] mb-1">
                      Volume: {Math.round(musicVolume * 100)}%
                    </label>
                    <div className="flex items-center gap-2">
                      <Volume2 size={14} className="text-[var(--text-muted)]" />
                      <input
                        type="range"
                        min={0}
                        max={1}
                        step={0.01}
                        value={musicVolume}
                        onChange={e => setMusicVolume(parseFloat(e.target.value))}
                        className="flex-1 accent-[var(--accent)]"
                      />
                    </div>
                  </div>
                </div>
                {musicTracks.length === 0 && (
                  <p className="text-[10px] text-[var(--text-muted)]">
                    Drop .mp3 / .wav files into <code className="text-[var(--text-secondary)]">data/music/</code> to add local tracks, or switch to Jamendo Search above.
                  </p>
                )}
              </div>
            )}

            {musicSource === 'jamendo' && (
              <div className="space-y-3">
                {!jamendoEnabled && (
                  <div className="p-2 rounded-lg border border-[var(--warning)]/30 bg-[var(--warning)]/5 text-xs text-[var(--warning)]">
                    Jamendo is not configured on the backend. Set JAMENDO_CLIENT_ID to enable royalty-free music search.
                  </div>
                )}

                {/* Mood presets */}
                <div>
                  <label className="block text-[10px] text-[var(--text-muted)] mb-1.5">Quick mood presets</label>
                  <div className="flex flex-wrap gap-1.5">
                    {MOOD_PRESETS.map(preset => (
                      <button
                        key={preset.label}
                        onClick={() => handleSearch(preset.query)}
                        disabled={searching || !jamendoEnabled}
                        className="px-2.5 py-1 rounded-md bg-[var(--bg-tertiary)] border border-[var(--border)] text-[10px] text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:border-[var(--accent)] transition-colors disabled:opacity-50"
                      >
                        {preset.label}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Custom search */}
                <div className="flex gap-2">
                  <div className="flex-1 relative">
                    <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--text-muted)]" />
                    <input
                      type="text"
                      value={searchQuery}
                      onChange={e => setSearchQuery(e.target.value)}
                      onKeyDown={e => e.key === 'Enter' && handleSearch(searchQuery)}
                      placeholder="Search Jamendo for instrumental music..."
                      disabled={searching || !jamendoEnabled}
                      className="w-full pl-9 pr-3 py-2 bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg text-sm text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none focus:border-[var(--border-focus)]"
                    />
                  </div>
                  <button
                    onClick={() => handleSearch(searchQuery)}
                    disabled={searching || !searchQuery.trim() || !jamendoEnabled}
                    className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white text-xs font-medium transition-colors disabled:opacity-50"
                  >
                    {searching ? <Loader2 size={12} className="animate-spin" /> : <Search size={12} />}
                    Search
                  </button>
                </div>

                {/* Search results */}
                {searchResults.length > 0 && (
                  <div className="space-y-1.5 max-h-56 overflow-y-auto">
                    <label className="block text-[10px] text-[var(--text-muted)]">Search results — click to download & select</label>
                    {searchResults.map(track => (
                      <div
                        key={track.id}
                        onClick={() => {
                          setSelectedTrack(track.id)
                        }}
                        className={`flex items-center justify-between p-2 rounded-lg border cursor-pointer transition-colors ${
                          selectedTrack === track.id
                            ? 'border-[var(--accent)] bg-[var(--accent)]/10'
                            : 'border-[var(--border)] bg-[var(--bg-tertiary)] hover:border-[var(--accent)]/50'
                        }`}
                      >
                        <div className="min-w-0 flex items-center gap-2">
                          <button
                            onClick={e => {
                              e.stopPropagation()
                              togglePreview(track)
                            }}
                            className="p-1.5 rounded-md border border-[var(--border)] hover:border-[var(--accent)] text-[var(--text-secondary)] hover:text-[var(--accent)] transition-colors shrink-0"
                            title="Preview track"
                          >
                            {playingTrackId === track.id ? <Pause size={12} /> : <Play size={12} />}
                          </button>
                          <div className="min-w-0">
                            <div className="text-xs text-[var(--text-primary)] truncate">{track.title}</div>
                            <div className="text-[10px] text-[var(--text-muted)]">
                              {track.artist} • {Math.round((track.duration || 0) / 60)}:{String(Math.round((track.duration || 0) % 60)).padStart(2, '0')}
                            </div>
                          </div>
                        </div>
                        {downloadingTrack === track.id ? (
                          <span className="flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded bg-[var(--bg-secondary)] text-[var(--text-muted)] shrink-0 ml-2">
                            <Loader2 size={8} className="animate-spin" /> Downloading
                          </span>
                        ) : selectedTrack === track.id && (
                          <span className="text-[10px] px-1.5 py-0.5 rounded bg-[var(--accent)] text-white shrink-0 ml-2">Selected</span>
                        )}
                      </div>
                    ))}
                  </div>
                )}

                {/* Volume slider for Jamendo too */}
                <div className="w-40">
                  <label className="block text-[10px] text-[var(--text-muted)] mb-1">
                    Volume: {Math.round(musicVolume * 100)}%
                  </label>
                  <div className="flex items-center gap-2">
                    <Volume2 size={14} className="text-[var(--text-muted)]" />
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.01}
                      value={musicVolume}
                      onChange={e => setMusicVolume(parseFloat(e.target.value))}
                      className="flex-1 accent-[var(--accent)]"
                    />
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

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
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium">Scene Breakdown</h3>
            <div className="flex items-center gap-2">
              <button
                onClick={handleSuggestAll}
                disabled={suggestingAll}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white text-xs font-medium transition-colors disabled:opacity-50"
              >
                {suggestingAll ? <Loader2 size={12} className="animate-spin" /> : <Wand2 size={12} />}
                {suggestingAll ? 'Suggesting...' : 'Suggest Music for All Scenes'}
              </button>
              <button
                onClick={handleClearAllSceneMusic}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-[var(--border)] text-xs text-[var(--text-secondary)] hover:text-[var(--error)] hover:border-[var(--error)]/50 transition-colors"
              >
                <Trash2 size={12} />
                Clear All Music
              </button>
            </div>
          </div>

          <div className="space-y-2">
            {scenes.map((scene, i) => {
              const suggestion = sceneSuggestions[i]
              const hasSceneMusic = !!scene.music_track
              return (
                <div
                  key={i}
                  className="rounded border border-[var(--border)] bg-[var(--bg-secondary)] overflow-hidden"
                >
                  {/* Scene info row */}
                  <div className="flex items-center gap-3 p-2 text-xs">
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

                  {/* Scene music controls */}
                  <div className="px-2 pb-2 space-y-2">
                    <div className="flex items-center gap-2">
                      <Music size={12} className="text-[var(--text-muted)] shrink-0" />
                      <select
                        value={scene.music_track || ''}
                        onChange={e => handleSetSceneMusic(i, e.target.value || null)}
                        className="flex-1 min-w-0 bg-[var(--bg-tertiary)] border border-[var(--border)] rounded px-2 py-1 text-xs text-[var(--text-primary)] focus:outline-none focus:border-[var(--border-focus)]"
                      >
                        <option value="">No scene music (use global track)</option>
                        {musicTracks.map(t => (
                          <option key={t.id} value={t.name || t.path || t.id}>
                            {t.title}{t.artist && t.artist !== 'Unknown' ? ` — ${t.artist}` : ''}
                          </option>
                        ))}
                      </select>
                      {hasSceneMusic && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-[var(--accent)]/10 text-[var(--accent)] shrink-0">
                          Active
                        </span>
                      )}
                    </div>

                    {/* Suggested tracks */}
                    {suggestion && suggestion.tracks.length > 0 && (
                      <div className="space-y-1">
                        <div className="flex items-center gap-1.5">
                          <span className="text-[10px] text-[var(--text-muted)]">Suggested for “{suggestion.query}”:</span>
                          <span className="text-[10px] text-[var(--text-muted)] truncate">{suggestion.reasoning}</span>
                        </div>
                        <div className="flex flex-wrap gap-1">
                          {suggestion.tracks.slice(0, 4).map((track, ti) => (
                            <button
                              key={`${i}-${ti}`}
                              onClick={() => {
                                const name = track.name || track.path || track.id
                                if (name) handleSetSceneMusic(i, name)
                              }}
                              className="px-2 py-1 rounded-md bg-[var(--bg-tertiary)] border border-[var(--border)] text-[10px] text-[var(--text-secondary)] hover:text-[var(--accent)] hover:border-[var(--accent)] transition-colors"
                            >
                              {track.title}
                              {track.artist && track.artist !== 'Unknown' ? ` — ${track.artist}` : ''}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                    {suggestion && suggestion.tracks.length === 0 && (
                      <div className="text-[10px] text-[var(--text-muted)]">No tracks found for “{suggestion.query}”</div>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}
