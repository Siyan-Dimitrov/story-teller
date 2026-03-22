import { useState, useEffect } from 'react'
import { Mic, Play, Loader2 } from 'lucide-react'
import type { ProjectState, VoiceProfile } from '../api'
import { api } from '../api'

interface Props {
  project: ProjectState
  onRefresh: () => void
  onNext: () => void
}

const DEFAULT_INSTRUCT =
  'Speak slowly and deliberately like a storyteller narrating a dark fairy tale. ' +
  'Use a calm, measured pace with dramatic pauses between sentences. ' +
  'Deep, atmospheric tone.'

export default function VoicePanel({ project, onRefresh, onNext }: Props) {
  const [profiles, setProfiles] = useState<VoiceProfile[]>([])
  const [selectedProfile, setSelectedProfile] = useState(project.voice_profile_id || '')
  const [language, setLanguage] = useState(project.voice_language || 'en')
  const [instruct, setInstruct] = useState(DEFAULT_INSTRUCT)
  const [generating, setGenerating] = useState(false)
  const [playingScene, setPlayingScene] = useState<number | null>(null)

  const scenes = project.script?.scenes || []
  const hasAudio = scenes.some(s => s.audio_path)

  useEffect(() => {
    api.profiles().then(p => {
      setProfiles(p)
      if (!selectedProfile && p.length > 0) setSelectedProfile(p[0].id)
    }).catch(() => {})
  }, [])

  const handleGenerate = async () => {
    if (!selectedProfile) { alert('Select a voice profile first'); return }
    setGenerating(true)
    try {
      await api.runVoice(project.project_id, {
        profile_id: selectedProfile,
        language,
        instruct: instruct || undefined,
      })
      onRefresh()
    } catch (e) {
      alert('Voice generation failed: ' + (e as Error).message)
      onRefresh()
    } finally {
      setGenerating(false)
    }
  }

  const playAudio = (sceneIndex: number, audioPath: string) => {
    const url = api.artifactUrl(project.project_id, audioPath)
    const audio = new Audio(url)
    setPlayingScene(sceneIndex)
    audio.onended = () => setPlayingScene(null)
    audio.play().catch(() => setPlayingScene(null))
  }

  if (!project.script) {
    return (
      <div className="text-center py-16 text-[var(--text-muted)]">
        <Mic size={40} className="mx-auto mb-3 opacity-30" />
        <p>Generate a script first before creating voice-over.</p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Voice settings */}
      <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] space-y-3">
        <div className="flex items-end gap-4">
          <div className="flex-1">
            <label className="block text-xs text-[var(--text-secondary)] mb-1.5">Voice Profile</label>
            <select
              value={selectedProfile}
              onChange={e => setSelectedProfile(e.target.value)}
              className="w-full bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--border-focus)]"
            >
              <option value="">Select a voice...</option>
              {profiles.map(p => (
                <option key={p.id} value={p.id}>{p.name} ({p.language})</option>
              ))}
            </select>
          </div>
          <div className="w-32">
            <label className="block text-xs text-[var(--text-secondary)] mb-1.5">Language</label>
            <select
              value={language}
              onChange={e => setLanguage(e.target.value)}
              className="w-full bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--border-focus)]"
            >
              {['en', 'de', 'fr', 'es', 'it', 'pt', 'ru', 'ja', 'ko', 'zh'].map(l => (
                <option key={l} value={l}>{l}</option>
              ))}
            </select>
          </div>
        </div>
        <div>
          <label className="block text-xs text-[var(--text-secondary)] mb-1.5">
            Voice Direction <span className="text-[var(--text-muted)]">(how the voice should sound)</span>
          </label>
          <textarea
            value={instruct}
            onChange={e => setInstruct(e.target.value)}
            rows={2}
            className="w-full bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--border-focus)] resize-none"
            placeholder="e.g. Speak slowly with dramatic pauses, like a dark fairy tale narrator..."
          />
        </div>
        <div className="flex justify-end">
          <button
            onClick={handleGenerate}
            disabled={generating || !selectedProfile}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white text-sm font-medium transition-colors disabled:opacity-50 shrink-0"
          >
            {generating ? <Loader2 size={14} className="animate-spin" /> : <Mic size={14} />}
            {generating ? 'Generating...' : hasAudio ? 'Regenerate All' : 'Generate Voice'}
          </button>
        </div>
      </div>

      {/* Scene audio preview */}
      <div className="space-y-2">
        {scenes.map((scene, i) => (
          <div
            key={i}
            className="flex items-center gap-3 p-3 rounded-lg border border-[var(--border)] bg-[var(--bg-secondary)]"
          >
            <span className="text-xs font-medium text-[var(--accent)] w-16 shrink-0">Scene {i + 1}</span>
            <p className="flex-1 text-xs text-[var(--text-secondary)] truncate">{scene.narration.slice(0, 100)}...</p>
            {scene.audio_duration && (
              <span className="text-xs text-[var(--text-muted)] shrink-0">{scene.audio_duration.toFixed(1)}s</span>
            )}
            {scene.audio_path ? (
              <button
                onClick={() => playAudio(i, scene.audio_path!)}
                className="p-1.5 rounded bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:text-[var(--accent)] transition-colors shrink-0"
              >
                {playingScene === i ? <Loader2 size={12} className="animate-spin" /> : <Play size={12} />}
              </button>
            ) : scene.voice_error ? (
              <span className="text-xs text-[var(--error)]">Error</span>
            ) : (
              <span className="text-xs text-[var(--text-muted)]">—</span>
            )}
          </div>
        ))}
      </div>

      {/* Total duration */}
      {hasAudio && (
        <div className="flex items-center justify-between">
          <span className="text-sm text-[var(--text-secondary)]">
            Total duration: {(scenes.reduce((sum, s) => sum + (s.audio_duration || 0), 0) / 60).toFixed(1)} min
          </span>
          <button
            onClick={onNext}
            className="px-4 py-2 rounded-lg bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white text-sm font-medium transition-colors"
          >
            Next: Images
          </button>
        </div>
      )}
    </div>
  )
}
