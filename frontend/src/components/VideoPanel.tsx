import { useState } from 'react'
import { Film, Download, Loader2, Play } from 'lucide-react'
import type { ProjectState } from '../api'
import { api } from '../api'

interface Props {
  project: ProjectState
  onRefresh: () => void
}

export default function VideoPanel({ project, onRefresh }: Props) {
  const [assembling, setAssembling] = useState(false)
  const isAssembled = project.step === 'assembled'
  const hasImages = project.script?.scenes?.some(s => s.image_path)
  const hasAudio = project.script?.scenes?.some(s => s.audio_path)

  const handleAssemble = async () => {
    setAssembling(true)
    try {
      await api.runAssemble(project.project_id)
      onRefresh()
    } catch (e) {
      alert('Assembly failed: ' + (e as Error).message)
      onRefresh()
    } finally {
      setAssembling(false)
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

  return (
    <div className="space-y-4">
      {/* Assemble controls */}
      <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)]">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-medium">Video Assembly</h3>
            <p className="text-xs text-[var(--text-muted)] mt-0.5">
              Combines all scene images with voice audio using Ken Burns effects
            </p>
          </div>
          <button
            onClick={handleAssemble}
            disabled={assembling}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white text-sm font-medium transition-colors disabled:opacity-50"
          >
            {assembling ? <Loader2 size={14} className="animate-spin" /> : <Film size={14} />}
            {assembling ? 'Assembling...' : isAssembled ? 'Re-assemble' : 'Assemble Video'}
          </button>
        </div>
      </div>

      {/* Status */}
      {!hasAudio && (
        <div className="p-3 rounded-lg border border-[var(--warning)]/30 bg-[var(--warning)]/5 text-xs text-[var(--warning)]">
          No audio generated yet — video will use duration hints for scene timing.
        </div>
      )}

      {/* Video preview */}
      {isAssembled && (
        <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] overflow-hidden">
          <video
            src={api.downloadUrl(project.project_id)}
            controls
            className="w-full aspect-video bg-black"
          >
            Your browser does not support video playback.
          </video>
          <div className="p-4 flex items-center justify-between">
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
        </div>
      )}

      {/* Scene breakdown */}
      {project.script?.scenes && (
        <div>
          <h3 className="text-sm font-medium mb-2">Scene Breakdown</h3>
          <div className="space-y-1">
            {project.script.scenes.map((scene, i) => (
              <div
                key={i}
                className="flex items-center gap-3 p-2 rounded text-xs border border-[var(--border)] bg-[var(--bg-secondary)]"
              >
                <span className="text-[var(--accent)] font-medium w-14 shrink-0">Scene {i + 1}</span>
                <span className={`w-16 shrink-0 ${scene.image_path ? 'text-[var(--success)]' : 'text-[var(--text-muted)]'}`}>
                  {scene.image_path ? 'Image OK' : 'No image'}
                </span>
                <span className={`w-16 shrink-0 ${scene.audio_path ? 'text-[var(--success)]' : 'text-[var(--text-muted)]'}`}>
                  {scene.audio_path ? `${(scene.audio_duration || 0).toFixed(1)}s` : 'No audio'}
                </span>
                <span className="text-[var(--text-muted)] capitalize w-16 shrink-0">{scene.kb_effect.replace('_', ' ')}</span>
                <span className="text-[var(--text-muted)] truncate flex-1">{scene.mood}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
