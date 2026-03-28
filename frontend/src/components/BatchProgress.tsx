import { useState, useEffect, useRef } from 'react'
import { CheckCircle, XCircle, Loader2, Clock, ArrowLeft, Play } from 'lucide-react'
import type { BatchProgress as BatchProgressType } from '../api'
import { api } from '../api'

const STEP_LABELS: Record<string, string> = {
  script: 'Script',
  voice: 'Voice',
  images: 'Images',
  qc: 'QC',
  animate: 'Animate',
  assemble: 'Assemble',
}

export default function BatchProgress({
  groupId,
  onBack,
  onSelectProject,
}: {
  groupId: string
  onBack: () => void
  onSelectProject: (id: string) => void
}) {
  const [progress, setProgress] = useState<BatchProgressType | null>(null)
  const [error, setError] = useState<string | null>(null)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const fetchProgress = () => {
    api.batchProgress(groupId)
      .then(setProgress)
      .catch(e => setError(e.message))
  }

  useEffect(() => {
    fetchProgress()
    timerRef.current = setInterval(fetchProgress, 3000)
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [groupId])

  // Stop polling when finished
  useEffect(() => {
    if (progress?.finished && timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
  }, [progress?.finished])

  if (error) {
    return (
      <div className="text-center py-16">
        <p className="text-[var(--error)] mb-4">Failed to load batch progress: {error}</p>
        <button onClick={onBack} className="text-sm text-[var(--accent)] hover:underline">Back to projects</button>
      </div>
    )
  }

  if (!progress) {
    return (
      <div className="flex items-center justify-center py-16 gap-2 text-[var(--text-muted)]">
        <Loader2 size={16} className="animate-spin" /> Loading batch progress...
      </div>
    )
  }

  const pct = progress.total > 0 ? Math.round(((progress.completed + progress.failed) / progress.total) * 100) : 0

  return (
    <div>
      <button
        onClick={onBack}
        className="flex items-center gap-1.5 text-sm text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors mb-4"
      >
        <ArrowLeft size={14} /> Back to projects
      </button>

      <div className="p-6 rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)]">
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-lg font-semibold">Batch Processing</h2>
            <p className="text-xs text-[var(--text-muted)] mt-0.5">
              {progress.completed} completed, {progress.failed} failed, {progress.total - progress.completed - progress.failed} remaining
            </p>
          </div>
          {progress.finished ? (
            <span className={`text-sm font-medium px-3 py-1 rounded-full ${
              progress.failed === 0
                ? 'bg-[var(--success)]/15 text-[var(--success)]'
                : 'bg-[var(--warning)]/15 text-[var(--warning)]'
            }`}>
              {progress.failed === 0 ? 'All Complete' : `Done with ${progress.failed} failures`}
            </span>
          ) : (
            <span className="flex items-center gap-2 text-sm text-[var(--accent)]">
              <Loader2 size={14} className="animate-spin" />
              Processing chapter {(progress.current_chapter ?? 0) + 1} — {STEP_LABELS[progress.current_step ?? ''] || progress.current_step}
            </span>
          )}
        </div>

        {/* Progress bar */}
        <div className="w-full h-2 bg-[var(--bg-tertiary)] rounded-full mb-4 overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-500 bg-[var(--accent)]"
            style={{ width: `${pct}%` }}
          />
        </div>

        {/* Chapter rows */}
        <div className="space-y-1.5">
          {progress.chapters.map(ch => (
            <button
              key={ch.project_id}
              onClick={() => onSelectProject(ch.project_id)}
              className={`w-full flex items-center gap-3 p-3 rounded-lg border transition-colors text-left ${
                ch.status === 'running'
                  ? 'border-[var(--accent)]/40 bg-[var(--accent)]/5'
                  : ch.status === 'failed'
                  ? 'border-[var(--error)]/30 bg-[var(--error)]/5'
                  : ch.status === 'completed'
                  ? 'border-[var(--success)]/20 bg-[var(--bg-tertiary)]'
                  : 'border-[var(--border)] bg-[var(--bg-tertiary)]'
              }`}
            >
              {/* Status icon */}
              {ch.status === 'completed' && <CheckCircle size={16} className="text-[var(--success)] shrink-0" />}
              {ch.status === 'failed' && <XCircle size={16} className="text-[var(--error)] shrink-0" />}
              {ch.status === 'running' && <Loader2 size={16} className="text-[var(--accent)] animate-spin shrink-0" />}
              {ch.status === 'pending' && <Clock size={16} className="text-[var(--text-muted)] shrink-0" />}

              {/* Chapter info */}
              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium truncate">{ch.title}</div>
                {ch.status === 'running' && ch.current_step && (
                  <div className="text-[10px] text-[var(--accent)] mt-0.5">
                    {STEP_LABELS[ch.current_step] || ch.current_step}...
                  </div>
                )}
                {ch.status === 'failed' && ch.error && (
                  <div className="text-[10px] text-[var(--error)] mt-0.5 truncate">
                    Failed at {STEP_LABELS[ch.failed_step ?? ''] || ch.failed_step}: {ch.error}
                  </div>
                )}
              </div>

              {/* Chapter number */}
              <span className="text-[10px] text-[var(--text-muted)] shrink-0">
                Ch. {ch.chapter_index + 1}
              </span>
            </button>
          ))}
        </div>

        {/* Summary when done */}
        {progress.finished && progress.failed > 0 && (
          <div className="mt-4 p-3 rounded-lg border border-[var(--warning)]/30 bg-[var(--warning)]/5">
            <p className="text-xs font-medium text-[var(--warning)] mb-1">
              {progress.failed} chapter{progress.failed > 1 ? 's' : ''} failed
            </p>
            <p className="text-[10px] text-[var(--text-muted)]">
              Click on a failed chapter to open it and retry manually from the failed step.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
