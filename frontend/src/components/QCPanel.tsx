import { useState, useEffect, useRef } from 'react'
import { ShieldCheck, Loader2, RefreshCw, CheckCircle2, XCircle, ChevronRight, Square, CheckSquare } from 'lucide-react'
import type { ProjectState } from '../api'
import { api } from '../api'

interface Props {
  project: ProjectState
  onRefresh: () => void
  onNext: () => void
}

const CRITERIA_LABELS: Record<string, string> = {
  prompt_adherence: 'Prompt',
  artistic_quality: 'Art',
  technical_quality: 'Tech',
  style_consistency: 'Style',
}

export default function QCPanel({ project, onRefresh, onNext }: Props) {
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState(0)
  const [phase, setPhase] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [threshold, setThreshold] = useState(3.0)
  const [retrying, setRetrying] = useState<string | null>(null)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const scenes = project.script?.scenes || []
  const hasImages = scenes.some(s => (s.image_paths && s.image_paths.length > 0) || s.image_path)
  const hasQCResults = scenes.some(s => s.qc_results && s.qc_results.length > 0)

  // Detect error: explicit error string OR empty scores with avg 0 (failed evaluation)
  const isQCError = (qr: any) =>
    (qr.error != null && qr.error !== '') || (Object.keys(qr.scores || {}).length === 0 && qr.average_score === 0)

  // Count all images
  const allImageKeys: string[] = []
  scenes.forEach((scene, i) => {
    const paths = scene.image_paths || (scene.image_path ? [scene.image_path] : [])
    paths.forEach((_: any, j: number) => allImageKeys.push(`${i}-${j}`))
  })

  useEffect(() => {
    if (project.step === 'qc_running') {
      setRunning(true)
      startPolling()
    }
    return () => stopPolling()
  }, [project.project_id])

  const startPolling = () => {
    stopPolling()
    pollRef.current = setInterval(async () => {
      try {
        const p = await api.qcProgress(project.project_id)
        setProgress(p.progress)
        setPhase(p.phase)

        if (!p.active) {
          stopPolling()
          setRunning(false)
          if (p.error) {
            setError(p.error)
          } else if (p.phase === 'done') {
            setError(null)
            setSelected(new Set())
          }
          onRefresh()
        }
      } catch {
        // Backend might be restarting
      }
    }, 1500)
  }

  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }

  const handleRunQC = async (targets?: { scene_index: number; image_index: number }[]) => {
    setRunning(true)
    setProgress(0)
    setPhase('starting')
    setError(null)
    try {
      await api.runQC(project.project_id, {
        pass_threshold: threshold,
        targets: targets,
      })
      startPolling()
    } catch (e) {
      setError('Failed to start QC: ' + (e as Error).message)
      setRunning(false)
    }
  }

  const handleRunSelected = () => {
    if (selected.size === 0) return
    const targets = Array.from(selected).map(key => {
      const [si, ii] = key.split('-').map(Number)
      return { scene_index: si, image_index: ii }
    })
    handleRunQC(targets)
  }

  const handleRunAll = () => handleRunQC()

  const handleRetry = async (sceneIndex: number, imageIndex: number) => {
    const key = `${sceneIndex}-${imageIndex}`
    setRetrying(key)
    try {
      await api.retryQCImage(project.project_id, sceneIndex, imageIndex)
      onRefresh()
    } catch (e) {
      alert('Retry failed: ' + (e as Error).message)
    } finally {
      setRetrying(null)
    }
  }

  const toggleSelect = (sceneIndex: number, imageIndex: number) => {
    const key = `${sceneIndex}-${imageIndex}`
    setSelected(prev => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }

  const selectAll = () => setSelected(new Set(allImageKeys))

  const selectAllFailed = () => {
    const keys = new Set<string>()
    scenes.forEach((scene, i) => {
      for (const qr of scene.qc_results || []) {
        if (!qr.passed) keys.add(`${i}-${qr.image_index}`)
      }
    })
    setSelected(keys)
  }

  const clearSelection = () => setSelected(new Set())

  const handleRegenerateSelected = async () => {
    if (selected.size === 0) return
    const targets = Array.from(selected).map(key => {
      const [si, ii] = key.split('-').map(Number)
      return { scene_index: si, image_index: ii }
    })
    setRunning(true)
    setProgress(0)
    setPhase('starting regeneration')
    setError(null)
    try {
      await api.regenerateQC(project.project_id, { targets })
      startPolling()
    } catch (e) {
      setError('Failed to start regeneration: ' + (e as Error).message)
      setRunning(false)
    }
  }

  if (!hasImages) {
    return (
      <div className="text-center py-16 text-[var(--text-muted)]">
        <ShieldCheck size={40} className="mx-auto mb-3 opacity-30" />
        <p>Generate images first before running quality checks.</p>
      </div>
    )
  }

  const pct = Math.round(progress * 100)

  // Compute overall stats
  let totalEvaluated = 0
  let totalPassed = 0
  let totalFailed = 0
  let totalErrors = 0
  for (const scene of scenes) {
    for (const qr of scene.qc_results || []) {
      totalEvaluated++
      if (isQCError(qr)) totalErrors++
      else if (qr.passed) totalPassed++
      else totalFailed++
    }
  }

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)]">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-medium">Image Quality Control</h3>
            <p className="text-xs text-[var(--text-muted)] mt-0.5">
              Select images to evaluate, or run on all
            </p>
          </div>
        </div>

        {/* Settings */}
        {!running && (
          <div className="mt-3 flex items-end gap-4">
            <div className="w-44">
              <label className="block text-xs text-[var(--text-secondary)] mb-1">
                Pass Threshold ({threshold.toFixed(1)}/5)
              </label>
              <input
                type="range"
                min="1"
                max="5"
                step="0.5"
                value={threshold}
                onChange={e => setThreshold(parseFloat(e.target.value))}
                className="w-full accent-[var(--accent)]"
              />
            </div>
          </div>
        )}

        {/* Progress bar */}
        {running && (
          <div className="mt-4 space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="text-[var(--text-secondary)] capitalize">
                {phase || 'Starting...'}
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

        {error && !running && (
          <div className="mt-3 p-2 rounded-lg border border-[var(--error)]/30 bg-[var(--error)]/5 text-xs text-[var(--error)]">
            {error}
          </div>
        )}
      </div>

      {/* Selection toolbar — always visible when not running */}
      {!running && (
        <div className="flex items-center gap-3 p-3 rounded-lg border border-[var(--border)] bg-[var(--bg-secondary)]">
          <span className="text-xs text-[var(--text-secondary)]">
            {selected.size}/{allImageKeys.length} selected
          </span>
          <button onClick={selectAll} className="text-xs text-[var(--accent)] hover:underline">
            Select all
          </button>
          {hasQCResults && (totalFailed > 0 || totalErrors > 0) && (
            <button onClick={selectAllFailed} className="text-xs text-orange-400 hover:underline">
              Select failed
            </button>
          )}
          {selected.size > 0 && (
            <button onClick={clearSelection} className="text-xs text-[var(--text-muted)] hover:underline">
              Clear
            </button>
          )}
          <div className="flex-1" />
          {selected.size > 0 && (
            <button
              onClick={handleRunSelected}
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white text-xs font-medium transition-colors"
            >
              <ShieldCheck size={12} />
              QC Selected ({selected.size})
            </button>
          )}
          <button
            onClick={handleRunAll}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[var(--accent)]/20 hover:bg-[var(--accent)]/30 text-[var(--accent)] text-xs font-medium transition-colors border border-[var(--accent)]/30"
          >
            <ShieldCheck size={12} />
            QC All
          </button>
          {hasQCResults && selected.size > 0 && (
            <button
              onClick={handleRegenerateSelected}
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-orange-500 hover:bg-orange-600 text-white text-xs font-medium transition-colors"
            >
              <RefreshCw size={12} />
              Regen ({selected.size})
            </button>
          )}
        </div>
      )}

      {/* Summary stats */}
      {hasQCResults && !running && (
        <div className="flex gap-3">
          <div className="flex-1 p-3 rounded-lg border border-[var(--border)] bg-[var(--bg-secondary)] text-center">
            <div className="text-lg font-bold text-[var(--text-primary)]">{totalEvaluated}</div>
            <div className="text-[10px] text-[var(--text-muted)]">Evaluated</div>
          </div>
          <div className="flex-1 p-3 rounded-lg border border-green-500/20 bg-green-500/5 text-center">
            <div className="text-lg font-bold text-green-400">{totalPassed}</div>
            <div className="text-[10px] text-green-400/70">Passed</div>
          </div>
          <div className="flex-1 p-3 rounded-lg border border-red-500/20 bg-red-500/5 text-center">
            <div className="text-lg font-bold text-red-400">{totalFailed}</div>
            <div className="text-[10px] text-red-400/70">Failed</div>
          </div>
          {totalErrors > 0 && (
            <div className="flex-1 p-3 rounded-lg border border-yellow-500/20 bg-yellow-500/5 text-center">
              <div className="text-lg font-bold text-yellow-400">{totalErrors}</div>
              <div className="text-[10px] text-yellow-400/70">Errors</div>
            </div>
          )}
        </div>
      )}

      {/* Per-scene image grid — always visible */}
      {!running && (
        <div className="space-y-3">
          {scenes.map((scene, i) => {
            const paths = scene.image_paths || (scene.image_path ? [scene.image_path] : [])
            const qcResults = scene.qc_results || []
            if (paths.length === 0) return null

            return (
              <div key={i} className="rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] overflow-hidden">
                <div className="px-3 py-2 border-b border-[var(--border)] flex items-center justify-between">
                  <span className="text-xs font-medium text-[var(--accent)]">Scene {i + 1}</span>
                  <div className="flex items-center gap-2">
                    {qcResults.length > 0 && (
                      scene.qc_passed ? (
                        <span className="flex items-center gap-1 text-[10px] text-green-400">
                          <CheckCircle2 size={10} /> All passed
                        </span>
                      ) : (
                        <span className="flex items-center gap-1 text-[10px] text-red-400">
                          <XCircle size={10} /> Has failures
                        </span>
                      )
                    )}
                  </div>
                </div>
                <div className="grid grid-cols-4 gap-1 p-1">
                  {paths.map((path: string, j: number) => {
                    const qr = qcResults.find((r: any) => r.image_index === j)
                    const isRetrying = retrying === `${i}-${j}`
                    const selKey = `${i}-${j}`
                    const isSelected = selected.has(selKey)

                    return (
                      <div key={j} className="relative group">
                        <img
                          src={api.artifactUrl(project.project_id, path)}
                          alt={`Scene ${i + 1} - Image ${j + 1}`}
                          className={`w-full aspect-video object-cover rounded ${isSelected ? 'ring-2 ring-[var(--accent)]' : ''}`}
                        />
                        {/* Selection checkbox — always visible */}
                        <button
                          onClick={() => toggleSelect(i, j)}
                          className="absolute top-1 right-1 z-10"
                        >
                          {isSelected ? (
                            <CheckSquare size={16} className="text-[var(--accent)] drop-shadow" />
                          ) : (
                            <Square size={16} className="text-white/50 hover:text-[var(--accent)] drop-shadow transition-colors" />
                          )}
                        </button>
                        {/* Pass/fail badge */}
                        {qr && (
                          <div className={`absolute top-1 left-1 flex items-center gap-0.5 px-1.5 py-0.5 rounded text-[9px] font-bold ${
                            isQCError(qr)
                              ? 'bg-yellow-500/80 text-white'
                              : qr.passed
                                ? 'bg-green-500/80 text-white'
                                : 'bg-red-500/80 text-white'
                          }`}>
                            {isQCError(qr) ? (
                              <span>ERR</span>
                            ) : qr.passed ? (
                              <CheckCircle2 size={8} />
                            ) : (
                              <XCircle size={8} />
                            )}
                            {isQCError(qr) ? '' : qr.average_score.toFixed(1)}
                          </div>
                        )}
                        {/* Attempts badge */}
                        {qr && qr.attempts > 1 && (
                          <div className="absolute bottom-1 right-1 px-1 py-0.5 rounded text-[8px] bg-black/60 text-white/80">
                            {qr.attempts} tries
                          </div>
                        )}
                        {/* Hover overlay with scores */}
                        <div className="absolute inset-0 bg-black/80 opacity-0 group-hover:opacity-100 transition-opacity rounded p-2 flex flex-col justify-between pointer-events-none group-hover:pointer-events-auto">
                          {qr ? (
                            <>
                              <div className="space-y-1">
                                {isQCError(qr) ? (
                                  <div className="text-[9px] text-yellow-300">{qr.error || 'Vision model evaluation failed — check that the model is pulled in Ollama'}</div>
                                ) : (
                                  Object.entries(qr.scores).map(([key, val]) => (
                                    <div key={key} className="flex items-center justify-between text-[9px]">
                                      <span className="text-white/70">{CRITERIA_LABELS[key] || key}</span>
                                      <div className="flex items-center gap-1">
                                        <div className="w-12 h-1 rounded-full bg-white/20 overflow-hidden">
                                          <div
                                            className={`h-full rounded-full ${(val as number) >= 4 ? 'bg-green-400' : (val as number) >= 3 ? 'bg-yellow-400' : 'bg-red-400'}`}
                                            style={{ width: `${((val as number) / 5) * 100}%` }}
                                          />
                                        </div>
                                        <span className="text-white/90 w-3 text-right">{val as number}</span>
                                      </div>
                                    </div>
                                  ))
                                )}
                              </div>
                              {qr.reasoning && (
                                <div className="text-[8px] text-white/60 leading-tight line-clamp-2">
                                  {qr.reasoning}
                                </div>
                              )}
                              {!qr.passed && (
                                <button
                                  onClick={() => handleRetry(i, j)}
                                  disabled={isRetrying}
                                  className="mt-1 w-full flex items-center justify-center gap-1 px-2 py-1 rounded text-[9px] bg-white/10 hover:bg-white/20 text-white transition-colors"
                                >
                                  {isRetrying ? <Loader2 size={8} className="animate-spin" /> : <RefreshCw size={8} />}
                                  Re-evaluate
                                </button>
                              )}
                            </>
                          ) : (
                            <div className="flex items-center justify-center h-full text-[9px] text-white/50">
                              Not evaluated
                            </div>
                          )}
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* Next */}
      {hasQCResults && !running && (
        <div className="flex justify-end">
          <button
            onClick={onNext}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white text-sm font-medium transition-colors"
          >
            <ChevronRight size={14} />
            Next: Animate
          </button>
        </div>
      )}
    </div>
  )
}
