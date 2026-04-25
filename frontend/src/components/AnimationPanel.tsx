import { useState, useEffect, useRef } from 'react'
import { Sparkles, Loader2, Play, Eye } from 'lucide-react'
import type { ProjectState } from '../api'
import { api } from '../api'

interface Props {
  project: ProjectState
  onRefresh: () => void
  onNext?: () => void
}

const MOTION_LABELS: Record<string, string> = {
  dolly_forward: 'Dolly Forward',
  dolly_backward: 'Dolly Backward',
  pan_left: 'Pan Left',
  pan_right: 'Pan Right',
  orbital_left: 'Orbital Left',
  orbital_right: 'Orbital Right',
  gentle_rise: 'Gentle Rise',
  gentle_float: 'Gentle Float',
  portrait_breathe: 'Portrait Breathe',
  portrait_reveal: 'Portrait Reveal',
  portrait_drift: 'Portrait Drift',
}

export default function AnimationPanel({ project, onRefresh, onNext }: Props) {
  const [animating, setAnimating] = useState(false)
  const [progress, setProgress] = useState(0)
  const [phase, setPhase] = useState('')
  const [error, setError] = useState<string | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const scenes = project.script?.scenes || []
  const hasImages = scenes.some(s => (s.image_paths && s.image_paths.length > 0) || s.image_path)
  const hasDepthMaps = scenes.some(s => s.depth_map_paths && s.depth_map_paths.length > 0)

  useEffect(() => {
    if (project.step === 'animating') {
      setAnimating(true)
      startPolling()
    }
    return () => stopPolling()
  }, [project.project_id])

  const startPolling = () => {
    stopPolling()
    pollRef.current = setInterval(async () => {
      try {
        const p = await api.animationProgress(project.project_id)
        setProgress(p.progress)
        setPhase(p.phase)

        if (!p.active) {
          stopPolling()
          setAnimating(false)
          if (p.error) {
            setError(p.error)
          } else if (p.phase === 'done') {
            setError(null)
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

  const handleAnimate = async () => {
    setAnimating(true)
    setProgress(0)
    setPhase('starting')
    setError(null)
    try {
      await api.runAnimate(project.project_id)
      startPolling()
    } catch (e) {
      setError('Failed to start animation: ' + (e as Error).message)
      setAnimating(false)
    }
  }

  if (!hasImages) {
    return (
      <div className="text-center py-16 text-[var(--text-muted)]">
        <Sparkles size={40} className="mx-auto mb-3 opacity-30" />
        <p>Generate images first before preparing animations.</p>
      </div>
    )
  }

  const pct = Math.round(progress * 100)

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)]">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-medium">Depth Parallax Animation</h3>
            <p className="text-xs text-[var(--text-muted)] mt-0.5">
              Classifies images and generates depth maps for 2.5D parallax motion
            </p>
          </div>
          {!animating ? (
            <button
              onClick={handleAnimate}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white text-sm font-medium transition-colors"
            >
              <Sparkles size={14} />
              {hasDepthMaps ? 'Re-classify & Regenerate' : 'Prepare Animations'}
            </button>
          ) : (
            <div className="flex items-center gap-2 text-sm text-[var(--text-muted)]">
              <Loader2 size={14} className="animate-spin" />
              Processing...
            </div>
          )}
        </div>

        {/* Progress bar */}
        {animating && (
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

        {error && !animating && (
          <div className="mt-3 p-2 rounded-lg border border-[var(--error)]/30 bg-[var(--error)]/5 text-xs text-[var(--error)]">
            {error}
          </div>
        )}
      </div>

      {/* Results */}
      {hasDepthMaps && !animating && (
        <>
          {/* Next step button */}
          {onNext && (
            <div className="flex justify-end">
              <button
                onClick={onNext}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white text-sm font-medium transition-colors"
              >
                <Play size={14} />
                Continue to Video
              </button>
            </div>
          )}

          {/* Scene animation breakdown */}
          <div>
            <h3 className="text-sm font-medium mb-2">Animation Assignments</h3>
            <div className="space-y-1">
              {scenes.map((scene, i) => {
                const types = scene.animation_types || []
                const presets = scene.motion_presets || []
                const depths = scene.depth_map_paths || []
                const imgCount = scene.image_paths?.length || (scene.image_path ? 1 : 0)

                return (
                  <div
                    key={i}
                    className="p-3 rounded-lg border border-[var(--border)] bg-[var(--bg-secondary)]"
                  >
                    <div className="flex items-center gap-3 text-xs">
                      <span className="text-[var(--accent)] font-medium w-14 shrink-0">Scene {i + 1}</span>
                      <span className="text-[var(--text-muted)] w-12 shrink-0">{imgCount} img{imgCount !== 1 ? 's' : ''}</span>
                      <span className="text-[var(--text-muted)] truncate flex-1">{scene.mood}</span>
                    </div>
                    {types.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-1.5">
                        {types.map((type, j) => (
                          <span
                            key={j}
                            className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-medium ${
                              type === 'portrait'
                                ? 'bg-purple-500/10 text-purple-400 border border-purple-500/20'
                                : 'bg-blue-500/10 text-blue-400 border border-blue-500/20'
                            }`}
                          >
                            {type === 'portrait' ? <Eye size={9} /> : <Sparkles size={9} />}
                            {MOTION_LABELS[presets[j]] || presets[j] || 'unknown'}
                            {depths[j] && <span className="opacity-50">+ depth</span>}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
