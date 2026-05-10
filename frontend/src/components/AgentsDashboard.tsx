import { useState, useEffect, useRef } from 'react'
import {
  ArrowLeft, Loader2, Pause, Play, AlertCircle, CheckCircle, XCircle,
  Wallet, Activity, ChevronDown, ChevronRight, Sparkles,
} from 'lucide-react'
import type {
  BatchProgress as BatchProgressType,
  AgentRun,
  BudgetStatus,
  Skill,
  VoiceProfile,
} from '../api'
import { api } from '../api'

const STEP_LABELS: Record<string, string> = {
  script: 'Script',
  voice: 'Voice',
  images: 'Images',
  assemble: 'Assemble',
  publish: 'Publish',
}

const SEVERITY_COLOR: Record<string, string> = {
  ok: 'var(--success)',
  minor: 'var(--text-secondary)',
  major: 'var(--warning)',
  fatal: 'var(--error)',
}

interface Props {
  groupId: string
  onBack: () => void
  onSelectProject: (id: string) => void
}

function fmtCents(c: number | undefined): string {
  if (c == null) return '—'
  if (c < 100) return `${c}¢`
  return `$${(c / 100).toFixed(2)}`
}

function fmtPercent(used?: number, cap?: number): number {
  if (!cap || cap <= 0) return 0
  return Math.min(100, Math.round(((used ?? 0) / cap) * 100))
}

function relTime(iso: string): string {
  const ms = Date.now() - new Date(iso).getTime()
  if (ms < 60_000) return 'just now'
  if (ms < 3_600_000) return `${Math.round(ms / 60_000)}m ago`
  if (ms < 86_400_000) return `${Math.round(ms / 3_600_000)}h ago`
  return `${Math.round(ms / 86_400_000)}d ago`
}

export default function AgentsDashboard({ groupId, onBack, onSelectProject }: Props) {
  const [progress, setProgress] = useState<BatchProgressType | null>(null)
  const [runs, setRuns] = useState<AgentRun[]>([])
  const [budget, setBudget] = useState<BudgetStatus | null>(null)
  const [skills, setSkills] = useState<Skill[]>([])
  const [profiles, setProfiles] = useState<VoiceProfile[]>([])
  const [error, setError] = useState<string | null>(null)
  const [editingCap, setEditingCap] = useState(false)
  const [capDraft, setCapDraft] = useState('')
  const [savingCap, setSavingCap] = useState(false)
  const [expandedProjects, setExpandedProjects] = useState<Set<string>>(new Set())
  const [pausing, setPausing] = useState(false)
  const [skillId, setSkillId] = useState<string>('grimm_gothic')
  const [voiceProfileId, setVoiceProfileId] = useState<string>('')
  const [starting, setStarting] = useState(false)
  const [startError, setStartError] = useState<string | null>(null)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const fetchAll = () => {
    api.batchProgress(groupId).then(setProgress).catch(e => setError(e.message))
    api.agentRuns(groupId, 200).then(r => setRuns(r.rows)).catch(() => {})
    api.agentBudget(groupId).then(setBudget).catch(() => {})
  }

  // One-shot loads
  useEffect(() => {
    api.agentSkills().then(r => setSkills(r.skills)).catch(() => {})
    api.profiles().then(setProfiles).catch(() => {})
  }, [])

  useEffect(() => {
    fetchAll()
    timerRef.current = setInterval(fetchAll, 3000)
    return () => { if (timerRef.current) clearInterval(timerRef.current) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [groupId])

  useEffect(() => {
    if ((progress?.finished || progress?.paused) && timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
  }, [progress?.finished, progress?.paused])

  const handleSaveCap = async () => {
    const cap = parseInt(capDraft, 10)
    if (!Number.isFinite(cap) || cap <= 0) {
      setEditingCap(false); return
    }
    setSavingCap(true)
    try {
      await api.agentBudgetSet(groupId, { cap_cents: cap })
      api.agentBudget(groupId).then(setBudget).catch(() => {})
      setEditingCap(false)
    } finally {
      setSavingCap(false)
    }
  }

  const handlePause = async () => {
    setPausing(true)
    try { await api.batchPause(groupId) } catch { setPausing(false) }
  }

  const handleStartProducer = async () => {
    setStarting(true)
    setStartError(null)
    try {
      await api.producerRun({
        group_id: groupId,
        skill_id: skillId || undefined,
        voice_profile_id: voiceProfileId || undefined,
      })
      // restart polling immediately
      fetchAll()
      if (!timerRef.current) {
        timerRef.current = setInterval(fetchAll, 3000)
      }
    } catch (e) {
      setStartError((e as Error).message)
    } finally {
      setStarting(false)
    }
  }

  const toggleExpanded = (pid: string) => {
    setExpandedProjects(prev => {
      const next = new Set(prev)
      if (next.has(pid)) next.delete(pid); else next.add(pid)
      return next
    })
  }

  if (error) {
    return (
      <div className="text-center py-16">
        <p className="text-[var(--error)] mb-4">Failed to load: {error}</p>
        <button onClick={onBack} className="text-sm text-[var(--accent)] hover:underline">Back</button>
      </div>
    )
  }

  if (!progress) {
    return (
      <div className="flex items-center justify-center py-16 gap-2 text-[var(--text-muted)]">
        <Loader2 size={16} className="animate-spin" /> Loading agents dashboard…
      </div>
    )
  }

  const budgetPct = fmtPercent(budget?.used_cents ?? progress.cost_cents, budget?.cap_cents ?? progress.cap_cents)
  const overWarn = budget && budget.percent >= budget.warn_pct
  const overCap = budget && !budget.ok

  return (
    <div>
      <button
        onClick={onBack}
        className="flex items-center gap-1.5 text-sm text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors mb-4"
      >
        <ArrowLeft size={14} /> Back to projects
      </button>

      {/* Header + status */}
      <div className="p-6 rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] mb-4">
        <div className="flex items-center justify-between mb-1">
          <div className="flex items-center gap-2">
            <Activity size={18} className="text-[var(--accent)]" />
            <h2 className="text-lg font-semibold">Agents Dashboard</h2>
            {progress.source === 'producer' && (
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-[var(--accent)]/15 text-[var(--accent)] font-medium">
                Producer
              </span>
            )}
          </div>
          {!progress.finished && !progress.paused && progress.source === 'producer' && (
            <button
              onClick={handlePause}
              disabled={pausing}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-[var(--border)] text-[var(--text-secondary)] text-sm hover:bg-[var(--bg-tertiary)] disabled:opacity-50"
            >
              {pausing ? <Loader2 size={14} className="animate-spin" /> : <Pause size={14} />}
              Pause
            </button>
          )}
        </div>
        <p className="text-xs text-[var(--text-muted)]">
          Group <code className="px-1 py-0.5 rounded bg-[var(--bg-tertiary)] text-[var(--text-secondary)]">{groupId.slice(0, 12)}</code>
          {' · '}{progress.completed} completed, {progress.failed} failed, {progress.total - progress.completed - progress.failed} pending
          {progress.finished && ' · finished'}
          {progress.paused && ' · paused'}
        </p>
      </div>

      {/* Budget bar */}
      <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] mb-4">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <Wallet size={14} className="text-[var(--text-secondary)]" />
            <h3 className="text-sm font-medium">Budget</h3>
          </div>
          {editingCap ? (
            <div className="flex items-center gap-2">
              <input
                type="number"
                value={capDraft}
                onChange={e => setCapDraft(e.target.value)}
                placeholder="cents"
                className="w-20 bg-[var(--bg-tertiary)] border border-[var(--border)] rounded px-2 py-1 text-xs text-[var(--text-primary)] focus:outline-none focus:border-[var(--border-focus)]"
              />
              <button onClick={handleSaveCap} disabled={savingCap} className="text-xs px-2 py-1 rounded bg-[var(--accent)] text-white disabled:opacity-50">
                {savingCap ? '…' : 'Save'}
              </button>
              <button onClick={() => setEditingCap(false)} className="text-xs px-2 py-1 rounded text-[var(--text-muted)]">Cancel</button>
            </div>
          ) : (
            <button
              onClick={() => { setCapDraft(String(budget?.cap_cents ?? 300)); setEditingCap(true) }}
              className="text-xs text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
            >
              Set cap
            </button>
          )}
        </div>
        <div className="flex items-baseline gap-2 mb-2">
          <span className="text-2xl font-semibold">{fmtCents(budget?.used_cents ?? progress.cost_cents)}</span>
          <span className="text-sm text-[var(--text-muted)]">used of {fmtCents(budget?.cap_cents ?? progress.cap_cents)} cap</span>
        </div>
        <div className="h-2 rounded-full bg-[var(--bg-tertiary)] overflow-hidden">
          <div
            className="h-full transition-all"
            style={{
              width: `${budgetPct}%`,
              background: overCap
                ? 'var(--error)'
                : overWarn
                ? 'var(--warning)'
                : 'var(--success)',
            }}
          />
        </div>
        {overCap && (
          <p className="text-xs text-[var(--error)] mt-2 flex items-center gap-1">
            <AlertCircle size={12} /> Cap reached. Producer will skip new paid steps until cap is raised.
          </p>
        )}
      </div>

      {/* Run with Producer */}
      {(progress.finished || !progress.source || progress.source === 'legacy') && (
        <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] mb-4">
          <div className="flex items-center gap-2 mb-3">
            <Sparkles size={14} className="text-[var(--accent)]" />
            <h3 className="text-sm font-medium">Run with Producer</h3>
          </div>
          <div className="grid grid-cols-2 gap-3 mb-3">
            <div>
              <label className="block text-[10px] text-[var(--text-muted)] mb-1">Skill</label>
              <select
                value={skillId}
                onChange={e => setSkillId(e.target.value)}
                className="w-full bg-[var(--bg-tertiary)] border border-[var(--border)] rounded px-2 py-1 text-xs text-[var(--text-primary)] focus:outline-none focus:border-[var(--border-focus)]"
              >
                <option value="">No skill (use project defaults)</option>
                {skills.map(s => (
                  <option key={s.id} value={s.id}>{s.name}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-[10px] text-[var(--text-muted)] mb-1">Voice Profile</label>
              <select
                value={voiceProfileId}
                onChange={e => setVoiceProfileId(e.target.value)}
                className="w-full bg-[var(--bg-tertiary)] border border-[var(--border)] rounded px-2 py-1 text-xs text-[var(--text-primary)] focus:outline-none focus:border-[var(--border-focus)]"
              >
                <option value="">Select voice…</option>
                {profiles.map(p => (
                  <option key={p.id} value={p.id}>{p.name} ({p.language})</option>
                ))}
              </select>
            </div>
          </div>
          {skillId && (() => {
            const sk = skills.find(s => s.id === skillId)
            return sk ? (
              <p className="text-[11px] text-[var(--text-secondary)] mb-3 italic">{sk.description}</p>
            ) : null
          })()}
          {startError && (
            <p className="text-[11px] text-[var(--error)] mb-2 flex items-center gap-1">
              <AlertCircle size={12} /> {startError}
            </p>
          )}
          <button
            onClick={handleStartProducer}
            disabled={starting || !voiceProfileId}
            className="flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white text-sm font-medium transition-colors disabled:opacity-50"
          >
            {starting ? <><Loader2 size={14} className="animate-spin" /> Starting…</> : <><Play size={14} /> Run Producer</>}
          </button>
          {!voiceProfileId && (
            <p className="text-[10px] text-[var(--text-muted)] mt-2">Pick a voice profile to enable.</p>
          )}
        </div>
      )}

      {/* Producer queue */}
      <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] mb-4">
        <h3 className="text-sm font-medium mb-3">Queue ({progress.chapters.length})</h3>
        <div className="space-y-1">
          {progress.chapters.map((ch) => {
            const isExpanded = expandedProjects.has(ch.project_id)
            const verdict = ch.critic_verdict
            const sev = verdict?.severity
            return (
              <div key={ch.project_id} className="border border-[var(--border)] rounded-lg overflow-hidden">
                <div
                  className="flex items-center gap-3 px-3 py-2 cursor-pointer hover:bg-[var(--bg-tertiary)]"
                  onClick={() => toggleExpanded(ch.project_id)}
                >
                  {isExpanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                  {ch.status === 'completed' && <CheckCircle size={14} className="text-[var(--success)] shrink-0" />}
                  {ch.status === 'failed' && <XCircle size={14} className="text-[var(--error)] shrink-0" />}
                  {ch.status === 'running' && <Loader2 size={14} className="text-[var(--accent)] animate-spin shrink-0" />}
                  {ch.status === 'pending' && <div className="w-3.5 h-3.5 rounded-full border border-[var(--text-muted)] shrink-0" />}
                  <div className="flex-1 min-w-0">
                    <div className="text-sm truncate">{ch.title || ch.project_id}</div>
                    <div className="text-[10px] text-[var(--text-muted)] flex items-center gap-2">
                      <span>{STEP_LABELS[ch.current_step ?? ''] ?? ch.current_step ?? ch.status}</span>
                      {(ch.critic_attempts ?? 0) > 1 && (
                        <span title="Critic regenerate attempts">↻ {ch.critic_attempts}</span>
                      )}
                      {sev && (
                        <span style={{ color: SEVERITY_COLOR[sev] }} title="Last critic verdict">
                          critic: {sev}
                        </span>
                      )}
                    </div>
                  </div>
                  <button
                    onClick={(e) => { e.stopPropagation(); onSelectProject(ch.project_id) }}
                    className="text-[10px] text-[var(--accent)] hover:underline shrink-0"
                  >
                    Open
                  </button>
                </div>
                {isExpanded && (
                  <div className="px-3 py-2 bg-[var(--bg-tertiary)] text-xs space-y-1">
                    {ch.error && (
                      <div className="text-[var(--error)]">
                        <strong>Error:</strong> {ch.error}
                      </div>
                    )}
                    {verdict?.feedback && (
                      <div className="text-[var(--text-secondary)] whitespace-pre-wrap">
                        <strong>Critic feedback:</strong>{'\n'}{verdict.feedback}
                      </div>
                    )}
                    {!ch.error && !verdict?.feedback && (
                      <div className="text-[var(--text-muted)]">No additional details yet.</div>
                    )}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>

      {/* Recent agent runs */}
      <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)]">
        <h3 className="text-sm font-medium mb-3">Recent agent runs ({runs.length})</h3>
        {runs.length === 0 ? (
          <p className="text-xs text-[var(--text-muted)]">No runs recorded yet.</p>
        ) : (
          <div className="space-y-1 max-h-80 overflow-y-auto pr-1">
            {[...runs].reverse().map((r, i) => (
              <div key={`${r.ts}-${i}`} className="flex items-center gap-2 text-[11px] py-1 border-b border-[var(--border)] last:border-0">
                <span className="text-[var(--text-muted)] w-16 shrink-0">{relTime(r.ts)}</span>
                <span
                  className="font-medium w-20 shrink-0"
                  style={{
                    color: r.phase === 'failed'
                      ? 'var(--error)'
                      : r.phase === 'done'
                      ? 'var(--success)'
                      : 'var(--text-secondary)',
                  }}
                >
                  {r.phase}
                </span>
                {r.step && <span className="text-[var(--text-secondary)] w-16 shrink-0">{r.step}</span>}
                <span className="text-[var(--text-muted)] truncate flex-1" title={r.project_id}>
                  {r.project_id.slice(0, 12)}
                </span>
                {r.critic_severity && (
                  <span
                    className="shrink-0"
                    style={{ color: SEVERITY_COLOR[r.critic_severity] ?? 'var(--text-muted)' }}
                  >
                    {r.critic_severity}
                  </span>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
