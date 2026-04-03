import { useState } from 'react'
import {
  Scroll, Mic, ImageIcon, ShieldCheck, Sparkles, Film,
  Check, Loader2, AlertCircle,
} from 'lucide-react'
import type { ProjectState } from '../api'
import { api } from '../api'

const ADAPTATION_TONES = [
  { value: 'dark', label: 'Dark & Gothic' },
  { value: 'humorous', label: 'Dark Humor' },
  { value: 'psychological horror', label: 'Psychological Horror' },
  { value: 'gothic noir', label: 'Gothic Noir' },
  { value: 'whimsical dark', label: 'Whimsical Dark' },
  { value: 'tragic', label: 'Tragic' },
  { value: 'satirical', label: 'Satirical' },
  { value: 'romantic gothic', label: 'Romantic Gothic' },
]
import ScriptPanel from './ScriptPanel'
import VoicePanel from './VoicePanel'
import ImagePanel from './ImagePanel'
import QCPanel from './QCPanel'
import AnimationPanel from './AnimationPanel'
import VideoPanel from './VideoPanel'

interface Props {
  project: ProjectState
  onRefresh: () => void
}

const STEPS = [
  { key: 'script', label: 'Script', icon: Scroll },
  { key: 'voice', label: 'Voice', icon: Mic },
  { key: 'images', label: 'Images', icon: ImageIcon },
  { key: 'qc', label: 'QC', icon: ShieldCheck },
  { key: 'animate', label: 'Animate', icon: Sparkles },
  { key: 'video', label: 'Video', icon: Film },
] as const

type StepKey = typeof STEPS[number]['key']

const STEP_ORDER: Record<string, number> = {
  created: 0,
  scripted: 1,
  generating_script: 0,
  voiced: 2,
  generating_voice: 1,
  illustrated: 3,
  generating_images: 2,
  qc_running: 3,
  qc_passed: 4,
  animating: 4,
  animated: 5,
  assembled: 6,
  assembling: 5,
}

function stepDone(projectStep: string, tabIndex: number): boolean {
  const order = STEP_ORDER[projectStep] ?? 0
  return order > tabIndex
}

function stepActive(projectStep: string, tabIndex: number): boolean {
  const order = STEP_ORDER[projectStep] ?? 0
  return order === tabIndex && (
    projectStep.startsWith('generating') || projectStep === 'assembling' || projectStep === 'animating' || projectStep === 'qc_running'
  )
}

export default function StoryWizard({ project, onRefresh }: Props) {
  const [activeTab, setActiveTab] = useState<StepKey>('script')
  const canEditSettings = project.step === 'created'

  const updateSetting = (updates: { tone?: string; target_minutes?: number; suggested_length?: string }) => {
    api.updateSettings(project.project_id, updates).then(() => onRefresh()).catch(() => {})
  }

  return (
    <div>
      {/* Title */}
      <div className="mb-4">
        <h2 className="text-xl font-semibold">{project.title || 'Untitled Story'}</h2>
        <div className="flex items-center gap-3 text-sm text-[var(--text-muted)] mt-1">
          <span>{project.source_tale || 'Custom story'}</span>
          {project.char_count && project.char_count > 0 && (
            <>
              <span>&middot;</span>
              <span className="text-[var(--accent)]" title="Estimated from source text">~{project.estimated_duration}m</span>
              <span>{project.char_count.toLocaleString()} chars</span>
            </>
          )}
          <span>&middot;</span>
          {canEditSettings ? (
            <>
              <span className="flex items-center gap-1">
                <input
                  type="number"
                  min="1"
                  step="0.5"
                  value={project.target_minutes}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value)
                    if (val > 0) updateSetting({ target_minutes: val })
                  }}
                  onKeyDown={(e) => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur() }}
                  className="w-14 bg-[var(--bg-tertiary)] border border-[var(--border)] rounded px-1.5 py-0.5 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--border-focus)]"
                />
                min target
              </span>
              <span>&middot;</span>
              <input
                type="text"
                value={project.suggested_length || ''}
                onChange={(e) => updateSetting({ suggested_length: e.target.value })}
                onKeyDown={(e) => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur() }}
                placeholder="e.g. 5 min"
                className="w-24 bg-[var(--bg-tertiary)] border border-[var(--border)] rounded px-1.5 py-0.5 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--border-focus)]"
              />
              <span>&middot;</span>
              <select
                value={project.tone || 'dark'}
                onChange={(e) => updateSetting({ tone: e.target.value })}
                className="bg-[var(--bg-tertiary)] border border-[var(--border)] rounded px-1.5 py-0.5 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--border-focus)] capitalize"
              >
                {project.tone && !ADAPTATION_TONES.some(t => t.value === project.tone) && (
                  <option value={project.tone}>{project.tone}</option>
                )}
                {ADAPTATION_TONES.map(t => (
                  <option key={t.value} value={t.value}>{t.label}</option>
                ))}
              </select>
            </>
          ) : (
            <span>{project.target_minutes} min target{project.suggested_length ? ` · ${project.suggested_length}` : ''}{project.tone ? ` · ${project.tone}` : ''}</span>
          )}
        </div>
      </div>

      {/* Error banner */}
      {project.error && (
        <div className="mb-4 rounded-lg border border-[var(--error)]/30 bg-[var(--error)]/5 p-4">
          <div className="flex items-start gap-3">
            <AlertCircle size={16} className="text-[var(--error)] shrink-0 mt-0.5" />
            <pre className="text-xs text-[var(--text-secondary)] whitespace-pre-wrap overflow-x-auto flex-1">
              {project.error}
            </pre>
          </div>
        </div>
      )}

      {/* Step tabs */}
      <div className="flex gap-1 mb-6 p-1 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border)]">
        {STEPS.map((step, i) => {
          const done = stepDone(project.step, i)
          const running = stepActive(project.step, i)
          const Icon = step.icon
          return (
            <button
              key={step.key}
              onClick={() => setActiveTab(step.key)}
              className={`flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                activeTab === step.key
                  ? 'bg-[var(--bg-tertiary)] text-[var(--text-primary)]'
                  : 'text-[var(--text-muted)] hover:text-[var(--text-secondary)]'
              }`}
            >
              {done ? (
                <Check size={14} className="text-[var(--success)]" />
              ) : running ? (
                <Loader2 size={14} className="text-[var(--accent)] animate-spin" />
              ) : (
                <Icon size={14} />
              )}
              {step.label}
            </button>
          )
        })}
      </div>

      {/* Panels */}
      {activeTab === 'script' && (
        <ScriptPanel project={project} onRefresh={onRefresh} onNext={() => setActiveTab('voice')} />
      )}
      {activeTab === 'voice' && (
        <VoicePanel project={project} onRefresh={onRefresh} onNext={() => setActiveTab('images')} />
      )}
      {activeTab === 'images' && (
        <ImagePanel project={project} onRefresh={onRefresh} onNext={() => setActiveTab('qc')} />
      )}
      {activeTab === 'qc' && (
        <QCPanel project={project} onRefresh={onRefresh} onNext={() => setActiveTab('animate')} />
      )}
      {activeTab === 'animate' && (
        <AnimationPanel project={project} onRefresh={onRefresh} onNext={() => setActiveTab('video')} />
      )}
      {activeTab === 'video' && (
        <VideoPanel project={project} onRefresh={onRefresh} />
      )}
    </div>
  )
}
