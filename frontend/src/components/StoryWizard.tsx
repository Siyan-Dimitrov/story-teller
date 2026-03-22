import { useState } from 'react'
import {
  Scroll, Mic, ImageIcon, Sparkles, Film,
  Check, Loader2, AlertCircle,
} from 'lucide-react'
import type { ProjectState } from '../api'
import ScriptPanel from './ScriptPanel'
import VoicePanel from './VoicePanel'
import ImagePanel from './ImagePanel'
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
  animating: 3,
  animated: 4,
  assembled: 5,
  assembling: 4,
}

function stepDone(projectStep: string, tabIndex: number): boolean {
  const order = STEP_ORDER[projectStep] ?? 0
  return order > tabIndex
}

function stepActive(projectStep: string, tabIndex: number): boolean {
  const order = STEP_ORDER[projectStep] ?? 0
  return order === tabIndex && (
    projectStep.startsWith('generating') || projectStep === 'assembling' || projectStep === 'animating'
  )
}

export default function StoryWizard({ project, onRefresh }: Props) {
  const [activeTab, setActiveTab] = useState<StepKey>('script')

  return (
    <div>
      {/* Title */}
      <div className="mb-4">
        <h2 className="text-xl font-semibold">{project.title || 'Untitled Story'}</h2>
        <p className="text-sm text-[var(--text-muted)]">
          {project.source_tale || 'Custom story'} &middot; {project.target_minutes} min target
        </p>
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
        <ImagePanel project={project} onRefresh={onRefresh} onNext={() => setActiveTab('animate')} />
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
