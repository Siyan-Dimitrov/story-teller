import { useState, useEffect } from 'react'
import { BookOpen, Heart, Activity, ListChecks } from 'lucide-react'
import type { HealthStatus, ProjectState } from './api'
import { api } from './api'
import ProjectList from './components/ProjectList'
import StoryWizard from './components/StoryWizard'
import BatchProgress from './components/BatchProgress'
import AgentsDashboard from './components/AgentsDashboard'

type BatchView = 'progress' | 'agents'

export default function App() {
  const [health, setHealth] = useState<HealthStatus | null>(null)
  const [activeProject, setActiveProject] = useState<string | null>(null)
  const [activeBatch, setActiveBatch] = useState<string | null>(null)
  const [batchView, setBatchView] = useState<BatchView>('progress')
  const [project, setProject] = useState<ProjectState | null>(null)

  useEffect(() => {
    api.health().then(setHealth).catch(() => {})
    const i = setInterval(() => {
      api.health().then(setHealth).catch(() => {})
    }, 15000)
    return () => clearInterval(i)
  }, [])

  useEffect(() => {
    if (!activeProject) { setProject(null); return }
    api.getProject(activeProject).then(setProject).catch(() => setActiveProject(null))
  }, [activeProject])

  const refreshProject = () => {
    if (activeProject) {
      api.getProject(activeProject).then(setProject).catch(() => {})
    }
  }

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b border-[var(--border)] bg-[var(--bg-secondary)] px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <BookOpen size={22} className="text-[var(--accent)]" />
          <h1 className="text-lg font-semibold">Story Teller</h1>
          <span className="text-xs text-[var(--text-muted)]">dark fairy tales for grown-ups</span>
        </div>
        <div className="flex items-center gap-4">
          {activeBatch && (
            <div className="flex items-center gap-1 p-0.5 rounded-lg bg-[var(--bg-tertiary)]">
              <button
                onClick={() => setBatchView('progress')}
                className={`flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-md transition-colors ${
                  batchView === 'progress'
                    ? 'bg-[var(--bg-secondary)] text-[var(--text-primary)]'
                    : 'text-[var(--text-muted)] hover:text-[var(--text-secondary)]'
                }`}
              >
                <ListChecks size={12} /> Progress
              </button>
              <button
                onClick={() => setBatchView('agents')}
                className={`flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-md transition-colors ${
                  batchView === 'agents'
                    ? 'bg-[var(--bg-secondary)] text-[var(--text-primary)]'
                    : 'text-[var(--text-muted)] hover:text-[var(--text-secondary)]'
                }`}
              >
                <Activity size={12} /> Agents
              </button>
            </div>
          )}
          {(activeProject || activeBatch) && (
            <button
              onClick={() => { setActiveProject(null); setActiveBatch(null); setBatchView('progress') }}
              className="text-sm text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
            >
              All Projects
            </button>
          )}
          {health && <StatusDots health={health} />}
        </div>
      </header>

      {/* Main */}
      <main className="flex-1 max-w-6xl mx-auto w-full px-6 py-6">
        {activeProject && project ? (
          <StoryWizard project={project} onRefresh={refreshProject} />
        ) : activeBatch ? (
          batchView === 'agents' ? (
            <AgentsDashboard
              groupId={activeBatch}
              onBack={() => { setActiveBatch(null); setBatchView('progress') }}
              onSelectProject={(id) => { setActiveBatch(null); setBatchView('progress'); setActiveProject(id) }}
            />
          ) : (
            <BatchProgress
              groupId={activeBatch}
              onBack={() => { setActiveBatch(null); setBatchView('progress') }}
              onSelectProject={(id) => { setActiveBatch(null); setBatchView('progress'); setActiveProject(id) }}
            />
          )
        ) : (
          <ProjectList
            onSelect={setActiveProject}
            onBatchStart={(groupId) => { setActiveBatch(groupId); setBatchView('progress') }}
          />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-[var(--border)] px-6 py-2 text-center text-xs text-[var(--text-muted)]">
        <span className="flex items-center justify-center gap-1">
          Made with <Heart size={10} className="text-[var(--error)]" /> and dark imagination
        </span>
      </footer>
    </div>
  )
}

function StatusDots({ health }: { health: HealthStatus }) {
  const services = [
    { name: 'Ollama', ok: health.ollama },
    { name: 'VoiceBox', ok: health.voicebox },
    { name: 'Replicate', ok: health.replicate },
    { name: 'OpenAI', ok: health.openai },
    { name: 'FFmpeg', ok: health.ffmpeg },
  ]
  return (
    <div className="flex items-center gap-3">
      {services.map(s => (
        <div key={s.name} className="flex items-center gap-1.5" title={s.name}>
          <div className={`w-1.5 h-1.5 rounded-full ${s.ok ? 'bg-[var(--success)]' : 'bg-[var(--text-muted)]'}`} />
          <span className="text-xs text-[var(--text-muted)]">{s.name}</span>
        </div>
      ))}
    </div>
  )
}
