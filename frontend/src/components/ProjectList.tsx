import { useState, useEffect } from 'react'
import { Plus, BookOpen, Trash2, Clock, ChevronRight } from 'lucide-react'
import type { ProjectSummary, Tale } from '../api'
import { api } from '../api'

const STEP_LABELS: Record<string, string> = {
  created: 'New',
  scripted: 'Script Ready',
  voiced: 'Voice Done',
  illustrated: 'Images Done',
  assembled: 'Complete',
  generating_script: 'Generating Script...',
  generating_voice: 'Generating Voice...',
  generating_images: 'Generating Images...',
  assembling: 'Assembling...',
}

export default function ProjectList({ onSelect }: { onSelect: (id: string) => void }) {
  const [projects, setProjects] = useState<ProjectSummary[]>([])
  const [tales, setTales] = useState<Tale[]>([])
  const [showCreate, setShowCreate] = useState(false)
  const [selectedTale, setSelectedTale] = useState('')
  const [targetMinutes, setTargetMinutes] = useState(5)
  const [ollamaModel, setOllamaModel] = useState('kimi-k2.5:cloud')
  const [creating, setCreating] = useState(false)

  const refresh = () => {
    api.listProjects().then(setProjects).catch(() => {})
  }

  useEffect(() => {
    refresh()
    api.tales().then(setTales).catch(() => {})
  }, [])

  const handleCreate = async () => {
    setCreating(true)
    try {
      const proj = await api.createProject({
        source_tale: selectedTale,
        target_minutes: targetMinutes,
        ollama_model: ollamaModel,
      })
      onSelect(proj.project_id)
    } catch (e) {
      alert('Failed to create project: ' + (e as Error).message)
    } finally {
      setCreating(false)
    }
  }

  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation()
    if (!confirm('Delete this project?')) return
    await api.deleteProject(id).catch(() => {})
    refresh()
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold">Projects</h2>
        <button
          onClick={() => setShowCreate(!showCreate)}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white text-sm font-medium transition-colors"
        >
          <Plus size={16} /> New Story
        </button>
      </div>

      {/* Create form */}
      {showCreate && (
        <div className="mb-6 p-6 rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)]">
          <h3 className="text-sm font-medium mb-4">New Story Project</h3>

          <div className="space-y-4">
            <div>
              <label className="block text-xs text-[var(--text-secondary)] mb-1.5">Source Tale</label>
              <select
                value={selectedTale}
                onChange={e => setSelectedTale(e.target.value)}
                className="w-full bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--border-focus)]"
              >
                <option value="">Custom / Original</option>
                {tales.map(t => (
                  <option key={t.id} value={t.id}>{t.title} — {t.origin}</option>
                ))}
              </select>
              {selectedTale && tales.find(t => t.id === selectedTale) && (
                <p className="mt-2 text-xs text-[var(--text-muted)] leading-relaxed">
                  {tales.find(t => t.id === selectedTale)!.description}
                </p>
              )}
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-[var(--text-secondary)] mb-1.5">Target Length (minutes)</label>
                <input
                  type="number"
                  min={1}
                  max={20}
                  value={targetMinutes}
                  onChange={e => setTargetMinutes(Number(e.target.value))}
                  className="w-full bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--border-focus)]"
                />
              </div>
              <div>
                <label className="block text-xs text-[var(--text-secondary)] mb-1.5">LLM Model</label>
                <input
                  type="text"
                  value={ollamaModel}
                  onChange={e => setOllamaModel(e.target.value)}
                  className="w-full bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--border-focus)]"
                />
              </div>
            </div>

            <button
              onClick={handleCreate}
              disabled={creating}
              className="px-4 py-2 rounded-lg bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white text-sm font-medium transition-colors disabled:opacity-50"
            >
              {creating ? 'Creating...' : 'Create Project'}
            </button>
          </div>
        </div>
      )}

      {/* Project list */}
      <div className="space-y-2">
        {projects.length === 0 && !showCreate && (
          <div className="text-center py-16 text-[var(--text-muted)]">
            <BookOpen size={40} className="mx-auto mb-3 opacity-30" />
            <p>No stories yet. Create your first dark fairy tale.</p>
          </div>
        )}
        {projects.map(p => (
          <button
            key={p.project_id}
            onClick={() => onSelect(p.project_id)}
            className="w-full flex items-center gap-4 p-4 rounded-lg border border-[var(--border)] bg-[var(--bg-secondary)] hover:bg-[var(--bg-hover)] transition-colors text-left"
          >
            <BookOpen size={18} className="text-[var(--accent)] shrink-0" />
            <div className="flex-1 min-w-0">
              <div className="font-medium text-sm truncate">{p.title || 'Untitled'}</div>
              <div className="flex items-center gap-3 mt-0.5 text-xs text-[var(--text-muted)]">
                <span>{p.source_tale || 'custom'}</span>
                <span className="flex items-center gap-1">
                  <Clock size={10} />
                  {new Date(p.created_at).toLocaleDateString()}
                </span>
              </div>
            </div>
            <span className={`shrink-0 text-xs px-2 py-0.5 rounded-full ${
              p.step === 'assembled' ? 'bg-[var(--success)]/15 text-[var(--success)]' :
              p.step.includes('generating') || p.step === 'assembling' ? 'bg-[var(--warning)]/15 text-[var(--warning)]' :
              'bg-[var(--bg-tertiary)] text-[var(--text-muted)]'
            }`}>
              {STEP_LABELS[p.step] || p.step}
            </span>
            <button
              onClick={e => handleDelete(e, p.project_id)}
              className="shrink-0 p-1 rounded text-[var(--text-muted)] hover:text-[var(--error)] transition-colors"
            >
              <Trash2 size={14} />
            </button>
            <ChevronRight size={14} className="text-[var(--text-muted)] shrink-0" />
          </button>
        ))}
      </div>
    </div>
  )
}
