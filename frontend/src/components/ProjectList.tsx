import { useState, useEffect } from 'react'
import { Plus, BookOpen, Trash2, Clock, ChevronRight, Search, Loader2 } from 'lucide-react'
import type { ProjectSummary, Tale, StorySearchResult } from '../api'
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

type SourceMode = 'grimm' | 'search' | 'custom'

export default function ProjectList({ onSelect }: { onSelect: (id: string) => void }) {
  const [projects, setProjects] = useState<ProjectSummary[]>([])
  const [tales, setTales] = useState<Tale[]>([])
  const [showCreate, setShowCreate] = useState(false)
  const [sourceMode, setSourceMode] = useState<SourceMode>('grimm')
  const [selectedTale, setSelectedTale] = useState('')
  const [targetMinutes, setTargetMinutes] = useState(5)
  const [ollamaModel, setOllamaModel] = useState('kimi-k2.5:cloud')
  const [tone, setTone] = useState('dark')
  const [creating, setCreating] = useState(false)

  // Story search state
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<StorySearchResult[]>([])
  const [searching, setSearching] = useState(false)
  const [selectedSearch, setSelectedSearch] = useState<StorySearchResult | null>(null)

  const refresh = () => {
    api.listProjects().then(setProjects).catch(() => {})
  }

  useEffect(() => {
    refresh()
    api.tales().then(setTales).catch(() => {})
  }, [])

  const handleSearch = async () => {
    if (!searchQuery.trim()) return
    setSearching(true)
    setSelectedSearch(null)
    try {
      const res = await api.searchStories(searchQuery)
      setSearchResults(res.results)
    } catch (e) {
      alert('Search failed: ' + (e as Error).message)
    } finally {
      setSearching(false)
    }
  }

  const handleCreate = async () => {
    setCreating(true)
    try {
      let source_tale = ''
      let custom_prompt = ''

      if (sourceMode === 'grimm') {
        source_tale = selectedTale
      } else if (sourceMode === 'search' && selectedSearch) {
        // Pass the searched story as a custom prompt with full synopsis
        custom_prompt = `Adapt this well-known story: "${selectedSearch.title}" by ${selectedSearch.author} (${selectedSearch.origin}).\n\nSynopsis: ${selectedSearch.synopsis}`
      }

      const proj = await api.createProject({
        source_tale,
        custom_prompt,
        target_minutes: targetMinutes,
        ollama_model: ollamaModel,
        tone,
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
            {/* Source mode tabs */}
            <div className="flex gap-1 p-1 rounded-lg bg-[var(--bg-tertiary)]">
              {([
                ['grimm', 'Grimm Tales'],
                ['search', 'Search Stories'],
                ['custom', 'Custom / Original'],
              ] as [SourceMode, string][]).map(([mode, label]) => (
                <button
                  key={mode}
                  onClick={() => setSourceMode(mode)}
                  className={`flex-1 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                    sourceMode === mode
                      ? 'bg-[var(--accent)] text-white'
                      : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>

            {/* Grimm Tales selector */}
            {sourceMode === 'grimm' && (
              <div>
                <label className="block text-xs text-[var(--text-secondary)] mb-1.5">Source Tale</label>
                <select
                  value={selectedTale}
                  onChange={e => setSelectedTale(e.target.value)}
                  className="w-full bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--border-focus)]"
                >
                  <option value="">Select a tale...</option>
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
            )}

            {/* Story search */}
            {sourceMode === 'search' && (
              <div className="space-y-3">
                <div>
                  <label className="block text-xs text-[var(--text-secondary)] mb-1.5">Search for Stories</label>
                  <div className="flex gap-2">
                    <input
                      type="text"
                      value={searchQuery}
                      onChange={e => setSearchQuery(e.target.value)}
                      onKeyDown={e => e.key === 'Enter' && handleSearch()}
                      placeholder="e.g. revenge, transformation, cursed prince, trickster..."
                      className="flex-1 bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--border-focus)]"
                    />
                    <button
                      onClick={handleSearch}
                      disabled={searching || !searchQuery.trim()}
                      className="flex items-center gap-2 px-4 py-2 rounded-lg bg-[var(--bg-tertiary)] border border-[var(--border)] text-sm text-[var(--text-primary)] hover:border-[var(--accent)] transition-colors disabled:opacity-50"
                    >
                      {searching ? <Loader2 size={14} className="animate-spin" /> : <Search size={14} />}
                      Search
                    </button>
                  </div>
                </div>

                {/* Search results */}
                {searchResults.length > 0 && (
                  <div className="space-y-2 max-h-64 overflow-y-auto">
                    {searchResults.map((story, i) => (
                      <button
                        key={i}
                        onClick={() => setSelectedSearch(story)}
                        className={`w-full text-left p-3 rounded-lg border transition-colors ${
                          selectedSearch?.title === story.title
                            ? 'border-[var(--accent)] bg-[var(--accent)]/10'
                            : 'border-[var(--border)] bg-[var(--bg-tertiary)] hover:border-[var(--border-focus)]'
                        }`}
                      >
                        <div className="flex items-start justify-between gap-2">
                          <div className="min-w-0">
                            <div className="text-sm font-medium truncate">{story.title}</div>
                            <div className="text-xs text-[var(--text-muted)] mt-0.5">
                              {story.author} — {story.origin}
                            </div>
                          </div>
                          <span className="shrink-0 text-[10px] px-2 py-0.5 rounded-full bg-[var(--bg-secondary)] text-[var(--text-muted)] capitalize">
                            {story.tone_suggestion}
                          </span>
                        </div>
                        <p className="text-xs text-[var(--text-secondary)] mt-1.5 leading-relaxed line-clamp-2">
                          {story.synopsis}
                        </p>
                        {story.themes.length > 0 && (
                          <div className="flex gap-1 mt-1.5 flex-wrap">
                            {story.themes.map(theme => (
                              <span key={theme} className="text-[10px] px-1.5 py-0.5 rounded bg-[var(--bg-secondary)] text-[var(--text-muted)]">
                                {theme}
                              </span>
                            ))}
                          </div>
                        )}
                      </button>
                    ))}
                  </div>
                )}

                {selectedSearch && (
                  <div className="p-3 rounded-lg border border-[var(--accent)]/30 bg-[var(--accent)]/5">
                    <div className="text-xs text-[var(--accent)] font-medium mb-1">Selected: {selectedSearch.title}</div>
                    <p className="text-xs text-[var(--text-secondary)] leading-relaxed">{selectedSearch.synopsis}</p>
                  </div>
                )}
              </div>
            )}

            {/* Custom prompt */}
            {sourceMode === 'custom' && (
              <div>
                <label className="block text-xs text-[var(--text-secondary)] mb-1.5">Story Idea</label>
                <p className="text-xs text-[var(--text-muted)] mb-2">
                  Leave empty for a completely original story, or describe your idea.
                </p>
              </div>
            )}

            {/* Adaptation tone */}
            <div>
              <label className="block text-xs text-[var(--text-secondary)] mb-1.5">Adaptation Tone</label>
              <select
                value={tone}
                onChange={e => setTone(e.target.value)}
                className="w-full bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--border-focus)]"
              >
                {ADAPTATION_TONES.map(t => (
                  <option key={t.value} value={t.value}>{t.label}</option>
                ))}
              </select>
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
              disabled={creating || (sourceMode === 'search' && !selectedSearch)}
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
