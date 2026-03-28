import { useState, useEffect } from 'react'
import { Plus, BookOpen, Trash2, Clock, ChevronRight, Search, Loader2, Globe, ChevronDown, Layers, Play, Palette } from 'lucide-react'
import type { ProjectSummary, Tale, StorySearchResult, GutenbergBook, AnalyzedChapter, VoiceProfile } from '../api'
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

const GUTENBERG_QUICK_TAGS = [
  'fairy tales', 'folklore', 'gothic fiction', 'mythology',
  'horror', 'fables', 'legends', 'ghost stories', 'fantasy',
  'dark romance', 'adventure', 'poetry',
]

const GUTENBERG_TOPICS = [
  { value: '', label: 'All topics' },
  { value: 'children', label: 'Children' },
  { value: 'crime', label: 'Crime & Mystery' },
  { value: 'fantasy', label: 'Fantasy' },
  { value: 'fiction', label: 'Fiction' },
  { value: 'horror', label: 'Horror' },
  { value: 'humor', label: 'Humor' },
  { value: 'mythology', label: 'Mythology' },
  { value: 'poetry', label: 'Poetry' },
  { value: 'romance', label: 'Romance' },
  { value: 'science fiction', label: 'Science Fiction' },
]

const GUTENBERG_LANGUAGES = [
  { value: 'en', label: 'English' },
  { value: '', label: 'All languages' },
  { value: 'de', label: 'German' },
  { value: 'fr', label: 'French' },
  { value: 'it', label: 'Italian' },
  { value: 'es', label: 'Spanish' },
  { value: 'pt', label: 'Portuguese' },
  { value: 'nl', label: 'Dutch' },
  { value: 'fi', label: 'Finnish' },
  { value: 'zh', label: 'Chinese' },
  { value: 'ja', label: 'Japanese' },
]

const STYLE_PRESETS = [
  { label: 'Tim Burton Gothic', prompt: 'Tim Burton style, dark whimsical illustration, exaggerated proportions, stark contrasts, eerie charm, spiral motifs, gothic fairy tale', loras: ['tim_burton'] },
  { label: 'Dark Fantasy', prompt: 'dark fantasy illustration, dramatic lighting, rich shadows, mythical atmosphere, intricate detail, oil painting style', loras: ['dark_fantasy'] },
  { label: 'Gothic Horror', prompt: 'dark gothic horror illustration, Victorian atmosphere, candlelit shadows, decaying grandeur, haunting beauty, sepia undertones', loras: ['dark_gothic'] },
  { label: 'Dark Storybook', prompt: 'dark fairy tale illustration, gothic storybook art, atmospheric, detailed, moody lighting, pen and ink with watercolor', loras: ['storybook'] },
  { label: 'Surreal Macabre', prompt: 'Mark Ryden style, pop surrealism, porcelain skin, unsettling beauty, meat and roses, wide-eyed figures, hyper-detailed oil painting', loras: ['mark_ryden'] },
  { label: 'Ghibli Dark', prompt: 'Studio Ghibli style, dark fairy tale mood, lush environments, melancholic atmosphere, hand-painted animation aesthetic', loras: [] },
  { label: 'No Style (default)', prompt: '', loras: [] },
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

type SourceMode = 'grimm' | 'search' | 'online' | 'custom'

export default function ProjectList({ onSelect, onBatchStart }: { onSelect: (id: string) => void; onBatchStart?: (groupId: string) => void }) {
  const [projects, setProjects] = useState<ProjectSummary[]>([])
  const [tales, setTales] = useState<Tale[]>([])
  const [showCreate, setShowCreate] = useState(false)
  const [sourceMode, setSourceMode] = useState<SourceMode>('grimm')
  const [selectedTale, setSelectedTale] = useState('')
  const [targetMinutes, setTargetMinutes] = useState(5)
  const [ollamaModel, setOllamaModel] = useState('kimi-k2.5:cloud')
  const [tone, setTone] = useState('dark')
  const [creating, setCreating] = useState(false)

  // Custom story state
  const [customPrompt, setCustomPrompt] = useState('')

  // Story search state
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<StorySearchResult[]>([])
  const [searching, setSearching] = useState(false)
  const [selectedSearch, setSelectedSearch] = useState<StorySearchResult | null>(null)

  // Gutenberg online search state
  const [onlineQuery, setOnlineQuery] = useState('')
  const [onlineResults, setOnlineResults] = useState<GutenbergBook[]>([])
  const [onlineSearching, setOnlineSearching] = useState(false)
  const [onlineResultCount, setOnlineResultCount] = useState(0)
  const [onlinePage, setOnlinePage] = useState(1)
  const [selectedOnline, setSelectedOnline] = useState<GutenbergBook | null>(null)
  const [onlinePreview, setOnlinePreview] = useState('')
  const [onlinePreviewTotal, setOnlinePreviewTotal] = useState(0)
  const [loadingPreview, setLoadingPreview] = useState(false)
  const [loadingFullText, setLoadingFullText] = useState(false)
  const [onlineTopic, setOnlineTopic] = useState('')
  const [onlineLanguage, setOnlineLanguage] = useState('en')

  // Chapter analysis state
  const [analyzedChapters, setAnalyzedChapters] = useState<AnalyzedChapter[]>([])
  const [analyzingChapters, setAnalyzingChapters] = useState(false)
  const [selectedChapters, setSelectedChapters] = useState<Set<number>>(new Set())
  const [batchSteps, setBatchSteps] = useState({ qc: false, animate: false })
  const [showChapterPanel, setShowChapterPanel] = useState(false)
  const [bookTitle, setBookTitle] = useState('')
  const [creatingBatch, setCreatingBatch] = useState(false)
  const [voiceProfiles, setVoiceProfiles] = useState<VoiceProfile[]>([])
  const [batchVoiceProfile, setBatchVoiceProfile] = useState('')
  const [batchImageBackend, setBatchImageBackend] = useState('replicate')
  const [batchStylePrompt, setBatchStylePrompt] = useState(STYLE_PRESETS[0].prompt)
  const [batchLoraKeys, setBatchLoraKeys] = useState<string[]>(STYLE_PRESETS[0].loras)
  const [selectedPreset, setSelectedPreset] = useState(0)

  // Collapsible book groups
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set())
  const [runConfigGroup, setRunConfigGroup] = useState<string | null>(null)
  const [runningGroup, setRunningGroup] = useState(false)

  const refresh = () => {
    api.listProjects().then(setProjects).catch(() => {})
  }

  useEffect(() => {
    refresh()
    api.tales().then(setTales).catch(() => {})
    api.profiles().then(p => {
      setVoiceProfiles(p)
      if (p.length > 0) setBatchVoiceProfile(p[0].id)
    }).catch(() => {})
  }, [])

  const handleSearch = async () => {
    if (!searchQuery.trim()) return
    setSearching(true)
    setSelectedSearch(null)
    try {
      const res = await api.searchStories(searchQuery, 6, ollamaModel)
      setSearchResults(res.results)
    } catch (e) {
      alert('Search failed: ' + (e as Error).message)
    } finally {
      setSearching(false)
    }
  }

  const handleOnlineSearch = async (page = 1, queryOverride?: string) => {
    const q = queryOverride ?? onlineQuery
    if (!q.trim()) return
    setOnlineSearching(true)
    setSelectedOnline(null)
    setOnlinePreview('')
    try {
      const res = await api.gutenbergSearch(q, page, onlineTopic, onlineLanguage)
      setOnlineResults(res.results)
      setOnlineResultCount(res.count)
      setOnlinePage(page)
    } catch (e) {
      alert('Gutenberg search failed: ' + (e as Error).message)
    } finally {
      setOnlineSearching(false)
    }
  }

  const handleSelectOnlineBook = async (book: GutenbergBook) => {
    setSelectedOnline(book)
    if (book.text_url) {
      setLoadingPreview(true)
      setOnlinePreview('')
      try {
        const res = await api.gutenbergText(book.text_url, 2000)
        setOnlinePreview(res.text)
        setOnlinePreviewTotal(res.total_chars)
      } catch {
        setOnlinePreview('Failed to load preview.')
      } finally {
        setLoadingPreview(false)
      }
    }
  }

  const handleUseFullText = async () => {
    if (!selectedOnline?.text_url) return
    setLoadingFullText(true)
    try {
      const res = await api.gutenbergText(selectedOnline.text_url, 0)
      setCustomPrompt(res.text)
      setSourceMode('custom')
    } catch (e) {
      alert('Failed to fetch full text: ' + (e as Error).message)
    } finally {
      setLoadingFullText(false)
    }
  }

  const handleAnalyzeChapters = async () => {
    if (!selectedOnline?.text_url) return
    setAnalyzingChapters(true)
    setAnalyzedChapters([])
    setShowChapterPanel(false)
    try {
      // Fetch full text first
      const textRes = await api.gutenbergText(selectedOnline.text_url, 0)
      // Send to LLM for analysis
      const res = await api.analyzeChapters(textRes.text, selectedOnline.title, ollamaModel)
      setAnalyzedChapters(res.chapters)
      setBookTitle(res.book_title)
      setSelectedChapters(new Set(res.chapters.map((_, i) => i)))
      setShowChapterPanel(true)
    } catch (e) {
      alert('Chapter analysis failed: ' + (e as Error).message)
    } finally {
      setAnalyzingChapters(false)
    }
  }

  const handleSaveBatch = async () => {
    if (selectedChapters.size === 0) return
    setCreatingBatch(true)
    try {
      const chaptersToCreate = analyzedChapters.filter((_, i) => selectedChapters.has(i))
      await api.batchCreate({
        book_title: bookTitle,
        chapters: chaptersToCreate,
        ollama_model: ollamaModel,
        voice_profile_id: batchVoiceProfile || undefined,
        voice_language: 'en',
        image_backend: batchImageBackend,
      })
      setShowChapterPanel(false)
      setAnalyzedChapters([])
      setShowCreate(false)
      refresh()
    } catch (e) {
      alert('Batch save failed: ' + (e as Error).message)
    } finally {
      setCreatingBatch(false)
    }
  }

  const handleCreateBatch = async () => {
    if (selectedChapters.size === 0) return
    setCreatingBatch(true)
    try {
      const chaptersToCreate = analyzedChapters.filter((_, i) => selectedChapters.has(i))
      const steps = ['script', 'voice', 'images', ...(batchSteps.qc ? ['qc'] : []), ...(batchSteps.animate ? ['animate'] : []), 'assemble']

      const createRes = await api.batchCreate({
        book_title: bookTitle,
        chapters: chaptersToCreate,
        ollama_model: ollamaModel,
        voice_profile_id: batchVoiceProfile || undefined,
        voice_language: 'en',
        image_backend: batchImageBackend,
      })

      await api.batchRun(createRes.book_group_id, {
        steps,
        voice_profile_id: batchVoiceProfile,
        voice_language: 'en',
        image_backend: batchImageBackend,
        ...(batchStylePrompt && { style_prompt: batchStylePrompt }),
        ...(batchLoraKeys.length > 0 && { lora_keys: batchLoraKeys }),
      })

      if (onBatchStart) {
        onBatchStart(createRes.book_group_id)
      }

      setShowChapterPanel(false)
      setAnalyzedChapters([])
      setShowCreate(false)
      refresh()
    } catch (e) {
      alert('Batch creation failed: ' + (e as Error).message)
    } finally {
      setCreatingBatch(false)
    }
  }

  const handleRunGroup = async (groupId: string) => {
    setRunningGroup(true)
    try {
      const steps = ['script', 'voice', 'images', ...(batchSteps.qc ? ['qc'] : []), ...(batchSteps.animate ? ['animate'] : []), 'assemble']
      await api.batchRun(groupId, {
        steps,
        voice_profile_id: batchVoiceProfile,
        voice_language: 'en',
        image_backend: batchImageBackend,
        ...(batchStylePrompt && { style_prompt: batchStylePrompt }),
        ...(batchLoraKeys.length > 0 && { lora_keys: batchLoraKeys }),
      })
      setRunConfigGroup(null)
      if (onBatchStart) {
        onBatchStart(groupId)
      }
    } catch (e) {
      alert('Batch run failed: ' + (e as Error).message)
    } finally {
      setRunningGroup(false)
    }
  }

  const toggleChapter = (index: number) => {
    setSelectedChapters(prev => {
      const next = new Set(prev)
      if (next.has(index)) next.delete(index)
      else next.add(index)
      return next
    })
  }

  const toggleGroup = (groupId: string) => {
    setExpandedGroups(prev => {
      const next = new Set(prev)
      if (next.has(groupId)) next.delete(groupId)
      else next.add(groupId)
      return next
    })
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
      } else if (sourceMode === 'online' && selectedOnline) {
        const authorStr = selectedOnline.authors.map(a => a.name).join(', ') || 'Unknown'
        custom_prompt = `Adapt this public domain story from Project Gutenberg: "${selectedOnline.title}" by ${authorStr}.\n\nSubjects: ${selectedOnline.subjects.join(', ')}\n\nText preview:\n${onlinePreview}`
      } else if (sourceMode === 'custom') {
        custom_prompt = customPrompt
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
                ['search', 'AI Suggestions'],
                ['online', 'Search Online'],
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

            {/* Gutenberg online search */}
            {sourceMode === 'online' && (
              <div className="space-y-3">
                <div>
                  <label className="block text-xs text-[var(--text-secondary)] mb-1.5">
                    <Globe size={12} className="inline mr-1" />
                    Search Project Gutenberg
                  </label>
                  <p className="text-xs text-[var(--text-muted)] mb-2">
                    Search 70,000+ free public domain books — fairy tales, folklore, classic fiction, and more.
                  </p>

                  {/* Quick-search tags */}
                  <div className="flex flex-wrap gap-1.5 mb-2">
                    {GUTENBERG_QUICK_TAGS.map(tag => (
                      <button
                        key={tag}
                        onClick={() => { setOnlineQuery(tag); handleOnlineSearch(1, tag) }}
                        className={`text-[10px] px-2 py-1 rounded-full border transition-colors ${
                          onlineQuery === tag
                            ? 'border-[var(--accent)] bg-[var(--accent)]/15 text-[var(--accent)]'
                            : 'border-[var(--border)] bg-[var(--bg-tertiary)] text-[var(--text-muted)] hover:border-[var(--accent)] hover:text-[var(--text-secondary)]'
                        }`}
                      >
                        {tag}
                      </button>
                    ))}
                  </div>

                  <div className="flex gap-2">
                    <input
                      type="text"
                      value={onlineQuery}
                      onChange={e => setOnlineQuery(e.target.value)}
                      onKeyDown={e => e.key === 'Enter' && handleOnlineSearch()}
                      placeholder="e.g. fairy tale, grimm, andersen, gothic horror..."
                      className="flex-1 bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--border-focus)]"
                    />
                    <button
                      onClick={() => handleOnlineSearch()}
                      disabled={onlineSearching || !onlineQuery.trim()}
                      className="flex items-center gap-2 px-4 py-2 rounded-lg bg-[var(--bg-tertiary)] border border-[var(--border)] text-sm text-[var(--text-primary)] hover:border-[var(--accent)] transition-colors disabled:opacity-50"
                    >
                      {onlineSearching ? <Loader2 size={14} className="animate-spin" /> : <Search size={14} />}
                      Search
                    </button>
                  </div>

                  {/* Topic + Language filters */}
                  <div className="flex gap-2 mt-2">
                    <select
                      value={onlineTopic}
                      onChange={e => setOnlineTopic(e.target.value)}
                      className="flex-1 bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg px-2 py-1.5 text-xs text-[var(--text-primary)] focus:outline-none focus:border-[var(--border-focus)]"
                    >
                      {GUTENBERG_TOPICS.map(t => (
                        <option key={t.value} value={t.value}>{t.label}</option>
                      ))}
                    </select>
                    <select
                      value={onlineLanguage}
                      onChange={e => setOnlineLanguage(e.target.value)}
                      className="flex-1 bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg px-2 py-1.5 text-xs text-[var(--text-primary)] focus:outline-none focus:border-[var(--border-focus)]"
                    >
                      {GUTENBERG_LANGUAGES.map(l => (
                        <option key={l.value} value={l.value}>{l.label}</option>
                      ))}
                    </select>
                  </div>

                  {onlineResultCount > 0 && (
                    <div className="text-[10px] text-[var(--text-muted)] mt-1">
                      {onlineResultCount.toLocaleString()} results found
                    </div>
                  )}
                </div>

                {/* Results list */}
                {onlineResults.length > 0 && (
                  <div className="space-y-2 max-h-64 overflow-y-auto">
                    {onlineResults.map(book => (
                      <button
                        key={book.gutenberg_id}
                        onClick={() => handleSelectOnlineBook(book)}
                        className={`w-full text-left p-3 rounded-lg border transition-colors ${
                          selectedOnline?.gutenberg_id === book.gutenberg_id
                            ? 'border-[var(--accent)] bg-[var(--accent)]/10'
                            : 'border-[var(--border)] bg-[var(--bg-tertiary)] hover:border-[var(--border-focus)]'
                        }`}
                      >
                        <div className="flex items-start justify-between gap-2">
                          <div className="min-w-0">
                            <div className="text-sm font-medium truncate">{book.title}</div>
                            <div className="text-xs text-[var(--text-muted)] mt-0.5">
                              {book.authors.map(a => a.name).join(', ') || 'Unknown'}
                              {book.authors[0]?.birth_year && ` (${book.authors[0].birth_year}\u2013${book.authors[0].death_year || '?'})`}
                            </div>
                          </div>
                          <span className="shrink-0 text-[10px] px-2 py-0.5 rounded-full bg-[var(--bg-secondary)] text-[var(--text-muted)]">
                            {book.download_count.toLocaleString()} downloads
                          </span>
                        </div>
                        {book.subjects.length > 0 && (
                          <div className="flex gap-1 mt-1.5 flex-wrap">
                            {book.subjects.slice(0, 4).map(s => (
                              <span key={s} className="text-[10px] px-1.5 py-0.5 rounded bg-[var(--bg-secondary)] text-[var(--text-muted)]">
                                {s}
                              </span>
                            ))}
                            {book.subjects.length > 4 && (
                              <span className="text-[10px] text-[var(--text-muted)]">
                                +{book.subjects.length - 4} more
                              </span>
                            )}
                          </div>
                        )}
                        {!book.text_url && (
                          <div className="text-[10px] text-[var(--warning)] mt-1">No plain text available</div>
                        )}
                      </button>
                    ))}

                    {/* Pagination */}
                    {onlineResultCount > 32 && (
                      <div className="flex items-center justify-between pt-2">
                        <button
                          disabled={onlinePage <= 1}
                          onClick={() => handleOnlineSearch(onlinePage - 1)}
                          className="text-xs px-3 py-1 rounded bg-[var(--bg-tertiary)] border border-[var(--border)] text-[var(--text-secondary)] hover:border-[var(--accent)] transition-colors disabled:opacity-30"
                        >
                          Previous
                        </button>
                        <span className="text-xs text-[var(--text-muted)]">Page {onlinePage}</span>
                        <button
                          onClick={() => handleOnlineSearch(onlinePage + 1)}
                          className="text-xs px-3 py-1 rounded bg-[var(--bg-tertiary)] border border-[var(--border)] text-[var(--text-secondary)] hover:border-[var(--accent)] transition-colors"
                        >
                          Next
                        </button>
                      </div>
                    )}
                  </div>
                )}

                {/* Preview panel */}
                {selectedOnline && (
                  <div className="p-3 rounded-lg border border-[var(--accent)]/30 bg-[var(--accent)]/5">
                    <div className="text-xs text-[var(--accent)] font-medium mb-1">
                      Selected: {selectedOnline.title}
                    </div>
                    {loadingPreview ? (
                      <div className="flex items-center gap-2 text-xs text-[var(--text-muted)] py-2">
                        <Loader2 size={12} className="animate-spin" /> Loading preview...
                      </div>
                    ) : onlinePreview ? (
                      <>
                        <p className="text-xs text-[var(--text-secondary)] leading-relaxed whitespace-pre-wrap max-h-32 overflow-y-auto">
                          {onlinePreview}{onlinePreviewTotal > 2000 && '...'}
                        </p>
                        <div className="flex items-center justify-between mt-2">
                          <span className="text-[10px] text-[var(--text-muted)]">
                            {onlinePreviewTotal.toLocaleString()} characters total
                          </span>
                          <div className="flex gap-2">
                            {onlinePreviewTotal > 5000 && (
                              <button
                                onClick={handleAnalyzeChapters}
                                disabled={analyzingChapters}
                                className="flex items-center gap-1 text-[10px] px-2 py-1 rounded bg-[var(--accent)]/10 border border-[var(--accent)]/30 text-[var(--accent)] hover:bg-[var(--accent)]/20 transition-colors disabled:opacity-50"
                              >
                                {analyzingChapters ? <Loader2 size={10} className="animate-spin" /> : <Layers size={10} />}
                                {analyzingChapters ? 'Analyzing...' : 'Analyze Chapters'}
                              </button>
                            )}
                            <button
                              onClick={handleUseFullText}
                              disabled={loadingFullText}
                              className="text-[10px] px-2 py-1 rounded bg-[var(--bg-tertiary)] border border-[var(--border)] text-[var(--text-primary)] hover:border-[var(--accent)] transition-colors disabled:opacity-50"
                            >
                              {loadingFullText ? 'Loading...' : 'Edit Full Text in Custom Tab'}
                            </button>
                          </div>
                        </div>
                      </>
                    ) : !selectedOnline.text_url ? (
                      <p className="text-xs text-[var(--text-muted)]">No plain text available for this book.</p>
                    ) : null}
                  </div>
                )}

                {/* Chapter analysis results */}
                {showChapterPanel && analyzedChapters.length > 0 && (
                  <div className="p-4 rounded-lg border border-[var(--border)] bg-[var(--bg-tertiary)]">
                    <div className="flex items-center justify-between mb-3">
                      <div>
                        <h4 className="text-sm font-medium">{bookTitle}</h4>
                        <p className="text-[10px] text-[var(--text-muted)]">
                          {analyzedChapters.length} chapters detected &middot; {selectedChapters.size} selected
                        </p>
                      </div>
                      <div className="flex gap-2">
                        <button
                          onClick={() => setSelectedChapters(new Set(analyzedChapters.map((_, i) => i)))}
                          className="text-[10px] px-2 py-0.5 rounded bg-[var(--bg-secondary)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
                        >
                          Select all
                        </button>
                        <button
                          onClick={() => setSelectedChapters(new Set())}
                          className="text-[10px] px-2 py-0.5 rounded bg-[var(--bg-secondary)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
                        >
                          Deselect all
                        </button>
                      </div>
                    </div>

                    {/* Chapter list */}
                    <div className="space-y-1.5 max-h-64 overflow-y-auto mb-3">
                      {analyzedChapters.map((ch, i) => (
                        <label
                          key={i}
                          className={`flex items-start gap-2.5 p-2 rounded-lg border cursor-pointer transition-colors ${
                            selectedChapters.has(i)
                              ? 'border-[var(--accent)]/40 bg-[var(--accent)]/5'
                              : 'border-transparent bg-[var(--bg-secondary)]'
                          }`}
                        >
                          <input
                            type="checkbox"
                            checked={selectedChapters.has(i)}
                            onChange={() => toggleChapter(i)}
                            className="mt-0.5 accent-[var(--accent)]"
                          />
                          <div className="flex-1 min-w-0">
                            <div className="text-xs font-medium truncate">{ch.title}</div>
                            <div className="flex items-center gap-3 mt-0.5 text-[10px] text-[var(--text-muted)]">
                              <span>{ch.char_count.toLocaleString()} chars</span>
                              <span>~{ch.estimated_duration} min</span>
                              <span className="capitalize">{ch.suggested_tone}</span>
                            </div>
                          </div>
                        </label>
                      ))}
                    </div>

                    {/* Pipeline step toggles */}
                    <div className="flex items-center gap-4 mb-3 p-2 rounded bg-[var(--bg-secondary)]">
                      <span className="text-[10px] text-[var(--text-muted)] font-medium">Pipeline:</span>
                      <span className="text-[10px] text-[var(--text-secondary)]">Script</span>
                      <span className="text-[10px] text-[var(--text-muted)]">&rarr;</span>
                      <span className="text-[10px] text-[var(--text-secondary)]">Voice</span>
                      <span className="text-[10px] text-[var(--text-muted)]">&rarr;</span>
                      <span className="text-[10px] text-[var(--text-secondary)]">Images</span>
                      <span className="text-[10px] text-[var(--text-muted)]">&rarr;</span>
                      <label className="flex items-center gap-1 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={batchSteps.qc}
                          onChange={e => setBatchSteps(s => ({ ...s, qc: e.target.checked }))}
                          className="accent-[var(--accent)]"
                        />
                        <span className={`text-[10px] ${batchSteps.qc ? 'text-[var(--text-secondary)]' : 'text-[var(--text-muted)]'}`}>QC</span>
                      </label>
                      <span className="text-[10px] text-[var(--text-muted)]">&rarr;</span>
                      <label className="flex items-center gap-1 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={batchSteps.animate}
                          onChange={e => setBatchSteps(s => ({ ...s, animate: e.target.checked }))}
                          className="accent-[var(--accent)]"
                        />
                        <span className={`text-[10px] ${batchSteps.animate ? 'text-[var(--text-secondary)]' : 'text-[var(--text-muted)]'}`}>Animate</span>
                      </label>
                      <span className="text-[10px] text-[var(--text-muted)]">&rarr;</span>
                      <span className="text-[10px] text-[var(--text-secondary)]">Assemble</span>
                    </div>

                    {/* Batch settings */}
                    <div className="grid grid-cols-2 gap-2 mb-3">
                      <div>
                        <label className="block text-[10px] text-[var(--text-muted)] mb-1">Voice Profile</label>
                        <select
                          value={batchVoiceProfile}
                          onChange={e => setBatchVoiceProfile(e.target.value)}
                          className="w-full bg-[var(--bg-secondary)] border border-[var(--border)] rounded px-2 py-1 text-xs text-[var(--text-primary)] focus:outline-none focus:border-[var(--border-focus)]"
                        >
                          <option value="">Select voice...</option>
                          {voiceProfiles.map(p => (
                            <option key={p.id} value={p.id}>{p.name} ({p.language})</option>
                          ))}
                        </select>
                      </div>
                      <div>
                        <label className="block text-[10px] text-[var(--text-muted)] mb-1">Image Backend</label>
                        <select
                          value={batchImageBackend}
                          onChange={e => setBatchImageBackend(e.target.value)}
                          className="w-full bg-[var(--bg-secondary)] border border-[var(--border)] rounded px-2 py-1 text-xs text-[var(--text-primary)] focus:outline-none focus:border-[var(--border-focus)]"
                        >
                          <option value="replicate">Replicate (FLUX)</option>
                          <option value="comfyui">ComfyUI (local)</option>
                          <option value="ollama">Ollama (local)</option>
                        </select>
                      </div>
                    </div>

                    {/* Image style prompt */}
                    <div className="mb-3">
                      <label className="flex items-center gap-1.5 text-[10px] text-[var(--text-muted)] font-medium mb-1.5">
                        <Palette size={10} /> Image Style
                      </label>
                      <div className="flex flex-wrap gap-1 mb-2">
                        {STYLE_PRESETS.map((preset, i) => (
                          <button
                            key={preset.label}
                            onClick={() => { setSelectedPreset(i); setBatchStylePrompt(preset.prompt); setBatchLoraKeys(preset.loras) }}
                            className={`text-[10px] px-2 py-0.5 rounded-full border transition-colors ${
                              selectedPreset === i
                                ? 'border-[var(--accent)] bg-[var(--accent)]/15 text-[var(--accent)]'
                                : 'border-[var(--border)] text-[var(--text-muted)] hover:text-[var(--text-secondary)] hover:border-[var(--border-focus)]'
                            }`}
                          >
                            {preset.label}
                          </button>
                        ))}
                      </div>
                      <textarea
                        value={batchStylePrompt}
                        onChange={e => { setBatchStylePrompt(e.target.value); setSelectedPreset(-1) }}
                        placeholder="Custom style prompt for image generation..."
                        rows={2}
                        className="w-full bg-[var(--bg-secondary)] border border-[var(--border)] rounded px-2 py-1.5 text-xs text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:outline-none focus:border-[var(--border-focus)] resize-none"
                      />
                      {batchLoraKeys.length > 0 && (
                        <p className="text-[10px] text-[var(--text-muted)] mt-1">LoRA: {batchLoraKeys.join(', ')}</p>
                      )}
                    </div>

                    {/* Save / Save & Run buttons */}
                    <div className="flex gap-2">
                      <button
                        onClick={handleSaveBatch}
                        disabled={creatingBatch || selectedChapters.size === 0}
                        className="flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-lg border border-[var(--accent)] text-[var(--accent)] text-sm font-medium transition-colors hover:bg-[var(--accent)]/10 disabled:opacity-50"
                      >
                        {creatingBatch ? (
                          <><Loader2 size={14} className="animate-spin" /> Saving...</>
                        ) : (
                          <><Layers size={14} /> Save {selectedChapters.size} Chapters</>
                        )}
                      </button>
                      <button
                        onClick={handleCreateBatch}
                        disabled={creatingBatch || selectedChapters.size === 0 || !batchVoiceProfile}
                        className="flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white text-sm font-medium transition-colors disabled:opacity-50"
                      >
                        {creatingBatch ? (
                          <><Loader2 size={14} className="animate-spin" /> Creating...</>
                        ) : (
                          <><Play size={14} /> Save & Run {selectedChapters.size}</>
                        )}
                      </button>
                    </div>
                    {!batchVoiceProfile && (
                      <p className="text-[10px] text-[var(--warning)] mt-1">Select a voice profile to run immediately</p>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* Custom prompt */}
            {sourceMode === 'custom' && (
              <div>
                <label className="block text-xs text-[var(--text-secondary)] mb-1.5">Story Text or Idea</label>
                <p className="text-xs text-[var(--text-muted)] mb-2">
                  Paste a full story text and the LLM will break it into scenes, or describe an idea for an original story. Leave empty for a completely AI-generated tale.
                </p>
                <textarea
                  value={customPrompt}
                  onChange={e => setCustomPrompt(e.target.value)}
                  placeholder="Paste your story here, or describe an idea...&#10;&#10;e.g. a full fairy tale text, a plot outline, or just a theme like 'a story about a cursed mirror that shows the truth'"
                  rows={8}
                  className="w-full bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--border-focus)] resize-y leading-relaxed"
                />
                {customPrompt.length > 0 && (
                  <div className="mt-1.5 text-[10px] text-[var(--text-muted)]">
                    {customPrompt.length.toLocaleString()} characters
                  </div>
                )}
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
              disabled={creating || (sourceMode === 'search' && !selectedSearch) || (sourceMode === 'online' && !selectedOnline)}
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
        {(() => {
          // Group projects: book groups first, then standalone projects
          const groups = new Map<string, ProjectSummary[]>()
          const standalone: ProjectSummary[] = []
          for (const p of projects) {
            if (p.book_group_id) {
              const list = groups.get(p.book_group_id) || []
              list.push(p)
              groups.set(p.book_group_id, list)
            } else {
              standalone.push(p)
            }
          }
          // Sort chapters within each group
          for (const list of groups.values()) {
            list.sort((a, b) => (a.chapter_index ?? 0) - (b.chapter_index ?? 0))
          }

          const renderProject = (p: ProjectSummary, indent = false) => (
            <button
              key={p.project_id}
              onClick={() => onSelect(p.project_id)}
              className={`w-full flex items-center gap-4 p-4 rounded-lg border border-[var(--border)] bg-[var(--bg-secondary)] hover:bg-[var(--bg-hover)] transition-colors text-left ${indent ? 'ml-6' : ''}`}
            >
              <BookOpen size={18} className="text-[var(--accent)] shrink-0" />
              <div className="flex-1 min-w-0">
                <div className="font-medium text-sm truncate">
                  {indent && p.chapter_index != null && <span className="text-[var(--text-muted)] mr-1.5">Ch. {p.chapter_index + 1}</span>}
                  {p.title || 'Untitled'}
                </div>
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
          )

          return (
            <>
              {/* Book groups */}
              {[...groups.entries()].map(([groupId, chapters]) => {
                const expanded = expandedGroups.has(groupId)
                const completedCount = chapters.filter(c => c.step === 'assembled').length
                const unprocessedCount = chapters.filter(c => c.step === 'created').length
                const groupTitle = chapters[0]?.source_tale || chapters[0]?.title || 'Book'
                const showRunConfig = runConfigGroup === groupId
                return (
                  <div key={groupId} className="rounded-xl border border-[var(--border)] overflow-hidden">
                    <button
                      onClick={() => toggleGroup(groupId)}
                      className="w-full flex items-center gap-3 p-4 bg-[var(--bg-secondary)] hover:bg-[var(--bg-hover)] transition-colors text-left"
                    >
                      <Layers size={18} className="text-[var(--accent)] shrink-0" />
                      <div className="flex-1 min-w-0">
                        <div className="font-medium text-sm truncate">{groupTitle}</div>
                        <div className="text-xs text-[var(--text-muted)] mt-0.5">
                          {chapters.length} chapters &middot; {completedCount} complete{unprocessedCount > 0 && ` · ${unprocessedCount} saved`}
                        </div>
                      </div>
                      <div className="flex items-center gap-2 shrink-0" onClick={e => e.stopPropagation()}>
                        {unprocessedCount > 0 && (
                          <button
                            onClick={() => setRunConfigGroup(showRunConfig ? null : groupId)}
                            className="text-[10px] px-2.5 py-1 rounded-full bg-[var(--accent)] text-white hover:bg-[var(--accent-hover)] transition-colors"
                          >
                            Start Processing
                          </button>
                        )}
                        {onBatchStart && completedCount < chapters.length && (
                          <button
                            onClick={() => onBatchStart(groupId)}
                            className="text-[10px] px-2.5 py-1 rounded-full bg-[var(--accent)]/10 text-[var(--accent)] hover:bg-[var(--accent)]/20 transition-colors"
                          >
                            View Progress
                          </button>
                        )}
                      </div>
                      <ChevronDown size={16} className={`text-[var(--text-muted)] shrink-0 transition-transform ${expanded ? '' : '-rotate-90'}`} />
                    </button>

                    {/* Run config panel */}
                    {showRunConfig && (
                      <div className="p-4 border-t border-[var(--border)] bg-[var(--bg-secondary)] space-y-3">
                        <div className="flex items-center gap-4 p-2 rounded bg-[var(--bg-tertiary)]">
                          <span className="text-[10px] text-[var(--text-muted)] font-medium">Pipeline:</span>
                          <span className="text-[10px] text-[var(--text-secondary)]">Script → Voice → Images →</span>
                          <label className="flex items-center gap-1 cursor-pointer">
                            <input type="checkbox" checked={batchSteps.qc} onChange={e => setBatchSteps(s => ({ ...s, qc: e.target.checked }))} className="accent-[var(--accent)]" />
                            <span className={`text-[10px] ${batchSteps.qc ? 'text-[var(--text-secondary)]' : 'text-[var(--text-muted)]'}`}>QC</span>
                          </label>
                          <span className="text-[10px] text-[var(--text-muted)]">→</span>
                          <label className="flex items-center gap-1 cursor-pointer">
                            <input type="checkbox" checked={batchSteps.animate} onChange={e => setBatchSteps(s => ({ ...s, animate: e.target.checked }))} className="accent-[var(--accent)]" />
                            <span className={`text-[10px] ${batchSteps.animate ? 'text-[var(--text-secondary)]' : 'text-[var(--text-muted)]'}`}>Animate</span>
                          </label>
                          <span className="text-[10px] text-[var(--text-muted)]">→ Assemble</span>
                        </div>
                        <div className="grid grid-cols-2 gap-2">
                          <div>
                            <label className="block text-[10px] text-[var(--text-muted)] mb-1">Voice Profile</label>
                            <select value={batchVoiceProfile} onChange={e => setBatchVoiceProfile(e.target.value)} className="w-full bg-[var(--bg-tertiary)] border border-[var(--border)] rounded px-2 py-1 text-xs text-[var(--text-primary)] focus:outline-none focus:border-[var(--border-focus)]">
                              <option value="">Select voice...</option>
                              {voiceProfiles.map(p => <option key={p.id} value={p.id}>{p.name} ({p.language})</option>)}
                            </select>
                          </div>
                          <div>
                            <label className="block text-[10px] text-[var(--text-muted)] mb-1">Image Backend</label>
                            <select value={batchImageBackend} onChange={e => setBatchImageBackend(e.target.value)} className="w-full bg-[var(--bg-tertiary)] border border-[var(--border)] rounded px-2 py-1 text-xs text-[var(--text-primary)] focus:outline-none focus:border-[var(--border-focus)]">
                              <option value="replicate">Replicate (FLUX)</option>
                              <option value="comfyui">ComfyUI (local)</option>
                              <option value="ollama">Ollama (local)</option>
                            </select>
                          </div>
                        </div>
                        {/* Image style prompt */}
                        <div>
                          <label className="flex items-center gap-1.5 text-[10px] text-[var(--text-muted)] font-medium mb-1.5">
                            <Palette size={10} /> Image Style
                          </label>
                          <div className="flex flex-wrap gap-1 mb-2">
                            {STYLE_PRESETS.map((preset, i) => (
                              <button
                                key={preset.label}
                                onClick={() => { setSelectedPreset(i); setBatchStylePrompt(preset.prompt); setBatchLoraKeys(preset.loras) }}
                                className={`text-[10px] px-2 py-0.5 rounded-full border transition-colors ${
                                  selectedPreset === i
                                    ? 'border-[var(--accent)] bg-[var(--accent)]/15 text-[var(--accent)]'
                                    : 'border-[var(--border)] text-[var(--text-muted)] hover:text-[var(--text-secondary)] hover:border-[var(--border-focus)]'
                                }`}
                              >
                                {preset.label}
                              </button>
                            ))}
                          </div>
                          <textarea
                            value={batchStylePrompt}
                            onChange={e => { setBatchStylePrompt(e.target.value); setSelectedPreset(-1) }}
                            placeholder="Custom style prompt for image generation..."
                            rows={2}
                            className="w-full bg-[var(--bg-tertiary)] border border-[var(--border)] rounded px-2 py-1.5 text-xs text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:outline-none focus:border-[var(--border-focus)] resize-none"
                          />
                          {batchLoraKeys.length > 0 && (
                            <p className="text-[10px] text-[var(--text-muted)] mt-1">LoRA: {batchLoraKeys.join(', ')}</p>
                          )}
                        </div>
                        <button
                          onClick={() => handleRunGroup(groupId)}
                          disabled={runningGroup || !batchVoiceProfile}
                          className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white text-sm font-medium transition-colors disabled:opacity-50"
                        >
                          {runningGroup ? <><Loader2 size={14} className="animate-spin" /> Starting...</> : <><Play size={14} /> Run {chapters.length} Chapters</>}
                        </button>
                        {!batchVoiceProfile && <p className="text-[10px] text-[var(--warning)]">Select a voice profile to continue</p>}
                      </div>
                    )}

                    {expanded && (
                      <div className="space-y-1 p-2 bg-[var(--bg-tertiary)]">
                        {chapters.map(p => renderProject(p, true))}
                      </div>
                    )}
                  </div>
                )
              })}
              {/* Standalone projects */}
              {standalone.map(p => renderProject(p))}
            </>
          )
        })()}
      </div>
    </div>
  )
}
