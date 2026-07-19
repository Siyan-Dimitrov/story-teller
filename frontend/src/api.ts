// ── Types ────────────────────────────────────────────────────

export interface HealthStatus {
  claude?: boolean
  comfyui: boolean
  replicate: boolean
  openai: boolean
  ffmpeg: boolean
}

export interface CastMember {
  id: string
  name: string
  role?: string
  description: string
  reference_prompt?: string
  reference_image_path?: string | null
  reference_image_error?: string
}

export interface Scene {
  index: number
  narration: string
  image_prompt: string
  image_prompts?: string[]
  characters?: string[]
  mood: string
  duration_hint: number
  audio_path?: string | null
  audio_duration?: number | null
  image_path?: string | null
  image_paths?: string[]
  image_errors?: (string | null)[]
  image_safety_error?: boolean
  kb_effect: string
  animation_types?: string[]
  motion_presets?: string[]
  depth_map_paths?: (string | null)[]
  animatediff_clip_paths?: (string | null)[]
  voice_error?: string
  voice_id?: string
  emotion?: string
  image_error?: string
  music_track?: string | null
  music_volume?: number | null
}

export interface Script {
  title: string
  synopsis: string
  cast?: CastMember[]
  scenes: Scene[]
  target_minutes?: number
  source_tale?: string
  tone?: string
}

export interface ShortSuggestion {
  scene_index: number
  hook: string
  reason: string
  score: number
}

export interface ShortItem {
  scene_index: number
  path: string
  duration: number
  hook: string
  pinned_comment?: string
}

export interface ImagesProgress {
  active: boolean
  phase: string
  generated: number
  total: number
  scene_index?: number | null
  error: string | null
}

export interface ShortsProgress {
  active: boolean
  done: number
  total: number
  error: string | null
  shorts: ShortItem[]
  stage?: string
}

export interface ProjectState {
  project_id: string
  step: string
  error?: string | null
  title: string
  source_tale: string
  voice_profile_id?: string | null
  voice_language: string
  claude_model?: string | null
  pipeline_writer_model?: string | null
  pipeline_critic_model?: string | null
  pipeline_reviser_model?: string | null
  image_backend: string
  target_minutes: number
  suggested_length?: string
  tone?: string
  created_at: string
  script?: Script
  output_dir?: string | null
  char_count?: number
  estimated_duration?: number
  music_track?: string | null
  music_volume?: number | null
}

export interface ProjectSummary {
  project_id: string
  title: string
  step: string
  source_tale: string
  created_at: string
  book_group_id?: string | null
  chapter_index?: number | null
  tone: string
  target_minutes: number
  suggested_length?: string
  estimated_duration: number
  char_count: number
}

export interface Tale {
  id: string
  title: string
  origin: string
  description: string
  themes: string[]
  synopsis?: string
}

export interface VoiceProfile {
  id: string
  name: string
  language: string
}

export interface StorySearchResult {
  title: string
  author: string
  origin: string
  synopsis: string
  themes: string[]
  tone_suggestion: string
}

export interface GutenbergAuthor {
  name: string
  birth_year: number | null
  death_year: number | null
}

export interface GutenbergBook {
  gutenberg_id: number
  title: string
  authors: GutenbergAuthor[]
  subjects: string[]
  bookshelves: string[]
  languages: string[]
  download_count: number
  text_url: string | null
}

export interface GutenbergSearchResponse {
  count: number
  next: string | null
  previous: string | null
  results: GutenbergBook[]
}

export interface GutenbergTextResponse {
  text: string
  total_chars: number
  truncated: boolean
}

// ── Batch chapter types ──────────────────────────────────────

export interface AnalyzedChapter {
  title: string
  text: string
  suggested_tone: string
  summary: string
  estimated_duration: number
  char_count: number
  parts: number
}

export interface AnalyzeChaptersResponse {
  book_title: string
  chapters: AnalyzedChapter[]
}

export interface BatchCreateResponse {
  book_group_id: string
  project_ids: string[]
}

export interface ChapterProgress {
  project_id: string
  chapter_index: number
  title: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  current_step?: string | null
  failed_step?: string | null
  error?: string | null
}

export interface BatchProgress {
  group_id: string
  total: number
  completed: number
  failed: number
  current_chapter?: number | null
  current_step?: string | null
  chapters: ChapterProgress[]
  finished: boolean
  paused: boolean
}

// ── HTTP client ─────────────────────────────────────────────

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, options)
  if (!res.ok) {
    const text = await res.text().catch(() => 'Unknown error')
    throw new Error(text)
  }
  return res.json() as Promise<T>
}

function post<T>(url: string, body?: unknown): Promise<T> {
  return request<T>(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  })
}

function put<T>(url: string, body: unknown): Promise<T> {
  return request<T>(url, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

function del<T>(url: string): Promise<T> {
  return request<T>(url, { method: 'DELETE' })
}

// ── API ─────────────────────────────────────────────────────

export interface LoraInfo {
  trigger: string
  file: string
  has_flux: boolean
  description?: string
}

export interface LorasResponse {
  available: Record<string, LoraInfo>
  defaults: string[]
}

export interface ImageStyle {
  id: string
  label: string
  description: string
  supports_loras: boolean
  default_lora_keys?: string[]
}

export interface ImageStylesResponse {
  styles: ImageStyle[]
  default_style_id: string
}

export interface MusicTrack {
  id: string
  title: string
  name?: string
  path?: string
  size_bytes?: number
  source: 'local' | 'jamendo'
  artist?: string
  duration?: number
  url?: string
  license?: string
}

export interface MusicListResponse {
  available: MusicTrack[]
  default_volume: number
  music_dir: string
  jamendo_enabled: boolean
}

export const api = {
  health: () => request<HealthStatus>('/api/health'),
  tales: () => request<Tale[]>('/api/tales'),
  tale: (id: string) => request<Tale>(`/api/tales/${id}`),
  profiles: () => request<VoiceProfile[]>('/api/profiles'),
  loras: () => request<LorasResponse>('/api/loras'),
  imageStyles: () => request<ImageStylesResponse>('/api/image-styles'),
  music: () => request<MusicListResponse>('/api/music'),
  musicSearch: (query: string, limit?: number) =>
    request<{ query: string; results: MusicTrack[] }>(`/api/music/search?query=${encodeURIComponent(query)}&limit=${limit || 8}`),
  musicDownload: (url: string) =>
    post<{ name: string; path: string; size_bytes: number }>('/api/music/download', { url }),

  suggestMusic: (id: string) =>
    post<{ scenes: { scene_index: number; query: string; reasoning: string; tracks: MusicTrack[]; assigned_track?: string | null }[] }>(`/api/projects/${id}/suggest-music`, {}),
  updateSceneMusic: (id: string, sceneIndex: number, body: { music_track?: string | null; music_volume?: number | null }) =>
    put<Scene>(`/api/projects/${id}/scenes/${sceneIndex}/music`, body),

  listProjects: () => request<ProjectSummary[]>('/api/projects'),
  searchStories: (query: string, count?: number) =>
    post<{ results: StorySearchResult[] }>('/api/search-stories', { query, count: count || 6 }),

  gutenbergSearch: (query: string, page?: number, topic?: string, languages?: string) =>
    post<GutenbergSearchResponse>('/api/gutenberg/search', { query, page: page || 1, topic: topic || '', languages: languages || '' }),

  gutenbergText: (text_url: string, max_chars?: number) =>
    post<GutenbergTextResponse>('/api/gutenberg/text', { text_url, max_chars: max_chars ?? 2000 }),

  analyzeChapters: (text: string, book_title?: string) =>
    post<AnalyzeChaptersResponse>('/api/analyze-chapters', { text, book_title: book_title || '' }),

  batchCreate: (body: {
    book_title: string
    chapters: AnalyzedChapter[]
    voice_profile_id?: string
    voice_language?: string
    image_backend?: string
  }) => post<BatchCreateResponse>('/api/batch/create', body),

  batchRun: (groupId: string, body: {
    steps: string[]
    project_ids?: string[]
    voice_profile_id: string
    voice_language?: string
    image_backend?: string
    style_id?: string
    custom_style_prompt?: string
    style_prompt?: string
    lora_keys?: string[]
    character_consistency?: boolean
  }) => post<{ status: string }>(`/api/batch/${groupId}/run`, body),

  batchProgress: (groupId: string) =>
    request<BatchProgress>(`/api/batch/${groupId}/progress`),

  batchPause: (groupId: string) =>
    post<{ status: string }>(`/api/batch/${groupId}/pause`, {}),

  batchResume: (groupId: string) =>
    post<{ status: string }>(`/api/batch/${groupId}/resume`, {}),

  createProject: (body: { source_tale: string; custom_prompt?: string; target_minutes: number; claude_model?: string; pipeline_writer_model?: string; pipeline_critic_model?: string; pipeline_reviser_model?: string; tone?: string }) =>
    post<ProjectState>('/api/projects', body),
  getProject: (id: string) => request<ProjectState>(`/api/projects/${id}`),
  duplicateProject: (id: string) => post<ProjectState>(`/api/projects/${id}/duplicate`),
  deleteProject: (id: string) => del<{ deleted: string }>(`/api/projects/${id}`),
  bulkDeleteProjects: (projectIds: string[]) =>
    post<{ deleted: string[]; not_found: string[] }>('/api/projects/bulk-delete', { project_ids: projectIds }),
  deleteBookGroup: (groupId: string) =>
    del<{ deleted: string[]; group_id: string }>(`/api/book-group/${groupId}`),
  updateSettings: (id: string, body: { tone?: string; target_minutes?: number; suggested_length?: string; music_track?: string | null; music_volume?: number | null }) =>
    put<Record<string, unknown>>(`/api/projects/${id}/settings`, body),

  runScript: (id: string, body: { claude_model?: string; pipeline_writer_model?: string; pipeline_critic_model?: string; pipeline_reviser_model?: string; target_minutes?: number; custom_prompt?: string }) =>
    post<Script>(`/api/projects/${id}/script`, body),
  updateScript: (id: string, body: { title: string; synopsis: string; scenes: Scene[] }) =>
    put<Script>(`/api/projects/${id}/script`, body),

  runVoice: (id: string, body: { profile_id: string; language: string }) =>
    post<{ scenes: Scene[] }>(`/api/projects/${id}/voice`, body),

  runImages: (id: string, body: { backend: string; style_id?: string; custom_style_prompt?: string; style_prompt?: string; lora_keys?: string[]; character_consistency?: boolean }) =>
    post<{ status: string }>(`/api/projects/${id}/images`, body),

  imagesProgress: (id: string) =>
    request<ImagesProgress>(`/api/projects/${id}/images-progress`),

  repairImages: (id: string, body: { backend: string; style_id?: string; custom_style_prompt?: string; style_prompt?: string; lora_keys?: string[]; character_consistency?: boolean; check_duplicates?: boolean }) =>
    post<{ status: string }>(`/api/projects/${id}/images/repair`, body),

  regenerateSceneImages: (id: string, sceneIndex: number, body: { backend: string; style_id?: string; custom_style_prompt?: string; style_prompt?: string; lora_keys?: string[]; character_consistency?: boolean }) =>
    post<{ scene: Scene }>(`/api/projects/${id}/images/${sceneIndex}`, body),

  regenerateSingleImage: (id: string, sceneIndex: number, imageIndex: number, body: { backend: string; style_id?: string; custom_style_prompt?: string; style_prompt?: string; lora_keys?: string[]; character_consistency?: boolean }) =>
    post<{ scene: Scene }>(`/api/projects/${id}/images/${sceneIndex}/${imageIndex}`, body),

  runAnimate: (id: string) =>
    post<{ status: string }>(`/api/projects/${id}/animate`, {}),

  animationProgress: (id: string) =>
    request<{ active: boolean; progress: number; phase: string; error: string | null }>(
      `/api/projects/${id}/animation-progress`
    ),

  runAssemble: (id: string, body?: { music_track?: string; music_volume?: number }) =>
    post<{ status: string }>(`/api/projects/${id}/assemble`, body || {}),

  assemblyProgress: (id: string) =>
    request<{ active: boolean; progress: number; phase: string; error: string | null }>(
      `/api/projects/${id}/assembly-progress`
    ),

  cancelAssembly: (id: string) =>
    post<{ cancelled: boolean }>(`/api/projects/${id}/assembly-cancel`, {}),

  artifactUrl: (projectId: string, filepath: string) =>
    `/api/projects/${projectId}/artifacts/${filepath}`,

  downloadUrl: (projectId: string) =>
    `/api/projects/${projectId}/download`,

  // Source text and splitting
  getSourceText: (projectId: string) =>
    request<{ text: string; char_count: number; project_id: string; title: string; book_group_id?: string; chapter_index?: number }>(`/api/projects/${projectId}/source-text`),
  splitProject: (projectId: string, parts: number) =>
    post<{ original_project_id: string; new_project_ids: string[]; parts: number }>(`/api/projects/${projectId}/split`, { parts }),
  splitProjectIntelligent: (projectId: string, parts: number) =>
    post<{ original_project_id: string; new_project_ids: string[]; parts: number; split_details: { title: string; summary: string; char_count: number }[] }>(`/api/projects/${projectId}/split-intelligent`, { parts }),

  // ── Cast bible & character references ───────────────────────
  generateCast: (id: string, body?: { overwrite?: boolean }) =>
    post<{ cast: CastMember[]; scenes: Scene[] }>(`/api/projects/${id}/cast`, body || {}),
  updateCastMember: (id: string, castId: string, body: { name?: string; role?: string; description?: string; reference_prompt?: string }) =>
    put<{ member: CastMember }>(`/api/projects/${id}/cast/${castId}`, body),
  generateCharacterRefs: (id: string, body: { backend?: string; style_id?: string; custom_style_prompt?: string; style_prompt?: string; cast_ids?: string[] }) =>
    post<{ cast: CastMember[] }>(`/api/projects/${id}/characters`, body),

  // ── Shorts ──────────────────────────────────────────────────
  suggestShorts: (id: string, body?: { count?: number }) =>
    post<{ suggestions: ShortSuggestion[] }>(`/api/projects/${id}/shorts/suggest`, body || {}),
  renderShorts: (id: string, body?: { scene_indices?: number[]; count?: number; hooks?: Record<number, string>; source?: string }) =>
    post<{ status: string; count?: number; scene_indices?: number[] }>(`/api/projects/${id}/shorts`, body || {}),
  listShorts: (id: string) =>
    request<{ shorts: ShortItem[]; progress: ShortsProgress }>(`/api/projects/${id}/shorts`),
  shortsProgress: (id: string) =>
    request<ShortsProgress>(`/api/projects/${id}/shorts/progress`),
}
