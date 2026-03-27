// ── Types ────────────────────────────────────────────────────

export interface HealthStatus {
  ollama: boolean
  voicebox: boolean
  comfyui: boolean
  replicate: boolean
  ffmpeg: boolean
}

export interface Scene {
  index: number
  narration: string
  image_prompt: string
  image_prompts?: string[]
  mood: string
  duration_hint: number
  audio_path?: string | null
  audio_duration?: number | null
  image_path?: string | null
  image_paths?: string[]
  kb_effect: string
  qc_results?: { image_index: number; passed: boolean; scores: Record<string, number>; average_score: number; reasoning: string; attempts: number }[]
  qc_passed?: boolean
  animation_types?: string[]
  motion_presets?: string[]
  depth_map_paths?: (string | null)[]
  animatediff_clip_paths?: (string | null)[]
  voice_error?: string
  image_error?: string
}

export interface Script {
  title: string
  synopsis: string
  scenes: Scene[]
  target_minutes?: number
  source_tale?: string
  tone?: string
}

export interface ProjectState {
  project_id: string
  step: string
  error?: string | null
  title: string
  source_tale: string
  voice_profile_id?: string | null
  voice_language: string
  ollama_model: string
  image_backend: string
  target_minutes: number
  created_at: string
  script?: Script
  output_dir?: string | null
}

export interface ProjectSummary {
  project_id: string
  title: string
  step: string
  source_tale: string
  created_at: string
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
}

export interface LorasResponse {
  available: Record<string, LoraInfo>
  defaults: string[]
}

export const api = {
  health: () => request<HealthStatus>('/api/health'),
  tales: () => request<Tale[]>('/api/tales'),
  tale: (id: string) => request<Tale>(`/api/tales/${id}`),
  profiles: () => request<VoiceProfile[]>('/api/profiles'),
  loras: () => request<LorasResponse>('/api/loras'),

  listProjects: () => request<ProjectSummary[]>('/api/projects'),
  searchStories: (query: string, count?: number, ollama_model?: string) =>
    post<{ results: StorySearchResult[] }>('/api/search-stories', { query, count: count || 6, ...(ollama_model && { ollama_model }) }),

  gutenbergSearch: (query: string, page?: number, topic?: string, languages?: string) =>
    post<GutenbergSearchResponse>('/api/gutenberg/search', { query, page: page || 1, topic: topic || '', languages: languages || '' }),

  gutenbergText: (text_url: string, max_chars?: number) =>
    post<GutenbergTextResponse>('/api/gutenberg/text', { text_url, max_chars: max_chars ?? 2000 }),

  createProject: (body: { source_tale: string; custom_prompt?: string; target_minutes: number; ollama_model: string; tone?: string }) =>
    post<ProjectState>('/api/projects', body),
  getProject: (id: string) => request<ProjectState>(`/api/projects/${id}`),
  deleteProject: (id: string) => del<{ deleted: string }>(`/api/projects/${id}`),

  runScript: (id: string, body: { ollama_model?: string; target_minutes?: number; custom_prompt?: string }) =>
    post<Script>(`/api/projects/${id}/script`, body),
  updateScript: (id: string, body: { title: string; synopsis: string; scenes: Scene[] }) =>
    put<Script>(`/api/projects/${id}/script`, body),

  runVoice: (id: string, body: { profile_id: string; language: string; instruct?: string }) =>
    post<{ scenes: Scene[] }>(`/api/projects/${id}/voice`, body),

  runImages: (id: string, body: { backend: string; style_prompt: string; lora_keys?: string[] }) =>
    post<{ scenes: Scene[] }>(`/api/projects/${id}/images`, body),

  runQC: (id: string, body: { vision_model?: string; pass_threshold?: number; style_prompt?: string; targets?: { scene_index: number; image_index: number }[] }) =>
    post<{ status: string }>(`/api/projects/${id}/qc`, body),

  regenerateQC: (id: string, body: { targets: { scene_index: number; image_index: number }[]; style_prompt?: string; lora_keys?: string[] }) =>
    post<{ status: string }>(`/api/projects/${id}/qc-regenerate`, body),

  qcProgress: (id: string) =>
    request<{ active: boolean; progress: number; phase: string; error: string | null }>(
      `/api/projects/${id}/qc-progress`
    ),

  retryQCImage: (id: string, sceneIndex: number, imageIndex: number) =>
    post<{ scores: Record<string, number>; average_score: number; reasoning: string }>(
      `/api/projects/${id}/qc-retry/${sceneIndex}/${imageIndex}`, {}
    ),

  runAnimate: (id: string) =>
    post<{ status: string }>(`/api/projects/${id}/animate`, {}),

  animationProgress: (id: string) =>
    request<{ active: boolean; progress: number; phase: string; error: string | null }>(
      `/api/projects/${id}/animation-progress`
    ),

  runAssemble: (id: string) =>
    post<{ status: string }>(`/api/projects/${id}/assemble`, {}),

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
}
