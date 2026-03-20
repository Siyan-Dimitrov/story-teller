// ── Types ────────────────────────────────────────────────────

export interface HealthStatus {
  ollama: boolean
  voicebox: boolean
  comfyui: boolean
  ffmpeg: boolean
}

export interface Scene {
  index: number
  narration: string
  image_prompt: string
  mood: string
  duration_hint: number
  audio_path?: string | null
  audio_duration?: number | null
  image_path?: string | null
  kb_effect: string
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
  createProject: (body: { source_tale: string; custom_prompt?: string; target_minutes: number; ollama_model: string }) =>
    post<ProjectState>('/api/projects', body),
  getProject: (id: string) => request<ProjectState>(`/api/projects/${id}`),
  deleteProject: (id: string) => del<{ deleted: string }>(`/api/projects/${id}`),

  runScript: (id: string, body: { ollama_model?: string; target_minutes?: number; custom_prompt?: string }) =>
    post<Script>(`/api/projects/${id}/script`, body),
  updateScript: (id: string, body: { title: string; synopsis: string; scenes: Scene[] }) =>
    put<Script>(`/api/projects/${id}/script`, body),

  runVoice: (id: string, body: { profile_id: string; language: string }) =>
    post<{ scenes: Scene[] }>(`/api/projects/${id}/voice`, body),

  runImages: (id: string, body: { backend: string; style_prompt: string; lora_keys?: string[] }) =>
    post<{ scenes: Scene[] }>(`/api/projects/${id}/images`, body),

  runAssemble: (id: string) =>
    post<{ video: string; duration: number | null }>(`/api/projects/${id}/assemble`, {}),

  artifactUrl: (projectId: string, filepath: string) =>
    `/api/projects/${projectId}/artifacts/${filepath}`,

  downloadUrl: (projectId: string) =>
    `/api/projects/${projectId}/download`,
}
