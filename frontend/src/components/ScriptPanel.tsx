import { useState } from 'react'
import { Wand2, Save, Plus, Trash2, GripVertical } from 'lucide-react'
import type { ProjectState, Scene } from '../api'
import { api } from '../api'

interface Props {
  project: ProjectState
  onRefresh: () => void
  onNext: () => void
}

export default function ScriptPanel({ project, onRefresh, onNext }: Props) {
  const script = project.script
  const [generating, setGenerating] = useState(false)
  const [saving, setSaving] = useState(false)
  const [customPrompt, setCustomPrompt] = useState('')

  // Editable local state
  const [title, setTitle] = useState(script?.title || project.title || '')
  const [synopsis, setSynopsis] = useState(script?.synopsis || '')
  const [scenes, setScenes] = useState<Scene[]>(script?.scenes || [])
  const [dirty, setDirty] = useState(false)

  const handleGenerate = async () => {
    setGenerating(true)
    try {
      const result = await api.runScript(project.project_id, {
        ollama_model: project.ollama_model,
        target_minutes: project.target_minutes,
        custom_prompt: customPrompt,
      })
      setTitle(result.title)
      setSynopsis(result.synopsis)
      setScenes(result.scenes)
      setDirty(false)
      onRefresh()
    } catch (e) {
      alert('Script generation failed: ' + (e as Error).message)
      onRefresh()
    } finally {
      setGenerating(false)
    }
  }

  const handleSave = async () => {
    setSaving(true)
    try {
      const result = await api.updateScript(project.project_id, { title, synopsis, scenes })
      setScenes(result.scenes)
      setDirty(false)
      onRefresh()
    } catch (e) {
      alert('Save failed: ' + (e as Error).message)
    } finally {
      setSaving(false)
    }
  }

  const updateScene = (index: number, updates: Partial<Scene>) => {
    setScenes(prev => prev.map((s, i) => i === index ? { ...s, ...updates } : s))
    setDirty(true)
  }

  const removeScene = (index: number) => {
    setScenes(prev => prev.filter((_, i) => i !== index).map((s, i) => ({ ...s, index: i })))
    setDirty(true)
  }

  const addScene = () => {
    setScenes(prev => [...prev, {
      index: prev.length,
      narration: '',
      image_prompt: '',
      mood: 'dark',
      duration_hint: 15,
      kb_effect: 'zoom_in',
    }])
    setDirty(true)
  }

  return (
    <div className="space-y-4">
      {/* Generate controls */}
      <div className="p-4 rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)]">
        <div className="flex items-end gap-3">
          <div className="flex-1">
            <label className="block text-xs text-[var(--text-secondary)] mb-1.5">Additional Direction (optional)</label>
            <input
              type="text"
              value={customPrompt}
              onChange={e => setCustomPrompt(e.target.value)}
              placeholder="e.g. 'Make it more psychological horror' or 'Focus on the villain's perspective'"
              className="w-full bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:outline-none focus:border-[var(--border-focus)]"
            />
          </div>
          <button
            onClick={handleGenerate}
            disabled={generating}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white text-sm font-medium transition-colors disabled:opacity-50 shrink-0"
          >
            <Wand2 size={14} className={generating ? 'animate-spin' : ''} />
            {generating ? 'Generating...' : scenes.length ? 'Regenerate' : 'Generate Script'}
          </button>
        </div>
      </div>

      {/* Script editor */}
      {scenes.length > 0 && (
        <>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-[var(--text-secondary)] mb-1.5">Title</label>
              <input
                type="text"
                value={title}
                onChange={e => { setTitle(e.target.value); setDirty(true) }}
                className="w-full bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--border-focus)]"
              />
            </div>
            <div>
              <label className="block text-xs text-[var(--text-secondary)] mb-1.5">Synopsis</label>
              <input
                type="text"
                value={synopsis}
                onChange={e => { setSynopsis(e.target.value); setDirty(true) }}
                className="w-full bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--border-focus)]"
              />
            </div>
          </div>

          {/* Scenes */}
          <div className="space-y-3">
            {scenes.map((scene, i) => (
              <div key={i} className="p-4 rounded-lg border border-[var(--border)] bg-[var(--bg-secondary)]">
                <div className="flex items-center gap-2 mb-3">
                  <GripVertical size={14} className="text-[var(--text-muted)]" />
                  <span className="text-xs font-medium text-[var(--accent)]">Scene {i + 1}</span>
                  <select
                    value={scene.mood}
                    onChange={e => updateScene(i, { mood: e.target.value })}
                    className="ml-auto bg-[var(--bg-tertiary)] border border-[var(--border)] rounded px-2 py-0.5 text-xs text-[var(--text-secondary)]"
                  >
                    {['dark', 'tense', 'whimsical', 'melancholy', 'horrifying', 'peaceful', 'ominous', 'triumphant', 'neutral'].map(m => (
                      <option key={m} value={m}>{m}</option>
                    ))}
                  </select>
                  <select
                    value={scene.kb_effect}
                    onChange={e => updateScene(i, { kb_effect: e.target.value })}
                    className="bg-[var(--bg-tertiary)] border border-[var(--border)] rounded px-2 py-0.5 text-xs text-[var(--text-secondary)]"
                  >
                    {['zoom_in', 'zoom_out', 'pan_left', 'pan_right', 'static'].map(e => (
                      <option key={e} value={e}>{e.replace('_', ' ')}</option>
                    ))}
                  </select>
                  <button
                    onClick={() => removeScene(i)}
                    className="p-1 rounded text-[var(--text-muted)] hover:text-[var(--error)] transition-colors"
                  >
                    <Trash2 size={12} />
                  </button>
                </div>

                <div className="space-y-2">
                  <div>
                    <label className="block text-[10px] uppercase tracking-wider text-[var(--text-muted)] mb-1">Narration</label>
                    <textarea
                      value={scene.narration}
                      onChange={e => updateScene(i, { narration: e.target.value })}
                      rows={4}
                      className="w-full bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--border-focus)] resize-y"
                    />
                  </div>
                  <div>
                    <label className="block text-[10px] uppercase tracking-wider text-[var(--text-muted)] mb-1">Image Prompt</label>
                    <textarea
                      value={scene.image_prompt}
                      onChange={e => updateScene(i, { image_prompt: e.target.value })}
                      rows={2}
                      className="w-full bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--border-focus)] resize-y"
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Actions */}
          <div className="flex items-center gap-3">
            <button
              onClick={addScene}
              className="flex items-center gap-1.5 px-3 py-2 rounded-lg border border-[var(--border)] text-sm text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:border-[var(--border-focus)] transition-colors"
            >
              <Plus size={14} /> Add Scene
            </button>
            <div className="flex-1" />
            {dirty && (
              <button
                onClick={handleSave}
                disabled={saving}
                className="flex items-center gap-1.5 px-4 py-2 rounded-lg border border-[var(--accent)] text-sm text-[var(--accent)] hover:bg-[var(--accent)]/10 transition-colors disabled:opacity-50"
              >
                <Save size={14} />
                {saving ? 'Saving...' : 'Save Changes'}
              </button>
            )}
            <button
              onClick={() => {
                if (dirty) { alert('Save your changes first!'); return }
                onNext()
              }}
              className="px-4 py-2 rounded-lg bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white text-sm font-medium transition-colors"
            >
              Next: Voice
            </button>
          </div>
        </>
      )}
    </div>
  )
}
