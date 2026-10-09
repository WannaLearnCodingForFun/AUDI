import { useEffect, useState } from 'react'
import { errorText, getConfig, getConfigVersions, putConfig } from '../api'
import { Card, ErrorBox, Loading, when } from '../components/ui'
import { useAsync } from '../hooks'

const NUM_GROUPS = [
  { key: 'severity', title: 'Severity S = (w1·M + w2·C)·Cf', fields: ['w1', 'w2', 'priority_threshold'] },
  { key: 'revisit_priority', title: 'Revisit R = w3·Sr + w4·T + w5·D', fields: ['w3', 'w4', 'w5', 'max_days_since_inspection', 'trend_window'] },
  { key: 'change_detection', title: 'Change M = α·a + (1−α)·b, gate threshold', fields: ['threshold', 'alpha'] },
]

function NumInput({ value, onChange, step = 0.01 }) {
  return <input type="number" step={step} value={value} onChange={(e) => onChange(e.target.value === '' ? '' : Number(e.target.value))} />
}

export default function AdminConfig() {
  const cfg = useAsync(getConfig, [])
  const versions = useAsync(getConfigVersions, [])
  const [draft, setDraft] = useState(null)
  const [error, setError] = useState(null)
  const [saved, setSaved] = useState(null)

  useEffect(() => { if (cfg.data) setDraft(structuredClone(cfg.data.config)) }, [cfg.data])
  if (cfg.loading || !draft) return <Loading />

  const set = (path, v) => setDraft((d) => {
    const n = structuredClone(d)
    let o = n
    path.slice(0, -1).forEach((k) => { o = o[k] })
    o[path[path.length - 1]] = v
    return n
  })

  const save = async () => {
    setError(null)
    setSaved(null)
    try {
      const r = await putConfig(draft)
      setSaved(`Saved as version ${r.version}`)
      cfg.reload()
      versions.reload()
    } catch (e) { setError(errorText(e)) }
  }

  const sumW12 = (Number(draft.severity.w1) + Number(draft.severity.w2)).toFixed(3)
  const sumW345 = (Number(draft.revisit_priority.w3) + Number(draft.revisit_priority.w4) + Number(draft.revisit_priority.w5)).toFixed(3)
  return (
    <div className="page">
      <header className="page-head">
        <h2>Configuration {cfg.data.version ? <span className="muted small">version {cfg.data.version}</span> : <span className="muted small">config.yaml defaults</span>}</h2>
        <div className="row gap">
          <button className="btn ghost" onClick={() => setDraft(structuredClone(cfg.data.config))}>Discard changes</button>
          <button className="btn primary" onClick={save}>Validate &amp; save</button>
        </div>
      </header>
      <p className="muted">Changes are validated on the server (weights must sum to 1, levels must increase) and stored as a new version; they apply to the running system immediately.</p>
      <ErrorBox error={error || cfg.error} />
      {saved && <div className="ok-box">{saved}</div>}

      <div className="grid three">
        {NUM_GROUPS.map((g) => (
          <Card key={g.key} title={g.title}>
            <div className="form-grid">
              {g.fields.map((f) => (
                <label key={f} className="field"><span>{f}</span>
                  <NumInput value={draft[g.key][f]} step={f === 'trend_window' || f === 'max_days_since_inspection' ? 1 : 0.01} onChange={(v) => set([g.key, f], v)} />
                </label>
              ))}
            </div>
            {g.key === 'severity' && <small className={sumW12 === '1.000' ? 'muted' : 'text-crit'}>w1 + w2 = {sumW12}</small>}
            {g.key === 'revisit_priority' && <small className={sumW345 === '1.000' ? 'muted' : 'text-crit'}>w3 + w4 + w5 = {sumW345}</small>}
          </Card>
        ))}
      </div>

      <div className="grid three">
        <Card title="Severity levels (lower bounds)">
          <div className="form-grid">
            {Object.keys(draft.severity.levels).map((k) => (
              <label key={k} className="field"><span>{k}</span><NumInput value={draft.severity.levels[k]} onChange={(v) => set(['severity', 'levels', k], v)} /></label>
            ))}
          </div>
        </Card>
        <Card title="Asset criticality C">
          <div className="form-grid">
            {Object.keys(draft.asset_criticality).map((k) => (
              <label key={k} className="field"><span>{k}</span><NumInput value={draft.asset_criticality[k]} onChange={(v) => set(['asset_criticality', k], v)} /></label>
            ))}
          </div>
        </Card>
        <Card title="Defects">
          <label className="field"><span>Consecutive normal missions before suggesting resolution</span>
            <NumInput step={1} value={draft.resolution_suggestion_missions} onChange={(v) => set(['resolution_suggestion_missions'], v)} />
          </label>
        </Card>
      </div>

      <Card title="Detection prompts (YOLO-World, one per line)">
        <div className="grid four">
          {Object.keys(draft.detection_prompts).map((k) => (
            <label key={k} className="field"><span>{k}</span>
              <textarea rows={3} value={draft.detection_prompts[k].join('\n')}
                onChange={(e) => set(['detection_prompts', k], e.target.value.split('\n').map((s) => s.trim()).filter(Boolean))} />
            </label>
          ))}
        </div>
      </Card>

      <Card title="Version history">
        {(versions.data || []).length === 0 ? <div className="muted">No saved versions — config.yaml defaults are active.</div> : (
          <table className="table">
            <thead><tr><th>Version</th><th>By</th><th>When</th><th>Changes</th></tr></thead>
            <tbody>{versions.data.map((v) => (
              <tr key={v.version}><td>{v.version}</td><td>{v.username}</td><td>{when(v.created_ts)}</td><td><code className="small">{JSON.stringify(v.changes)}</code></td></tr>
            ))}</tbody>
          </table>
        )}
      </Card>
    </div>
  )
}
