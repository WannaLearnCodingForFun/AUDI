import { useState } from 'react'
import { Link } from 'react-router-dom'
import { errorText, getDefects, getSites, updateDefect } from '../api'
import { hasRole, useAuth } from '../auth'
import { Card, Empty, ErrorBox, fmt, Loading, StatusPill, when } from '../components/ui'
import { useAsync } from '../hooks'

export const STATUSES = ['Open', 'In Progress', 'Resolved']

export function StatusChanger({ defect, onChanged }) {
  const { user } = useAuth()
  const [notes, setNotes] = useState(defect.notes || '')
  const [error, setError] = useState(null)
  if (!hasRole(user, 'maintenance')) return <StatusPill status={defect.status} />
  const change = async (status) => {
    try {
      await updateDefect(defect.defect_id, status, notes)
      onChanged()
    } catch (e) { setError(errorText(e)) }
  }
  return (
    <div className="stack tight">
      <div className="row gap">
        {STATUSES.map((s) => (
          <button key={s} className={`btn small ${defect.status === s ? 'primary' : 'ghost'}`} disabled={defect.status === s} onClick={() => change(s)}>{s}</button>
        ))}
      </div>
      <input value={notes} onChange={(e) => setNotes(e.target.value)} aria-label="Notes" title="Notes saved with the next status change" />
      <ErrorBox error={error} />
    </div>
  )
}

export default function Defects() {
  const sites = useAsync(getSites, [])
  const [siteId, setSiteId] = useState('')
  const [status, setStatus] = useState('Open')
  const list = useAsync(() => getDefects({ site_id: siteId || undefined, status: status || undefined }), [siteId, status])
  return (
    <div className="page">
      <header className="page-head">
        <h2>Defects</h2>
        <div className="row gap">
          <select value={siteId} onChange={(e) => setSiteId(e.target.value)} aria-label="Site">
            <option value="">All sites</option>
            {(sites.data || []).map((s) => <option key={s.site_id} value={s.site_id}>{s.name}</option>)}
          </select>
          <select value={status} onChange={(e) => setStatus(e.target.value)} aria-label="Status">
            <option value="">Any status</option>
            {STATUSES.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>
      </header>
      <ErrorBox error={list.error} />
      <Card>
        {list.loading ? <Loading /> : (list.data || []).length === 0 ? <Empty>No defects match.</Empty> : (
          <table className="table">
            <thead><tr><th>Defect</th><th>Asset</th><th>Site</th><th>Severity (max)</th><th>First / last seen</th><th>Resolution</th><th>Status</th></tr></thead>
            <tbody>
              {list.data.map((d) => (
                <tr key={d.defect_id}>
                  <td><Link to={`/defects/${d.defect_id}`}>{d.defect_type}</Link></td>
                  <td><Link to={`/assets/${d.asset_id}`} className="mono">{d.asset_id}</Link> <span className="muted small">{d.asset_type}</span></td>
                  <td>{d.site_id}</td>
                  <td>{fmt(d.severity_score)} ({fmt(d.max_severity)})</td>
                  <td className="small">{when(d.first_detected_ts)}<br />{when(d.last_seen_ts)}</td>
                  <td className="small">{d.resolution_suggested ? <span className="pill ok">suggested — normal for {d.normal_streak} missions</span> : '—'}</td>
                  <td><StatusChanger defect={d} onChanged={list.reload} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </div>
  )
}
