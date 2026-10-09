import { useState } from 'react'
import { Link } from 'react-router-dom'
import { getSessions, getSites } from '../api'
import { Card, Empty, ErrorBox, Loading, Simulated, StatusPill, when } from '../components/ui'
import { useAsync } from '../hooks'

export default function MissionHistory() {
  const sites = useAsync(getSites, [])
  const [siteId, setSiteId] = useState('')
  const sessions = useAsync(() => getSessions(siteId), [siteId])
  return (
    <div className="page">
      <header className="page-head">
        <h2>Missions &amp; reports</h2>
        <select value={siteId} onChange={(e) => setSiteId(e.target.value)} aria-label="Site">
          <option value="">All sites</option>
          {(sites.data || []).map((s) => <option key={s.site_id} value={s.site_id}>{s.name}</option>)}
        </select>
      </header>
      <ErrorBox error={sessions.error} />
      <Card>
        {sessions.loading ? <Loading /> : (sessions.data || []).length === 0 ? <Empty>No missions yet.</Empty> : (
          <table className="table">
            <thead><tr><th>Session</th><th>Site</th><th>Operator</th><th>Date</th><th>Mode</th><th>Status</th><th>Assets</th><th /></tr></thead>
            <tbody>
              {sessions.data.map((s) => (
                <tr key={s.session_id}>
                  <td className="mono">{s.session_id}</td><td>{s.site_id}</td><td>{s.operator_id}</td><td>{when(s.started_ts)}</td>
                  <td><StatusPill status={s.mode} /></td><td><StatusPill status={s.status} /></td><td>{s.assets_processed}</td>
                  <td className="row gap">
                    <Simulated on={s.simulated} />
                    {s.status === 'ended' ? <Link to={`/report/${s.session_id}`}>Report</Link> : <Link to={`/flight/${s.session_id}`}>Open</Link>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </div>
  )
}
