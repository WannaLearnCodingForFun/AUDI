import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { endSession, errorText, getInterrupted, getSites, resumeSession } from '../api'
import { Card, Empty, ErrorBox, Loading, StatusPill, when } from '../components/ui'
import { useAsync } from '../hooks'

export default function Sites() {
  const sites = useAsync(getSites, [])
  const interrupted = useAsync(getInterrupted, [])
  const [error, setError] = useState(null)
  const nav = useNavigate()

  const resume = async (id) => {
    try {
      await resumeSession(id)
      nav(`/flight/${id}`)
    } catch (e) { setError(errorText(e)) }
  }
  const close = async (id) => {
    try {
      await endSession(id)
      nav(`/report/${id}`)
    } catch (e) { setError(errorText(e)) }
  }

  return (
    <div className="page">
      <header className="page-head"><h2>Select a site</h2></header>
      <ErrorBox error={error || sites.error || interrupted.error} />
      {(interrupted.data || []).length > 0 && (
        <Card title="Unfinished sessions (crash recovery)" className="warn-card">
          <table className="table">
            <thead><tr><th>Session</th><th>Site</th><th>Started</th><th>Status</th><th>Assets</th><th /></tr></thead>
            <tbody>
              {interrupted.data.map((s) => (
                <tr key={s.session_id}>
                  <td className="mono">{s.session_id}</td><td>{s.site_id}</td><td>{when(s.started_ts)}</td>
                  <td><StatusPill status={s.status} /></td><td>{s.assets_processed}</td>
                  <td className="row gap">
                    <button className="btn small" onClick={() => resume(s.session_id)}>Resume</button>
                    <button className="btn small ghost" onClick={() => close(s.session_id)}>Close &amp; report</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}
      <Card>
        {sites.loading ? <Loading /> : (sites.data || []).length === 0 ? <Empty>No sites. An admin can create one.</Empty> : (
          <div className="site-grid">
            {sites.data.map((s) => (
              <Link key={s.site_id} to={`/briefing/${s.site_id}`} className="site-tile">
                <strong>{s.name}</strong>
                <span className="mono muted small">{s.site_id}</span>
              </Link>
            ))}
          </div>
        )}
      </Card>
    </div>
  )
}
