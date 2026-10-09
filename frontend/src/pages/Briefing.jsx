import { useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { errorText, getBriefing, startSession } from '../api'
import { Card, Empty, ErrorBox, fmt, Loading, Simulated, StatusPill, when } from '../components/ui'
import { useAsync } from '../hooks'

export default function Briefing() {
  const { siteId } = useParams()
  const brief = useAsync(() => getBriefing(siteId), [siteId])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [sync, setSync] = useState(null)
  const nav = useNavigate()

  const start = async () => {
    setBusy(true)
    setError(null)
    try {
      const r = await startSession(siteId)
      setSync(r.sync)
      nav(`/flight/${r.session.session_id}`, { state: { sync: r.sync } })
    } catch (e) {
      setError(errorText(e))
    } finally {
      setBusy(false)
    }
  }

  const b = brief.data
  return (
    <div className="page">
      <header className="page-head">
        <h2>Pre-flight briefing · <span className="mono">{siteId}</span></h2>
        <button className="btn primary" onClick={start} disabled={busy}>{busy ? 'Syncing with cloud…' : 'Start session'}</button>
      </header>
      <p className="muted">Starting a session checks the cloud (FR-1), pulls the latest site memory when online (FR-2) or runs from the local database when offline (FR-3).</p>
      <ErrorBox error={error || brief.error} />
      {sync && <div className="muted">Sync: {sync.mode}</div>}
      {brief.loading ? <Loading /> : b && (
        <div className="grid two">
          <Card title={`Assets due for recheck (${b.assets_due.length}), ranked by R`}>
            {b.assets_due.length === 0 ? <Empty>No assets known at this site yet — the first flight registers them.</Empty> : (
              <table className="table">
                <thead><tr><th>#</th><th>Asset</th><th>Type</th><th>Condition</th><th title="R = 0.5·Sr + 0.3·T + 0.2·D">R</th><th>Sr / T / D</th><th>Last checked</th><th /></tr></thead>
                <tbody>
                  {b.assets_due.map((a, i) => (
                    <tr key={a.asset_id}>
                      <td>{i + 1}</td>
                      <td><Link to={`/assets/${a.asset_id}`} className="mono">{a.asset_id}</Link>{a.marker_id !== null && a.marker_id !== undefined && <span className="muted small"> · ArUco {a.marker_id}</span>}</td>
                      <td>{a.asset_type}</td>
                      <td>{a.current_condition}</td>
                      <td><strong>{fmt(a.revisit_priority)}</strong></td>
                      <td className="small muted">{a.components ? `${fmt(a.components.recent_severity)} / ${fmt(a.components.trend)} / ${fmt(a.components.days_since_inspection)}` : '—'}</td>
                      <td>{when(a.last_checked_ts)}</td>
                      <td><Simulated on={a.simulated} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </Card>
          <Card title={`Unresolved defects (${b.unresolved_defects.length})`}>
            {b.unresolved_defects.length === 0 ? <Empty>None.</Empty> : (
              <table className="table">
                <thead><tr><th>Asset</th><th>Defect</th><th>Severity</th><th>Status</th><th>Last seen</th></tr></thead>
                <tbody>
                  {b.unresolved_defects.map((d) => (
                    <tr key={d.defect_id}>
                      <td className="mono">{d.asset_id}</td><td>{d.defect_type}</td><td>{fmt(d.severity_score)}</td>
                      <td><StatusPill status={d.status} /></td><td>{when(d.last_seen_ts)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </Card>
        </div>
      )}
    </div>
  )
}
