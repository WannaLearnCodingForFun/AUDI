import { Link, useParams } from 'react-router-dom'
import { getDefect, getSummary } from '../api'
import SeverityChart from '../components/SeverityChart'
import { Card, ErrorBox, fmt, Loading, when } from '../components/ui'
import { useAsync } from '../hooks'
import { StatusChanger } from './Defects'

export default function DefectDetail() {
  const { defectId } = useParams()
  const d = useAsync(() => getDefect(defectId), [defectId])
  const levels = useAsync(() => getSummary(), [])
  if (d.loading) return <Loading />
  const x = d.data
  return (
    <div className="page">
      <ErrorBox error={d.error} />
      {x && (
        <>
          <header className="page-head">
            <h2>{x.defect_type} on <Link to={`/assets/${x.asset_id}`} className="mono">{x.asset_id}</Link></h2>
          </header>
          <div className="grid two">
            <Card title="Lifecycle (FR-17)">
              <dl className="kv">
                <dt>Status</dt><dd><StatusChanger defect={x} onChanged={d.reload} /></dd>
                <dt>Severity (latest / max)</dt><dd>{fmt(x.severity_score)} / {fmt(x.max_severity)}</dd>
                <dt>First detected</dt><dd>{when(x.first_detected_ts)}</dd>
                <dt>Last seen</dt><dd>{when(x.last_seen_ts)}</dd>
                <dt>Normal streak</dt><dd>{x.normal_streak} mission(s){x.resolution_suggested ? ' — resolution suggested' : ''}</dd>
                <dt>Resolved</dt><dd>{when(x.resolved_ts)}</dd>
                <dt>Notes</dt><dd>{x.notes || '—'}</dd>
              </dl>
            </Card>
            <Card title={`Severity trend (${x.trend.length} observations)`}>
              <SeverityChart points={x.trend.map((t) => ({ ts: t.ts, severity: t.severity_score }))} thresholds={levels.data?.severity_levels} />
            </Card>
          </div>
        </>
      )}
    </div>
  )
}
