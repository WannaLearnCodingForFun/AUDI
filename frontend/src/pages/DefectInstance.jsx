import { Link, useParams } from 'react-router-dom'
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { getInstance } from '../api'
import { Card, ErrorBox, fmt, Loading, when } from '../components/ui'
import { useAsync } from '../hooks'

/** 2.2 / 2.3: one physical defect, its size history and its deterioration forecast. */
export default function DefectInstance() {
  const { instanceId } = useParams()
  const inst = useAsync(() => getInstance(instanceId), [instanceId])
  if (inst.loading) return <Loading />
  const d = inst.data
  if (!d) return <ErrorBox error={inst.error} />
  const f = d.forecast || {}
  const metric = f.metric || 'area_norm'
  const pts = d.measurements.map((m) => ({ ts: new Date(m.ts).toLocaleDateString(), value: m[metric] }))
  return (
    <div className="page">
      <header className="page-head"><h2>{d.defect_type} on <Link className="mono" to={`/assets/${d.asset_id}`}>{d.asset_id}</Link></h2><span className="pill">{d.status}</span></header>
      <div className="grid two">
        <Card title="Forecast">
          <dl className="kv">
            <dt>Model</dt><dd>{f.model || '—'} {f.params && <span className="small muted mono">{Object.entries(f.params).map(([k, v]) => `${k}=${fmt(v, 4)}`).join(' ')}</span>}</dd>
            <dt>Current / critical</dt><dd>{fmt(f.current, 3)} / {fmt(f.critical, 3)} {metric}</dd>
            <dt>Growth since last visit</dt><dd>{f.growth_since_last_pct == null ? '—' : `${fmt(f.growth_since_last_pct, 0)}%`}</dd>
            <dt>Predicted critical</dt><dd>{d.predicted_critical_ts ? `${when(d.predicted_critical_ts)} (in ${fmt(f.days_to_critical, 0)} days)` : 'not within the horizon'}</dd>
            <dt>Fit RMSE</dt><dd>{fmt(f.rmse, 4)}</dd>
            <dt>First / last seen</dt><dd>{when(d.first_seen_ts)} / {when(d.last_seen_ts)}</dd>
          </dl>
        </Card>
        <Card title={`Size history (${metric})`}>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={pts}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="ts" /><YAxis /><Tooltip /><Line dataKey="value" strokeWidth={2} /></LineChart>
          </ResponsiveContainer>
        </Card>
      </div>
    </div>
  )
}
