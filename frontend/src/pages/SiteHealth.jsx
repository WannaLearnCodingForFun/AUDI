import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { errorText, getOwnerReportPdf, getRoutePlan, getRoutePlanFile, getSiteHealth, getUnseen, saveBlob } from '../api'
import { Card, Empty, ErrorBox, fmt, Loading, Stat } from '../components/ui'
import { useAsync } from '../hooks'

const BAND_TONE = { good: 'ok', fair: 'warn', poor: 'high', critical: 'crit', unknown: '' }

/** 5.5 health index, 2.5 unseen heatmap, 4.5 advisory revisit route, 6.4 owner report. */
export default function SiteHealth() {
  const { siteId } = useParams()
  const health = useAsync(() => getSiteHealth(siteId), [siteId])
  const unseen = useAsync(() => getUnseen(siteId), [siteId])
  const [route, setRoute] = useState(null)
  const [minP, setMinP] = useState(0.3)
  const [month, setMonth] = useState(new Date().toISOString().slice(0, 7))
  const [err, setErr] = useState(null)
  const plan = async () => { setErr(null); try { setRoute(await getRoutePlan(siteId, { min_priority: Number(minP) })) } catch (e) { setErr(errorText(e)) } }
  const planFile = async () => { try { saveBlob(await getRoutePlanFile(siteId, { min_priority: Number(minP) }), `audi_revisit_${siteId}.plan`) } catch (e) { setErr(errorText(e)) } }
  const report = async () => { try { saveBlob(await getOwnerReportPdf(siteId, month), `audi_owner_${siteId}_${month}.pdf`) } catch (e) { setErr(errorText(e)) } }
  if (health.loading) return <Loading />
  const h = health.data
  const maxDays = Math.max(1, ...((unseen.data?.cells || []).map((c) => c.max_days_unseen)))
  return (
    <div className="page">
      <header className="page-head"><h2>Site health · {siteId}</h2><Link to={`/twin/${siteId}`} className="btn ghost">3D twin</Link></header>
      <ErrorBox error={err || health.error} />
      <div className="grid four">
        <Stat label="Site health index" value={fmt(h?.health_index, 0)} tone={BAND_TONE[h?.band]} />
        <Stat label="Assets" value={h?.assets.length ?? 0} />
        <Stat label="Poor or critical" value={(h?.assets || []).filter((a) => ['poor', 'critical'].includes(a.band)).length} tone="high" />
        <Stat label="Max days unseen" value={fmt(Math.max(0, ...(unseen.data?.assets || []).map((a) => a.days_unseen)), 0)} />
      </div>
      <div className="grid two">
        <Card title="Assets by health (worst first)">
          {!h?.assets.length ? <Empty>No assets.</Empty> : (
            <table className="table"><thead><tr><th>Asset</th><th>Type</th><th>Health</th></tr></thead>
              <tbody>{h.assets.slice(0, 30).map((a) => (
                <tr key={a.asset_id}><td><Link className="mono" to={`/assets/${a.asset_id}`}>{a.asset_id}</Link></td><td>{a.asset_type}</td>
                  <td><span className={`pill ${BAND_TONE[a.band]}`}>{fmt(a.health_index, 0)} {a.band}</span></td></tr>
              ))}</tbody></table>
          )}
        </Card>
        <Card title="Unseen for N days (heatmap)">
          {!unseen.data?.cells?.length ? <Empty>No located assets.</Empty> : (
            <div className="heatgrid">
              {unseen.data.cells.map((c) => (
                <div key={`${c.row},${c.col}`} className="heatcell" title={`${c.assets} assets · up to ${fmt(c.max_days_unseen, 0)} days`}
                  style={{ gridRow: 1 + (Math.max(...unseen.data.cells.map((x) => x.row)) - c.row), gridColumn: 1 + c.col,
                    background: `rgba(248, 81, 73, ${0.15 + 0.85 * c.max_days_unseen / maxDays})` }}>{fmt(c.max_days_unseen, 0)}</div>
              ))}
            </div>
          )}
        </Card>
      </div>
      <div className="grid two">
        <Card title="Advisory revisit route" actions={<>
          <label className="row gap small">min R <input type="number" step="0.05" min="0" max="1" value={minP} onChange={(e) => setMinP(e.target.value)} style={{ width: '5rem' }} /></label>
          <button className="btn small" onClick={plan}>Plan</button><button className="btn small ghost" onClick={planFile}>QGroundControl .plan</button></>}>
          <p className="small muted">Suggested waypoints only — the operator flies manually; AUDI never controls the drone.</p>
          {route && (
            <>
              <div className="small">{route.waypoints.length} waypoints · {fmt(route.route_length_m, 0)} m · coverage {fmt(100 * route.coverage, 0)}%{route.skipped_no_location.length ? ` · ${route.skipped_no_location.length} without location skipped` : ''}</div>
              <ol className="small">{route.waypoints.map((w) => <li key={w.asset_id} className="mono">{w.asset_id} · {fmt(w.altitude_m, 1)} m AGL · standoff {fmt(w.standoff_m, 0)} m</li>)}</ol>
            </>
          )}
        </Card>
        <Card title="Monthly owner report" actions={<><input type="month" value={month} onChange={(e) => setMonth(e.target.value)} aria-label="Month" /><button className="btn small" onClick={report}>Download PDF</button></>}>
          <p className="small muted">Health index and its trend, inspection coverage, defects opened / resolved, the cost of deferring repairs and the estimated solar energy loss.</p>
        </Card>
      </div>
    </div>
  )
}
