import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { getAsset, getAssetHistory, getAssetInstances, getSummary, imageUrl, sendFeedback } from '../api'
import SeverityChart from '../components/SeverityChart'
import { Card, Empty, ErrorBox, fmt, Loading, Severity, Simulated, StatusPill, when } from '../components/ui'
import { useAsync } from '../hooks'

export default function AssetDetail() {
  const { assetId } = useParams()
  const asset = useAsync(() => getAsset(assetId), [assetId])
  const hist = useAsync(() => getAssetHistory(assetId), [assetId])
  const levels = useAsync(() => getSummary(), [])
  const instances = useAsync(() => getAssetInstances(assetId), [assetId])
  const [fb, setFb] = useState({})
  const [heat, setHeat] = useState(false)
  const [selected, setSelected] = useState(null)

  if (asset.loading || hist.loading) return <Loading />
  const a = asset.data
  const obs = hist.data?.observations || []
  const sel = selected ? obs.find((o) => o.observation_id === selected) : obs[obs.length - 1]
  return (
    <div className="page">
      <ErrorBox error={asset.error || hist.error} />
      {a && (
        <>
          <header className="page-head">
            <h2 className="mono">{a.asset_id}</h2>
            <div className="row gap"><Simulated on={a.simulated} /><span className="pill">{a.asset_type}</span></div>
          </header>
          <div className="grid two">
            <Card title="Asset">
              <dl className="kv">
                <dt>Site</dt><dd>{a.site_id}</dd>
                <dt>Current condition</dt><dd className={a.current_is_defect ? 'text-crit' : ''}>{a.current_condition}</dd>
                <dt>Reference (last confirmed)</dt><dd>{a.reference_condition?.label}</dd>
                <dt>Criticality C</dt><dd>{fmt(a.criticality)}</dd>
                <dt>Health index</dt><dd>{fmt(a.health_index, 0)} / 100</dd>
                {a.revisit_components?.trend_source === 'forecast' && <><dt>Forecast</dt><dd>critical in about {fmt(a.revisit_components.days_to_critical, 0)} days (drives T)</dd></>}
                {a.revisit_components?.graph_boost ? <><dt>Dependency boost</dt><dd>+{fmt(a.revisit_components.graph_boost)} from <Link className="mono" to={`/assets/${a.revisit_components.graph_source}`}>{a.revisit_components.graph_source}</Link></dd></> : null}
                <dt>Revisit priority R</dt><dd>{fmt(a.revisit_priority_score)}{a.revisit_components && <span className="muted small"> · Sr {fmt(a.revisit_components.recent_severity)} · T {fmt(a.revisit_components.trend)} · D {fmt(a.revisit_components.days_since_inspection)}</span>}</dd>
                <dt>ArUco marker</dt><dd>{a.marker_id ?? '—'}</dd>
                <dt>Location</dt><dd>{a.location ? `${fmt(a.location.latitude, 6)}, ${fmt(a.location.longitude, 6)} (${a.location_samples} samples)` : '—'}</dd>
                <dt>First observed / last checked</dt><dd>{when(a.first_observed_ts)} / {when(a.last_checked_ts)}</dd>
              </dl>
            </Card>
            <Card title="Severity per mission">
              {obs.length === 0 ? <Empty>No observations.</Empty> : (
                <SeverityChart points={obs.map((o) => ({ ts: o.observation_ts, severity: o.severity_score }))} thresholds={levels.data?.severity_levels} />
              )}
            </Card>
          </div>

          {sel && (
            <Card title="Image history" actions={<label className="row gap small"><input type="checkbox" checked={heat} onChange={(e) => setHeat(e.target.checked)} /> change heatmap</label>}>
              <div className="viewer">
                <img src={imageUrl(heat && sel.heatmap_reference ? sel.heatmap_reference : sel.image_reference)} alt={`Observation ${sel.observation_id}`} />
                <div className="stack">
                  <div><strong>{sel.condition_description}</strong>{sel.defect_type ? ` · ${sel.defect_type}` : ''}</div>
                  <div><Severity score={sel.severity_score} level={sel.severity_level} /></div>
                  <div className="small muted">M {fmt(sel.change_magnitude)} (condition {fmt(sel.condition_delta)}, visual {fmt(sel.visual_delta)}) · Cf {fmt(sel.observation_confidence)} · match {sel.match_stage} ({fmt(sel.match_confidence)})</div>
                  <div className="small muted">{when(sel.observation_ts)} · session <Link to={`/report/${sel.session_id}`} className="mono">{sel.session_id}</Link></div>
                  {heat && !sel.heatmap_reference && <div className="small muted">No heatmap for this observation (first view of the asset).</div>}
                  {sel.metadata?.simulation_note && <div className="small muted">Simulation: {sel.metadata.simulation_note}</div>}
                  {sel.metadata?.mask_reference && <a className="small" href={imageUrl(sel.metadata.mask_reference)} target="_blank" rel="noreferrer">change mask</a>}
                  <div className="row gap small">
                    {fb[sel.observation_id] ? <span className="muted">feedback: {fb[sel.observation_id]}</span> : ['confirmed', 'false_alarm', 'wrong_asset'].map((l) => (
                      <button key={l} className="btn small ghost" onClick={() => sendFeedback(sel.observation_id, l).then(() => setFb({ ...fb, [sel.observation_id]: l }))}>{l.replace('_', ' ')}</button>
                    ))}
                  </div>
                </div>
              </div>
              <div className="filmstrip">
                {obs.map((o) => (
                  <button key={o.observation_id} className={`film ${o.observation_id === sel.observation_id ? 'on' : ''}`} onClick={() => setSelected(o.observation_id)} title={when(o.observation_ts)}>
                    <img src={imageUrl(o.image_reference)} alt="" />
                    <span className={`bar ${o.severity_level}`} />
                  </button>
                ))}
              </div>
            </Card>
          )}

          <div className="grid two">
            <Card title={`Observation timeline (${obs.length})`}>
              <table className="table">
                <thead><tr><th>Time</th><th>Status</th><th>Condition</th><th>Severity</th><th>Match</th></tr></thead>
                <tbody>
                  {[...obs].reverse().map((o) => (
                    <tr key={o.observation_id} onClick={() => setSelected(o.observation_id)} className="clickable">
                      <td>{when(o.observation_ts)}</td><td>{o.status}</td><td>{o.condition_description}</td>
                      <td><Severity score={o.severity_score} level={o.severity_level} /></td><td>{o.match_stage}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
            <Card title="Tracked defect instances">
              {!(instances.data || []).length ? <Empty>No tracked instances.</Empty> : (
                <table className="table"><thead><tr><th>Defect</th><th>Size</th><th>Growth</th><th>Critical by</th><th>Status</th></tr></thead>
                  <tbody>{instances.data.map((i) => (
                    <tr key={i.instance_id}><td><Link to={`/defect-instances/${i.instance_id}`}>{i.defect_type}</Link></td>
                      <td>{i.area_cm2 != null ? `${fmt(i.area_cm2, 1)} cm²` : `${fmt(100 * i.area_norm, 2)}%`}{i.length_cm != null ? ` · ${fmt(i.length_cm, 1)} cm` : ''}</td>
                      <td>{i.forecast?.growth_since_last_pct != null ? `${fmt(i.forecast.growth_since_last_pct, 0)}%` : '—'}</td>
                      <td>{i.predicted_critical_ts ? when(i.predicted_critical_ts) : '—'}</td><td>{i.status}</td></tr>))}</tbody></table>
              )}
            </Card>
            <Card title="Defects">
              {(hist.data?.defects || []).length === 0 ? <Empty>None recorded.</Empty> : (
                <table className="table">
                  <thead><tr><th>Defect</th><th>Severity</th><th>Status</th><th>Notes</th></tr></thead>
                  <tbody>{hist.data.defects.map((d) => (
                    <tr key={d.defect_id}><td><Link to={`/defects/${d.defect_id}`}>{d.defect_type}</Link></td><td>{fmt(d.severity_score)}</td><td><StatusPill status={d.status} /></td><td className="small">{d.notes || '—'}</td></tr>
                  ))}</tbody>
                </table>
              )}
            </Card>
          </div>
        </>
      )}
    </div>
  )
}
