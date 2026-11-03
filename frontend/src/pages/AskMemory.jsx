import { useState } from 'react'
import { Link } from 'react-router-dom'
import { askMemory, getSites } from '../api'
import { Card, Empty, ErrorBox, fmt } from '../components/ui'
import { useAsync } from '../hooks'

const EXAMPLES = [
  'poles with corrosion getting worse in the last 3 months',
  'which defects will become critical within 14 days',
  'assets not inspected for 30 days',
  'top 5 assets with the worst health',
]

/** 5.2: a question becomes a structured query; filtering and ranking are deterministic. */
export default function AskMemory() {
  const sites = useAsync(getSites, [])
  const [q, setQ] = useState(EXAMPLES[0])
  const [site, setSite] = useState('')
  const [res, setRes] = useState(null)
  const [err, setErr] = useState(null)
  const [busy, setBusy] = useState(false)
  const run = async (question = q) => {
    setBusy(true); setErr(null)
    try { setRes(await askMemory(question, site)) } catch (e) { setErr(e?.response?.data?.detail || e.message) } finally { setBusy(false) }
  }
  return (
    <div className="page">
      <header className="page-head"><h2>Ask the memory</h2></header>
      <Card>
        <form className="row gap wrap" onSubmit={(e) => { e.preventDefault(); run() }}>
          <input className="grow" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Question" />
          <select value={site} onChange={(e) => setSite(e.target.value)} aria-label="Site">
            <option value="">All sites</option>
            {(sites.data || []).map((s) => <option key={s.site_id} value={s.site_id}>{s.name}</option>)}
          </select>
          <button className="btn primary" disabled={busy}>{busy ? 'Searching…' : 'Ask'}</button>
        </form>
        <div className="row gap wrap small" style={{ marginTop: '0.5rem' }}>
          {EXAMPLES.map((e) => <button key={e} className="btn ghost small" onClick={() => { setQ(e); run(e) }}>{e}</button>)}
        </div>
      </Card>
      <ErrorBox error={err} />
      {res && (
        <>
          <Card title="Interpreted as">
            <pre className="small">{JSON.stringify(Object.fromEntries(Object.entries(res.query).filter(([, v]) => v !== null && !(Array.isArray(v) && !v.length))), null, 1)}</pre>
            <div className="small muted">translator: {res.translator}</div>
          </Card>
          <Card title={`${res.count} result${res.count === 1 ? '' : 's'}`}>
            {res.results.length === 0 ? <Empty>No assets match.</Empty> : (
              <table className="table">
                <thead><tr><th>Asset</th><th>Type</th><th>Priority</th><th>Health</th><th>Days to critical</th><th>Why</th></tr></thead>
                <tbody>{res.results.map((r) => (
                  <tr key={r.asset_id}>
                    <td><Link className="mono" to={`/assets/${r.asset_id}`}>{r.asset_id}</Link></td><td>{r.asset_type}</td>
                    <td>{fmt(r.revisit_priority)}</td><td>{fmt(r.health_index, 0)}</td><td>{fmt(r.days_to_critical, 0)}</td>
                    <td className="small">{r.why.join(' · ')}</td>
                  </tr>
                ))}</tbody>
              </table>
            )}
          </Card>
        </>
      )}
    </div>
  )
}
