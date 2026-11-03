import { useState } from 'react'
import { Link } from 'react-router-dom'
import { addRelation, deleteRelation, errorText, getAssetHistory, getAudit, getDrones, getRelations, getSites, mergeAssets, registerDrone, revokeDrone, splitAsset } from '../api'
import { Card, Empty, ErrorBox, Field, when } from '../components/ui'
import { useAsync } from '../hooks'

/** 1.4 merge / split + audit log, 5.1 asset relations, 6.3 drone registry. */
export default function AdminMemory() {
  const sites = useAsync(getSites, [])
  const [site, setSite] = useState('site_demo')
  const audit = useAsync(() => getAudit(site), [site])
  const rels = useAsync(() => getRelations(site), [site])
  const drones = useAsync(getDrones, [])
  const [err, setErr] = useState(null)
  const [msg, setMsg] = useState(null)
  const [m, setM] = useState({ keep: '', merge: '' })
  const [sp, setSp] = useState({ asset: '', obs: [] })
  const [spObs, setSpObs] = useState([])
  const [rel, setRel] = useState({ parent: '', child: '', relation: 'contains' })
  const [dr, setDr] = useState({ name: '', sites: '' })
  const act = async (fn, ok) => { setErr(null); setMsg(null); try { const r = await fn(); setMsg(ok(r)); audit.reload(); rels.reload(); drones.reload() } catch (e) { setErr(errorText(e)) } }
  const loadSplit = async () => { try { setSpObs((await getAssetHistory(sp.asset)).observations); setSp({ ...sp, obs: [] }) } catch (e) { setErr(errorText(e)) } }
  return (
    <div className="page">
      <header className="page-head"><h2>Memory administration</h2>
        <select value={site} onChange={(e) => setSite(e.target.value)} aria-label="Site">{(sites.data || []).map((s) => <option key={s.site_id} value={s.site_id}>{s.name}</option>)}</select>
      </header>
      <ErrorBox error={err} />{msg && <div className="ok-box">{msg}</div>}
      <div className="grid two">
        <Card title="Merge two records of one asset">
          <div className="stack">
            <Field label="Keep asset id"><input value={m.keep} onChange={(e) => setM({ ...m, keep: e.target.value.trim() })} /></Field>
            <Field label="Merge (delete) asset id" hint="its observations, defects and history move to the kept asset"><input value={m.merge} onChange={(e) => setM({ ...m, merge: e.target.value.trim() })} /></Field>
            <button className="btn danger" disabled={!m.keep || !m.merge} onClick={() => act(() => mergeAssets(m.keep, m.merge), (r) => `Merged: ${r.counts.observations} observations moved`)}>Merge</button>
          </div>
        </Card>
        <Card title="Split one record into two">
          <div className="stack">
            <div className="row gap"><input className="grow" placeholder="asset id" value={sp.asset} onChange={(e) => setSp({ ...sp, asset: e.target.value.trim() })} /><button className="btn" onClick={loadSplit}>Load</button></div>
            {spObs.length > 0 && <div className="scroll small">{spObs.map((o) => (
              <label key={o.observation_id} className="row gap"><input type="checkbox" checked={sp.obs.includes(o.observation_id)}
                onChange={(e) => setSp({ ...sp, obs: e.target.checked ? [...sp.obs, o.observation_id] : sp.obs.filter((x) => x !== o.observation_id) })} />
                {when(o.observation_ts)} · {o.condition_description} · {o.match_stage}</label>))}</div>}
            <button className="btn danger" disabled={!sp.obs.length} onClick={() => act(() => splitAsset(sp.asset, sp.obs), (r) => `Split: new asset ${r.new_asset.asset_id}`)}>Move selected to a new asset</button>
          </div>
        </Card>
        <Card title="Asset relations (knowledge graph)">
          <div className="row gap wrap">
            <input placeholder="parent (depends on child)" value={rel.parent} onChange={(e) => setRel({ ...rel, parent: e.target.value.trim() })} />
            <select value={rel.relation} onChange={(e) => setRel({ ...rel, relation: e.target.value })}>{['contains', 'feeds', 'supports', 'part_of'].map((r) => <option key={r}>{r}</option>)}</select>
            <input placeholder="child" value={rel.child} onChange={(e) => setRel({ ...rel, child: e.target.value.trim() })} />
            <button className="btn" onClick={() => act(() => addRelation(rel.parent, rel.child, rel.relation), () => 'Relation added')}>Add</button>
          </div>
          {!rels.data?.length ? <Empty>No relations.</Empty> : (
            <table className="table"><tbody>{rels.data.map((r) => (
              <tr key={r.relation_id}><td className="mono"><Link to={`/assets/${r.parent_id}`}>{r.parent_id}</Link></td><td>{r.relation}</td>
                <td className="mono"><Link to={`/assets/${r.child_id}`}>{r.child_id}</Link></td>
                <td><button className="btn small ghost" onClick={() => act(() => deleteRelation(r.relation_id), () => 'Relation removed')}>remove</button></td></tr>))}</tbody></table>
          )}
        </Card>
        <Card title="Drones">
          <div className="row gap wrap">
            <input placeholder="name" value={dr.name} onChange={(e) => setDr({ ...dr, name: e.target.value })} />
            <input placeholder="site ids, comma separated" value={dr.sites} onChange={(e) => setDr({ ...dr, sites: e.target.value })} />
            <button className="btn" onClick={() => act(() => registerDrone({ name: dr.name, site_ids: dr.sites.split(',').map((x) => x.trim()).filter(Boolean) }),
              (r) => `Drone ${r.drone_id} registered. API key (shown once): ${r.api_key}`)}>Register</button>
          </div>
          <table className="table"><tbody>{(drones.data || []).map((d) => (
            <tr key={d.drone_id}><td>{d.name}</td><td className="mono small">{d.api_key_prefix}…</td><td className="small">{d.site_ids.join(', ')}</td>
              <td className="small">{d.scopes.join(' ')}</td><td>{d.active ? 'active' : 'revoked'}</td><td className="small">{when(d.last_seen_ts)}</td>
              <td>{d.active && <button className="btn small ghost" onClick={() => act(() => revokeDrone(d.drone_id), () => 'Key revoked')}>revoke</button>}</td></tr>))}</tbody></table>
        </Card>
      </div>
      <Card title="Audit log">
        {!audit.data?.length ? <Empty>No entries.</Empty> : (
          <table className="table"><thead><tr><th>Time</th><th>Actor</th><th>Action</th><th>Entity</th><th>Details</th></tr></thead>
            <tbody>{audit.data.map((a) => (
              <tr key={a.audit_id}><td>{when(a.ts)}</td><td>{a.actor}</td><td>{a.action}</td><td className="mono small">{a.entity_id}</td>
                <td className="small mono">{JSON.stringify(a.details?.counts || a.details || {}).slice(0, 140)}</td></tr>))}</tbody></table>
        )}
      </Card>
    </div>
  )
}
