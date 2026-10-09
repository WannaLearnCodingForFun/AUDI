import { useState } from 'react'
import { createSite, createUser, errorText, getSites, getUsers, setUserSites, updateUser } from '../api'
import { useAuth } from '../auth'
import { Card, ErrorBox, Field, Loading, when } from '../components/ui'
import { useAsync } from '../hooks'

const ROLES = ['operator', 'maintenance', 'admin']

function UserRow({ u, sites, self, onChanged, onError }) {
  const [pw, setPw] = useState('')
  const act = (fn) => fn().then(onChanged).catch((e) => onError(errorText(e)))
  const toggleSite = (sid) => {
    const next = u.sites.includes(sid) ? u.sites.filter((x) => x !== sid) : [...u.sites, sid]
    act(() => setUserSites(u.user_id, next))
  }
  return (
    <tr className={u.active ? '' : 'muted'}>
      <td>{u.username}{self && <span className="muted small"> (you)</span>}</td>
      <td>
        <select value={u.role} disabled={self} onChange={(e) => act(() => updateUser(u.user_id, { role: e.target.value }))} aria-label="Role">
          {ROLES.map((r) => <option key={r}>{r}</option>)}
        </select>
      </td>
      <td>{u.email || '—'}</td>
      <td>
        <div className="chips">
          {sites.map((s) => (
            <button key={s.site_id} className={`chip ${u.sites.includes(s.site_id) ? 'on' : ''}`} onClick={() => toggleSite(s.site_id)} title="Toggle subscription (escalations go to subscribed maintenance users)">{s.site_id}</button>
          ))}
        </div>
      </td>
      <td>{when(u.created_ts)}</td>
      <td className="row gap">
        <button className="btn small ghost" disabled={self} onClick={() => act(() => updateUser(u.user_id, { active: !u.active }))}>{u.active ? 'Deactivate' : 'Activate'}</button>
        <input type="password" value={pw} onChange={(e) => setPw(e.target.value)} aria-label={`New password for ${u.username}`} title="New password (min 8 characters)" className="narrow" />
        <button className="btn small" disabled={pw.length < 8} onClick={() => act(() => updateUser(u.user_id, { password: pw })).then(() => setPw(''))}>Set password</button>
      </td>
    </tr>
  )
}

export default function AdminUsers() {
  const { user } = useAuth()
  const users = useAsync(getUsers, [])
  const sites = useAsync(getSites, [])
  const [error, setError] = useState(null)
  const [site, setSite] = useState({ site_id: '', name: '', latitude: '', longitude: '' })
  const [nu, setNu] = useState({ username: '', password: '', role: 'operator', email: '' })

  const addSite = async (e) => {
    e.preventDefault()
    setError(null)
    try {
      await createSite({
        site_id: site.site_id.trim(), name: site.name.trim(),
        latitude: site.latitude === '' ? null : Number(site.latitude), longitude: site.longitude === '' ? null : Number(site.longitude),
      })
      setSite({ site_id: '', name: '', latitude: '', longitude: '' })
      sites.reload()
    } catch (err) { setError(errorText(err)) }
  }
  const addUser = async (e) => {
    e.preventDefault()
    setError(null)
    try {
      await createUser({ ...nu, email: nu.email || null, sites: [] })
      setNu({ username: '', password: '', role: 'operator', email: '' })
      users.reload()
    } catch (err) { setError(errorText(err)) }
  }

  return (
    <div className="page">
      <header className="page-head"><h2>Sites &amp; users</h2></header>
      <ErrorBox error={error || users.error || sites.error} />
      <div className="grid two">
        <Card title="Create site">
          <form className="form-grid" onSubmit={addSite}>
            <Field label="Site id" hint="letters, digits, _ and -"><input value={site.site_id} onChange={(e) => setSite({ ...site, site_id: e.target.value })} required /></Field>
            <Field label="Name"><input value={site.name} onChange={(e) => setSite({ ...site, name: e.target.value })} required /></Field>
            <Field label="Latitude (optional)"><input type="number" step="any" value={site.latitude} onChange={(e) => setSite({ ...site, latitude: e.target.value })} /></Field>
            <Field label="Longitude (optional)"><input type="number" step="any" value={site.longitude} onChange={(e) => setSite({ ...site, longitude: e.target.value })} /></Field>
            <button className="btn primary">Create site</button>
          </form>
          <ul className="plain small">{(sites.data || []).map((s) => <li key={s.site_id}><span className="mono">{s.site_id}</span> — {s.name}</li>)}</ul>
        </Card>
        <Card title="Create user">
          <form className="form-grid" onSubmit={addUser}>
            <Field label="Username"><input value={nu.username} minLength={3} onChange={(e) => setNu({ ...nu, username: e.target.value })} required /></Field>
            <Field label="Password" hint="at least 8 characters"><input type="password" minLength={8} value={nu.password} onChange={(e) => setNu({ ...nu, password: e.target.value })} required /></Field>
            <Field label="Role"><select value={nu.role} onChange={(e) => setNu({ ...nu, role: e.target.value })}>{ROLES.map((r) => <option key={r}>{r}</option>)}</select></Field>
            <Field label="Email (escalation copies)"><input type="email" value={nu.email} onChange={(e) => setNu({ ...nu, email: e.target.value })} /></Field>
            <button className="btn primary">Create user</button>
          </form>
        </Card>
      </div>
      <Card title="Users">
        {users.loading ? <Loading /> : (
          <table className="table">
            <thead><tr><th>User</th><th>Role</th><th>Email</th><th>Site subscriptions</th><th>Created</th><th /></tr></thead>
            <tbody>
              {(users.data || []).map((u) => (
                <UserRow key={u.user_id} u={u} sites={sites.data || []} self={u.user_id === user.user_id} onChanged={users.reload} onError={setError} />
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </div>
  )
}
