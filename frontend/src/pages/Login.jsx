import { useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { errorText } from '../api'
import { useAuth } from '../auth'
import { ErrorBox, Field } from '../components/ui'

export default function Login() {
  const { login } = useAuth()
  const nav = useNavigate()
  const loc = useLocation()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)

  const submit = async (e) => {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      const u = await login(username, password)
      nav(loc.state?.from && loc.state.from !== '/login' ? loc.state.from : u.role === 'admin' ? '/admin/metrics' : '/', { replace: true })
    } catch (err) {
      setError(err.response?.status === 401 ? 'Wrong username or password' : errorText(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="login-wrap">
      <form className="card login" onSubmit={submit}>
        <h1>AUDI</h1>
        <p className="muted">Sign in as operator, maintenance or admin. Accounts are created by <code>scripts/seed_users.py</code>.</p>
        <Field label="Username">
          <input value={username} onChange={(e) => setUsername(e.target.value)} autoComplete="username" autoFocus required />
        </Field>
        <Field label="Password">
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="current-password" required />
        </Field>
        <ErrorBox error={error} />
        <button className="btn primary" disabled={busy}>{busy ? 'Signing in…' : 'Sign in'}</button>
      </form>
    </div>
  )
}
