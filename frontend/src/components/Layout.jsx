import { useEffect, useState } from 'react'
import { NavLink, Outlet, useNavigate } from 'react-router-dom'
import { getHealth, getNotifications } from '../api'
import { hasRole, useAuth } from '../auth'
import { useSocket } from '../hooks'

const NAV = [
  { to: '/', label: 'Dashboard', roles: ['operator', 'maintenance'], end: true },
  { to: '/sites', label: 'Pre-flight briefing', roles: ['operator'] },
  { to: '/missions', label: 'Missions & reports', roles: ['operator', 'maintenance'] },
  { to: '/inbox', label: 'Notifications', roles: ['maintenance'], badge: true },
  { to: '/defects', label: 'Defects', roles: ['maintenance', 'operator'] },
  { to: '/assets', label: 'Assets', roles: ['operator', 'maintenance'] },
  { to: '/memory', label: 'Analyse & search', roles: ['operator', 'maintenance'] },
  { to: '/admin/config', label: 'Configuration', roles: [] },
  { to: '/admin/users', label: 'Sites & users', roles: [] },
  { to: '/admin/metrics', label: 'Metrics & sync', roles: [] },
]

export default function Layout() {
  const { user, logout } = useAuth()
  const nav = useNavigate()
  const [health, setHealth] = useState(null)
  const [unread, setUnread] = useState(0)
  const [toast, setToast] = useState(null)
  const receivesEscalations = hasRole(user, 'maintenance')

  useEffect(() => {
    const load = () => getHealth().then(setHealth).catch(() => setHealth(null))
    load()
    const t = setInterval(load, 15000)
    return () => clearInterval(t)
  }, [])

  useEffect(() => {
    if (receivesEscalations) getNotifications(true).then((n) => setUnread(n.length)).catch(() => {})
  }, [receivesEscalations])

  useSocket('/ws/notifications', (msg) => {
    if (msg.type === 'hello') setUnread(msg.unread)
    if (msg.type === 'notification') {
      setUnread((u) => u + 1)
      setToast(msg.notification)
      setTimeout(() => setToast(null), 8000)
    }
  }, receivesEscalations)

  const online = health?.sync?.online
  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">
          <h1>AUDI</h1>
          <p>Autonomous inspection memory</p>
        </div>
        <nav>
          {NAV.filter((n) => hasRole(user, ...n.roles)).map((n) => (
            <NavLink key={n.to} to={n.to} end={n.end} className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              {n.label}
              {n.badge && unread > 0 && <span className="badge">{unread}</span>}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-foot">
          <div className="muted small">
            {health ? (
              <>
                <span className={`dot ${online ? 'ok' : 'warn'}`} /> cloud {online ? 'online' : 'offline'} · {health.backbone}
              </>
            ) : (
              <><span className="dot crit" /> API unreachable</>
            )}
          </div>
          <div className="row between">
            <span>{user.username} <span className="muted small">({user.role})</span></span>
            <button className="btn ghost small" onClick={() => { logout(); nav('/login') }}>Log out</button>
          </div>
        </div>
      </aside>
      <main className="main">
        {toast && (
          <div className="toast crit" onClick={() => { setToast(null); nav('/inbox') }}>
            <strong>Critical defect</strong> — {toast.message}
          </div>
        )}
        <Outlet context={{ setUnread }} />
      </main>
    </div>
  )
}
