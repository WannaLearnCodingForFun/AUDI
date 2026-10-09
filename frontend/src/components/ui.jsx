export const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? '—' : Number(v).toFixed(d))
export const pct = (v) => (v === null || v === undefined ? '—' : `${(v * 100).toFixed(1)}%`)
export const when = (ts) => (ts ? new Date(ts).toLocaleString() : '—')

export function Loading({ what = 'Loading' }) {
  return <div className="muted pad">{what}…</div>
}

export function ErrorBox({ error }) {
  return error ? <div className="error-box">{error}</div> : null
}

export function Card({ title, actions, children, className = '' }) {
  return (
    <section className={`card ${className}`}>
      {(title || actions) && (
        <header className="card-head">
          {title && <h3>{title}</h3>}
          {actions && <div className="row gap">{actions}</div>}
        </header>
      )}
      {children}
    </section>
  )
}

export function Stat({ label, value, tone }) {
  return (
    <div className={`stat ${tone || ''}`}>
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  )
}

const LEVEL_TONE = { low: 'ok', medium: 'warn', priority: 'high', critical: 'crit' }

export function Severity({ score, level }) {
  return <span className={`pill ${LEVEL_TONE[level] || 'ok'}`}>{fmt(score)} {level}</span>
}

export function StatusPill({ status }) {
  const tone = { Open: 'crit', 'In Progress': 'warn', Resolved: 'ok', active: 'warn', ended: 'ok', interrupted: 'crit', online: 'ok', offline: 'warn' }[status] || ''
  return <span className={`pill ${tone}`}>{status}</span>
}

export function Simulated({ on }) {
  return on ? <span className="pill sim" title="Generated for demonstration, not a real inspection">simulated</span> : null
}

export function Empty({ children }) {
  return <div className="muted pad">{children}</div>
}

export function Field({ label, children, hint }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
      {hint && <small className="muted">{hint}</small>}
    </label>
  )
}
