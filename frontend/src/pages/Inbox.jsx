import { useCallback, useState } from 'react'
import { Link, useOutletContext } from 'react-router-dom'
import { errorText, getNotifications, markRead } from '../api'
import { Card, Empty, ErrorBox, fmt, Loading, when } from '../components/ui'
import { useAsync, useSocket } from '../hooks'

export default function Inbox() {
  const [unreadOnly, setUnreadOnly] = useState(false)
  const list = useAsync(() => getNotifications(unreadOnly), [unreadOnly])
  const [error, setError] = useState(null)
  const { setUnread } = useOutletContext()
  const reload = list.reload
  const onMessage = useCallback((m) => { if (m.type === 'notification') reload() }, [reload])
  const sock = useSocket('/ws/notifications', onMessage, true)

  const read = async (id) => {
    try {
      await markRead(id)
      setUnread((u) => Math.max(0, u - 1))
      reload()
    } catch (e) { setError(errorText(e)) }
  }

  return (
    <div className="page">
      <header className="page-head">
        <h2>Notifications</h2>
        <div className="row gap">
          <span className="small muted"><span className={`dot ${sock.connected ? 'ok' : 'warn'}`} /> live {sock.connected ? 'connected' : 'reconnecting'}</span>
          <label className="row gap small"><input type="checkbox" checked={unreadOnly} onChange={(e) => setUnreadOnly(e.target.checked)} /> unread only</label>
        </div>
      </header>
      <p className="muted">Critical-severity defects at the sites you are subscribed to (FR-19). Email copies are sent when SMTP is configured.</p>
      <ErrorBox error={error || list.error} />
      <Card>
        {list.loading ? <Loading /> : (list.data || []).length === 0 ? <Empty>No notifications.</Empty> : (
          <table className="table">
            <thead><tr><th>Received</th><th>Site</th><th>Asset</th><th>Severity</th><th>Message</th><th>Email</th><th /></tr></thead>
            <tbody>
              {list.data.map((n) => (
                <tr key={n.notification_id} className={n.read ? 'muted' : 'unread'}>
                  <td>{when(n.created_ts)}</td><td>{n.site_id}</td>
                  <td>{n.asset_id && <Link to={`/assets/${n.asset_id}`} className="mono">{n.asset_id}</Link>}</td>
                  <td>{fmt(n.severity_score)}</td>
                  <td>{n.defect_id ? <Link to={`/defects/${n.defect_id}`}>{n.message}</Link> : n.message}</td>
                  <td className="small">{n.email_status || '—'}</td>
                  <td>{!n.read && <button className="btn small" onClick={() => read(n.notification_id)}>Mark read</button>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </div>
  )
}
