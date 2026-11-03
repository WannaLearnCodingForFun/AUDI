import { useState } from 'react'
import { Link } from 'react-router-dom'
import { errorText, getWorkOrders, updateWorkOrder } from '../api'
import { hasRole, useAuth } from '../auth'
import { Card, Empty, ErrorBox, Loading, when } from '../components/ui'
import { useAsync } from '../hooks'

const NEXT = { Open: ['Assigned'], Assigned: ['Fixed', 'Open'], Fixed: ['Assigned'], Verified: [] }
const TONE = { Open: 'crit', Assigned: 'warn', Fixed: 'high', Verified: 'ok' }

/** 6.2: Open → Assigned → Fixed → Verified (set automatically by the next flight). */
export default function WorkOrders() {
  const { user } = useAuth()
  const [status, setStatus] = useState('')
  const list = useAsync(() => getWorkOrders({ status: status || undefined }), [status])
  const [assignee, setAssignee] = useState({})
  const [err, setErr] = useState(null)
  const canEdit = hasRole(user, 'maintenance')
  const move = async (w, to) => {
    setErr(null)
    try { await updateWorkOrder(w.work_order_id, { status: to, assignee: assignee[w.work_order_id] || w.assignee || undefined }); list.reload() } catch (e) { setErr(errorText(e)) }
  }
  return (
    <div className="page">
      <header className="page-head">
        <h2>Work orders</h2>
        <select value={status} onChange={(e) => setStatus(e.target.value)} aria-label="Status filter">
          <option value="">All</option>{Object.keys(NEXT).map((s) => <option key={s}>{s}</option>)}
        </select>
      </header>
      <ErrorBox error={err || list.error} />
      <Card>
        {list.loading ? <Loading /> : !list.data?.length ? <Empty>No work orders.</Empty> : (
          <table className="table">
            <thead><tr><th>Title</th><th>Asset</th><th>Priority</th><th>Status</th><th>Assignee</th><th>History</th><th></th></tr></thead>
            <tbody>{list.data.map((w) => (
              <tr key={w.work_order_id}>
                <td>{w.title}</td>
                <td><Link className="mono" to={`/assets/${w.asset_id}`}>{w.asset_id}</Link></td>
                <td>{w.priority}</td>
                <td><span className={`pill ${TONE[w.status]}`}>{w.status}</span></td>
                <td>{canEdit && w.status === 'Open'
                  ? <input value={assignee[w.work_order_id] || ''} placeholder="crew" onChange={(e) => setAssignee({ ...assignee, [w.work_order_id]: e.target.value })} aria-label="Assignee" />
                  : (w.assignee || '—')}</td>
                <td className="small">{w.history.map((h) => `${h.status} ${when(h.ts)}${h.note ? ` (${h.note})` : ''}`).join(' → ')}</td>
                <td className="row gap">{canEdit && NEXT[w.status].map((to) => <button key={to} className="btn small" onClick={() => move(w, to)}>{to}</button>)}</td>
              </tr>
            ))}</tbody>
          </table>
        )}
      </Card>
    </div>
  )
}
