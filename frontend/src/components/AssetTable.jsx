import { Link } from 'react-router-dom'
import { Empty, fmt, Simulated, when } from './ui'

export default function AssetTable({ assets }) {
  if (!assets.length) return <Empty>No assets yet — fly a mission or run <code>make memory</code>.</Empty>
  return (
    <table className="table">
      <thead>
        <tr>
          <th>Asset</th><th>Type</th><th>Site</th><th>Condition</th><th>Revisit R</th><th>Last checked</th><th />
        </tr>
      </thead>
      <tbody>
        {assets.map((a) => (
          <tr key={a.asset_id}>
            <td><Link to={`/assets/${a.asset_id}`} className="mono">{a.asset_id}</Link></td>
            <td>{a.asset_type}</td>
            <td>{a.site_id}</td>
            <td className={a.current_is_defect ? 'text-crit' : ''}>{a.current_condition}</td>
            <td>{fmt(a.revisit_priority_score)}</td>
            <td>{when(a.last_checked_ts)}</td>
            <td><Simulated on={a.simulated} /></td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
