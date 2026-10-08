export default function AssetTable({ assets, onSelect }) {
  if (!assets?.length) return <p className="loading">No assets in semantic memory.</p>

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Asset ID</th>
            <th>Condition</th>
            <th>Severity</th>
            <th>Revisit Priority</th>
            <th>Last Inspected</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {assets.map(a => (
            <tr key={a.asset_id} onClick={() => onSelect?.(a.asset_id)} style={{ cursor: onSelect ? 'pointer' : 'default' }}>
              <td style={{ fontFamily: 'var(--mono)', fontSize: '0.8rem' }}>{a.asset_id}</td>
              <td>{a.condition}</td>
              <td>{(a.severity_score ?? 0).toFixed(2)}</td>
              <td>{(a.revisit_priority ?? 0).toFixed(2)}</td>
              <td>{a.last_inspected ? a.last_inspected.slice(0, 16).replace('T', ' ') : '—'}</td>
              <td>
                {(a.severity_score ?? 0) > 0.5
                  ? <span className="badge badge-high">Priority</span>
                  : a.condition !== 'Normal'
                    ? <span className="badge badge-warn">Defect</span>
                    : <span className="badge badge-normal">Normal</span>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
