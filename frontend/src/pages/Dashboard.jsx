import { useState } from 'react'
import { Link } from 'react-router-dom'
import { getAssets, getSites, getSummary } from '../api'
import AssetTable from '../components/AssetTable'
import { Card, ErrorBox, Loading, Stat } from '../components/ui'
import { useAsync } from '../hooks'

export default function Dashboard() {
  const sites = useAsync(getSites, [])
  const [siteId, setSiteId] = useState('')
  const summary = useAsync(() => getSummary(siteId), [siteId])
  const assets = useAsync(() => getAssets(siteId), [siteId])

  return (
    <div className="page">
      <header className="page-head">
        <h2>Dashboard</h2>
        <select value={siteId} onChange={(e) => setSiteId(e.target.value)} aria-label="Site">
          <option value="">All sites</option>
          {(sites.data || []).map((s) => <option key={s.site_id} value={s.site_id}>{s.name}</option>)}
        </select>
      </header>
      <ErrorBox error={summary.error || assets.error} />
      {summary.data && (
        <div className="stats">
          <Stat label="Assets in memory" value={summary.data.total_assets} />
          <Stat label="Unresolved defects" value={summary.data.active_defects} tone={summary.data.active_defects ? 'crit' : ''} />
          <Stat label="Priority assets (latest S above threshold)" value={summary.data.priority_assets} tone={summary.data.priority_assets ? 'warn' : ''} />
          <Stat label="Missions" value={summary.data.missions} />
        </div>
      )}
      <Card title="Assets by revisit priority" actions={<Link to="/assets">All assets →</Link>}>
        {assets.loading ? <Loading /> : <AssetTable assets={(assets.data || []).slice(0, 15)} />}
      </Card>
    </div>
  )
}
