import { useState } from 'react'
import { getAssets, getSites } from '../api'
import AssetTable from '../components/AssetTable'
import { Card, ErrorBox, Loading } from '../components/ui'
import { useAsync } from '../hooks'

export default function Assets() {
  const sites = useAsync(getSites, [])
  const [siteId, setSiteId] = useState('')
  const [type, setType] = useState('')
  const assets = useAsync(() => getAssets(siteId), [siteId])
  const rows = (assets.data || []).filter((a) => !type || a.asset_type === type)
  const types = [...new Set((assets.data || []).map((a) => a.asset_type))]
  return (
    <div className="page">
      <header className="page-head">
        <h2>Assets ({rows.length})</h2>
        <div className="row gap">
          <select value={siteId} onChange={(e) => setSiteId(e.target.value)} aria-label="Site">
            <option value="">All sites</option>
            {(sites.data || []).map((s) => <option key={s.site_id} value={s.site_id}>{s.name}</option>)}
          </select>
          <select value={type} onChange={(e) => setType(e.target.value)} aria-label="Asset type">
            <option value="">All types</option>
            {types.map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
        </div>
      </header>
      <ErrorBox error={assets.error} />
      <Card>{assets.loading ? <Loading /> : <AssetTable assets={rows} />}</Card>
    </div>
  )
}
