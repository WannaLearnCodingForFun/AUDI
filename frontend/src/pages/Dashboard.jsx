import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { getSummary } from '../api'
import SummaryCard from '../components/SummaryCard'
import AssetTable from '../components/AssetTable'

export default function Dashboard() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const navigate = useNavigate()

  useEffect(() => {
    getSummary()
      .then(setData)
      .catch(e => setError(e.response?.data?.detail || e.message))
  }, [])

  if (error) return <div className="error-box">API unavailable: {error}</div>
  if (!data) return <div className="loading">Loading dashboard...</div>

  const priority = data.priority_assets_list || []

  return (
    <div>
      <h2 className="page-title">Dashboard</h2>
      <p className="page-subtitle">Persistent semantic memory overview</p>

      <div className="banner">
        <strong>Research Demonstration</strong> — Simulated multi-mission histories built on Kaggle PV defect images.
        Not production drone inspection data.
      </div>

      <div className="cards">
        <SummaryCard label="Total Assets" value={data.total_assets} color="var(--accent2)" />
        <SummaryCard label="Active Defects" value={data.active_defects} color="var(--warning)" />
        <SummaryCard label="Priority Assets" value={data.priority_assets} color="var(--danger)" />
        <SummaryCard label="Missions" value={data.missions} />
      </div>

      <h3 style={{ marginBottom: '1rem', fontSize: '1rem' }}>Priority Assets</h3>
      <AssetTable assets={priority} onSelect={id => navigate(`/assets/${id}`)} />
    </div>
  )
}
