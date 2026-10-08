import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { getAssetHistory } from '../api'
import ObservationTimeline from '../components/ObservationTimeline'

export default function AssetDetail() {
  const { id } = useParams()
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    getAssetHistory(id)
      .then(setData)
      .catch(e => setError(e.response?.data?.detail || e.message))
  }, [id])

  if (error) return <div className="error-box">{error}</div>
  if (!data) return <div className="loading">Loading asset...</div>

  return (
    <div>
      <Link to="/assets" style={{ fontSize: '0.875rem' }}>← Back to Assets</Link>
      <h2 className="page-title" style={{ marginTop: '1rem' }}>{data.asset_id}</h2>
      <p className="page-subtitle">{data.asset_type} · {data.current_condition}</p>

      {data.simulated_history && (
        <div className="banner">
          <strong>Simulated Multi-Mission History</strong> — Observations are constructed from independent Kaggle images, not real repeated inspections of the same physical panel.
        </div>
      )}

      <div className="cards">
        <div className="card">
          <div className="card-label">Current Condition</div>
          <div className="card-value" style={{ fontSize: '1.1rem' }}>{data.current_condition}</div>
        </div>
        <div className="card">
          <div className="card-label">Observations</div>
          <div className="card-value" style={{ fontSize: '1.1rem' }}>{data.observations?.length ?? 0}</div>
        </div>
      </div>

      <h3 style={{ marginTop: '1.5rem', marginBottom: '0.5rem' }}>Observation Timeline</h3>
      <ObservationTimeline
        observations={data.observations}
        revisitInfo={data.revisit_priority}
      />
    </div>
  )
}
