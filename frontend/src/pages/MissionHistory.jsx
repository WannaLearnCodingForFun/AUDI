import { useState } from 'react'
import { simulateMission, getMissions } from '../api'
import MissionSummary from '../components/MissionSummary'

export default function MissionHistory() {
  const [result, setResult] = useState(null)
  const [missions, setMissions] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const runMission = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await simulateMission()
      setResult(data)
      const m = await getMissions()
      setMissions(m.missions)
    } catch (e) {
      setError(e.response?.data?.detail || e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <h2 className="page-title">Mission History</h2>
      <p className="page-subtitle">Simulated drone inspection missions</p>

      <div className="banner">
        <strong>Research Demonstration</strong> — Missions process sample PV panel images through the full semantic memory pipeline, demonstrating persistence across flights.
      </div>

      <button className="btn btn-primary" onClick={runMission} disabled={loading}>
        {loading ? 'Running Mission...' : 'Simulate Mission'}
      </button>

      {error && <div className="error-box" style={{ marginTop: '1rem' }}>{error}</div>}
      <MissionSummary result={result} />

      {missions && (
        <div style={{ marginTop: '2rem' }}>
          <h3 style={{ marginBottom: '1rem' }}>Recorded Missions</h3>
          {Object.entries(missions).map(([mid, obs]) => (
            <div key={mid} className="card" style={{ marginBottom: '0.75rem' }}>
              <div style={{ fontWeight: 600, fontFamily: 'var(--mono)' }}>{mid}</div>
              <div style={{ fontSize: '0.8rem', color: 'var(--muted)', marginTop: '0.25rem' }}>
                {obs.length} observation(s)
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
