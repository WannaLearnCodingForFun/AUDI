import { useEffect, useState } from 'react'
import { searchMemory, getAssetHistory, getAssets } from '../api'
import ObservationTimeline from '../components/ObservationTimeline'

export default function SemanticMemory() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [selected, setSelected] = useState(null)
  const [history, setHistory] = useState(null)
  const [error, setError] = useState(null)

  const handleSearch = async (q) => {
    setQuery(q)
    if (!q.trim()) { setResults([]); return }
    try {
      const data = await searchMemory(q)
      setResults(data.results || [])
      setError(null)
    } catch (e) {
      setError(e.message)
    }
  }

  const selectAsset = async (assetId) => {
    setSelected(assetId)
    try {
      const data = await getAssetHistory(assetId)
      setHistory(data)
    } catch (e) {
      setError(e.message)
    }
  }

  useEffect(() => {
    getAssets()
      .then(d => setResults(d.assets || []))
      .catch(e => setError(e.message))
  }, [])

  return (
    <div>
      <h2 className="page-title">Semantic Memory</h2>
      <p className="page-subtitle">Search and explore persistent asset memory</p>

      <input
        className="search-input"
        placeholder="Search asset..."
        value={query}
        onChange={e => handleSearch(e.target.value)}
      />

      {error && <div className="error-box">{error}</div>}

      <div style={{ display: 'grid', gridTemplateColumns: selected ? '1fr 2fr' : '1fr', gap: '1.5rem' }}>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Asset</th>
                <th>Condition</th>
                <th>Observations</th>
                <th>Similarity</th>
              </tr>
            </thead>
            <tbody>
              {results.map(r => (
                <tr
                  key={r.asset_id}
                  onClick={() => selectAsset(r.asset_id)}
                  style={{ cursor: 'pointer', background: selected === r.asset_id ? 'var(--surface2)' : undefined }}
                >
                  <td style={{ fontFamily: 'var(--mono)', fontSize: '0.75rem' }}>{r.asset_id}</td>
                  <td>{r.condition}</td>
                  <td>{r.observations_count}</td>
                  <td>{r.simulated_history ? 'Demo' : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {history && (
          <div>
            <h3 style={{ marginBottom: '0.5rem' }}>{history.asset_id}</h3>
            {history.simulated_history && (
              <div className="banner" style={{ marginBottom: '1rem' }}>
                <strong>Simulated Multi-Mission History</strong>
              </div>
            )}
            <ObservationTimeline
              observations={history.observations}
              revisitInfo={history.revisit_priority}
            />
          </div>
        )}
      </div>
    </div>
  )
}
