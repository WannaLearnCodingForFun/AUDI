export default function AnalysisResult({ result }) {
  if (!result) return null
  if (result.error) return <div className="error-box">{result.error}</div>

  const isNew = result.is_new_asset

  return (
    <div className="result-panel">
      <h3>{isNew ? '🆕 NEW ASSET DETECTED' : '✓ MATCHED ASSET'}</h3>
      <div className="result-grid">
        <div className="result-item">
          <label>Asset ID</label>
          <span>{result.asset_id || '—'}</span>
        </div>
        <div className="result-item">
          <label>Match Confidence</label>
          <span>{((result.match_confidence ?? 0) * 100).toFixed(0)}%</span>
        </div>
        <div className="result-item">
          <label>Current Condition</label>
          <span>{result.current_condition || '—'}</span>
        </div>
        {!isNew && (
          <div className="result-item">
            <label>Previous Condition</label>
            <span>{result.previous_condition || '—'}</span>
          </div>
        )}
        {result.inferred_source_class && (
          <div className="result-item">
            <label>Inferred Source Class</label>
            <span>{result.inferred_source_class}</span>
          </div>
        )}
        <div className="result-item">
          <label>Change Magnitude</label>
          <span>{(result.change_magnitude ?? 0).toFixed(2)}</span>
        </div>
        <div className="result-item">
          <label>Severity</label>
          <span>{(result.severity_score ?? 0).toFixed(2)}</span>
        </div>
        <div className="result-item">
          <label>Priority</label>
          <span className={result.priority ? 'badge badge-high' : 'badge badge-normal'}>
            {result.priority ? 'HIGH' : 'NORMAL'}
          </span>
        </div>
        <div className="result-item">
          <label>Revisit Priority</label>
          <span>{(result.revisit_priority ?? 0).toFixed(2)}</span>
        </div>
      </div>
    </div>
  )
}
