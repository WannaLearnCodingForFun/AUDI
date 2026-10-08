export default function MissionSummary({ result }) {
  if (!result) return null

  return (
    <div className="mission-result">
      <h3 style={{ color: 'var(--accent2)' }}>{result.mission_id?.toUpperCase()}</h3>
      <div className="mission-stats">
        <div className="stat">
          <div className="stat-value">{result.assets_processed}</div>
          <div className="stat-label">Assets Processed</div>
        </div>
        <div className="stat">
          <div className="stat-value">{result.existing_matched}</div>
          <div className="stat-label">Existing Matched</div>
        </div>
        <div className="stat">
          <div className="stat-value">{result.new_assets}</div>
          <div className="stat-label">New Assets</div>
        </div>
        <div className="stat">
          <div className="stat-value">{result.changes_detected}</div>
          <div className="stat-label">Changes Detected</div>
        </div>
        <div className="stat">
          <div className="stat-value">{result.priority_alerts}</div>
          <div className="stat-label">Priority Alerts</div>
        </div>
      </div>
    </div>
  )
}
