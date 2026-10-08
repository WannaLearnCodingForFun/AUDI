import SeverityChart from './SeverityChart'

export default function ObservationTimeline({ observations, revisitInfo }) {
  if (!observations?.length) return <p className="loading">No observations recorded.</p>

  return (
    <div>
      {revisitInfo && (
        <div className="cards" style={{ marginBottom: '1rem' }}>
          <div className="card">
            <div className="card-label">Recent Severity</div>
            <div className="card-value" style={{ fontSize: '1.25rem' }}>
              {(revisitInfo.recent_severity ?? 0).toFixed(2)}
            </div>
          </div>
          <div className="card">
            <div className="card-label">Trend Rate</div>
            <div className="card-value" style={{ fontSize: '1.25rem' }}>
              {(revisitInfo.trend ?? 0).toFixed(2)}
            </div>
          </div>
          <div className="card">
            <div className="card-label">Days Since Inspection</div>
            <div className="card-value" style={{ fontSize: '1.25rem' }}>
              {(revisitInfo.days_since_inspection ?? 0).toFixed(2)}
            </div>
          </div>
          <div className="card">
            <div className="card-label">Revisit Priority</div>
            <div className="card-value" style={{ fontSize: '1.25rem', color: 'var(--accent2)' }}>
              {(revisitInfo.revisit_priority ?? 0).toFixed(2)}
            </div>
          </div>
        </div>
      )}

      <div className="timeline">
        {observations.map((obs, i) => (
          <div className="timeline-item" key={obs.observation_id || i}>
            <div style={{ fontSize: '0.75rem', color: 'var(--muted)' }}>
              {obs.mission_id || `Observation ${i + 1}`} · {obs.observation_ts?.slice(0, 16).replace('T', ' ')}
            </div>
            <div style={{ fontWeight: 600, marginTop: '0.25rem' }}>{obs.condition_description}</div>
            {obs.source_class && (
              <div style={{ fontSize: '0.75rem', color: 'var(--muted)' }}>Source: {obs.source_class}</div>
            )}
            <div style={{ fontSize: '0.8rem', marginTop: '0.25rem', fontFamily: 'var(--mono)' }}>
              Severity: {(obs.severity_score ?? 0).toFixed(2)} · Change M: {(obs.change_magnitude ?? 0).toFixed(2)}
            </div>
            {obs.image_url && (
              <img src={obs.image_url} alt={obs.condition_description} />
            )}
          </div>
        ))}
      </div>

      <SeverityChart observations={observations} />
    </div>
  )
}
