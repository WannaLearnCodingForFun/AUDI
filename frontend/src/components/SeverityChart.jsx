import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

export default function SeverityChart({ observations }) {
  if (!observations?.length) return null

  const data = observations.map((o, i) => ({
    name: o.mission_id || `M${i + 1}`,
    severity: o.severity_score ?? 0,
    change: o.change_magnitude ?? 0,
  }))

  return (
    <div style={{ marginTop: '1.5rem' }}>
      <h4 style={{ marginBottom: '0.75rem', color: 'var(--muted)', fontSize: '0.8rem', textTransform: 'uppercase' }}>
        Severity Trend
      </h4>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2a3548" />
          <XAxis dataKey="name" stroke="#94a3b8" fontSize={12} />
          <YAxis domain={[0, 1]} stroke="#94a3b8" fontSize={12} />
          <Tooltip contentStyle={{ background: '#111827', border: '1px solid #2a3548' }} />
          <Line type="monotone" dataKey="severity" stroke="#3b82f6" strokeWidth={2} dot={{ r: 4 }} name="Severity" />
          <Line type="monotone" dataKey="change" stroke="#06b6d4" strokeWidth={2} dot={{ r: 4 }} name="Change M" strokeDasharray="4 4" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
