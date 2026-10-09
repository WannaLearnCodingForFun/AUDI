import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

/** points: [{ ts, severity }] ; thresholds: { priority, critical } */
export default function SeverityChart({ points, thresholds, height = 220 }) {
  const data = points.map((p) => ({ ...p, label: new Date(p.ts).toLocaleDateString() }))
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
        <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
        <XAxis dataKey="label" stroke="var(--muted)" fontSize={11} />
        <YAxis domain={[0, 1]} stroke="var(--muted)" fontSize={11} />
        <Tooltip contentStyle={{ background: 'var(--surface2)', border: '1px solid var(--border)' }} />
        {thresholds?.priority !== undefined && <ReferenceLine y={thresholds.priority} stroke="var(--warning)" strokeDasharray="4 4" />}
        {thresholds?.critical !== undefined && <ReferenceLine y={thresholds.critical} stroke="var(--danger)" strokeDasharray="4 4" />}
        <Line type="monotone" dataKey="severity" stroke="var(--accent2)" strokeWidth={2} dot={{ r: 3 }} />
      </LineChart>
    </ResponsiveContainer>
  )
}
