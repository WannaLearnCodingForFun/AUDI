export default function SummaryCard({ label, value, color }) {
  return (
    <div className="card">
      <div className="card-label">{label}</div>
      <div className="card-value" style={color ? { color } : {}}>{value ?? '—'}</div>
    </div>
  )
}
