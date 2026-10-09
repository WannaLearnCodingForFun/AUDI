import { useState } from 'react'
import { errorText, flushSync, getMetrics, getSync } from '../api'
import { Card, Empty, ErrorBox, fmt, Loading, Stat, when } from '../components/ui'
import { useAsync } from '../hooks'

function EvalTable({ ev }) {
  const types = Object.keys(ev.results.classifier.clean)
  return (
    <table className="table">
      <thead><tr><th>Asset type</th><th>Model</th><th>n</th><th>Acc clean</th><th>Macro-F1 clean</th><th>Acc webcam</th><th>Macro-F1 webcam</th></tr></thead>
      <tbody>
        {types.flatMap((t) => ['baseline', 'classifier'].map((m) => {
          const c = ev.results[m].clean[t]
          const w = ev.results[m].webcam[t]
          return (
            <tr key={`${t}-${m}`} className={m === 'classifier' ? '' : 'muted'}>
              <td>{t}</td><td>{m}</td><td>{c.n}</td><td>{fmt(c.accuracy, 3)}</td><td>{fmt(c.macro_f1, 3)}</td><td>{fmt(w.accuracy, 3)}</td><td>{fmt(w.macro_f1, 3)}</td>
            </tr>
          )
        }))}
      </tbody>
    </table>
  )
}

export default function AdminMetrics() {
  const m = useAsync(getMetrics, [])
  const sync = useAsync(getSync, [])
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)

  const flush = async () => {
    setBusy(true)
    try {
      await flushSync()
      sync.reload()
    } catch (e) { setError(errorText(e)) } finally { setBusy(false) }
  }

  const s = sync.data
  const ev = m.data?.evaluation
  const tr = m.data?.training
  return (
    <div className="page">
      <header className="page-head"><h2>Metrics &amp; sync</h2></header>
      <ErrorBox error={error || m.error || sync.error} />

      <Card title="Cloud sync (FR-6, FR-7)" actions={<button className="btn" onClick={flush} disabled={busy}>{busy ? 'Syncing…' : 'Sync now'}</button>}>
        {s ? (
          <div className="stats">
            <Stat label="Cloud" value={s.online ? 'online' : 'offline'} tone={s.online ? '' : 'warn'} />
            <Stat label="Outbox pending" value={s.outbox?.pending ?? 0} tone={s.outbox?.pending ? 'warn' : ''} />
            <Stat label="Outbox sent" value={s.outbox?.sent ?? 0} />
            <Stat label="Last sync" value={when(s.last_sync_ts)} />
          </div>
        ) : <Loading />}
      </Card>

      {m.loading ? <Loading /> : (
        <>
          <Card title="Condition classifier — test split (from reports/eval)">
            {ev ? (
              <>
                <div className="stats">
                  <Stat label="Baseline accuracy (clean)" value={fmt(ev.summary.baseline_clean.accuracy, 3)} />
                  <Stat label="Classifier accuracy (clean)" value={fmt(ev.summary.classifier_clean.accuracy, 3)} />
                  <Stat label="Baseline macro-F1 (webcam)" value={fmt(ev.summary.baseline_webcam.mean_macro_f1, 3)} />
                  <Stat label="Classifier macro-F1 (webcam)" value={fmt(ev.summary.classifier_webcam.mean_macro_f1, 3)} />
                </div>
                <EvalTable ev={ev} />
              </>
            ) : <Empty>No evaluation yet — run <code>python training/evaluate.py</code>.</Empty>}
          </Card>
          <div className="grid two">
            <Card title="Training">
              {tr ? (
                <dl className="kv">
                  <dt>Trained</dt><dd>{when(tr.trained_at)}</dd>
                  <dt>Validation macro-F1</dt><dd>{fmt(tr.val_macro_f1, 4)} (best epoch {tr.best_epoch})</dd>
                  <dt>Per type</dt><dd className="small">{Object.entries(tr.val_per_type_macro_f1 || {}).map(([k, v]) => `${k} ${fmt(v, 3)}`).join(' · ')}</dd>
                  <dt>Fine-tuned blocks</dt><dd>{tr.fine_tune_blocks}</dd>
                </dl>
              ) : <Empty>No training metrics.</Empty>}
            </Card>
            <Card title="Benchmarks (reports/bench.md)">
              {m.data?.bench_md ? <pre className="pre small">{m.data.bench_md}</pre> : <Empty>Run <code>python scripts/bench.py</code>.</Empty>}
            </Card>
          </div>
          <Card title="Dataset (dataset/processed/STATS.md)">
            {m.data?.dataset_stats_md ? <pre className="pre small">{m.data.dataset_stats_md}</pre> : <Empty>Run <code>make data</code>.</Empty>}
          </Card>
        </>
      )}
    </div>
  )
}
