import { imageUrl, sendFeedback } from '../api'
import { fmt } from './ui'
import { useState } from 'react'

/** 5.3 explainable alert: before / after / mask, the S = (w1·M + w2·C)·Cf terms, reasons; 5.4 feedback buttons. */
export default function Explanation({ event, compact = false }) {
  const x = event?.explanation
  const [sent, setSent] = useState(null)
  if (!x) return null
  const t = x.terms || {}
  const give = async (label) => {
    try { await sendFeedback(event.observation_id, label); setSent(label) } catch (e) { setSent(`error: ${e?.response?.data?.detail || e.message}`) }
  }
  return (
    <div className="explain">
      {!compact && (
        <div className="explain-images">
          {x.before_image && <figure><img src={imageUrl(x.before_image)} alt="Before" /><figcaption>before</figcaption></figure>}
          {x.after_image && <figure><img src={imageUrl(x.after_image)} alt="After" /><figcaption>after</figcaption></figure>}
          {x.change_mask && <figure><img src={imageUrl(x.change_mask)} alt="Change mask" /><figcaption>change mask</figcaption></figure>}
          {!x.change_mask && x.heatmap && <figure><img src={imageUrl(x.heatmap)} alt="Change heatmap" /><figcaption>heatmap</figcaption></figure>}
        </div>
      )}
      <div className="small mono">
        S = ({fmt(t.w1, 1)}·M {fmt(t.M)} + {fmt(t.w2, 1)}·C {fmt(t.C)}) · Cf {fmt(t.Cf)} = <strong>{fmt(t.S)}</strong>
        <span className="muted"> · M = α·a {fmt(t.M_condition_a)} + (1−α)·b {fmt(t.M_visual_b)} · gate {fmt(t.change_threshold)}</span>
      </div>
      <ul className="small reasons">{(x.reasons || []).map((r, i) => <li key={i}>{r}</li>)}</ul>
      {event.observation_id && (
        <div className="row gap small">
          {sent ? <span className="muted">feedback: {sent}</span> : (
            <>
              <span className="muted">Was this right?</span>
              <button className="btn small" onClick={() => give('confirmed')}>Confirmed</button>
              <button className="btn small ghost" onClick={() => give('false_alarm')}>False alarm</button>
              <button className="btn small ghost" onClick={() => give('wrong_asset')}>Wrong asset</button>
            </>
          )}
        </div>
      )}
    </div>
  )
}
