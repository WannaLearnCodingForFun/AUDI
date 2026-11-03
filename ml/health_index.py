"""Asset health index, 0 (failed) – 100 (as new) (upgrade 5.5).

H = 100 · (1 − clip(w_s·S + w_d·D + w_g·G + w_t·T, 0, 1))
  S  latest severity of the asset ∈ [0, 1]
  D  open-defect load: Σ over open defects of their severity, divided by defects_cap, clipped
  G  growth: forecast urgency of the fastest-deteriorating defect instance (2.3), else the
     severity trend component
  T  time since inspection: the revisit-priority D term (days / max_days, clipped)
Site roll-up: criticality-weighted mean of its assets' indices.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

from ml.config import get_nested


def health_index(
    latest_severity: float,
    open_defect_severities: Sequence[float],
    growth: Optional[float],
    staleness: float,
    config: Dict[str, Any],
) -> float:
    c = get_nested(config, "health_index", default={}) or {}
    ws, wd, wg, wt = (float(c.get(k, d)) for k, d in (("w_severity", 0.4), ("w_defects", 0.25), ("w_growth", 0.25), ("w_staleness", 0.1)))
    cap = float(c.get("defects_cap", 3.0))
    d = min(1.0, sum(max(0.0, float(x)) for x in open_defect_severities) / max(cap, 1e-9)) if open_defect_severities else 0.0
    pen = ws * max(0.0, min(1.0, latest_severity)) + wd * d + wg * max(0.0, min(1.0, growth or 0.0)) + wt * max(0.0, min(1.0, staleness))
    return round(100.0 * (1.0 - max(0.0, min(1.0, pen))), 1)


def site_health(assets: Iterable[Dict[str, Any]]) -> Optional[float]:
    num = den = 0.0
    for a in assets:
        if a.get("health_index") is None:
            continue
        w = float(a.get("criticality") or 0.5)
        num += w * float(a["health_index"])
        den += w
    return round(num / den, 1) if den else None


def health_band(h: Optional[float]) -> str:
    if h is None:
        return "unknown"
    if h >= 80:
        return "good"
    if h >= 60:
        return "fair"
    if h >= 40:
        return "poor"
    return "critical"
