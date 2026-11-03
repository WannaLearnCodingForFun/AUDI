"""Deterioration forecast per defect instance (upgrade 2.3).

Interpretable, deterministic least-squares growth models, one per defect family:

  crack      a(t) = a0 · (1 + k·t)^m          (Paris-law style power growth; k ≥ 0, m ∈ [m_min, m_max])
  corrosion  a(t) = a0 + r·t                  (area grows linearly with time)
  soiling    a(t) = a_reset + r·(t − t_reset)  (linear accumulation since the last reset; a drop of
                                              more than `reset_drop` of the previous value — rain or
                                              cleaning — starts a new segment)

t is in days since the first measurement. The fitted curve is solved for the configured
critical size, giving the predicted date the defect becomes critical. Revisit priority uses
that date (ml/revisit_priority.py) in place of the slope of past severity scores whenever a
forecast exists.

No randomness and no Bayesian machinery: the crack fit is a bounded grid search over m with a
closed-form linear solve for (a0, a0·k-ish) refined by scipy least_squares from a fixed start,
so the same series always yields the same forecast.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ml.config import get_nested
from ml.revisit_priority import parse_ts

CRACK_TYPES = {"crack", "cracks", "physical_damage", "microcrack", "broken", "spalling"}
CORROSION_TYPES = {"rust", "corrosion", "corroded", "pitted", "scratches", "oxidation"}
SOILING_TYPES = {"dust", "soiling", "dirty", "bird_drop", "bird_droppings", "snow", "dusty", "snow_covered"}


def family(defect_type: Optional[str]) -> str:
    t = (defect_type or "").lower()
    if t in CRACK_TYPES or "crack" in t:
        return "crack"
    if t in CORROSION_TYPES or "rust" in t or "corro" in t:
        return "corrosion"
    if t in SOILING_TYPES or "dust" in t or "soil" in t or "bird" in t or "snow" in t:
        return "soiling"
    return "corrosion"   # unknown defects: the conservative linear model


@dataclass
class Forecast:
    family: str
    model: str
    metric: str
    params: Dict[str, float]
    rmse: float
    n_points: int
    current: float
    critical: float
    days_to_critical: Optional[float]       # from the last measurement; 0 when already critical
    predicted_critical_ts: Optional[str]
    growth_since_last_pct: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _days(ts: Sequence[str]) -> np.ndarray:
    t0 = parse_ts(ts[0])
    return np.array([(parse_ts(t) - t0).total_seconds() / 86400.0 for t in ts], dtype=np.float64)


def fit_linear(t: np.ndarray, a: np.ndarray) -> Tuple[float, float]:
    """(a0, r) least squares; a single point gives r = 0."""
    if len(t) < 2 or np.ptp(t) < 1e-9:
        return float(a[-1]), 0.0
    A = np.stack([np.ones_like(t), t], axis=1)
    (a0, r), *_ = np.linalg.lstsq(A, a, rcond=None)
    return float(a0), float(r)


def power_curve(t: np.ndarray, a0: float, k: float, m: float) -> np.ndarray:
    return a0 * np.power(1.0 + k * np.maximum(t, 0.0), m)


def fit_power(t: np.ndarray, a: np.ndarray, m_bounds: Tuple[float, float]) -> Tuple[float, float, float]:
    """(a0, k, m) of a = a0·(1 + k·t)^m by bounded least squares from a deterministic start."""
    a = np.maximum(a, 1e-9)
    if len(t) < 3:
        # two points cannot identify both k and m: fix m = 1 (linear relative growth)
        a0, r = float(a[0]), 0.0
        if len(t) == 2 and t[1] > t[0]:
            r = max(0.0, (a[1] / a[0] - 1.0) / (t[1] - t[0]))
        return a0, r, 1.0
    from scipy.optimize import least_squares

    best = None
    for m0 in np.linspace(m_bounds[0], m_bounds[1], 8):
        # start: k from the end points for this m
        ratio = max(a[-1] / a[0], 1.0)
        k0 = max(1e-6, (ratio ** (1.0 / m0) - 1.0) / max(t[-1], 1e-9))
        x0 = np.array([a[0], k0, m0])
        res = least_squares(
            lambda x: power_curve(t, x[0], x[1], x[2]) - a,
            x0,
            bounds=([1e-12, 0.0, m_bounds[0]], [np.inf, np.inf, m_bounds[1]]),
            method="trf",
        )
        cost = float(np.sum(res.fun ** 2))
        if best is None or cost < best[0] - 1e-15:
            best = (cost, res.x)
    a0, k, m = best[1]
    return float(a0), float(k), float(m)


def _soiling_segment(t: np.ndarray, a: np.ndarray, reset_drop: float, resets: Sequence[float]) -> int:
    """Index of the first point after the last reset (detected drop or supplied reset time)."""
    start = 0
    for i in range(1, len(a)):
        if a[i] < a[i - 1] * (1.0 - reset_drop):
            start = i
    for r in resets:
        idx = int(np.searchsorted(t, r, side="left"))
        if idx < len(t):
            start = max(start, idx)
    return start


def choose_metric(measurements: Sequence[Dict[str, Any]], fam: str) -> str:
    if fam == "crack" and all(m.get("length_cm") for m in measurements):
        return "length_cm"
    if fam != "soiling" and all(m.get("area_cm2") for m in measurements):
        return "area_cm2"
    return "area_norm"


def forecast_series(
    measurements: Sequence[Dict[str, Any]],
    defect_type: Optional[str],
    config: Dict[str, Any],
    resets_ts: Sequence[str] = (),
) -> Optional[Forecast]:
    """Fit the family's growth model to a time-ordered measurement series."""
    if not measurements:
        return None
    fc = get_nested(config, "forecast", default={}) or {}
    fam = family(defect_type)
    ms = sorted(measurements, key=lambda m: str(m["ts"]))
    metric = choose_metric(ms, fam)
    ts = [m["ts"] for m in ms]
    t = _days(ts)
    a = np.array([float(m.get(metric) or 0.0) for m in ms], dtype=np.float64)
    critical = float(get_nested(fc, "critical", fam, metric, default=get_nested(fc, "critical", "default", metric, default=1.0)))
    horizon = float(fc.get("max_horizon_days", 3650.0))

    params: Dict[str, float]
    if fam == "crack":
        a0, k, m = fit_power(t, a, (float(fc.get("crack_m_min", 0.5)), float(fc.get("crack_m_max", 4.0))))
        pred = power_curve(t, a0, k, m)
        params = {"a0": a0, "k": k, "m": m}
        model = "power"

        def solve(target: float) -> Optional[float]:
            if a0 >= target:
                return 0.0
            if k <= 0 or m <= 0:
                return None
            return ((target / a0) ** (1.0 / m) - 1.0) / k
    elif fam == "soiling":
        t_resets = [(parse_ts(r) - parse_ts(ts[0])).total_seconds() / 86400.0 for r in resets_ts]
        s = _soiling_segment(t, a, float(fc.get("soiling_reset_drop", 0.3)), t_resets)
        ts_seg, a_seg = t[s:], a[s:]
        a0, r = fit_linear(ts_seg - ts_seg[0], a_seg)
        pred = np.concatenate([a[:s], a0 + r * (ts_seg - ts_seg[0])])
        params = {"a0": a0, "rate_per_day": r, "segment_start_day": float(ts_seg[0])}
        model = "linear_reset"
        t_base = float(ts_seg[0])

        def solve(target: float) -> Optional[float]:
            if a0 >= target:
                return t_base
            if r <= 0:
                return None
            return t_base + (target - a0) / r
    else:
        a0, r = fit_linear(t, a)
        pred = a0 + r * t
        params = {"a0": a0, "rate_per_day": r}
        model = "linear"

        def solve(target: float) -> Optional[float]:
            if a0 >= target:
                return 0.0
            if r <= 0:
                return None
            return (target - a0) / r

    rmse = float(np.sqrt(np.mean((pred - a) ** 2)))
    current = float(a[-1])
    t_last = float(t[-1])
    if current >= critical:
        days = 0.0
    else:
        hit = solve(critical)
        days = None if hit is None else max(0.0, hit - t_last)
        if days is not None and days > horizon:
            days = None
    predicted = None
    if days is not None:
        predicted = (parse_ts(ts[-1]) + timedelta(days=days)).isoformat()
    growth = None
    if len(a) >= 2 and a[-2] > 0:
        growth = float((a[-1] - a[-2]) / a[-2] * 100.0)
    return Forecast(fam, model, metric, params, rmse, len(a), current, critical, days, predicted, growth)


def days_until(predicted_ts: Optional[str], now: Optional[datetime]) -> Optional[float]:
    if not predicted_ts:
        return None
    now = now or datetime.now(parse_ts(predicted_ts).tzinfo)
    return (parse_ts(predicted_ts) - now).total_seconds() / 86400.0


def holdout_error_days(
    measurements: Sequence[Dict[str, Any]], defect_type: Optional[str], config: Dict[str, Any], target: Optional[float] = None,
) -> Optional[float]:
    """Fit on all but the last visit, predict when the last visit's size is reached; error in days.

    Positive = forecast later than reality (optimistic), negative = earlier (conservative).
    """
    ms = sorted(measurements, key=lambda m: str(m["ts"]))
    if len(ms) < 3:
        return None
    fam = family(defect_type)
    metric = choose_metric(ms, fam)
    goal = float(target if target is not None else ms[-1].get(metric) or 0.0)
    cfg_copy = {"forecast": {**(get_nested(config, "forecast", default={}) or {}),
                             "critical": {fam: {metric: goal}}, "max_horizon_days": 1e9}}
    f = forecast_series(ms[:-1], defect_type, cfg_copy)
    if f is None or f.predicted_critical_ts is None:
        return None
    return (parse_ts(f.predicted_critical_ts) - parse_ts(ms[-1]["ts"])).total_seconds() / 86400.0


def forecast_urgency(days_to_critical: Optional[float], config: Dict[str, Any]) -> Optional[float]:
    """Map days-to-critical onto [0, 1]: 1 = critical now, 0 = beyond the urgency horizon."""
    if days_to_critical is None:
        return None
    h = float(get_nested(config, "forecast", "urgency_horizon_days", default=90.0))
    return float(max(0.0, min(1.0, 1.0 - max(0.0, days_to_critical) / h)))


def growth_summary(f: Optional[Forecast]) -> Optional[str]:
    """Human sentence: 'crack grew 40% since last visit; critical in about 34 days'."""
    if f is None:
        return None
    parts: List[str] = []
    if f.growth_since_last_pct is not None:
        parts.append(f"{f.family} grew {f.growth_since_last_pct:.0f}% since last visit")
    if f.days_to_critical is not None:
        parts.append("critical now" if f.days_to_critical <= 0 else f"critical in about {math.ceil(f.days_to_critical)} days")
    return "; ".join(parts) or None
