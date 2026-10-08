"""Revisit priority scoring per AUDI SRS specification."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

from ml.config import get_nested, load_config


def _parse_ts(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def calculate_trend(severity_scores: Sequence[float]) -> float:
    """
    Estimate historical severity trend rate T in [0, 1].

    Uses simple linear slope normalized by number of observations.
    """
    if len(severity_scores) < 2:
        return 0.0

    scores = list(severity_scores)
    n = len(scores)
    x_mean = (n - 1) / 2.0
    y_mean = sum(scores) / n
    numerator = sum((i - x_mean) * (scores[i] - y_mean) for i in range(n))
    denominator = sum((i - x_mean) ** 2 for i in range(n)) or 1.0
    slope = numerator / denominator
    return float(max(0.0, min(1.0, slope * n)))


def days_since_last_inspection(
    observation_timestamps: Sequence[str],
    current_timestamp: Optional[str] = None,
    max_days: float = 30.0,
) -> float:
    """
    Compute normalized time since last inspection D in [0, 1].
    """
    if not observation_timestamps:
        return 0.0

    now = _parse_ts(current_timestamp) if current_timestamp else datetime.now(timezone.utc)
    last_ts = _parse_ts(observation_timestamps[-1])
    delta_days = max(0.0, (now - last_ts).total_seconds() / 86400.0)
    return float(min(1.0, delta_days / max(max_days, 1e-6)))


def calculate_revisit_priority(
    recent_severity: float,
    trend_rate: float,
    days_since_inspection: float,
    w3: float | None = None,
    w4: float | None = None,
    w5: float | None = None,
    config: Dict | None = None,
) -> float:
    """
    Compute revisit priority R = w3 * Sr + w4 * T + w5 * D.

    All components normalized to [0, 1].
    """
    cfg = config or load_config()
    if w3 is None:
        w3 = float(get_nested(cfg, "revisit_priority", "w3", default=0.5))
    if w4 is None:
        w4 = float(get_nested(cfg, "revisit_priority", "w4", default=0.3))
    if w5 is None:
        w5 = float(get_nested(cfg, "revisit_priority", "w5", default=0.2))

    sr = max(0.0, min(1.0, recent_severity))
    t = max(0.0, min(1.0, trend_rate))
    d = max(0.0, min(1.0, days_since_inspection))

    priority = w3 * sr + w4 * t + w5 * d
    return float(max(0.0, min(1.0, priority)))


def compute_revisit_from_history(
    severity_history: List[float],
    observation_timestamps: List[str],
    current_timestamp: Optional[str] = None,
    config: Dict | None = None,
) -> Dict[str, float]:
    """Compute revisit priority components from historical severity sequence."""
    cfg = config or load_config()
    max_days = float(get_nested(cfg, "revisit_priority", "max_days_since_inspection", default=30.0))

    recent_severity = severity_history[-1] if severity_history else 0.0
    trend = calculate_trend(severity_history)
    days_norm = days_since_last_inspection(
        observation_timestamps[:-1] if len(observation_timestamps) > 1 else observation_timestamps,
        current_timestamp,
        max_days,
    )
    priority = calculate_revisit_priority(recent_severity, trend, days_norm, config=cfg)

    return {
        "recent_severity": recent_severity,
        "trend": trend,
        "days_since_inspection": days_norm,
        "revisit_priority": priority,
    }
