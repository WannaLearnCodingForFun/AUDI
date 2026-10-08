"""Severity scoring per AUDI SRS specification."""

from __future__ import annotations

from typing import Dict

from ml.config import get_nested, load_config


def calculate_severity(
    change_magnitude: float,
    asset_criticality: float,
    observation_confidence: float,
    w1: float | None = None,
    w2: float | None = None,
    config: Dict | None = None,
) -> float:
    """
    Compute severity score S = (w1 * M + w2 * C) * Cf.

    Args:
        change_magnitude: M, normalized change in [0, 1]
        asset_criticality: C, asset importance in [0, 1]
        observation_confidence: Cf, match/observation confidence in [0, 1]
        w1, w2: configurable weights (default from config.yaml)

    Returns:
        Severity score in [0, 1]
    """
    cfg = config or load_config()
    if w1 is None:
        w1 = float(get_nested(cfg, "severity", "w1", default=0.6))
    if w2 is None:
        w2 = float(get_nested(cfg, "severity", "w2", default=0.4))

    m = max(0.0, min(1.0, change_magnitude))
    c = max(0.0, min(1.0, asset_criticality))
    cf = max(0.0, min(1.0, observation_confidence))

    severity = (w1 * m + w2 * c) * cf
    return float(max(0.0, min(1.0, severity)))


def is_priority(severity: float, config: Dict | None = None) -> bool:
    """Return True when severity exceeds configured priority threshold."""
    cfg = config or load_config()
    threshold = float(get_nested(cfg, "severity", "priority_threshold", default=0.5))
    return severity > threshold
