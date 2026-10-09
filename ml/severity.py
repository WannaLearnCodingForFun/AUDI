"""Severity scoring, SRS §8.1 with the amendments in docs/SRS_AMENDMENTS.md.

S = (w1·M + w2·C)·Cf
  M  change magnitude ∈ [0,1]
  C  asset-type criticality ∈ [0,1]
  Cf OBSERVATION confidence = min(detector confidence, condition-classifier
     confidence) — never the identity-match score (D1).
Gate (D4): if M < change_detection.threshold the change is not significant,
S = 0 and the asset follows the FR-16 "confirmed" path.
"""

from __future__ import annotations

from typing import Dict, Tuple

from ml.config import cfg


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def observation_confidence(detector_confidence: float, classifier_confidence: float) -> float:
    return _clip01(min(detector_confidence, classifier_confidence))


def calculate_severity(change_magnitude: float, criticality: float, confidence: float, config: Dict) -> float:
    """Ungated §8.1 formula."""
    w1 = float(cfg(config, "severity", "w1"))
    w2 = float(cfg(config, "severity", "w2"))
    return _clip01((w1 * _clip01(change_magnitude) + w2 * _clip01(criticality)) * _clip01(confidence))


def gated_severity(
    change_magnitude: float, criticality: float, confidence: float, config: Dict
) -> Tuple[float, bool]:
    """Return (S, significant_change) applying the D4 M-gate."""
    threshold = float(cfg(config, "change_detection", "threshold"))
    if change_magnitude < threshold:
        return 0.0, False
    return calculate_severity(change_magnitude, criticality, confidence, config), True


def severity_level(severity: float, config: Dict) -> str:
    """Lowest level is the floor; every other level applies when S > its lower bound."""
    ordered = sorted(cfg(config, "severity", "levels").items(), key=lambda kv: float(kv[1]))
    level = ordered[0][0]
    for name, lower in ordered[1:]:
        if severity > float(lower):
            level = name
    return level


def is_priority(severity: float, config: Dict) -> bool:
    return severity > float(cfg(config, "severity", "priority_threshold"))


def is_critical(severity: float, config: Dict) -> bool:
    return severity > float(cfg(config, "severity", "levels", "critical"))


def asset_criticality(asset_type: str, config: Dict) -> float:
    table = cfg(config, "asset_criticality")
    return float(table.get(asset_type, table["default"]))
