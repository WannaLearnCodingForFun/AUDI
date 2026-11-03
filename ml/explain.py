"""Explainable alerts (upgrade 5.3).

Every processed observation carries an `explanation`: before/after images, the change mask
and heatmap, the measured growth, the M / C / Cf terms of S = (w1·M + w2·C)·Cf with their
values, and plain-language reasons for crossing (or not crossing) each threshold.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ml.config import cfg


def explain(
    config: Dict[str, Any],
    *,
    change: Dict[str, Any],
    criticality: float,
    confidence: float,
    detector_confidence: float,
    classifier_confidence: float,
    severity: float,
    level: str,
    significant: bool,
    before_image: Optional[str],
    after_image: Optional[str],
    heatmap: Optional[str],
    mask: Optional[str],
    condition: Dict[str, Any],
    reference_label: Optional[str],
    instances: Optional[List[Dict[str, Any]]] = None,
    growth_text: Optional[str] = None,
) -> Dict[str, Any]:
    w1 = float(cfg(config, "severity", "w1"))
    w2 = float(cfg(config, "severity", "w2"))
    thr = float(cfg(config, "change_detection", "threshold"))
    alpha = float(cfg(config, "change_detection", "alpha"))
    levels = dict(cfg(config, "severity", "levels"))
    M = float(change.get("change_magnitude") or 0.0)
    a = change.get("condition_delta")
    b = change.get("visual_delta")
    reasons: List[str] = []
    if reference_label is not None and condition.get("label") != reference_label:
        reasons.append(f"condition changed from '{reference_label}' to '{condition.get('label')}' "
                       f"(classifier confidence {classifier_confidence:.2f})")
    if b is not None:
        reasons.append(f"visual change b = {b:.2f} from aligned patch comparison")
    cm = change.get("mask") or {}
    if cm.get("changed_area_cm2"):
        reasons.append(f"changed area ≈ {cm['changed_area_cm2']:.1f} cm² ({cm.get('change_type') or 'unknown'} type)")
    elif cm.get("changed_fraction"):
        reasons.append(f"{100 * cm['changed_fraction']:.1f}% of the asset view changed ({cm.get('change_type') or 'unknown'} type)")
    if growth_text:
        reasons.append(growth_text)
    if significant:
        reasons.append(f"M = {M:.2f} ≥ change threshold {thr:.2f}, so the change is significant")
    else:
        reasons.append(f"M = {M:.2f} < change threshold {thr:.2f}: no significant change, S = 0 (asset confirmed)")
    crossed = [n for n, lo in sorted(levels.items(), key=lambda kv: kv[1]) if severity > float(lo)]
    if significant:
        reasons.append(f"S = ({w1}·{M:.2f} + {w2}·{criticality:.2f})·{confidence:.2f} = {severity:.2f} → level '{level}'"
                       + (f", above the {crossed[-1]} bound {levels[crossed[-1]]}" if crossed else ""))
    return {
        "before_image": before_image,
        "after_image": after_image,
        "heatmap": heatmap,
        "change_mask": mask,
        "terms": {
            "M": M, "M_condition_a": a, "M_visual_b": b, "alpha": alpha,
            "C": criticality, "Cf": confidence, "Cf_detector": detector_confidence, "Cf_classifier": classifier_confidence,
            "w1": w1, "w2": w2, "S": severity, "change_threshold": thr, "levels": levels,
        },
        "growth": growth_text,
        "defect_instances": [{k: i.get(k) for k in ("instance_id", "defect_type", "area_cm2", "length_cm", "area_norm", "predicted_critical_ts")}
                             for i in (instances or [])],
        "reasons": reasons,
        "summary": reasons[0] if reasons else "",
    }
