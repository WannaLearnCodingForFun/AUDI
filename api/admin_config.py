"""Admin-editable scoring configuration: view, validation, application.

Edits are stored as versioned snapshots in the `config_versions` table and
applied over config.yaml at startup (config.yaml itself is never rewritten,
so its comments survive).
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

WEIGHT_TOL = 1e-6
LEVEL_ORDER = ("low", "medium", "priority", "critical")


class ConfigValidationError(ValueError):
    """An admin config change failed validation (unknown key, range, or weight sum)."""


def editable_view(config: Dict[str, Any]) -> Dict[str, Any]:
    sev, rev, chg = config["severity"], config["revisit_priority"], config["change_detection"]
    return {
        "severity": {
            "w1": sev["w1"], "w2": sev["w2"], "priority_threshold": sev["priority_threshold"],
            "levels": dict(sev["levels"]),
        },
        "revisit_priority": {
            "w3": rev["w3"], "w4": rev["w4"], "w5": rev["w5"],
            "max_days_since_inspection": rev["max_days_since_inspection"], "trend_window": rev["trend_window"],
        },
        "change_detection": {"threshold": chg["threshold"], "alpha": chg["alpha"]},
        "asset_criticality": dict(config["asset_criticality"]),
        "detection_prompts": {k: list(v) for k, v in config["detection"]["prompts"].items()},
        "resolution_suggestion_missions": config["defects"]["resolution_suggestion_missions"],
    }


def _deep_merge(base: Dict[str, Any], changes: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in changes.items():
        if k not in out:
            raise ConfigValidationError(f"Unknown setting: {k}")
        if isinstance(out[k], dict) and isinstance(v, dict) and k != "detection_prompts":
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _unit(name: str, v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(f"{name} must be a number") from exc
    if not 0.0 <= f <= 1.0:
        raise ConfigValidationError(f"{name} must be within [0, 1]")
    return f


def validate(view: Dict[str, Any], asset_types: List[str]) -> Dict[str, Any]:
    sev, rev, chg = view["severity"], view["revisit_priority"], view["change_detection"]
    for k in ("w1", "w2", "priority_threshold"):
        sev[k] = _unit(f"severity.{k}", sev[k])
    if abs(sev["w1"] + sev["w2"] - 1.0) > WEIGHT_TOL:
        raise ConfigValidationError("severity weights w1 + w2 must sum to 1")
    levels = sev["levels"]
    if set(levels) != set(LEVEL_ORDER):
        raise ConfigValidationError(f"severity.levels must define exactly {LEVEL_ORDER}")
    bounds = [_unit(f"severity.levels.{k}", levels[k]) for k in LEVEL_ORDER]
    if bounds[0] != 0.0 or any(b2 <= b1 for b1, b2 in zip(bounds, bounds[1:])):
        raise ConfigValidationError("severity levels must start at 0 and strictly increase low < medium < priority < critical")
    sev["levels"] = dict(zip(LEVEL_ORDER, bounds))

    for k in ("w3", "w4", "w5"):
        rev[k] = _unit(f"revisit_priority.{k}", rev[k])
    if abs(rev["w3"] + rev["w4"] + rev["w5"] - 1.0) > WEIGHT_TOL:
        raise ConfigValidationError("revisit weights w3 + w4 + w5 must sum to 1")
    if float(rev["max_days_since_inspection"]) <= 0:
        raise ConfigValidationError("max_days_since_inspection must be positive")
    if int(rev["trend_window"]) < 2:
        raise ConfigValidationError("trend_window must be at least 2 missions")
    rev["max_days_since_inspection"] = float(rev["max_days_since_inspection"])
    rev["trend_window"] = int(rev["trend_window"])

    chg["threshold"] = _unit("change_detection.threshold", chg["threshold"])
    chg["alpha"] = _unit("change_detection.alpha", chg["alpha"])

    crit = view["asset_criticality"]
    for k in list(crit):
        crit[k] = _unit(f"asset_criticality.{k}", crit[k])
    if "default" not in crit:
        raise ConfigValidationError("asset_criticality.default is required")

    prompts = view["detection_prompts"]
    unknown = set(prompts) - set(asset_types)
    if unknown:
        raise ConfigValidationError(f"detection prompts for unknown asset types: {sorted(unknown)}")
    for k, v in prompts.items():
        if not isinstance(v, list) or not v or not all(isinstance(p, str) and p.strip() for p in v):
            raise ConfigValidationError(f"detection_prompts.{k} must be a non-empty list of strings")
        prompts[k] = [p.strip() for p in v]
    if int(view["resolution_suggestion_missions"]) < 1:
        raise ConfigValidationError("resolution_suggestion_missions must be ≥ 1")
    view["resolution_suggestion_missions"] = int(view["resolution_suggestion_missions"])
    return view


def merge_and_validate(config: Dict[str, Any], changes: Dict[str, Any], asset_types: List[str]) -> Dict[str, Any]:
    return validate(_deep_merge(editable_view(config), changes), asset_types)


def apply_view(config: Dict[str, Any], view: Dict[str, Any]) -> None:
    """Write a validated view into the live config dict (in place, shared by all services)."""
    config["severity"].update({k: view["severity"][k] for k in ("w1", "w2", "priority_threshold")})
    config["severity"]["levels"] = dict(view["severity"]["levels"])
    config["revisit_priority"].update(view["revisit_priority"])
    config["change_detection"].update(view["change_detection"])
    config["asset_criticality"] = dict(view["asset_criticality"])
    config["detection"]["prompts"] = {k: list(v) for k, v in view["detection_prompts"].items()}
    config["defects"]["resolution_suggestion_missions"] = view["resolution_suggestion_missions"]
