"""In-flight inspection guidance (upgrade 2.4).

When a high-priority asset is in view the overlay tells the operator where to move, e.g.
"B-14: move 3 m closer, tilt camera 15° down". Computed from the required resolution for the
asset type (guidance.required_px_per_cm) and the current pose / metric scale:

  current px/cm   = 1 / (100 · m_per_px)            (ml/camera_model.py scale)
  needed range    = range · current_px_per_cm / required_px_per_cm
  move closer by  = range − needed range             (only when resolution is short)
  tilt / pan      = angle of the bbox centre from the optical axis, so the asset is centred

Advisory only: AUDI never commands the drone (SRS).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ml.camera_model import CameraIntrinsics, MetricScale
from ml.config import get_nested


@dataclass
class Guidance:
    asset_label: str
    message: str
    resolution_ok: bool
    current_px_per_cm: Optional[float]
    required_px_per_cm: float
    move_closer_m: float = 0.0
    tilt_deg: float = 0.0        # + = tilt camera down
    pan_deg: float = 0.0         # + = turn right
    actions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def short_label(asset_id: Optional[str], marker_id: Optional[int] = None, name: Optional[str] = None) -> str:
    if name:
        return name
    if marker_id is not None:
        return f"M-{marker_id}"
    return (asset_id or "new")[-6:]


def compute_guidance(
    config: Dict[str, Any],
    asset_type: str,
    bbox: Sequence[float],
    intr: CameraIntrinsics,
    scale: Optional[MetricScale],
    label: str,
) -> Guidance:
    g = get_nested(config, "guidance", default={}) or {}
    req_table = g.get("required_px_per_cm", {}) or {}
    required = float(req_table.get(asset_type, req_table.get("default", 2.0)))
    tol_deg = float(g.get("center_tolerance_deg", 5.0))
    min_move = float(g.get("min_move_m", 0.5))
    min_range = float(g.get("min_standoff_m", 1.5))

    actions: List[str] = []
    current = None
    move = 0.0
    ok = True
    if scale is not None and scale.m_per_px > 0:
        current = 1.0 / (100.0 * scale.m_per_px)
        if current < required:
            ok = False
            needed_range = max(min_range, scale.distance_m * current / required)
            move = max(0.0, scale.distance_m - needed_range)
            if move >= min_move:
                actions.append(f"move {move:.0f} m closer" if move >= 1.5 else f"move {move:.1f} m closer")
            else:
                actions.append("zoom in")
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    tilt = math.degrees(math.atan((cy - intr.cy) / intr.fy))
    pan = math.degrees(math.atan((cx - intr.cx) / intr.fx))
    if abs(tilt) > tol_deg:
        actions.append(f"tilt camera {abs(tilt):.0f}° {'down' if tilt > 0 else 'up'}")
    if abs(pan) > tol_deg:
        actions.append(f"turn {abs(pan):.0f}° {'right' if pan > 0 else 'left'}")
    if not actions:
        actions.append("hold — required resolution reached")
    return Guidance(label, f"{label}: " + ", ".join(actions), ok, current, required, move,
                    tilt if abs(tilt) > tol_deg else 0.0, pan if abs(pan) > tol_deg else 0.0, actions)
