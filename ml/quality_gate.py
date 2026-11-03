"""Image quality gate (upgrade 3.4): reject bad input before it reaches memory.

Frame level (cheap, on a downscaled grey image):
  blur       variance of the Laplacian < min_laplacian_var
  exposure   mean luminance outside [min_mean, max_mean], or more than max_clipped_fraction
             of pixels crushed to black / blown to white
Detection level (when metric scale / pose are known):
  distance   estimated range outside [min_distance_m, max_distance_m]
  angle      incidence angle between the view ray and the asset surface normal > max_incidence_deg
  size       crop smaller than min_crop_px on its short side
A rejected frame or detection is counted and reported, never stored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
from PIL import Image

from ml.config import get_nested


@dataclass
class QualityReport:
    ok: bool
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "reasons": list(self.reasons), "metrics": dict(self.metrics)}


class QualityGate:
    def __init__(self, config: Dict[str, Any]):
        c = get_nested(config, "quality_gate", default={}) or {}
        self.enabled = bool(c.get("enabled", True))
        self.side = int(c.get("analysis_side", 256))
        self.min_lap = float(c.get("min_laplacian_var", 15.0))
        self.min_mean = float(c.get("min_mean", 20.0))
        self.max_mean = float(c.get("max_mean", 235.0))
        self.max_clipped = float(c.get("max_clipped_fraction", 0.4))
        self.min_dist = float(c.get("min_distance_m", 0.3))
        self.max_dist = float(c.get("max_distance_m", 60.0))
        self.max_incidence = float(c.get("max_incidence_deg", 70.0))
        self.min_crop = int(c.get("min_crop_px", 24))
        self.flat_types = set(c.get("flat_asset_types", ["solar_panel"]))
        self.counters: Dict[str, int] = {}

    def _count(self, reasons: Sequence[str]) -> None:
        for r in reasons:
            self.counters[r] = self.counters.get(r, 0) + 1
        self.counters["checked"] = self.counters.get("checked", 0) + 1
        if reasons:
            self.counters["rejected"] = self.counters.get("rejected", 0) + 1

    def image_metrics(self, img: Image.Image) -> Dict[str, float]:
        g = np.asarray(img.convert("L"))
        h, w = g.shape
        s = min(1.0, self.side / float(max(h, w)))
        if s < 1.0:
            g = cv2.resize(g, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        return {
            "laplacian_var": float(cv2.Laplacian(g, cv2.CV_64F).var()),
            "mean": float(g.mean()),
            "clipped_fraction": float(((g <= 3) | (g >= 252)).mean()),
        }

    def check_frame(self, img: Image.Image) -> QualityReport:
        if not self.enabled:
            return QualityReport(True)
        m = self.image_metrics(img)
        reasons = []
        if m["laplacian_var"] < self.min_lap:
            reasons.append("blur")
        if m["mean"] < self.min_mean:
            reasons.append("underexposed")
        elif m["mean"] > self.max_mean:
            reasons.append("overexposed")
        if m["clipped_fraction"] > self.max_clipped:
            reasons.append("clipped")
        self._count(reasons)
        return QualityReport(not reasons, reasons, m)

    def check_detection(
        self,
        crop: Image.Image,
        asset_type: str,
        distance_m: Optional[float] = None,
        view_ray_pitch_deg: Optional[float] = None,
    ) -> QualityReport:
        """`view_ray_pitch_deg`: depression angle of the ray to the asset (90 = straight down)."""
        if not self.enabled:
            return QualityReport(True)
        reasons, m = [], {}
        if min(crop.size) < self.min_crop:
            reasons.append("too_small")
        if distance_m is not None:
            m["distance_m"] = float(distance_m)
            if distance_m < self.min_dist:
                reasons.append("too_close")
            elif distance_m > self.max_dist:
                reasons.append("too_far")
        if view_ray_pitch_deg is not None:
            # flat assets face up (normal = vertical); upright ones face sideways (normal horizontal)
            incidence = 90.0 - view_ray_pitch_deg if asset_type in self.flat_types else abs(view_ray_pitch_deg)
            m["incidence_deg"] = float(incidence)
            if incidence > self.max_incidence:
                reasons.append("steep_angle")
        self._count(reasons)
        return QualityReport(not reasons, reasons, m)


def ray_depression_deg(ray_z: float) -> float:
    return math.degrees(math.asin(max(-1.0, min(1.0, -ray_z))))
