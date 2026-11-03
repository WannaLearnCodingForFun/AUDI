"""Thermal + RGB fusion (upgrade 3.1).

Inputs: a thermal frame as °C (radiometric, e.g. FLIR Lepton 3.5 in TLinear mode) or as 8-bit
with a known linear scale. Steps:
  register  thermal → RGB homography: from the rig calibration if given, refined by ECC
            (MOTION_HOMOGRAPHY) on gradient-magnitude images (edges survive the modality gap)
  hotspots  inside the asset box: cells hotter than the asset's median by ≥ hotspot_delta_c
            (IEC TS 62446-3 classes: ΔT ≥ 10 K is a serious module fault)
  fuse      per-asset feature = [RGB embedding, thermal statistics] for the fused condition head
            trained by training/train_thermal.py on marcosgabriel/infrared-solar-modules
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from ml.config import get_nested


@dataclass
class ThermalFrame:
    celsius: np.ndarray                     # float32 (H, W)

    @classmethod
    def from_8bit(cls, img: np.ndarray, t_min: float, t_max: float) -> "ThermalFrame":
        g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cls((g.astype(np.float32) / 255.0) * (t_max - t_min) + t_min)

    @classmethod
    def from_centikelvin(cls, raw: np.ndarray) -> "ThermalFrame":
        return cls(raw.astype(np.float32) / 100.0 - 273.15)


def _edges(x: np.ndarray) -> np.ndarray:
    x = cv2.GaussianBlur(x.astype(np.float32), (5, 5), 0)
    g = cv2.magnitude(cv2.Sobel(x, cv2.CV_32F, 1, 0), cv2.Sobel(x, cv2.CV_32F, 0, 1))
    return cv2.normalize(g, None, 0, 1, cv2.NORM_MINMAX)


def register(thermal: ThermalFrame, rgb: np.ndarray, H0: Optional[np.ndarray] = None, iterations: int = 200) -> Optional[np.ndarray]:
    """Homography mapping thermal pixels → RGB pixels (None when ECC does not converge).

    ECC aligns gradient-magnitude images (edges survive the modality gap). `H0` is the rig
    calibration (thermal → RGB) used as the starting point; without it the views are assumed
    roughly co-boresighted and only rescaled."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    h, w = gray.shape
    th, tw = thermal.celsius.shape
    S = np.diag([w / float(tw), h / float(th), 1.0])            # thermal px → resized-thermal px
    t = cv2.resize(thermal.celsius, (w, h), interpolation=cv2.INTER_LINEAR)
    # ECC warp W maps template (RGB) coords → input (resized thermal) coords
    W0 = np.eye(3) if H0 is None else S @ np.linalg.inv(H0)
    W0 = (W0 / W0[2, 2]).astype(np.float32)
    try:
        _, W = cv2.findTransformECC(_edges(gray), _edges(t), W0, cv2.MOTION_HOMOGRAPHY,
                                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, 1e-6), None, 5)
    except cv2.error:
        return None
    H = np.linalg.inv(W.astype(np.float64)) @ S
    return H / H[2, 2]


def warp_to_rgb(thermal: ThermalFrame, H: np.ndarray, size) -> np.ndarray:
    return cv2.warpPerspective(thermal.celsius, H, size, flags=cv2.INTER_LINEAR, borderValue=float("nan"))


@dataclass
class HotspotReport:
    median_c: float
    max_c: float
    delta_c: float
    hotspots: List[Dict[str, Any]] = field(default_factory=list)
    severity: str = "none"                  # none | minor | serious | critical

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def hotspots(celsius_crop: np.ndarray, config: Dict[str, Any]) -> HotspotReport:
    c = get_nested(config, "thermal", default={}) or {}
    thr = float(c.get("hotspot_delta_c", 10.0))
    minor = float(c.get("minor_delta_c", 3.0))
    crit = float(c.get("critical_delta_c", 20.0))
    t = np.nan_to_num(celsius_crop.astype(np.float32), nan=float(np.nanmedian(celsius_crop)))
    med = float(np.median(t))
    hot = (t - med >= minor).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(hot, connectivity=8)
    spots = []
    h, w = t.shape
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < max(2, 0.0005 * h * w):
            continue
        m = lab == i
        spots.append({"centroid": [float(cents[i][0]) / w, float(cents[i][1]) / h], "area_norm": float(m.mean()),
                      "max_c": float(t[m].max()), "delta_c": float(t[m].max() - med)})
    d = max([s["delta_c"] for s in spots], default=float(t.max() - med))
    sev = "critical" if d >= crit else "serious" if d >= thr else "minor" if d >= minor else "none"
    return HotspotReport(med, float(t.max()), float(d), sorted(spots, key=lambda s: -s["delta_c"]), sev)


def thermal_features(celsius_crop: np.ndarray) -> np.ndarray:
    """Fixed-length thermal descriptor: ΔT percentiles, hot-area fractions, gradient energy."""
    t = np.nan_to_num(celsius_crop.astype(np.float32), nan=float(np.nanmedian(celsius_crop)))
    d = t - np.median(t)
    pct = np.percentile(d, [50, 75, 90, 95, 99, 100])
    frac = [float((d >= k).mean()) for k in (2, 5, 10, 20)]
    gy, gx = np.gradient(t)
    return np.array(list(pct) + frac + [float(np.mean(np.hypot(gx, gy))), float(d.std())], dtype=np.float32)


def fuse(rgb_embedding: np.ndarray, thermal_feat: np.ndarray, thermal_weight: float = 1.0) -> np.ndarray:
    t = thermal_feat / (np.linalg.norm(thermal_feat) + 1e-6)
    v = np.concatenate([rgb_embedding.astype(np.float32), thermal_weight * t])
    return v / (np.linalg.norm(v) + 1e-12)
