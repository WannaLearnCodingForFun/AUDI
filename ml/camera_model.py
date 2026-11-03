"""Camera intrinsics and metric scale (upgrade 3.6).

`CameraIntrinsics` holds the pinhole parameters, loaded from a calibration file written by
`scripts/calibrate_camera.py` (chessboard, OpenCV) or derived from a horizontal field of view
when no calibration exists. Parameters are rescaled to the actual frame size, so one
calibration at full resolution serves every capture size.

Scale: the ground sampling distance (GSD) at range d is d / f metres per pixel, so a length
of n pixels on a surface facing the camera is n · d / f metres. `scale_for_bbox` estimates the
range from the known physical size of the asset type when no range sensor is available.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ml.config import get_nested, resolve_path


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: List[float] = field(default_factory=list)
    source: str = "hfov"

    @classmethod
    def from_hfov(cls, width: int, height: int, hfov_deg: float) -> "CameraIntrinsics":
        f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        return cls(f, f, width / 2.0, height / 2.0, int(width), int(height), [], "hfov")

    @classmethod
    def load(cls, path: Path) -> "CameraIntrinsics":
        d = json.loads(Path(path).read_text())
        return cls(
            float(d["fx"]), float(d["fy"]), float(d["cx"]), float(d["cy"]),
            int(d["width"]), int(d["height"]), [float(x) for x in d.get("distortion", [])], "calibration",
        )

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    def scaled(self, width: int, height: int) -> "CameraIntrinsics":
        if width == self.width and height == self.height:
            return self
        sx, sy = width / float(self.width), height / float(self.height)
        return CameraIntrinsics(self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy, int(width), int(height),
                                list(self.distortion), self.source)

    @property
    def hfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan((self.width / 2.0) / self.fx))

    @property
    def vfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan((self.height / 2.0) / self.fy))

    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64)

    def undistort_points(self, pts: np.ndarray) -> np.ndarray:
        """Pixel points (N, 2) → undistorted pixel points (no-op without distortion)."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if not self.distortion or not any(self.distortion):
            return pts
        import cv2

        K = self.matrix()
        out = cv2.undistortPoints(pts.reshape(-1, 1, 2), K, np.asarray(self.distortion), P=K)
        return out.reshape(-1, 2)

    def gsd_m_per_px(self, distance_m: float) -> float:
        """Ground sampling distance of a fronto-parallel surface at `distance_m`."""
        return float(distance_m) / float((self.fx + self.fy) / 2.0)


def load_intrinsics(config: Dict[str, Any], width: int, height: int) -> CameraIntrinsics:
    """Calibration file if present, otherwise the configured horizontal FOV, at the frame size."""
    cm = get_nested(config, "camera_model", default={}) or {}
    path = cm.get("calibration_file")
    if path:
        p = resolve_path(path)
        if p.exists():
            return CameraIntrinsics.load(p).scaled(width, height)
    return CameraIntrinsics.from_hfov(width, height, float(cm.get("hfov_deg", 78.0)))


def physical_size_m(config: Dict[str, Any], asset_type: str) -> Optional[float]:
    """Nominal width of an asset type in metres (used for range-from-size)."""
    table = get_nested(config, "camera_model", "asset_width_m", default={}) or {}
    v = table.get(asset_type)
    return float(v) if v else None


def distance_from_bbox(intr: CameraIntrinsics, bbox: Sequence[float], width_m: float) -> float:
    """Range to an object of known width from its pixel width (pinhole model)."""
    px = max(1.0, float(bbox[2]) - float(bbox[0]))
    return width_m * intr.fx / px


@dataclass
class MetricScale:
    m_per_px: float
    distance_m: float
    source: str          # range | altitude | size | none

    @property
    def cm2_per_px(self) -> float:
        return (self.m_per_px * 100.0) ** 2

    def to_dict(self) -> Dict[str, Any]:
        return {"m_per_px": self.m_per_px, "distance_m": self.distance_m, "source": self.source}


def scale_for_bbox(
    config: Dict[str, Any],
    intr: CameraIntrinsics,
    bbox: Sequence[float],
    asset_type: str,
    range_m: Optional[float] = None,
    altitude_m: Optional[float] = None,
    pitch_deg: Optional[float] = None,
) -> Optional[MetricScale]:
    """Best available metric scale for an object crop, in frame pixels.

    Priority: measured range (LiDAR / rangefinder) → altitude and pitch (slant range to the
    ground point) → known physical width of the asset type. None when nothing is known.
    """
    if range_m is not None and range_m > 0:
        return MetricScale(intr.gsd_m_per_px(range_m), float(range_m), "range")
    if altitude_m is not None and altitude_m > 0 and pitch_deg is not None and pitch_deg > 1.0:
        slant = altitude_m / math.sin(math.radians(min(90.0, pitch_deg)))
        return MetricScale(intr.gsd_m_per_px(slant), slant, "altitude")
    w = physical_size_m(config, asset_type)
    if w:
        d = distance_from_bbox(intr, bbox, w)
        return MetricScale(intr.gsd_m_per_px(d), d, "size")
    return None
