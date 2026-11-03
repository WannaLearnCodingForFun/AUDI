"""Geo-anchored identity: project a detection onto the ground plane (upgrade 1.1).

Given the drone's GPS fix, its height above ground, the camera heading / pitch / roll and the
pinhole intrinsics, the ray through a bounding-box anchor pixel is intersected with the flat
ground plane z = 0 in a local East-North-Up frame centred under the drone. The hit point is
converted back to latitude / longitude with a local tangent-plane approximation (error well
under a centimetre over the few hundred metres a camera can see).

Conventions
  heading_deg  0 = north, 90 = east (clockwise, compass)
  pitch_deg    angle of the optical axis BELOW the horizon: 0 = level, 90 = nadir
  roll_deg     clockwise rotation of the image about the optical axis
  anchor       which bbox point touches the ground: "center" for flat assets seen from above
               (solar panels), "bottom" for upright ones (poles, walls, machinery)

The uncertainty estimate combines the GPS error with the angular error of the attitude
solution propagated over the slant range, and grows quickly for grazing rays.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ml.asset_identity import EARTH_RADIUS_M, GeoLocation
from ml.camera_model import CameraIntrinsics
from ml.config import get_nested


@dataclass
class CameraPose:
    latitude: float
    longitude: float
    altitude_agl: float          # metres above the ground plane
    heading_deg: float = 0.0
    pitch_deg: float = 90.0
    roll_deg: float = 0.0
    gps_accuracy_m: Optional[float] = None
    altitude_msl: Optional[float] = None
    range_m: Optional[float] = None   # rangefinder / LiDAR distance along the optical axis, if any

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> Optional["CameraPose"]:
        if not d or d.get("latitude") is None or d.get("longitude") is None:
            return None
        alt = d.get("altitude_agl", d.get("relative_altitude", d.get("altitude")))
        if alt is None:
            return None
        def f(k, default=None):
            v = d.get(k, default)
            return None if v is None else float(v)
        return cls(
            float(d["latitude"]), float(d["longitude"]), float(alt),
            f("heading_deg", 0.0), f("pitch_deg", 90.0), f("roll_deg", 0.0),
            f("gps_accuracy_m"), f("altitude_msl"), f("range_m"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class GroundFix:
    latitude: float
    longitude: float
    east_m: float                # offset from the point under the drone
    north_m: float
    slant_range_m: float
    uncertainty_m: float

    def to_location(self) -> GeoLocation:
        return GeoLocation(self.latitude, self.longitude, 0.0)

    def to_dict(self) -> Dict[str, float]:
        return dict(self.__dict__)


def offset_to_latlon(lat: float, lon: float, east_m: float, north_m: float) -> Tuple[float, float]:
    dlat = north_m / EARTH_RADIUS_M
    dlon = east_m / (EARTH_RADIUS_M * math.cos(math.radians(lat)))
    return lat + math.degrees(dlat), lon + math.degrees(dlon)


def latlon_to_offset(lat0: float, lon0: float, lat: float, lon: float) -> Tuple[float, float]:
    """(east, north) metres of (lat, lon) relative to (lat0, lon0), local tangent plane."""
    north = math.radians(lat - lat0) * EARTH_RADIUS_M
    east = math.radians(lon - lon0) * EARTH_RADIUS_M * math.cos(math.radians(lat0))
    return east, north


def camera_axes(heading_deg: float, pitch_deg: float, roll_deg: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World-ENU unit vectors of the camera's right, down and forward axes."""
    h, p, r = (math.radians(v) for v in (heading_deg, pitch_deg, roll_deg))
    forward = np.array([math.sin(h) * math.cos(p), math.cos(h) * math.cos(p), -math.sin(p)])
    right0 = np.array([math.cos(h), -math.sin(h), 0.0])
    down0 = np.cross(forward, right0)
    right = math.cos(r) * right0 + math.sin(r) * down0
    down = math.cos(r) * down0 - math.sin(r) * right0
    return right, down, forward


def pixel_ray(u: float, v: float, intr: CameraIntrinsics, pose: CameraPose) -> np.ndarray:
    pt = intr.undistort_points(np.array([[u, v]]))[0]
    xn, yn = (pt[0] - intr.cx) / intr.fx, (pt[1] - intr.cy) / intr.fy
    right, down, forward = camera_axes(pose.heading_deg, pose.pitch_deg, pose.roll_deg)
    ray = xn * right + yn * down + forward
    return ray / np.linalg.norm(ray)


def project_pixel(
    u: float, v: float, intr: CameraIntrinsics, pose: CameraPose,
    gps_accuracy_m: float = 1.5, attitude_error_deg: float = 1.0, max_range_m: float = 500.0,
) -> Optional[GroundFix]:
    """Ground point seen at pixel (u, v), or None when the ray does not hit the ground in range."""
    ray = pixel_ray(u, v, intr, pose)
    if ray[2] >= -1e-6 or pose.altitude_agl <= 0:
        return None
    t = pose.altitude_agl / -ray[2]
    if t > max_range_m:
        return None
    east, north = float(t * ray[0]), float(t * ray[1])
    lat, lon = offset_to_latlon(pose.latitude, pose.longitude, east, north)
    # angular error δ on a ray with depression angle e moves the hit point by ≈ t·δ / sin(e)
    depression = math.asin(min(1.0, -ray[2]))
    angular = t * math.radians(attitude_error_deg) / max(math.sin(depression), 0.05)
    gps = pose.gps_accuracy_m if pose.gps_accuracy_m is not None else gps_accuracy_m
    return GroundFix(lat, lon, east, north, float(t), float(math.hypot(gps, angular)))


def anchor_pixel(bbox: Sequence[float], anchor: str) -> Tuple[float, float]:
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if anchor == "bottom":
        return (x1 + x2) / 2.0, y2
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


class GeoProjector:
    """Config-driven projection of detections into world coordinates."""

    def __init__(self, config: Dict[str, Any]):
        g = get_nested(config, "geo", default={}) or {}
        self.enabled = bool(g.get("enabled", True))
        self.anchors: Dict[str, str] = dict(g.get("anchor", {}))
        self.gps_accuracy_m = float(g.get("gps_accuracy_m", 1.5))
        self.attitude_error_deg = float(g.get("attitude_error_deg", 1.0))
        self.max_range_m = float(g.get("max_range_m", 500.0))
        self.config = config

    def project(self, bbox: Sequence[float], asset_type: str, pose: CameraPose, intr: CameraIntrinsics) -> Optional[GroundFix]:
        if not self.enabled:
            return None
        u, v = anchor_pixel(bbox, self.anchors.get(asset_type, self.anchors.get("default", "center")))
        return project_pixel(u, v, intr, pose, self.gps_accuracy_m, self.attitude_error_deg, self.max_range_m)
