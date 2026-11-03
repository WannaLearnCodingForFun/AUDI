"""ROS-independent conversions used by the audi_ros nodes (unit-tested without ROS).

MAVROS conventions: NavSatFix gives WGS-84 lat/lon; `/mavros/local_position/pose` gives an
ENU orientation quaternion of the vehicle (x east, y north, z up). The camera is on a gimbal
whose pitch comes from `/mavros/mount_control/orientation` (or a fixed mount angle).
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Optional, Tuple


def quaternion_to_euler_enu(x: float, y: float, z: float, w: float) -> Tuple[float, float, float]:
    """(roll, pitch, yaw) in radians, ENU frame, yaw 0 = east, counter-clockwise."""
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    s = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = math.asin(s)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def enu_yaw_to_compass_heading(yaw_rad: float) -> float:
    """ENU yaw (0 = east, CCW) → compass heading (0 = north, CW), degrees."""
    return (90.0 - math.degrees(yaw_rad)) % 360.0


def camera_pose(fix: Dict[str, float], quat: Optional[Tuple[float, float, float, float]], rel_alt_m: float,
                gimbal_pitch_down_deg: float, gimbal_yaw_offset_deg: float = 0.0, gps_accuracy_m: Optional[float] = None,
                range_m: Optional[float] = None) -> Dict[str, Any]:
    """Build the engine's CameraPose dict from MAVROS data."""
    heading, roll = 0.0, 0.0
    if quat is not None:
        r, _, yaw = quaternion_to_euler_enu(*quat)
        heading = (enu_yaw_to_compass_heading(yaw) + gimbal_yaw_offset_deg) % 360.0
        roll = 0.0     # a stabilised gimbal keeps the image level
    d = {"latitude": fix["latitude"], "longitude": fix["longitude"], "altitude_agl": float(rel_alt_m),
         "heading_deg": heading, "pitch_deg": float(gimbal_pitch_down_deg), "roll_deg": roll}
    if gps_accuracy_m is not None:
        d["gps_accuracy_m"] = float(gps_accuracy_m)
    if range_m is not None and math.isfinite(range_m):
        d["range_m"] = float(range_m)
    return d


def navsat_accuracy(position_covariance, covariance_type: int) -> Optional[float]:
    """Horizontal 1-σ from a NavSatFix covariance (type 0 = unknown)."""
    if covariance_type == 0 or not position_covariance:
        return None
    return math.sqrt(max(0.0, (float(position_covariance[0]) + float(position_covariance[4])) / 2.0))


EVENT_FIELDS = ("session_id", "type", "asset_id", "asset_type", "track_id", "bbox", "condition_label", "condition_confidence",
                "identity_stage", "identity_confidence", "change_magnitude", "severity_score", "severity_level", "priority",
                "critical", "latitude", "longitude", "location_uncertainty_m", "guidance", "growth", "observation_id", "json")


def event_fields(ev: Dict[str, Any]) -> Dict[str, Any]:
    """AssetEvent dict → audi_msgs/AssetEvent field values."""
    fix = ((ev.get("geo") or {}).get("ground_fix")) or {}
    cond = ev.get("condition") or {}
    ident = ev.get("identity") or {}
    return {
        "session_id": ev.get("session_id") or "",
        "type": ev.get("type") or "",
        "asset_id": ev.get("asset_id") or "",
        "asset_type": ev.get("asset_type") or "",
        "track_id": int(ev["track_id"]) if ev.get("track_id") is not None else -1,
        "bbox": [int(v) for v in (ev.get("bbox") or [0, 0, 0, 0])],
        "condition_label": cond.get("label") or "",
        "condition_confidence": float(cond.get("confidence") or 0.0),
        "identity_stage": ident.get("stage") or "",
        "identity_confidence": float(ident.get("confidence") or 0.0),
        "change_magnitude": float((ev.get("change") or {}).get("change_magnitude") or 0.0),
        "severity_score": float(ev.get("severity_score") or 0.0),
        "severity_level": ev.get("severity_level") or "",
        "priority": bool(ev.get("priority")),
        "critical": bool(ev.get("critical")),
        "latitude": float(fix.get("latitude", float("nan"))),
        "longitude": float(fix.get("longitude", float("nan"))),
        "location_uncertainty_m": float(fix.get("uncertainty_m", float("nan"))),
        "guidance": (ev.get("guidance") or {}).get("message") or "",
        "growth": ev.get("growth") or "",
        "observation_id": ev.get("observation_id") or "",
        "json": json.dumps(ev, default=str),
    }


def event_signature(ev: Dict[str, Any]) -> Tuple:
    """What a replay must reproduce (launch test vs smoke_e2e): type, asset type, condition, persisted."""
    return (ev.get("type"), ev.get("asset_type"), (ev.get("condition") or {}).get("label"), bool(ev.get("persisted")))
