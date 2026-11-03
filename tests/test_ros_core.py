"""4.2 ROS 2 bridge conversions (no ROS needed): MAVROS → CameraPose, AssetEvent → message fields."""

import math

import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ros2" / "audi_ros"))

from audi_ros.core import camera_pose, enu_yaw_to_compass_heading, event_fields, event_signature, navsat_accuracy, quaternion_to_euler_enu  # noqa: E402


def quat_from_yaw(yaw):
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def test_enu_yaw_to_compass():
    assert enu_yaw_to_compass_heading(0.0) == pytest.approx(90.0)          # facing east
    assert enu_yaw_to_compass_heading(math.pi / 2) == pytest.approx(0.0)   # facing north
    assert enu_yaw_to_compass_heading(math.pi) == pytest.approx(270.0)


def test_camera_pose_from_mavros():
    p = camera_pose({"latitude": 47.0, "longitude": 8.0}, quat_from_yaw(math.pi / 2), 25.0, 60.0, gps_accuracy_m=1.2, range_m=float("nan"))
    assert p["heading_deg"] == pytest.approx(0.0, abs=1e-6) and p["pitch_deg"] == 60.0 and p["altitude_agl"] == 25.0
    assert "range_m" not in p and p["gps_accuracy_m"] == 1.2
    r, pitch, yaw = quaternion_to_euler_enu(*quat_from_yaw(0.3))
    assert yaw == pytest.approx(0.3) and abs(r) < 1e-9 and abs(pitch) < 1e-9


def test_navsat_accuracy():
    assert navsat_accuracy([4.0, 0, 0, 0, 1.0, 0, 0, 0, 9.0], 2) == pytest.approx(math.sqrt(2.5))
    assert navsat_accuracy([0] * 9, 0) is None


def test_event_fields_roundtrip():
    ev = {"type": "change", "session_id": "s", "asset_id": "a", "asset_type": "solar_panel", "track_id": 3, "bbox": [1, 2, 3, 4],
          "condition": {"label": "crack", "confidence": 0.9}, "identity": {"stage": "geometric", "confidence": 0.8},
          "change": {"change_magnitude": 0.5}, "severity_score": 0.6, "severity_level": "priority", "priority": True,
          "critical": False, "geo": {"ground_fix": {"latitude": 47.1, "longitude": 8.2, "uncertainty_m": 1.5}},
          "guidance": {"message": "A: move 3 m closer"}, "growth": "crack grew 40%", "observation_id": "o", "persisted": True}
    f = event_fields(ev)
    assert f["latitude"] == 47.1 and f["guidance"].startswith("A:") and f["identity_stage"] == "geometric"
    assert event_signature(ev) == ("change", "solar_panel", "crack", True)
    assert math.isnan(event_fields({"type": "new_asset"})["latitude"])
