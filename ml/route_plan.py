"""Advisory revisit route (upgrade 4.5) → QGroundControl .plan file.

Covers the assets that are due (revisit priority ≥ threshold, or an explicit list) at each
asset type's required standoff distance. Order: nearest-neighbour tour from home refined by
2-opt on local East-North coordinates. Waypoints:
  flat assets (solar)   hover above at altitude = standoff, camera pointed down
  upright assets        stand off `standoff` metres on the approach side at mid height,
                        yaw toward the asset
Every inspection waypoint is preceded by MAV_CMD_DO_SET_ROI_LOCATION on the asset.

The plan is ADVISORY: the operator reviews and flies it manually; the SRS forbids AUDI from
controlling the drone, so nothing here talks to the autopilot.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ml.config import get_nested
from ml.geo_projection import latlon_to_offset, offset_to_latlon

MAV_CMD_NAV_WAYPOINT = 16
MAV_CMD_NAV_TAKEOFF = 22
MAV_CMD_NAV_RETURN_TO_LAUNCH = 20
MAV_CMD_DO_SET_ROI_LOCATION = 195
MAV_FRAME_GLOBAL_RELATIVE_ALT = 3


def tour_length(points: Sequence[Tuple[float, float]], order: Sequence[int], home: Tuple[float, float], closed: bool = True) -> float:
    if not order:
        return 0.0
    path = [home] + [points[i] for i in order] + ([home] if closed else [])
    return sum(math.dist(path[i], path[i + 1]) for i in range(len(path) - 1))


def plan_order(points: Sequence[Tuple[float, float]], home: Tuple[float, float]) -> List[int]:
    left = set(range(len(points)))
    order: List[int] = []
    cur = home
    while left:
        nxt = min(left, key=lambda i: (math.dist(cur, points[i]), i))
        order.append(nxt)
        left.remove(nxt)
        cur = points[nxt]
    improved = True
    while improved and len(order) > 3:
        improved = False
        for i in range(len(order) - 1):
            for j in range(i + 2, len(order) + 1):
                cand = order[:i] + order[i:j][::-1] + order[j:]
                if tour_length(points, cand, home) < tour_length(points, order, home) - 1e-9:
                    order, improved = cand, True
    return order


def _item(seq: int, command: int, params: List[Optional[float]], alt: float) -> Dict[str, Any]:
    return {"AMSLAltAboveTerrain": None, "Altitude": alt, "AltitudeMode": 1, "autoContinue": True, "command": command,
            "doJumpId": seq, "frame": MAV_FRAME_GLOBAL_RELATIVE_ALT, "params": params, "type": "SimpleItem"}


def build_plan(
    assets: Sequence[Dict[str, Any]],
    home: Tuple[float, float],
    config: Dict[str, Any],
    cruise_alt_m: Optional[float] = None,
) -> Dict[str, Any]:
    rp = get_nested(config, "route_plan", default={}) or {}
    standoff = rp.get("standoff_m", {}) or {}
    upright_h = rp.get("upright_height_m", {}) or {}
    flat = set(rp.get("flat_asset_types", ["solar_panel"]))
    hold_s = float(rp.get("hold_s", 3.0))
    cruise = float(cruise_alt_m if cruise_alt_m is not None else rp.get("cruise_alt_m", 30.0))
    located = [a for a in assets if a.get("location")]
    pts = [latlon_to_offset(home[0], home[1], a["location"]["latitude"], a["location"]["longitude"]) for a in located]
    order = plan_order(pts, (0.0, 0.0))
    items: List[Dict[str, Any]] = [_item(1, MAV_CMD_NAV_TAKEOFF, [0, 0, 0, None, home[0], home[1], cruise], cruise)]
    seq = 2
    waypoints = []
    prev = (0.0, 0.0)
    for idx in order:
        a = located[idx]
        lat, lon = a["location"]["latitude"], a["location"]["longitude"]
        d = float(standoff.get(a["asset_type"], standoff.get("default", 8.0)))
        e, n = pts[idx]
        if a["asset_type"] in flat:
            wlat, wlon, alt, yaw = lat, lon, d, None
        else:
            # approach side: from the previous waypoint toward the asset
            vx, vy = e - prev[0], n - prev[1]
            norm = math.hypot(vx, vy) or 1.0
            we, wn = e - vx / norm * d, n - vy / norm * d
            wlat, wlon = offset_to_latlon(home[0], home[1], we, wn)
            alt = float(upright_h.get(a["asset_type"], upright_h.get("default", 6.0))) / 2.0 + 1.0
            yaw = (math.degrees(math.atan2(vx, vy)) + 360.0) % 360.0
            prev = (we, wn)
        if a["asset_type"] in flat:
            prev = (e, n)
        items.append(_item(seq, MAV_CMD_DO_SET_ROI_LOCATION, [0, 0, 0, 0, lat, lon, 0], 0))
        seq += 1
        items.append(_item(seq, MAV_CMD_NAV_WAYPOINT, [hold_s, 0, 0, yaw, wlat, wlon, alt], alt))
        seq += 1
        waypoints.append({"asset_id": a["asset_id"], "asset_type": a["asset_type"], "latitude": wlat, "longitude": wlon,
                          "altitude_m": alt, "standoff_m": d, "yaw_deg": yaw})
    items.append(_item(seq, MAV_CMD_NAV_RETURN_TO_LAUNCH, [0, 0, 0, 0, 0, 0, 0], 0))
    plan = {
        "fileType": "Plan", "version": 1, "groundStation": "AUDI (advisory)",
        "geoFence": {"circles": [], "polygons": [], "version": 2},
        "rallyPoints": {"points": [], "version": 2},
        "mission": {"cruiseSpeed": float(rp.get("cruise_speed_mps", 5.0)), "hoverSpeed": float(rp.get("hover_speed_mps", 3.0)),
                    "firmwareType": 12, "vehicleType": 2, "globalPlanAltitudeMode": 1,
                    "plannedHomePosition": [home[0], home[1], 0], "version": 2, "items": items},
    }
    return {
        "plan": plan,
        "waypoints": waypoints,
        "skipped_no_location": [a["asset_id"] for a in assets if not a.get("location")],
        "route_length_m": tour_length(pts, order, (0.0, 0.0)),
        "coverage": len(located) / len(assets) if assets else 1.0,
        "advisory": True,
    }


def compare_with_manual(assets: Sequence[Dict[str, Any]], home: Tuple[float, float], manual_order: Sequence[str]) -> Dict[str, float]:
    """Route length of the planned tour vs a manual visiting order (asset ids), same assets."""
    located = [a for a in assets if a.get("location")]
    pts = [latlon_to_offset(home[0], home[1], a["location"]["latitude"], a["location"]["longitude"]) for a in located]
    ids = [a["asset_id"] for a in located]
    manual = [ids.index(x) for x in manual_order if x in ids]
    planned = plan_order(pts, (0.0, 0.0))
    lm, lp = tour_length(pts, manual, (0.0, 0.0)), tour_length(pts, planned, (0.0, 0.0))
    return {"manual_length_m": lm, "planned_length_m": lp, "saving_pct": (100.0 * (lm - lp) / lm) if lm else 0.0,
            "manual_coverage": len(set(manual)) / len(located) if located else 1.0, "planned_coverage": 1.0 if located else 1.0}


def to_plan_file(result: Dict[str, Any]) -> str:
    return json.dumps(result["plan"], indent=2)
