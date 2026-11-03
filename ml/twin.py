"""3D digital twin scene (upgrade 4.4).

Builds a browser scene (frontend Twin page, three.js) from memory: every asset becomes a box
at its local East-North-Up position, sized by asset type, coloured by health; every active
defect instance is a marker placed on the asset's inspected face from its normalised
position on the reference view. When a photogrammetry mesh exists for the site
(OpenDroneMap `odm_textured_model_geo.obj` / glTF, or a Gaussian-splat .ply, registered as
site_maps.meta.mesh_path) it is referenced so the viewer loads it underneath.

Clicking a defect in the viewer opens its history page (`/defect-instances/{instance_id}`).
"""

from __future__ import annotations

from typing import Any, Dict, List

from ml.config import get_nested
from ml.geo_projection import latlon_to_offset
from ml.health_index import health_band

DEFAULT_DIMS = {  # width (E), depth (N), height (U) in metres
    "solar_panel": (1.0, 1.7, 0.05),
    "utility_pole": (0.3, 0.3, 10.0),
    "building": (6.0, 0.4, 4.0),
    "machinery": (1.0, 1.0, 1.2),
}


def defect_anchor(asset_type: str, dims, cx: float, cy: float) -> List[float]:
    """Local (x, y, z) of a defect on the inspected face from normalised reference-view coords."""
    w, d, h = dims
    if asset_type == "solar_panel":       # seen from above: image x → east, image y → south
        return [(cx - 0.5) * w, (0.5 - cy) * d, h]
    # upright assets seen from the side (south face): image x → east, image y → down
    return [(cx - 0.5) * w, -d / 2.0, (1.0 - cy) * h]


def build_scene(store, site_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    dims_cfg = get_nested(config, "twin", "dims_m", default={}) or {}
    assets = store.list_assets(site_id)
    site = store.get_site(site_id) or {}
    located = [a for a in assets if a.get("location")]
    if site.get("latitude") is not None:
        origin = (site["latitude"], site["longitude"])
    elif located:
        origin = (sum(a["location"]["latitude"] for a in located) / len(located),
                  sum(a["location"]["longitude"] for a in located) / len(located))
    else:
        origin = None
    instances = store.list_defect_instances(site_id=site_id, status="active")
    nodes = []
    for a in located:
        e, n = latlon_to_offset(origin[0], origin[1], a["location"]["latitude"], a["location"]["longitude"])
        dims = tuple(dims_cfg.get(a["asset_type"], DEFAULT_DIMS.get(a["asset_type"], (1.0, 1.0, 1.0))))
        defects = []
        for i in instances:
            if i["asset_id"] != a["asset_id"]:
                continue
            defects.append({
                "instance_id": i["instance_id"], "defect_type": i["defect_type"],
                "local": defect_anchor(a["asset_type"], dims, i["centroid_x"], i["centroid_y"]),
                "area_cm2": i.get("area_cm2"), "length_cm": i.get("length_cm"),
                "predicted_critical_ts": i.get("predicted_critical_ts"),
                "history_url": f"/api/defect-instances/{i['instance_id']}",
            })
        nodes.append({
            "asset_id": a["asset_id"], "asset_type": a["asset_type"], "position": [e, n, 0.0], "dims": list(dims),
            "health_index": a.get("health_index"), "health_band": health_band(a.get("health_index")),
            "revisit_priority": a["revisit_priority_score"], "condition": a["current_condition"], "defects": defects,
            "history_url": f"/api/assets/{a['asset_id']}/history",
        })
    smap = store.active_site_map(site_id)
    mesh = (smap or {}).get("meta", {}) or {}
    return {
        "site_id": site_id,
        "origin": {"latitude": origin[0], "longitude": origin[1]} if origin else None,
        "frame": "ENU metres (x east, y north, z up)",
        "assets": nodes,
        "unlocated_assets": [a["asset_id"] for a in assets if not a.get("location")],
        "mesh": {"path": mesh.get("mesh_path"), "format": mesh.get("mesh_format")} if mesh.get("mesh_path") else None,
    }
