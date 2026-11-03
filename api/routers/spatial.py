"""Spatial memory: site maps and footprints (1.2), coverage (2.5), telemetry (1.1/2.4),
3D twin (4.4) and advisory revisit routes (4.5)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ml.config import get_nested, resolve_path
from ml.coverage import coverage, unseen_heatmap
from ml.route_plan import build_plan, to_plan_file
from ml.schemas import new_id
from ml.severity import asset_criticality
from ml.twin import build_scene

from api.deps import admin_only, any_user, field_roles, get_services, operator_only
from api.services import Services

router = APIRouter(tags=["spatial"])


class SiteMapIn(BaseModel):
    image_path: str
    geotransform: Optional[List[float]] = None
    mesh_path: Optional[str] = None
    mesh_format: Optional[str] = None


class FootprintIn(BaseModel):
    polygon: Optional[List[List[float]]] = Field(None, description="[[lat, lon], ...]; null clears")


class MappedAssetIn(BaseModel):
    asset_type: str
    polygon: List[List[float]]


class Pose(BaseModel):
    latitude: float
    longitude: float
    altitude_agl: float
    heading_deg: float = 0.0
    pitch_deg: float = 90.0
    roll_deg: float = 0.0
    gps_accuracy_m: Optional[float] = None
    range_m: Optional[float] = None


class RouteRequest(BaseModel):
    asset_ids: Optional[List[str]] = None
    min_priority: Optional[float] = None
    home: Optional[List[float]] = Field(None, description="[lat, lon]; default: site location or first asset")
    cruise_alt_m: Optional[float] = None


def _site(services: Services, site_id: str) -> Dict[str, Any]:
    s = services.store.get_site(site_id)
    if s is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown site {site_id}")
    return s


def _register_map(services: Services, site_id: str, path, gt, actor: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    from PIL import Image

    from ml.site_map import read_geotransform

    gt = gt or read_geotransform(path)
    if not gt or len(gt) != 6:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "A 6-term geotransform is required (GeoTIFF tags or explicit)")
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
        w, h = im.size
    row = services.store.add_site_map(site_id, str(path), gt, w, h, actor, meta)
    services.engine.invalidate_site(site_id)
    return row


@router.get("/api/sites/{site_id}/map")
def get_map(site_id: str, _: Dict = Depends(any_user), services: Services = Depends(get_services)) -> Dict[str, Any]:
    _site(services, site_id)
    return {"map": services.store.active_site_map(site_id), "footprints": services.store.footprint_assets(site_id)}


@router.post("/api/sites/{site_id}/map", status_code=201)
def set_map(site_id: str, body: SiteMapIn, user: Dict = Depends(admin_only), services: Services = Depends(get_services)) -> Dict[str, Any]:
    _site(services, site_id)
    path = resolve_path(body.image_path)
    if not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No such image {body.image_path}")
    meta = {k: v for k, v in {"mesh_path": body.mesh_path, "mesh_format": body.mesh_format}.items() if v}
    return _register_map(services, site_id, path, body.geotransform, user["username"], meta)


@router.post("/api/sites/{site_id}/map/upload", status_code=201)
async def upload_map(site_id: str, file: UploadFile = File(...), geotransform: Optional[str] = Form(None),
                     user: Dict = Depends(admin_only), services: Services = Depends(get_services)) -> Dict[str, Any]:
    _site(services, site_id)
    suffix = ".tif" if (file.filename or "").lower().endswith((".tif", ".tiff")) else ".png"
    dest = resolve_path(get_nested(services.config, "site_map", "dir", default="data/site_maps")) / site_id / f"{new_id('ortho')}{suffix}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(await file.read())
    gt = json.loads(geotransform) if geotransform else None
    return _register_map(services, site_id, dest, gt, user["username"], {"uploaded_name": file.filename})


@router.put("/api/assets/{asset_id}/footprint")
def put_footprint(asset_id: str, body: FootprintIn, user: Dict = Depends(admin_only), services: Services = Depends(get_services)) -> Dict[str, Any]:
    try:
        a = services.store.set_footprint(asset_id, body.polygon, user["username"])
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown asset {asset_id}") from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    services.engine.invalidate_site(a["site_id"])
    return a


@router.post("/api/sites/{site_id}/map/assets", status_code=201)
def add_mapped_asset(site_id: str, body: MappedAssetIn, user: Dict = Depends(admin_only), services: Services = Depends(get_services)) -> Dict[str, Any]:
    _site(services, site_id)
    if len(body.polygon) < 3:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "A footprint needs at least 3 vertices")
    a = services.store.register_mapped_asset(site_id, body.asset_type, body.polygon, asset_criticality(body.asset_type, services.config), user["username"])
    services.engine.invalidate_site(site_id)
    return a


@router.get("/api/sessions/{session_id}/coverage")
def session_coverage(session_id: str, due_only: bool = False, _: Dict = Depends(field_roles), services: Services = Depends(get_services)) -> Dict[str, Any]:
    try:
        return coverage(services.store, session_id, services.config, due_only)
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown session {session_id}") from exc


@router.get("/api/sites/{site_id}/unseen")
def site_unseen(site_id: str, _: Dict = Depends(any_user), services: Services = Depends(get_services)) -> Dict[str, Any]:
    _site(services, site_id)
    return unseen_heatmap(services.store, site_id, float(get_nested(services.config, "coverage", "heatmap_cell_m", default=25.0)))


@router.post("/api/live/telemetry")
def live_telemetry(body: Pose, _: Dict = Depends(operator_only), services: Services = Depends(get_services)) -> Dict[str, Any]:
    """Latest drone pose for geo-anchoring and guidance (also fed by the ROS 2 bridge)."""
    return {"accepted": services.live.update_pose(body.model_dump())}


@router.get("/api/sites/{site_id}/twin")
def twin(site_id: str, _: Dict = Depends(any_user), services: Services = Depends(get_services)) -> Dict[str, Any]:
    _site(services, site_id)
    return build_scene(services.store, site_id, services.config)


@router.post("/api/sites/{site_id}/route-plan")
def route_plan(site_id: str, body: RouteRequest, format: str = Query("json", pattern="^(json|plan)$"),
               _: Dict = Depends(field_roles), services: Services = Depends(get_services)):
    site = _site(services, site_id)
    ranked = services.store.recompute_revisit(site_id, enqueue=False)
    if body.asset_ids:
        wanted = set(body.asset_ids)
        assets = [a for a in ranked if a["asset_id"] in wanted]
    else:
        thr = body.min_priority if body.min_priority is not None else float(get_nested(services.config, "route_plan", "min_priority", default=0.3))
        assets = [a for a in ranked if a["revisit_priority_score"] >= thr]
    home = body.home
    if home is None:
        if site.get("latitude") is not None:
            home = [site["latitude"], site["longitude"]]
        else:
            loc = next((a["location"] for a in assets if a.get("location")), None)
            if loc is None:
                raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "No home position and no located assets")
            home = [loc["latitude"], loc["longitude"]]
    res = build_plan(assets, (home[0], home[1]), services.config, body.cruise_alt_m)
    if format == "plan":
        return Response(to_plan_file(res), media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="audi_revisit_{site_id}.plan"'})
    return res
