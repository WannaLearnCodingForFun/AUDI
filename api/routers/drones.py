"""Multi-drone operation (6.3) with scoped, revocable API keys and signed uploads (6.5).

Admin
  POST /api/admin/drones                 register; the API key is returned ONCE
  GET  /api/admin/drones                 list (no secrets)
  POST /api/admin/drones/{id}/revoke     disable the key
Drone (header X-AUDI-Key; each route needs a scope; the drone may only touch its sites)
  POST /api/drone/sessions/start         sessions:start
  POST /api/drone/sessions/{id}/end      sessions:end
  POST /api/drone/telemetry              telemetry:write
  POST /api/drone/frames                 frames:upload — raw JPEG body, signed:
                                         X-AUDI-Timestamp + X-AUDI-Signature = HMAC-SHA256(sha256(key), "<ts>.<body>")
                                         X-AUDI-Session, optional X-AUDI-Pose (JSON)
"""

from __future__ import annotations

import io
import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from ml.config import cfg
from ml.mission import SessionStateError

from api.deps import admin_only, get_services, require_drone_scope, verify_signed_body
from api.routers.spatial import Pose
from api.services import Services

router = APIRouter(tags=["drones"])


class DroneIn(BaseModel):
    name: str
    site_ids: List[str]
    model: Optional[str] = None
    scopes: Optional[List[str]] = None


class DroneSessionIn(BaseModel):
    site_id: str


def _check_site(drone: Dict[str, Any], site_id: str) -> None:
    if site_id not in (drone.get("site_ids") or []):
        raise HTTPException(status.HTTP_403_FORBIDDEN, f"Drone {drone['drone_id']} is not registered for site {site_id}")


@router.post("/api/admin/drones", status_code=201)
def register(body: DroneIn, user: Dict = Depends(admin_only), services: Services = Depends(get_services)) -> Dict[str, Any]:
    allowed = list(cfg(services.config, "security", "drone_scopes"))
    scopes = body.scopes if body.scopes is not None else allowed
    bad = [s for s in scopes if s not in allowed]
    if bad:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"Unknown scopes {bad}")
    try:
        return services.store.register_drone(body.name, body.site_ids, scopes, body.model, user["username"])
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc


@router.get("/api/admin/drones")
def list_drones(_: Dict = Depends(admin_only), services: Services = Depends(get_services)) -> List[Dict[str, Any]]:
    return services.store.list_drones()


@router.post("/api/admin/drones/{drone_id}/revoke")
def revoke(drone_id: str, user: Dict = Depends(admin_only), services: Services = Depends(get_services)) -> Dict[str, Any]:
    try:
        return services.store.set_drone_active(drone_id, False, user["username"])
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown drone {drone_id}") from exc


@router.post("/api/drone/sessions/start")
def drone_start(body: DroneSessionIn, drone: Dict = Depends(require_drone_scope("sessions:start")),
                services: Services = Depends(get_services)) -> Dict[str, Any]:
    _check_site(drone, body.site_id)
    try:
        return services.mission.start(body.site_id, f"drone:{drone['name']}", drone_id=drone["drone_id"])
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc


def _own_session(services: Services, drone: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    s = services.store.get_session(session_id)
    if s is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown session {session_id}")
    if s.get("drone_id") != drone["drone_id"]:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Session belongs to another drone")
    return s


@router.post("/api/drone/sessions/{session_id}/end")
def drone_end(session_id: str, drone: Dict = Depends(require_drone_scope("sessions:end")), services: Services = Depends(get_services)) -> Dict[str, Any]:
    _own_session(services, drone, session_id)
    try:
        return services.mission.end(session_id)
    except SessionStateError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc


@router.post("/api/drone/telemetry")
def drone_telemetry(body: Pose, drone: Dict = Depends(require_drone_scope("telemetry:write")), services: Services = Depends(get_services)) -> Dict[str, Any]:
    sid = services.live.session_id if services.live is not None else None
    if sid is None:
        return {"accepted": False}
    _own_session(services, drone, sid)
    return {"accepted": services.live.update_pose(body.model_dump())}


@router.post("/api/drone/frames")
async def drone_frames(request: Request, drone: Dict = Depends(require_drone_scope("frames:upload")),
                       services: Services = Depends(get_services)) -> Dict[str, Any]:
    secret = services.store.drone_secret(drone["drone_id"])
    body = await verify_signed_body(request, services, secret)
    limit = int(cfg(services.config, "api", "max_upload_mb")) * 1024 * 1024
    if len(body) > limit:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Frame too large")
    sid = request.headers.get("x-audi-session", "")
    sess = _own_session(services, drone, sid)
    if sess["status"] != "active":
        raise HTTPException(status.HTTP_409_CONFLICT, "Session is not active")
    pose = None
    if request.headers.get("x-audi-pose"):
        try:
            pose = Pose(**json.loads(request.headers["x-audi-pose"])).model_dump()
        except (ValueError, TypeError) as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Bad X-AUDI-Pose") from exc
    try:
        with Image.open(io.BytesIO(body)) as im:
            frame = im.convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Not an image") from exc
    import asyncio

    events = await asyncio.to_thread(services.engine.process_frame, frame, sid, None, None, None, None, pose)
    return {"events": [e.to_dict() for e in events], "quality": services.engine.last_frame_quality}
