"""Assets, history (FR-12), stored images, single-image analysis (D6) and memory search."""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from ml.config import cfg

from api.deps import any_user, field_roles, get_services, user_from_token
from api.services import Services

router = APIRouter(tags=["assets"])


@router.get("/api/health")
def health(services: Services = Depends(get_services)) -> Dict[str, Any]:
    return {
        "status": "ok",
        "backbone": services.embedder.active_model_name,
        "embedding_dim": services.embedder.embedding_dim,
        "detector": type(services.detector).__name__,
        "db_ok": services.store.integrity["ok"],
        "sync": services.sync.status(),
    }


@router.get("/api/summary")
def summary(site_id: Optional[str] = None, _: Dict = Depends(any_user), services: Services = Depends(get_services)) -> Dict[str, Any]:
    return {**services.store.summary(site_id), "severity_levels": dict(cfg(services.config, "severity", "levels"))}


@router.get("/api/assets")
def list_assets(site_id: Optional[str] = None, _: Dict = Depends(any_user), services: Services = Depends(get_services)) -> List[Dict[str, Any]]:
    return services.store.list_assets(site_id)


@router.get("/api/assets/{asset_id}")
def get_asset(asset_id: str, _: Dict = Depends(any_user), services: Services = Depends(get_services)) -> Dict[str, Any]:
    a = services.store.get_asset(asset_id)
    if a is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown asset {asset_id}")
    return a


@router.get("/api/assets/{asset_id}/history")
def history(asset_id: str, _: Dict = Depends(any_user), services: Services = Depends(get_services)) -> Dict[str, Any]:
    h = services.engine.get_asset_history(asset_id)
    if not h["found"]:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown asset {asset_id}")
    return h


@router.get("/api/images/{ref:path}")
def image(ref: str, request: Request, token: Optional[str] = Query(None), services: Services = Depends(get_services)):
    """Stored crops / heatmaps. <img> tags cannot send headers, so `?token=` is accepted too."""
    bearer = request.headers.get("authorization", "")
    raw = bearer[7:] if bearer.lower().startswith("bearer ") else token
    if not raw:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated")
    user_from_token(services, raw)
    root = services.store.images_dir.resolve()
    path = (root / ref).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Image not found")
    return FileResponse(path)


async def _read_image(file: UploadFile, services: Services) -> Image.Image:
    limit = int(cfg(services.config, "api", "max_upload_mb")) * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Image too large")
    try:
        with Image.open(io.BytesIO(data)) as im:
            return im.convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Not an image") from exc


@router.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    asset_type: str = Form(...),
    session_id: Optional[str] = Form(None),
    add_to_memory: bool = Form(False),
    _: Dict = Depends(field_roles),
    services: Services = Depends(get_services),
) -> Dict[str, Any]:
    """Classify one close-up asset image. Only `add_to_memory=true` (with an active session) writes to memory (D6, D7)."""
    if not services.classifier.supports(asset_type):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Unsupported asset_type {asset_type}")
    if add_to_memory:
        s = services.store.get_session(session_id) if session_id else None
        if s is None or s["status"] != "active":
            raise HTTPException(status.HTTP_409_CONFLICT, "add_to_memory requires an active session_id")
    img = await _read_image(file, services)
    ev = services.engine.observe(img, asset_type, session_id, persist=add_to_memory)
    if ev.type == "error":
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, ev.error or "analysis failed")
    return ev.to_dict()


@router.post("/api/memory/search")
async def memory_search(
    file: UploadFile = File(...),
    asset_type: str = Form(...),
    site_id: Optional[str] = Form(None),
    top_k: int = Form(5),
    _: Dict = Depends(any_user),
    services: Services = Depends(get_services),
) -> List[Dict[str, Any]]:
    """Visually most similar stored assets (global embedding only — identity uses the full cascade)."""
    img = await _read_image(file, services)
    vec = services.embedder.encode_image(img)
    site = site_id or cfg(services.config, "sites", "default_site_id")
    out = []
    for c in services.engine.index.search(vec, site, asset_type, max(1, min(top_k, 50))):
        asset = services.store.get_asset(c.asset_id)
        if asset:
            out.append({"asset": asset, "similarity": c.similarity})
    return out
