"""Admin: scoring configuration (validated, versioned), sites, users, sync, model metrics."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from auth.security import hash_password
from memory.db import ROLES
from ml.change_detector import ConditionVisualChangeDetector
from ml.condition_mapping import UNIFIED_CONDITIONS
from ml.config import cfg, resolve_path
from ml.detector import build_detector

from api.admin_config import ConfigValidationError, apply_view, editable_view, merge_and_validate
from api.deps import admin_only, get_services
from api.services import Services

router = APIRouter(prefix="/api/admin", tags=["admin"])

SITE_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def apply_to_services(services: Services, view: Dict[str, Any]) -> None:
    prompts_changed = view["detection_prompts"] != editable_view(services.config)["detection_prompts"]
    apply_view(services.config, view)
    engine = services.engine
    engine.change_detector = ConditionVisualChangeDetector(services.config, services.embedder)
    engine.resolution_missions = int(view["resolution_suggestion_missions"])
    if prompts_changed and cfg(services.config, "detection", "mode") == "yolo_world":
        detector = build_detector(services.config)
        services.detector = detector
        engine.detector = detector


def load_saved_overrides(services: Services) -> Optional[int]:
    """Apply the latest stored config version at startup; returns its version number."""
    versions = services.store.list_config_versions(include_snapshot=True)
    if not versions:
        return None
    latest = versions[0]
    view = merge_and_validate(services.config, latest["snapshot"], list(UNIFIED_CONDITIONS))
    apply_to_services(services, view)
    return latest["version"]


class ConfigChange(BaseModel):
    changes: Dict[str, Any]


@router.get("/config")
def get_config(_: Dict = Depends(admin_only), services: Services = Depends(get_services)) -> Dict[str, Any]:
    versions = services.store.list_config_versions()
    return {"config": editable_view(services.config), "version": versions[0]["version"] if versions else None}


@router.put("/config")
def update_config(body: ConfigChange, user: Dict = Depends(admin_only), services: Services = Depends(get_services)) -> Dict[str, Any]:
    try:
        view = merge_and_validate(services.config, body.changes, list(UNIFIED_CONDITIONS))
    except ConfigValidationError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    try:
        apply_to_services(services, view)
    except Exception as exc:  # noqa: BLE001 - e.g. detector rebuild failure; nothing is versioned then
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Could not apply configuration: {exc}") from exc
    version = services.store.record_config_version(user["username"], body.changes, view)
    return {"config": view, "version": version["version"]}


@router.get("/config/versions")
def config_versions(_: Dict = Depends(admin_only), services: Services = Depends(get_services)) -> List[Dict[str, Any]]:
    return services.store.list_config_versions()


class SiteCreate(BaseModel):
    site_id: str
    name: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None


@router.post("/sites", status_code=201)
def create_site(body: SiteCreate, _: Dict = Depends(admin_only), services: Services = Depends(get_services)) -> Dict[str, Any]:
    if not SITE_ID_RE.match(body.site_id):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "site_id: letters, digits, _ and - only (max 64)")
    if services.store.get_site(body.site_id) is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "Site already exists")
    return services.store.ensure_site(body.site_id, body.name, body.latitude, body.longitude)


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=8)
    role: str
    email: Optional[str] = None
    sites: List[str] = []


class UserUpdate(BaseModel):
    role: Optional[str] = None
    email: Optional[str] = None
    active: Optional[bool] = None
    password: Optional[str] = Field(default=None, min_length=8)


class Subscriptions(BaseModel):
    sites: List[str]


@router.get("/users")
def list_users(_: Dict = Depends(admin_only), services: Services = Depends(get_services)) -> List[Dict[str, Any]]:
    return services.store.list_users()


@router.post("/users", status_code=201)
def create_user(body: UserCreate, _: Dict = Depends(admin_only), services: Services = Depends(get_services)) -> Dict[str, Any]:
    if body.role not in ROLES:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"role must be one of {ROLES}")
    if services.store.get_user(body.username) is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "Username taken")
    user = services.store.create_user(body.username, hash_password(body.password), body.role, body.email)
    if body.sites:
        try:
            services.store.set_subscriptions(user["user_id"], body.sites)
        except KeyError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    user.pop("password_hash", None)
    return {**user, "sites": body.sites}


@router.patch("/users/{user_id}")
def update_user(user_id: int, body: UserUpdate, admin: Dict = Depends(admin_only), services: Services = Depends(get_services)) -> Dict[str, Any]:
    fields = body.model_dump(exclude_none=True)
    if "password" in fields:
        fields["password_hash"] = hash_password(fields.pop("password"))
    if user_id == admin["user_id"] and (fields.get("active") is False or fields.get("role", "admin") != "admin"):
        raise HTTPException(status.HTTP_409_CONFLICT, "Admins cannot deactivate or demote themselves")
    try:
        return services.store.update_user(user_id, **fields)
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown user") from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc


@router.put("/users/{user_id}/sites")
def set_sites(user_id: int, body: Subscriptions, _: Dict = Depends(admin_only), services: Services = Depends(get_services)) -> Dict[str, Any]:
    if services.store.get_user_by_id(user_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown user")
    try:
        services.store.set_subscriptions(user_id, body.sites)
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return {"user_id": user_id, "sites": body.sites}


@router.get("/sync")
def sync_status(_: Dict = Depends(admin_only), services: Services = Depends(get_services)) -> Dict[str, Any]:
    return {**services.sync.status(), "outbox": services.store.outbox_counts()}


@router.post("/sync/flush")
def sync_flush(_: Dict = Depends(admin_only), services: Services = Depends(get_services)) -> Dict[str, Any]:
    result = services.sync.end_sync()
    return {**result, "outbox": services.store.outbox_counts()}


@router.get("/metrics")
def metrics(_: Dict = Depends(admin_only), services: Services = Depends(get_services)) -> Dict[str, Any]:
    """Measured model/dataset metrics from reports/ (nothing is shown if a report does not exist)."""
    def read_json(path) -> Optional[Dict[str, Any]]:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    eval_dir = resolve_path(cfg(services.config, "reports", "eval_dir"))
    ev = read_json(eval_dir / "metrics.json")
    train = read_json(resolve_path(cfg(services.config, "condition_classifier", "metrics_path")))
    stats = resolve_path(cfg(services.config, "datasets", "prepare", "output_dir")) / "STATS.md"
    bench = resolve_path(cfg(services.config, "reports", "dir")) / "bench.md"
    return {
        "evaluation": {"summary": ev["summary"], "split": ev["split"], "results": ev["results"]} if ev else None,
        "training": {k: train[k] for k in ("trained_at", "val_macro_f1", "val_per_type_macro_f1", "best_epoch", "counts", "fine_tune_blocks")} if train else None,
        "dataset_stats_md": stats.read_text(encoding="utf-8") if stats.exists() else None,
        "bench_md": bench.read_text(encoding="utf-8") if bench.exists() else None,
    }
