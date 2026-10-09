"""Defect log with trend (FR-17) and maintenance status workflow."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from memory.db import DEFECT_STATUSES

from api.deps import any_user, get_services, maintenance_only
from api.services import Services

router = APIRouter(prefix="/api/defects", tags=["defects"])


class StatusUpdate(BaseModel):
    status: str
    notes: Optional[str] = None


@router.get("")
def list_defects(
    site_id: Optional[str] = None,
    asset_id: Optional[str] = None,
    status_filter: Optional[str] = Query(None, alias="status"),
    _: Dict = Depends(any_user),
    services: Services = Depends(get_services),
) -> List[Dict[str, Any]]:
    return services.store.list_defects(site_id=site_id, asset_id=asset_id, status=status_filter)


@router.get("/{defect_id}")
def get_defect(defect_id: str, _: Dict = Depends(any_user), services: Services = Depends(get_services)) -> Dict[str, Any]:
    d = services.store.get_defect(defect_id)
    if d is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown defect {defect_id}")
    return d


@router.patch("/{defect_id}")
def update_defect(defect_id: str, body: StatusUpdate, _: Dict = Depends(maintenance_only), services: Services = Depends(get_services)) -> Dict[str, Any]:
    if body.status not in DEFECT_STATUSES:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"status must be one of {DEFECT_STATUSES}")
    try:
        return services.store.update_defect_status(defect_id, body.status, body.notes)
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown defect {defect_id}") from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
