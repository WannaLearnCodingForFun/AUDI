"""Closing the loop: work orders (6.2), inspector feedback (5.4), owner reports (6.4)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response
from pydantic import BaseModel

from memory.db import FEEDBACK_LABELS, WORK_ORDER_STATUSES
from ml.owner_report import build_owner_report, render_owner_pdf

from api.deps import any_user, field_roles, get_services, maintenance_only
from api.services import Services

router = APIRouter(tags=["maintenance"])


class WorkOrderIn(BaseModel):
    asset_id: str
    defect_id: Optional[str] = None
    title: str
    priority: str = "priority"


class WorkOrderUpdate(BaseModel):
    status: str
    assignee: Optional[str] = None
    note: Optional[str] = None


class FeedbackIn(BaseModel):
    label: str
    correct_asset_id: Optional[str] = None
    correct_condition: Optional[str] = None
    notes: Optional[str] = None


@router.get("/api/work-orders")
def list_work_orders(site_id: Optional[str] = None, status_: Optional[str] = Query(None, alias="status"), asset_id: Optional[str] = None,
                     _: Dict = Depends(any_user), services: Services = Depends(get_services)) -> List[Dict[str, Any]]:
    return services.store.list_work_orders(site_id, status_, asset_id)


@router.post("/api/work-orders", status_code=201)
def create_work_order(body: WorkOrderIn, user: Dict = Depends(maintenance_only), services: Services = Depends(get_services)) -> Dict[str, Any]:
    try:
        return services.store.create_work_order(body.asset_id, body.defect_id, body.title, body.priority, user["username"])
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown asset {body.asset_id}") from exc


@router.get("/api/work-orders/{work_order_id}")
def get_work_order(work_order_id: str, _: Dict = Depends(any_user), services: Services = Depends(get_services)) -> Dict[str, Any]:
    w = services.store.get_work_order(work_order_id)
    if w is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown work order {work_order_id}")
    return w


@router.patch("/api/work-orders/{work_order_id}")
def update_work_order(work_order_id: str, body: WorkOrderUpdate, user: Dict = Depends(maintenance_only),
                      services: Services = Depends(get_services)) -> Dict[str, Any]:
    if body.status not in WORK_ORDER_STATUSES:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"status must be one of {WORK_ORDER_STATUSES}")
    try:
        return services.store.update_work_order(work_order_id, body.status, user["username"], body.assignee, body.note)
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown work order {work_order_id}") from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc


@router.post("/api/observations/{observation_id}/feedback", status_code=201)
def feedback(observation_id: str, body: FeedbackIn, user: Dict = Depends(field_roles), services: Services = Depends(get_services)) -> Dict[str, Any]:
    if body.label not in FEEDBACK_LABELS:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"label must be one of {FEEDBACK_LABELS}")
    try:
        return services.store.add_feedback(observation_id, user["username"], body.label, body.correct_asset_id,
                                           body.correct_condition, body.notes)
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown observation {observation_id}") from exc


@router.get("/api/feedback")
def list_feedback(site_id: Optional[str] = None, _: Dict = Depends(any_user), services: Services = Depends(get_services)) -> List[Dict[str, Any]]:
    return services.store.list_feedback(site_id)


@router.get("/api/feedback/stats")
def feedback_stats(site_id: Optional[str] = None, _: Dict = Depends(any_user), services: Services = Depends(get_services)) -> Dict[str, Any]:
    return services.store.feedback_stats(site_id)


@router.get("/api/sites/{site_id}/owner-report")
def owner_report(site_id: str, month: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}$"), format: str = Query("json", pattern="^(json|pdf)$"),
                 _: Dict = Depends(any_user), services: Services = Depends(get_services)):
    if services.store.get_site(site_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown site {site_id}")
    month = month or datetime.now(timezone.utc).strftime("%Y-%m")
    rep = build_owner_report(services.store, site_id, month, services.config)
    if format == "pdf":
        return Response(render_owner_pdf(rep), media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="audi_owner_{site_id}_{month}.pdf"'})
    return rep
