"""Per-user notification inbox (FR-19) and the authenticated WebSocket."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.exceptions import HTTPException as FastAPIHTTPException

from api.deps import any_user, get_services, user_from_token
from api.services import Services

logger = logging.getLogger(__name__)
router = APIRouter(tags=["notifications"])


@router.get("/api/notifications")
def list_notifications(unread_only: bool = False, user: Dict = Depends(any_user), services: Services = Depends(get_services)) -> List[Dict[str, Any]]:
    return services.store.list_notifications(user["user_id"], unread_only)


@router.post("/api/notifications/{notification_id}/read")
def mark_read(notification_id: str, user: Dict = Depends(any_user), services: Services = Depends(get_services)) -> Dict[str, Any]:
    try:
        return services.store.mark_notification_read(user["user_id"], notification_id)
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown notification") from exc


@router.websocket("/ws/notifications")
async def ws_notifications(ws: WebSocket) -> None:
    services: Services = ws.app.state.services
    try:
        user = user_from_token(services, ws.query_params.get("token", ""))
    except FastAPIHTTPException:
        await ws.close(code=4401)
        return
    await ws.accept()
    client = services.hub.register(ws, user["user_id"], user["role"], {"notifications"})
    pump = asyncio.create_task(services.hub.pump(client))
    try:
        unread = services.store.list_notifications(user["user_id"], unread_only=True)
        await ws.send_json({"type": "hello", "unread": len(unread)})
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        logger.debug("notification socket closed for %s", user["username"])
    finally:
        pump.cancel()
        services.hub.unregister(client)
