"""FastAPI application factory for the AUDI field/edge API."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ml.config import cfg, load_config
from ml.live.manager import LiveManager

from api.routers import admin, assets, auth, defects, live, notifications, sessions
from api.services import Services, build_services

logger = logging.getLogger(__name__)


def create_app(config: Optional[Dict[str, Any]] = None, services: Optional[Services] = None, background_sync: bool = True) -> FastAPI:
    config = config if config is not None else load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        svc = app.state.services
        svc.hub.bind_loop(asyncio.get_running_loop())
        if background_sync:
            svc.sync.start_background()
        integrity = svc.store.integrity
        if integrity["active_sessions"]:
            logger.warning("Unfinished session(s) from a previous run: %s — resume or end them", integrity["active_sessions"])
        try:
            yield
        finally:
            if svc.live is not None:
                svc.live.stop()
            svc.sync.stop_background()

    app = FastAPI(title="AUDI API", version="2.0", lifespan=lifespan)
    svc = services or build_services(config)
    if svc.live is None:
        svc.live = LiveManager(svc.config, svc.engine, svc.store, lambda m: svc.hub.publish("live", m))
    version = admin.load_saved_overrides(svc)
    if version is not None:
        logger.info("Applied admin configuration version %d", version)
    app.state.services = svc

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cfg(config, "api", "cors_origins")),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )
    for r in (auth, sessions, assets, defects, notifications, admin, live):
        app.include_router(r.router)
    return app
