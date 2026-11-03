"""FastAPI application factory for the AUDI field/edge API."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

from ml.config import cfg, load_config
from ml.live.manager import LiveManager

from api.observability import ObservabilityMiddleware, Registry, TokenBucket, configure_logging, install_default_collectors
from api.routers import admin, assets, auth, defects, drones, live, maintenance, memory_ops, notifications, sessions, spatial
from api.services import Services, build_services

ROUTERS = (auth, sessions, assets, defects, notifications, admin, live, spatial, memory_ops, maintenance, drones)

logger = logging.getLogger(__name__)


def create_app(config: Optional[Dict[str, Any]] = None, services: Optional[Services] = None, background_sync: bool = True) -> FastAPI:
    config = config if config is not None else load_config()
    configure_logging(config)

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

    registry = Registry()
    install_default_collectors(registry, svc)
    app.state.metrics = registry
    sec = config.get("security") or {}
    per_min = float(sec.get("rate_limit_per_minute", 0) or 0)
    limiter = TokenBucket(per_min, float(sec.get("rate_limit_burst", 60))) if per_min > 0 else None
    app.add_middleware(ObservabilityMiddleware, registry=registry, limiter=limiter)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cfg(config, "api", "cors_origins")),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "X-AUDI-Key", "X-AUDI-Timestamp", "X-AUDI-Signature", "X-AUDI-Session", "X-AUDI-Pose"],
    )
    for r in ROUTERS:
        app.include_router(r.router)

    @app.get("/metrics", include_in_schema=False)
    def metrics(request: Request):
        """Prometheus scrape: bearer = observability.metrics_token_env value, or an admin JWT."""
        import os
        import secrets as _secrets

        from api.deps import user_from_token

        bearer = request.headers.get("authorization", "")
        token = bearer[7:] if bearer.lower().startswith("bearer ") else ""
        scrape = os.environ.get((config.get("observability") or {}).get("metrics_token_env", "AUDI_METRICS_TOKEN"))
        if not (scrape and token and _secrets.compare_digest(token, scrape)):
            if not token:
                raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated")
            user = user_from_token(svc, token)
            if user["role"] != "admin":
                raise HTTPException(status.HTTP_403_FORBIDDEN, "Requires admin")
        return PlainTextResponse(registry.render(), media_type="text/plain; version=0.0.4")

    return app
