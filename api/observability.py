"""Observability and request hardening for the edge API (upgrades 6.5, 6.6).

  Metrics       a dependency-free Prometheus registry (counters, gauges, histograms rendered
                in text exposition format 0.0.4) served at GET /metrics. Request latency per
                route, live pipeline fps / latency / counters, outbox queue depth, quality-gate
                rejections, active sessions. Scrapers authenticate with the bearer token in
                observability.metrics_token_env, or an admin JWT.
  Logs          observability.json_logs → one JSON object per log line (ts, level, logger,
                msg, plus request id / route / status / duration for access logs).
  Rate limiting token bucket per client (JWT subject, drone key prefix or IP):
                security.rate_limit_per_minute sustained, security.rate_limit_burst burst;
                429 with Retry-After when exhausted.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
import uuid
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from ml.config import get_nested

logger = logging.getLogger("audi.access")

DEFAULT_BUCKETS = (5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000)


def _labels(labels: Dict[str, str]) -> str:
    if not labels:
        return ""
    esc = lambda v: str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return "{" + ",".join(f'{k}="{esc(v)}"' for k, v in sorted(labels.items())) + "}"


class Registry:
    def __init__(self):
        self._lock = threading.Lock()
        self.counters: Dict[Tuple[str, str], float] = defaultdict(float)
        self.gauges: Dict[Tuple[str, str], float] = {}
        self.hist: Dict[Tuple[str, str], List[float]] = {}
        self.help: Dict[str, Tuple[str, str]] = {}
        self.collectors: List[Callable[["Registry"], None]] = []

    def describe(self, name: str, kind: str, text: str) -> None:
        self.help[name] = (kind, text)

    def inc(self, name: str, value: float = 1.0, **labels) -> None:
        with self._lock:
            self.counters[(name, _labels(labels))] += value

    def set(self, name: str, value: float, **labels) -> None:
        with self._lock:
            self.gauges[(name, _labels(labels))] = float(value)

    def observe(self, name: str, value: float, **labels) -> None:
        key = (name, _labels(labels))
        with self._lock:
            h = self.hist.setdefault(key, [0.0] * (len(DEFAULT_BUCKETS) + 2))  # buckets…, +Inf count, sum
            for i, b in enumerate(DEFAULT_BUCKETS):
                if value <= b:
                    h[i] += 1
            h[len(DEFAULT_BUCKETS)] += 1
            h[len(DEFAULT_BUCKETS) + 1] += value

    def render(self) -> str:
        for c in list(self.collectors):
            try:
                c(self)
            except Exception:  # noqa: BLE001 - a broken collector must not break /metrics
                logging.getLogger(__name__).exception("metrics collector failed")
        out: List[str] = []
        with self._lock:
            names = sorted({k[0] for k in self.counters} | {k[0] for k in self.gauges} | {k[0] for k in self.hist})
            for name in names:
                kind, text = self.help.get(name, ("untyped", name))
                out.append(f"# HELP {name} {text}")
                out.append(f"# TYPE {name} {kind}")
                for (n, lab), v in sorted(self.counters.items()):
                    if n == name:
                        out.append(f"{n}{lab} {v:g}")
                for (n, lab), v in sorted(self.gauges.items()):
                    if n == name:
                        out.append(f"{n}{lab} {v:g}" if math.isfinite(v) else f"{n}{lab} NaN")
                for (n, lab), h in sorted(self.hist.items()):
                    if n != name:
                        continue
                    inner = lab[1:-1] if lab else ""
                    for i, b in enumerate(DEFAULT_BUCKETS):
                        out.append(f'{n}_bucket{{{inner + "," if inner else ""}le="{b}"}} {h[i]:g}')
                    out.append(f'{n}_bucket{{{inner + "," if inner else ""}le="+Inf"}} {h[len(DEFAULT_BUCKETS)]:g}')
                    out.append(f"{n}_count{lab} {h[len(DEFAULT_BUCKETS)]:g}")
                    out.append(f"{n}_sum{lab} {h[len(DEFAULT_BUCKETS) + 1]:g}")
        return "\n".join(out) + "\n"


def install_default_collectors(reg: Registry, services) -> None:
    reg.describe("audi_http_requests_total", "counter", "HTTP requests by route, method and status")
    reg.describe("audi_http_request_duration_ms", "histogram", "HTTP request duration in milliseconds")
    reg.describe("audi_outbox_pending", "gauge", "Outbox records waiting for cloud sync")
    reg.describe("audi_outbox_sent_total", "gauge", "Outbox records acknowledged by the cloud")
    reg.describe("audi_cloud_online", "gauge", "1 when the cloud was reachable at the last check")
    reg.describe("audi_live_running", "gauge", "1 while a live session is running")
    reg.describe("audi_live_processed_fps", "gauge", "Frames processed per second by the live pipeline")
    reg.describe("audi_live_capture_fps", "gauge", "Frames captured per second")
    reg.describe("audi_live_latency_p95_ms", "gauge", "p95 latency per live pipeline stage")
    reg.describe("audi_live_frames", "gauge", "Live pipeline frame counters (seen, processed, skipped, rejected, alerts)")
    reg.describe("audi_quality_gate", "gauge", "Quality-gate decisions by reason since start")
    reg.describe("audi_assets", "gauge", "Assets in memory")
    reg.describe("audi_active_sessions", "gauge", "Inspection sessions currently active")
    reg.describe("audi_rate_limited_total", "counter", "Requests rejected by the rate limiter")

    def collect(r: Registry) -> None:
        counts = services.store.outbox_counts()
        r.set("audi_outbox_pending", counts["pending"])
        r.set("audi_outbox_sent_total", counts["sent"])
        r.set("audi_cloud_online", 1.0 if services.sync.online else 0.0)
        live = services.live.status() if services.live is not None else {"running": False}
        r.set("audi_live_running", 1.0 if live.get("running") else 0.0)
        if live.get("running"):
            r.set("audi_live_processed_fps", live.get("processed_fps") or 0.0)
            r.set("audi_live_capture_fps", live.get("capture_fps") or 0.0)
            for stage, st in (live.get("latency_ms") or {}).items():
                if st.get("p95") is not None:
                    r.set("audi_live_latency_p95_ms", st["p95"], stage=stage)
            for k, v in (live.get("counters") or {}).items():
                r.set("audi_live_frames", v, counter=k)
        for reason, n in services.engine.quality.counters.items():
            r.set("audi_quality_gate", n, reason=reason)
        s = services.store.summary()
        r.set("audi_assets", s["total_assets"])
        r.set("audi_active_sessions", len([x for x in services.store.list_sessions() if x["status"] == "active"]))

    reg.collectors.append(collect)


class TokenBucket:
    def __init__(self, per_minute: float, burst: float):
        self.rate = per_minute / 60.0
        self.burst = burst
        self._state: Dict[str, Tuple[float, float]] = {}
        self._lock = threading.Lock()

    def take(self, key: str) -> Tuple[bool, float]:
        now = time.monotonic()
        with self._lock:
            tokens, last = self._state.get(key, (self.burst, now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            if tokens >= 1.0:
                self._state[key] = (tokens - 1.0, now)
                return True, 0.0
            self._state[key] = (tokens, now)
            return False, (1.0 - tokens) / self.rate if self.rate > 0 else 60.0


def client_key(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer ") and len(auth) > 20:
        return "tok:" + auth[-24:]
    key = request.headers.get("x-audi-key")
    if key:
        return "drone:" + key[:12]
    return "ip:" + (request.client.host if request.client else "unknown")


class ObservabilityMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, registry: Registry, limiter: Optional[TokenBucket], exempt: Iterable[str] = ("/api/health", "/metrics")):
        super().__init__(app)
        self.registry = registry
        self.limiter = limiter
        self.exempt = set(exempt)

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        if self.limiter is not None and path not in self.exempt and not path.startswith("/ws/"):
            ok, wait = self.limiter.take(client_key(request))
            if not ok:
                self.registry.inc("audi_rate_limited_total")
                return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429,
                                    headers={"Retry-After": str(max(1, int(math.ceil(wait)))), "X-Request-ID": rid})
        t0 = time.perf_counter()
        status = 500
        try:
            response: Response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = rid
            return response
        finally:
            ms = (time.perf_counter() - t0) * 1000
            route = request.scope.get("route")
            tmpl = getattr(route, "path", None) or ("/api/images/*" if path.startswith("/api/images/") else "unmatched")
            self.registry.inc("audi_http_requests_total", route=tmpl, method=request.method, status=str(status))
            self.registry.observe("audi_http_request_duration_ms", ms, route=tmpl)
            logger.info("request", extra={"request_id": rid, "route": tmpl, "method": request.method, "status": status,
                                          "duration_ms": round(ms, 2)})


class JsonFormatter(logging.Formatter):
    STD = set(vars(logging.LogRecord("x", 0, "", 0, "", None, None)).keys()) | {"message", "asctime"}

    def format(self, record: logging.LogRecord) -> str:
        d: Dict[str, Any] = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
                             "level": record.levelname, "logger": record.name, "msg": record.getMessage()}
        for k, v in record.__dict__.items():
            if k not in self.STD and not k.startswith("_"):
                d[k] = v
        if record.exc_info:
            d["exc"] = self.formatException(record.exc_info)
        return json.dumps(d, default=str)


def configure_logging(config: Dict[str, Any]) -> None:
    if not get_nested(config, "observability", "json_logs", default=False):
        return
    root = logging.getLogger()
    h = logging.StreamHandler()
    h.setFormatter(JsonFormatter())
    root.handlers = [h]
    root.setLevel(get_nested(config, "logging", "level", default="INFO"))
