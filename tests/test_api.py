"""API: JWT auth + role guards on every route (D14), sessions/report (FR-4/18), D6, D16,
defect workflow (FR-17), admin config validation + versioning, FR-19 notifications over WS."""

import io
import time

import pytest
from fastapi.routing import APIRoute, APIWebSocketRoute
from fastapi.testclient import TestClient

from api.app import create_app
from api.services import build_services
from auth.security import hash_password
from ml.detector import FullFrameDetector
from tests.conftest import add_crack, make_panel

PASSWORDS = {"admin": "admin-pass-1", "operator": "operator-pass-1", "maintenance": "maint-pass-1", "other_maint": "maint-pass-2"}
PUBLIC = {("POST", "/api/auth/token"), ("GET", "/api/health")}


def jpeg(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    return buf.getvalue()


@pytest.fixture
def stack(config, embedder, classifier):
    svc = build_services(config, embedder=embedder, classifier=classifier,
                         detector=FullFrameDetector("solar_panel", 0.95), use_cloud=False)
    site = config["sites"]["default_site_id"]
    svc.store.ensure_site("site_other", "Other site")
    for name, role in (("admin", "admin"), ("operator", "operator"), ("maintenance", "maintenance"), ("other_maint", "maintenance")):
        u = svc.store.create_user(name, hash_password(PASSWORDS[name]), role)
        if name in {"operator", "maintenance"}:
            svc.store.set_subscriptions(u["user_id"], [site])
        if name == "other_maint":
            svc.store.set_subscriptions(u["user_id"], ["site_other"])
    app = create_app(config, services=svc, background_sync=False)
    with TestClient(app) as client:
        yield client, svc


def login(client, name):
    r = client.post("/api/auth/token", data={"username": name, "password": PASSWORDS[name]})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def H(token):
    return {"Authorization": f"Bearer {token}"}


def fill_path(path):
    return (path.replace("{session_id}", "sess_x").replace("{asset_id}", "asset_x").replace("{defect_id}", "def_x")
            .replace("{notification_id}", "n_x").replace("{user_id}", "1").replace("{site_id}", "site_demo")
            .replace("{ref:path}", "a/b.jpg"))


def all_routes():
    from api.routers import admin, assets, auth, defects, live, notifications, sessions

    return [r for m in (auth, sessions, assets, defects, notifications, admin, live) for r in m.router.routes]


def test_every_route_requires_auth(stack):
    client, _ = stack
    checked = 0
    for route in all_routes():
        if isinstance(route, APIRoute):
            for method in route.methods:
                if (method, route.path) in PUBLIC:
                    continue
                r = client.request(method, fill_path(route.path), params={"token": "not-a-jwt"})
                assert r.status_code == 401, f"{method} {route.path} → {r.status_code}"
                checked += 1
        elif isinstance(route, APIWebSocketRoute):
            with pytest.raises(Exception):
                with client.websocket_connect(f"{route.path}?token=bad") as ws:
                    ws.receive_json()
            checked += 1
    assert checked >= 30


def test_bad_password_rejected(stack):
    client, _ = stack
    assert client.post("/api/auth/token", data={"username": "admin", "password": "wrong"}).status_code == 401


def test_role_guards(stack):
    client, _ = stack
    op, mt, ad = login(client, "operator"), login(client, "maintenance"), login(client, "admin")
    assert client.post("/api/sessions/start", json={"site_id": "site_demo"}, headers=H(mt)).status_code == 403
    assert client.patch("/api/defects/def_x", json={"status": "Resolved"}, headers=H(op)).status_code == 403
    assert client.get("/api/admin/config", headers=H(op)).status_code == 403
    assert client.get("/api/admin/users", headers=H(mt)).status_code == 403
    assert client.post("/api/live/start", json={"session_id": "s"}, headers=H(mt)).status_code == 403
    assert client.get("/api/admin/config", headers=H(ad)).status_code == 200
    me = client.get("/api/auth/me", headers=H(mt)).json()
    assert me["role"] == "maintenance" and me["sites"] == ["site_demo"] and "password_hash" not in me


def test_cors_restricted_to_config_origins(stack, config):
    client, _ = stack
    allowed = config["api"]["cors_origins"][0]
    ok = client.options("/api/sites", headers={"Origin": allowed, "Access-Control-Request-Method": "GET"})
    assert ok.headers.get("access-control-allow-origin") == allowed
    bad = client.options("/api/sites", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "GET"})
    assert "access-control-allow-origin" not in bad.headers


def test_session_flow_d6_d16_report(stack):
    client, svc = stack
    op = login(client, "operator")
    start = client.post("/api/sessions/start", json={"site_id": "site_demo"}, headers=H(op)).json()
    sid = start["session"]["session_id"]
    assert start["sync"]["mode"] == "offline" and "assets_due" in start["briefing"]

    panel = make_panel(41)
    # D6: analysis without add_to_memory never writes
    r = client.post("/api/analyze", files={"file": ("p.jpg", jpeg(panel), "image/jpeg")},
                    data={"asset_type": "solar_panel", "session_id": sid, "add_to_memory": "false"}, headers=H(op))
    assert r.status_code == 200 and r.json()["persisted"] is False
    assert svc.store.list_assets("site_demo") == []
    # add_to_memory requires an active session (D7)
    r = client.post("/api/analyze", files={"file": ("p.jpg", jpeg(panel), "image/jpeg")},
                    data={"asset_type": "solar_panel", "add_to_memory": "true"}, headers=H(op))
    assert r.status_code == 409
    r = client.post("/api/analyze", files={"file": ("p.jpg", jpeg(panel), "image/jpeg")},
                    data={"asset_type": "solar_panel", "session_id": sid, "add_to_memory": "true"}, headers=H(op))
    ev = r.json()
    assert ev["persisted"] and ev["type"] == "new_asset"
    asset_id = ev["asset_id"]

    hist = client.get(f"/api/assets/{asset_id}/history", headers=H(op)).json()
    assert hist["found"] and len(hist["observations"]) == 1
    ref = hist["observations"][0]["image_reference"]
    assert client.get(f"/api/images/{ref}", params={"token": op}).status_code == 200
    assert client.get("/api/images/../../config.yaml", params={"token": op}).status_code == 404

    report = client.post(f"/api/sessions/{sid}/end", headers=H(op)).json()
    assert report["counts"]["new_assets"] == 1
    assert client.post(f"/api/sessions/{sid}/end", headers=H(op)).status_code == 409
    html = client.get(f"/api/sessions/{sid}/report", params={"format": "html"}, headers=H(op))
    assert html.status_code == 200 and "<html" in html.text.lower()
    pdf = client.get(f"/api/sessions/{sid}/report", params={"format": "pdf"}, headers=H(op))
    assert pdf.content[:4] == b"%PDF"

    # D16: missions counted from the sessions table
    client.post("/api/sessions/start", json={"site_id": "site_demo"}, headers=H(op))
    summary = client.get("/api/summary", params={"site_id": "site_demo"}, headers=H(op)).json()
    assert summary["missions"] == 2 == len(client.get("/api/sessions", headers=H(op)).json())
    assert summary["total_assets"] == 1


def test_memory_search_returns_registered_asset(stack):
    client, _ = stack
    op = login(client, "operator")
    sid = client.post("/api/sessions/start", json={"site_id": "site_demo"}, headers=H(op)).json()["session"]["session_id"]
    panel = make_panel(77)
    client.post("/api/analyze", files={"file": ("p.jpg", jpeg(panel), "image/jpeg")},
                data={"asset_type": "solar_panel", "session_id": sid, "add_to_memory": "true"}, headers=H(op))
    res = client.post("/api/memory/search", files={"file": ("p.jpg", jpeg(panel), "image/jpeg")},
                      data={"asset_type": "solar_panel", "site_id": "site_demo"}, headers=H(op)).json()
    assert res and res[0]["similarity"] > 0.99


def test_crash_recovery_endpoints(stack):
    client, _ = stack
    op = login(client, "operator")
    sid = client.post("/api/sessions/start", json={"site_id": "site_demo"}, headers=H(op)).json()["session"]["session_id"]
    assert sid in [s["session_id"] for s in client.get("/api/sessions/interrupted", headers=H(op)).json()]
    assert client.post(f"/api/sessions/{sid}/resume", headers=H(op)).json()["status"] == "active"


def test_admin_config_validated_and_versioned(stack, config, embedder, classifier):
    client, svc = stack
    ad = login(client, "admin")
    bad = client.put("/api/admin/config", json={"changes": {"severity": {"w1": 0.7}}}, headers=H(ad))
    assert bad.status_code == 422 and "sum to 1" in bad.json()["detail"]
    bad = client.put("/api/admin/config", json={"changes": {"severity": {"levels": {"low": 0.0, "medium": 0.6, "priority": 0.5, "critical": 0.8}}}}, headers=H(ad))
    assert bad.status_code == 422
    ok = client.put("/api/admin/config", json={"changes": {"severity": {"w1": 0.7, "w2": 0.3}, "asset_criticality": {"solar_panel": 0.95}}}, headers=H(ad))
    assert ok.status_code == 200 and ok.json()["version"] == 1
    assert svc.config["severity"]["w1"] == 0.7 and svc.config["asset_criticality"]["solar_panel"] == 0.95
    versions = client.get("/api/admin/config/versions", headers=H(ad)).json()
    assert versions[0]["username"] == "admin" and versions[0]["changes"]["severity"]["w1"] == 0.7

    # a restarted API applies the latest stored version over config.yaml
    import copy
    fresh_cfg = copy.deepcopy(config)
    svc2 = build_services(fresh_cfg, store=svc.store, embedder=embedder, classifier=classifier,
                          detector=FullFrameDetector("solar_panel", 0.95), use_cloud=False)
    create_app(fresh_cfg, services=svc2, background_sync=False)
    assert svc2.config["severity"]["w1"] == 0.7


def test_admin_users_and_sites(stack):
    client, svc = stack
    ad = login(client, "admin")
    assert client.post("/api/admin/sites", json={"site_id": "bad id!", "name": "x"}, headers=H(ad)).status_code == 422
    assert client.post("/api/admin/sites", json={"site_id": "site_new", "name": "New"}, headers=H(ad)).status_code == 201
    r = client.post("/api/admin/users", json={"username": "tech2", "password": "longpassword", "role": "maintenance", "sites": ["site_new"]}, headers=H(ad))
    assert r.status_code == 201 and "password_hash" not in r.json()
    uid = r.json()["user_id"]
    assert client.patch(f"/api/admin/users/{uid}", json={"active": False}, headers=H(ad)).json()["active"] is False
    assert client.post("/api/auth/token", data={"username": "tech2", "password": "longpassword"}).status_code == 401
    me = next(u for u in client.get("/api/admin/users", headers=H(ad)).json() if u["username"] == "admin")
    assert client.patch(f"/api/admin/users/{me['user_id']}", json={"role": "operator"}, headers=H(ad)).status_code == 409
    assert client.get("/api/admin/sync", headers=H(ad)).json()["enabled"] is False
    assert "evaluation" in client.get("/api/admin/metrics", headers=H(ad)).json()


def test_fr19_critical_notifies_only_subscribed_maintenance_over_ws(stack):
    client, svc = stack
    svc.config["severity"]["levels"]["critical"] = 0.3
    op, mt, other = login(client, "operator"), login(client, "maintenance"), login(client, "other_maint")
    sid = client.post("/api/sessions/start", json={"site_id": "site_demo"}, headers=H(op)).json()["session"]["session_id"]
    with client.websocket_connect(f"/ws/notifications?token={mt}") as ws:
        assert ws.receive_json()["type"] == "hello"
        ev = client.post("/api/analyze", files={"file": ("p.jpg", jpeg(add_crack(make_panel(5))), "image/jpeg")},
                         data={"asset_type": "solar_panel", "session_id": sid, "add_to_memory": "true"}, headers=H(op)).json()
        assert ev["critical"], ev
        msg = ws.receive_json()
        assert msg["type"] == "notification" and msg["notification"]["asset_id"] == ev["asset_id"]
    mine = client.get("/api/notifications", headers=H(mt)).json()
    assert len(mine) == 1 and client.get("/api/notifications", headers=H(other)).json() == []
    assert client.get("/api/notifications", headers=H(op)).json() == []
    nid = mine[0]["notification_id"]
    assert client.post(f"/api/notifications/{nid}/read", headers=H(other)).status_code == 404
    assert client.post(f"/api/notifications/{nid}/read", headers=H(mt)).json()["read"] is True

    defects = client.get("/api/defects", params={"site_id": "site_demo"}, headers=H(mt)).json()
    assert defects and defects[0]["status"] == "Open"
    did = defects[0]["defect_id"]
    assert client.patch(f"/api/defects/{did}", json={"status": "Bogus"}, headers=H(mt)).status_code == 400
    assert client.patch(f"/api/defects/{did}", json={"status": "In Progress"}, headers=H(mt)).json()["status"] == "In Progress"
    detail = client.get(f"/api/defects/{did}", headers=H(mt)).json()
    assert detail["trend"], "defect trend must be recorded (FR-17)"
    report = client.post(f"/api/sessions/{sid}/end", headers=H(op)).json()
    assert [e["username"] for e in report["escalations"]] == ["maintenance"]


def test_live_browser_mode_over_websocket(stack):
    client, svc = stack
    op = login(client, "operator")
    sid = client.post("/api/sessions/start", json={"site_id": "site_demo"}, headers=H(op)).json()["session"]["session_id"]
    svc.config["camera"]["frame_skip"] = 1
    with client.websocket_connect(f"/ws/live?token={op}") as ws:
        r = client.post("/api/live/start", json={"session_id": sid, "mode": "browser", "detection_mode": "full_frame", "asset_type": "solar_panel"}, headers=H(op))
        assert r.status_code == 200, r.text
        assert client.post("/api/live/start", json={"session_id": sid, "mode": "browser"}, headers=H(op)).status_code == 409
        ws.send_bytes(jpeg(make_panel(9)))
        deadline = time.time() + 20
        event = None
        while time.time() < deadline:
            msg = ws.receive_json()
            if msg["type"] == "event":
                event = msg["event"]
                break
        assert event is not None and event["type"] == "new_asset" and event["persisted"]
        st = client.get("/api/live/status", headers=H(op)).json()
        assert st["running"] and st["counters"]["processed"] >= 1 and st["latency_ms"]["capture_to_result"]["n"] >= 1
        assert client.post("/api/live/capture", json={}, headers=H(op)).status_code == 200
        stopped = client.post("/api/live/stop", headers=H(op)).json()
        assert stopped["running"] is False
    assert client.get("/api/live/status", headers=H(op)).json() == {"running": False}
