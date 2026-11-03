"""API surface of the upgrades: merge/split (1.4), coverage (2.5), telemetry + guidance (2.4),
twin (4.4), route plan (4.5), graph propagation (5.1), feedback (5.4), health (5.5),
cloud image sync (6.1), work orders (6.2), drones + signed uploads (6.3/6.5),
rate limiting and encryption at rest (6.5), metrics (6.6), owner report (6.4)."""

import io
import json
import time

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.services import build_services
from auth.image_crypto import MAGIC, sign_payload
from auth.security import hash_password
from memory.store_ext import hash_api_key
from ml.detector import FullFrameDetector
from ml.schemas import utc_now
from tests.conftest import add_crack, make_panel

PW = {"admin": "admin-pass-1", "operator": "operator-pass-1", "maintenance": "maint-pass-1"}
SITE = "site_demo"


def jpeg(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    return buf.getvalue()


def make_stack(config, embedder, classifier, cloud=None):
    svc = build_services(config, embedder=embedder, classifier=classifier, detector=FullFrameDetector("solar_panel", 0.95),
                         cloud=cloud, use_cloud=cloud is not None)
    for name in PW:
        u = svc.store.create_user(name, hash_password(PW[name]), name)
        if name == "maintenance":
            svc.store.set_subscriptions(u["user_id"], [SITE])
    return svc


@pytest.fixture
def stack(config, embedder, classifier):
    config["geo"]["identity_mode"] = "appearance"
    svc = make_stack(config, embedder, classifier)
    with TestClient(create_app(config, services=svc, background_sync=False)) as client:
        yield client, svc


def H(client, name):
    r = client.post("/api/auth/token", data={"username": name, "password": PW[name]})
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def start(client, h):
    return client.post("/api/sessions/start", json={"site_id": SITE}, headers=h).json()["session"]["session_id"]


def observe(client, h, sid, img):
    r = client.post("/api/analyze", files={"file": ("p.jpg", jpeg(img), "image/jpeg")},
                    data={"asset_type": "solar_panel", "session_id": sid, "add_to_memory": "true"}, headers=h)
    assert r.status_code == 200, r.text
    return r.json()


def test_merge_split_and_audit_over_api(stack):
    client, svc = stack
    op, admin = H(client, "operator"), H(client, "admin")
    s1 = start(client, op)
    a = observe(client, op, s1, make_panel(51))
    b = observe(client, op, s1, make_panel(52))
    assert client.post("/api/assets/merge", json={"keep_id": a["asset_id"], "merge_id": b["asset_id"]}, headers=op).status_code == 403
    r = client.post("/api/assets/merge", json={"keep_id": a["asset_id"], "merge_id": b["asset_id"]}, headers=admin)
    assert r.status_code == 200 and r.json()["counts"]["observations"] == 1
    assert client.get(f"/api/assets/{b['asset_id']}", headers=admin).status_code == 404
    obs = client.get(f"/api/assets/{a['asset_id']}/history", headers=admin).json()["observations"]
    r = client.post(f"/api/assets/{a['asset_id']}/split", json={"observation_ids": [obs[-1]["observation_id"]]}, headers=admin)
    assert r.status_code == 200
    actions = [x["action"] for x in client.get("/api/audit", headers=admin).json()]
    assert actions[:2] == ["asset.split", "asset.merge"]


def test_coverage_twin_route_and_health(stack):
    client, svc = stack
    op = H(client, "operator")
    s1 = start(client, op)
    for i, seed in enumerate((61, 62, 63)):
        svc.engine.observe(make_panel(seed), "solar_panel", s1, location={"latitude": 47.0 + i * 1e-4, "longitude": 8.0})
    client.post(f"/api/sessions/{s1}/end", headers=op)
    s2 = start(client, op)
    seen = observe(client, op, s2, make_panel(61))
    cov = client.get(f"/api/sessions/{s2}/coverage", headers=op).json()
    assert cov["planned"] == 3 and cov["seen"] == 1 and len(cov["missing"]) == 2
    assert seen["asset_id"] not in [m["asset_id"] for m in cov["missing"]]
    unseen = client.get(f"/api/sites/{SITE}/unseen", headers=op).json()
    assert len(unseen["assets"]) == 3 and unseen["cells"]
    twin = client.get(f"/api/sites/{SITE}/twin", headers=op).json()
    assert len(twin["assets"]) == 3 and twin["origin"]
    r = client.post(f"/api/sites/{SITE}/route-plan", json={"min_priority": 0.0}, headers=op)
    plan = r.json()
    assert plan["advisory"] and len(plan["waypoints"]) == 3
    items = plan["plan"]["mission"]["items"]
    assert items[0]["command"] == 22 and items[-1]["command"] == 20
    f = client.post(f"/api/sites/{SITE}/route-plan?format=plan", json={"min_priority": 0.0}, headers=op)
    assert json.loads(f.text)["fileType"] == "Plan"
    h = client.get(f"/api/sites/{SITE}/health", headers=op).json()
    assert h["health_index"] is not None and len(h["assets"]) == 3


def test_telemetry_pose_reaches_live_pipeline_and_guidance(config, embedder, classifier):
    from ml.live.camera import Frame, PushFrameSource
    from ml.live.pipeline import LivePipeline

    import cv2
    import numpy as np

    svc = make_stack(config, embedder, classifier)
    svc.store.ensure_site(SITE, "Demo")
    sid = svc.store.create_session(SITE, "op", "offline", {})["session_id"]
    pipe = LivePipeline(svc.config, svc.engine, PushFrameSource(), sid, lambda m: None)
    pipe.set_pose({"latitude": 47.0, "longitude": 8.0, "altitude_agl": 25.0, "pitch_deg": 90})
    assert pipe.current_pose()["altitude_agl"] == 25.0
    # force guidance: everything is "due"
    svc.config["guidance"]["min_revisit_priority"] = 0.0
    img = cv2.cvtColor(np.asarray(make_panel(71)), cv2.COLOR_RGB2BGR)
    pipe.process(Frame(img, time.time(), time.perf_counter(), 1))
    ev = svc.engine.process_frame(make_panel(71), sid, pose={"latitude": 47.0, "longitude": 8.0, "altitude_agl": 25.0, "pitch_deg": 90})
    assert ev and ev[0].geo and ev[0].geo["ground_fix"]["slant_range_m"] == pytest.approx(25.0)
    # pose is too stale after max age
    pipe.pose_max_age = 0.0
    assert pipe.current_pose() is None


def test_guidance_message():
    from ml.camera_model import CameraIntrinsics, MetricScale
    from ml.config import load_config
    from ml.guidance import compute_guidance

    k = CameraIntrinsics.from_hfov(640, 480, 90)
    g = compute_guidance(load_config(), "solar_panel", (400, 300, 500, 400), k, MetricScale(0.02, 12.8, "range"), "B-14")
    # 0.5 px/cm now, 2 px/cm needed → range 12.8 → 3.2 m: move ~10 m closer; asset is right / below centre
    assert not g.resolution_ok and g.move_closer_m == pytest.approx(9.6, abs=0.01)
    assert g.message.startswith("B-14: move 10 m closer") and "down" in g.message and "right" in g.message


def test_graph_propagation_changes_ranking(stack):
    client, svc = stack
    op, admin = H(client, "operator"), H(client, "admin")
    s1 = start(client, op)
    panel = observe(client, op, s1, make_panel(81))
    inverter = observe(client, op, s1, make_panel(82))
    other = observe(client, op, s1, make_panel(83))
    client.post(f"/api/sessions/{s1}/end", headers=op)
    s2 = start(client, op)
    svc.engine.observe(add_crack(make_panel(81), 10), "solar_panel", s2)   # the panel deteriorates
    before = {a["asset_id"]: a["revisit_priority_score"] for a in svc.store.recompute_revisit(SITE)}
    assert client.post("/api/relations", json={"parent_id": inverter["asset_id"], "child_id": panel["asset_id"]}, headers=admin).status_code == 201
    after = {a["asset_id"]: a for a in svc.store.recompute_revisit(SITE)}
    assert after[inverter["asset_id"]]["revisit_priority_score"] > before[inverter["asset_id"]]
    assert after[inverter["asset_id"]]["revisit_components"]["graph_source"] == panel["asset_id"]
    assert after[inverter["asset_id"]]["revisit_priority_score"] > after[other["asset_id"]]["revisit_priority_score"]
    g = client.get(f"/api/assets/{panel['asset_id']}/graph", headers=op).json()
    assert g["depends_on_it"] == [{"asset_id": inverter["asset_id"], "depth": 1}]


def test_feedback_and_stats(stack):
    client, svc = stack
    op = H(client, "operator")
    s1 = start(client, op)
    ev = observe(client, op, s1, make_panel(91))
    assert client.post(f"/api/observations/{ev['observation_id']}/feedback", json={"label": "nonsense"}, headers=op).status_code == 422
    r = client.post(f"/api/observations/{ev['observation_id']}/feedback", json={"label": "false_alarm"}, headers=op)
    assert r.status_code == 201
    st = client.get("/api/feedback/stats", headers=op).json()
    assert st["totals"]["false_alarm"] == 1 and st["by_session"][0]["session_id"] == s1


def test_work_order_full_cycle(config, embedder, classifier):
    config["geo"]["identity_mode"] = "appearance"
    config["severity"]["levels"] = {**config["severity"]["levels"], "priority": 0.2, "critical": 0.3}
    svc = make_stack(config, embedder, classifier)
    with TestClient(create_app(config, services=svc, background_sync=False)) as client:
        op, mt = H(client, "operator"), H(client, "maintenance")
        s1 = start(client, op)
        base = observe(client, op, s1, make_panel(101))
        client.post(f"/api/sessions/{s1}/end", headers=op)
        s2 = start(client, op)
        bad = observe(client, op, s2, add_crack(make_panel(101), 10))
        assert bad["critical"], bad["severity_score"]
        wos = client.get("/api/work-orders", params={"asset_id": base["asset_id"]}, headers=mt).json()
        assert len(wos) == 1 and wos[0]["status"] == "Open"
        wid = wos[0]["work_order_id"]
        assert client.patch(f"/api/work-orders/{wid}", json={"status": "Assigned"}, headers=mt).status_code == 409   # needs assignee
        assert client.patch(f"/api/work-orders/{wid}", json={"status": "Assigned", "assignee": "crew-3"}, headers=mt).json()["status"] == "Assigned"
        assert client.patch(f"/api/work-orders/{wid}", json={"status": "Verified"}, headers=mt).status_code == 409   # automatic only
        assert client.patch(f"/api/work-orders/{wid}", json={"status": "Fixed", "note": "panel replaced"}, headers=mt).json()["status"] == "Fixed"
        client.post(f"/api/sessions/{s2}/end", headers=op)
        s3 = start(client, op)
        observe(client, op, s3, make_panel(101))          # next flight: repaired
        w = client.get(f"/api/work-orders/{wid}", headers=mt).json()
        assert w["status"] == "Verified" and w["verified_session_id"] == s3
        assert [h["status"] for h in w["history"]] == ["Open", "Assigned", "Fixed", "Verified"]
        d = client.get(f"/api/defects/{w['defect_id']}", headers=mt).json()
        assert d["status"] == "Resolved"


def test_drone_keys_scopes_and_signed_frames(stack):
    client, svc = stack
    admin = H(client, "admin")
    svc.store.ensure_site("site_other", "Other")
    r = client.post("/api/admin/drones", json={"name": "D1", "site_ids": [SITE]}, headers=admin)
    assert r.status_code == 201
    key, drone_id = r.json()["api_key"], r.json()["drone_id"]
    assert "api_key" not in client.get("/api/admin/drones", headers=admin).json()[0]
    K = {"X-AUDI-Key": key}
    assert client.post("/api/drone/sessions/start", json={"site_id": "site_other"}, headers=K).status_code == 403
    sid = client.post("/api/drone/sessions/start", json={"site_id": SITE}, headers=K).json()["session"]["session_id"]
    assert svc.store.get_session(sid)["drone_id"] == drone_id
    body = jpeg(make_panel(111))
    unsigned = client.post("/api/drone/frames", content=body, headers={**K, "X-AUDI-Session": sid})
    assert unsigned.status_code == 401
    sig = sign_payload(hash_api_key(key), body)
    ok = client.post("/api/drone/frames", content=body, headers={**K, **sig, "X-AUDI-Session": sid,
                                                                  "X-AUDI-Pose": json.dumps({"latitude": 47, "longitude": 8, "altitude_agl": 20})})
    assert ok.status_code == 200, ok.text
    assert ok.json()["events"][0]["type"] == "new_asset"
    stale = sign_payload(hash_api_key(key), body, ts=int(time.time()) - 3600)
    assert client.post("/api/drone/frames", content=body, headers={**K, **stale, "X-AUDI-Session": sid}).status_code == 401
    # second drone limited to telemetry cannot start sessions
    k2 = client.post("/api/admin/drones", json={"name": "D2", "site_ids": [SITE], "scopes": ["telemetry:write"]}, headers=admin).json()["api_key"]
    assert client.post("/api/drone/sessions/start", json={"site_id": SITE}, headers={"X-AUDI-Key": k2}).status_code == 403
    assert client.post(f"/api/drone/sessions/{sid}/end", headers=K).status_code == 200
    client.post(f"/api/admin/drones/{drone_id}/revoke", headers=admin)
    assert client.post("/api/drone/sessions/start", json={"site_id": SITE}, headers=K).status_code == 401


def test_rate_limit_and_metrics(config, embedder, classifier):
    config["security"]["rate_limit_per_minute"] = 60
    config["security"]["rate_limit_burst"] = 5
    svc = make_stack(config, embedder, classifier)
    with TestClient(create_app(config, services=svc, background_sync=False)) as client:
        admin = H(client, "admin")
        codes = [client.get("/api/sites", headers=admin).status_code for _ in range(8)]
        assert codes[:4] == [200] * 4 and 429 in codes
        r = next(client.get("/api/sites", headers=admin) for _ in range(1))
        assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1
        assert client.get("/metrics").status_code == 401
        m = client.get("/metrics", headers=admin)
        assert m.status_code == 200
        assert "audi_http_requests_total" in m.text and "audi_outbox_pending" in m.text and "audi_rate_limited_total" in m.text


def test_images_encrypted_at_rest(config, embedder, classifier):
    config["security"]["encrypt_images"] = True
    svc = make_stack(config, embedder, classifier)
    with TestClient(create_app(config, services=svc, background_sync=False)) as client:
        op = H(client, "operator")
        sid = start(client, op)
        ev = observe(client, op, sid, make_panel(121))
        raw = svc.store.image_path(ev["image_reference"]).read_bytes()
        assert raw.startswith(MAGIC)
        r = client.get(f"/api/images/{ev['image_reference']}", headers=op)
        assert r.status_code == 200 and r.content[:2] == b"\xff\xd8"      # decrypted JPEG
        client.post(f"/api/sessions/{sid}/end", headers=op)
        s2 = start(client, op)
        assert observe(client, op, s2, make_panel(121))["asset_id"] == ev["asset_id"]   # re-ID reads sealed prototypes


def test_cloud_receives_images_and_merge(config, embedder, classifier):
    from cloud.main import create_app as cloud_create
    from tests.test_sync_mission import InProcessCloud

    config["geo"]["identity_mode"] = "appearance"
    capp = cloud_create(config, secret="cloud-secret")
    capp.state.seed_user("sync", "sync-pass")
    cloud = InProcessCloud(capp, "sync", "sync-pass")
    cloud.up = False
    svc = make_stack(config, embedder, classifier, cloud)
    start_ = svc.mission.start(SITE, "op")
    sid = start_["session"]["session_id"]
    a = svc.engine.observe(make_panel(131), "solar_panel", sid)
    b = svc.engine.observe(make_panel(132), "solar_panel", sid)
    svc.mission.end(sid)                                     # offline: queued
    svc.store.merge_assets(a.asset_id, b.asset_id, "admin")
    cloud.up = True
    assert svc.sync.flush_once() > 0                        # edge back online → everything syncs
    cc = TestClient(capp)
    h = {"Authorization": f"Bearer {cloud._token}"}
    img = cc.get(f"/cloud/images/{a.image_reference}", headers=h)
    assert img.status_code == 200 and img.content == svc.store.read_image_bytes(a.image_reference)
    pulled = cc.get(f"/cloud/sites/{SITE}/pull", headers=h).json()["records"]
    assert "image" not in pulled
    assert b.asset_id not in {x["asset_id"] for x in pulled["asset"]}
    assert {o["asset_id"] for o in pulled["observation"]} == {a.asset_id}


def test_owner_report_from_real_data(stack):
    client, svc = stack
    op = H(client, "operator")
    s1 = start(client, op)
    # through the engine (no JPEG round trip: compression blocks give distinct synthetic panels shared SIFT matches)
    x = svc.engine.observe(make_panel(141), "solar_panel", s1)
    y = svc.engine.observe(add_crack(make_panel(142), 8), "solar_panel", s1)
    assert x.asset_id != y.asset_id
    month = utc_now().strftime("%Y-%m")
    rep = client.get(f"/api/sites/{SITE}/owner-report", params={"month": month}, headers=op).json()
    assert rep["activity"]["missions"] == 1 and rep["activity"]["assets_inspected"] == 2
    assert rep["health"]["site"] is not None and len(rep["health"]["assets"]) == 2
    pdf = client.get(f"/api/sites/{SITE}/owner-report", params={"month": month, "format": "pdf"}, headers=op)
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"
