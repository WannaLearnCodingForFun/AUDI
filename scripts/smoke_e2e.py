#!/usr/bin/env python3
"""End-to-end smoke test against real processes (Phase 9 gate).

Starts cloud/ and the API as subprocesses on free ports with a throw-away
database and generated credentials, then:

  1. logs in as admin, operator and maintenance (and checks the role guards);
  2. admin creates a site, subscribes the maintenance user, raises building
     criticality through the versioned config endpoint;
  3. operator flies mission 1 from a video file of real test-split building
     images through the live pipeline (POST /api/live/start, camera mode);
  4. briefing for mission 2 lists the assets from mission 1 (FR-4);
  5. operator flies mission 2 from a video of the same assets with synthetic
     cracks while WebSockets collect live alerts (operator) and escalations
     (maintenance);
  6. the post-flight report (JSON + PDF) flags the defects (FR-18), maintenance
     receives the notification (FR-19) and moves a defect to In Progress (FR-17);
  7. the outbox is flushed to the cloud (FR-6/7) and the cloud holds the records.

Prints one PASS/FAIL line per check and exits non-zero on any failure.
Usage: python scripts/smoke_e2e.py [--keep]
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402
import yaml  # noqa: E402

from ml.config import cfg, load_config  # noqa: E402
from ml.net import ensure_ca_bundle  # noqa: E402
from scripts.bench import build_video, flight_images, free_port  # noqa: E402

PY = sys.executable
SITE = "smoke_site"


class Checks:
    def __init__(self):
        self.results: List[tuple] = []

    def check(self, name: str, ok: bool, detail: Any = "") -> bool:
        self.results.append((name, bool(ok), detail))
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail != "" else ""), flush=True)
        return bool(ok)

    @property
    def failed(self) -> List[str]:
        return [n for n, ok, _ in self.results if not ok]


def write_config(base: Dict[str, Any], tmp: Path) -> tuple[Path, int, int]:
    c = yaml.safe_load(yaml.safe_dump(base))
    api_port, cloud_port = free_port(), free_port()
    c["storage"]["db_url"] = f"sqlite:///{tmp / 'local.db'}"
    c["storage"]["images_dir"] = str(tmp / "images")
    c["auth"]["secret_file"] = str(tmp / ".jwt_secret")
    c["auth"]["credentials_file"] = str(tmp / "credentials.env")
    c["cloud"]["base_url"] = f"http://127.0.0.1:{cloud_port}"
    c["cloud"]["server"]["port"] = cloud_port
    c["cloud"]["server"]["db_url"] = f"sqlite:///{tmp / 'cloud.db'}"
    c["cloud"]["server"]["secret_file"] = str(tmp / ".cloud_jwt_secret")
    c["api"]["port"] = api_port
    path = tmp / "config.yaml"
    path.write_text(yaml.safe_dump(c, sort_keys=False))
    return path, api_port, cloud_port


def wait_http(url: str, timeout: float, proc: subprocess.Popen) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if proc.poll() is not None:
            return False
        try:
            if httpx.get(url, timeout=1.0).status_code == 200:
                return True
        except httpx.HTTPError:
            time.sleep(0.5)
    return False


def wait_for(pred: Callable[[], bool], timeout: float, step: float = 0.5) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(step)
    return False


class WsCollector:
    """Collects JSON messages from one WebSocket in a background thread."""

    def __init__(self, url: str):
        from websockets.sync.client import connect

        self.ws = connect(url, open_timeout=10)
        self.messages: List[Dict[str, Any]] = []
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            for raw in self.ws:
                if isinstance(raw, str):
                    self.messages.append(json.loads(raw))
        except Exception:  # noqa: BLE001 - the socket closes when the smoke test ends
            return

    def of_type(self, t: str) -> List[Dict[str, Any]]:
        return [m for m in list(self.messages) if m.get("type") == t]

    def close(self):
        self.ws.close()
        self.thread.join(timeout=5)


def fly(c: Checks, op: httpx.Client, video: Path, label: str, timeout: float) -> str:
    started = op.post("/api/sessions/start", json={"site_id": SITE})
    c.check(f"{label}: session started", started.status_code == 200, started.status_code)
    sid = started.json()["session"]["session_id"]
    live = op.post("/api/live/start", json={"session_id": sid, "camera_source": str(video), "mode": "camera",
                                          "detection_mode": "full_frame", "asset_type": "building"})
    c.check(f"{label}: live pipeline started on the video file", live.status_code == 200, live.text[:200] if live.status_code != 200 else "")
    finished = wait_for(lambda: not op.get("/api/live/status").json().get("running", False), timeout)
    st = op.get("/api/live/status").json()
    c.check(f"{label}: video played through the pipeline", finished and st.get("counters", {}).get("processed", 0) > 0,
            {k: st.get("counters", {}).get(k) for k in ("seen", "processed", "events", "alerts")})
    return sid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keep", action="store_true", help="keep the temporary directory (logs, databases)")
    args = parser.parse_args()

    ensure_ca_bundle()
    base = load_config()
    b = cfg(base, "bench")
    tmp = Path(tempfile.mkdtemp(prefix="audi_smoke_"))
    config_path, api_port, cloud_port = write_config(base, tmp)
    pw = {r: secrets.token_urlsafe(18) for r in ("ADMIN", "OPERATOR", "MAINTENANCE", "CLOUD")}
    env = {
        **os.environ, "AUDI_CONFIG": str(config_path), "PYTHONPATH": str(ROOT),
        "AUDI_JWT_SECRET": secrets.token_urlsafe(32), "AUDI_CLOUD_JWT_SECRET": secrets.token_urlsafe(32),
        **{f"AUDI_{r}_PASSWORD": p for r, p in pw.items()},
    }
    users = {r.lower(): (e["default_username"], pw[r]) for e in base["auth"]["seed_users"]
             for r in pw if e["password_env"] == f"AUDI_{r}_PASSWORD" and r != "CLOUD"}
    c = Checks()
    procs: List[subprocess.Popen] = []
    timeout = float(b["video_seconds_per_asset"]) * int(b["video_assets"]) * 4 + 60
    try:
        seed = subprocess.run([PY, "scripts/seed_users.py"], cwd=ROOT, env=env, capture_output=True, text=True)
        c.check("seed_users.py created one user per role", seed.returncode == 0, seed.stderr[-300:] if seed.returncode else "")

        images = flight_images(base)
        v1 = build_video(base, tmp / "flight1.avi", images, damaged=False)
        v2 = build_video(base, tmp / "flight2.avi", images, damaged=True)
        c.check("flight videos written", v1.stat().st_size > 0 and v2.stat().st_size > 0, f"{len(images)} building assets")

        logs = {n: open(tmp / f"{n}.log", "w") for n in ("cloud", "api")}
        procs.append(subprocess.Popen([PY, "-m", "cloud.main"], cwd=ROOT, env=env, stdout=logs["cloud"], stderr=subprocess.STDOUT))
        c.check("cloud service up", wait_http(f"http://127.0.0.1:{cloud_port}/cloud/health", 60, procs[-1]))
        procs.append(subprocess.Popen([PY, "run_api.py"], cwd=ROOT, env=env, stdout=logs["api"], stderr=subprocess.STDOUT))
        api_url = f"http://127.0.0.1:{api_port}"
        if not c.check("API up (models loaded)", wait_http(f"{api_url}/api/health", 300, procs[-1])):
            raise RuntimeError("API did not start; see api.log")

        clients: Dict[str, httpx.Client] = {}
        tokens: Dict[str, str] = {}
        for role, (username, password) in users.items():
            r = httpx.post(f"{api_url}/api/auth/token", data={"username": username, "password": password})
            ok = r.status_code == 200
            tokens[role] = r.json()["access_token"] if ok else ""
            clients[role] = httpx.Client(base_url=api_url, headers={"Authorization": f"Bearer {tokens[role]}"}, timeout=120)
            me = clients[role].get("/api/auth/me").json() if ok else {}
            c.check(f"login as {role}", ok and me.get("role") == role, me.get("role"))
        admin, op, maint = clients["admin"], clients["operator"], clients["maintenance"]
        c.check("unauthenticated request rejected", httpx.get(f"{api_url}/api/assets").status_code == 401)
        c.check("operator cannot reach admin config", op.get("/api/admin/config").status_code == 403)
        c.check("maintenance cannot start a session", maint.post("/api/sessions/start", json={"site_id": SITE}).status_code == 403)

        c.check("admin created the site", admin.post("/api/admin/sites", json={"site_id": SITE, "name": "Smoke test site"}).status_code == 201)
        ulist = admin.get("/api/admin/users").json()
        for role in ("maintenance", "operator"):
            u = next(x for x in ulist if x["username"] == users[role][0])
            r = admin.put(f"/api/admin/users/{u['user_id']}/sites", json={"sites": sorted(set(u["sites"]) | {SITE})})
            c.check(f"{role} subscribed to {SITE}", r.status_code == 200)
        r = admin.put("/api/admin/config", json={"changes": {"asset_criticality": {"building": 0.95}}})
        versions = admin.get("/api/admin/config/versions").json()
        c.check("admin config change validated and versioned", r.status_code == 200 and len(versions) >= 1, f"version {versions[0]['version'] if versions else None}")
        bad = admin.put("/api/admin/config", json={"changes": {"severity": {"w1": 0.9}}})
        c.check("invalid weights rejected (w1+w2≠1)", bad.status_code == 422)

        sid1 = fly(c, op, v1, "mission 1", timeout)
        end1 = op.post(f"/api/sessions/{sid1}/end").json()
        c.check("mission 1: new assets registered", len(end1.get("new_assets", [])) == len(images), f"{len(end1.get('new_assets', []))} new")

        brief = op.get(f"/api/sites/{SITE}/briefing").json()
        due = brief.get("assets_due", [])
        rs = [a["revisit_priority"] for a in due]
        c.check("briefing ranks every known asset by revisit priority (FR-4)", len(due) == len(images) and rs == sorted(rs, reverse=True), f"{len(due)} assets")

        live_ws = WsCollector(f"ws://127.0.0.1:{api_port}/ws/live?token={tokens['operator']}")
        note_ws = WsCollector(f"ws://127.0.0.1:{api_port}/ws/notifications?token={tokens['maintenance']}")
        sid2 = fly(c, op, v2, "mission 2", timeout)
        alerts = [m for m in live_ws.of_type("event") if m.get("alert")]
        c.check("live alerts pushed over /ws/live", len(alerts) > 0,
                f"{len(alerts)} alerts, latency p95 {op.get('/api/live/status').json().get('latency_ms', {}).get('alert', {}).get('p95')}")
        c.check("live status pushed over /ws/live", len(live_ws.of_type("status")) > 0)
        stream = op.get("/api/live/stream", params={"token": tokens["operator"]})
        c.check("MJPEG stream endpoint answers after the flight", stream.status_code == 200)

        end2 = op.post(f"/api/sessions/{sid2}/end").json()
        report = op.get(f"/api/sessions/{sid2}/report", params={"format": "json"}).json()
        c.check("post-flight report flags the cracked assets (FR-18)", len(report.get("flagged_defects", [])) > 0,
                f"{len(report.get('flagged_defects', []))} flagged / {len(images)}")
        pdf = op.get(f"/api/sessions/{sid2}/report", params={"format": "pdf"})
        c.check("PDF report downloads", pdf.status_code == 200 and pdf.content[:4] == b"%PDF", f"{len(pdf.content)} bytes")

        wait_for(lambda: len(note_ws.of_type("notification")) > 0, 10)
        inbox = maint.get("/api/notifications").json()
        c.check("maintenance received the escalation (FR-19)", len(inbox) > 0 and len(note_ws.of_type("notification")) > 0,
                f"{len(inbox)} in inbox, {len(note_ws.of_type('notification'))} over WebSocket")
        c.check("operator did not receive escalations", len(op.get("/api/notifications").json()) == 0)
        live_ws.close()
        note_ws.close()

        defects = maint.get("/api/defects", params={"site_id": SITE, "status": "Open"}).json()
        moved = maint.patch(f"/api/defects/{defects[0]['defect_id']}", json={"status": "In Progress", "notes": "smoke test"}) if defects else None
        c.check("defect opened and moved to In Progress (FR-17)", moved is not None and moved.status_code == 200 and moved.json()["status"] == "In Progress",
                f"{len(defects)} open")

        flush = admin.post("/api/admin/sync/flush").json()
        sync = admin.get("/api/admin/sync").json()
        c.check("outbox flushed to the cloud (FR-6/7)", end2.get("sync", {}).get("online") and sync["outbox"].get("pending", 1) == 0,
                {"end_sync_online": end2.get("sync", {}).get("online"), "outbox": sync["outbox"], "flush_online": flush.get("online")})
        cloud = httpx.Client(base_url=f"http://127.0.0.1:{cloud_port}")
        ctok = cloud.post("/cloud/auth/token", data={"username": "sync", "password": pw["CLOUD"]}).json()["access_token"]
        pulled = cloud.get(f"/cloud/sites/{SITE}/pull", headers={"Authorization": f"Bearer {ctok}"}).json()["records"]
        c.check("cloud holds the site's assets and observations",
                len(pulled.get("asset", [])) == len(images) + len(end2.get("new_assets", []))
                and len(pulled.get("observation", [])) >= 2 * len(images),
                {**{k: len(v) for k, v in pulled.items()}, "re-registered_in_mission_2": len(end2.get("new_assets", []))})
        metrics = admin.get("/api/admin/metrics")
        c.check("admin metrics endpoint answers", metrics.status_code == 200)
    except Exception as exc:  # noqa: BLE001 - report and fail instead of a traceback mid-run
        c.check("smoke test ran to completion", False, repr(exc))
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()
        print(f"\nlogs and databases: {tmp}" if args.keep or c.failed else "")
        if not args.keep and not c.failed:
            shutil.rmtree(tmp, ignore_errors=True)

    print(f"{len(c.results) - len(c.failed)}/{len(c.results)} checks passed")
    sys.exit(1 if c.failed else 0)


if __name__ == "__main__":
    main()
