#!/usr/bin/env python3
"""Benchmarks for the SRS non-functional requirements → reports/bench.md.

NFR-1  end-to-end alert latency: a recorded video (built from real test-split
       images; the second pass carries a synthetic crack) is played at its
       native FPS through CameraReader → LivePipeline with the trained models.
       Latency = frame capture (monotonic clock at grab) → alert event published.
       `--source <index|url|file>` measures on a live camera or your own video
       instead (alerts then depend on what the camera sees).
NFR-2  matching latency: the identity cascade (index top-k + SIFT/RANSAC
       verification) with `--assets` real-image assets in ONE (site, type)
       partition, queried with re-photographed (webcam-augmented) views.
NFR-3  sync bound: session start against a cloud that is down, black-holed
       (accepts TCP, never answers), slow (answers health, stalls pull/push)
       and healthy, through the real HTTP client.

All work happens in a temporary store; the real database is not touched.
Usage: python scripts/bench.py [--assets 5000] [--source 0] [--skip nfr1,nfr2,nfr3]
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import socket
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from ml.config import ROOT as CFG_ROOT, cfg, load_config, resolve_path  # noqa: E402

logger = logging.getLogger("bench")


def pct(values: List[float], q: float) -> Optional[float]:
    return float(np.percentile(np.asarray(values), q)) if values else None


def fmt(v: Optional[float], nd: int = 1) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def temp_config(config: Dict[str, Any], tmp: Path) -> Dict[str, Any]:
    c = copy.deepcopy(config)
    c["storage"]["db_url"] = f"sqlite:///{tmp / 'bench.db'}"
    c["storage"]["images_dir"] = str(tmp / "images")
    return c


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def manifest_rows(config, asset_type: Optional[str] = None, splits=("test",), condition: Optional[str] = None):
    from training.common import load_manifest

    return [
        r for r in load_manifest(config)
        if r.split in splits and (asset_type is None or r.asset_type == asset_type) and (condition is None or r.condition == condition)
    ]


# ── NFR-2 ─────────────────────────────────────────────────────────────────
def bench_matching(config: Dict[str, Any], embedder, n_assets: int) -> Dict[str, Any]:
    from memory.vector_memory import PrototypeIndex
    from ml.asset_identity import AssetIdentifier
    from ml.augment import WebcamAugment
    from training.common import extract_features

    b = cfg(config, "bench")
    rng = random.Random(int(b["seed"]))
    rows = manifest_rows(config, splits=("train", "val", "test"))
    rng.shuffle(rows)
    if len(rows) < n_assets:
        raise SystemExit(f"NFR-2 needs {n_assets} images; the manifest has {len(rows)}")
    rows = rows[:n_assets]
    paths = [r.path for r in rows]
    t0 = time.perf_counter()
    feats = extract_features(embedder, paths, config, view=0, tag="bench_assets")
    logger.info("embedded %d asset images in %.1fs", len(paths), time.perf_counter() - t0)

    site, atype = "bench_site", b["matching_asset_type"]
    mcfg = cfg(config, "matching")
    index = PrototypeIndex(embedder.embedding_dim, cfg(config, "memory", "vector_backend"),
                           int(mcfg["max_prototypes_per_asset"]), float(mcfg["prototype_novelty"]))
    for i, v in enumerate(feats):
        index.add(i, f"asset_{i}", site, atype, v)

    def loader(pid: int) -> Optional[Image.Image]:
        with Image.open(CFG_ROOT / paths[pid]) as im:
            return im.convert("RGB")

    ident = AssetIdentifier({**config, "aruco": {**config["aruco"], "enabled": False}}, index, loader)
    q_ids = rng.sample(range(n_assets), min(int(b["matching_queries"]), n_assets))
    view = WebcamAugment(cfg(config, "memory_build", "view_augment"))
    queries = [view(loader(i), random.Random(i)) for i in q_ids]
    q_emb = embedder.encode_batch(queries)

    def run(label: str) -> Dict[str, Any]:
        total, search = [], []
        correct = new = 0
        stages: Dict[str, int] = {}
        for qi, img, emb in zip(q_ids, queries, q_emb):
            s0 = time.perf_counter()
            index.search(emb, site, atype, int(mcfg["top_k"]))
            search.append((time.perf_counter() - s0) * 1000)
            t = time.perf_counter()
            res = ident.identify(img, None, img, emb, atype, site, None, [], {})
            total.append((time.perf_counter() - t) * 1000)
            stages[res.stage] = stages.get(res.stage, 0) + 1
            correct += res.asset_id == f"asset_{qi}"
            new += res.asset_id is None
        return {
            "pass": label, "n": len(total), "p50_ms": pct(total, 50), "p95_ms": pct(total, 95),
            "search_p50_ms": pct(search, 50), "search_p95_ms": pct(search, 95),
            "top1": correct / len(total), "false_new": new / len(total), "stages": stages,
        }

    cold = run("cold: no reference keypoints cached")
    ident._feature_cache.clear()
    budget = min(int(mcfg["feature_cache_size"]), n_assets)
    t0 = time.perf_counter()
    for pid in range(budget):
        ident.reference_features(pid)
    prefetch_s = time.perf_counter() - t0
    covered = sum(1 for i in q_ids if i < budget) / len(q_ids)
    prefetched = run(f"after session-start prefetch ({budget} of {n_assets} references, {covered:.0%} of queried assets covered)")
    warm = run("warm: every queried reference cached")
    return {"assets": n_assets, "prototypes": len(index), "backend": index.backend_name, "cold": cold,
            "prefetched": prefetched, "warm": warm, "prefetch_s": prefetch_s, "prefetch_budget": budget}


# ── NFR-1 ─────────────────────────────────────────────────────────────────
def flight_images(config: Dict[str, Any]) -> List[Image.Image]:
    """`bench.video_assets` test-split images labelled normal, of `bench.video_asset_type`."""
    b = cfg(config, "bench")
    rows = manifest_rows(config, b["video_asset_type"], condition="normal")
    random.Random(int(b["seed"])).shuffle(rows)
    images = []
    for r in rows[: int(b["video_assets"])]:
        with Image.open(CFG_ROOT / r.path) as im:
            images.append(im.convert("RGB"))
    return images


def build_video(config: Dict[str, Any], path: Path, images: List[Image.Image], damaged: bool) -> Path:
    """One re-photographed view per asset, held for `bench.video_seconds_per_asset`; optionally cracked."""
    from ml.augment import WebcamAugment
    from ml.synthetic import draw_crack

    b = cfg(config, "bench")
    w, h = (int(x) for x in b["video_size"])
    fps = int(b["video_fps"])
    per = max(1, int(round(fps * float(b["video_seconds_per_asset"]))))
    view = WebcamAugment(cfg(config, "memory_build", "view_augment"))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    for i, img in enumerate(images):
        rng = random.Random(int(b["seed"]) + i)
        shot = view(img, rng)
        if damaged:
            shot = draw_crack(shot, rng, float(b["crack_extent"]))
        frame = cv2.cvtColor(np.asarray(shot.resize((w, h))), cv2.COLOR_RGB2BGR)
        for _ in range(per):
            writer.write(frame)
    writer.release()
    return path


def bench_alert_latency(config: Dict[str, Any], tmp: Path, source: Optional[str], embedder, classifier) -> Dict[str, Any]:
    from api.services import build_services
    from ml.detector import FullFrameDetector, build_detector
    from ml.live.camera import CameraReader
    from ml.live.devices import resolve_source
    from ml.live.pipeline import LivePipeline

    b = cfg(config, "bench")
    atype = b["video_asset_type"]
    detector = build_detector(config) if source is not None else FullFrameDetector(atype, float(cfg(config, "detection", "full_frame_confidence")))
    svc = build_services(config, embedder=embedder, classifier=classifier, detector=detector, use_cloud=False)
    site = cfg(config, "sites", "default_site_id")

    if source is None:
        images = flight_images(config)
        first = svc.mission.start(site, "bench")["session"]["session_id"]
        for img in images:
            svc.engine.observe(img, atype, first)
        svc.mission.end(first)
        src = str(build_video(config, tmp / "bench_flight.avi", images, damaged=True))
        label = f"recorded video: {len(images)} {atype} assets, second flight with synthetic cracks"
    else:
        src = resolve_source(source)
        label = f"live source {source!r} (detector: {cfg(config, 'detection', 'mode')})"

    sid = svc.mission.start(site, "bench")["session"]["session_id"]
    events: List[Dict[str, Any]] = []
    cam = CameraReader.from_config({**config, "camera": {**config["camera"], "loop_video": False}}, src)
    pipe = LivePipeline(config, svc.engine, cam, sid, events.append)
    pipe.start()
    t0 = time.monotonic()
    limit = float(cfg(config, "bench", "video_seconds_per_asset")) * int(b["video_assets"]) * 4 + 30
    while pipe.running and time.monotonic() - t0 < limit:
        time.sleep(0.2)
    pipe.stop()
    report = svc.mission.end(sid)
    st = pipe.status()
    lat = st["latency_ms"]
    alerts = [e for e in events if e.get("alert")]
    event_types = Counter(e["event"]["type"] for e in events if e.get("type") == "event")
    return {
        "label": label, "counters": st["counters"], "latency": lat, "event_types": dict(event_types),
        "alert_p50_ms": lat.get("alert", {}).get("p50"), "alert_p95_ms": lat.get("alert", {}).get("p95"),
        "alerts": len(alerts), "alert_levels": sorted({e["alert"] for e in alerts}),
        "flagged_in_report": len(report.get("flagged_defects", [])),
        "capture_fps": cam.measured_fps(),
    }


# ── NFR-3 ─────────────────────────────────────────────────────────────────
class _Uvicorn:
    def __init__(self, app, port: int):
        import uvicorn

        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        end = time.time() + 15
        while not self.server.started and time.time() < end:
            time.sleep(0.05)
        return self

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(timeout=10)


class _BlackHole:
    """Accepts TCP connections and never answers."""

    def __init__(self, port: int):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", port))
        self.sock.listen(64)
        self.conns: List[socket.socket] = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        self.sock.settimeout(0.2)
        while not self.stop.is_set():
            try:
                c, _ = self.sock.accept()
                self.conns.append(c)
            except socket.timeout:
                continue

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(timeout=2)
        for c in self.conns:
            c.close()
        self.sock.close()


def bench_sync(config: Dict[str, Any], tmp: Path, embedder, classifier) -> Dict[str, Any]:
    from api.services import build_services
    from cloud.main import create_app as create_cloud
    from memory.cloud_store_interface import HttpCloudStore
    from ml.detector import FullFrameDetector

    b = cfg(config, "bench")
    user, password = "bench_sync", "bench-" + str(random.Random(int(b["seed"])).getrandbits(64))
    cloud_app = create_cloud(config, db_url=f"sqlite:///{tmp / 'cloud.db'}", secret="bench-secret-" + password)
    cloud_app.state.seed_user(user, password)
    delay = float(b["slow_cloud_delay_s"])

    @cloud_app.middleware("http")
    async def stall(request, call_next):
        import asyncio

        if getattr(cloud_app.state, "slow", False) and ("/pull" in request.url.path or "/push" in request.url.path):
            await asyncio.sleep(delay)
        return await call_next(request)

    det = FullFrameDetector("solar_panel", 1.0)
    site = cfg(config, "sites", "default_site_id")
    results = []

    def scenario(name: str, url: str) -> None:
        work = tmp / f"sync_{len(results)}"
        work.mkdir(exist_ok=True)
        c = temp_config(config, work)
        svc = build_services(c, embedder=embedder, classifier=classifier, detector=det, cloud=HttpCloudStore(url, user, password))
        t0 = time.monotonic()
        started = svc.mission.start(site, "bench")
        t_start = time.monotonic() - t0
        sid = started["session"]["session_id"]
        t1 = time.monotonic()
        report = svc.mission.end(sid)
        t_end = time.monotonic() - t1
        results.append({
            "scenario": name, "start_s": t_start, "end_s": t_end, "mode": started["session"]["mode"],
            "start_timed_out": bool(started["sync"].get("timed_out")), "end_online": bool(report["sync"].get("online")),
        })

    scenario("cloud down (connection refused)", f"http://127.0.0.1:{free_port()}")
    port = free_port()
    with _BlackHole(port):
        scenario("cloud black-holed (no response)", f"http://127.0.0.1:{port}")
    port = free_port()
    with _Uvicorn(cloud_app, port):
        cloud_app.state.slow = True
        scenario(f"cloud slow (pull/push stall {delay:.0f} s)", f"http://127.0.0.1:{port}")
        cloud_app.state.slow = False
        scenario("cloud healthy", f"http://127.0.0.1:{port}")
    return {"timeout_s": float(cfg(config, "cloud", "sync_timeout_s")), "scenarios": results}


# ── report ────────────────────────────────────────────────────────────────
def verdict(value: Optional[float], target: float) -> str:
    if value is None:
        return "NOT MEASURED"
    return "PASS" if value <= target else "FAIL"


def write_report(config, out: Path, res: Dict[str, Any], env: Dict[str, Any]) -> str:
    t = cfg(config, "bench", "targets")
    margin = float(cfg(config, "bench", "sync_margin_s"))
    lines = ["# AUDI benchmarks (NFR-1/2/3)", "", f"Generated {env['ts']} on {env['platform']} · device `{env['device']}` · backbone `{env['backbone']}`.", ""]
    lines += ["| Requirement | Measured | Target | Result |", "|---|---|---|---|"]
    if "nfr1" in res:
        r = res["nfr1"]
        lines.append(f"| NFR-1 alert latency p95 | {fmt(r['alert_p95_ms'])} ms (p50 {fmt(r['alert_p50_ms'])}, n={r['alerts']}) | ≤ {t['alert_p95_ms']:.0f} ms | {verdict(r['alert_p95_ms'], t['alert_p95_ms'])} |")
    if "nfr2" in res:
        r = res["nfr2"]
        for key, name in (("prefetched", "after session-start prefetch"), ("cold", "cold cache"), ("warm", "warm cache")):
            p = r[key]
            lines.append(f"| NFR-2 matching p95 @ {r['assets']} assets, {name} | {fmt(p['p95_ms'])} ms (p50 {fmt(p['p50_ms'])}) | ≤ {t['matching_p95_ms']:.0f} ms | {verdict(p['p95_ms'], t['matching_p95_ms'])} |")
    if "nfr3" in res:
        worst = max(max(s["start_s"], s["end_s"]) for s in res["nfr3"]["scenarios"])
        bound = min(float(t["sync_s"]), res["nfr3"]["timeout_s"]) + margin
        lines.append(f"| NFR-3 worst start/end sync | {worst:.2f} s | ≤ {t['sync_s']:.0f} s (+{margin:.0f} s slack) | {'PASS' if worst <= bound else 'FAIL'} |")
    lines.append("")

    if "nfr1" in res:
        r = res["nfr1"]
        lines += ["## NFR-1 — live pipeline", "", f"Input: {r['label']}. Capture FPS {r['capture_fps']:.1f}; alerts raised: {r['alerts']} ({', '.join(r['alert_levels']) or 'none'}); defects flagged in the post-flight report: {r['flagged_in_report']}.", "",
                  f"Counters: `{json.dumps(r['counters'])}`", "",
                  f"Event types: `{json.dumps(r['event_types'])}`. On the recorded video every asset was stored in the first flight, "
                  "so `change` = re-identified with its damage detected and `new_asset` = re-identification failed (the asset is "
                  "registered again; it still alerts because the classifier sees the crack).", "",
                  "| Stage | p50 ms | p95 ms | n |", "|---|---|---|---|"]
        for k, v in sorted(r["latency"].items()):
            lines.append(f"| {k} | {fmt(v['p50'])} | {fmt(v['p95'])} | {v['n']} |")
        lines.append("")
    if "nfr2" in res:
        r = res["nfr2"]
        lines += ["## NFR-2 — identity cascade", "", f"{r['assets']} assets / {r['prototypes']} prototypes in one partition, `{r['backend']}` backend; queries are webcam-augmented re-photographs of stored assets.", "",
                  "| Pass | p50 ms | p95 ms | index search p95 ms | top-1 | false new | stages |", "|---|---|---|---|---|---|---|"]
        for p in (r["cold"], r["prefetched"], r["warm"]):
            lines.append(f"| {p['pass']} | {fmt(p['p50_ms'])} | {fmt(p['p95_ms'])} | {fmt(p['search_p95_ms'], 2)} | {p['top1']:.3f} | {p['false_new']:.3f} | `{json.dumps(p['stages'])}` |")
        lines += ["", f"The session-start prefetch (`SemanticMemoryEngine.prefetch_references`, run in the background by `MissionService.start`) "
                  f"took {r['prefetch_s']:.1f} s for {r['prefetch_budget']} references (`matching.feature_cache_size`).",
                  "Embedding the query crop is part of NFR-1 (the `embed` stage in the live pipeline), not matching.", ""]
    if "nfr3" in res:
        lines += ["## NFR-3 — sync bound", "", f"`cloud.sync_timeout_s` = {res['nfr3']['timeout_s']}", "", "| Scenario | start s | end s | session mode | start timed out | end online |", "|---|---|---|---|---|---|"]
        for s in res["nfr3"]["scenarios"]:
            lines.append(f"| {s['scenario']} | {s['start_s']:.2f} | {s['end_s']:.2f} | {s['mode']} | {s['start_timed_out']} | {s['end_online']} |")
        lines.append("")
    text = "\n".join(lines)
    out.write_text(text)
    (out.parent / "bench.json").write_text(json.dumps({"env": env, **res}, indent=2, default=str))
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--assets", type=int, default=5000)
    parser.add_argument("--source", default=None, help="camera index / URL / video file for NFR-1 (default: generated flight video)")
    parser.add_argument("--skip", default="", help="comma list of nfr1,nfr2,nfr3")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    import platform

    from ml.condition_classifier import load_condition_classifier
    from ml.embedding_model import AssetEmbeddingModel
    from ml.schemas import utc_now_iso

    base = load_config(args.config)
    tmp = Path(tempfile.mkdtemp(prefix="audi_bench_"))
    config = temp_config(base, tmp)
    embedder = AssetEmbeddingModel.from_config(config)
    classifier = load_condition_classifier(config, embedder)
    env = {"ts": utc_now_iso(), "platform": f"{platform.system()} {platform.machine()} · Python {platform.python_version()}",
           "device": str(getattr(embedder.backbone, "device", "cpu")), "backbone": embedder.active_model_name}

    res: Dict[str, Any] = {}
    if "nfr2" not in skip:
        res["nfr2"] = bench_matching(config, embedder, args.assets)
        logger.info("NFR-2 %s", {k: res["nfr2"][k] for k in ("cold", "warm")})
    if "nfr1" not in skip:
        res["nfr1"] = bench_alert_latency(config, tmp, args.source, embedder, classifier)
        logger.info("NFR-1 alert p95 %s ms", res["nfr1"]["alert_p95_ms"])
    if "nfr3" not in skip:
        res["nfr3"] = bench_sync(config, tmp, embedder, classifier)
        logger.info("NFR-3 %s", res["nfr3"])

    out = resolve_path(cfg(base, "reports", "dir")) / "bench.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(write_report(base, out, res, env))


if __name__ == "__main__":
    main()
