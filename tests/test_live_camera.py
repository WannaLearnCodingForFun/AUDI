"""Live input (FR-8, FR-15, D8): fake VideoCapture with disconnect/reconnect, latest-frame
semantics, video-file end, duplicate/frame-skip in the pipeline, non-obstructive overlay."""

import time

import cv2
import numpy as np

from memory.store import LocalStore
from ml.detector import FullFrameDetector
from ml.live.camera import CameraReader, PushFrameSource
from ml.live.overlay import OverlayRenderer
from ml.live.pipeline import LivePipeline
from ml.semantic_memory import SemanticMemoryEngine
from tests.conftest import make_panel


def bgr(seed):
    return cv2.cvtColor(np.asarray(make_panel(seed)), cv2.COLOR_RGB2BGR)


class FakeCapture:
    """Serves frames; `fail_after` frames then reports a grab failure (unplugged)."""

    opened = 0

    def __init__(self, frames, fail_after=None, fps=0.0):
        self.frames, self.fail_after, self.fps = frames, fail_after, fps
        self.i = 0
        FakeCapture.opened += 1

    def isOpened(self):
        return True

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.i = int(value)
        return True

    def get(self, prop):
        return self.fps if prop == cv2.CAP_PROP_FPS else 0.0

    def read(self):
        if self.fail_after is not None and self.i >= self.fail_after:
            return False, None
        if self.i >= len(self.frames):
            return False, None
        f = self.frames[self.i]
        self.i += 1
        time.sleep(0.005)
        return True, f

    def release(self):
        return None


def wait_until(pred, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_camera_reconnects_after_unplug():
    FakeCapture.opened = 0
    frames = [bgr(i) for i in range(5)]
    attempts = {"n": 0}

    def factory(source, backend):
        attempts["n"] += 1
        if attempts["n"] == 2:
            raise RuntimeError("device busy")          # first reconnect attempt fails
        return FakeCapture(frames, fail_after=3 if attempts["n"] == 1 else None)

    cam = CameraReader(0, "any", reconnect_seconds=0.05, capture_factory=factory)
    cam.start()
    try:
        assert wait_until(lambda: cam.reconnects >= 1 and cam.connected and cam.status()["frames"] >= 5)
        st = cam.status()
        assert st["connected"] and st["reconnects"] >= 1 and attempts["n"] >= 3
        assert cam.latest().seq == st["frames"]          # only the latest frame is held
    finally:
        cam.stop()


def test_camera_keeps_only_latest_frame():
    frames = [bgr(i) for i in range(30)]
    cam = CameraReader(0, "any", reconnect_seconds=0.05, capture_factory=lambda s, b: FakeCapture(frames * 100))
    cam.start()
    try:
        assert wait_until(lambda: cam.status()["frames"] > 20)
        f1 = cam.latest()
        time.sleep(0.1)
        f2 = cam.wait_frame(f1.seq, 1.0)
        assert f2 is not None and f2.seq > f1.seq + 1, "slow consumers must skip stale frames"
        assert cam.measured_fps() > 0
    finally:
        cam.stop()


def test_video_file_source_finishes(tmp_path):
    path = tmp_path / "clip.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 50, (320, 240))
    for i in range(10):
        w.write(bgr(i))
    w.release()
    cam = CameraReader(str(path), "auto", reconnect_seconds=0.05)
    assert cam.is_file
    cam.start()
    try:
        assert wait_until(lambda: cam.finished, timeout=10)
        assert cam.status()["frames"] == 10
    finally:
        cam.stop()


def _engine(config, embedder, classifier):
    store = LocalStore(config)
    store.ensure_site("site_live", "Live")
    engine = SemanticMemoryEngine(config, store, embedder, classifier, FullFrameDetector("solar_panel", 0.95))
    sid = store.create_session("site_live", "op", "offline", {})["session_id"]
    return store, engine, sid


def test_pipeline_skips_duplicate_frames_and_frame_skip(config, embedder, classifier):
    store, engine, sid = _engine(config, embedder, classifier)
    config["camera"]["frame_skip"] = 1
    events = []
    src = PushFrameSource()
    pipe = LivePipeline(config, engine, src, sid, events.append)
    pipe.start()
    try:
        frame = cv2.imencode(".jpg", bgr(3))[1].tobytes()
        for _ in range(3):
            src.push_jpeg(frame)
            time.sleep(0.3)
        assert wait_until(lambda: pipe.counters["seen"] >= 3)
        assert pipe.counters["processed"] == 1 and pipe.counters["skipped_duplicate"] == 2   # D8
        assert len(store.list_assets("site_live")) == 1
        assert [e["event"]["type"] for e in events if e["type"] == "event"] == ["new_asset"]
    finally:
        pipe.stop()

    pipe2 = LivePipeline({**config, "camera": {**config["camera"], "frame_skip": 3}}, engine, PushFrameSource(), sid, events.append)
    pipe2.source.start()
    pipe2.start()
    try:
        for i in range(6):
            pipe2.source.push_jpeg(cv2.imencode(".jpg", bgr(100 + i))[1].tobytes())
            time.sleep(0.2)
        assert wait_until(lambda: pipe2.counters["seen"] >= 6)
        assert pipe2.counters["skipped_frame_skip"] == 4 and pipe2.counters["processed"] == 2
    finally:
        pipe2.stop()


def test_tracked_asset_processed_once_per_track(config, embedder, classifier):
    """Consecutive frames of one asset → one observation (tracker, D8)."""
    store, engine, sid = _engine(config, embedder, classifier)
    pipe = LivePipeline(config, engine, PushFrameSource(), sid, lambda m: None)
    base = bgr(11)
    from ml.live.camera import Frame

    for i in range(4):
        noisy = np.clip(base.astype(np.int16) + np.random.default_rng(i).integers(-3, 4, base.shape), 0, 255).astype(np.uint8)
        pipe.process(Frame(noisy, time.time(), time.perf_counter(), i + 1))
    assert len(store.session_observations(sid)) == 1
    forced = pipe.force_capture()
    pipe.process(Frame(base, time.time(), time.perf_counter(), 9), forced)
    assert len(store.session_observations(sid)) == 2, "capture button forces a full observation"


def test_manager_restart_after_video_end_restores_detector(config, embedder, classifier, tmp_path):
    """A finished video source must not leak the full_frame detector into the next run."""
    from ml.live.manager import LiveManager

    path = tmp_path / "clip.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 50, (320, 240))
    for i in range(3):
        w.write(bgr(i))
    w.release()
    store, engine, sid = _engine(config, embedder, classifier)
    configured = engine.detector
    mgr = LiveManager(config, engine, store, lambda m: None)
    for _ in range(2):
        mgr.start(sid, str(path), "camera", "full_frame", "solar_panel")
        assert engine.detector is not configured
        assert wait_until(lambda: not mgr.running, timeout=10)
    mgr.stop()
    assert engine.detector is configured


def test_nfr1_alert_latency_measured_from_capture(config, embedder, classifier):
    """FR-15/NFR-1: a changed asset raises an alert whose capture→alert latency is recorded."""
    # The weight-free test classifier is low-confidence (Cf ≈ 0.55), so the alert levels are lowered;
    # the latency path under test is unchanged.
    config = {**config, "severity": {**config["severity"], "levels": {**config["severity"]["levels"], "priority": 0.2, "critical": 0.3}}}
    store, engine, sid = _engine(config, embedder, classifier)
    base = make_panel(21)
    engine.observe(base, "solar_panel", sid)
    sid2 = store.create_session("site_live", "op", "offline", {})["session_id"]
    events = []
    pipe = LivePipeline(config, engine, PushFrameSource(), sid2, events.append)
    from ml.live.camera import Frame
    from tests.conftest import add_crack

    img = cv2.cvtColor(np.asarray(add_crack(base, 10)), cv2.COLOR_RGB2BGR)
    pipe.process(Frame(img, time.time(), time.perf_counter(), 1))
    alerts = [e for e in events if e.get("alert")]
    assert alerts and alerts[0]["alert_latency_ms"] > 0
    p95 = pipe.latency.p95("alert")
    assert p95 is not None and p95 <= config["bench"]["targets"]["alert_p95_ms"]
    assert pipe.alerts and pipe.alerts[0]["asset_id"] == alerts[0]["event"]["asset_id"]


def test_overlay_is_non_obstructive(config):
    r = OverlayRenderer(config)
    frame = np.full((480, 640, 3), 127, np.uint8)
    track = {"bbox": [200, 150, 440, 330], "asset_id": "asset_1", "condition": {"label": "crack"},
             "severity_score": 0.9, "critical": True, "priority": True}
    alert = {"asset_id": "asset_1", "label": "crack", "severity_score": 0.9, "critical": True, "received": time.time()}
    out = r.render(frame, [track], [alert], {"session_id": "s", "camera_connected": True})
    centre = (slice(200, 280), slice(260, 380))
    assert np.array_equal(out[centre], frame[centre]), "the asset centre must stay unobstructed (FR-15)"
    assert not np.array_equal(out[: r.bar_h], frame[: r.bar_h]), "status bar drawn at the top edge"
    assert not np.array_equal(out[:, -5:], frame[:, -5:]), "critical alert → red border"
