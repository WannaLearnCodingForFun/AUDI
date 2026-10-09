"""Live inspection pipeline.

capture (own thread, latest frame only) → frame_skip → duplicate-frame skip (D8)
→ detect → track → identity → condition → change → severity → store → events.

Inference runs on a worker thread and never blocks capture. Every processed
frame records per-stage latency and capture→event latency (NFR-1 is measured on
events that raise an alert).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from ml.config import cfg
from ml.live.camera import Frame, FrameSource
from ml.semantic_memory import AssetEvent, SemanticMemoryEngine
from ml.tracker import SessionTracker

logger = logging.getLogger(__name__)

Publish = Callable[[Dict[str, Any]], None]


def percentile(values, q: float) -> Optional[float]:
    return float(np.percentile(np.asarray(values), q)) if len(values) else None


class LatencyStats:
    def __init__(self, window: int):
        self.window = window
        self.samples: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()

    def add(self, name: str, value_ms: float) -> None:
        with self._lock:
            self.samples.setdefault(name, deque(maxlen=self.window)).append(float(value_ms))

    def summary(self) -> Dict[str, Dict[str, Optional[float]]]:
        with self._lock:
            return {k: {"p50": percentile(v, 50), "p95": percentile(v, 95), "n": len(v)} for k, v in self.samples.items()}

    def p95(self, name: str) -> Optional[float]:
        with self._lock:
            return percentile(self.samples.get(name, []), 95)


class LivePipeline:
    def __init__(self, config: Dict[str, Any], engine: SemanticMemoryEngine, source: FrameSource, session_id: str, publish: Publish):
        self.config = config
        self.engine = engine
        self.source = source
        self.session_id = session_id
        self.publish = publish
        live, cam, rt = cfg(config, "live"), cfg(config, "camera"), cfg(config, "runtime")
        self.frame_skip = max(1, int(cam["frame_skip"]))
        self.dup_mad = float(rt["duplicate_frame_mad"])
        self.dup_size = int(rt["duplicate_frame_size"])
        self.cooldown = float(live["alert_cooldown_s"])
        self.max_alerts = int(live["max_alerts_shown"])
        self.frame_wait = float(live["frame_wait_s"])
        self.tracker = SessionTracker.from_config(config)
        self.latency = LatencyStats(int(live["latency_window"]))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._force_track: Optional[int] = None
        self._last_gray: Optional[np.ndarray] = None
        self._last_alert: Dict[str, float] = {}
        self._processed_stamps: Deque[float] = deque(maxlen=30)
        self.tracks: List[Dict[str, Any]] = []
        self.alerts: Deque[Dict[str, Any]] = deque(maxlen=self.max_alerts)
        self.counters = {"seen": 0, "processed": 0, "skipped_frame_skip": 0, "skipped_duplicate": 0, "events": 0, "alerts": 0, "errors": 0}
        self.started_ts: Optional[float] = None

    # ── control ───────────────────────────────────────────────────────────
    def start(self) -> None:
        self.source.start()
        self._stop.clear()
        self.started_ts = time.time()
        self._thread = threading.Thread(target=self._run, name="live-pipeline", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.source.stop()
        if self._thread:
            self._thread.join(timeout=10)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def force_capture(self, track_id: Optional[int] = None) -> Optional[int]:
        """Force a full observation of `track_id` (default: the largest visible track) on the next frame."""
        with self._lock:
            if track_id is None and self.tracks:
                largest = max(self.tracks, key=lambda t: (t["bbox"][2] - t["bbox"][0]) * (t["bbox"][3] - t["bbox"][1]))
                track_id = largest.get("track_id")
            self._force_track = track_id
            return track_id

    # ── worker ────────────────────────────────────────────────────────────
    def _is_duplicate(self, bgr: np.ndarray) -> bool:
        gray = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (self.dup_size, self.dup_size), interpolation=cv2.INTER_AREA).astype(np.float32)
        dup = self._last_gray is not None and float(np.mean(np.abs(gray - self._last_gray))) < self.dup_mad
        if not dup:
            self._last_gray = gray
        return dup

    def _run(self) -> None:
        last_seq = 0
        while not self._stop.is_set():
            frame = self.source.wait_frame(last_seq, self.frame_wait)
            if frame is None:
                if self.source.finished:
                    logger.info("Frame source finished; live pipeline stopping")
                    self.publish({"type": "source_finished", "session_id": self.session_id})
                    break
                continue
            last_seq = frame.seq
            self.counters["seen"] += 1
            with self._lock:
                forced = self._force_track
            if forced is None and self.counters["seen"] % self.frame_skip != 0:
                self.counters["skipped_frame_skip"] += 1
                continue
            if forced is None and self._is_duplicate(frame.image):
                self.counters["skipped_duplicate"] += 1
                continue
            self.process(frame, forced)

    def process(self, frame: Frame, forced: Optional[int] = None) -> List[AssetEvent]:
        t0 = time.perf_counter()
        pil = Image.fromarray(cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB))
        try:
            events = self.engine.process_frame(pil, self.session_id, self.tracker, force_track_id=forced)
        except Exception as exc:  # noqa: BLE001 - one bad frame must not stop the live session
            self.counters["errors"] += 1
            logger.exception("Live frame failed")
            self.publish({"type": "error", "error": str(exc)})
            return []
        if forced is not None:
            with self._lock:
                self._force_track = None
        done = time.perf_counter()
        self.latency.add("frame_total", (done - t0) * 1000)
        self.latency.add("capture_to_result", (done - frame.mono) * 1000)
        self.counters["processed"] += 1
        self._processed_stamps.append(done)

        tracks = []
        for ev in events:
            for stage, ms in ev.latency_ms.items():
                self.latency.add(stage, ms)
            tracks.append({
                "track_id": ev.track_id, "bbox": ev.bbox, "asset_id": ev.asset_id, "type": ev.type,
                "condition": {"label": ev.condition.get("label"), "confidence": ev.condition.get("confidence")},
                "severity_score": ev.severity_score, "severity_level": ev.severity_level,
                "priority": ev.priority, "critical": ev.critical,
            })
            if ev.type == "tracked":
                continue
            self.counters["events"] += 1
            alert = self._alert_for(ev)
            payload = {"type": "event", "event": ev.to_dict(), "alert": alert, "capture_ts": frame.ts}
            if alert:
                latency_ms = (time.perf_counter() - frame.mono) * 1000
                self.latency.add("alert", latency_ms)
                payload["alert_latency_ms"] = latency_ms
            self.publish(payload)
        with self._lock:
            self.tracks = tracks
        return events

    def _alert_for(self, ev: AssetEvent) -> Optional[str]:
        if ev.type == "error" or not (ev.priority or ev.critical):
            return None
        key = f"{ev.asset_id}:{ev.condition.get('label')}"
        now = time.time()
        if now - self._last_alert.get(key, 0.0) < self.cooldown:
            return None
        self._last_alert[key] = now
        level = "critical" if ev.critical else "priority"
        self.counters["alerts"] += 1
        self.alerts.appendleft({
            "asset_id": ev.asset_id, "label": ev.condition.get("label"), "severity_score": ev.severity_score,
            "critical": ev.critical, "received": now, "level": level,
        })
        return level

    # ── views ─────────────────────────────────────────────────────────────
    def processed_fps(self) -> float:
        s = list(self._processed_stamps)
        if len(s) < 2 or s[-1] == s[0]:
            return 0.0
        return (len(s) - 1) / (s[-1] - s[0])

    def status(self) -> Dict[str, Any]:
        src = self.source.status()
        return {
            "running": self.running,
            "session_id": self.session_id,
            "camera": src,
            "camera_connected": bool(src.get("connected")),
            "capture_fps": src.get("fps", 0.0),
            "processed_fps": round(self.processed_fps(), 2),
            "latency_ms": self.latency.summary(),
            "latency_p95_ms": self.latency.p95("alert") or self.latency.p95("capture_to_result"),
            "counters": dict(self.counters),
            "tracks": list(self.tracks),
            "alerts": list(self.alerts),
        }
