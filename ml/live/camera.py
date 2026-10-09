"""Frame sources for the live pipeline.

`CameraReader` runs OpenCV capture on its own thread and keeps only the LATEST
frame (older frames are overwritten, so latency stays bounded however slow
inference is). It reconnects automatically when the device disappears and
accepts a webcam index / name, an RTSP/HTTP URL (drone feed) or a video file
(paced at the file's native frame rate). `PushFrameSource` exposes the same
interface for frames sent by the browser over the WebSocket.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import cv2
import numpy as np

from ml.config import cfg
from ml.live.devices import backend_for, resolve_source

logger = logging.getLogger(__name__)

FPS_WINDOW = 30


@dataclass
class Frame:
    image: np.ndarray        # BGR
    ts: float                # time.time() at capture
    mono: float              # time.perf_counter() at capture (latency measurement)
    seq: int


class FrameSource:
    """Latest-frame slot with a condition variable; subclasses fill it."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._latest: Optional[Frame] = None
        self._seq = 0
        self._stamps: deque = deque(maxlen=FPS_WINDOW)
        self.connected = False
        self.finished = False
        self.last_error: Optional[str] = None

    def _publish(self, image: np.ndarray) -> None:
        now_mono = time.perf_counter()
        with self._cond:
            self._seq += 1
            self._latest = Frame(image, time.time(), now_mono, self._seq)
            self._stamps.append(now_mono)
            self._cond.notify_all()

    def latest(self) -> Optional[Frame]:
        with self._cond:
            return self._latest

    def wait_frame(self, after_seq: int, timeout: float) -> Optional[Frame]:
        with self._cond:
            self._cond.wait_for(lambda: (self._latest is not None and self._latest.seq > after_seq) or self.finished, timeout)
            f = self._latest
            return f if f is not None and f.seq > after_seq else None

    def measured_fps(self) -> float:
        with self._cond:
            if len(self._stamps) < 2:
                return 0.0
            span = self._stamps[-1] - self._stamps[0]
            return (len(self._stamps) - 1) / span if span > 0 else 0.0

    def status(self) -> Dict[str, Any]:
        return {
            "connected": self.connected,
            "finished": self.finished,
            "frames": self._seq,
            "fps": round(self.measured_fps(), 2),
            "last_error": self.last_error,
        }

    def start(self) -> None:
        return None

    def stop(self) -> None:
        with self._cond:
            self.finished = True
            self._cond.notify_all()


class PushFrameSource(FrameSource):
    """Frames pushed from elsewhere (browser getUserMedia → WebSocket JPEG)."""

    kind = "browser"

    def push_jpeg(self, data: bytes) -> bool:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            self.last_error = "undecodable frame"
            return False
        self.connected = True
        self._publish(img)
        return True

    def status(self) -> Dict[str, Any]:
        return {**super().status(), "source": "browser"}


CaptureFactory = Callable[[Union[int, str], int], Any]


def _default_factory(source: Union[int, str], backend: int):
    return cv2.VideoCapture(source, backend)


class CameraReader(FrameSource):
    kind = "camera"

    def __init__(
        self,
        source: Union[int, str],
        backend: str = "auto",
        width: int = 0,
        height: int = 0,
        fps: float = 0,
        warmup_frames: int = 0,
        reconnect_seconds: float = 2.0,
        loop_video: bool = False,
        capture_factory: Optional[CaptureFactory] = None,
    ):
        super().__init__()
        self.source = resolve_source(source)
        self.backend_name = backend
        self.backend = backend_for(backend, self.source)
        self.width, self.height, self.fps = int(width), int(height), float(fps)
        self.warmup_frames = int(warmup_frames)
        self.reconnect_seconds = float(reconnect_seconds)
        self.loop_video = loop_video
        self.is_file = isinstance(self.source, str) and Path(self.source).exists()
        self._factory = capture_factory or _default_factory
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.reconnects = 0
        self.resolution = (0, 0)

    @classmethod
    def from_config(cls, config: Dict[str, Any], source: Optional[Union[int, str]] = None, **kw) -> "CameraReader":
        c = cfg(config, "camera")
        return cls(
            c["source"] if source is None else source, c["backend"], c["width"], c["height"], c["fps"],
            c["warmup_frames"], c["reconnect_seconds"], c["loop_video"], **kw,
        )

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.finished = False
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        super().stop()
        if self._thread:
            self._thread.join(timeout=self.reconnect_seconds + 2)

    def _open(self):
        cap = self._factory(self.source, self.backend)
        if cap is None or not cap.isOpened():
            raise RuntimeError(f"cannot open camera source {self.source!r}")
        if not self.is_file:
            if self.width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            if self.height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if self.fps:
                cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            for _ in range(self.warmup_frames):
                cap.read()
        return cap

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                cap = self._open()
            except Exception as exc:  # noqa: BLE001 - keep retrying until stopped
                self.connected = False
                self.last_error = str(exc)
                logger.warning("Camera %r unavailable (%s); retrying in %.1fs", self.source, exc, self.reconnect_seconds)
                self._stop.wait(self.reconnect_seconds)
                continue
            self.connected = True
            self.last_error = None
            file_fps = cap.get(cv2.CAP_PROP_FPS) if self.is_file else 0.0
            period = 1.0 / file_fps if file_fps and file_fps > 0 else 0.0
            next_t = time.perf_counter()
            try:
                while not self._stop.is_set():
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        if self.is_file and self.loop_video:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        if self.is_file:
                            self.connected = False
                            super().stop()
                            return
                        raise RuntimeError("frame grab failed (device unplugged?)")
                    self.resolution = (frame.shape[1], frame.shape[0])
                    self._publish(frame)
                    if period:
                        next_t += period
                        delay = next_t - time.perf_counter()
                        if delay > 0:
                            self._stop.wait(delay)
                        else:
                            next_t = time.perf_counter()
            except Exception as exc:  # noqa: BLE001
                self.last_error = str(exc)
                self.connected = False
                self.reconnects += 1
                logger.warning("Camera %r lost (%s); reconnecting", self.source, exc)
                self._stop.wait(self.reconnect_seconds)
            finally:
                cap.release()

    def status(self) -> Dict[str, Any]:
        return {
            **super().status(),
            "source": self.source,
            "backend": self.backend_name,
            "resolution": list(self.resolution),
            "reconnects": self.reconnects,
            "is_file": self.is_file,
        }
