"""Local ↔ Cloud synchronisation (FR-1/2/3/6/7, NFR-3, SRS §5.3).

Every local write is already in `sync_outbox`. `SyncManager` flushes it
whenever the cloud is reachable: at session start/end (bounded by
cloud.sync_timeout_s — NFR-3) and opportunistically in a background thread.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, Optional

from memory.cloud_store_interface import CloudStoreInterface, CloudUnavailable
from memory.store import LocalStore
from ml.config import cfg
from ml.schemas import utc_now_iso

logger = logging.getLogger(__name__)


class SyncManager:
    def __init__(self, config: Dict[str, Any], store: LocalStore, cloud: Optional[CloudStoreInterface]):
        c = cfg(config, "cloud")
        self.enabled = bool(c["enabled"]) and cloud is not None
        self.cloud = cloud
        self.store = store
        self.connect_timeout = float(c["connect_timeout_s"])
        self.sync_timeout = float(c["sync_timeout_s"])
        self.flush_interval = float(c["flush_interval_s"])
        self.batch_size = int(c["batch_size"])
        self.online = False
        self.last_sync_ts: Optional[str] = None
        self.last_error: Optional[str] = None
        self._flush_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # FR-1
    def check_connectivity(self) -> bool:
        if not self.enabled:
            self.online = False
            return False
        self.online = self.cloud.is_available(self.connect_timeout)
        return self.online

    def flush_once(self, deadline: Optional[float] = None, cancel: Optional[threading.Event] = None) -> int:
        """Push pending outbox batches until empty, deadline, or error. Returns records acked."""
        if not self.enabled:
            return 0
        sent = 0
        with self._flush_lock:
            while not (cancel and cancel.is_set()):
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    break
                batch = self.store.pending_outbox(self.batch_size)
                if not batch:
                    break
                timeout = self.sync_timeout if remaining is None else max(0.1, remaining)
                try:
                    acked = self.cloud.push_changes(batch, timeout)
                except CloudUnavailable as exc:
                    self.store.bump_outbox_attempts([b["change_id"] for b in batch])
                    self.online = False
                    self.last_error = str(exc)
                    raise
                self.store.mark_outbox_sent(acked)
                sent += len(acked)
                if not acked:
                    break
        if sent:
            self.last_sync_ts = utc_now_iso()
        return sent

    def _bounded(self, work: Callable[[threading.Event, float], Dict[str, Any]]) -> Dict[str, Any]:
        """Run `work` with a hard wall-clock limit (NFR-3); on timeout proceed offline."""
        started = time.monotonic()
        deadline = started + self.sync_timeout
        cancel = threading.Event()
        box: Dict[str, Any] = {}

        def runner():
            try:
                box["result"] = work(cancel, deadline)
            except Exception as exc:  # noqa: BLE001 - surfaced as offline mode
                box["error"] = str(exc)

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        t.join(self.sync_timeout)
        elapsed = time.monotonic() - started
        if t.is_alive():
            cancel.set()
            self.online = False
            self.last_error = f"sync exceeded {self.sync_timeout:.1f}s"
            return {"online": False, "mode": "offline", "timed_out": True, "elapsed_s": elapsed, "error": self.last_error}
        if "error" in box:
            self.online = False
            self.last_error = box["error"]
            return {"online": False, "mode": "offline", "timed_out": False, "elapsed_s": elapsed, "error": box["error"]}
        return {**box["result"], "online": True, "mode": "online", "timed_out": False, "elapsed_s": elapsed}

    # FR-1/2/3 at mission start
    def start_sync(self, site_id: str) -> Dict[str, Any]:
        if not self.check_connectivity():
            return {"online": False, "mode": "offline", "timed_out": False, "elapsed_s": 0.0, "error": "cloud unreachable"}

        def work(cancel: threading.Event, deadline: float) -> Dict[str, Any]:
            pushed = self.flush_once(deadline, cancel)
            records = self.cloud.pull_site(site_id, max(0.1, deadline - time.monotonic()))
            if cancel.is_set():
                return {"pushed": pushed, "pulled": {}}
            pulled = self.store.apply_remote(records)
            self.last_sync_ts = utc_now_iso()
            return {"pushed": pushed, "pulled": pulled}

        return self._bounded(work)

    # FR-6/FR-7 at mission end
    def end_sync(self) -> Dict[str, Any]:
        if not self.check_connectivity():
            return {
                "online": False,
                "mode": "offline",
                "timed_out": False,
                "elapsed_s": 0.0,
                "queued": self.store.outbox_counts()["pending"],
                "error": "cloud unreachable",
            }

        def work(cancel: threading.Event, deadline: float) -> Dict[str, Any]:
            return {"pushed": self.flush_once(deadline, cancel)}

        result = self._bounded(work)
        result["queued"] = self.store.outbox_counts()["pending"]
        return result

    # FR-7 / §5.3 opportunistic flushing
    def _loop(self) -> None:
        while not self._stop.wait(self.flush_interval):
            if self.store.outbox_counts()["pending"] == 0:
                continue
            if not self.check_connectivity():
                continue
            try:
                n = self.flush_once()
                if n:
                    logger.info("Background sync pushed %d changes", n)
            except CloudUnavailable as exc:
                logger.info("Background sync deferred: %s", exc)

    def start_background(self) -> None:
        if not self.enabled or (self._thread and self._thread.is_alive()):
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="audi-sync", daemon=True)
        self._thread.start()

    def stop_background(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.flush_interval)

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "online": self.online,
            "last_sync_ts": self.last_sync_ts,
            "last_error": self.last_error,
            "outbox": self.store.outbox_counts(),
        }
