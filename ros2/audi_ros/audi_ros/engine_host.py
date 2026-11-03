"""One engine per process, shared by perception_node and mission_node when run in the same
container (the launch file composes them in one process via a MultiThreadedExecutor)."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image


class EngineHost:
    _shared: Optional["EngineHost"] = None
    _lock = threading.Lock()

    def __init__(self, config_path: Optional[str] = None):
        from api.services import build_services
        from ml.config import load_config

        config = load_config(Path(config_path)) if config_path else load_config()
        self.services = build_services(config)
        self.session_id: Optional[str] = None

    @classmethod
    def shared(cls, config_path: Optional[str] = None) -> "EngineHost":
        with cls._lock:
            if cls._shared is None:
                cls._shared = cls(config_path)
            return cls._shared

    def start(self, site_id: str, operator: str, simulated: bool = False) -> Dict[str, Any]:
        res = self.services.mission.start(site_id, operator, simulated=simulated)
        self.session_id = res["session"]["session_id"]
        return res

    def end(self, session_id: str) -> Dict[str, Any]:
        rep = self.services.mission.end(session_id)
        if self.session_id == session_id:
            self.session_id = None
        return rep

    def process(self, rgb: np.ndarray, pose: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        evs = self.services.engine.process_frame(Image.fromarray(rgb), self.session_id, pose=pose)
        return [e.to_dict() for e in evs]
