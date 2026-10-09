"""Non-obstructive live overlay (FR-15, SRS §6.3).

Assets get thin corner brackets (no filled boxes over the asset) with a small
label on the bracket edge; status lives in a thin top bar and recent alerts in
a narrow side panel, so the centre of the operator's view stays clear.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np

from ml.config import cfg

Color = Tuple[int, int, int]


class OverlayRenderer:
    def __init__(self, config: Dict[str, Any]):
        o = cfg(config, "live", "overlay")
        self.bar_h = int(o["bar_height"])
        self.side_w = int(o["side_width"])
        self.thick = int(o["box_thickness"])
        self.font_scale = float(o["font_scale"])
        self.colors: Dict[str, Color] = {k: tuple(int(c) for c in v) for k, v in o["colors"].items()}
        self.bracket_fraction = float(o["bracket_fraction"])
        self.panel_alpha = float(o["panel_alpha"])
        self.gap = int(o["line_gap"])
        self.line_h = int(o["line_height"])
        self.alert_display_s = float(cfg(config, "live", "alert_display_s"))
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def color_for(self, item: Dict[str, Any]) -> Color:
        if item.get("critical"):
            return self.colors["critical"]
        if item.get("priority"):
            return self.colors["priority"]
        if item.get("type") == "new_asset":
            return self.colors["new"]
        if item.get("severity_level") == "medium":
            return self.colors["medium"]
        return self.colors["normal"]

    def _text(self, img: np.ndarray, text: str, org: Tuple[int, int], color: Color, scale: float = None) -> None:
        cv2.putText(img, text, org, self.font, scale or self.font_scale, color, 1, cv2.LINE_AA)

    def _panel(self, img: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
        roi = img[y0:y1, x0:x1]
        if roi.size:
            img[y0:y1, x0:x1] = (roi * (1 - self.panel_alpha)).astype(np.uint8)

    def _brackets(self, img: np.ndarray, box: Sequence[int], color: Color) -> None:
        x1, y1, x2, y2 = [int(v) for v in box]
        lx, ly = max(self.thick * 2, int((x2 - x1) * self.bracket_fraction)), max(self.thick * 2, int((y2 - y1) * self.bracket_fraction))
        for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(img, (cx, cy), (cx + dx * lx, cy), color, self.thick, cv2.LINE_AA)
            cv2.line(img, (cx, cy), (cx, cy + dy * ly), color, self.thick, cv2.LINE_AA)

    def render(self, frame: np.ndarray, tracks: List[Dict[str, Any]], alerts: List[Dict[str, Any]], status: Dict[str, Any]) -> np.ndarray:
        img = frame.copy()
        h, w = img.shape[:2]
        line_h = self.line_h

        for t in tracks:
            color = self.color_for(t)
            self._brackets(img, t["bbox"], color)
            cond = t.get("condition", {}) or {}
            label = f"{t.get('asset_id') or 'new'} · {cond.get('label', '?')} · S={t.get('severity_score', 0):.2f}"
            x1, y1 = int(t["bbox"][0]), int(t["bbox"][1])
            y_text = y1 - self.gap if y1 - self.gap > self.bar_h + line_h else int(t["bbox"][3]) + line_h
            self._text(img, label, (x1, y_text), color)

        self._panel(img, 0, 0, w, self.bar_h)
        cam = "camera ok" if status.get("camera_connected") else "camera DISCONNECTED"
        bar = (f"session {status.get('session_id', '-')} | {cam} | capture {status.get('capture_fps', 0):.1f} fps | "
               f"processed {status.get('processed_fps', 0):.1f} fps | alert latency p95 {status.get('latency_p95_ms') or 0:.0f} ms")
        self._text(img, bar, (8, int(self.bar_h * 0.65)), (255, 255, 255))

        recent = [a for a in alerts if time.time() - a.get("received", 0) < self.alert_display_s]
        if recent:
            x0 = w - self.side_w
            panel_h = min(h, self.bar_h + self.gap + line_h * (2 * len(recent) + 1))
            self._panel(img, x0, self.bar_h, w, panel_h)
            y = self.bar_h + line_h
            for a in recent:
                color = self.colors["critical"] if a.get("critical") else self.colors["priority"]
                self._text(img, f"{'CRITICAL' if a.get('critical') else 'PRIORITY'} {a.get('asset_id')}", (x0 + 8, y), color)
                self._text(img, f"{a.get('label')} S={a.get('severity_score', 0):.2f}", (x0 + 8, y + line_h), (255, 255, 255))
                y += 2 * line_h
            if any(a.get("critical") for a in recent):
                cv2.rectangle(img, (0, 0), (w - 1, h - 1), self.colors["critical"], max(2, self.thick * 2))
        return img
