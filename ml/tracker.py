"""Per-session IoU + embedding + appearance tracker (D8, NFR-1).

Consecutive detections of one physical asset are linked into a track so the
identity cascade and the store run once per track rather than once per frame.
A detection continues a track only if its box overlaps, its global embedding is
similar AND a small grey thumbnail correlates with the track's last one: global
embeddings of different assets of one type overlap (e.g. two concrete walls), and
with full-frame detection the box test always passes. Splitting a track by
mistake only costs one identity-cascade run; merging two assets would lose one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from ml.imaging import iou


def thumbnail(crop: Image.Image, size: int) -> np.ndarray:
    """Zero-mean, unit-norm grey thumbnail; the dot product of two is their NCC."""
    g = cv2.resize(np.asarray(crop.convert("L")), (size, size), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    g -= g.mean()
    return g / (float(np.linalg.norm(g)) + 1e-6)


@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]
    asset_type: str
    embedding: np.ndarray
    last_frame: int
    hits: int = 1
    asset_id: Optional[str] = None
    identity_stage: Optional[str] = None
    last_label: Optional[str] = None
    observed: bool = False
    thumb: Optional[np.ndarray] = None
    info: Dict = field(default_factory=dict)


class SessionTracker:
    def __init__(
        self,
        iou_threshold: float,
        embedding_similarity: float,
        max_missed_frames: int,
        thumbnail_size: int,
        thumbnail_ncc: float,
    ):
        self.iou_threshold = iou_threshold
        self.embedding_similarity = embedding_similarity
        self.max_missed = max_missed_frames
        self.thumbnail_size = thumbnail_size
        self.thumbnail_ncc = thumbnail_ncc
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1
        self.frame_index = 0

    @classmethod
    def from_config(cls, config: Dict) -> "SessionTracker":
        t = config["tracker"]
        return cls(
            float(t["iou_threshold"]), float(t["embedding_similarity"]), int(t["max_missed_frames"]),
            int(t["thumbnail_size"]), float(t["thumbnail_ncc"]),
        )

    def update(
        self,
        boxes: Sequence[Tuple[int, int, int, int]],
        asset_types: Sequence[str],
        embeddings: np.ndarray,
        crops: Optional[Sequence[Image.Image]] = None,
    ) -> List[Track]:
        """Associate detections with tracks (greedy by IoU, gated by embedding cosine and thumbnail NCC)."""
        self.frame_index += 1
        thumbs = [thumbnail(c, self.thumbnail_size) for c in crops] if crops is not None else [None] * len(boxes)
        pairs = []
        for di, (box, atype) in enumerate(zip(boxes, asset_types)):
            for tid, tr in self.tracks.items():
                if tr.asset_type != atype:
                    continue
                ov = iou(box, tr.bbox)
                if ov < self.iou_threshold:
                    continue
                sim = float(np.dot(embeddings[di], tr.embedding))
                if sim < self.embedding_similarity:
                    continue
                if thumbs[di] is not None and tr.thumb is not None and float(np.dot(thumbs[di], tr.thumb)) < self.thumbnail_ncc:
                    continue
                pairs.append((ov + sim, di, tid))
        pairs.sort(reverse=True)
        assigned: Dict[int, int] = {}
        used_tracks = set()
        for _, di, tid in pairs:
            if di in assigned or tid in used_tracks:
                continue
            assigned[di] = tid
            used_tracks.add(tid)

        out: List[Track] = []
        for di, (box, atype) in enumerate(zip(boxes, asset_types)):
            if di in assigned:
                tr = self.tracks[assigned[di]]
                tr.bbox = tuple(box)
                tr.embedding = embeddings[di]
                tr.last_frame = self.frame_index
                tr.hits += 1
            else:
                tr = Track(self._next_id, tuple(box), atype, embeddings[di], self.frame_index)
                self.tracks[tr.track_id] = tr
                self._next_id += 1
            tr.thumb = thumbs[di]
            out.append(tr)

        # A track survives missed frames only while its region stays free: once another
        # track occupies that region the view has moved on, and re-attaching to the old
        # track later would merge two different assets.
        current = {tr.track_id for tr in out}
        stale = {
            t for t, tr in self.tracks.items()
            if self.frame_index - tr.last_frame > self.max_missed
            or (t not in current and any(iou(tr.bbox, o.bbox) >= self.iou_threshold for o in out))
        }
        for tid in stale:
            del self.tracks[tid]
        return out

    def get(self, track_id: int) -> Optional[Track]:
        return self.tracks.get(track_id)

    def reset(self) -> None:
        self.tracks.clear()
        self.frame_index = 0
