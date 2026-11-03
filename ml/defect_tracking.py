"""Defect-level tracking across visits (upgrade 2.2).

Each defect instance (one crack, one rust patch) is its own object, positioned in the
normalised coordinates of its asset's reference view. On a new visit the defect regions found
in the current view (segmentation ∪ defect-like change regions, warped into the reference
view) are matched to the asset's active instances by optimal assignment (Hungarian) on a cost
that combines centroid distance and polygon-box overlap; same defect family only.

  matched region   → new measurement on that instance (size history grows)
  unmatched region → a new instance
  unmatched instance → missed_visits += 1; after `resolve_after_missed` consecutive misses
                       while the asset is in view, the instance is marked resolved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ml.config import get_nested
from ml.forecast import family


@dataclass
class MeasuredRegion:
    polygon: List[List[float]]
    bbox: List[float]
    centroid: List[float]
    area_norm: float
    area_px: float
    length_px: float
    defect_type: str
    area_cm2: Optional[float] = None
    length_cm: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def _box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def polygon_bbox(poly: Sequence[Sequence[float]]) -> List[float]:
    arr = np.asarray(poly, dtype=np.float64)
    return [float(arr[:, 0].min()), float(arr[:, 1].min()), float(arr[:, 0].max()), float(arr[:, 1].max())]


def match_regions(
    instances: Sequence[Dict[str, Any]],
    regions: Sequence[MeasuredRegion],
    max_distance: float = 0.15,
    iou_weight: float = 0.5,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Return (pairs (instance_idx, region_idx), unmatched region idx, unmatched instance idx)."""
    if not instances or not regions:
        return [], list(range(len(regions))), list(range(len(instances)))
    from scipy.optimize import linear_sum_assignment

    big = 1e6
    cost = np.full((len(instances), len(regions)), big)
    for i, inst in enumerate(instances):
        ib = polygon_bbox(inst["polygon"])
        # a growing crack's centroid moves along its path: allow the box extent as extra slack
        slack = 0.5 * max(ib[2] - ib[0], ib[3] - ib[1])
        for j, r in enumerate(regions):
            if family(inst["defect_type"]) != family(r.defect_type):
                continue
            d = float(np.hypot(inst["centroid_x"] - r.centroid[0], inst["centroid_y"] - r.centroid[1]))
            ov = _box_iou(ib, r.bbox)
            if d > max_distance + slack and ov <= 0:
                continue
            cost[i, j] = d - iou_weight * ov
    rows, cols = linear_sum_assignment(cost)
    pairs = [(int(r), int(c)) for r, c in zip(rows, cols) if cost[r, c] < big]
    used_r = {c for _, c in pairs}
    used_i = {r for r, _ in pairs}
    return pairs, [j for j in range(len(regions)) if j not in used_r], [i for i in range(len(instances)) if i not in used_i]


def merge_overlapping(regions: List[MeasuredRegion], gap: float = 0.02) -> List[MeasuredRegion]:
    """Union regions of the same family whose boxes touch (a crack split by a seam is one crack)."""
    out: List[MeasuredRegion] = []
    for r in sorted(regions, key=lambda x: -x.area_norm):
        hit = None
        for o in out:
            if family(o.defect_type) != family(r.defect_type):
                continue
            a, b = o.bbox, r.bbox
            if a[0] - gap <= b[2] and b[0] - gap <= a[2] and a[1] - gap <= b[3] and b[1] - gap <= a[3]:
                hit = o
                break
        if hit is None:
            out.append(r)
            continue
        tot = hit.area_norm + r.area_norm
        hit.centroid = [(hit.centroid[k] * hit.area_norm + r.centroid[k] * r.area_norm) / max(tot, 1e-12) for k in range(2)]
        hit.bbox = [min(hit.bbox[0], r.bbox[0]), min(hit.bbox[1], r.bbox[1]), max(hit.bbox[2], r.bbox[2]), max(hit.bbox[3], r.bbox[3])]
        hit.polygon = [[hit.bbox[0], hit.bbox[1]], [hit.bbox[2], hit.bbox[1]], [hit.bbox[2], hit.bbox[3]], [hit.bbox[0], hit.bbox[3]]]
        hit.area_norm = tot
        hit.area_px += r.area_px
        hit.length_px += r.length_px
        if hit.area_cm2 is not None and r.area_cm2 is not None:
            hit.area_cm2 += r.area_cm2
        if hit.length_cm is not None and r.length_cm is not None:
            hit.length_cm += r.length_cm
    return out


def tracking_params(config: Dict[str, Any]) -> Dict[str, float]:
    c = get_nested(config, "defect_tracking", default={}) or {}
    return {
        "max_distance": float(c.get("max_centroid_distance", 0.15)),
        "iou_weight": float(c.get("iou_weight", 0.5)),
        "resolve_after_missed": int(c.get("resolve_after_missed", 3)),
        "min_area_norm": float(c.get("min_area_fraction", 0.0005)),
    }
