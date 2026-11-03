"""Site map as the memory backbone (upgrade 1.2).

A site is flown once and stitched into an orthomosaic (OpenDroneMap, see
`scripts/build_site_map.py`). Each asset is a polygon (`assets.footprint`, lat/lon vertices)
on that map. On every flight a frame is registered to the map (SIFT + RANSAC homography,
optionally restricted to a window around the drone's GPS fix), each detection's anchor pixel
is mapped into the orthomosaic, converted to lat/lon with the map's geotransform, and looked
up against the footprints.

Geotransform (GDAL order): lon = gt[0] + col·gt[1] + row·gt[2]; lat = gt[3] + col·gt[4] + row·gt[5].
It is read from GeoTIFF tags (ModelTiepoint + ModelPixelScale, EPSG:4326) or a sidecar
`<image>.json` holding {"geotransform": [...]}.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from ml.config import get_nested

logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None    # orthomosaics are large by design

TAG_MODEL_PIXEL_SCALE = 33550
TAG_MODEL_TIEPOINT = 33922


def read_geotransform(path: Path) -> Optional[List[float]]:
    side = Path(str(path) + ".json")
    if side.exists():
        gt = json.loads(side.read_text()).get("geotransform")
        if gt:
            return [float(v) for v in gt]
    try:
        with Image.open(path) as im:
            tags = getattr(im, "tag_v2", None)
            if tags and TAG_MODEL_PIXEL_SCALE in tags and TAG_MODEL_TIEPOINT in tags:
                sx, sy = tags[TAG_MODEL_PIXEL_SCALE][:2]
                i, j, _, x, y, _ = tags[TAG_MODEL_TIEPOINT][:6]
                x0 = x - i * sx
                y0 = y + j * sy
                return [float(x0), float(sx), 0.0, float(y0), 0.0, -float(sy)]
    except Exception:  # noqa: BLE001 - unreadable tags → caller supplies a geotransform
        logger.exception("Could not read GeoTIFF tags from %s", path)
    return None


def pixel_to_geo(gt: Sequence[float], col: float, row: float) -> Tuple[float, float]:
    """(lat, lon) of map pixel (col, row)."""
    lon = gt[0] + col * gt[1] + row * gt[2]
    lat = gt[3] + col * gt[4] + row * gt[5]
    return lat, lon


def geo_to_pixel(gt: Sequence[float], lat: float, lon: float) -> Tuple[float, float]:
    a = np.array([[gt[1], gt[2]], [gt[4], gt[5]]], dtype=np.float64)
    col, row = np.linalg.solve(a, np.array([lon - gt[0], lat - gt[3]]))
    return float(col), float(row)


def point_in_polygon(x: float, y: float, poly: Sequence[Sequence[float]]) -> bool:
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i][0], poly[i][1]
        xj, yj = poly[j][0], poly[j][1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def polygon_centroid(poly: Sequence[Sequence[float]]) -> Tuple[float, float]:
    arr = np.asarray(poly, dtype=np.float64)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


@dataclass
class Footprint:
    asset_id: str
    asset_type: str
    polygon_px: List[Tuple[float, float]]   # in map pixels (col, row)


@dataclass
class Registration:
    homography: np.ndarray          # frame px → map px
    inliers: int
    inlier_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return {"inliers": self.inliers, "inlier_ratio": self.inlier_ratio}


@dataclass
class FootprintHit:
    asset_id: str
    asset_type: str
    map_px: Tuple[float, float]
    latitude: float
    longitude: float


@dataclass
class SiteMap:
    map_id: str
    site_id: str
    image: np.ndarray                   # grey uint8 orthomosaic, possibly downscaled
    geotransform: List[float]           # for the stored (downscaled) image
    footprints: List[Footprint] = field(default_factory=list)

    @classmethod
    def load(cls, map_id: str, site_id: str, image_path: Path, geotransform: Optional[Sequence[float]],
             max_side: int = 4096) -> "SiteMap":
        with Image.open(image_path) as im:
            gray = np.asarray(im.convert("L"))
        gt = list(geotransform) if geotransform else read_geotransform(image_path)
        if gt is None:
            raise ValueError(f"No geotransform for site map {image_path}")
        h, w = gray.shape
        scale = min(1.0, max_side / float(max(h, w)))
        if scale < 1.0:
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            gt = [gt[0], gt[1] / scale, gt[2] / scale, gt[3], gt[4] / scale, gt[5] / scale]
        return cls(map_id, site_id, gray, gt)

    def set_footprints(self, assets: Sequence[Dict[str, Any]]) -> None:
        """Assets with a lat/lon `footprint` polygon ([[lat, lon], ...])."""
        out = []
        for a in assets:
            poly = a.get("footprint")
            if not poly or len(poly) < 3:
                continue
            out.append(Footprint(a["asset_id"], a["asset_type"], [geo_to_pixel(self.geotransform, p[0], p[1]) for p in poly]))
        self.footprints = out

    def lookup_px(self, col: float, row: float, asset_type: Optional[str] = None) -> Optional[FootprintHit]:
        for fp in self.footprints:
            if asset_type is not None and fp.asset_type != asset_type:
                continue
            if point_in_polygon(col, row, fp.polygon_px):
                lat, lon = pixel_to_geo(self.geotransform, col, row)
                return FootprintHit(fp.asset_id, fp.asset_type, (col, row), lat, lon)
        return None

    def lookup_geo(self, lat: float, lon: float, asset_type: Optional[str] = None) -> Optional[FootprintHit]:
        col, row = geo_to_pixel(self.geotransform, lat, lon)
        return self.lookup_px(col, row, asset_type)


class MapRegistrar:
    """Registers live frames to a site map with SIFT + RANSAC homography."""

    def __init__(self, config: Dict[str, Any]):
        c = get_nested(config, "site_map", default={}) or {}
        self.frame_max_side = int(c.get("frame_max_side", 960))
        self.window_m = float(c.get("search_window_m", 120.0))
        self.min_inliers = int(c.get("min_inliers", 25))
        self.min_ratio = float(c.get("min_inlier_ratio", 0.25))
        self.ratio = float(c.get("lowe_ratio", 0.75))
        self.reproj = float(c.get("ransac_reproj_threshold", 6.0))
        n = int(c.get("max_keypoints", 4000))
        self._sift = cv2.SIFT_create(nfeatures=n) if hasattr(cv2, "SIFT_create") else cv2.ORB_create(nfeatures=n)
        self._norm = cv2.NORM_L2 if hasattr(cv2, "SIFT_create") else cv2.NORM_HAMMING
        self._matcher = cv2.BFMatcher(self._norm)
        self._map_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._lock = threading.Lock()

    def _map_features(self, site_map: SiteMap):
        with self._lock:
            hit = self._map_cache.get(site_map.map_id)
            if hit is None:
                kps, desc = self._sift.detectAndCompute(site_map.image, None)
                pts = np.array([k.pt for k in kps], dtype=np.float32).reshape(-1, 2)
                hit = (pts, desc)
                self._map_cache[site_map.map_id] = hit
            return hit

    def register(self, frame: Image.Image, site_map: SiteMap, near: Optional[Tuple[float, float]] = None) -> Optional[Registration]:
        """Frame → map homography, or None when the frame cannot be placed on the map."""
        gray = np.asarray(frame.convert("L"))
        h, w = gray.shape
        s = min(1.0, self.frame_max_side / float(max(h, w)))
        small = cv2.resize(gray, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1.0 else gray
        with self._lock:
            kps, desc = self._sift.detectAndCompute(small, None)
        if desc is None or len(kps) < 8:
            return None
        q = np.array([k.pt for k in kps], dtype=np.float32) / s
        m_pts, m_desc = self._map_features(site_map)
        if m_desc is None or len(m_pts) < 8:
            return None
        if near is not None:
            # GPS prior: only map keypoints within the search window of the drone position
            col, row = geo_to_pixel(site_map.geotransform, near[0], near[1])
            m_per_px = abs(site_map.geotransform[5]) * 111_320.0
            radius_px = self.window_m / max(m_per_px, 1e-9)
            keep = np.hypot(m_pts[:, 0] - col, m_pts[:, 1] - row) <= radius_px
            if keep.sum() >= 8:
                m_pts, m_desc = m_pts[keep], m_desc[keep]
        pairs = self._matcher.knnMatch(desc, m_desc, k=2)
        good = [m for m, *rest in pairs if rest and m.distance < self.ratio * rest[0].distance]
        if len(good) < max(8, self.min_inliers // 2):
            return None
        src = q[[m.queryIdx for m in good]]
        dst = m_pts[[m.trainIdx for m in good]]
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.reproj)
        if H is None or mask is None:
            return None
        inliers = int(mask.sum())
        ratio = inliers / float(len(good))
        if inliers < self.min_inliers or ratio < self.min_ratio:
            return None
        return Registration(H, inliers, ratio)

    @staticmethod
    def frame_to_map(reg: Registration, u: float, v: float) -> Tuple[float, float]:
        p = reg.homography @ np.array([u, v, 1.0])
        return float(p[0] / p[2]), float(p[1] / p[2])


def detect_footprints(site_map: SiteMap, detector, tile: int = 1024, overlap: int = 128) -> List[Dict[str, Any]]:
    """Run an asset detector over orthomosaic tiles; return lat/lon rectangle footprints."""
    rgb = Image.fromarray(site_map.image).convert("RGB")
    W, H = rgb.size
    out: List[Dict[str, Any]] = []
    step = max(1, tile - overlap)
    for y in range(0, max(1, H - overlap), step):
        for x in range(0, max(1, W - overlap), step):
            crop = rgb.crop((x, y, min(W, x + tile), min(H, y + tile)))
            for d in detector.detect(crop):
                x1, y1, x2, y2 = d.bbox
                corners = [(x + x1, y + y1), (x + x2, y + y1), (x + x2, y + y2), (x + x1, y + y2)]
                poly = [list(pixel_to_geo(site_map.geotransform, c, r)) for c, r in corners]
                out.append({"asset_type": d.asset_type, "confidence": d.confidence, "footprint": poly,
                            "bbox_px": [x + x1, y + y1, x + x2, y + y2]})
    return _nms_footprints(out)


def _nms_footprints(items: List[Dict[str, Any]], iou_thr: float = 0.5) -> List[Dict[str, Any]]:
    from ml.imaging import iou

    items = sorted(items, key=lambda d: d["confidence"], reverse=True)
    kept: List[Dict[str, Any]] = []
    for it in items:
        if all(k["asset_type"] != it["asset_type"] or iou(k["bbox_px"], it["bbox_px"]) < iou_thr for k in kept):
            kept.append(it)
    return kept
