"""Asset re-identification cascade, separate from condition (D2, D3).

Stages (each swappable and configurable):
  1. Fiducial: ArUco marker id inside/near the bbox → asset via the site marker table.
  2. Location gate: only assets of the same type within matching.location_radius_m.
  3. Visual shortlist: global-embedding cosine top-k from the prototype index.
  4. Geometric verification: SIFT/ORB + RANSAC homography against the
     shortlisted assets' reference views.
  5. Otherwise a new asset (FR-11).

Match confidence (identity only, never used in severity — D1):
  aruco          → 1.0
  geometric      → 0.5 + 0.5·min(1, inliers / (2·min_inliers))
  visual_only    → global cosine similarity (only when the query or all
                   reference views lack keypoints, e.g. texture-less surfaces)
  new            → 0.0 (1.0 when an unknown ArUco marker certifies a new asset)
"""

from __future__ import annotations

import logging
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
from PIL import Image

from memory.vector_memory import Candidate, PrototypeIndex
from ml.imaging import expand_bbox, pil_to_bgr, to_gray

logger = logging.getLogger(__name__)

EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


@dataclass
class GeoLocation:
    latitude: float
    longitude: float
    altitude: float = 0.0

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> Optional["GeoLocation"]:
        if not d or d.get("latitude") is None or d.get("longitude") is None:
            return None
        return cls(float(d["latitude"]), float(d["longitude"]), float(d.get("altitude") or 0.0))

    def to_dict(self) -> Dict[str, float]:
        return {"latitude": self.latitude, "longitude": self.longitude, "altitude": self.altitude}


@dataclass
class AssetRef:
    """Minimal asset view the cascade needs (supplied by the store)."""

    asset_id: str
    asset_type: str
    location: Optional[GeoLocation] = None
    marker_id: Optional[int] = None


@dataclass
class IdentityResult:
    asset_id: Optional[str]
    stage: str                      # aruco | geometric | visual_only | new
    confidence: float
    marker_id: Optional[int] = None
    inliers: int = 0
    inlier_ratio: float = 0.0
    similarity: float = 0.0
    reference_proto_id: Optional[int] = None
    homography: Optional[np.ndarray] = None   # query-crop px → reference-image px
    candidates: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_new(self) -> bool:
        return self.asset_id is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "stage": self.stage,
            "confidence": self.confidence,
            "marker_id": self.marker_id,
            "inliers": self.inliers,
            "inlier_ratio": self.inlier_ratio,
            "similarity": self.similarity,
            "candidates": self.candidates,
        }


class ArucoReader:
    def __init__(self, dictionary: str, margin: float):
        if not hasattr(cv2.aruco, dictionary):
            raise ValueError(f"Unknown ArUco dictionary: {dictionary}")
        self.margin = margin
        self._detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary)),
            cv2.aruco.DetectorParameters(),
        )

    def read(self, frame: Image.Image, bbox: Optional[Sequence[float]]) -> List[int]:
        region = frame
        if bbox is not None:
            box = expand_bbox(bbox, self.margin, frame.width, frame.height)
            if box:
                region = frame.crop(box)
        gray = cv2.cvtColor(pil_to_bgr(region), cv2.COLOR_BGR2GRAY)
        _, ids, _ = self._detector.detectMarkers(gray)
        return [] if ids is None else sorted(int(i) for i in ids.flatten())


@dataclass
class Features:
    keypoints: np.ndarray        # (K, 2) float32 at verification scale
    descriptors: Optional[np.ndarray]
    scale: float                 # verification px = original px · scale
    size: Tuple[int, int]        # original (width, height)


class GeometricVerifier:
    def __init__(self, mcfg: Dict[str, Any]):
        self.kind = mcfg["feature"]
        self.max_side = int(mcfg["verify_max_side"])
        self.min_keypoints = int(mcfg["min_keypoints"])
        self.ratio = float(mcfg["lowe_ratio"])
        self.reproj = float(mcfg["ransac_reproj_threshold"])
        self.min_inliers = int(mcfg["min_inliers"])
        self.min_ratio = float(mcfg["min_inlier_ratio"])
        n = int(mcfg["max_keypoints"])
        if self.kind == "sift" and hasattr(cv2, "SIFT_create"):
            self._extractor = cv2.SIFT_create(nfeatures=n)
            self._norm = cv2.NORM_L2
        else:
            self.kind = "orb"
            self._extractor = cv2.ORB_create(nfeatures=n)
            self._norm = cv2.NORM_HAMMING
        self._matcher = cv2.BFMatcher(self._norm)
        self._lock = threading.Lock()

    def extract(self, pil: Image.Image) -> Features:
        gray = to_gray(pil, self.max_side)
        scale = gray.shape[1] / float(pil.width)
        with self._lock:
            kps, desc = self._extractor.detectAndCompute(gray, None)
        pts = np.array([k.pt for k in kps], dtype=np.float32).reshape(-1, 2)
        return Features(pts, desc, scale, (pil.width, pil.height))

    def has_enough(self, f: Features) -> bool:
        return f.descriptors is not None and len(f.keypoints) >= self.min_keypoints

    def verify(self, q: Features, r: Features) -> Tuple[int, float, Optional[np.ndarray]]:
        """Return (inliers, inlier_ratio, H) with H mapping query px → reference px (original scale)."""
        if not (self.has_enough(q) and self.has_enough(r)):
            return 0, 0.0, None
        pairs = self._matcher.knnMatch(q.descriptors, r.descriptors, k=2)
        good = [m for m, *rest in pairs if rest and m.distance < self.ratio * rest[0].distance]
        if len(good) < 4:
            return 0, 0.0, None
        src = q.keypoints[[m.queryIdx for m in good]]
        dst = r.keypoints[[m.trainIdx for m in good]]
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.reproj)
        if H is None or mask is None:
            return 0, 0.0, None
        inliers = int(mask.sum())
        s_q = np.diag([q.scale, q.scale, 1.0])
        s_r_inv = np.diag([1.0 / r.scale, 1.0 / r.scale, 1.0])
        return inliers, inliers / float(len(good)), s_r_inv @ H @ s_q

    def accepts(self, inliers: int, ratio: float) -> bool:
        return inliers >= self.min_inliers and ratio >= self.min_ratio

    def confidence(self, inliers: int) -> float:
        return 0.5 + 0.5 * min(1.0, inliers / (2.0 * self.min_inliers))


ReferenceLoader = Callable[[int], Optional[Image.Image]]


class AssetIdentifier:
    def __init__(
        self,
        config: Dict[str, Any],
        index: PrototypeIndex,
        reference_loader: ReferenceLoader,
    ):
        mcfg = config["matching"]
        acfg = config["aruco"]
        self.top_k = int(mcfg["top_k"])
        self.shortlist_min = float(mcfg["shortlist_min_similarity"])
        self.visual_only = float(mcfg["visual_only_similarity"])
        self.radius_m = float(mcfg["location_radius_m"])
        self.early_accept_inliers = int(mcfg["early_accept_inliers"])
        self.index = index
        self.reference_loader = reference_loader
        self.verifier = GeometricVerifier(mcfg)
        self.aruco = ArucoReader(acfg["dictionary"], float(acfg["bbox_margin"])) if acfg["enabled"] else None
        self._feature_cache: "OrderedDict[int, Features]" = OrderedDict()
        self._cache_size = int(mcfg["feature_cache_size"])
        self._cache_lock = threading.Lock()

    def reference_features(self, proto_id: int) -> Optional[Features]:
        with self._cache_lock:
            if proto_id in self._feature_cache:
                self._feature_cache.move_to_end(proto_id)
                return self._feature_cache[proto_id]
        pil = self.reference_loader(proto_id)
        if pil is None:
            return None
        feats = self.verifier.extract(pil)
        with self._cache_lock:
            self._feature_cache[proto_id] = feats
            while len(self._feature_cache) > self._cache_size:
                self._feature_cache.popitem(last=False)
        return feats

    def forget(self, proto_id: int) -> None:
        with self._cache_lock:
            self._feature_cache.pop(proto_id, None)

    def location_gate(
        self, assets: Sequence[AssetRef], asset_type: str, location: Optional[GeoLocation]
    ) -> Optional[Set[str]]:
        """Asset ids allowed by location, or None when no gating applies."""
        if location is None:
            return None
        allowed = set()
        for a in assets:
            if a.asset_type != asset_type:
                continue
            if a.location is None or haversine_m(
                location.latitude, location.longitude, a.location.latitude, a.location.longitude
            ) <= self.radius_m:
                allowed.add(a.asset_id)
        return allowed

    def identify(
        self,
        frame: Image.Image,
        bbox: Optional[Sequence[float]],
        crop: Image.Image,
        embedding: np.ndarray,
        asset_type: str,
        site_id: str,
        location: Optional[GeoLocation],
        site_assets: Sequence[AssetRef],
        marker_table: Dict[int, str],
    ) -> IdentityResult:
        # 1. Fiducial
        marker_id: Optional[int] = None
        if self.aruco is not None:
            markers = self.aruco.read(frame, bbox)
            if markers:
                marker_id = markers[0]
                if marker_id in marker_table:
                    return IdentityResult(marker_table[marker_id], "aruco", 1.0, marker_id=marker_id)
                # A marker that is not in the site table denotes a new tagged asset.
                return IdentityResult(None, "new", 1.0, marker_id=marker_id)

        # 2. Location gate
        allowed = self.location_gate(site_assets, asset_type, location)
        if allowed is not None and not allowed:
            return IdentityResult(None, "new", 0.0)

        # 3. Visual shortlist
        shortlist: List[Candidate] = [
            c
            for c in self.index.search(embedding, site_id, asset_type, self.top_k, allowed)
            if c.similarity >= self.shortlist_min
        ]
        cand_info = [{"asset_id": c.asset_id, "similarity": round(c.similarity, 4)} for c in shortlist]
        if not shortlist:
            return IdentityResult(None, "new", 0.0, candidates=cand_info)

        # 4. Geometric verification against every prototype of each shortlisted asset
        q_feats = self.verifier.extract(crop)
        best: Optional[IdentityResult] = None
        reference_has_geometry = False
        for cand in shortlist:
            if best is not None and best.inliers >= self.early_accept_inliers:
                break
            for proto in self.index.prototypes(cand.asset_id):
                r_feats = self.reference_features(proto.proto_id)
                if r_feats is None:
                    continue
                reference_has_geometry = reference_has_geometry or self.verifier.has_enough(r_feats)
                inliers, ratio, H = self.verifier.verify(q_feats, r_feats)
                if self.verifier.accepts(inliers, ratio) and (best is None or inliers > best.inliers):
                    best = IdentityResult(
                        cand.asset_id,
                        "geometric",
                        self.verifier.confidence(inliers),
                        inliers=inliers,
                        inlier_ratio=ratio,
                        similarity=cand.similarity,
                        reference_proto_id=proto.proto_id,
                        homography=H,
                    )
        if best is not None:
            best.candidates = cand_info
            return best

        top = shortlist[0]
        geometry_possible = self.verifier.has_enough(q_feats) and reference_has_geometry
        if not geometry_possible and top.similarity >= self.visual_only:
            return IdentityResult(
                top.asset_id,
                "visual_only",
                top.similarity,
                similarity=top.similarity,
                reference_proto_id=top.proto_id,
                candidates=cand_info,
            )

        # 5. New asset
        return IdentityResult(None, "new", 0.0, similarity=top.similarity, candidates=cand_info)
