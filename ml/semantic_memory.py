"""Persistent semantic memory engine.

Per detected asset:
  crop → embedding → condition classifier → identity cascade → change
  detector → gated severity → one atomic Local DB commit → events.

Identity (re-ID) and condition are separate models (D2). Severity uses the
observation confidence min(detector, classifier) (D1) and the M-gate (D4).

Upgrade path wired in here:
  3.4 quality gate on every frame and detection         ml/quality_gate.py
  1.1 ground projection of each detection from the pose  ml/geo_projection.py
  1.2 frame → site-map registration, footprint lookup    ml/site_map.py
  1.3 identity embedding from the re-ID head             ml/reid_head.py
  2.1 pixel change mask, area in cm²                     ml/pixel_change.py, ml/camera_model.py
  3.3 defect masks → 2.2 defect instances → 2.3 forecast ml/segmentation.py, memory/store_ext.py
  2.4 in-flight guidance, 5.3 explanation per event      ml/guidance.py, ml/explain.py
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from memory.store import LocalStore
from memory.vector_memory import PrototypeIndex
from ml.asset_identity import AssetIdentifier, GeoLocation, IdentityResult
from ml.camera_model import CameraIntrinsics, MetricScale, load_intrinsics, scale_for_bbox
from ml.change_detector import (
    ChangeAssessment,
    ChangeDetector,
    ConditionVisualChangeDetector,
    ReferenceCondition,
    render_heatmap,
)
from ml.condition_classifier import NORMAL_LABEL, ConditionClassifier, ConditionPrediction
from ml.config import cfg, get_nested, resolve_path
from ml.defect_tracking import MeasuredRegion, merge_overlapping
from ml.detector import Detection, Detector
from ml.embedding_model import AssetEmbeddingModel
from ml.explain import explain
from ml.forecast import Forecast, family, growth_summary
from ml.geo_projection import CameraPose, GeoProjector, pixel_ray
from ml.guidance import compute_guidance, short_label
from ml.imaging import crop as crop_image
from ml.pixel_change import ChangeMask, PixelChangeDetector, homography_area_scale, render_mask, warp_to_reference
from ml.quality_gate import QualityGate, ray_depression_deg
from ml.reid_head import IdentityEncoder
from ml.segmentation import build_segmenter, regions_from_mask
from ml.site_map import MapRegistrar, SiteMap
from ml.schemas import new_id, utc_now_iso
from ml.severity import (
    asset_criticality,
    gated_severity,
    is_critical,
    is_priority,
    observation_confidence,
    severity_level,
)
from ml.tracker import SessionTracker

logger = logging.getLogger(__name__)


@dataclass
class AssetEvent:
    type: str                      # new_asset | confirmed | change | duplicate | tracked | error
    session_id: Optional[str]
    timestamp: str
    asset_type: str
    bbox: List[int]
    asset_id: Optional[str] = None
    track_id: Optional[int] = None
    identity: Dict[str, Any] = field(default_factory=dict)
    condition: Dict[str, Any] = field(default_factory=dict)
    change: Optional[Dict[str, Any]] = None
    detector_confidence: float = 0.0
    observation_confidence: float = 0.0
    severity_score: float = 0.0
    severity_level: str = "low"
    priority: bool = False
    critical: bool = False
    persisted: bool = False
    observation_id: Optional[str] = None
    defect: Optional[Dict[str, Any]] = None
    revisit: Optional[Dict[str, float]] = None
    history: Optional[Dict[str, Any]] = None     # FR-12: what memory held for a matched asset
    image_reference: Optional[str] = None
    heatmap_reference: Optional[str] = None
    mask_reference: Optional[str] = None
    geo: Optional[Dict[str, Any]] = None           # 1.1/1.2 ground fix, site-map hit
    quality: Optional[Dict[str, Any]] = None       # 3.4 gate result
    change_mask: Optional[Dict[str, Any]] = None   # 2.1 pixel change
    defect_instances: List[Dict[str, Any]] = field(default_factory=list)   # 2.2/2.3
    growth: Optional[str] = None                   # 2.3 "crack grew 40% …; critical in about 34 days"
    guidance: Optional[Dict[str, Any]] = None      # 2.4
    explanation: Optional[Dict[str, Any]] = None   # 5.3
    work_orders: List[Dict[str, Any]] = field(default_factory=list)        # 6.2
    latency_ms: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


Listener = Callable[[AssetEvent], None]


def _reference_from_prediction(p: ConditionPrediction) -> Dict[str, Any]:
    return {"label": p.label, "scores": p.scores, "is_defect": p.is_defect, "defect_type": p.defect_type}


def _implied_normal(labels: Sequence[str]) -> Dict[str, Any]:
    """Reference for a never-seen asset: assume it was commissioned in normal condition."""
    return {
        "label": NORMAL_LABEL,
        "scores": {l: (1.0 if l == NORMAL_LABEL else 0.0) for l in labels},
        "is_defect": False,
        "defect_type": None,
    }


class SemanticMemoryEngine:
    def __init__(
        self,
        config: Dict[str, Any],
        store: LocalStore,
        embedder: AssetEmbeddingModel,
        classifier: ConditionClassifier,
        detector: Detector,
        change_detector: Optional[ChangeDetector] = None,
    ):
        self.config = config
        self.store = store
        self.embedder = embedder
        self.classifier = classifier
        self.detector = detector
        self.change_detector = change_detector or ConditionVisualChangeDetector(config, embedder)
        self.identity_encoder = IdentityEncoder.from_config(config, embedder.embedding_dim)
        self.geo = GeoProjector(config)
        self.quality = QualityGate(config)
        self.pixel_change = PixelChangeDetector(config)
        self.segmenter = build_segmenter(config)
        self.map_registrar = MapRegistrar(config)
        self.site_map_enabled = bool(get_nested(config, "site_map", "enabled", default=True))
        self.sync_images = bool(get_nested(config, "cloud", "sync_images", default=True))
        self._site_maps: Dict[str, Optional[SiteMap]] = {}
        self._intrinsics: Dict[tuple, CameraIntrinsics] = {}
        self.last_frame_quality: Optional[Dict[str, Any]] = None
        m = cfg(config, "matching")
        self.index = PrototypeIndex(
            self.identity_dim,
            cfg(config, "memory", "vector_backend"),
            int(m["max_prototypes_per_asset"]),
            float(m["prototype_novelty"]),
        )
        self.identifier = AssetIdentifier(config, self.index, self._load_prototype_image)
        self.resolution_missions = int(cfg(config, "defects", "resolution_suggestion_missions"))
        self.jpeg_quality = int(cfg(config, "storage", "image_jpeg_quality"))
        self.heatmap_dir = resolve_path(cfg(config, "change_detection", "heatmap_dir"))
        self.heatmap_opacity = float(cfg(config, "change_detection", "heatmap_opacity"))
        self._listeners: List[Listener] = []
        self._session_cache: Dict[str, Dict[str, Any]] = {}
        self._site_cache: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._sync_backbone()
        self._load_index()

    # ── setup ─────────────────────────────────────────────────────────────
    @property
    def identity_dim(self) -> int:
        return self.identity_encoder.dim(self.embedder.embedding_dim)

    def identity_embedding(self, embedding: np.ndarray) -> np.ndarray:
        return self.identity_encoder.encode(embedding)

    def intrinsics(self, width: int, height: int) -> CameraIntrinsics:
        key = (width, height)
        if key not in self._intrinsics:
            self._intrinsics[key] = load_intrinsics(self.config, width, height)
        return self._intrinsics[key]

    def site_map(self, site_id: str) -> Optional[SiteMap]:
        """Active orthomosaic + footprints of a site (cached; `invalidate_site` refreshes)."""
        if not self.site_map_enabled:
            return None
        if site_id not in self._site_maps:
            row = self.store.active_site_map(site_id)
            smap = None
            if row is not None:
                try:
                    smap = SiteMap.load(row["map_id"], site_id, resolve_path(row["image_path"]), row["geotransform"],
                                        int(get_nested(self.config, "site_map", "max_side", default=4096)))
                    smap.set_footprints(self.store.footprint_assets(site_id))
                except Exception:  # noqa: BLE001 - a broken map must not stop inspection
                    logger.exception("Site map %s could not be loaded", row["map_id"])
            self._site_maps[site_id] = smap
        return self._site_maps[site_id]

    def invalidate_site(self, site_id: str) -> None:
        self._site_maps.pop(site_id, None)
        self._site_cache.pop(site_id, None)

    def _sync_backbone(self) -> None:
        """Re-embed stored prototypes if the backbone (or re-ID head) changed since they were written (D12)."""
        current = {"name": self.embedder.active_model_name, "dim": self.identity_dim, "identity": self.identity_encoder.name}
        stored = self.store.get_meta("backbone")
        if stored == current:
            return
        protos = self.store.all_prototypes()
        if protos and stored is not None:
            logger.warning("Backbone changed %s → %s; re-embedding %d prototypes", stored, current, len(protos))
            from memory import db

            with self.store.txn() as s:
                for p in protos:
                    path = self.store.image_path(p["image_reference"])
                    row = s.get(db.AssetPrototype, p["proto_id"])
                    if not path.exists():
                        s.delete(row)
                        continue
                    im = self._open_image_file(path)
                    vec = self.identity_embedding(self.embedder.encode_image(im))
                    row.embedding = vec.astype(np.float32).tobytes()
                    row.dim = int(vec.shape[0])
        self.store.set_meta("backbone", current)

    def _load_index(self) -> None:
        for p in self.store.all_prototypes():
            if p["embedding"].shape[0] != self.identity_dim:
                logger.warning("Skipping prototype %s with dim %d", p["proto_id"], p["embedding"].shape[0])
                continue
            self.index.add(p["proto_id"], p["asset_id"], p["site_id"], p["asset_type"], p["embedding"])
        logger.info("Prototype index: %d prototypes for %d assets (%s)", len(self.index), self.index.asset_count(), self.index.backend_name)

    def reload_index(self) -> None:
        with self._lock:
            self.index = PrototypeIndex(
                self.identity_dim,
                cfg(self.config, "memory", "vector_backend"),
                self.index.max_prototypes,
                self.index.novelty,
            )
            self.identifier.index = self.index
            self._site_cache.clear()
            self._load_index()

    def prefetch_references(self, site_id: str, stop: Optional[threading.Event] = None) -> int:
        """Extract reference keypoints for the site's prototypes, highest revisit priority first.

        Bounded by `matching.feature_cache_size`, so matching during the flight does not pay
        the extraction cost on the first sighting of a known asset (NFR-2).
        """
        known = self.index.asset_ids(site_id)
        order = [a["asset_id"] for a in self.store.list_assets(site_id) if a["asset_id"] in known]
        budget = int(cfg(self.config, "matching", "feature_cache_size"))
        done = 0
        for aid in order:
            for proto in self.index.prototypes(aid):
                if done >= budget or (stop is not None and stop.is_set()):
                    return done
                if self.identifier.reference_features(proto.proto_id) is not None:
                    done += 1
        return done

    def start_prefetch(self, site_id: str) -> threading.Thread:
        t = threading.Thread(target=self.prefetch_references, args=(site_id,), name=f"prefetch-{site_id}", daemon=True)
        t.start()
        return t

    def _open_image_file(self, path) -> Image.Image:
        import io

        from auth.image_crypto import maybe_decrypt

        with Image.open(io.BytesIO(maybe_decrypt(self.config, path.read_bytes()))) as im:
            return im.convert("RGB")

    def _load_prototype_image(self, proto_id: int) -> Optional[Image.Image]:
        path = self.store.prototype_image_path(proto_id)
        if path is None or not path.exists():
            return None
        return self._open_image_file(path)

    def add_listener(self, fn: Listener) -> None:
        self._listeners.append(fn)

    def remove_listener(self, fn: Listener) -> None:
        if fn in self._listeners:
            self._listeners.remove(fn)

    def _emit(self, event: AssetEvent) -> None:
        for fn in list(self._listeners):
            try:
                fn(event)
            except Exception:  # noqa: BLE001 - a failing listener must not break inspection
                logger.exception("Event listener failed")

    def _session(self, session_id: str) -> Dict[str, Any]:
        sess = self._session_cache.get(session_id)
        if sess is None:
            sess = self.store.get_session(session_id)
            if sess is None:
                raise KeyError(f"Unknown session: {session_id}")
            self._session_cache[session_id] = sess
        if sess["status"] != "active":
            fresh = self.store.get_session(session_id)
            self._session_cache[session_id] = fresh
            sess = fresh
        return sess

    def _site_context(self, site_id: str):
        ctx = self._site_cache.get(site_id)
        if ctx is None:
            ctx = (self.store.asset_refs(site_id), self.store.marker_table(site_id))
            self._site_cache[site_id] = ctx
        return ctx

    # ── public API ────────────────────────────────────────────────────────
    def process_frame(
        self,
        frame: Image.Image,
        session_id: str,
        tracker: Optional[SessionTracker] = None,
        location: Optional[Dict[str, float]] = None,
        timestamp: Optional[str] = None,
        force_track_id: Optional[int] = None,
        pose: Optional[Dict[str, Any]] = None,
    ) -> List[AssetEvent]:
        """Quality gate → detect → track → process each new/changed track once (D8).

        `pose` (optional): drone GPS + height above ground + camera heading/pitch/roll, see
        ml/geo_projection.CameraPose. With a pose every detection is geo-anchored (1.1).
        """
        t_q = time.perf_counter()
        fq = self.quality.check_frame(frame)
        self.last_frame_quality = fq.to_dict()
        if not fq.ok:
            return []
        t_q = (time.perf_counter() - t_q) * 1000
        t0 = time.perf_counter()
        detections = self.detector.detect(frame)
        t_det = (time.perf_counter() - t0) * 1000
        if not detections:
            return []
        crops = [crop_image(frame, d.bbox) for d in detections]
        t1 = time.perf_counter()
        embeddings = self.embedder.encode_batch(crops)
        t_emb = (time.perf_counter() - t1) * 1000
        t2 = time.perf_counter()
        spatial = self._frame_spatial(frame, detections, session_id, pose, location)
        t_spatial = (time.perf_counter() - t2) * 1000
        tracks = (
            tracker.update([d.bbox for d in detections], [d.asset_type for d in detections], embeddings, crops)
            if tracker else [None] * len(detections)
        )

        events: List[AssetEvent] = []
        for det, crop, emb, track, sp in zip(detections, crops, embeddings, tracks, spatial):
            forced = track is not None and force_track_id == track.track_id
            if track is not None and track.observed and not forced:
                pred = self.classifier.predict_embeddings(emb[None, :], det.asset_type)[0]
                if pred.label == track.last_label:
                    events.append(self._tracked_event(det, track, session_id, timestamp))
                    continue
            ev = self._process(frame, det, crop, emb, session_id, location, timestamp, True, track, forced, sp)
            ev.latency_ms.update({"quality": t_q, "detect": t_det, "embed_batch": t_emb, "spatial": t_spatial})
            events.append(ev)
        return events

    # ── spatial context (1.1, 1.2, 3.6) ───────────────────────────────────
    def _frame_spatial(self, frame: Image.Image, detections: Sequence[Detection], session_id: Optional[str],
                       pose: Optional[Dict[str, Any]], location: Optional[Dict[str, float]]) -> List[Dict[str, Any]]:
        pose_obj = CameraPose.from_dict(pose) if pose else None
        intr = self.intrinsics(frame.width, frame.height)
        site_id = None
        if session_id:
            try:
                site_id = self._session(session_id)["site_id"]
            except KeyError:
                site_id = None
        smap = self.site_map(site_id) if site_id else None
        reg = None
        if smap is not None and smap.footprints:
            near = (pose_obj.latitude, pose_obj.longitude) if pose_obj else (
                (location["latitude"], location["longitude"]) if location and location.get("latitude") is not None else None)
            reg = self.map_registrar.register(frame, smap, near)
        return [self._detection_spatial(d, intr, pose_obj, smap, reg, location) for d in detections]

    def _detection_spatial(self, det: Detection, intr: CameraIntrinsics, pose: Optional[CameraPose], smap, reg,
                           location: Optional[Dict[str, float]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {"intrinsics": intr, "pose": pose}
        geo: Dict[str, Any] = {}
        if pose is not None:
            fix = self.geo.project(det.bbox, det.asset_type, pose, intr)
            if fix is not None:
                out["location"] = {"latitude": fix.latitude, "longitude": fix.longitude, "altitude": 0.0}
                out["uncertainty_m"] = fix.uncertainty_m
                geo["ground_fix"] = fix.to_dict()
            cx, cy = (det.bbox[0] + det.bbox[2]) / 2.0, (det.bbox[1] + det.bbox[3]) / 2.0
            ray = pixel_ray(cx, cy, intr, pose)
            out["ray_pitch_deg"] = ray_depression_deg(float(ray[2]))
            slant = None if fix is None else fix.slant_range_m
            out["scale"] = scale_for_bbox(self.config, intr, det.bbox, det.asset_type, range_m=pose.range_m or slant)
        else:
            if location is not None:
                out["location"] = location
            out["scale"] = scale_for_bbox(self.config, intr, det.bbox, det.asset_type)
        if smap is not None and reg is not None:
            from ml.geo_projection import anchor_pixel

            u, v = anchor_pixel(det.bbox, self.geo.anchors.get(det.asset_type, "center"))
            col, row = self.map_registrar.frame_to_map(reg, u, v)
            hit = smap.lookup_px(col, row, det.asset_type)
            geo["site_map"] = {"map_id": smap.map_id, "registration": reg.to_dict(), "map_px": [col, row],
                               "asset_id": hit.asset_id if hit else None}
            if hit is not None:
                out["map_asset_id"] = hit.asset_id
                out["location"] = {"latitude": hit.latitude, "longitude": hit.longitude, "altitude": 0.0}
                out.setdefault("uncertainty_m", 1.0)
        if out.get("scale") is not None:
            geo["scale"] = out["scale"].to_dict()
        out["geo"] = geo or None
        return out

    def observe(
        self,
        image: Image.Image,
        asset_type: str,
        session_id: Optional[str],
        persist: bool = True,
        detector_confidence: Optional[float] = None,
        location: Optional[Dict[str, float]] = None,
        timestamp: Optional[str] = None,
        force: bool = False,
        pose: Optional[Dict[str, Any]] = None,
    ) -> AssetEvent:
        """Process a single close-up image of one asset (whole image = the asset)."""
        conf = float(cfg(self.config, "detection", "full_frame_confidence")) if detector_confidence is None else detector_confidence
        det = Detection((0, 0, image.width, image.height), asset_type, conf, "image")
        emb = self.embedder.encode_image(image)
        intr = self.intrinsics(image.width, image.height)
        pose_obj = CameraPose.from_dict(pose) if pose else None
        spatial = self._detection_spatial(det, intr, pose_obj, None, None, location)
        return self._process(image, det, image, emb, session_id, location, timestamp, persist, None, force, spatial)

    # ── core ──────────────────────────────────────────────────────────────
    def _tracked_event(self, det: Detection, track, session_id, timestamp) -> AssetEvent:
        info = track.info
        return AssetEvent(
            type="tracked",
            session_id=session_id,
            timestamp=timestamp or utc_now_iso(),
            asset_type=det.asset_type,
            bbox=list(det.bbox),
            asset_id=track.asset_id,
            track_id=track.track_id,
            condition=info.get("condition", {}),
            detector_confidence=det.confidence,
            severity_score=info.get("severity_score", 0.0),
            severity_level=info.get("severity_level", "low"),
            priority=info.get("priority", False),
            critical=info.get("critical", False),
        )

    def _process(
        self,
        frame: Image.Image,
        det: Detection,
        crop: Image.Image,
        embedding: np.ndarray,
        session_id: Optional[str],
        location: Optional[Dict[str, float]],
        timestamp: Optional[str],
        persist: bool,
        track,
        force: bool,
        spatial: Optional[Dict[str, Any]] = None,
    ) -> AssetEvent:
        spatial = spatial or {}
        ts = timestamp or utc_now_iso()
        lat: Dict[str, float] = {}
        t_start = time.perf_counter()
        ev = AssetEvent(
            type="error",
            session_id=session_id,
            timestamp=ts,
            asset_type=det.asset_type,
            bbox=list(det.bbox),
            track_id=track.track_id if track else None,
            detector_confidence=det.confidence,
        )
        try:
            if not self.classifier.supports(det.asset_type):
                raise ValueError(f"No condition model for asset_type '{det.asset_type}'")
            session = self._session(session_id) if session_id else None
            if persist and (session is None or session["status"] != "active"):
                raise ValueError("Persisting an observation requires an active session (D7)")
            site_id = session["site_id"] if session else cfg(self.config, "sites", "default_site_id")
            ev.geo = spatial.get("geo")
            scale: Optional[MetricScale] = spatial.get("scale")
            dq = self.quality.check_detection(crop, det.asset_type, scale.distance_m if scale else None, spatial.get("ray_pitch_deg"))
            ev.quality = dq.to_dict()
            if not dq.ok:
                ev.type = "rejected"
                lat["total"] = (time.perf_counter() - t_start) * 1000
                ev.latency_ms = lat
                return ev
            if spatial.get("location") is not None:
                location = spatial["location"]

            t = time.perf_counter()
            pred = self.classifier.predict_embeddings(embedding[None, :], det.asset_type)[0]
            lat["classify"] = (time.perf_counter() - t) * 1000
            cf = observation_confidence(det.confidence, pred.confidence)
            ev.condition = pred.to_dict()
            ev.observation_confidence = cf

            with self._lock:
                t = time.perf_counter()
                refs, markers = self._site_context(site_id)
                geo = GeoLocation.from_dict(location)
                id_embedding = self.identity_embedding(embedding)
                identity: IdentityResult = self.identifier.identify(
                    frame, det.bbox, crop, id_embedding, det.asset_type, site_id, geo, refs, markers,
                    location_uncertainty_m=spatial.get("uncertainty_m"), map_asset_id=spatial.get("map_asset_id"),
                )
                lat["identity"] = (time.perf_counter() - t) * 1000
                ev.identity = identity.to_dict()

                if not identity.is_new and persist and session_id and not force:
                    last = self.store.last_session_observation(session_id, identity.asset_id)
                    if last is not None and last["condition_description"] == pred.label:
                        ev.type = "duplicate"
                        ev.asset_id = identity.asset_id
                        ev.severity_score = last["severity_score"]
                        ev.severity_level = last["severity_level"]
                        self._update_track(track, ev, pred)
                        lat["total"] = (time.perf_counter() - t_start) * 1000
                        ev.latency_ms = lat
                        return ev

                labels = self.classifier.label_maps[det.asset_type]
                t = time.perf_counter()
                asset_row = None
                if identity.is_new:
                    # A healthy-looking new asset is its own baseline (M = 0); a defective one is
                    # compared with the implied commissioning state so the defect is scored.
                    ref_dict = _implied_normal(labels) if pred.is_defect else _reference_from_prediction(pred)
                    reference_image = None
                    homography = None
                else:
                    asset_row = self.store.get_asset(identity.asset_id)
                    ev.history = self._history_summary(asset_row)
                    ref_dict = asset_row["reference_condition"]
                    ref_proto = identity.reference_proto_id or asset_row["reference_proto_id"]
                    reference_image = self._load_prototype_image(ref_proto) if ref_proto else None
                    homography = identity.homography if identity.reference_proto_id == ref_proto else None
                reference = ReferenceCondition(
                    ref_dict["label"], ref_dict["scores"], ref_dict["is_defect"], ref_dict["defect_type"]
                )
                change: ChangeAssessment = self.change_detector.assess(pred, reference, crop, reference_image, homography)
                lat["change"] = (time.perf_counter() - t) * 1000
                t = time.perf_counter()
                cmask: Optional[ChangeMask] = None
                cm2_per_px = scale.cm2_per_px if scale else None
                if reference_image is not None:
                    cmask = self.pixel_change.detect(crop, reference_image, homography, cm2_per_px)
                regions = self._defect_regions(pred, crop, reference_image, homography, cmask, cm2_per_px)
                lat["pixel_change"] = (time.perf_counter() - t) * 1000

                criticality = asset_criticality(det.asset_type, self.config)
                severity, significant = gated_severity(change.magnitude, criticality, cf, self.config)
                ev.change = change.to_dict()
                if cmask is not None:
                    ev.change_mask = cmask.to_dict()
                    ev.change["mask"] = {k: v for k, v in ev.change_mask.items() if k != "regions"}
                ev.severity_score = severity
                ev.severity_level = severity_level(severity, self.config)
                ev.priority = is_priority(severity, self.config)
                ev.critical = is_critical(severity, self.config)
                if identity.is_new:
                    ev.type = "new_asset"
                    status = "new"
                else:
                    ev.type = "change" if significant else "confirmed"
                    status = "changed" if significant else "confirmed"
                    ev.asset_id = identity.asset_id

                if persist:
                    t = time.perf_counter()
                    heat_base = reference_image if change.warped else crop
                    self._persist(
                        ev, det, crop, id_embedding, pred, identity, change, status, significant, session, site_id,
                        geo, criticality, cf, heat_base, labels, cmask, reference_image, regions,
                    )
                    lat["store"] = (time.perf_counter() - t) * 1000
                ev.growth = self._growth_text(ev.defect_instances)
                ev.explanation = explain(
                    self.config, change=ev.change, criticality=criticality, confidence=cf,
                    detector_confidence=det.confidence, classifier_confidence=pred.confidence,
                    severity=ev.severity_score, level=ev.severity_level, significant=significant,
                    before_image=(ev.history or {}).get("previous_image"), after_image=ev.image_reference,
                    heatmap=ev.heatmap_reference, mask=ev.mask_reference, condition=ev.condition,
                    reference_label=None if identity.is_new else reference.label,
                    instances=ev.defect_instances, growth_text=ev.growth,
                )
                ev.guidance = self._guidance(ev, det, spatial, asset_row)
                self._update_track(track, ev, pred)
        except Exception as exc:  # noqa: BLE001 - errors are reported per asset, never crash a mission
            logger.exception("Processing failed")
            ev.type = "error"
            ev.error = str(exc)
        lat["total"] = (time.perf_counter() - t_start) * 1000
        ev.latency_ms = lat
        self._emit(ev)
        return ev

    def _update_track(self, track, ev: AssetEvent, pred: ConditionPrediction) -> None:
        if track is None:
            return
        track.observed = True
        track.asset_id = ev.asset_id
        track.identity_stage = ev.identity.get("stage")
        track.last_label = pred.label
        track.info = {
            "condition": ev.condition,
            "severity_score": ev.severity_score,
            "severity_level": ev.severity_level,
            "priority": ev.priority,
            "critical": ev.critical,
        }

    # ── defect regions (2.2 / 3.3) ────────────────────────────────────────
    def _defect_regions(
        self, pred: ConditionPrediction, crop: Image.Image, reference_image: Optional[Image.Image],
        homography: Optional[np.ndarray], cmask: Optional[ChangeMask], cm2_per_crop_px: Optional[float],
    ) -> Optional[List[MeasuredRegion]]:
        """Defect instances of this view in the asset's reference-view coordinates.

        [] = the asset was seen without defects (open instances count a miss); None = unknown.
        """
        if not pred.is_defect or not pred.defect_type:
            return []
        try:
            seg = self.segmenter.segment(crop, pred.defect_type)
        except Exception:  # noqa: BLE001 - segmentation is best-effort
            logger.exception("Defect segmentation failed")
            return None
        if reference_image is not None:
            size = reference_image.size
            seg_rgb = Image.fromarray((seg > 0).astype(np.uint8) * 255).convert("RGB")
            warped, _ = warp_to_reference(seg_rgb, size, homography)
            seg_ref = (warped[..., 0] > 127).astype(np.uint8)
            hscale = homography_area_scale(homography, crop.width / 2.0, crop.height / 2.0) if homography is not None \
                else (size[0] * size[1]) / float(crop.width * crop.height)
        else:
            seg_ref, hscale = seg, 1.0
        min_area = float(get_nested(self.config, "defect_tracking", "min_area_fraction", default=0.0005))
        regions = regions_from_mask(seg_ref, min_area, pred.defect_type)
        if cmask is not None:
            fam = family(pred.defect_type)
            for r in cmask.regions:
                if r.change_type and r.change_type != "other" and family(r.change_type) == fam:
                    # change regions are already reported in full reference-view pixels
                    regions.append(type(r)(r.polygon, r.bbox, r.centroid, r.area_px, r.area_norm, r.length_px, pred.defect_type))
        out: List[MeasuredRegion] = []
        cm2_ref = (cm2_per_crop_px / max(hscale, 1e-9)) if cm2_per_crop_px else None
        for r in regions:
            out.append(MeasuredRegion(
                polygon=r.polygon, bbox=r.bbox, centroid=r.centroid, area_norm=r.area_norm, area_px=r.area_px,
                length_px=r.length_px, defect_type=pred.defect_type,
                area_cm2=(r.area_px * cm2_ref) if cm2_ref else None,
                length_cm=(r.length_px * float(np.sqrt(cm2_ref))) if cm2_ref else None,
            ))
        return merge_overlapping(out)

    @staticmethod
    def _growth_text(instances: Sequence[Dict[str, Any]]) -> Optional[str]:
        texts = []
        for i in instances:
            f = i.get("forecast")
            if f and f.get("model"):
                t = growth_summary(Forecast(**{k: f.get(k) for k in Forecast.__dataclass_fields__}))
                if t:
                    texts.append((f.get("days_to_critical") if f.get("days_to_critical") is not None else 1e9, t))
        return min(texts)[1] if texts else None

    def _guidance(self, ev: AssetEvent, det: Detection, spatial: Dict[str, Any], asset_row) -> Optional[Dict[str, Any]]:
        g = get_nested(self.config, "guidance", default={}) or {}
        if not g.get("enabled", True) or "intrinsics" not in spatial:
            return None
        min_r = float(g.get("min_revisit_priority", 0.5))
        due = asset_row is not None and float(asset_row.get("revisit_priority_score") or 0.0) >= min_r
        if not (ev.priority or ev.critical or due):
            return None
        label = short_label(ev.asset_id, (asset_row or {}).get("marker_id"), ((asset_row or {}).get("metadata") or {}).get("name"))
        return compute_guidance(self.config, det.asset_type, det.bbox, spatial["intrinsics"], spatial.get("scale"), label).to_dict()

    def _persist(
        self, ev, det, crop, embedding, pred, identity, change, status, significant, session, site_id,
        geo, criticality, cf, heat_base, labels, cmask=None, reference_image=None, regions=None,
    ) -> None:
        obs_id = new_id("obs")
        is_new = identity.is_new
        asset_id = new_id("asset") if is_new else identity.asset_id
        self.store.asset_image_dir(site_id, asset_id)
        image_ref = f"{site_id}/{asset_id}/{obs_id}.jpg"
        self.store.save_pil(image_ref, crop, "JPEG", self.jpeg_quality)
        heat_ref = None
        if change.heatmap is not None:
            heat_ref = f"{site_id}/{asset_id}/{obs_id}_heat.png"
            render_heatmap(heat_base, change.heatmap, self.store.image_path(heat_ref), self.heatmap_opacity)
            self.store.seal(heat_ref)
        mask_ref = None
        if cmask is not None and reference_image is not None and cmask.changed_area_px > 0:
            mask_ref = f"{site_id}/{asset_id}/{obs_id}_mask.png"
            render_mask(reference_image, cmask.mask, self.store.image_path(mask_ref))
            self.store.seal(mask_ref)

        add_proto, evict = (True, None) if is_new else self.index.plan_update(asset_id, embedding)
        prototype = {"embedding": embedding.tolist(), "image_reference": image_ref} if add_proto else None

        if is_new:
            reference = _reference_from_prediction(pred) if not pred.is_defect else _implied_normal(labels)
        else:
            reference = _reference_from_prediction(pred) if not significant else None

        observation = {
            "observation_id": obs_id,
            "asset_id": asset_id,
            "session_id": session["session_id"],
            "observation_ts": ev.timestamp,
            "condition_description": pred.label,
            "condition_scores": pred.scores,
            "is_defect": pred.is_defect,
            "defect_type": pred.defect_type,
            "image_reference": image_ref,
            "heatmap_reference": heat_ref,
            "severity_score": ev.severity_score,
            "severity_level": ev.severity_level,
            "change_magnitude": change.magnitude,
            "condition_delta": change.condition_delta,
            "visual_delta": change.visual_delta,
            "observation_confidence": cf,
            "detector_confidence": det.confidence,
            "classifier_confidence": pred.confidence,
            "match_stage": identity.stage,
            "match_confidence": identity.confidence,
            "status": status,
            "latitude": geo.latitude if geo else None,
            "longitude": geo.longitude if geo else None,
            "altitude": geo.altitude if geo else None,
            "bbox": list(det.bbox),
            "simulated": bool(session.get("simulated")),
            "meta": {"detector_label": det.label, "identity": identity.to_dict(), "geo": ev.geo, "quality": ev.quality,
                     "mask_reference": mask_ref,
                     "change_mask": {k: v for k, v in (ev.change_mask or {}).items() if k != "regions"} or None},
        }
        if is_new:
            asset = {
                "asset_id": asset_id,
                "site_id": site_id,
                "asset_type": det.asset_type,
                "marker_id": identity.marker_id,
                "first_observed_ts": ev.timestamp,
                "last_checked_ts": ev.timestamp,
                "current_condition": pred.label,
                "current_is_defect": pred.is_defect,
                "reference_condition": reference,
                "criticality": criticality,
                "revisit_priority_score": 0.0,
                "location_samples": 0,
                "simulated": bool(session.get("simulated")),
                "meta": {},
                "updated_ts": ev.timestamp,
            }
        else:
            asset = {"asset_id": asset_id}

        result = self.store.commit_observation(
            asset=asset,
            observation=observation,
            is_new=is_new,
            confirmed=status == "confirmed",
            reference_condition=reference,
            prototype=prototype,
            evict_proto_id=evict,
            resolution_missions=self.resolution_missions,
            defect_regions=regions,
        )
        if result["evicted_proto_id"] is not None:
            self.index.remove(result["evicted_proto_id"])
            self.identifier.forget(result["evicted_proto_id"])
        if result["proto_id"] is not None:
            self.index.add(result["proto_id"], asset_id, site_id, det.asset_type, embedding)
        if is_new:
            self._site_cache.pop(site_id, None)

        ev.asset_id = asset_id
        ev.persisted = True
        ev.observation_id = obs_id
        ev.image_reference = image_ref
        ev.heatmap_reference = heat_ref
        ev.mask_reference = mask_ref
        ev.defect = result["defect"]
        ev.revisit = result["revisit"]
        ev.defect_instances = result.get("instances") or []
        ev.work_orders = result.get("work_orders") or []
        if self.sync_images:
            self.store.enqueue_images(site_id, [image_ref, heat_ref, mask_ref])

    # ── history (FR-12) ───────────────────────────────────────────────────
    def _history_summary(self, asset_row: Dict[str, Any]) -> Dict[str, Any]:
        """Historical condition, latest prior image and unresolved defect notes of a matched asset."""
        last = self.store.latest_observation(asset_row["asset_id"])
        return {
            "previous_condition": asset_row["current_condition"],
            "reference_condition": asset_row["reference_condition"]["label"],
            "last_checked_ts": asset_row["last_checked_ts"],
            "previous_image": last["image_reference"] if last else None,
            "previous_severity": last["severity_score"] if last else None,
            "open_defects": [
                {k: d[k] for k in ("defect_id", "defect_type", "status", "severity_score", "notes")}
                for d in self.store.list_defects(asset_id=asset_row["asset_id"])
                if d["status"] != "Resolved"
            ],
        }

    def get_asset_history(self, asset_id: str) -> Dict[str, Any]:
        asset = self.store.get_asset(asset_id)
        if asset is None:
            return {"asset_id": asset_id, "found": False}
        obs = self.store.observations(asset_id)
        return {
            "asset_id": asset_id,
            "found": True,
            "asset": asset,
            "observations": obs,
            "previous_images": [o["image_reference"] for o in obs],
            "previous_conditions": [o["condition_description"] for o in obs],
            "severity_scores": [o["severity_score"] for o in obs],
            "defects": self.store.list_defects(asset_id=asset_id),
            "mission_severities": self.store.mission_severities(asset_id),
        }

    def end_session_hook(self, session_id: str) -> None:
        self._session_cache.pop(session_id, None)

    def recompute_site(self, site_id: str, now: Optional[datetime] = None) -> List[Dict[str, Any]]:
        return self.store.recompute_revisit(site_id, now)
