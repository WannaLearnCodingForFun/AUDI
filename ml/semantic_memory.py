"""Persistent semantic memory engine — end-to-end ML pipeline."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union 

import numpy as np
from PIL import Image

from memory.local_store import LocalSemanticMemoryStore
from ml.asset_encoder import AssetEncoder, BBox
from ml.change_detector import ChangeDetector
from ml.config import get_nested, load_config
from ml.memory_matcher import AssetMatcher
from ml.revisit_priority import compute_revisit_from_history
from ml.schemas import (
    AssetRecord,
    Location,
    Observation,
    ProcessingMetrics,
    generate_asset_id,
    generate_observation_id,
    utc_now_iso,
)
from ml.severity import calculate_severity, is_priority

logger = logging.getLogger(__name__)

ImageInput = Union[Image.Image, np.ndarray, str, Path]


def _load_image(image: ImageInput) -> Image.Image:
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image.astype(np.uint8)).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")


def _asset_criticality(asset_type: str, config: Dict) -> float:
    mapping = get_nested(config, "asset_criticality", default={}) or {}
    return float(mapping.get(asset_type, mapping.get("default", 0.5)))


class SemanticMemoryEngine:
    """
    End-to-end semantic memory pipeline for drone inspection observations.

    Decoupled from UI and database specifics via SemanticMemoryStore abstraction.
    """

    def __init__(
        self,
        store: Optional[LocalSemanticMemoryStore] = None,
        encoder: Optional[AssetEncoder] = None,
        matcher: Optional[AssetMatcher] = None,
        change_detector: Optional[ChangeDetector] = None,
        config: Optional[Dict] = None,
    ):
        self.config = config or load_config()
        self.store = store or LocalSemanticMemoryStore(config=self.config)
        model_name = get_nested(self.config, "model", "name", default="dinov2")
        device = get_nested(self.config, "model", "device", default="auto")
        self.encoder = encoder or AssetEncoder(model_name=model_name, device=device)
        self.matcher = matcher or AssetMatcher()
        change_threshold = float(get_nested(self.config, "change_detection", "threshold", default=0.30))
        self.change_detector = change_detector or ChangeDetector(threshold=change_threshold)

        self._frame_counter = 0
        self._last_embedding: Optional[np.ndarray] = None

    def _should_process_frame(self) -> bool:
        frame_skip = int(get_nested(self.config, "runtime", "frame_skip", default=1))
        self._frame_counter += 1
        return self._frame_counter % max(frame_skip, 1) == 0

    def process_observation(
        self,
        image: ImageInput,
        asset_type: Optional[str] = None,
        location: Optional[Dict[str, float]] = None,
        timestamp: Optional[str] = None,
        condition: Optional[str] = None,
        bbox: Optional[BBox] = None,
        image_reference: Optional[str] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Full pipeline: embed → search → match/register → change analysis → update memory.
        """
        if not force and not self._should_process_frame():
            return {"skipped": True, "reason": "frame_skip"}

        t_total_start = time.perf_counter()
        metrics = ProcessingMetrics()

        pil = _load_image(image)
        asset_type = asset_type or "unknown"
        location_obj = Location.from_dict(location)
        ts = timestamp or utc_now_iso()
        image_ref = image_reference or (str(image) if isinstance(image, (str, Path)) else "")

        # Embedding
        t0 = time.perf_counter()
        try:
            embedding = self.encoder.encode_asset(pil, bbox=bbox)
        except Exception as exc:
            logger.error("Embedding failed: %s", exc, exc_info=True)
            return {"error": str(exc), "success": False}
        metrics.embedding_latency_ms = (time.perf_counter() - t0) * 1000

        # Skip if embedding nearly identical to last (avoid redundant work)
        if self._last_embedding is not None:
            from ml.similarity import cosine_similarity

            if cosine_similarity(embedding, self._last_embedding) > 0.999:
                logger.debug("Skipping near-duplicate frame")
        self._last_embedding = embedding

        match_threshold = float(get_nested(self.config, "matching", "similarity_threshold", default=0.75))
        top_k = int(get_nested(self.config, "matching", "top_k", default=5))

        # Search memory — build gallery from all observation embeddings
        t1 = time.perf_counter()
        logger.info("Searching semantic memory")
        candidate_ids, candidate_matrix = self._build_embedding_gallery()
        metrics.memory_lookup_latency_ms = (time.perf_counter() - t1) * 1000

        # Match
        t2 = time.perf_counter()
        match = self.matcher.match(embedding, candidate_matrix, candidate_ids, match_threshold)
        metrics.matching_latency_ms = (time.perf_counter() - t2) * 1000

        if match.matched and match.asset_id:
            logger.info("Existing asset matched: %s", match.asset_id)
            logger.info("Match confidence: %.2f", match.confidence)
            result = update_existing_asset(
                asset_id=match.asset_id,
                image=pil,
                condition=condition or "normal",
                location=location_obj,
                timestamp=ts,
                store=self.store,
                encoder=self.encoder,
                change_detector=self.change_detector,
                config=self.config,
                embedding=embedding,
                image_reference=image_ref,
                match_confidence=match.confidence,
            )
            result["is_new_asset"] = False
            result["match_confidence"] = match.confidence
        else:
            logger.info("No match found — registering new asset")
            result = register_new_asset(
                image=pil,
                asset_type=asset_type,
                location=location_obj,
                timestamp=ts,
                store=self.store,
                encoder=self.encoder,
                config=self.config,
                embedding=embedding,
                condition=condition or "normal",
                image_reference=image_ref,
            )
            result["is_new_asset"] = True
            result["match_confidence"] = result.get("confidence", 1.0)

        metrics.total_processing_latency_ms = (time.perf_counter() - t_total_start) * 1000
        result["metrics"] = metrics.to_dict()
        result["priority"] = is_priority(result.get("severity_score", 0.0), self.config)
        logger.info("Semantic memory updated")
        return result

    def get_asset_history(self, asset_id: str) -> Dict[str, Any]:
        return get_asset_history(asset_id, self.store)

    def compare_with_history(self, asset_id: str, current_observation: Dict[str, Any]) -> Dict[str, Any]:
        return compare_with_history(asset_id, current_observation, self.store, self.change_detector)

    def get_memory_state(self) -> List[Dict[str, Any]]:
        return self.store.list_assets()

    def _build_embedding_gallery(self) -> tuple[List[str], np.ndarray]:
        """Collect all observation embeddings for gallery-based matching."""
        ids: List[str] = []
        vectors: List[np.ndarray] = []
        for asset in self.store.list_assets():
            aid = asset["asset_id"]
            for obs in asset.get("observations", []):
                emb = obs.get("embedding")
                if emb:
                    ids.append(aid)
                    vectors.append(np.array(emb, dtype=np.float32))
        if not vectors:
            return [], np.zeros((0, 0), dtype=np.float32)
        return ids, np.stack(vectors)


def register_new_asset(
    image: ImageInput,
    asset_type: str,
    location: Location,
    timestamp: str,
    store: LocalSemanticMemoryStore,
    encoder: AssetEncoder,
    config: Dict,
    embedding: Optional[np.ndarray] = None,
    condition: str = "normal",
    image_reference: str = "",
    bbox: Optional[BBox] = None,
) -> Dict[str, Any]:
    """Register a new asset with baseline observation and vector memory entry."""
    logger.info("Asset detected — registering new asset type=%s", asset_type)

    pil = _load_image(image) if not isinstance(image, Image.Image) else image
    if embedding is None:
        embedding = encoder.encode_asset(pil, bbox=bbox)

    asset_id = generate_asset_id()
    criticality = _asset_criticality(asset_type, config)

    observation = Observation(
        observation_id=generate_observation_id(),
        observation_ts=timestamp,
        condition_description=condition,
        image_reference=image_reference,
        embedding=embedding.tolist(),
        severity_score=0.0,
        change_magnitude=0.0,
        confidence=1.0,
    )

    record = AssetRecord(
        asset_id=asset_id,
        asset_type=asset_type,
        location=location,
        first_observed_ts=timestamp,
        current_condition=condition,
        current_embedding=embedding.tolist(),
        condition_history=[condition],
        observations=[observation],
        asset_criticality=criticality,
    )

    store.save_asset(record.to_dict())
    logger.info("New asset registered: %s", asset_id)

    return {
        "asset_id": asset_id,
        "is_new": True,
        "embedding": embedding.tolist(),
        "confidence": 1.0,
        "change_magnitude": 0.0,
        "severity_score": 0.0,
        "current_condition": condition,
        "previous_condition": None,
        "asset_record": record.to_dict(),
    }


def update_existing_asset(
    asset_id: str,
    image: ImageInput,
    condition: str,
    location: Location,
    timestamp: str,
    store: LocalSemanticMemoryStore,
    encoder: AssetEncoder,
    change_detector: ChangeDetector,
    config: Dict,
    embedding: Optional[np.ndarray] = None,
    image_reference: str = "",
    match_confidence: float = 1.0,
    bbox: Optional[BBox] = None,
) -> Dict[str, Any]:
    """Append observation to existing asset; never overwrite historical data."""
    asset_dict = store.get_asset(asset_id)
    if asset_dict is None:
        raise KeyError(f"Asset not found: {asset_id}")

    record = AssetRecord.from_dict(asset_dict)
    previous_condition = record.current_condition

    pil = _load_image(image) if not isinstance(image, Image.Image) else image
    if embedding is None:
        embedding = encoder.encode_asset(pil, bbox=bbox)

    historical_embeddings = [obs.embedding for obs in record.observations if obs.embedding]
    change = change_detector.compare(embedding, historical_embeddings)
    logger.info("Change magnitude: %.2f", change.change_magnitude)

    severity = calculate_severity(
        change.change_magnitude,
        record.asset_criticality,
        match_confidence,
        config=config,
    )
    logger.info("Severity score: %.2f", severity)

    observation = Observation(
        observation_id=generate_observation_id(),
        observation_ts=timestamp,
        condition_description=condition,
        image_reference=image_reference,
        embedding=embedding.tolist(),
        severity_score=severity,
        change_magnitude=change.change_magnitude,
        confidence=match_confidence,
    )

    record.observations.append(observation)
    record.condition_history.append(condition)
    record.current_condition = condition
    record.current_embedding = embedding.tolist()
    record.location = location

    severity_history = [obs.severity_score for obs in record.observations]
    timestamps = [obs.observation_ts for obs in record.observations]
    revisit_info = compute_revisit_from_history(severity_history, timestamps, timestamp, config)
    record.revisit_priority_score = revisit_info["revisit_priority"]

    store.save_asset(record.to_dict())

    return {
        "asset_id": asset_id,
        "is_new": False,
        "change_magnitude": change.change_magnitude,
        "severity_score": severity,
        "current_condition": condition,
        "previous_condition": previous_condition,
        "match_confidence": match_confidence,
        "revisit_priority": revisit_info["revisit_priority"],
        "trend": revisit_info["trend"],
        "historical_observations": [obs.to_dict() for obs in record.observations],
        "asset_record": record.to_dict(),
    }


def get_asset_history(asset_id: str, store: LocalSemanticMemoryStore) -> Dict[str, Any]:
    """Retrieve complete historical data for an asset."""
    asset = store.get_asset(asset_id)
    if not asset:
        return {"asset_id": asset_id, "found": False}

    observations = asset.get("observations", [])
    return {
        "asset_id": asset_id,
        "found": True,
        "asset_type": asset.get("asset_type"),
        "previous_embeddings": [o.get("embedding", []) for o in observations],
        "previous_images": [o.get("image_reference", "") for o in observations],
        "previous_conditions": [o.get("condition_description", "") for o in observations],
        "severity_scores": [o.get("severity_score", 0.0) for o in observations],
        "defect_list": asset.get("defect_list", []),
        "timestamps": [o.get("observation_ts", "") for o in observations],
        "observations": observations,
        "condition_history": asset.get("condition_history", []),
    }


def compare_with_history(
    asset_id: str,
    current_observation: Dict[str, Any],
    store: LocalSemanticMemoryStore,
    change_detector: ChangeDetector,
) -> Dict[str, Any]:
    """Compare current observation against full historical sequence."""
    history = get_asset_history(asset_id, store)
    if not history.get("found"):
        return {"asset_id": asset_id, "found": False}

    current_embedding = current_observation.get("embedding", [])
    historical_embeddings = history["previous_embeddings"]

    change = change_detector.compare(current_embedding, historical_embeddings)
    previous_condition = history["previous_conditions"][-1] if history["previous_conditions"] else "unknown"
    current_condition = current_observation.get("condition_description", "unknown")

    severity_history = history["severity_scores"]
    from ml.revisit_priority import calculate_trend

    trend = calculate_trend(severity_history)

    return {
        "asset_id": asset_id,
        "match_confidence": current_observation.get("confidence", 0.0),
        "change_magnitude": change.change_magnitude,
        "previous_condition": previous_condition,
        "current_condition": current_condition,
        "historical_observations": history["observations"],
        "trend": trend,
        "changed": change.changed,
    }
