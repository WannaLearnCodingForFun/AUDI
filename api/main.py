"""FastAPI application — AUDI semantic memory API."""

from __future__ import annotations

import io
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from memory.local_store import LocalSemanticMemoryStore
from ml.condition_mapping import map_pv_class, normalize_class_name
from ml.config import load_config
from ml.semantic_memory import SemanticMemoryEngine
from ml.severity import is_priority

logger = logging.getLogger("audi.api")

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".heic", ".heif"}


def _load_upload_image(file: UploadFile, contents: bytes):
    """Validate and open an uploaded image (lenient MIME — macOS drag-drop often sends octet-stream)."""
    from PIL import Image

    if not contents:
        raise HTTPException(400, "Empty file uploaded.")

    ext = Path(file.filename or "").suffix.lower()
    content_type = (file.content_type or "").lower()

    is_image_mime = content_type.startswith("image/")
    is_known_ext = ext in ALLOWED_IMAGE_EXTENSIONS
    is_octet = content_type in ("application/octet-stream", "binary/octet-stream", "")

    if not (is_image_mime or is_known_ext or is_octet):
        raise HTTPException(
            400,
            f"Unsupported file type '{content_type or ext or 'unknown'}'. Upload JPEG, PNG, or WebP.",
        )

    try:
        return Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"Could not read image: {exc}") from exc


def _infer_condition_from_memory(
    engine: SemanticMemoryEngine,
    store: LocalSemanticMemoryStore,
    embedding,
    threshold: float = 0.85,
) -> tuple[Optional[str], Optional[str], float]:
    """
    Infer condition by finding the nearest stored observation embedding.
    Returns (condition_description, source_class, similarity).
    """
    import numpy as np
    from ml.similarity import batch_cosine_similarity

    best_sim = -1.0
    best_condition = None
    best_source = None

    for asset in store.list_assets():
        for obs in asset.get("observations", []):
            emb = obs.get("embedding")
            if not emb:
                continue
            sim = float(batch_cosine_similarity(embedding, np.array(emb, dtype=np.float32).reshape(1, -1))[0])
            if sim > best_sim:
                best_sim = sim
                best_condition = obs.get("condition_description")
                best_source = obs.get("source_class")

    if best_sim >= threshold:
        return best_condition, best_source, best_sim
    return None, None, best_sim

# ── App setup ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="AUDI Semantic Memory API",
    description="AI-Powered Persistent Semantic Memory for Drone Inspection",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine: Optional[SemanticMemoryEngine] = None
_store: Optional[LocalSemanticMemoryStore] = None


def get_engine() -> SemanticMemoryEngine:
    global _engine, _store
    if _engine is None:
        config = load_config(reload=True)
        store_path = ROOT / config.get("memory", {}).get("store_path", "data/pv_semantic_memory.json")
        if not store_path.exists():
            alt = ROOT / "data" / "pv_semantic_memory.json"
            store_path = alt if alt.exists() else store_path
        _store = LocalSemanticMemoryStore(store_path=store_path, config=config)
        _engine = SemanticMemoryEngine(store=_store, config=config)
        logger.info("Engine initialized with %d assets", len(_store.list_assets()))
    return _engine


def get_store() -> LocalSemanticMemoryStore:
    get_engine()
    assert _store is not None
    return _store


# Serve dataset images for frontend
_dataset_dir = ROOT / "dataset"
if _dataset_dir.exists():
    app.mount("/images", StaticFiles(directory=str(_dataset_dir)), name="images")


# ── Response models ────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model: str
    assets_count: int
    research_demonstration: bool = True


class AssetSummary(BaseModel):
    asset_id: str
    asset_type: str
    condition: str
    severity_score: float
    revisit_priority: float
    last_inspected: Optional[str] = None
    observations_count: int = 0
    demo_asset: bool = False
    simulated_history: bool = False


class AnalyzeResponse(BaseModel):
    asset_id: Optional[str] = None
    is_new_asset: bool = False
    match_confidence: float = 0.0
    current_condition: Optional[str] = None
    previous_condition: Optional[str] = None
    inferred_source_class: Optional[str] = None
    change_magnitude: float = 0.0
    severity_score: float = 0.0
    priority: bool = False
    revisit_priority: float = 0.0
    metrics: Optional[Dict[str, float]] = None
    error: Optional[str] = None


class MissionResult(BaseModel):
    mission_id: str
    assets_processed: int
    existing_matched: int
    new_assets: int
    changes_detected: int
    priority_alerts: int
    results: List[Dict[str, Any]]


# ── Helpers ────────────────────────────────────────────────────────────────

def _asset_summary(asset: Dict) -> AssetSummary:
    obs = asset.get("observations", [])
    last = obs[-1] if obs else {}
    meta = asset.get("metadata", {})
    return AssetSummary(
        asset_id=asset["asset_id"],
        asset_type=asset.get("asset_type", "unknown"),
        condition=asset.get("current_condition", "unknown"),
        severity_score=float(last.get("severity_score", 0.0)),
        revisit_priority=float(asset.get("revisit_priority_score", 0.0)),
        last_inspected=last.get("observation_ts"),
        observations_count=len(obs),
        demo_asset=bool(meta.get("demo_asset")),
        simulated_history=bool(meta.get("simulated_history")),
    )


def _resolve_image_path(ref: str) -> Optional[Path]:
    if not ref:
        return None
    p = ROOT / "dataset" / ref if not ref.startswith("/") else Path(ref)
    if p.exists():
        return p
    p2 = ROOT / ref
    return p2 if p2.exists() else None


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
def health():
    try:
        engine = get_engine()
        config = load_config()
        model_name = config.get("model", {}).get("name", "unknown")
        count = len(engine.get_memory_state())
        return HealthResponse(status="ok", model=model_name, assets_count=count)
    except Exception as exc:
        logger.error("Health check failed: %s", exc)
        return HealthResponse(status="degraded", model="unavailable", assets_count=0)


@app.get("/api/summary")
def summary():
    try:
        assets = get_store().list_assets()
    except Exception as exc:
        raise HTTPException(503, f"Memory store unavailable: {exc}")

    if not assets:
        return {
            "total_assets": 0,
            "active_defects": 0,
            "priority_assets": 0,
            "missions": 0,
            "research_demonstration": True,
            "message": "Semantic memory is empty. Run scripts/build_semantic_memory.py",
        }

    config = load_config()
    priority_threshold = float(config.get("severity", {}).get("priority_threshold", 0.5))

    defects = sum(1 for a in assets if a.get("current_condition", "Normal") != "Normal")
    priority = sum(
        1 for a in assets
        if (a.get("observations") or [{}])[-1].get("severity_score", 0) > priority_threshold
    )
    mission_ids = set()
    for a in assets:
        for o in a.get("observations", []):
            if o.get("mission_id"):
                mission_ids.add(o["mission_id"])

    summaries = sorted(
        [_asset_summary(a).model_dump() for a in assets],
        key=lambda x: x["revisit_priority"],
        reverse=True,
    )

    return {
        "total_assets": len(assets),
        "active_defects": defects,
        "priority_assets": priority,
        "missions": len(mission_ids) or max(len(o) for o in [[obs for a in assets for obs in a.get("observations", [])]] or [[]]),
        "research_demonstration": True,
        "priority_assets_list": summaries[:10],
    }


@app.get("/api/assets")
def list_assets():
    try:
        assets = get_store().list_assets()
        return {"assets": [_asset_summary(a).model_dump() for a in assets]}
    except Exception as exc:
        raise HTTPException(503, f"Failed to load assets: {exc}")


@app.get("/api/assets/{asset_id}")
def get_asset(asset_id: str):
    asset = get_store().get_asset(asset_id)
    if not asset:
        raise HTTPException(404, f"Asset not found: {asset_id}")
    summary = _asset_summary(asset)
    return {**asset, **summary.model_dump()}


@app.get("/api/assets/{asset_id}/history")
def get_asset_history(asset_id: str):
    asset = get_store().get_asset(asset_id)
    if not asset:
        raise HTTPException(404, f"Asset not found: {asset_id}")

    observations = []
    for obs in asset.get("observations", []):
        img_url = None
        ref = obs.get("image_reference", "")
        if ref:
            img_url = f"/images/{ref}" if not ref.startswith("/") else ref
        observations.append({**obs, "image_url": img_url})

    from ml.revisit_priority import compute_revisit_from_history

    severity_hist = [o.get("severity_score", 0) for o in observations]
    timestamps = [o.get("observation_ts", "") for o in observations]
    revisit = compute_revisit_from_history(severity_hist, timestamps)

    return {
        "asset_id": asset_id,
        "asset_type": asset.get("asset_type"),
        "current_condition": asset.get("current_condition"),
        "demo_asset": asset.get("metadata", {}).get("demo_asset", False),
        "simulated_history": asset.get("metadata", {}).get("simulated_history", False),
        "research_demonstration": True,
        "observations": observations,
        "severity_history": severity_hist,
        "revisit_priority": revisit,
    }


@app.get("/api/memory/search")
def search_memory(q: str = Query(..., min_length=1)):
    assets = get_store().list_assets()
    q_lower = q.lower()
    matches = [
        _asset_summary(a).model_dump()
        for a in assets
        if q_lower in a["asset_id"].lower()
        or q_lower in a.get("current_condition", "").lower()
        or q_lower in a.get("asset_type", "").lower()
        or q_lower in str(a.get("metadata", {})).lower()
    ]
    return {"query": q, "results": matches}


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_image(
    file: UploadFile = File(...),
    asset_type: str = "solar_panel",
    condition: Optional[str] = None,
    add_to_memory: bool = True,
):
    contents = await file.read()
    image = _load_upload_image(file, contents)

    try:
        engine = get_engine()
        store = get_store()

        # Infer condition from nearest stored observation (not hard-coded "Normal")
        embedding = engine.encoder.encode_asset(image)
        inferred_condition, inferred_source, infer_sim = _infer_condition_from_memory(
            engine, store, embedding
        )
        condition_to_use = condition or inferred_condition or "Unknown"

        result = engine.process_observation(
            image,
            asset_type=asset_type,
            condition=condition_to_use,
            timestamp=datetime.now(timezone.utc).isoformat(),
            image_reference=file.filename or "upload",
            force=True,
        )

        if result.get("error"):
            return AnalyzeResponse(error=result["error"], is_new_asset=False)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Analysis failed: %s", exc, exc_info=True)
        return AnalyzeResponse(error=str(exc), is_new_asset=False)

    return AnalyzeResponse(
        asset_id=result.get("asset_id"),
        is_new_asset=bool(result.get("is_new_asset")),
        match_confidence=float(result.get("match_confidence", 0.0)),
        current_condition=result.get("current_condition"),
        previous_condition=result.get("previous_condition"),
        inferred_source_class=inferred_source,
        change_magnitude=float(result.get("change_magnitude", 0.0)),
        severity_score=float(result.get("severity_score", 0.0)),
        priority=is_priority(result.get("severity_score", 0.0)),
        revisit_priority=float(result.get("revisit_priority", 0.0)),
        metrics=result.get("metrics"),
    )


@app.post("/api/simulate-mission", response_model=MissionResult)
def simulate_mission():
    """Process sample images as a simulated drone inspection mission."""
    metadata_path = ROOT / "dataset" / "small" / "metadata.csv"
    if not metadata_path.exists():
        raise HTTPException(
            404,
            "Dataset not prepared. Run: python scripts/prepare_pv_dataset.py",
        )

    import csv
    import random

    with open(metadata_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise HTTPException(404, "No images in metadata.csv")

    # Pick 8 random images for this mission
    sample = random.sample(rows, min(8, len(rows)))
    mission_id = f"mission_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    engine = get_engine()

    matched = new = changes = priority = 0
    results = []

    for row in sample:
        img_path = ROOT / "dataset" / row["image_path"]
        if not img_path.exists():
            continue

        mapped = map_pv_class(row["class"])
        try:
            result = engine.process_observation(
                img_path,
                asset_type="solar_panel",
                condition=mapped["condition_description"],
                timestamp=datetime.now(timezone.utc).isoformat(),
                image_reference=row["image_path"],
                force=True,
            )
            if result.get("is_new_asset"):
                new += 1
            else:
                matched += 1
            if result.get("change_magnitude", 0) > 0.3:
                changes += 1
            if is_priority(result.get("severity_score", 0)):
                priority += 1
            results.append({
                "asset_id": result.get("asset_id"),
                "is_new_asset": result.get("is_new_asset"),
                "match_confidence": result.get("match_confidence"),
                "condition": result.get("current_condition"),
                "change_magnitude": result.get("change_magnitude"),
                "severity_score": result.get("severity_score"),
                "image_path": row["image_path"],
            })
        except Exception as exc:
            logger.error("Mission processing error: %s", exc)
            results.append({"error": str(exc), "image_path": row["image_path"]})

    return MissionResult(
        mission_id=mission_id,
        assets_processed=len(results),
        existing_matched=matched,
        new_assets=new,
        changes_detected=changes,
        priority_alerts=priority,
        results=results,
    )


@app.get("/api/missions")
def list_missions():
    assets = get_store().list_assets()
    missions: Dict[str, List] = {}
    for asset in assets:
        for obs in asset.get("observations", []):
            mid = obs.get("mission_id", "unknown")
            missions.setdefault(mid, []).append({
                "asset_id": asset["asset_id"],
                "condition": obs.get("condition_description"),
                "severity": obs.get("severity_score"),
                "timestamp": obs.get("observation_ts"),
                "image_reference": obs.get("image_reference"),
            })
    return {"missions": missions, "research_demonstration": True}
