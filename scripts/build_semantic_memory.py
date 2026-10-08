#!/usr/bin/env python3
"""
Build AUDI semantic memory from the prepared PV panel dataset.

Constructs SIMULATED MULTI-MISSION HISTORIES for demo assets using the
existing SemanticMemoryEngine pipeline (embedding → match → change → severity).
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from memory.local_store import LocalSemanticMemoryStore
from ml.condition_mapping import map_pv_class
from ml.config import load_config
from ml.semantic_memory import SemanticMemoryEngine, register_new_asset, update_existing_asset
from ml.schemas import Location

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("build_memory")


def load_metadata(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def group_by_asset(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["asset_id"]].append(row)
    for aid in grouped:
        grouped[aid].sort(key=lambda r: r.get("mission_id", ""))
    return grouped


def mission_timestamp(mission_id: str, base: datetime) -> str:
    num = int(mission_id.split("_")[-1]) if "_" in mission_id else 1
    return (base + timedelta(days=7 * (num - 1))).isoformat()


def build_memory(
    metadata_path: Path,
    store_path: Path,
    config_path: Path | None = None,
) -> SemanticMemoryEngine:
    if config_path:
        import os
        os.environ["AUDI_CONFIG"] = str(config_path)

    config = load_config(reload=True)
    if store_path.exists():
        store_path.unlink()

    store = LocalSemanticMemoryStore(store_path=store_path, config=config)
    engine = SemanticMemoryEngine(store=store, config=config)

    rows = load_metadata(metadata_path)
    grouped = group_by_asset(rows)
    base_time = datetime(2026, 2, 1, 9, 0, 0, tzinfo=timezone.utc)

    logger.info("Building semantic memory for %d demo assets", len(grouped))

    demo_to_real: dict[str, str] = {}

    for asset_id, observations in grouped.items():
        is_simulated = any(r.get("simulated_history") == "true" for r in observations)
        label = "SIMULATED MULTI-MISSION HISTORY" if is_simulated else "DEMO ASSET"
        logger.info("Processing %s (%s) — %d observations", asset_id, label, len(observations))

        lat = 37.0 + hash(asset_id) % 100 / 1000
        lon = -122.0 + hash(asset_id) % 100 / 1000
        location = Location(latitude=lat, longitude=lon, altitude=45.0)

        for i, obs_row in enumerate(observations):
            img_path = ROOT / "dataset" / obs_row["image_path"]
            if not img_path.exists():
                logger.warning("Image not found: %s", img_path)
                continue

            cls = obs_row["class"]
            mapped = map_pv_class(cls)
            ts = mission_timestamp(obs_row["mission_id"], base_time)

            if i == 0 or asset_id not in demo_to_real:
                embedding = engine.encoder.encode_asset(str(img_path))
                result = register_new_asset(
                    image=str(img_path),
                    asset_type="solar_panel",
                    location=location,
                    timestamp=ts,
                    store=store,
                    encoder=engine.encoder,
                    config=config,
                    embedding=embedding,
                    condition=mapped["condition_description"],
                    image_reference=str(obs_row["image_path"]),
                )
                demo_to_real[asset_id] = result["asset_id"]
            else:
                real_id = demo_to_real[asset_id]
                embedding = engine.encoder.encode_asset(str(img_path))
                result = update_existing_asset(
                    asset_id=real_id,
                    image=str(img_path),
                    condition=mapped["condition_description"],
                    location=location,
                    timestamp=ts,
                    store=store,
                    encoder=engine.encoder,
                    change_detector=engine.change_detector,
                    config=config,
                    embedding=embedding,
                    image_reference=str(obs_row["image_path"]),
                    match_confidence=0.95,
                )

            stored = store.get_asset(demo_to_real[asset_id])
            if stored and stored.get("observations"):
                last = stored["observations"][-1]
                last["mission_id"] = obs_row["mission_id"]
                last["source_class"] = cls
                last["metadata"] = {
                    "demo_asset": True,
                    "simulated_history": obs_row.get("simulated_history") == "true",
                    "note": obs_row.get("note", ""),
                    "original_asset_id": asset_id,
                }
                if cls not in stored.get("defect_list", []) and mapped["defect_category"] != "none":
                    stored.setdefault("defect_list", []).append(mapped["defect_category"])
                stored.setdefault("metadata", {}).update({
                    "demo_asset": True,
                    "demo_asset_id": asset_id,
                    "simulated_history": is_simulated,
                    "research_demonstration": True,
                })
                store.save_asset(stored)

    logger.info("Semantic memory built: %d assets", len(engine.get_memory_state()))
    return engine


def main():
    parser = argparse.ArgumentParser(description="Build semantic memory from PV dataset")
    parser.add_argument("--metadata", type=str, default="dataset/small/metadata.csv")
    parser.add_argument("--output", type=str, default="data/pv_semantic_memory.json")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    build_memory(
        metadata_path=ROOT / args.metadata,
        store_path=ROOT / args.output,
        config_path=ROOT / args.config if args.config else None,
    )
    print(f"Memory saved to {ROOT / args.output}")


if __name__ == "__main__":
    main()
