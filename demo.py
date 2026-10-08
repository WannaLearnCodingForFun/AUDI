#!/usr/bin/env python3
"""
AUDI Semantic Memory Demo — multi-mission persistent memory simulation.

Demonstrates that AUDI maintains asset-level memory across inspection missions,
not treating each flight independently.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from memory.local_store import LocalSemanticMemoryStore
from ml.semantic_memory import SemanticMemoryEngine

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger("audi.demo")


def print_separator(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_memory_state(engine: SemanticMemoryEngine) -> None:
    assets = engine.get_memory_state()
    print(f"\n--- Semantic Memory State ({len(assets)} assets) ---")
    for asset in assets:
        n_obs = len(asset.get("observations", []))
        print(
            f"  {asset['asset_id']} | type={asset['asset_type']} | "
            f"condition={asset['current_condition']} | observations={n_obs} | "
            f"revisit_priority={asset.get('revisit_priority_score', 0):.2f}"
        )
        for obs in asset.get("observations", []):
            print(
                f"    └ obs {obs['observation_id'][:12]}… | "
                f"ts={obs['observation_ts'][:19]} | "
                f"condition={obs['condition_description']} | "
                f"change_M={obs.get('change_magnitude', 0):.2f} | "
                f"severity={obs.get('severity_score', 0):.2f}"
            )


def find_demo_image(dataset_dir: Path, asset_name: str, split: str = "train", index: int = 0) -> Path:
    """Locate demo image by asset name prefix."""
    for type_dir in (dataset_dir / split).iterdir():
        if not type_dir.is_dir():
            continue
        matches = sorted(type_dir.glob(f"{asset_name}_*.png"))
        if matches:
            return matches[min(index, len(matches) - 1)]
    raise FileNotFoundError(f"No demo image for {asset_name} in {dataset_dir}")


def run_demo() -> None:
    # Use demo config with stricter matching threshold for offline embedder
    os.environ["AUDI_CONFIG"] = str(ROOT / "config_demo.yaml")

    dataset_dir = ROOT / "dataset"
    if not (dataset_dir / "manifest.json").exists():
        logger.info("Generating demo dataset…")
        from scripts.generate_demo_dataset import generate_demo_dataset

        generate_demo_dataset(dataset_dir)

    # Fresh memory store for demo
    store_path = ROOT / "data" / "demo_semantic_memory.json"
    if store_path.exists():
        store_path.unlink()

    from ml.config import load_config

    config = load_config(reload=True)
    store = LocalSemanticMemoryStore(store_path=store_path, config=config)
    engine = SemanticMemoryEngine(store=store, config=config)

    base_time = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

    def ts(days: float = 0, hours: float = 0) -> str:
        return (base_time + timedelta(days=days, hours=hours)).isoformat()

    # ── MISSION 1 ──────────────────────────────────────────
    print_separator("MISSION 1 — Initial Inspection (all new assets)")

    observations_m1 = [
        ("solar_panel_a", "solar_panel", "normal", ts(0, 0)),
        ("solar_panel_b", "solar_panel", "normal", ts(0, 1)),
        ("utility_pole_a", "utility_pole", "normal", ts(0, 2)),
    ]

    for asset_name, asset_type, condition, timestamp in observations_m1:
        img_path = find_demo_image(dataset_dir, asset_name)
        result = engine.process_observation(
            img_path,
            asset_type=asset_type,
            location={"latitude": 37.0, "longitude": -122.0, "altitude": 50.0},
            timestamp=timestamp,
            condition=condition,
            force=True,
        )
        status = "NEW ASSET" if result.get("is_new_asset") else "MATCHED"
        print(f"  {asset_name} → {status} (id={result['asset_id']})")

    print_memory_state(engine)

    # ── MISSION 2 ──────────────────────────────────────────
    print_separator("MISSION 2 — Re-inspection + new asset")

    observations_m2 = [
        ("solar_panel_a", "solar_panel", "normal", ts(7, 0)),
        ("solar_panel_b", "solar_panel", "normal", ts(7, 1)),
        ("utility_pole_a", "utility_pole", "normal", ts(7, 2)),
        ("solar_panel_c", "solar_panel", "normal", ts(7, 3)),
    ]

    for asset_name, asset_type, condition, timestamp in observations_m2:
        img_path = find_demo_image(dataset_dir, asset_name)
        result = engine.process_observation(
            img_path,
            asset_type=asset_type,
            location={"latitude": 37.0, "longitude": -122.0, "altitude": 50.0},
            timestamp=timestamp,
            condition=condition,
            force=True,
        )
        status = "NEW ASSET" if result.get("is_new_asset") else "MATCHED"
        conf = result.get("match_confidence", 0)
        print(f"  {asset_name} → {status} (confidence={conf:.2f}, id={result['asset_id']})")

    print_memory_state(engine)

    # ── MISSION 3 ──────────────────────────────────────────
    print_separator("MISSION 3 — Condition change + revisit priority")

    # Solar panel A with degraded condition (use damaged image variant)
    img_a_damaged = find_demo_image(dataset_dir, "solar_panel_a", index=2)
    result_a = engine.process_observation(
        img_a_damaged,
        asset_type="solar_panel",
        location={"latitude": 37.0, "longitude": -122.0, "altitude": 50.0},
        timestamp=ts(14, 0),
        condition="minor_damage",
        force=True,
    )
    status = "MATCHED" if not result_a.get("is_new_asset") else "NEW ASSET"
    print(
        f"  solar_panel_a → {status} | change_M={result_a.get('change_magnitude', 0):.2f} | "
        f"severity={result_a.get('severity_score', 0):.2f} | "
        f"priority={result_a.get('priority', False)}"
    )

    # Re-inspect panel B (unchanged)
    img_b = find_demo_image(dataset_dir, "solar_panel_b")
    result_b = engine.process_observation(
        img_b,
        asset_type="solar_panel",
        location={"latitude": 37.0, "longitude": -122.0, "altitude": 50.0},
        timestamp=ts(14, 1),
        condition="normal",
        force=True,
    )
    print(
        f"  solar_panel_b → MATCHED | change_M={result_b.get('change_magnitude', 0):.2f} | "
        f"revisit_priority={result_b.get('revisit_priority', 0):.2f}"
    )

    # Completely new Asset B (building)
    img_building = find_demo_image(dataset_dir, "building_a")
    result_new = engine.process_observation(
        img_building,
        asset_type="building",
        location={"latitude": 37.1, "longitude": -122.1, "altitude": 45.0},
        timestamp=ts(14, 2),
        condition="normal",
        force=True,
    )
    status_new = "NEW ASSET" if result_new.get("is_new_asset") else "MATCHED"
    print(f"  building_a → {status_new} (id={result_new['asset_id']})")

    print_memory_state(engine)

    # Show full history for solar_panel_a
    if not result_a.get("is_new_asset"):
        history = engine.get_asset_history(result_a["asset_id"])
        print(f"\n--- Full history for {result_a['asset_id']} ({len(history.get('observations', []))} observations) ---")
        for obs in history.get("observations", []):
            print(
                f"  {obs['observation_ts'][:19]} | {obs['condition_description']} | "
                f"M={obs.get('change_magnitude', 0):.2f} | S={obs.get('severity_score', 0):.2f}"
            )

    print_separator("DEMO COMPLETE")
    print("Persistent semantic memory demonstrated across 3 missions.")
    print(f"Memory persisted to: {store_path}")


if __name__ == "__main__":
    run_demo()
