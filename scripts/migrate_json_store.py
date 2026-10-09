#!/usr/bin/env python3
"""One-off migration of the old JSON semantic-memory store into the SQLite DB.

The legacy store (storage.legacy_json_store) holds simulated multi-mission
histories built from DIFFERENT physical panels, scored with the old (defective)
formulas (D1/D4/D5). It is preserved for traceability, not treated as evidence:

- everything goes into its own site (`storage.legacy_site_id`) so it can never be
  matched against real assets of an inspection site;
- assets, observations and sessions are flagged `simulated` and `meta.legacy`;
- the original severity / change numbers are kept in `meta.legacy_scores` only;
  stored severity/change fields are 0 (they were not produced by this system);
- one ended session per legacy mission_id;
- images under dataset/ are copied to data/images/<site>/<asset>/ and re-embedded
  with the configured backbone as identity prototypes.

Idempotent: assets that already exist are skipped.

Usage: python scripts/migrate_json_store.py [--source data/pv_semantic_memory.json]
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from memory import db  # noqa: E402
from memory.store import LocalStore  # noqa: E402
from ml.condition_mapping import UNIFIED_CONDITIONS, match_rule  # noqa: E402
from ml.config import cfg, load_config, resolve_path  # noqa: E402
from ml.embedding_model import AssetEmbeddingModel  # noqa: E402
from ml.schemas import new_id, utc_now_iso  # noqa: E402
from ml.severity import asset_criticality  # noqa: E402

DESCRIPTION_MAP = {
    "normal": "normal", "clean": "normal", "soiling": "soiling", "dusty": "soiling",
    "surface contamination": "bird_drop", "bird-drop": "bird_drop", "bird_drop": "bird_drop",
    "electrical defect": "electrical_damage", "electrical-damage": "electrical_damage",
    "physical defect": "physical_damage", "physical-damage": "physical_damage",
    "surface obstruction": "snow_cover", "snow-covered": "snow_cover", "snow": "snow_cover",
}


def parse(value: Any) -> Any:
    """Legacy files sometimes hold Python reprs instead of JSON values."""
    if isinstance(value, str) and value[:1] in "[{":
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return ast.literal_eval(value)
    return value


def unified_label(asset_type: str, obs: Dict[str, Any]) -> Optional[str]:
    src = obs.get("source_class")
    if src:
        rule = match_rule("pv_defect", str(src))
        if rule is not None and rule.asset_type == asset_type:
            return rule.condition
    label = DESCRIPTION_MAP.get(str(obs.get("condition_description", "")).strip().lower())
    return label if label in UNIFIED_CONDITIONS.get(asset_type, []) else None


def find_image(ref: Optional[str]) -> Optional[Path]:
    if not ref:
        return None
    for base in (ROOT / "dataset", ROOT):
        p = base / ref
        if p.is_file():
            return p
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--source", default=None, help="legacy JSON store (default: storage.legacy_json_store)")
    args = parser.parse_args()

    config = load_config(args.config)
    scfg = cfg(config, "storage")
    src = resolve_path(args.source or scfg["legacy_json_store"])
    if not src.exists():
        print(f"[skip] no legacy store at {src}; nothing to migrate")
        return
    data = json.loads(src.read_text(encoding="utf-8"))
    assets = data["assets"] if isinstance(data, dict) else data
    site_id = scfg["legacy_site_id"]
    store = LocalStore(config)
    store.ensure_site(site_id, scfg["legacy_site_name"])
    embedder = AssetEmbeddingModel.from_config(config)

    sessions: Dict[str, str] = {}
    counts = defaultdict(int)
    for raw in assets:
        a = {k: parse(v) for k, v in raw.items()}
        asset_id = a["asset_id"]
        if store.get_asset(asset_id) is not None:
            counts["skipped_existing"] += 1
            continue
        asset_type = a.get("asset_type") or "solar_panel"
        if asset_type not in UNIFIED_CONDITIONS:
            counts["skipped_unknown_type"] += 1
            continue
        labels = UNIFIED_CONDITIONS[asset_type]
        observations = [parse(o) if not isinstance(o, dict) else o for o in (a.get("observations") or [])]
        observations = [{k: parse(v) for k, v in o.items()} for o in observations]
        observations.sort(key=lambda o: o.get("observation_ts", ""))
        meta = a.get("metadata") or {}
        loc = a.get("location") or {}
        last_label = unified_label(asset_type, observations[-1]) if observations else None
        current = last_label or "normal"
        now = utc_now_iso()
        img_dir = store.asset_image_dir(site_id, asset_id)

        with store.txn() as s:
            asset = db.Asset(
                asset_id=asset_id, site_id=site_id, asset_type=asset_type,
                latitude=loc.get("latitude"), longitude=loc.get("longitude"), altitude=loc.get("altitude"),
                location_samples=1 if loc else 0,
                first_observed_ts=a.get("first_observed_ts") or (observations[0]["observation_ts"] if observations else now),
                last_checked_ts=observations[-1]["observation_ts"] if observations else now,
                current_condition=current, current_is_defect=current != "normal",
                reference_condition={"label": current, "scores": {l: float(l == current) for l in labels},
                                     "is_defect": current != "normal", "defect_type": None if current == "normal" else current},
                criticality=asset_criticality(asset_type, config),
                simulated=True,
                meta={"legacy": True, "legacy_source": src.name, **meta},
                updated_ts=now,
            )
            s.add(asset)
            s.flush()
            last_image = None
            for o in observations:
                mission = str(o.get("mission_id") or "legacy_unknown_mission")
                if mission not in sessions:
                    sid = new_id("sess")
                    ts = o.get("observation_ts") or now
                    s.add(db.Session(
                        session_id=sid, site_id=site_id, operator_id="legacy_import", mission_date=ts[:10],
                        started_ts=ts, ended_ts=ts, status="ended", mode="offline", assets_processed=0, simulated=True,
                        start_sync={"legacy_mission_id": mission},
                    ))
                    s.flush()
                    sessions[mission] = sid
                label = unified_label(asset_type, o) or "normal"
                image_ref = None
                img = find_image(o.get("image_reference"))
                if img is not None:
                    img_dir.mkdir(parents=True, exist_ok=True)
                    dest = img_dir / f"{o.get('observation_id', new_id('obs'))}{img.suffix.lower()}"
                    shutil.copy2(img, dest)
                    image_ref = dest.relative_to(store.images_dir).as_posix()
                    last_image = (dest, image_ref, o.get("observation_id"))
                    counts["images_copied"] += 1
                s.add(db.Observation(
                    observation_id=o.get("observation_id") or new_id("obs"), asset_id=asset_id, session_id=sessions[mission],
                    observation_ts=o.get("observation_ts") or now, condition_description=label,
                    condition_scores={l: float(l == label) for l in labels}, is_defect=label != "normal",
                    defect_type=None if label == "normal" else label, image_reference=image_ref,
                    severity_score=0.0, severity_level="low", change_magnitude=0.0,
                    observation_confidence=0.0, detector_confidence=0.0, classifier_confidence=0.0,
                    match_stage="legacy_import", match_confidence=0.0, status="confirmed", simulated=True,
                    meta={"legacy": True, "legacy_scores": {
                        "severity_score": o.get("severity_score"), "change_magnitude": o.get("change_magnitude"),
                        "confidence": o.get("confidence"), "condition_description": o.get("condition_description"),
                    }, **(o.get("metadata") or {})},
                ))
                counts["observations"] += 1
                sess = s.get(db.Session, sessions[mission])
                sess.assets_processed += 1
            if last_image is not None:
                with Image.open(last_image[0]) as im:
                    vec = embedder.encode_image(im.convert("RGB"))
                proto = db.AssetPrototype(
                    proto_uid=new_id("proto"), asset_id=asset_id, observation_id=last_image[2],
                    embedding=vec.astype("float32").tobytes(), dim=int(vec.shape[0]),
                    image_reference=last_image[1], created_ts=now,
                )
                s.add(proto)
                s.flush()
                asset.reference_proto_id = proto.proto_id
                counts["prototypes"] += 1
        counts["assets"] += 1

    store.recompute_revisit(site_id, enqueue=False)
    if store.get_meta("backbone") is None:
        store.set_meta("backbone", {"name": embedder.active_model_name, "dim": embedder.embedding_dim})
    print(f"[done] migrated from {src.name} into site '{site_id}' (all flagged simulated/legacy): {dict(counts)}")
    print(f"       legacy sessions created: {len(sessions)}")


if __name__ == "__main__":
    main()
