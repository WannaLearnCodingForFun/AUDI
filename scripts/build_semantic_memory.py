#!/usr/bin/env python3
"""Build a SIMULATED inspection history through the real pipeline (D5).

For each asset type, `memory_build.assets_per_type` test-split images that the
dataset labels `normal` become simulated physical assets. Each mission is a
real session (MissionService start → observe → end, so FR-4/FR-18/FR-20 run)
marked `simulated`. Every mission re-photographs the SAME image through the
webcam degradation model (new viewpoint, blur, colour cast); from
`damage_from_mission` on, a fraction of assets gets a progressively stronger
synthetic crack or soiling overlay. Identity, change, severity, defects and
revisit priority are all computed by the system — nothing is hard-coded.

This is demonstration data, not evidence: re-ID accuracy must come from real
repeated views (scripts/capture_reid_set.py + training/evaluate_reid.py).

Usage: python scripts/build_semantic_memory.py [--reset]
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from collections import Counter
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402
from sqlalchemy import select  # noqa: E402

from api.services import build_services  # noqa: E402
from memory import db  # noqa: E402
from ml.augment import WebcamAugment  # noqa: E402
from ml.config import ROOT as CFG_ROOT, cfg, load_config  # noqa: E402
from ml.detector import FullFrameDetector  # noqa: E402
from ml.schemas import utc_now  # noqa: E402
from ml.synthetic import apply_defect  # noqa: E402
from training.common import load_manifest  # noqa: E402

logger = logging.getLogger("build_semantic_memory")

DEFECT_KIND = {"solar_panel": "soiling", "building": "crack", "utility_pole": "crack", "machinery": "crack"}


def reset_site(store, site_id: str) -> int:
    """Delete a previous simulated build of this site (only simulated sessions/assets)."""
    with store.txn() as s:
        assets = list(s.scalars(select(db.Asset).where(db.Asset.site_id == site_id, db.Asset.simulated.is_(True))))
        for a in assets:
            s.delete(a)
        for sess in s.scalars(select(db.Session).where(db.Session.site_id == site_id, db.Session.simulated.is_(True))):
            for o in s.scalars(select(db.Observation).where(db.Observation.session_id == sess.session_id)):
                s.delete(o)
            s.delete(sess)
    return len(assets)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--reset", action="store_true", help="remove a previous simulated build of the site first")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = load_config(args.config)
    mb = cfg(config, "memory_build")
    rng = random.Random(int(mb["seed"]))
    services = build_services(
        config,
        detector=FullFrameDetector(cfg(config, "detection", "full_frame_asset_type"), float(cfg(config, "detection", "full_frame_confidence"))),
        use_cloud=False,
    )
    store, engine, mission = services.store, services.engine, services.mission
    site_id = mb["site_id"]
    store.ensure_site(site_id, mb["site_name"])
    if args.reset:
        n = reset_site(store, site_id)
        engine.reload_index()
        logger.info("removed %d simulated assets from %s", n, site_id)

    rows = load_manifest(config)
    view = WebcamAugment(mb["view_augment"])
    assets = []
    for asset_type in services.classifier.label_maps:
        pool = [r for r in rows if r.split == "test" and r.asset_type == asset_type and r.condition == "normal"]
        rng.shuffle(pool)
        for r in pool[: int(mb["assets_per_type"])]:
            with Image.open(CFG_ROOT / r.path) as im:
                base = im.convert("RGB")
            assets.append({
                "asset_type": asset_type, "image": base, "source": r.path,
                "damaged": rng.random() < float(mb["damaged_fraction"]), "seed": rng.randrange(1 << 30),
            })
    logger.info("%d simulated assets (%s)", len(assets), dict(Counter(a["asset_type"] for a in assets)))

    n_missions = int(mb["missions"])
    start = utc_now() - timedelta(days=int(mb["days_between_missions"]) * n_missions)
    stats = Counter()
    for m in range(n_missions):
        mission_ts = start + timedelta(days=int(mb["days_between_missions"]) * m)
        started = mission.start(site_id, "simulator", simulated=True)
        sid = started["session"]["session_id"]
        store.update_session(sid, started_ts=mission_ts.isoformat(), mission_date=mission_ts.date().isoformat())
        for i, a in enumerate(assets):
            img = view(a["image"], random.Random(a["seed"] + 7919 * m))
            note = "re-photographed (simulated viewpoint)"
            if a["damaged"] and m >= int(mb["damage_from_mission"]):
                steps = m - int(mb["damage_from_mission"]) + 1
                strength = min(float(mb["damage_max"]), float(mb["damage_step"]) * steps)
                img, note = apply_defect(img, DEFECT_KIND[a["asset_type"]], random.Random(a["seed"]), strength)
            ts = (mission_ts + timedelta(minutes=i)).isoformat()
            ev = engine.observe(img, a["asset_type"], sid, persist=True, timestamp=ts)
            if ev.type == "error":
                logger.error("mission %d asset %d: %s", m + 1, i, ev.error)
                stats["errors"] += 1
                continue
            stats[ev.type] += 1
            if ev.asset_id and ev.persisted:
                with store.txn() as s:
                    obs = s.get(db.Observation, ev.observation_id)
                    obs.simulated = True
                    obs.meta = {**(obs.meta or {}), "simulated": True, "simulation_note": note, "source_image": a["source"]}
                    asset = s.get(db.Asset, ev.asset_id)
                    asset.simulated = True
        report = mission.end(sid)
        store.update_session(sid, ended_ts=(mission_ts + timedelta(hours=1)).isoformat())
        logger.info("mission %d/%d (%s): %s", m + 1, n_missions, sid, report["counts"])

    summary = store.summary(site_id)
    logger.info("done — events %s; site summary %s", dict(stats), summary)
    print(f"Simulated history built in site '{site_id}' (all sessions/assets flagged simulated): {dict(stats)} | {summary}")


if __name__ == "__main__":
    main()
