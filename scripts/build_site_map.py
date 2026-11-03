#!/usr/bin/env python3
"""Build a site map (upgrade 1.2): orthomosaic with OpenDroneMap, then asset footprints.

  # 1. stitch a survey flight (Docker; images with GPS EXIF)
  python scripts/build_site_map.py odm --images flights/survey_2026-11-03 --project data/odm/site_demo
  # 2. register the orthophoto as the site's map
  python scripts/build_site_map.py register --site site_demo --ortho data/odm/site_demo/odm_orthophoto/odm_orthophoto.tif
  # 3. detect assets on the orthophoto and create footprints (or draw them in the admin UI)
  python scripts/build_site_map.py detect --site site_demo
  # verification: share of detections inside the correct polygon, on a recorded / simulated flight
  python scripts/build_site_map.py verify --site site_demo --missions-dir runs/gz
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.config import load_config, resolve_path  # noqa: E402


def cmd_odm(args) -> int:
    if shutil.which("docker") is None:
        print("docker not found: install Docker or run OpenDroneMap / WebODM yourself")
        return 1
    proj = Path(args.project).resolve()
    (proj / "images").mkdir(parents=True, exist_ok=True)
    for p in Path(args.images).glob("*"):
        if p.suffix.lower() in (".jpg", ".jpeg", ".tif", ".png"):
            shutil.copy(p, proj / "images" / p.name)
    cmd = ["docker", "run", "--rm", "-v", f"{proj.parent}:/datasets", "opendronemap/odm", "--project-path", "/datasets", proj.name,
           "--orthophoto-resolution", str(args.resolution_cm), "--dsm", "--mesh-size", "200000", "--texturing-skip-global-seam-leveling"]
    print(" ".join(cmd))
    return subprocess.call(cmd)


def _store():
    from memory.store import LocalStore

    return LocalStore(load_config())


def cmd_register(args) -> int:
    from PIL import Image

    from ml.site_map import read_geotransform

    Image.MAX_IMAGE_PIXELS = None
    path = resolve_path(args.ortho)
    gt = json.loads(args.geotransform) if args.geotransform else read_geotransform(path)
    if not gt:
        print("no geotransform in the file; pass --geotransform '[lon0, dlon, 0, lat0, 0, -dlat]'")
        return 1
    with Image.open(path) as im:
        w, h = im.size
    meta = {}
    mesh = path.parent.parent / "odm_texturing" / "odm_textured_model_geo.obj"
    if mesh.exists():
        meta = {"mesh_path": str(mesh), "mesh_format": "obj"}
    row = _store().add_site_map(args.site, str(path), gt, w, h, "cli", meta)
    print(json.dumps(row, indent=2, default=str))
    return 0


def cmd_detect(args) -> int:
    from ml.detector import build_detector
    from ml.severity import asset_criticality
    from ml.site_map import SiteMap, detect_footprints

    config = load_config()
    store = _store()
    row = store.active_site_map(args.site)
    if row is None:
        print("register a map first")
        return 1
    smap = SiteMap.load(row["map_id"], args.site, resolve_path(row["image_path"]), row["geotransform"])
    found = detect_footprints(smap, build_detector(config), args.tile)
    for f in found:
        store.register_mapped_asset(args.site, f["asset_type"], f["footprint"], asset_criticality(f["asset_type"], config), "cli")
    print(f"{len(found)} footprints registered")
    return 0


def cmd_verify(args) -> int:
    from sim.metrics import site_map_hit_rate
    from sim.render import read_mission

    store = _store()
    from api.services import build_services

    svc = build_services(load_config(), store=store, use_cloud=False)
    log = []
    for folder in sorted(Path(args.missions_dir).iterdir()):
        if not (folder / "frames.jsonl").exists():
            continue
        sid = store.create_session(args.site, "verify", "offline", {}, simulated=True)["session_id"]
        for fr in read_mission(folder):
            for ev, g in zip(svc.engine.process_frame(fr.image, sid, pose=fr.pose, timestamp=fr.ts), fr.gt):
                log.append({"gt": g, "event": ev.to_dict()})
        store.update_session(sid, status="ended")
    print(json.dumps({"detections": len(log), "inside_correct_polygon": site_map_hit_rate(log)}, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("odm")
    a.add_argument("--images", required=True)
    a.add_argument("--project", required=True)
    a.add_argument("--resolution-cm", type=float, default=2.0)
    b = sub.add_parser("register")
    b.add_argument("--site", required=True)
    b.add_argument("--ortho", required=True)
    b.add_argument("--geotransform", default=None)
    c = sub.add_parser("detect")
    c.add_argument("--site", required=True)
    c.add_argument("--tile", type=int, default=1024)
    d = sub.add_parser("verify")
    d.add_argument("--site", required=True)
    d.add_argument("--missions-dir", required=True)
    args = ap.parse_args()
    return {"odm": cmd_odm, "register": cmd_register, "detect": cmd_detect, "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
