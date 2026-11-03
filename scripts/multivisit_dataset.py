#!/usr/bin/env python3
"""Real multi-visit dataset tooling (upgrade 7.1) — see docs/DATASET_PROTOCOL.md.

  # 1. package each visit (photos with GPS EXIF, or a flight-log CSV) into the mission-folder format
  python scripts/multivisit_dataset.py package --visit visits/2026-11-03 --out release/v1/visit_01 \
         [--log visits/2026-11-03/flight.csv] [--altitude-agl 12 --pitch 90 --heading 0]
  # 2. propose labels with the current engine (asset ids across visits, defect boxes)
  python scripts/multivisit_dataset.py prelabel --release release/v1 --site campus_roof
  # 3. correct labels/<visit>.csv by hand (asset_id, asset_type, bbox, defect, change), then import
  python scripts/multivisit_dataset.py import --release release/v1
  # 4. evaluate AUDI on it exactly like a simulated run
  python scripts/run_sim_eval.py --missions-dir release/v1

Flight-log CSV columns: filename, latitude, longitude, altitude_agl, heading_deg, pitch_deg[, roll_deg, gps_accuracy_m]
(PX4 / DJI logs export to this with their usual tools).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import ExifTags, Image  # noqa: E402

LABEL_COLS = ["frame", "asset_id", "asset_type", "x1", "y1", "x2", "y2", "defect", "defect_extent", "changed_since_last_visit", "notes"]


def _dms(v) -> float:
    d, m, s = (float(x) for x in v)
    return d + m / 60.0 + s / 3600.0


def exif_pose(path: Path) -> Dict[str, Any]:
    with Image.open(path) as im:
        exif = im.getexif()
        ts = exif.get(36867) or exif.get(306)
        gps = exif.get_ifd(ExifTags.IFD.GPSInfo) if hasattr(ExifTags, "IFD") else {}
    out: Dict[str, Any] = {}
    if gps and 2 in gps and 4 in gps:
        lat, lon = _dms(gps[2]), _dms(gps[4])
        out["latitude"] = -lat if gps.get(1) == "S" else lat
        out["longitude"] = -lon if gps.get(3) == "W" else lon
        if 6 in gps:
            out["altitude_msl"] = float(gps[6])
        if 17 in gps:
            out["heading_deg"] = float(gps[17])
    if ts:
        out["ts"] = datetime.strptime(str(ts), "%Y:%m:%d %H:%M:%S").replace(tzinfo=timezone.utc).isoformat()
    return out


def cmd_package(a) -> int:
    visit, out = Path(a.visit), Path(a.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    log: Dict[str, Dict[str, str]] = {}
    if a.log:
        with open(a.log, newline="") as f:
            log = {r["filename"]: r for r in csv.DictReader(f)}
    n = 0
    with (out / "frames.jsonl").open("w") as fj:
        for p in sorted(visit.iterdir()):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            ex = exif_pose(p)
            row = log.get(p.name, {})
            pose = {
                "latitude": float(row.get("latitude") or ex.get("latitude") or "nan"),
                "longitude": float(row.get("longitude") or ex.get("longitude") or "nan"),
                "altitude_agl": float(row.get("altitude_agl") or a.altitude_agl),
                "heading_deg": float(row.get("heading_deg") or ex.get("heading_deg") or a.heading),
                "pitch_deg": float(row.get("pitch_deg") or a.pitch),
                "roll_deg": float(row.get("roll_deg") or 0.0),
            }
            if row.get("gps_accuracy_m"):
                pose["gps_accuracy_m"] = float(row["gps_accuracy_m"])
            rel = f"frames/{n:05d}{p.suffix.lower()}"
            shutil.copy(p, out / rel)
            fj.write(json.dumps({"frame": rel, "source": p.name, "ts": ex.get("ts") or datetime.now(timezone.utc).isoformat(),
                                 "pose": pose, "gt": []}) + "\n")
            n += 1
    print(f"{n} frames → {out}")
    return 0


def _visits(release: Path) -> List[Path]:
    return [p for p in sorted(release.iterdir()) if (p / "frames.jsonl").exists()]


def cmd_prelabel(a) -> int:
    from api.services import build_services
    from ml.config import load_config
    from sim.render import read_mission

    release = Path(a.release)
    (release / "labels").mkdir(exist_ok=True)
    svc = build_services(load_config(), use_cloud=False)
    svc.store.ensure_site(a.site, a.site)
    for v in _visits(release):
        sid = svc.store.create_session(a.site, "prelabel", "offline", {})["session_id"]
        rows = []
        lines = (v / "frames.jsonl").read_text().splitlines()
        for line, fr in zip(lines, read_mission(v)):
            d = json.loads(line)
            pose = d["pose"] if d["pose"]["latitude"] == d["pose"]["latitude"] else None   # NaN check
            for ev in svc.engine.process_frame(fr.image, sid, pose=pose, timestamp=fr.ts):
                if ev.type in ("error", "rejected"):
                    continue
                x1, y1, x2, y2 = ev.bbox
                rows.append({"frame": d["frame"], "asset_id": ev.asset_id, "asset_type": ev.asset_type, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                             "defect": ev.condition.get("defect_type") or "", "defect_extent": "", "changed_since_last_visit": int(ev.type == "change"),
                             "notes": "prelabel"})
        svc.mission.end(sid)
        with (release / "labels" / f"{v.name}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, LABEL_COLS)
            w.writeheader()
            w.writerows(rows)
        print(f"{v.name}: {len(rows)} proposed labels → labels/{v.name}.csv (review asset_id consistency across visits)")
    return 0


def cmd_import(a) -> int:
    release = Path(a.release)
    for v in _visits(release):
        lab = release / "labels" / f"{v.name}.csv"
        if not lab.exists():
            print(f"{v.name}: no labels, skipped")
            continue
        by_frame: Dict[str, List[Dict[str, Any]]] = {}
        with lab.open(newline="") as f:
            for r in csv.DictReader(f):
                ext = float(r["defect_extent"]) if r.get("defect_extent") else (1.0 if r.get("defect") else 0.0)
                by_frame.setdefault(r["frame"], []).append({
                    "asset_id": r["asset_id"], "asset_type": r["asset_type"],
                    "bbox": [int(float(r[k])) for k in ("x1", "y1", "x2", "y2")],
                    "damage": {r["defect"]: ext} if r.get("defect") else {}, "damage_area_px": 0,
                    "changed": bool(int(r.get("changed_since_last_visit") or 0)),
                })
        lines = []
        for line in (v / "frames.jsonl").read_text().splitlines():
            d = json.loads(line)
            d["gt"] = by_frame.get(d["frame"], [])
            lines.append(json.dumps(d))
        (v / "frames.jsonl").write_text("\n".join(lines) + "\n")
        print(f"{v.name}: {sum(len(x) for x in by_frame.values())} labels imported")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("package")
    p.add_argument("--visit", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--log", default=None)
    p.add_argument("--altitude-agl", type=float, default=12.0)
    p.add_argument("--pitch", type=float, default=90.0)
    p.add_argument("--heading", type=float, default=0.0)
    q = sub.add_parser("prelabel")
    q.add_argument("--release", required=True)
    q.add_argument("--site", required=True)
    r = sub.add_parser("import")
    r.add_argument("--release", required=True)
    a = ap.parse_args()
    return {"package": cmd_package, "prelabel": cmd_prelabel, "import": cmd_import}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
