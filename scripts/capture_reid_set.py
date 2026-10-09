#!/usr/bin/env python3
"""Record a real re-identification set with the webcam (replaces simulated evidence, D5).

Photograph the same physical objects in several sessions (different days,
angles, lighting), optionally with a printed ArUco marker and optionally after
adding a simulated defect (tape or marker "crack", dust):

  python scripts/capture_reid_set.py --object mug_01 --session day1
  python scripts/capture_reid_set.py --object mug_01 --session day2 --condition damaged
  python scripts/capture_reid_set.py --object box_02 --session day2 --auto 5   # headless: 5 frames, no window

Window keys: SPACE = save frame, q / ESC = quit. Images go to
`reid.dir`/<session>/<object>/ and one row per image is appended to
`reid.dir`/manifest.csv (session, object_id, condition, path, ts, marker_id).
Evaluate with: python training/evaluate_reid.py
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
from PIL import Image  # noqa: E402

from ml.asset_identity import ArucoReader  # noqa: E402
from ml.config import cfg, load_config, resolve_path  # noqa: E402
from ml.live.camera import CameraReader  # noqa: E402
from ml.live.devices import resolve_source  # noqa: E402
from ml.schemas import utc_now_iso  # noqa: E402

FIELDS = ["session", "object_id", "condition", "path", "ts", "marker_id"]
SAFE = re.compile(r"^[A-Za-z0-9_.-]+$")


def append_rows(manifest: Path, rows) -> None:
    new = not manifest.exists()
    with manifest.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--object", required=True, help="object id, e.g. mug_01")
    parser.add_argument("--session", required=True, help="session label, e.g. day1")
    parser.add_argument("--condition", default="normal", choices=["normal", "damaged"])
    parser.add_argument("--source", default=None, help="camera index / name / URL (default: camera.source)")
    parser.add_argument("--auto", type=int, default=0, help="capture N frames without a window")
    args = parser.parse_args()
    for v in (args.object, args.session):
        if not SAFE.match(v):
            sys.exit(f"Use letters, digits, '_', '-', '.' only: {v!r}")

    config = load_config()
    r = cfg(config, "reid")
    root = resolve_path(r["dir"])
    out = root / args.session / args.object
    out.mkdir(parents=True, exist_ok=True)
    acfg = cfg(config, "aruco")
    aruco = ArucoReader(acfg["dictionary"], float(acfg["bbox_margin"]))

    source = resolve_source(args.source) if args.source is not None else cfg(config, "camera", "source")
    cam = CameraReader.from_config(config, source)
    cam.start()
    if cam.wait_frame(0, float(cfg(config, "camera", "reconnect_seconds")) * 3) is None:
        cam.stop()
        sys.exit("No frames from the camera. Run scripts/camera_probe.py; on macOS grant Camera access to your terminal/IDE.")

    rows = []

    def save(frame) -> None:
        n = len(list(out.glob(f"{args.condition}_*.jpg")))
        path = out / f"{args.condition}_{n:03d}.jpg"
        cv2.imwrite(str(path), frame.image)
        markers = aruco.read(Image.fromarray(cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)), None)
        rows.append({"session": args.session, "object_id": args.object, "condition": args.condition,
                     "path": str(path.relative_to(root)), "ts": utc_now_iso(), "marker_id": markers[0] if markers else ""})
        print(f"saved {path}" + (f" (ArUco {markers[0]})" if markers else ""))

    try:
        if args.auto:
            seq = 0
            for _ in range(args.auto):
                f = cam.wait_frame(seq, 5.0)
                if f is None:
                    break
                seq = f.seq
                save(f)
                time.sleep(float(r["capture_interval_s"]))
        else:
            title = f"AUDI re-ID capture: {args.session}/{args.object} ({args.condition}) — SPACE save, q quit"
            seq = 0
            while True:
                f = cam.wait_frame(seq, 1.0)
                if f is not None:
                    seq = f.seq
                    cv2.imshow(title, f.image)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" ") and f is not None:
                    save(f)
            cv2.destroyAllWindows()
    finally:
        cam.stop()
    if rows:
        append_rows(root / "manifest.csv", rows)
    print(f"{len(rows)} image(s) recorded for {args.object} in {args.session}")


if __name__ == "__main__":
    main()
