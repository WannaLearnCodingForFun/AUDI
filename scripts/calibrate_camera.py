#!/usr/bin/env python3
"""Calibrate the inspection camera (upgrade 3.6) → camera_model.calibration_file.

  python scripts/calibrate_camera.py --images calib/*.jpg --cols 9 --rows 6 --square-mm 25
  python scripts/calibrate_camera.py --camera 0 --cols 9 --rows 6 --square-mm 25 --views 20   # live capture

Verification against a tape measure:
  python scripts/calibrate_camera.py --check photo.jpg --p1 120,340 --p2 880,352 --distance-m 2.0 --truth-m 1.20
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from ml.calibration import calibrate, find_corners, measure_length  # noqa: E402
from ml.camera_model import load_intrinsics  # noqa: E402
from ml.config import cfg, load_config, resolve_path  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="*", default=[])
    ap.add_argument("--camera", default=None)
    ap.add_argument("--views", type=int, default=20)
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--square-mm", type=float, default=25.0)
    ap.add_argument("--check", default=None)
    ap.add_argument("--p1")
    ap.add_argument("--p2")
    ap.add_argument("--distance-m", type=float)
    ap.add_argument("--truth-m", type=float)
    args = ap.parse_args()
    config = load_config()
    if args.check:
        im = cv2.imread(args.check)
        intr = load_intrinsics(config, im.shape[1], im.shape[0])
        p1 = tuple(float(v) for v in args.p1.split(","))
        p2 = tuple(float(v) for v in args.p2.split(","))
        m = measure_length(p1, p2, intr, args.distance_m)
        err = None if args.truth_m is None else 100.0 * (m - args.truth_m) / args.truth_m
        print(json.dumps({"measured_m": m, "truth_m": args.truth_m, "error_pct": err, "intrinsics": intr.source}, indent=2))
        return 0
    pattern = (args.cols, args.rows)
    grays = []
    if args.camera is not None:
        cap = cv2.VideoCapture(int(args.camera) if str(args.camera).isdigit() else args.camera)
        while len(grays) < args.views:
            ok, f = cap.read()
            if not ok:
                break
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            if find_corners(g, pattern) is not None:
                grays.append(g)
                print(f"view {len(grays)}/{args.views}")
                time.sleep(0.7)
        cap.release()
    for pat in args.images:
        for p in sorted(glob.glob(pat)):
            grays.append(cv2.imread(p, cv2.IMREAD_GRAYSCALE))
    intr, stats = calibrate(grays, pattern, args.square_mm / 1000.0)
    out = resolve_path(cfg(config, "camera_model", "calibration_file"))
    intr.save(out)
    print(json.dumps({"file": str(out), **stats, "hfov_deg": intr.hfov_deg, "fx": intr.fx, "fy": intr.fy}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
