#!/usr/bin/env python3
"""Find the Philips (or any UVC) webcam: probe camera indices and save a snapshot of each.

For every index 0…camera.probe_max_index that opens with the OS backend
(AVFoundation on macOS, DirectShow/MSMF on Windows, V4L2 on Linux) this prints
resolution and measured FPS and writes reports/camera_probe/cam_<i>.jpg so you
can see which index is the external camera. Then set `camera.source` in
config.yaml (index or a unique part of the device name).

Usage: python scripts/camera_probe.py [--max-index 9] [--seconds 2]
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from ml.config import cfg, load_config, resolve_path  # noqa: E402
from ml.live.devices import BACKENDS, default_backend, list_device_names, raw_device_listing  # noqa: E402


def probe(index: int, backend: int, seconds: float, width: int, height: int, out_dir: Path):
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    frames, last, t0 = 0, None, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        ok, frame = cap.read()
        if ok and frame is not None:
            frames += 1
            last = frame
    elapsed = time.perf_counter() - t0
    reported_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if last is None:
        return {"index": index, "opened": True, "frames": 0}
    out_dir.mkdir(parents=True, exist_ok=True)
    snap = out_dir / f"cam_{index}.jpg"
    cv2.imwrite(str(snap), last)
    return {
        "index": index, "opened": True, "frames": frames,
        "resolution": f"{last.shape[1]}x{last.shape[0]}",
        "measured_fps": frames / elapsed if elapsed > 0 else 0.0,
        "reported_fps": reported_fps, "snapshot": snap,
    }


def main() -> None:
    config = load_config()
    c = cfg(config, "camera")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-index", type=int, default=int(c["probe_max_index"]))
    parser.add_argument("--seconds", type=float, default=float(c["probe_seconds"]))
    parser.add_argument("--backend", default=c["backend"], choices=["auto", *BACKENDS])
    args = parser.parse_args()

    system = platform.system()
    backend = default_backend() if args.backend == "auto" else BACKENDS[args.backend]
    out_dir = resolve_path(cfg(config, "reports", "dir")) / "camera_probe"
    print(f"OS: {system} | OpenCV {cv2.__version__} | backend id {backend} | probing indices 0–{args.max_index}")
    if system == "Darwin":
        print("Note: macOS requires Camera permission for the app running this script "
              "(System Settings → Privacy & Security → Camera → enable Terminal / Cursor / iTerm).")

    listing = raw_device_listing()
    if listing:
        print("\nOS device listing:\n" + listing)
    names = list_device_names()

    found = []
    for i in range(args.max_index + 1):
        r = probe(i, backend, args.seconds, int(c["width"]), int(c["height"]), out_dir)
        if r is None:
            continue
        found.append(r)
        name = names.get(i, "")
        if r["frames"] == 0:
            print(f"[{i}] opened but delivered no frames {name} (permission denied or device busy?)")
        else:
            print(f"[{i}] {name or '(name unavailable)'}: {r['resolution']} | measured {r['measured_fps']:.1f} fps "
                  f"(driver reports {r['reported_fps']:.0f}) | snapshot {r['snapshot'].relative_to(ROOT)}")

    if not found:
        print("\nNo camera opened. Plug in the webcam, grant camera permission, and re-run.")
        return
    print(f"\nCurrent config camera.source = {c['source']!r}. Set it to the index (or a unique part of the name) "
          "of the Philips camera, e.g.  camera:\\n  source: 1")


if __name__ == "__main__":
    main()
