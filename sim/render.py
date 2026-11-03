"""Fly simulated missions: frames, drone poses and ground truth in the mission-folder format.

missions/<k>/frames/<i>.jpg
missions/<k>/frames.jsonl    one line per frame:
  {"frame": "frames/0003.jpg", "ts": ISO, "pose": {...CameraPose...},
   "gt": [{"asset_id", "asset_type", "bbox": [x1,y1,x2,y2], "damage": {label: extent}, "damage_area_px": n}]}

The Gazebo recorder (sim/gazebo/record_mission.py) writes the same format, so sim.runner and
sim.metrics evaluate either source.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance

from ml.camera_model import CameraIntrinsics
from ml.geo_projection import offset_to_latlon
from sim.world import SimAsset, SimWorld

FRAME = (480, 360)
T0 = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)


@dataclass
class SimFrame:
    image: Image.Image
    ts: str
    pose: Dict[str, Any]
    gt: List[Dict[str, Any]]


def background(seed: int) -> Image.Image:
    r = np.random.default_rng(seed)
    w, h = FRAME
    g = r.normal(0, 18, (h // 8 + 1, w // 8 + 1)).astype(np.float32)
    import cv2

    g = cv2.resize(g, (w, h))
    arr = np.stack([95 + g, 110 + g * 0.8, 70 + g * 0.5], -1) + r.normal(0, 4, (h, w, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def render_view(asset: SimAsset, mission: int, world: SimWorld, rng: random.Random, altitude_m: float,
                gps_sigma_m: float, intr: CameraIntrinsics, frame_seed: int) -> SimFrame:
    img, extents, mask = asset.appearance(mission)
    # re-photographed: lighting, scale, small rotation (identity unchanged)
    scale = rng.uniform(0.85, 1.1)
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.8, 1.2))
    img = img.resize((int(img.width * scale), int(img.height * scale)), Image.BILINEAR)
    mask_img = Image.fromarray(mask * 255).resize(img.size, Image.NEAREST)
    ang = rng.uniform(-4, 4)
    img = img.rotate(ang, resample=Image.BILINEAR, expand=False)
    mask_img = mask_img.rotate(ang, resample=Image.NEAREST, expand=False)
    frame = background(frame_seed)
    w, h = FRAME
    x0 = int(rng.uniform(0.25, 0.75) * w - img.width / 2)
    y0 = int(rng.uniform(0.3, 0.7) * h - img.height / 2)
    x0 = max(0, min(w - img.width, x0))
    y0 = max(0, min(h - img.height, y0))
    frame.paste(img, (x0, y0))
    bbox = [x0, y0, x0 + img.width, y0 + img.height]
    # camera geometry: nadir over flat assets, oblique side view of upright ones; the drone position
    # is solved so that the bbox anchor pixel's ground ray lands on the asset's true position
    from ml.geo_projection import CameraPose, pixel_ray

    flat = asset.asset_type == "solar_panel"
    pitch = 90.0 if flat else rng.uniform(20.0, 35.0)
    alt = altitude_m if flat else rng.uniform(3.0, 5.0)
    if flat:
        u, v = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
    else:
        u, v = (bbox[0] + bbox[2]) / 2.0, float(bbox[3])
    ray = pixel_ray(u, v, intr, CameraPose(0.0, 0.0, alt, 0.0, pitch, 0.0))
    t = alt / max(1e-6, -ray[2])
    e = asset.east - t * ray[0] + rng.gauss(0, gps_sigma_m)
    n = asset.north - t * ray[1] + rng.gauss(0, gps_sigma_m)
    lat, lon = offset_to_latlon(world.origin[0], world.origin[1], e, n)
    ts = (T0 + timedelta(days=world.days_between_missions * mission, seconds=frame_seed % 3600)).isoformat()
    pose = {"latitude": lat, "longitude": lon, "altitude_agl": alt, "heading_deg": 0.0, "pitch_deg": pitch,
            "roll_deg": 0.0, "gps_accuracy_m": max(0.3, gps_sigma_m)}
    gt = [{"asset_id": asset.asset_id, "asset_type": asset.asset_type, "bbox": bbox, "damage": extents,
           "damage_area_px": int((np.asarray(mask_img) > 0).sum())}]
    return SimFrame(frame, ts, pose, gt)


def fly_mission(world: SimWorld, mission: int, seed: int = 0, coverage: float = 1.0, altitude_m: float = 12.0,
                gps_sigma_m: float = 0.8) -> List[SimFrame]:
    rng = random.Random(seed * 7919 + mission)
    intr = CameraIntrinsics.from_hfov(FRAME[0], FRAME[1], 78.0)
    frames = []
    for i, a in enumerate(world.assets):
        if mission > 0 and rng.random() > coverage:
            continue        # missed this flight (coverage < 1)
        frames.append(render_view(a, mission, world, rng, altitude_m, gps_sigma_m, intr, seed * 100_000 + mission * 1000 + i))
    return frames


def write_mission(frames: List[SimFrame], out: Path) -> Path:
    (out / "frames").mkdir(parents=True, exist_ok=True)
    with (out / "frames.jsonl").open("w") as f:
        for i, fr in enumerate(frames):
            rel = f"frames/{i:04d}.jpg"
            fr.image.save(out / rel, quality=92)
            f.write(json.dumps({"frame": rel, "ts": fr.ts, "pose": fr.pose, "gt": fr.gt}) + "\n")
    return out


def read_mission(folder: Path) -> Iterator[SimFrame]:
    for line in (folder / "frames.jsonl").read_text().splitlines():
        d = json.loads(line)
        with Image.open(folder / d["frame"]) as im:
            yield SimFrame(im.convert("RGB"), d["ts"], d["pose"], d["gt"])
