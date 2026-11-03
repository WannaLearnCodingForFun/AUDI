"""Simulated site: assets, appearances, damage schedules (all deterministic from a seed)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ml.damage_synth import DamageSpec, render_damage
from ml.geo_projection import offset_to_latlon

KINDS_BY_TYPE = {"solar_panel": ["crack", "soiling"], "building": ["crack"], "utility_pole": ["rust", "crack"], "machinery": ["rust"]}
LABEL = {"crack": "crack", "rust": "corrosion", "soiling": "soiling"}
SIZE = {"solar_panel": (200, 130), "building": (220, 150), "utility_pole": (90, 200), "machinery": (160, 140)}


def _texture(asset_type: str, seed: int, shared_seed: Optional[int] = None) -> Image.Image:
    """Appearance of one asset. Solar panels share `shared_seed` (they look alike) plus a little
    individual variation, which is exactly what makes appearance-only re-ID ambiguous."""
    w, h = SIZE[asset_type]
    r = np.random.default_rng(shared_seed if shared_seed is not None else seed)
    if asset_type == "solar_panel":
        arr = np.full((h, w, 3), (40, 60, 120), np.float32) + r.normal(0, 6, (h, w, 3))
        for x in range(0, w, 25):
            arr[:, x:x + 1] = (190, 190, 200)
        for y in range(0, h, 26):
            arr[y:y + 1, :] = (190, 190, 200)
        own = np.random.default_rng(seed)
        for _ in range(3):   # small individual marks (labels, dirt specks)
            x, y = int(own.integers(5, w - 10)), int(own.integers(5, h - 10))
            arr[y:y + 4, x:x + 4] = own.integers(60, 230, 3)
    else:
        base = {"building": (150, 148, 140), "utility_pole": (110, 85, 60), "machinery": (120, 128, 135)}[asset_type]
        arr = np.full((h, w, 3), base, np.float32)
        noise = cv2.resize(r.normal(0, 25, (h // 6 + 1, w // 6 + 1)).astype(np.float32), (w, h))
        arr += noise[..., None] + r.normal(0, 5, (h, w, 3))
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    rr = random.Random(seed)
    n = 4 if asset_type == "solar_panel" else 12
    for _ in range(n):    # textured details so SIFT has something to verify
        x, y = rr.randint(0, w - 12), rr.randint(0, h - 12)
        d.rectangle([x, y, x + rr.randint(6, 20), y + rr.randint(6, 20)],
                    outline=tuple(rr.randint(0, 255) for _ in range(3)), width=2)
    return img


@dataclass
class Damage:
    spec: DamageSpec
    start_mission: int
    law: str              # power | linear
    rate: float           # extent growth per mission (linear) or k (power)
    exponent: float = 1.5
    e0: float = 0.12
    repaired_at: Optional[int] = None

    def extent(self, mission: int) -> float:
        if mission < self.start_mission or (self.repaired_at is not None and mission >= self.repaired_at):
            return 0.0
        t = mission - self.start_mission
        if self.law == "power":
            return float(min(1.0, self.e0 * (1 + self.rate * t) ** self.exponent))
        return float(min(1.0, self.e0 + self.rate * t))


@dataclass
class SimAsset:
    asset_id: str
    asset_type: str
    east: float
    north: float
    latitude: float
    longitude: float
    texture: Image.Image
    damages: List[Damage] = field(default_factory=list)

    def appearance(self, mission: int) -> Tuple[Image.Image, Dict[str, float], np.ndarray]:
        img = self.texture
        extents = {}
        mask = np.zeros((img.height, img.width), np.uint8)
        for d in self.damages:
            e = d.extent(mission)
            extents[LABEL[d.spec.kind]] = max(extents.get(LABEL[d.spec.kind], 0.0), e)
            if e > 0:
                img, m = render_damage(img, d.spec, e)
                mask |= m
        return img, extents, mask


@dataclass
class SimWorld:
    origin: Tuple[float, float]
    assets: List[SimAsset]
    days_between_missions: float

    def by_id(self) -> Dict[str, SimAsset]:
        return {a.asset_id: a for a in self.assets}


def build_world(seed: int = 7, rows: int = 2, panels_per_row: int = 6, poles: int = 4, walls: int = 2, machines: int = 2,
                damaged_fraction: float = 0.4, missions: int = 5, spacing_m: float = 4.0, days_between_missions: float = 14.0,
                origin: Tuple[float, float] = (47.3769, 8.5417)) -> SimWorld:
    rng = random.Random(seed)
    assets: List[SimAsset] = []

    def add(asset_type: str, e: float, n: float, shared: Optional[int] = None) -> None:
        idx = len(assets)
        lat, lon = offset_to_latlon(origin[0], origin[1], e, n)
        assets.append(SimAsset(f"gt_{asset_type[:4]}_{idx:03d}", asset_type, e, n, lat, lon,
                               _texture(asset_type, seed * 1000 + idx, shared)))

    for r in range(rows):                               # solar farm: rows of near-identical panels
        for p in range(panels_per_row):
            add("solar_panel", p * spacing_m, r * spacing_m * 1.5, shared=seed)
    for i in range(poles):                              # a line of poles to the east
        add("utility_pole", 40.0 + i * 30.0, 0.0)
    for i in range(walls):                              # a building wall to the north
        add("building", i * 12.0, 40.0)
    for i in range(machines):
        add("machinery", -20.0, i * 8.0)

    for a in assets:
        if rng.random() >= damaged_fraction:
            continue
        kind = rng.choice(KINDS_BY_TYPE[a.asset_type])
        spec = DamageSpec(kind, rng.randrange(1 << 30), (rng.uniform(0.25, 0.65), rng.uniform(0.25, 0.65)))
        start = rng.randint(1, max(1, missions - 3))
        if kind == "crack":
            a.damages.append(Damage(spec, start, "power", rate=rng.uniform(0.2, 0.4), exponent=rng.uniform(1.2, 2.0), e0=0.15))
        else:
            a.damages.append(Damage(spec, start, "linear", rate=rng.uniform(0.06, 0.12), e0=0.12))
    return SimWorld(origin, assets, days_between_missions)
