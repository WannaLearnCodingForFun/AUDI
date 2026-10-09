"""Synthetic defect overlays for simulated histories and tests (always labelled simulated)."""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def draw_crack(img: Image.Image, rng: random.Random, extent: float, width_px: int = 3) -> Image.Image:
    """Dark random-walk crack covering roughly `extent` (0–1) of the image diagonal."""
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    w, h = out.size
    x, y = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
    angle = rng.uniform(0, 2 * np.pi)
    steps = max(4, int(extent * 24))
    step = np.hypot(w, h) * extent / steps
    pts = [(x, y)]
    for _ in range(steps):
        angle += rng.uniform(-0.6, 0.6)
        x = float(np.clip(x + step * np.cos(angle), 0, w - 1))
        y = float(np.clip(y + step * np.sin(angle), 0, h - 1))
        pts.append((x, y))
    d.line(pts, fill=(25, 22, 20), width=max(1, width_px))
    return out


def add_soiling(img: Image.Image, rng: random.Random, coverage: float) -> Image.Image:
    """Brownish translucent blotches covering about `coverage` of the image."""
    base = img.convert("RGB")
    w, h = base.size
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    n = max(1, int(coverage * 30))
    r_max = max(4, int(min(w, h) * np.sqrt(coverage / n) * 0.9))
    for _ in range(n):
        cx, cy, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(r_max * 0.5, r_max)
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=int(rng.uniform(120, 200)))
    mask = mask.filter(ImageFilter.GaussianBlur(max(1, r_max // 3)))
    dirt = Image.new("RGB", (w, h), (120, 96, 60))
    return Image.composite(dirt, base, mask)


def apply_defect(img: Image.Image, kind: str, rng: random.Random, strength: float) -> Tuple[Image.Image, str]:
    if kind == "soiling":
        return add_soiling(img, rng, strength), "synthetic soiling"
    return draw_crack(img, rng, strength), "synthetic crack"
