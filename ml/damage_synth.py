"""Deterministic, growable synthetic damage with ground-truth masks.

Used by the Siamese change-detector training pairs (2.1), the defect-tracking growth
sequences (2.2/2.3), the simulation testbed (4.3) and copy-paste augmentation for weak classes
(3.5). A `DamageSpec` fixes the defect's seed and anchor, so rendering it at a larger
`extent` grows the SAME defect (the crack path is extended, the rust patch dilates) — exactly
what a longitudinal benchmark needs.

Every render returns (image, mask) with the mask in image pixels.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


@dataclass(frozen=True)
class DamageSpec:
    kind: str            # crack | rust | soiling
    seed: int
    anchor: Tuple[float, float] = (0.5, 0.5)   # normalised start / centre

    def path(self, size: Tuple[int, int], max_steps: int = 64) -> List[Tuple[float, float]]:
        """Full random-walk crack path; a render at extent e uses its first e·max_steps steps."""
        rng = random.Random(self.seed)
        w, h = size
        x, y = self.anchor[0] * w, self.anchor[1] * h
        angle = rng.uniform(0, 2 * math.pi)
        step = math.hypot(w, h) / max_steps
        pts = [(x, y)]
        for _ in range(max_steps):
            angle += rng.uniform(-0.5, 0.5)
            x = float(np.clip(x + step * math.cos(angle), 0, w - 1))
            y = float(np.clip(y + step * math.sin(angle), 0, h - 1))
            pts.append((x, y))
        return pts


def render_damage(img: Image.Image, spec: DamageSpec, extent: float, width_px: int = 3) -> Tuple[Image.Image, np.ndarray]:
    """Apply `spec` at `extent` ∈ [0, 1]. Returns (damaged RGB image, uint8 mask)."""
    base = img.convert("RGB").copy()
    w, h = base.size
    mask = Image.new("L", (w, h), 0)
    extent = float(np.clip(extent, 0.0, 1.0))
    if extent <= 0:
        return base, np.zeros((h, w), np.uint8)
    if spec.kind == "crack":
        pts = spec.path((w, h))
        n = max(2, int(round(extent * (len(pts) - 1))) + 1)
        seg = pts[:n]
        ImageDraw.Draw(base).line(seg, fill=(22, 20, 18), width=max(1, width_px))
        ImageDraw.Draw(mask).line(seg, fill=255, width=max(1, width_px))
    elif spec.kind == "rust":
        rng = random.Random(spec.seed)
        cx, cy = spec.anchor[0] * w, spec.anchor[1] * h
        r_max = 0.35 * min(w, h)
        d = ImageDraw.Draw(mask)
        for i in range(12):
            ang = rng.uniform(0, 2 * math.pi)
            off = rng.uniform(0, 0.5) * r_max * extent
            r = r_max * extent * rng.uniform(0.35, 0.7)
            x, y = cx + off * math.cos(ang), cy + off * math.sin(ang)
            d.ellipse((x - r, y - r, x + r, y + r), fill=255)
        m = np.asarray(mask) > 0
        arr = np.asarray(base).astype(np.float32)
        tex = np.random.default_rng(spec.seed).normal(0, 12, arr.shape)
        rust = np.array([150, 70, 25], np.float32) + tex
        arr[m] = 0.2 * arr[m] + 0.8 * rust[m]
        base = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    elif spec.kind == "soiling":
        rng = random.Random(spec.seed)
        d = ImageDraw.Draw(mask)
        n = 6
        r_max = max(4, int(min(w, h) * math.sqrt(max(extent, 1e-3) / n) * 0.9))
        for _ in range(n):
            x, y = rng.uniform(0, w), rng.uniform(0, h)
            r = rng.uniform(0.6, 1.0) * r_max
            d.ellipse((x - r, y - r, x + r, y + r), fill=255)
        soft = mask.filter(ImageFilter.GaussianBlur(max(1, r_max // 4)))
        dirt = Image.new("RGB", (w, h), (205, 195, 175))
        base = Image.composite(dirt, base, soft.point(lambda v: int(v * 0.75)))
    else:
        raise ValueError(f"Unknown damage kind: {spec.kind}")
    return base, (np.asarray(mask) > 127).astype(np.uint8)


def change_pair(
    img: Image.Image, spec: DamageSpec, before: float, after: float, jitter: Optional[random.Random] = None,
) -> Tuple[Image.Image, Image.Image, np.ndarray]:
    """(before image, after image, change mask = after-mask minus before-mask) with optional
    photometric jitter on the after view (lighting changes are not change)."""
    b, mb = render_damage(img, spec, before)
    a, ma = render_damage(img, spec, after)
    if jitter is not None:
        arr = np.asarray(a).astype(np.float32)
        gain = jitter.uniform(0.8, 1.2)
        bias = jitter.uniform(-15, 15)
        a = Image.fromarray(np.clip(arr * gain + bias, 0, 255).astype(np.uint8))
    return b, a, ((ma > 0) & ~(mb > 0)).astype(np.uint8)


def copy_paste(clean: Image.Image, donor: Image.Image, donor_mask: np.ndarray, rng: random.Random,
               scale_range: Sequence[float] = (0.5, 1.0)) -> Tuple[Image.Image, np.ndarray]:
    """Paste a masked defect from `donor` onto `clean` with random scale / rotation / position
    and Poisson-free alpha feathering (copy-paste augmentation, 3.5)."""
    clean = clean.convert("RGB")
    W, H = clean.size
    ys, xs = np.nonzero(donor_mask)
    if len(xs) == 0:
        return clean, np.zeros((H, W), np.uint8)
    x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    patch = np.asarray(donor.convert("RGB"))[y1:y2, x1:x2]
    pm = donor_mask[y1:y2, x1:x2].astype(np.float32)
    s = rng.uniform(*scale_range) * min(W / max(1, patch.shape[1]), H / max(1, patch.shape[0]), 1.0)
    ang = rng.uniform(0, 360)
    ph, pw = patch.shape[:2]
    M = cv2.getRotationMatrix2D((pw / 2, ph / 2), ang, s)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(ph * sin + pw * cos) + 1, int(ph * cos + pw * sin) + 1
    M[0, 2] += nw / 2 - pw / 2
    M[1, 2] += nh / 2 - ph / 2
    patch = cv2.warpAffine(patch, M, (nw, nh))
    pm = cv2.warpAffine(pm, M, (nw, nh))
    nw, nh = min(nw, W), min(nh, H)
    patch, pm = patch[:nh, :nw], pm[:nh, :nw]
    ox, oy = rng.randint(0, W - nw), rng.randint(0, H - nh)
    out = np.asarray(clean).astype(np.float32).copy()
    alpha = cv2.GaussianBlur(pm, (3, 3), 0)[..., None]
    region = out[oy:oy + nh, ox:ox + nw]
    out[oy:oy + nh, ox:ox + nw] = region * (1 - alpha) + patch.astype(np.float32) * alpha
    mask = np.zeros((H, W), np.uint8)
    mask[oy:oy + nh, ox:ox + nw] = (pm > 0.5).astype(np.uint8)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), mask
