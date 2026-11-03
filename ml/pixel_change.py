"""Pixel-level change between an asset's reference view and the current view (upgrade 2.1).

The current crop is warped onto the reference view with the identity homography (from the
re-ID SIFT verification or the site-map registration), photometrically normalised to the
reference inside the overlap, and compared. Two mask producers share one output format:

  SiameseChangeNet   (ml/siamese_change.py) when change_mask.checkpoint exists — a BIT-style
                     Siamese network trained on synthetic before/after pairs + LEVIR-CD.
  classical          training-free: robust z-score of a Lab colour + gradient difference,
                     morphological clean-up, minimum component size.

Every changed component gets a change type from its colour shift (darker + thin → crack,
redder / yellower → rust, brighter + desaturated → soiling, otherwise "other"), and the area
is converted to cm² with the metric scale of the current crop (ml/camera_model.py) corrected by
the homography's local area scale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from ml.config import get_nested, resolve_path
from ml.segmentation import Region, regions_from_mask

logger = logging.getLogger(__name__)


@dataclass
class ChangeMask:
    mask: np.ndarray                         # uint8 (H, W) in reference-view pixels
    valid: np.ndarray                        # overlap of the warped current view
    regions: List[Region] = field(default_factory=list)
    changed_fraction: float = 0.0            # changed / valid pixels
    changed_area_px: float = 0.0             # in reference pixels
    changed_area_cm2: Optional[float] = None
    change_type: Optional[str] = None        # dominant type by area
    backend: str = "classical"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "changed_fraction": self.changed_fraction,
            "changed_area_px": self.changed_area_px,
            "changed_area_cm2": self.changed_area_cm2,
            "change_type": self.change_type,
            "backend": self.backend,
            "regions": [r.to_dict() for r in self.regions],
        }


def homography_area_scale(H: Optional[np.ndarray], x: float, y: float) -> float:
    """Local area magnification of H at (x, y): reference px² per current px²."""
    if H is None:
        return 1.0
    eps = 1.0
    def m(px, py):
        p = H @ np.array([px, py, 1.0])
        return p[:2] / p[2]
    c, dx, dy = m(x, y), m(x + eps, y), m(x, y + eps)
    J = np.stack([dx - c, dy - c], axis=1)
    return float(abs(np.linalg.det(J))) or 1.0


def warp_to_reference(current: Image.Image, reference_size, H: Optional[np.ndarray]):
    w, h = reference_size
    src = np.asarray(current.convert("RGB"))
    if H is None:
        return cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA), np.full((h, w), 255, np.uint8)
    warped = cv2.warpPerspective(src, H, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    valid = cv2.warpPerspective(np.full(src.shape[:2], 255, np.uint8), H, (w, h), flags=cv2.INTER_NEAREST)
    valid = cv2.erode(valid, np.ones((5, 5), np.uint8))
    return warped, valid


def normalise_to(src: np.ndarray, ref: np.ndarray, valid: np.ndarray, iterations: int = 4) -> np.ndarray:
    """Per-channel gain/bias match of `src` to `ref` (global lighting), robust to large changes.

    Least squares src·g + b ≈ ref, iteratively re-fitted on the pixels whose residual is below
    2.5 robust sigmas, so a defect covering a large part of the view (soiling) is not absorbed
    into the lighting correction.
    """
    out = src.astype(np.float32).copy()
    v = valid > 0
    if v.sum() < 16:
        return out
    for c in range(src.shape[2]):
        x = out[..., c][v]
        y = ref[..., c].astype(np.float32)[v]
        keep = np.ones_like(x, dtype=bool)
        g, b = 1.0, 0.0
        for _ in range(iterations):
            xs, ys = x[keep], y[keep]
            if len(xs) < 16:
                break
            var = float(xs.var())
            g = float(((xs - xs.mean()) * (ys - ys.mean())).mean() / var) if var > 1e-6 else 1.0
            g = float(np.clip(g, 0.3, 3.0))
            b = float(ys.mean() - g * xs.mean())
            res = np.abs(g * x + b - y)
            sigma = 1.4826 * float(np.median(res[keep])) + 1.0
            keep = res <= 2.5 * sigma
        out[..., c] = out[..., c] * g + b
    return np.clip(out, 0, 255)


def classify_change(cur_lab: np.ndarray, ref_lab: np.ndarray, comp: np.ndarray, elongation: float) -> str:
    m = comp > 0
    dL = float(cur_lab[..., 0][m].mean() - ref_lab[..., 0][m].mean())
    da = float(cur_lab[..., 1][m].mean() - ref_lab[..., 1][m].mean())
    db = float(cur_lab[..., 2][m].mean() - ref_lab[..., 2][m].mean())
    chroma_cur = np.hypot(cur_lab[..., 1][m] - 128, cur_lab[..., 2][m] - 128).mean()
    chroma_ref = np.hypot(ref_lab[..., 1][m] - 128, ref_lab[..., 2][m] - 128).mean()
    if da > 6 and db > 6:
        return "rust"
    if dL < -20 and elongation >= 3.0:
        return "crack"
    if dL > 10 and chroma_cur < chroma_ref:
        return "soiling"
    if dL < -20:
        return "crack" if elongation >= 2.0 else "other"
    return "other"


class PixelChangeDetector:
    def __init__(self, config: Dict[str, Any]):
        c = get_nested(config, "change_mask", default={}) or {}
        self.enabled = bool(c.get("enabled", True))
        self.z = float(c.get("z_threshold", 4.0))
        self.abs_floor = float(c.get("abs_floor", 18.0))
        self.min_area_frac = float(c.get("min_area_fraction", 0.0015))
        self.work_side = int(c.get("work_max_side", 320))
        self.net = None
        ckpt = c.get("checkpoint")
        if ckpt and resolve_path(ckpt).is_file():
            try:
                from ml.siamese_change import SiameseChangeNet

                self.net = SiameseChangeNet.load(resolve_path(ckpt))
            except Exception:  # noqa: BLE001
                logger.exception("Could not load Siamese change model %s; using classical masks", ckpt)

    def _classical(self, cur: np.ndarray, ref: np.ndarray, valid: np.ndarray) -> np.ndarray:
        cur_lab = cv2.cvtColor(cur.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB).astype(np.float32)
        d_col = np.sqrt(((cur_lab - ref_lab) ** 2).sum(axis=-1))
        g = lambda x: cv2.GaussianBlur(cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_RGB2GRAY), (3, 3), 0)
        gc, gr = g(cur).astype(np.float32), g(ref).astype(np.float32)
        d_grad = np.abs(cv2.Laplacian(gc, cv2.CV_32F) - cv2.Laplacian(gr, cv2.CV_32F))
        d = cv2.GaussianBlur(d_col + 0.25 * d_grad, (5, 5), 0)
        v = valid > 0
        if v.sum() < 16:
            return np.zeros(valid.shape, np.uint8)
        med = float(np.median(d[v]))
        mad = float(np.median(np.abs(d[v] - med))) * 1.4826 + 1e-3
        m = ((d - med) / mad >= self.z) & (d >= self.abs_floor) & v
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    def detect(
        self,
        current: Image.Image,
        reference: Image.Image,
        homography: Optional[np.ndarray],
        cm2_per_current_px: Optional[float] = None,
    ) -> Optional[ChangeMask]:
        if not self.enabled:
            return None
        rw, rh = reference.size
        s = min(1.0, self.work_side / float(max(rw, rh)))
        W, Hh = max(8, int(rw * s)), max(8, int(rh * s))
        S = np.diag([s, s, 1.0])
        H = None if homography is None else S @ homography
        if homography is None:
            # no geometry: compare the views at the reference size
            H_full = np.diag([rw / float(current.width), rh / float(current.height), 1.0])
            H = S @ H_full
        ref_small = np.asarray(reference.convert("RGB").resize((W, Hh), Image.BILINEAR))
        cur_w, valid = warp_to_reference(current, (W, Hh), H)
        cur_n = normalise_to(cur_w, ref_small, valid)
        backend = "classical"
        if self.net is not None:
            mask = self.net.predict_mask(cur_n.astype(np.uint8), ref_small) & (valid > 0).astype(np.uint8)
            backend = "siamese"
        else:
            mask = self._classical(cur_n, ref_small, valid)

        cur_lab = cv2.cvtColor(cur_n.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(ref_small, cv2.COLOR_RGB2LAB).astype(np.float32)
        regions = regions_from_mask(mask, self.min_area_frac)
        n, labels = cv2.connectedComponents(mask, connectivity=8)
        type_area: Dict[str, float] = {}
        for r in regions:
            cx, cy = int(r.centroid[0] * W), int(r.centroid[1] * Hh)
            lbl = labels[min(Hh - 1, cy), min(W - 1, cx)]
            comp = (labels == lbl).astype(np.uint8) if lbl > 0 else np.zeros_like(mask)
            if not comp.any():
                x1, y1, x2, y2 = (int(v) for v in (r.bbox[0] * W, r.bbox[1] * Hh, r.bbox[2] * W, r.bbox[3] * Hh))
                comp = np.zeros_like(mask)
                comp[y1:y2 + 1, x1:x2 + 1] = mask[y1:y2 + 1, x1:x2 + 1]
            bw, bh = (r.bbox[2] - r.bbox[0]) * W, (r.bbox[3] - r.bbox[1]) * Hh
            elong = max(r.length_px, max(bw, bh)) / max(1.0, r.area_px / max(r.length_px, 1.0))
            r.change_type = classify_change(cur_lab, ref_lab, comp, elong)
            type_area[r.change_type] = type_area.get(r.change_type, 0.0) + r.area_px
        # report areas at full reference resolution
        scale_back = 1.0 / (s * s)
        changed_px = float(mask.sum()) * scale_back
        for r in regions:
            r.area_px *= scale_back
            r.length_px /= s
        cm2 = None
        if cm2_per_current_px is not None:
            # reference px² per current px² at the view centre
            hscale = homography_area_scale(homography, current.width / 2.0, current.height / 2.0) if homography is not None \
                else (rw * rh) / float(current.width * current.height)
            cm2 = changed_px * cm2_per_current_px / max(hscale, 1e-9)
        vcount = float((valid > 0).sum())
        return ChangeMask(
            mask=cv2.resize(mask, (rw, rh), interpolation=cv2.INTER_NEAREST),
            valid=cv2.resize(valid, (rw, rh), interpolation=cv2.INTER_NEAREST),
            regions=regions,
            changed_fraction=float(mask.sum()) / max(vcount, 1.0),
            changed_area_px=changed_px,
            changed_area_cm2=cm2,
            change_type=max(type_area, key=type_area.get) if type_area else None,
            backend=backend,
        )


def render_mask(reference: Image.Image, mask: np.ndarray, out_path, color=(255, 0, 60), opacity: float = 0.55) -> None:
    base = np.asarray(reference.convert("RGB")).astype(np.float32)
    m = (mask > 0)[..., None].astype(np.float32) * opacity
    blended = base * (1 - m) + np.array(color, np.float32) * m
    edges = cv2.Canny((mask > 0).astype(np.uint8) * 255, 50, 150) > 0
    blended[edges] = color
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(blended.astype(np.uint8)).save(out_path)
