"""Defect segmentation → masks for area and length (upgrade 3.3).

Backends behind `DefectSegmenter`:
  classical  training-free: morphological black-hat for cracks (thin dark structures), an
             HSV rust-hue window for corrosion, low-saturation / bright film for soiling.
  yolo_seg   an Ultralytics YOLO-seg checkpoint (segmentation.yolo_seg_weights) fine-tuned on
             the defect classes; used when the file exists.
  sam2       SAM2 prompted with the detection box (segmentation.sam2_checkpoint); the classical
             mask's peak is passed as a positive point prompt.

Every backend returns a binary mask the size of the crop; `regions_from_mask` turns it into
per-instance polygons with area and skeleton length.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from ml.config import get_nested, resolve_path
from ml.forecast import family

logger = logging.getLogger(__name__)


@dataclass
class Region:
    polygon: List[List[float]]      # normalised [0,1] (x, y) vertices
    bbox: List[float]               # normalised x1, y1, x2, y2
    centroid: List[float]           # normalised
    area_px: float
    area_norm: float                # fraction of the image
    length_px: float                # skeleton length (cracks)
    defect_type: Optional[str] = None
    change_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def skeleton_length(mask: np.ndarray) -> float:
    m = (mask > 0).astype(np.uint8) * 255
    if not m.any():
        return 0.0
    if hasattr(cv2, "ximgproc"):
        sk = cv2.ximgproc.thinning(m)
    else:  # morphological skeleton
        sk = np.zeros_like(m)
        el = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        img = m.copy()
        while img.any():
            eroded = cv2.erode(img, el)
            sk |= cv2.subtract(img, cv2.dilate(eroded, el))
            img = eroded
    pts = (sk > 0).astype(np.uint8)
    # 8-connected path length: diagonal steps count √2
    n = float(pts.sum())
    diag = float((pts[1:, 1:] & pts[:-1, :-1]).sum() + (pts[1:, :-1] & pts[:-1, 1:]).sum())
    orth = float((pts[1:, :] & pts[:-1, :]).sum() + (pts[:, 1:] & pts[:, :-1]).sum())
    return max(n, orth + diag * np.sqrt(2.0)) if n > 1 else n


def regions_from_mask(mask: np.ndarray, min_area_frac: float = 0.0005, defect_type: Optional[str] = None) -> List[Region]:
    h, w = mask.shape[:2]
    m = (mask > 0).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
    out: List[Region] = []
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area / float(h * w) < min_area_frac:
            continue
        comp = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key=cv2.contourArea)
        approx = cv2.approxPolyDP(cnt, 1.5, True).reshape(-1, 2)
        if len(approx) < 3:
            x, y, bw, bh = cv2.boundingRect(cnt)
            approx = np.array([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]])
        x, y, bw, bh = (int(v) for v in stats[i, :4])
        out.append(Region(
            polygon=[[float(px) / w, float(py) / h] for px, py in approx],
            bbox=[x / w, y / h, (x + bw) / w, (y + bh) / h],
            centroid=[float(cents[i][0]) / w, float(cents[i][1]) / h],
            area_px=area,
            area_norm=area / float(h * w),
            length_px=skeleton_length(comp),
            defect_type=defect_type,
        ))
    return out


class DefectSegmenter(ABC):
    @abstractmethod
    def segment(self, crop: Image.Image, defect_type: Optional[str]) -> np.ndarray:
        """Binary uint8 mask (H, W) of defect pixels in the crop."""


class ClassicalSegmenter(DefectSegmenter):
    def __init__(self, config: Dict[str, Any]):
        c = get_nested(config, "segmentation", default={}) or {}
        self.kernel_frac = float(c.get("crack_kernel_fraction", 0.04))
        self.crack_min_contrast = float(c.get("crack_min_contrast", 25.0))
        self.min_elongation = float(c.get("crack_min_elongation", 3.0))
        self.strong_factor = float(c.get("crack_strong_factor", 1.8))
        self.line_angle_tol = float(c.get("structural_line_angle_deg", 2.0))
        self.line_span = float(c.get("structural_line_span", 0.5))
        self.line_max_thickness = float(c.get("structural_line_max_px", 4.0))
        self.rust_hue = tuple(c.get("rust_hue_range", [5, 25]))        # OpenCV hue 0–180
        self.rust_min_sat = int(c.get("rust_min_saturation", 90))
        self.soil_max_sat = int(c.get("soiling_max_saturation", 40))
        self.soil_min_val = int(c.get("soiling_min_value", 150))

    def crack_mask(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape
        k = max(3, int(round(min(h, w) * self.kernel_frac)) | 1)
        bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        thr = max(self.crack_min_contrast, float(cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0]))
        weak = (bh >= thr).astype(np.uint8)
        # hysteresis: a crack has a high-contrast core; faint thin lines (grid, seams, texture) do not
        strong = bh >= max(thr * self.strong_factor, self.crack_min_contrast * self.strong_factor)
        n, labels = cv2.connectedComponents(weak, connectivity=8)
        keep_lbl = np.unique(labels[strong & (weak > 0)])
        m = np.isin(labels, keep_lbl[keep_lbl > 0]).astype(np.uint8)
        # remove structural lines first (they often touch a crack and would join its component):
        # an opening with a long 1-px kernel keeps only straight runs at least that long
        lh = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((1, max(3, int(self.line_span * w))), np.uint8))
        lv = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((max(3, int(self.line_span * h)), 1), np.uint8))
        lines = cv2.dilate(lh | lv, np.ones((3, 3), np.uint8))
        m = m & (1 - lines)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        # keep elongated components only (cracks are thin)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        keep = np.zeros_like(m)
        for i in range(1, n):
            comp = (labels == i).astype(np.uint8)
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not cnts or len(cnts[0]) < 5:
                continue
            (_, _), (a, b), ang = cv2.minAreaRect(cnts[0])
            elong = max(a, b) / max(1.0, min(a, b))
            if elong < self.min_elongation:
                continue
            # structural lines (PV cell grid, busbars, panel seams, wall joints) are long,
            # straight and axis-aligned; cracks are not
            x, y, bw, bh = cv2.boundingRect(cnts[0])
            axis_aligned = min(abs(ang) % 90, 90 - abs(ang) % 90) <= self.line_angle_tol
            spans = bw >= self.line_span * w or bh >= self.line_span * h
            straight = min(a, b) <= self.line_max_thickness
            if axis_aligned and spans and straight:
                continue
            keep |= comp
        return keep

    def rust_mask(self, rgb: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        lo, hi = self.rust_hue
        m = ((hsv[..., 0] >= lo) & (hsv[..., 0] <= hi) & (hsv[..., 1] >= self.rust_min_sat) & (hsv[..., 2] >= 40)).astype(np.uint8)
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    def soiling_mask(self, rgb: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        m = ((hsv[..., 1] <= self.soil_max_sat) & (hsv[..., 2] >= self.soil_min_val)).astype(np.uint8)
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    def segment(self, crop: Image.Image, defect_type: Optional[str]) -> np.ndarray:
        rgb = np.asarray(crop.convert("RGB"))
        fam = family(defect_type)
        if fam == "crack":
            return self.crack_mask(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        if fam == "soiling":
            return self.soiling_mask(rgb)
        return self.rust_mask(rgb)


class YoloSegSegmenter(DefectSegmenter):
    def __init__(self, weights: str, conf: float = 0.25):
        from ultralytics import YOLO

        self.model = YOLO(weights, task="segment")
        self.conf = conf

    def segment(self, crop: Image.Image, defect_type: Optional[str]) -> np.ndarray:
        res = self.model.predict(np.asarray(crop.convert("RGB"))[..., ::-1], conf=self.conf, verbose=False)[0]
        out = np.zeros((crop.height, crop.width), np.uint8)
        if res.masks is None:
            return out
        for m in res.masks.data.cpu().numpy():
            out |= (cv2.resize(m, (crop.width, crop.height)) > 0.5).astype(np.uint8)
        return out


class Sam2Segmenter(DefectSegmenter):
    """SAM2 prompted with the whole crop box plus the classical mask's strongest point."""

    def __init__(self, config: Dict[str, Any], checkpoint: str, model_cfg: str):
        from sam2.build_sam import build_sam2  # type: ignore
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

        self.predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
        self.prior = ClassicalSegmenter(config)

    def segment(self, crop: Image.Image, defect_type: Optional[str]) -> np.ndarray:
        rgb = np.asarray(crop.convert("RGB"))
        prior = self.prior.segment(crop, defect_type)
        self.predictor.set_image(rgb)
        kwargs: Dict[str, Any] = {"box": np.array([0, 0, crop.width, crop.height]), "multimask_output": False}
        if prior.any():
            ys, xs = np.nonzero(prior)
            kwargs.update(point_coords=np.array([[xs[len(xs) // 2], ys[len(ys) // 2]]]), point_labels=np.array([1]))
        masks, _, _ = self.predictor.predict(**kwargs)
        return (masks[0] > 0).astype(np.uint8)


def build_segmenter(config: Dict[str, Any]) -> DefectSegmenter:
    c = get_nested(config, "segmentation", default={}) or {}
    backend = c.get("backend", "classical")
    try:
        if backend == "yolo_seg" and resolve_path(c.get("yolo_seg_weights", "")).is_file():
            return YoloSegSegmenter(str(resolve_path(c["yolo_seg_weights"])), float(c.get("confidence", 0.25)))
        if backend == "sam2" and resolve_path(c.get("sam2_checkpoint", "")).is_file():
            return Sam2Segmenter(config, str(resolve_path(c["sam2_checkpoint"])), c.get("sam2_model_cfg", "sam2_hiera_s.yaml"))
    except Exception:  # noqa: BLE001 - optional backends fall back to the classical one
        logger.exception("Segmentation backend %s unavailable; using classical", backend)
    return ClassicalSegmenter(config)
