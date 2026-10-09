"""Condition change detection for a re-identified asset (FR-13).

M = clip(α·a + (1−α)·b, 0, 1)
  a  condition delta: condition_l1_weight · L1(current class scores, last
     CONFIRMED class scores) + defect_flip_step if is_defect flips
     False→True + defect_type_change_step if the defect type changes; clipped.
  b  visual local delta: the current crop is warped onto the reference view
     with the identity homography, both are encoded into patch tokens, and
     b = clip((P_pct − P_50) / patch_scale, 0, 1) where P_pct is the
     patch_percentile-th percentile and P_50 the median of per-patch cosine
     dissimilarity. The median is the pair's own baseline (global lighting /
     residual view noise raises every patch), so only localised change counts;
     a high percentile keeps a small new crack from being averaged away. The
     baseline is per pair, not an absolute floor, so it holds for any backbone.
When no reference image is available b is undefined and M = a.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import cv2
import numpy as np
from PIL import Image

from ml.condition_classifier import ConditionPrediction
from ml.embedding_model import AssetEmbeddingModel


@dataclass
class ReferenceCondition:
    """Last confirmed condition of an asset."""

    label: str
    scores: Dict[str, float]
    is_defect: bool
    defect_type: Optional[str]


@dataclass
class ChangeAssessment:
    magnitude: float
    condition_delta: float
    visual_delta: Optional[float]
    patch_percentile_value: Optional[float]
    significant: bool
    warped: bool
    heatmap: Optional[np.ndarray] = None   # (G, G) in [0, 1]

    def to_dict(self) -> Dict:
        return {
            "change_magnitude": self.magnitude,
            "condition_delta": self.condition_delta,
            "visual_delta": self.visual_delta,
            "patch_percentile_value": self.patch_percentile_value,
            "significant": self.significant,
            "warped": self.warped,
        }


class ChangeDetector(ABC):
    @abstractmethod
    def assess(
        self,
        current: ConditionPrediction,
        reference: ReferenceCondition,
        current_crop: Image.Image,
        reference_image: Optional[Image.Image],
        homography: Optional[np.ndarray],
    ) -> ChangeAssessment:
        """Compare the current observation with the asset's confirmed reference."""


class ConditionVisualChangeDetector(ChangeDetector):
    def __init__(self, config: Dict, embedder: AssetEmbeddingModel):
        c = config["change_detection"]
        self.threshold = float(c["threshold"])
        self.alpha = float(c["alpha"])
        self.l1_weight = float(c["condition_l1_weight"])
        self.flip_step = float(c["defect_flip_step"])
        self.type_step = float(c["defect_type_change_step"])
        self.percentile = float(c["patch_percentile"])
        self.scale = float(c["patch_scale"])
        self.min_coverage = float(c["patch_min_coverage"])
        self.embedder = embedder

    def condition_delta(self, current: ConditionPrediction, reference: ReferenceCondition) -> float:
        labels: Sequence[str] = sorted(set(current.scores) | set(reference.scores))
        cur = np.array([current.scores.get(l, 0.0) for l in labels])
        ref = np.array([reference.scores.get(l, 0.0) for l in labels])
        a = self.l1_weight * float(np.abs(cur - ref).sum())
        if current.is_defect and not reference.is_defect:
            a += self.flip_step
        elif current.is_defect and reference.is_defect and current.defect_type != reference.defect_type:
            a += self.type_step
        return float(np.clip(a, 0.0, 1.0))

    def _warp(self, crop: Image.Image, ref_size: tuple, H: np.ndarray):
        src = np.asarray(crop.convert("RGB"))
        w, h = ref_size
        warped = cv2.warpPerspective(src, H, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        mask = cv2.warpPerspective(np.full(src.shape[:2], 255, np.uint8), H, (w, h), flags=cv2.INTER_NEAREST)
        return Image.fromarray(warped), mask

    def visual_delta(
        self,
        current_crop: Image.Image,
        reference_image: Image.Image,
        homography: Optional[np.ndarray],
    ):
        mask = None
        query = current_crop
        if homography is not None:
            query, mask = self._warp(current_crop, reference_image.size, homography)
        _, patches = self.embedder.encode_with_patches([query, reference_image])
        q, r = patches[0], patches[1]
        grid = q.shape[0]
        dissim = 1.0 - np.clip((q * r).sum(axis=-1), -1.0, 1.0)
        valid = np.ones((grid, grid), dtype=bool)
        if mask is not None:
            coverage = cv2.resize((mask > 0).astype(np.float32), (grid, grid), interpolation=cv2.INTER_AREA)
            valid = coverage >= self.min_coverage
        if not valid.any():
            return None, None, None
        p = float(np.percentile(dissim[valid], self.percentile))
        baseline = float(np.median(dissim[valid]))
        b = float(np.clip((p - baseline) / self.scale, 0.0, 1.0))
        heat = np.where(valid, np.clip((dissim - baseline) / self.scale, 0.0, 1.0), 0.0).astype(np.float32)
        return b, p, heat

    def assess(self, current, reference, current_crop, reference_image, homography) -> ChangeAssessment:
        a = self.condition_delta(current, reference)
        b = p = heat = None
        if reference_image is not None:
            b, p, heat = self.visual_delta(current_crop, reference_image, homography)
        m = a if b is None else self.alpha * a + (1.0 - self.alpha) * b
        m = float(np.clip(m, 0.0, 1.0))
        return ChangeAssessment(
            magnitude=m,
            condition_delta=a,
            visual_delta=b,
            patch_percentile_value=p,
            significant=m >= self.threshold,
            warped=homography is not None and b is not None,
            heatmap=heat,
        )


def render_heatmap(image: Image.Image, heat: np.ndarray, out_path, opacity: float) -> None:
    """Blend a (G, G) change heatmap over `image` and write it as PNG."""
    base = np.asarray(image.convert("RGB"))
    h, w = base.shape[:2]
    up = cv2.resize((heat * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    colour = cv2.cvtColor(cv2.applyColorMap(up, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    alpha = (up.astype(np.float32) / 255.0)[..., None] * opacity
    blended = (base * (1 - alpha) + colour * alpha).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(blended).save(out_path)
