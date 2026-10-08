"""Condition change detection — separate from asset identity matching."""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np

from ml.schemas import ChangeResult
from ml.similarity import batch_cosine_similarity, cosine_similarity


class ChangeDetector:
    """
    Detect physical condition change between observations of the same asset.

    Asset identity similarity answers: "Is this the same physical asset?"
    Condition change magnitude answers: "Has the condition changed?"
    """

    def __init__(self, threshold: float = 0.30):
        self.threshold = threshold

    def compare(
        self,
        current_embedding: Union[np.ndarray, List[float]],
        historical_embeddings: Union[np.ndarray, List[List[float]]],
        current_image=None,
        historical_image=None,
    ) -> ChangeResult:
        """
        Compare current observation against historical embedding sequence.

        Change magnitude is normalized to [0, 1]:
          0 → no meaningful change (high similarity to baseline/recent)
          1 → very large change (low similarity)

        The optional image arguments are reserved for a future image-based
        change detection model; they are not used in the baseline implementation.
        """
        _ = current_image, historical_image  # reserved for future CNN-based detector

        hist = np.asarray(historical_embeddings, dtype=np.float32)
        if hist.size == 0:
            return ChangeResult(change_magnitude=0.0, confidence=0.5, changed=False)

        if hist.ndim == 1:
            hist = hist.reshape(1, -1)

        sims = batch_cosine_similarity(current_embedding, hist)
        max_sim = float(np.max(sims))
        mean_sim = float(np.mean(sims))

        # Use both best-match and average similarity for robust change estimate
        combined_sim = 0.6 * max_sim + 0.4 * mean_sim
        change_magnitude = float(np.clip(1.0 - combined_sim, 0.0, 1.0))

        # Confidence increases with number of historical observations
        confidence = float(np.clip(0.5 + 0.1 * min(len(hist), 5), 0.0, 1.0))
        changed = change_magnitude >= self.threshold

        return ChangeResult(
            change_magnitude=change_magnitude,
            confidence=confidence,
            changed=changed,
        )

    def compare_to_baseline(
        self,
        current_embedding: Union[np.ndarray, List[float]],
        baseline_embedding: Union[np.ndarray, List[float]],
    ) -> ChangeResult:
        """Compare directly against the first (baseline) observation."""
        sim = cosine_similarity(current_embedding, baseline_embedding)
        change_magnitude = float(np.clip(1.0 - sim, 0.0, 1.0))
        changed = change_magnitude >= self.threshold
        return ChangeResult(change_magnitude=change_magnitude, confidence=0.8, changed=changed)
