"""Asset identity matching via cosine similarity."""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np

from ml.schemas import MatchResult
from ml.similarity import batch_cosine_similarity


class AssetMatcher:
    """Match query embeddings against candidate asset embeddings."""

    def match(
        self,
        query_embedding: Union[np.ndarray, List[float]],
        candidate_embeddings: np.ndarray,
        candidate_asset_ids: List[str],
        threshold: float,
    ) -> MatchResult:
        """
        Find best matching asset using cosine similarity.

        When multiple embeddings exist per asset (observation gallery),
        uses the maximum similarity across each asset's embeddings.
        """
        if len(candidate_asset_ids) == 0 or candidate_embeddings.size == 0:
            return MatchResult(matched=False, asset_id=None, similarity=0.0, confidence=0.0)

        sims = batch_cosine_similarity(query_embedding, candidate_embeddings)

        # Aggregate: best similarity per asset_id
        best_per_asset: dict[str, float] = {}
        for idx, aid in enumerate(candidate_asset_ids):
            best_per_asset[aid] = max(best_per_asset.get(aid, -1.0), float(sims[idx]))

        best_id = max(best_per_asset, key=best_per_asset.get)
        best_sim = best_per_asset[best_id]

        matched = best_sim >= threshold
        confidence = self._similarity_to_confidence(best_sim, threshold)

        return MatchResult(
            matched=matched,
            asset_id=best_id if matched else None,
            similarity=best_sim,
            confidence=confidence if matched else best_sim,
        )

    @staticmethod
    def _similarity_to_confidence(similarity: float, threshold: float) -> float:
        """Map similarity above threshold to [0, 1] confidence."""
        if similarity < threshold:
            return max(0.0, similarity)
        span = max(1.0 - threshold, 1e-6)
        return float(np.clip((similarity - threshold) / span, 0.0, 1.0))

    def match_dict(
        self,
        query_embedding: Union[np.ndarray, List[float]],
        candidate_embeddings: np.ndarray,
        candidate_asset_ids: List[str],
        threshold: float,
    ) -> Dict:
        """Return match result as dictionary (spec-compatible)."""
        result = self.match(query_embedding, candidate_embeddings, candidate_asset_ids, threshold)
        return result.to_dict()
