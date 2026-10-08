"""Cosine similarity utilities for normalized embeddings."""

from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np


ArrayLike = Union[np.ndarray, List[float]]


def to_unit_vector(vector: ArrayLike) -> np.ndarray:
    """L2-normalize a vector; return zero vector if norm is negligible."""
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(arr)
    if norm < 1e-12:
        return arr
    return arr / norm


def cosine_similarity(a: ArrayLike, b: ArrayLike) -> float:
    """Compute cosine similarity between two vectors."""
    va = to_unit_vector(a)
    vb = to_unit_vector(b)
    return float(np.clip(np.dot(va, vb), -1.0, 1.0))


def batch_cosine_similarity(query: ArrayLike, candidates: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between one query and a batch of candidate vectors.

    Args:
        query: 1-D embedding vector.
        candidates: 2-D array of shape (N, D).

    Returns:
        1-D array of similarities of shape (N,).
    """
    q = to_unit_vector(query)
    if candidates.size == 0:
        return np.array([], dtype=np.float32)

    cand = np.asarray(candidates, dtype=np.float32)
    if cand.ndim == 1:
        cand = cand.reshape(1, -1)

    norms = np.linalg.norm(cand, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    cand_norm = cand / norms
    return np.clip(cand_norm @ q, -1.0, 1.0)


def top_k_similar(
    query: ArrayLike,
    candidates: np.ndarray,
    asset_ids: List[str],
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """Return top-k (asset_id, similarity) pairs sorted by descending similarity."""
    if len(asset_ids) == 0:
        return []

    sims = batch_cosine_similarity(query, candidates)
    k = min(top_k, len(asset_ids))
    indices = np.argsort(sims)[::-1][:k]
    return [(asset_ids[i], float(sims[i])) for i in indices]
