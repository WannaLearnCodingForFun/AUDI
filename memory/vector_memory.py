"""Lightweight vector search layer with pluggable backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ml.similarity import batch_cosine_similarity, to_unit_vector


class VectorBackend(ABC):
    @abstractmethod
    def add(self, asset_id: str, embedding: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> None:
        ...

    @abstractmethod
    def search(self, embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        ...

    @abstractmethod
    def remove(self, asset_id: str) -> bool:
        ...

    @abstractmethod
    def get_all_embeddings(self) -> Tuple[List[str], np.ndarray]:
        ...


class NumpyVectorBackend(VectorBackend):
    """Development-mode in-memory vector store using NumPy cosine similarity."""

    def __init__(self):
        self._asset_ids: List[str] = []
        self._embeddings: List[np.ndarray] = []
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def add(self, asset_id: str, embedding: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> None:
        vec = to_unit_vector(embedding)
        if asset_id in self._metadata:
            idx = self._asset_ids.index(asset_id)
            self._embeddings[idx] = vec
        else:
            self._asset_ids.append(asset_id)
            self._embeddings.append(vec)
        self._metadata[asset_id] = metadata or {}

    def search(self, embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        if not self._asset_ids:
            return []

        matrix = np.stack(self._embeddings)
        sims = batch_cosine_similarity(embedding, matrix)
        k = min(top_k, len(self._asset_ids))
        indices = np.argsort(sims)[::-1][:k]
        return [
            (self._asset_ids[i], float(sims[i]), dict(self._metadata[self._asset_ids[i]]))
            for i in indices
        ]

    def remove(self, asset_id: str) -> bool:
        if asset_id not in self._metadata:
            return False
        idx = self._asset_ids.index(asset_id)
        del self._asset_ids[idx]
        del self._embeddings[idx]
        del self._metadata[asset_id]
        return True

    def get_all_embeddings(self) -> Tuple[List[str], np.ndarray]:
        if not self._asset_ids:
            return [], np.zeros((0, 0), dtype=np.float32)
        return list(self._asset_ids), np.stack(self._embeddings)


class FaissVectorBackend(VectorBackend):
    """
    Scalable FAISS-backed vector store (optional).

    Install faiss-cpu to enable. Falls back to NumPy if unavailable.
    """

    def __init__(self, dim: int):
        try:
            import faiss  # type: ignore
        except ImportError as exc:
            raise ImportError("faiss-cpu is required for FaissVectorBackend") from exc

        self._faiss = faiss
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._asset_ids: List[str] = []
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._id_to_idx: Dict[str, int] = {}

    def add(self, asset_id: str, embedding: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> None:
        vec = to_unit_vector(embedding).astype(np.float32).reshape(1, -1)
        if asset_id in self._id_to_idx:
            # FAISS flat index does not support in-place update; rebuild is expensive.
            # For v1, remove and re-add via metadata tracking.
            self.remove(asset_id)
        self._index.add(vec)
        idx = len(self._asset_ids)
        self._asset_ids.append(asset_id)
        self._id_to_idx[asset_id] = idx
        self._metadata[asset_id] = metadata or {}

    def search(self, embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        if not self._asset_ids:
            return []
        vec = to_unit_vector(embedding).astype(np.float32).reshape(1, -1)
        k = min(top_k, len(self._asset_ids))
        sims, indices = self._index.search(vec, k)
        results = []
        for sim, idx in zip(sims[0], indices[0]):
            if idx < 0 or idx >= len(self._asset_ids):
                continue
            aid = self._asset_ids[idx]
            results.append((aid, float(sim), dict(self._metadata[aid])))
        return results

    def remove(self, asset_id: str) -> bool:
        # Flat index removal requires rebuild — not implemented in v1
        if asset_id not in self._metadata:
            return False
        del self._metadata[asset_id]
        return True

    def get_all_embeddings(self) -> Tuple[List[str], np.ndarray]:
        if self._index.ntotal == 0:
            return [], np.zeros((0, self.dim), dtype=np.float32)
        # Reconstruct not supported on flat IP without stored vectors
        return list(self._asset_ids), np.zeros((len(self._asset_ids), self.dim), dtype=np.float32)


class VectorMemory:
    """
    Vector search facade decoupled from backend implementation.

    Default backend: NumPy (development mode).
    Future: plug in FAISS or external vector DB via VectorBackend interface.
    """

    def __init__(self, backend: Optional[VectorBackend] = None, backend_name: str = "numpy", dim: int = 384):
        if backend is not None:
            self._backend = backend
        elif backend_name == "faiss":
            self._backend = FaissVectorBackend(dim=dim)
        else:
            self._backend = NumpyVectorBackend()

    def add(self, asset_id: str, embedding: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> None:
        self._backend.add(asset_id, embedding, metadata)

    def search(self, embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        return self._backend.search(embedding, top_k=top_k)

    def remove(self, asset_id: str) -> bool:
        return self._backend.remove(asset_id)

    def get_candidate_matrix(self) -> Tuple[List[str], np.ndarray]:
        return self._backend.get_all_embeddings()
