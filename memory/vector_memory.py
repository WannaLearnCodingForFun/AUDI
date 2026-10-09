"""Partitioned prototype index for asset re-identification (D10).

Each asset keeps a bounded set (≤ matching.max_prototypes_per_asset) of
representative embeddings. Vectors are partitioned by (site_id, asset_type) so
a query only scans candidates of the same site and type. The default backend
is a NumPy matrix; FAISS is opt-in (memory.vector_backend: faiss).
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

PartitionKey = Tuple[str, str]


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def faiss_available() -> bool:
    try:
        import faiss  # noqa: F401
    except ImportError:
        return False
    return True


class VectorBackend(ABC):
    """Inner-product index over unit vectors with integer ids."""

    @abstractmethod
    def add(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        """Insert vectors with the given int64 ids."""

    @abstractmethod
    def remove(self, ids: Iterable[int]) -> None:
        """Delete vectors by id."""

    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (ids, similarities) of the k nearest vectors."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of stored vectors."""


class NumpyBackend(VectorBackend):
    def __init__(self, dim: int):
        self.dim = dim
        self._ids = np.zeros((0,), dtype=np.int64)
        self._mat = np.zeros((0, dim), dtype=np.float32)

    def add(self, ids, vectors):
        self._ids = np.concatenate([self._ids, np.asarray(ids, dtype=np.int64)])
        self._mat = np.vstack([self._mat, np.asarray(vectors, dtype=np.float32).reshape(-1, self.dim)])

    def remove(self, ids):
        drop = np.isin(self._ids, np.fromiter(ids, dtype=np.int64))
        self._ids, self._mat = self._ids[~drop], self._mat[~drop]

    def search(self, query, k):
        if len(self._ids) == 0:
            return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
        sims = self._mat @ query
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return self._ids[top], sims[top]

    def __len__(self):
        return int(len(self._ids))


class FaissBackend(VectorBackend):
    def __init__(self, dim: int):
        import faiss

        self.dim = dim
        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

    def add(self, ids, vectors):
        self._index.add_with_ids(
            np.ascontiguousarray(np.asarray(vectors, dtype=np.float32).reshape(-1, self.dim)),
            np.asarray(ids, dtype=np.int64),
        )

    def remove(self, ids):
        arr = np.fromiter(ids, dtype=np.int64)
        if arr.size:
            self._index.remove_ids(arr)

    def search(self, query, k):
        n = self._index.ntotal
        if n == 0:
            return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
        sims, ids = self._index.search(np.ascontiguousarray(query.reshape(1, -1)), min(k, n))
        keep = ids[0] >= 0
        return ids[0][keep], sims[0][keep]

    def __len__(self):
        return int(self._index.ntotal)


@dataclass
class Prototype:
    proto_id: int
    asset_id: str
    vector: np.ndarray


@dataclass
class Candidate:
    asset_id: str
    similarity: float
    proto_id: int


class PrototypeIndex:
    def __init__(self, dim: int, backend: str, max_prototypes: int, novelty: float):
        if backend not in {"faiss", "numpy"}:
            raise ValueError(f"Unknown vector backend: {backend}")
        if backend == "faiss" and not faiss_available():
            raise ImportError("memory.vector_backend=faiss but faiss is not installed")
        self.backend_name = backend
        self.dim = dim
        self.max_prototypes = max_prototypes
        self.novelty = novelty
        self._backends: Dict[PartitionKey, VectorBackend] = {}
        self._protos: Dict[int, Prototype] = {}
        self._proto_partition: Dict[int, PartitionKey] = {}
        self._asset_protos: Dict[str, List[int]] = {}
        self._lock = threading.RLock()

    def _backend(self, key: PartitionKey) -> VectorBackend:
        if key not in self._backends:
            self._backends[key] = FaissBackend(self.dim) if self.backend_name == "faiss" else NumpyBackend(self.dim)
        return self._backends[key]

    def __len__(self) -> int:
        return len(self._protos)

    def asset_count(self) -> int:
        return len(self._asset_protos)

    def add(self, proto_id: int, asset_id: str, site_id: str, asset_type: str, vector: np.ndarray) -> None:
        vec = _unit(vector)
        if vec.shape[0] != self.dim:
            raise ValueError(f"Embedding dim {vec.shape[0]} != index dim {self.dim}")
        with self._lock:
            key = (site_id, asset_type)
            self._backend(key).add(np.array([proto_id]), vec[None, :])
            self._protos[proto_id] = Prototype(proto_id, asset_id, vec)
            self._proto_partition[proto_id] = key
            self._asset_protos.setdefault(asset_id, []).append(proto_id)

    def remove(self, proto_id: int) -> None:
        with self._lock:
            proto = self._protos.pop(proto_id, None)
            if proto is None:
                return
            self._backends[self._proto_partition.pop(proto_id)].remove([proto_id])
            ids = self._asset_protos.get(proto.asset_id, [])
            if proto_id in ids:
                ids.remove(proto_id)
            if not ids:
                self._asset_protos.pop(proto.asset_id, None)

    def remove_asset(self, asset_id: str) -> None:
        for pid in list(self._asset_protos.get(asset_id, [])):
            self.remove(pid)

    def prototypes(self, asset_id: str) -> List[Prototype]:
        return [self._protos[p] for p in self._asset_protos.get(asset_id, [])]

    def asset_ids(self, site_id: str) -> Set[str]:
        with self._lock:
            return {self._protos[p].asset_id for p, key in self._proto_partition.items() if key[0] == site_id}

    def plan_update(self, asset_id: str, vector: np.ndarray) -> Tuple[bool, Optional[int]]:
        """Decide whether `vector` should become a prototype of `asset_id`.

        Returns (add, evict_proto_id). A vector is added only if it is novel
        (max cosine to existing prototypes < 1 − novelty). When the asset is at
        capacity the most redundant prototype (highest mean similarity to the
        others) is evicted so the set stays diverse and bounded.
        """
        vec = _unit(vector)
        protos = self.prototypes(asset_id)
        if not protos:
            return True, None
        mat = np.stack([p.vector for p in protos])
        if float((mat @ vec).max()) >= 1.0 - self.novelty:
            return False, None
        if len(protos) < self.max_prototypes:
            return True, None
        gram = mat @ mat.T
        np.fill_diagonal(gram, 0.0)
        redundancy = gram.sum(axis=1) / max(len(protos) - 1, 1)
        return True, protos[int(np.argmax(redundancy))].proto_id

    def search(
        self,
        vector: np.ndarray,
        site_id: str,
        asset_type: str,
        top_k: int,
        allowed_assets: Optional[Set[str]] = None,
    ) -> List[Candidate]:
        """Top-k assets (max similarity over their prototypes) in one partition."""
        q = _unit(vector)
        with self._lock:
            if allowed_assets is not None:
                pairs = [
                    (pid, float(self._protos[pid].vector @ q))
                    for aid in allowed_assets
                    for pid in self._asset_protos.get(aid, [])
                    if self._proto_partition.get(pid) == (site_id, asset_type)
                ]
            else:
                backend = self._backends.get((site_id, asset_type))
                if backend is None or len(backend) == 0:
                    return []
                ids, sims = backend.search(q, top_k * self.max_prototypes)
                pairs = list(zip(ids.tolist(), sims.tolist()))
            best: Dict[str, Candidate] = {}
            for pid, sim in pairs:
                aid = self._protos[pid].asset_id
                if aid not in best or sim > best[aid].similarity:
                    best[aid] = Candidate(aid, float(sim), int(pid))
        return sorted(best.values(), key=lambda c: c.similarity, reverse=True)[:top_k]
