"""Local file-backed semantic memory store for offline operation."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from memory.vector_memory import VectorMemory
from ml.config import get_nested, load_config
from ml.schemas import AssetRecord, Observation

logger = logging.getLogger(__name__)


class SemanticMemoryStore:
    """Abstract store interface — implemented by LocalSemanticMemoryStore."""

    def save_asset(self, asset: Dict[str, Any]) -> None:
        raise NotImplementedError

    def get_asset(self, asset_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def search_assets(self, embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def add_observation(self, asset_id: str, observation: Dict[str, Any]) -> None:
        raise NotImplementedError

    def get_history(self, asset_id: str) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def list_assets(self) -> List[Dict[str, Any]]:
        raise NotImplementedError


class LocalSemanticMemoryStore(SemanticMemoryStore):
    """
    JSON file + in-memory vector index for persistent semantic memory.

    Operates fully offline; cloud DB is optional and accessed via abstraction.
    """

    def __init__(
        self,
        store_path: Optional[Path] = None,
        vector_memory: Optional[VectorMemory] = None,
        config: Optional[Dict] = None,
    ):
        self.config = config or load_config()
        path_str = store_path or get_nested(self.config, "memory", "store_path", default="data/semantic_memory.json")
        self.store_path = Path(path_str)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

        backend = get_nested(self.config, "memory", "vector_backend", default="numpy")
        dim = int(get_nested(self.config, "model", "embedding_dim", default=384))
        self.vector_memory = vector_memory or VectorMemory(backend_name=backend, dim=dim)

        self._assets: Dict[str, AssetRecord] = {}
        self._lock = threading.RLock()
        self._load()

    def _load(self) -> None:
        if not self.store_path.exists():
            logger.info("No existing semantic memory at %s; starting fresh", self.store_path)
            return

        with open(self.store_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for asset_dict in data.get("assets", []):
            record = AssetRecord.from_dict(asset_dict)
            self._assets[record.asset_id] = record
            if record.current_embedding:
                import numpy as np

                self.vector_memory.add(
                    record.asset_id,
                    np.array(record.current_embedding, dtype=np.float32),
                    metadata={"asset_type": record.asset_type},
                )
        logger.info("Loaded %d assets from semantic memory", len(self._assets))

    def _persist(self) -> None:
        payload = {"assets": [asset.to_dict() for asset in self._assets.values()]}
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def save_asset(self, asset: Dict[str, Any]) -> None:
        with self._lock:
            record = AssetRecord.from_dict(asset)
            self._assets[record.asset_id] = record
            if record.current_embedding:
                import numpy as np

                self.vector_memory.add(
                    record.asset_id,
                    np.array(record.current_embedding, dtype=np.float32),
                    metadata={"asset_type": record.asset_type},
                )
            self._persist()

    def get_asset(self, asset_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            record = self._assets.get(asset_id)
            return record.to_dict() if record else None

    def search_assets(self, embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        import numpy as np

        results = self.vector_memory.search(np.array(embedding, dtype=np.float32), top_k=top_k)
        output = []
        for asset_id, similarity, _meta in results:
            asset = self.get_asset(asset_id)
            if asset:
                asset = dict(asset)
                asset["similarity"] = similarity
                output.append(asset)
        return output

    def add_observation(self, asset_id: str, observation: Dict[str, Any]) -> None:
        with self._lock:
            record = self._assets.get(asset_id)
            if record is None:
                raise KeyError(f"Asset not found: {asset_id}")

            obs = Observation.from_dict(observation)
            record.observations.append(obs)
            record.condition_history.append(obs.condition_description)
            record.current_condition = obs.condition_description
            if obs.embedding:
                record.current_embedding = obs.embedding
            self._persist()

    def get_history(self, asset_id: str) -> List[Dict[str, Any]]:
        asset = self.get_asset(asset_id)
        if not asset:
            return []
        return list(asset.get("observations", []))

    def list_assets(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [a.to_dict() for a in self._assets.values()]

    def clear(self) -> None:
        """Clear all assets (primarily for testing)."""
        with self._lock:
            self._assets.clear()
            self.vector_memory = VectorMemory(
                backend_name=get_nested(self.config, "memory", "vector_backend", default="numpy"),
                dim=int(get_nested(self.config, "model", "embedding_dim", default=384)),
            )
            if self.store_path.exists():
                self.store_path.unlink()
