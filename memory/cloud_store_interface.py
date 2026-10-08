"""Abstract cloud store interface for future remote persistence."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class CloudStoreInterface(ABC):
    """
    Abstract interface for cloud-backed semantic memory.

    The local store implements the same contract so the ML pipeline
    can operate offline without a cloud database connection.
    """

    @abstractmethod
    def connect(self) -> bool:
        """Attempt connection; return True on success."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return whether cloud store is reachable."""

    @abstractmethod
    def save_asset(self, asset: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def get_asset(self, asset_id: str) -> Optional[Dict[str, Any]]:
        ...

    @abstractmethod
    def search_assets(self, embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def add_observation(self, asset_id: str, observation: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def get_history(self, asset_id: str) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def sync_from_local(self, local_assets: List[Dict[str, Any]]) -> int:
        """Push local assets to cloud; return number synced."""

    @abstractmethod
    def sync_to_local(self) -> List[Dict[str, Any]]:
        """Pull assets from cloud for offline operation."""


class UnavailableCloudStore(CloudStoreInterface):
    """Placeholder cloud store that gracefully degrades to local-only mode."""

    def connect(self) -> bool:
        return False

    def is_available(self) -> bool:
        return False

    def save_asset(self, asset: Dict[str, Any]) -> None:
        raise ConnectionError("Cloud store unavailable")

    def get_asset(self, asset_id: str) -> Optional[Dict[str, Any]]:
        return None

    def search_assets(self, embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        return []

    def add_observation(self, asset_id: str, observation: Dict[str, Any]) -> None:
        raise ConnectionError("Cloud store unavailable")

    def get_history(self, asset_id: str) -> List[Dict[str, Any]]:
        return []

    def sync_from_local(self, local_assets: List[Dict[str, Any]]) -> int:
        return 0

    def sync_to_local(self) -> List[Dict[str, Any]]:
        return []
