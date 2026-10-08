"""Data schemas for AUDI persistent semantic memory."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_asset_id() -> str:
    return f"asset_{uuid4().hex[:8]}"


def generate_observation_id() -> str:
    return f"obs_{uuid4().hex[:8]}"


@dataclass
class Location:
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Location":
        if not data:
            return cls()
        return cls(
            latitude=float(data.get("latitude", 0.0)),
            longitude=float(data.get("longitude", 0.0)),
            altitude=float(data.get("altitude", 0.0)),
        )


@dataclass
class Observation:
    observation_id: str
    observation_ts: str
    condition_description: str
    image_reference: str
    embedding: List[float]
    severity_score: float = 0.0
    change_magnitude: float = 0.0
    confidence: float = 0.0
    mission_id: Optional[str] = None
    source_class: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if not d.get("metadata"):
            d.pop("metadata", None)
        if d.get("mission_id") is None:
            d.pop("mission_id", None)
        if d.get("source_class") is None:
            d.pop("source_class", None)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Observation":
        return cls(
            observation_id=data.get("observation_id", generate_observation_id()),
            observation_ts=data["observation_ts"],
            condition_description=data.get("condition_description", "normal"),
            image_reference=data.get("image_reference", ""),
            embedding=list(data.get("embedding", [])),
            severity_score=float(data.get("severity_score", 0.0)),
            change_magnitude=float(data.get("change_magnitude", 0.0)),
            confidence=float(data.get("confidence", 0.0)),
            mission_id=data.get("mission_id"),
            source_class=data.get("source_class"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class AssetRecord:
    asset_id: str
    asset_type: str
    location: Location
    first_observed_ts: str
    current_condition: str
    current_embedding: List[float]
    condition_history: List[str] = field(default_factory=list)
    defect_list: List[str] = field(default_factory=list)
    observations: List[Observation] = field(default_factory=list)
    revisit_priority_score: float = 0.0
    asset_criticality: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "asset_type": self.asset_type,
            "location": self.location.to_dict(),
            "first_observed_ts": self.first_observed_ts,
            "current_condition": self.current_condition,
            "current_embedding": self.current_embedding,
            "condition_history": list(self.condition_history),
            "defect_list": list(self.defect_list),
            "observations": [obs.to_dict() for obs in self.observations],
            "revisit_priority_score": self.revisit_priority_score,
            "asset_criticality": self.asset_criticality,
            "metadata": dict(self.metadata) if self.metadata else {},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AssetRecord":
        observations = [Observation.from_dict(o) for o in data.get("observations", [])]
        return cls(
            asset_id=data["asset_id"],
            asset_type=data.get("asset_type", "unknown"),
            location=Location.from_dict(data.get("location")),
            first_observed_ts=data["first_observed_ts"],
            current_condition=data.get("current_condition", "normal"),
            current_embedding=list(data.get("current_embedding", [])),
            condition_history=list(data.get("condition_history", [])),
            defect_list=list(data.get("defect_list", [])),
            observations=observations,
            revisit_priority_score=float(data.get("revisit_priority_score", 0.0)),
            asset_criticality=float(data.get("asset_criticality", 0.5)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class MatchResult:
    matched: bool
    asset_id: Optional[str]
    similarity: float
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChangeResult:
    change_magnitude: float
    confidence: float
    changed: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessingMetrics:
    embedding_latency_ms: float = 0.0
    matching_latency_ms: float = 0.0
    memory_lookup_latency_ms: float = 0.0
    total_processing_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)
