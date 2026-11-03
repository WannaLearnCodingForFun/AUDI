"""Memory administration and the semantic layer: merge / split + audit (1.4), defect
instances (2.2/2.3), knowledge graph (5.1), natural-language queries (5.2), health (5.5)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from ml.health_index import health_band, site_health
from ml.knowledge_graph import ancestors, descendants, subgraph
from ml.nl_query import ask

from api.deps import admin_only, any_user, get_services
from api.services import Services

router = APIRouter(tags=["memory"])


class MergeIn(BaseModel):
    keep_id: str
    merge_id: str
    allow_type_mismatch: bool = False


class SplitIn(BaseModel):
    observation_ids: List[str]


class RelationIn(BaseModel):
    parent_id: str
    child_id: str
    relation: str = "contains"


class QueryIn(BaseModel):
    question: str
    site_id: Optional[str] = None


@router.post("/api/assets/merge")
def merge(body: MergeIn, user: Dict = Depends(admin_only), services: Services = Depends(get_services)) -> Dict[str, Any]:
    try:
        res = services.store.merge_assets(body.keep_id, body.merge_id, user["username"], body.allow_type_mismatch)
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown asset {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    services.engine.reload_index()
    return res


@router.post("/api/assets/{asset_id}/split")
def split(asset_id: str, body: SplitIn, user: Dict = Depends(admin_only), services: Services = Depends(get_services)) -> Dict[str, Any]:
    try:
        res = services.store.split_asset(asset_id, body.observation_ids, user["username"])
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown asset {asset_id}") from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    services.engine.reload_index()
    return res


@router.get("/api/audit")
def audit(site_id: Optional[str] = None, limit: int = 200, _: Dict = Depends(admin_only), services: Services = Depends(get_services)) -> List[Dict[str, Any]]:
    return services.store.list_audit(site_id, max(1, min(limit, 2000)))


@router.get("/api/assets/{asset_id}/defect-instances")
def asset_instances(asset_id: str, _: Dict = Depends(any_user), services: Services = Depends(get_services)) -> List[Dict[str, Any]]:
    if services.store.get_asset(asset_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown asset {asset_id}")
    return services.store.list_defect_instances(asset_id=asset_id)


@router.get("/api/defect-instances/{instance_id}")
def instance(instance_id: str, _: Dict = Depends(any_user), services: Services = Depends(get_services)) -> Dict[str, Any]:
    rows = [i for i in services.store.list_defect_instances() if i["instance_id"] == instance_id]
    if not rows:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown defect instance {instance_id}")
    return {**rows[0], "measurements": services.store.instance_measurements(instance_id)}


@router.get("/api/sites/{site_id}/relations")
def relations(site_id: str, _: Dict = Depends(any_user), services: Services = Depends(get_services)) -> List[Dict[str, Any]]:
    return services.store.list_relations(site_id)


@router.post("/api/relations", status_code=201)
def add_relation(body: RelationIn, user: Dict = Depends(admin_only), services: Services = Depends(get_services)) -> Dict[str, Any]:
    try:
        return services.store.add_relation(body.parent_id, body.child_id, body.relation, user["username"])
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown asset {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc


@router.delete("/api/relations/{relation_id}", status_code=204)
def delete_relation(relation_id: str, user: Dict = Depends(admin_only), services: Services = Depends(get_services)) -> None:
    try:
        services.store.remove_relation(relation_id, user["username"])
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown relation {relation_id}") from exc


@router.get("/api/sites/{site_id}/graph")
def graph(site_id: str, _: Dict = Depends(any_user), services: Services = Depends(get_services)) -> Dict[str, Any]:
    return subgraph(services.store, site_id)


@router.get("/api/assets/{asset_id}/graph")
def asset_graph(asset_id: str, _: Dict = Depends(any_user), services: Services = Depends(get_services)) -> Dict[str, Any]:
    if services.store.get_asset(asset_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown asset {asset_id}")
    return {"asset_id": asset_id, "depends_on_it": ancestors(services.store, asset_id), "depends_on": descendants(services.store, asset_id)}


@router.post("/api/query")
def query(body: QueryIn, _: Dict = Depends(any_user), services: Services = Depends(get_services)) -> Dict[str, Any]:
    if not body.question.strip():
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Empty question")
    return ask(services.store, body.site_id, body.question, services.config)


@router.get("/api/sites/{site_id}/health")
def health(site_id: str, _: Dict = Depends(any_user), services: Services = Depends(get_services)) -> Dict[str, Any]:
    assets = services.store.list_assets(site_id)
    h = site_health(assets)
    return {"site_id": site_id, "health_index": h, "band": health_band(h),
            "assets": sorted(({"asset_id": a["asset_id"], "asset_type": a["asset_type"], "health_index": a.get("health_index"),
                               "band": health_band(a.get("health_index")), "location": a["location"]} for a in assets),
                             key=lambda x: (x["health_index"] if x["health_index"] is not None else 101))}
