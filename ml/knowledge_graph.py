"""Asset knowledge graph (upgrade 5.1).

Edges live in `asset_relations` as parent → child, where the PARENT depends on the CHILD:
inverter → string → panel, feeder → span → pole, building → floor → wall. Queries use
recursive CTEs (SQLite and Postgres share the syntax), so the hierarchy can be walked in the
database.

Propagation: a deteriorating component raises the priority of everything that depends on it.
For every asset, boost = graph.weight · max over its descendants d of R_d · decay^(depth−1),
counted only above graph.min_source_priority. A degrading string therefore flags its inverter,
and a failing panel flags its string (full weight) and inverter (decayed).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Sequence

from sqlalchemy import text

from ml.config import get_nested

ANCESTORS_SQL = """
WITH RECURSIVE up(asset_id, depth) AS (
    SELECT parent_id, 1 FROM asset_relations WHERE child_id = :asset_id
    UNION
    SELECT r.parent_id, up.depth + 1 FROM asset_relations r JOIN up ON r.child_id = up.asset_id
    WHERE up.depth < :max_depth
)
SELECT asset_id, MIN(depth) AS depth FROM up GROUP BY asset_id ORDER BY depth
"""

DESCENDANTS_SQL = """
WITH RECURSIVE down(asset_id, depth) AS (
    SELECT child_id, 1 FROM asset_relations WHERE parent_id = :asset_id
    UNION
    SELECT r.child_id, down.depth + 1 FROM asset_relations r JOIN down ON r.parent_id = down.asset_id
    WHERE down.depth < :max_depth
)
SELECT asset_id, MIN(depth) AS depth FROM down GROUP BY asset_id ORDER BY depth
"""


def ancestors(store, asset_id: str, max_depth: int = 8) -> List[Dict[str, Any]]:
    with store.engine.connect() as conn:
        return [{"asset_id": r[0], "depth": int(r[1])} for r in conn.execute(text(ANCESTORS_SQL), {"asset_id": asset_id, "max_depth": max_depth})]


def descendants(store, asset_id: str, max_depth: int = 8) -> List[Dict[str, Any]]:
    with store.engine.connect() as conn:
        return [{"asset_id": r[0], "depth": int(r[1])} for r in conn.execute(text(DESCENDANTS_SQL), {"asset_id": asset_id, "max_depth": max_depth})]


def propagate_priority(priorities: Dict[str, float], relations: Sequence[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    g = get_nested(config, "graph", default={}) or {}
    weight = float(g.get("weight", 0.3))
    decay = float(g.get("decay", 0.5))
    min_src = float(g.get("min_source_priority", 0.3))
    max_depth = int(g.get("max_depth", 6))
    parents: Dict[str, List[str]] = defaultdict(list)
    for r in relations:
        parents[r["child_id"]].append(r["parent_id"])
    best: Dict[str, Dict[str, Any]] = {}
    for src, pr in priorities.items():
        if pr < min_src:
            continue
        frontier, seen, depth = [src], {src}, 0
        while frontier and depth < max_depth:
            depth += 1
            nxt = []
            for node in frontier:
                for p in parents.get(node, []):
                    if p in seen:
                        continue
                    seen.add(p)
                    nxt.append(p)
                    b = weight * pr * decay ** (depth - 1)
                    if b > best.get(p, {}).get("boost", 0.0):
                        best[p] = {"boost": b, "source": src, "depth": depth}
            frontier = nxt
    return best


def subgraph(store, site_id: str) -> Dict[str, Any]:
    """Nodes (assets with health / priority) and edges of a site, for the UI and NL queries."""
    assets = store.list_assets(site_id)
    return {
        "nodes": [{k: a.get(k) for k in ("asset_id", "asset_type", "current_condition", "revisit_priority_score", "health_index", "location")}
                  for a in assets],
        "edges": store.list_relations(site_id),
    }
