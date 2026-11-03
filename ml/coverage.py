"""Mission coverage (upgrade 2.5).

  coverage(store, session_id)   planned assets of the site vs assets seen in this session;
                                the missing list (before landing) is sorted by revisit priority.
  unseen_heatmap(store, site)   days since each asset was last inspected, plus a coarse
                                lat/lon grid of the maximum "unseen for N days" per cell.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, Optional

from ml.config import get_nested
from ml.revisit_priority import parse_ts
from ml.schemas import utc_now


def coverage(store, session_id: str, config: Dict[str, Any], planned_only_due: bool = False) -> Dict[str, Any]:
    sess = store.get_session(session_id)
    if sess is None:
        raise KeyError(session_id)
    seen = set(store.session_assets(session_id))
    assets = store.list_assets(sess["site_id"])
    due_thr = float(get_nested(config, "coverage", "due_priority", default=0.0))
    planned = [a for a in assets if not planned_only_due or a["revisit_priority_score"] >= due_thr or a["asset_id"] in seen]
    planned_ids = {a["asset_id"] for a in planned}
    missing = sorted((a for a in planned if a["asset_id"] not in seen), key=lambda a: -a["revisit_priority_score"])
    new = [a["asset_id"] for a in assets if a["asset_id"] in seen and a["first_observed_ts"] >= sess["started_ts"]]
    return {
        "session_id": session_id,
        "site_id": sess["site_id"],
        "planned": len(planned),
        "seen": len(seen & planned_ids),
        "coverage": (len(seen & planned_ids) / len(planned)) if planned else 1.0,
        "new_assets": sorted(set(new)),
        "missing": [{"asset_id": a["asset_id"], "asset_type": a["asset_type"], "revisit_priority": a["revisit_priority_score"],
                     "location": a["location"], "last_checked_ts": a["last_checked_ts"]} for a in missing],
    }


def unseen_heatmap(store, site_id: str, cell_m: float = 25.0, now: Optional[datetime] = None) -> Dict[str, Any]:
    now = now or utc_now()
    assets = store.list_assets(site_id)
    points = []
    for a in assets:
        days = max(0.0, (now - parse_ts(a["last_checked_ts"])).total_seconds() / 86400.0)
        points.append({"asset_id": a["asset_id"], "asset_type": a["asset_type"], "days_unseen": round(days, 2), "location": a["location"]})
    located = [p for p in points if p["location"]]
    cells: Dict[str, Dict[str, Any]] = {}
    if located:
        lat0 = min(p["location"]["latitude"] for p in located)
        lon0 = min(p["location"]["longitude"] for p in located)
        dlat = cell_m / 111_320.0
        dlon = cell_m / (111_320.0 * max(0.01, math.cos(math.radians(lat0))))
        for p in located:
            i = int((p["location"]["latitude"] - lat0) / dlat)
            j = int((p["location"]["longitude"] - lon0) / dlon)
            c = cells.setdefault(f"{i},{j}", {"row": i, "col": j, "lat": lat0 + (i + 0.5) * dlat, "lon": lon0 + (j + 0.5) * dlon,
                                              "max_days_unseen": 0.0, "assets": 0})
            c["max_days_unseen"] = max(c["max_days_unseen"], p["days_unseen"])
            c["assets"] += 1
    return {"site_id": site_id, "generated_ts": now.isoformat(), "cell_m": cell_m, "assets": points, "cells": list(cells.values())}
