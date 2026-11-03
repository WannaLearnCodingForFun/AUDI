"""Local DB access for the upgrade entities, mixed into `LocalStore`.

  audit log ................ 1.4 / 6.5
  site maps, footprints .... 1.2
  merge / split ............ 1.4 (observations and defects move, nothing lost or duplicated)
  defect instances ......... 2.2 / 2.3 (called inside `commit_observation`'s transaction)
  asset relations .......... 5.1
  alert feedback ........... 5.4
  work orders .............. 6.2
  drones ................... 6.3 / 6.5

Every write enqueues outbox records like the base store, so the cloud copy follows.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence

from sqlalchemy import or_, select
from sqlalchemy.orm import Session as OrmSession

from memory import db
from ml.defect_tracking import MeasuredRegion, match_regions, tracking_params
from ml.forecast import days_until, forecast_series
from ml.schemas import new_id, utc_now, utc_now_iso


def _row(obj) -> Dict[str, Any]:
    return {c.name: getattr(obj, c.name) for c in obj.__table__.columns}


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class StoreExtMixin:
    # ── audit ─────────────────────────────────────────────────────────────
    def _audit(self, s: OrmSession, actor: str, action: str, entity: str, entity_id: str,
               site_id: Optional[str], details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        a = db.AuditLog(audit_id=new_id("aud"), ts=utc_now_iso(), actor=actor, action=action, entity=entity,
                        entity_id=entity_id, site_id=site_id, details=details or {})
        s.add(a)
        return _row(a)

    def audit(self, actor: str, action: str, entity: str, entity_id: str, site_id: Optional[str] = None,
              details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with self.txn() as s:
            return self._audit(s, actor, action, entity, entity_id, site_id, details)

    def list_audit(self, site_id: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
        with self.read() as s:
            q = select(db.AuditLog).order_by(db.AuditLog.ts.desc()).limit(limit)
            if site_id:
                q = q.where(db.AuditLog.site_id == site_id)
            return [_row(a) for a in s.scalars(q)]

    # ── site maps (1.2) ───────────────────────────────────────────────────
    def add_site_map(self, site_id: str, image_path: str, geotransform: Sequence[float], width: int, height: int,
                     actor: str = "system", meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with self.txn() as s:
            if s.get(db.Site, site_id) is None:
                raise KeyError(f"Unknown site: {site_id}")
            for old in s.scalars(select(db.SiteMap).where(db.SiteMap.site_id == site_id, db.SiteMap.active.is_(True))):
                old.active = False
            m = db.SiteMap(map_id=new_id("map"), site_id=site_id, image_path=str(image_path), geotransform=list(geotransform),
                           width=int(width), height=int(height), active=True, created_ts=utc_now_iso(), meta=meta or {})
            s.add(m)
            s.flush()
            self._enqueue(s, "site_map", m.map_id, site_id, _row(m))
            self._audit(s, actor, "site_map.add", "site_map", m.map_id, site_id, {"image_path": str(image_path)})
            return _row(m)

    def active_site_map(self, site_id: str) -> Optional[Dict[str, Any]]:
        with self.read() as s:
            m = s.scalar(select(db.SiteMap).where(db.SiteMap.site_id == site_id, db.SiteMap.active.is_(True))
                         .order_by(db.SiteMap.created_ts.desc()).limit(1))
            return _row(m) if m else None

    def set_footprint(self, asset_id: str, polygon: Optional[Sequence[Sequence[float]]], actor: str = "system") -> Dict[str, Any]:
        if polygon is not None and len(polygon) < 3:
            raise ValueError("A footprint needs at least 3 vertices")
        with self.txn() as s:
            a = s.get(db.Asset, asset_id)
            if a is None:
                raise KeyError(asset_id)
            a.footprint = [[float(p[0]), float(p[1])] for p in polygon] if polygon else None
            a.updated_ts = utc_now_iso()
            if polygon and a.latitude is None:
                a.latitude = sum(p[0] for p in polygon) / len(polygon)
                a.longitude = sum(p[1] for p in polygon) / len(polygon)
            self._enqueue(s, "asset", a.asset_id, a.site_id, _row(a))
            self._audit(s, actor, "asset.footprint", "asset", asset_id, a.site_id, {"vertices": len(polygon or [])})
            from memory.store import asset_dict

            return asset_dict(a)

    def footprint_assets(self, site_id: str) -> List[Dict[str, Any]]:
        with self.read() as s:
            return [{"asset_id": a.asset_id, "asset_type": a.asset_type, "footprint": a.footprint}
                    for a in s.scalars(select(db.Asset).where(db.Asset.site_id == site_id, db.Asset.footprint.is_not(None)))]

    def register_mapped_asset(self, site_id: str, asset_type: str, footprint: Sequence[Sequence[float]],
                              criticality: float, actor: str = "system") -> Dict[str, Any]:
        """Create an asset from a site-map polygon before it has ever been observed."""
        now = utc_now_iso()
        lat = sum(p[0] for p in footprint) / len(footprint)
        lon = sum(p[1] for p in footprint) / len(footprint)
        with self.txn() as s:
            a = db.Asset(
                asset_id=new_id("asset"), site_id=site_id, asset_type=asset_type, latitude=lat, longitude=lon,
                altitude=0.0, location_samples=0, first_observed_ts=now, last_checked_ts=now, current_condition="unknown",
                current_is_defect=False, reference_condition={"label": "normal", "scores": {}, "is_defect": False, "defect_type": None},
                criticality=criticality, revisit_priority_score=0.0, simulated=False, meta={"source": "site_map"},
                updated_ts=now, footprint=[[float(p[0]), float(p[1])] for p in footprint],
            )
            s.add(a)
            s.flush()
            self._enqueue(s, "asset", a.asset_id, site_id, _row(a))
            self._audit(s, actor, "asset.register_from_map", "asset", a.asset_id, site_id, {})
            from memory.store import asset_dict

            return asset_dict(a)

    # ── merge / split (1.4) ───────────────────────────────────────────────
    def _refresh_from_observations(self, s: OrmSession, a: db.Asset) -> None:
        obs = list(s.scalars(select(db.Observation).where(db.Observation.asset_id == a.asset_id).order_by(db.Observation.observation_ts)))
        if obs:
            a.first_observed_ts = obs[0].observation_ts
            a.last_checked_ts = obs[-1].observation_ts
            a.current_condition = obs[-1].condition_description
            a.current_is_defect = obs[-1].is_defect
            located = [o for o in obs if o.latitude is not None and o.status in ("new", "confirmed")]
            if located:
                a.latitude = sum(o.latitude for o in located) / len(located)
                a.longitude = sum(o.longitude for o in located) / len(located)
                a.altitude = sum((o.altitude or 0.0) for o in located) / len(located)
                a.location_samples = len(located)
        protos = list(s.scalars(select(db.AssetPrototype.proto_id).where(db.AssetPrototype.asset_id == a.asset_id)))
        if a.reference_proto_id not in protos:
            a.reference_proto_id = protos[0] if protos else None
        a.updated_ts = utc_now_iso()

    def merge_assets(self, keep_id: str, merge_id: str, actor: str, allow_type_mismatch: bool = False,
                     enqueue: bool = True, op_id: Optional[str] = None) -> Dict[str, Any]:
        """Fold `merge_id` into `keep_id`. Every observation, prototype, defect, defect instance,
        work order, feedback and relation of `merge_id` ends up on `keep_id`; `merge_id` is deleted."""
        if keep_id == merge_id:
            raise ValueError("Cannot merge an asset into itself")
        with self.txn() as s:
            keep, gone = s.get(db.Asset, keep_id), s.get(db.Asset, merge_id)
            if keep is None or gone is None:
                raise KeyError(keep_id if keep is None else merge_id)
            if keep.site_id != gone.site_id:
                raise ValueError("Assets belong to different sites")
            if keep.asset_type != gone.asset_type and not allow_type_mismatch:
                raise ValueError(f"Asset types differ ({keep.asset_type} vs {gone.asset_type})")
            site_id = keep.site_id
            counts = {"observations": 0, "prototypes": 0, "defects_moved": 0, "defects_folded": 0, "instances": 0,
                      "work_orders": 0, "feedback": 0, "relations": 0}
            n_keep = keep.location_samples or 0
            n_gone = gone.location_samples or 0
            for o in s.scalars(select(db.Observation).where(db.Observation.asset_id == merge_id)):
                o.asset_id = keep_id
                counts["observations"] += 1
            for p in s.scalars(select(db.AssetPrototype).where(db.AssetPrototype.asset_id == merge_id)):
                p.asset_id = keep_id
                counts["prototypes"] += 1
            keep_open = {d.defect_type: d for d in s.scalars(select(db.Defect).where(db.Defect.asset_id == keep_id, db.Defect.status != "Resolved"))}
            for d in list(s.scalars(select(db.Defect).where(db.Defect.asset_id == merge_id))):
                target = keep_open.get(d.defect_type) if d.status != "Resolved" else None
                if target is None:
                    d.asset_id = keep_id
                    counts["defects_moved"] += 1
                    continue
                for t in s.scalars(select(db.DefectTrend).where(db.DefectTrend.defect_id == d.defect_id)):
                    t.defect_id = target.defect_id
                target.max_severity = max(target.max_severity, d.max_severity)
                target.first_detected_ts = min(target.first_detected_ts, d.first_detected_ts)
                if d.last_seen_ts > target.last_seen_ts:
                    target.last_seen_ts, target.severity_score = d.last_seen_ts, d.severity_score
                target.notes = "\n".join(x for x in (target.notes, d.notes) if x) or None
                target.updated_ts = utc_now_iso()
                for w in s.scalars(select(db.WorkOrder).where(db.WorkOrder.defect_id == d.defect_id)):
                    w.defect_id = target.defect_id
                s.flush()
                s.delete(d)
                counts["defects_folded"] += 1
            for i in s.scalars(select(db.DefectInstance).where(db.DefectInstance.asset_id == merge_id)):
                i.asset_id = keep_id
                counts["instances"] += 1
            for w in s.scalars(select(db.WorkOrder).where(db.WorkOrder.asset_id == merge_id)):
                w.asset_id = keep_id
                counts["work_orders"] += 1
            for f in s.scalars(select(db.AlertFeedback).where(or_(db.AlertFeedback.asset_id == merge_id, db.AlertFeedback.correct_asset_id == merge_id))):
                if f.asset_id == merge_id:
                    f.asset_id = keep_id
                if f.correct_asset_id == merge_id:
                    f.correct_asset_id = keep_id
                counts["feedback"] += 1
            existing = {(r.parent_id, r.child_id, r.relation) for r in s.scalars(select(db.AssetRelation).where(
                or_(db.AssetRelation.parent_id == keep_id, db.AssetRelation.child_id == keep_id)))}
            for r in list(s.scalars(select(db.AssetRelation).where(or_(db.AssetRelation.parent_id == merge_id, db.AssetRelation.child_id == merge_id)))):
                p = keep_id if r.parent_id == merge_id else r.parent_id
                c = keep_id if r.child_id == merge_id else r.child_id
                if p == c or (p, c, r.relation) in existing:
                    s.delete(r)
                else:
                    r.parent_id, r.child_id = p, c
                    existing.add((p, c, r.relation))
                counts["relations"] += 1
            marker = gone.marker_id
            footprint = gone.footprint
            gl = (gone.latitude, gone.longitude, gone.altitude)
            s.flush()
            s.delete(gone)
            s.flush()
            if keep.marker_id is None and marker is not None:
                keep.marker_id = marker
            if keep.footprint is None and footprint:
                keep.footprint = footprint
            if gl[0] is not None and n_gone:
                if keep.latitude is None or not n_keep:
                    keep.latitude, keep.longitude, keep.altitude = gl
                    keep.location_samples = n_gone
                else:
                    n = n_keep + n_gone
                    keep.latitude = (keep.latitude * n_keep + gl[0] * n_gone) / n
                    keep.longitude = (keep.longitude * n_keep + gl[1] * n_gone) / n
                    keep.altitude = ((keep.altitude or 0.0) * n_keep + (gl[2] or 0.0) * n_gone) / n
                    keep.location_samples = n
            self._refresh_from_observations(s, keep)
            self._apply_revisit(s, keep, None)
            op = {"op_id": op_id or new_id("op"), "op": "merge", "keep_id": keep_id, "merge_id": merge_id, "ts": utc_now_iso(), "actor": actor}
            self._audit(s, actor, "asset.merge", "asset", keep_id, site_id, {**op, "counts": counts})
            if enqueue:
                self._enqueue(s, "asset_op", op["op_id"], site_id, op)
                self._enqueue(s, "asset", keep.asset_id, site_id, _row(keep))
            from memory.store import asset_dict

            return {"asset": asset_dict(keep), "merged_id": merge_id, "counts": counts, "op": op}

    def split_asset(self, asset_id: str, observation_ids: Sequence[str], actor: str, enqueue: bool = True,
                    new_asset_id: Optional[str] = None, op_id: Optional[str] = None) -> Dict[str, Any]:
        """Move `observation_ids` (and what was derived only from them) to a new asset."""
        moved = set(observation_ids)
        if not moved:
            raise ValueError("Nothing to split")
        with self.txn() as s:
            src = s.get(db.Asset, asset_id)
            if src is None:
                raise KeyError(asset_id)
            obs = list(s.scalars(select(db.Observation).where(db.Observation.asset_id == asset_id)))
            ids = {o.observation_id for o in obs}
            if not moved <= ids:
                raise ValueError(f"Observations not on {asset_id}: {sorted(moved - ids)}")
            if moved == ids:
                raise ValueError("A split must leave at least one observation on the original asset")
            now = utc_now_iso()
            dst = db.Asset(
                asset_id=new_asset_id or new_id("asset"), site_id=src.site_id, asset_type=src.asset_type, marker_id=None,
                location_samples=0, first_observed_ts=now, last_checked_ts=now, current_condition=src.current_condition,
                current_is_defect=False, reference_condition=dict(src.reference_condition), criticality=src.criticality,
                revisit_priority_score=0.0, simulated=src.simulated, meta={"split_from": asset_id}, updated_ts=now,
            )
            s.add(dst)
            s.flush()
            counts = {"observations": 0, "prototypes": 0, "defects": 0, "instances": 0}
            for o in obs:
                if o.observation_id in moved:
                    o.asset_id = dst.asset_id
                    counts["observations"] += 1
            for p in s.scalars(select(db.AssetPrototype).where(db.AssetPrototype.asset_id == asset_id)):
                if p.observation_id in moved:
                    p.asset_id = dst.asset_id
                    counts["prototypes"] += 1
            for d in s.scalars(select(db.Defect).where(db.Defect.asset_id == asset_id)):
                trend_obs = {t.observation_id for t in s.scalars(select(db.DefectTrend).where(db.DefectTrend.defect_id == d.defect_id))}
                if trend_obs and trend_obs <= moved:
                    d.asset_id = dst.asset_id
                    counts["defects"] += 1
            for i in s.scalars(select(db.DefectInstance).where(db.DefectInstance.asset_id == asset_id)):
                m_obs = {m.observation_id for m in s.scalars(select(db.DefectMeasurement).where(db.DefectMeasurement.instance_id == i.instance_id))}
                if m_obs and m_obs <= moved:
                    i.asset_id = dst.asset_id
                    counts["instances"] += 1
            s.flush()
            for a in (src, dst):
                self._refresh_from_observations(s, a)
                self._apply_revisit(s, a, None)
            op = {"op_id": op_id or new_id("op"), "op": "split", "asset_id": asset_id, "new_asset_id": dst.asset_id,
                  "observation_ids": sorted(moved), "ts": now, "actor": actor}
            self._audit(s, actor, "asset.split", "asset", asset_id, src.site_id, {**op, "counts": counts})
            if enqueue:
                self._enqueue(s, "asset_op", op["op_id"], src.site_id, op)
                self._enqueue(s, "asset", src.asset_id, src.site_id, _row(src))
                self._enqueue(s, "asset", dst.asset_id, src.site_id, _row(dst))
            from memory.store import asset_dict

            return {"asset": asset_dict(src), "new_asset": asset_dict(dst), "counts": counts, "op": op}

    def apply_asset_ops(self, ops: Iterable[Dict[str, Any]]) -> int:
        """Replay merge/split ops pulled from the cloud (idempotent via the audit log)."""
        done = 0
        with self.read() as s:
            seen = {(a.details or {}).get("op_id") for a in s.scalars(select(db.AuditLog).where(db.AuditLog.action.in_(["asset.merge", "asset.split"])))}
        for op in sorted(ops, key=lambda o: o.get("ts", "")):
            if op.get("op_id") in seen:
                continue
            try:
                if op["op"] == "merge":
                    self.merge_assets(op["keep_id"], op["merge_id"], op.get("actor", "cloud"), True, enqueue=False, op_id=op["op_id"])
                elif op["op"] == "split":
                    self.split_asset(op["asset_id"], op["observation_ids"], op.get("actor", "cloud"), enqueue=False,
                                     new_asset_id=op["new_asset_id"], op_id=op["op_id"])
                done += 1
            except (KeyError, ValueError):
                continue   # already reflected in the pulled records
        return done

    # ── defect instances (2.2 / 2.3) ──────────────────────────────────────
    def _apply_instances(self, s: OrmSession, a: db.Asset, obs: db.Observation,
                         regions: Optional[List[MeasuredRegion]]) -> List[Dict[str, Any]]:
        if regions is None:
            return []
        params = tracking_params(self.config)
        active = list(s.scalars(select(db.DefectInstance).where(db.DefectInstance.asset_id == a.asset_id,
                                                                db.DefectInstance.status == "active")))
        inst_dicts = [_row(i) for i in active]
        regions = [r for r in regions if r.area_norm >= params["min_area_norm"]]
        pairs, new_regions, missed = match_regions(inst_dicts, regions, params["max_distance"], params["iou_weight"])
        touched: List[db.DefectInstance] = []

        def measure(inst: db.DefectInstance, r: MeasuredRegion) -> None:
            inst.polygon, inst.centroid_x, inst.centroid_y = r.polygon, r.centroid[0], r.centroid[1]
            inst.area_norm, inst.area_cm2, inst.length_cm = r.area_norm, r.area_cm2, r.length_cm
            inst.last_seen_ts, inst.missed_visits, inst.updated_ts = obs.observation_ts, 0, obs.observation_ts
            m = db.DefectMeasurement(
                measurement_id=new_id("dms"), instance_id=inst.instance_id, observation_id=obs.observation_id,
                session_id=obs.session_id, ts=obs.observation_ts, area_norm=r.area_norm, area_px=r.area_px,
                area_cm2=r.area_cm2, length_px=r.length_px, length_cm=r.length_cm, polygon=r.polygon,
            )
            s.add(m)
            s.flush()
            self._enqueue(s, "defect_measurement", m.measurement_id, a.site_id, _row(m))
            touched.append(inst)

        for i_idx, r_idx in pairs:
            measure(active[i_idx], regions[r_idx])
        for r_idx in new_regions:
            r = regions[r_idx]
            inst = db.DefectInstance(
                instance_id=new_id("dfi"), asset_id=a.asset_id, defect_type=r.defect_type, status="active",
                polygon=r.polygon, centroid_x=r.centroid[0], centroid_y=r.centroid[1], area_norm=r.area_norm,
                first_seen_ts=obs.observation_ts, last_seen_ts=obs.observation_ts, missed_visits=0, updated_ts=obs.observation_ts,
            )
            s.add(inst)
            s.flush()
            measure(inst, r)
        for i_idx in missed:
            inst = active[i_idx]
            seen_this_session = s.scalar(select(db.DefectMeasurement.measurement_id).where(
                db.DefectMeasurement.instance_id == inst.instance_id, db.DefectMeasurement.session_id == obs.session_id).limit(1))
            last_missed_session = (inst.forecast or {}).get("last_missed_session")
            if seen_this_session or last_missed_session == obs.session_id:
                continue
            inst.missed_visits += 1
            inst.forecast = {**(inst.forecast or {}), "last_missed_session": obs.session_id}
            if inst.missed_visits >= params["resolve_after_missed"]:
                inst.status = "resolved"
            inst.updated_ts = obs.observation_ts
            self._enqueue(s, "defect_instance", inst.instance_id, a.site_id, _row(inst))
        out = []
        for inst in touched:
            ms = [_row(m) for m in s.scalars(select(db.DefectMeasurement).where(
                db.DefectMeasurement.instance_id == inst.instance_id).order_by(db.DefectMeasurement.ts))]
            f = forecast_series(ms, inst.defect_type, self.config)
            inst.forecast = f.to_dict() if f else None
            inst.predicted_critical_ts = f.predicted_critical_ts if f else None
            s.flush()
            self._enqueue(s, "defect_instance", inst.instance_id, a.site_id, _row(inst))
            out.append({**_row(inst), "measurements": len(ms)})
        return out

    def _days_to_critical(self, s: OrmSession, asset_id: str, now: Optional[datetime]) -> Optional[float]:
        rows = s.scalars(select(db.DefectInstance.predicted_critical_ts).where(
            db.DefectInstance.asset_id == asset_id, db.DefectInstance.status == "active",
            db.DefectInstance.predicted_critical_ts.is_not(None)))
        days = [days_until(ts, now or utc_now()) for ts in rows]
        days = [d for d in days if d is not None]
        return min(days) if days else None

    def list_defect_instances(self, asset_id: Optional[str] = None, site_id: Optional[str] = None,
                              status: Optional[str] = None) -> List[Dict[str, Any]]:
        with self.read() as s:
            q = select(db.DefectInstance, db.Asset.site_id, db.Asset.asset_type).join(db.Asset, db.Asset.asset_id == db.DefectInstance.asset_id)
            if asset_id:
                q = q.where(db.DefectInstance.asset_id == asset_id)
            if site_id:
                q = q.where(db.Asset.site_id == site_id)
            if status:
                q = q.where(db.DefectInstance.status == status)
            return [{**_row(i), "site_id": sid, "asset_type": t} for i, sid, t in s.execute(q.order_by(db.DefectInstance.first_seen_ts)).all()]

    def instance_measurements(self, instance_id: str) -> List[Dict[str, Any]]:
        with self.read() as s:
            return [_row(m) for m in s.scalars(select(db.DefectMeasurement).where(
                db.DefectMeasurement.instance_id == instance_id).order_by(db.DefectMeasurement.ts))]

    # ── relations (5.1) ───────────────────────────────────────────────────
    def add_relation(self, parent_id: str, child_id: str, relation: str, actor: str = "system") -> Dict[str, Any]:
        if parent_id == child_id:
            raise ValueError("An asset cannot relate to itself")
        with self.txn() as s:
            p, c = s.get(db.Asset, parent_id), s.get(db.Asset, child_id)
            if p is None or c is None:
                raise KeyError(parent_id if p is None else child_id)
            if p.site_id != c.site_id:
                raise ValueError("Relations are within one site")
            r = s.scalar(select(db.AssetRelation).where(db.AssetRelation.parent_id == parent_id, db.AssetRelation.child_id == child_id,
                                                        db.AssetRelation.relation == relation))
            if r is None:
                r = db.AssetRelation(relation_id=new_id("rel"), site_id=p.site_id, parent_id=parent_id, child_id=child_id,
                                     relation=relation, created_ts=utc_now_iso())
                s.add(r)
                s.flush()
                self._enqueue(s, "asset_relation", r.relation_id, p.site_id, _row(r))
                self._audit(s, actor, "relation.add", "asset_relation", r.relation_id, p.site_id, _row(r))
            return _row(r)

    def remove_relation(self, relation_id: str, actor: str = "system") -> None:
        with self.txn() as s:
            r = s.get(db.AssetRelation, relation_id)
            if r is None:
                raise KeyError(relation_id)
            self._audit(s, actor, "relation.remove", "asset_relation", relation_id, r.site_id, _row(r))
            s.delete(r)

    def list_relations(self, site_id: str) -> List[Dict[str, Any]]:
        with self.read() as s:
            return [_row(r) for r in s.scalars(select(db.AssetRelation).where(db.AssetRelation.site_id == site_id))]

    # ── feedback (5.4) ────────────────────────────────────────────────────
    def add_feedback(self, observation_id: str, username: str, label: str, correct_asset_id: Optional[str] = None,
                     correct_condition: Optional[str] = None, notes: Optional[str] = None) -> Dict[str, Any]:
        if label not in db.FEEDBACK_LABELS:
            raise ValueError(f"label must be one of {db.FEEDBACK_LABELS}")
        with self.txn() as s:
            o = s.get(db.Observation, observation_id)
            if o is None:
                raise KeyError(observation_id)
            a = s.get(db.Asset, o.asset_id)
            f = db.AlertFeedback(feedback_id=new_id("fbk"), observation_id=observation_id, asset_id=o.asset_id, site_id=a.site_id,
                                 session_id=o.session_id, username=username, label=label, correct_asset_id=correct_asset_id,
                                 correct_condition=correct_condition, notes=notes, ts=utc_now_iso(), used_in_training=False)
            s.add(f)
            s.flush()
            self._enqueue(s, "alert_feedback", f.feedback_id, a.site_id, _row(f))
            return _row(f)

    def list_feedback(self, site_id: Optional[str] = None, unused_only: bool = False) -> List[Dict[str, Any]]:
        with self.read() as s:
            q = select(db.AlertFeedback).order_by(db.AlertFeedback.ts)
            if site_id:
                q = q.where(db.AlertFeedback.site_id == site_id)
            if unused_only:
                q = q.where(db.AlertFeedback.used_in_training.is_(False))
            return [_row(f) for f in s.scalars(q)]

    def mark_feedback_used(self, ids: Sequence[str]) -> None:
        with self.txn() as s:
            for f in s.scalars(select(db.AlertFeedback).where(db.AlertFeedback.feedback_id.in_(list(ids)))):
                f.used_in_training = True

    def feedback_stats(self, site_id: Optional[str] = None) -> Dict[str, Any]:
        """False alarms per mission, per session in time order (5.4 verification)."""
        fb = self.list_feedback(site_id)
        per_session: Dict[str, Dict[str, int]] = {}
        for f in fb:
            d = per_session.setdefault(f["session_id"] or "", {l: 0 for l in db.FEEDBACK_LABELS})
            d[f["label"]] += 1
        sessions = {x["session_id"]: x["started_ts"] for x in self.list_sessions(site_id)}
        ordered = sorted(per_session.items(), key=lambda kv: sessions.get(kv[0], ""))
        return {"by_session": [{"session_id": k, **v} for k, v in ordered],
                "totals": {l: sum(v[l] for v in per_session.values()) for l in db.FEEDBACK_LABELS}}

    # ── work orders (6.2) ─────────────────────────────────────────────────
    def _wo_event(self, w: db.WorkOrder, status: str, actor: str, note: Optional[str] = None) -> None:
        w.history = list(w.history or []) + [{"ts": utc_now_iso(), "status": status, "actor": actor, "note": note}]
        w.status = status
        w.updated_ts = utc_now_iso()

    def create_work_order(self, asset_id: str, defect_id: Optional[str], title: str, priority: str, actor: str = "system",
                          s: Optional[OrmSession] = None) -> Dict[str, Any]:
        def go(sess: OrmSession) -> Dict[str, Any]:
            a = sess.get(db.Asset, asset_id)
            if a is None:
                raise KeyError(asset_id)
            if defect_id:
                existing = sess.scalar(select(db.WorkOrder).where(db.WorkOrder.defect_id == defect_id, db.WorkOrder.status != "Verified"))
                if existing is not None:
                    return _row(existing)
            now = utc_now_iso()
            w = db.WorkOrder(work_order_id=new_id("wo"), site_id=a.site_id, asset_id=asset_id, defect_id=defect_id, title=title,
                             status="Open", priority=priority, created_ts=now, updated_ts=now, history=[])
            self._wo_event(w, "Open", actor)
            sess.add(w)
            sess.flush()
            self._enqueue(sess, "work_order", w.work_order_id, a.site_id, _row(w))
            return _row(w)

        if s is not None:
            return go(s)
        with self.txn() as sess:
            return go(sess)

    def update_work_order(self, work_order_id: str, status: str, actor: str, assignee: Optional[str] = None,
                          note: Optional[str] = None) -> Dict[str, Any]:
        if status not in db.WORK_ORDER_STATUSES:
            raise ValueError(f"status must be one of {db.WORK_ORDER_STATUSES}")
        order = {k: i for i, k in enumerate(db.WORK_ORDER_STATUSES)}
        with self.txn() as s:
            w = s.get(db.WorkOrder, work_order_id)
            if w is None:
                raise KeyError(work_order_id)
            if status == "Verified":
                raise ValueError("Verified is set automatically when the next flight confirms the repair")
            if order[status] < order[w.status] and not (w.status == "Fixed" and status in ("Open", "Assigned")):
                raise ValueError(f"Cannot move a work order from {w.status} back to {status}")
            if status == "Assigned" and not (assignee or w.assignee):
                raise ValueError("Assigned requires an assignee")
            if assignee:
                w.assignee = assignee
            if status == "Fixed":
                w.fixed_ts = utc_now_iso()
            self._wo_event(w, status, actor, note)
            self._enqueue(s, "work_order", w.work_order_id, w.site_id, _row(w))
            return _row(w)

    def _verify_work_orders(self, s: OrmSession, a: db.Asset, obs: db.Observation) -> List[Dict[str, Any]]:
        """A Fixed work order is Verified when the next flight sees the asset without the defect."""
        out = []
        for w in s.scalars(select(db.WorkOrder).where(db.WorkOrder.asset_id == a.asset_id, db.WorkOrder.status == "Fixed")):
            if w.fixed_ts and obs.observation_ts <= w.fixed_ts:
                continue
            d = s.get(db.Defect, w.defect_id) if w.defect_id else None
            still = obs.is_defect and (d is None or obs.defect_type == d.defect_type)
            if still:
                self._wo_event(w, "Assigned" if w.assignee else "Open", "audi", f"repair not confirmed by observation {obs.observation_id}")
            else:
                self._wo_event(w, "Verified", "audi", f"confirmed by observation {obs.observation_id}")
                w.verified_ts, w.verified_session_id = obs.observation_ts, obs.session_id
                if d is not None and d.status != "Resolved":
                    d.status, d.resolved_ts, d.updated_ts = "Resolved", obs.observation_ts, obs.observation_ts
                    self._enqueue(s, "defect", d.defect_id, a.site_id, _row(d))
            self._enqueue(s, "work_order", w.work_order_id, a.site_id, _row(w))
            out.append(_row(w))
        return out

    def list_work_orders(self, site_id: Optional[str] = None, status: Optional[str] = None, asset_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self.read() as s:
            q = select(db.WorkOrder).order_by(db.WorkOrder.created_ts.desc())
            if site_id:
                q = q.where(db.WorkOrder.site_id == site_id)
            if status:
                q = q.where(db.WorkOrder.status == status)
            if asset_id:
                q = q.where(db.WorkOrder.asset_id == asset_id)
            return [_row(w) for w in s.scalars(q)]

    def get_work_order(self, work_order_id: str) -> Optional[Dict[str, Any]]:
        with self.read() as s:
            w = s.get(db.WorkOrder, work_order_id)
            return _row(w) if w else None

    # ── drones (6.3 / 6.5) ────────────────────────────────────────────────
    def register_drone(self, name: str, site_ids: Sequence[str], scopes: Sequence[str], model: Optional[str] = None,
                       actor: str = "system") -> Dict[str, Any]:
        """Returns the drone row plus its API key (shown once; only the hash is stored)."""
        key = "audi_" + secrets.token_urlsafe(32)
        with self.txn() as s:
            for sid in site_ids:
                if s.get(db.Site, sid) is None:
                    raise KeyError(f"Unknown site: {sid}")
            d = db.Drone(drone_id=new_id("drone"), name=name, model=model, site_ids=list(site_ids), api_key_hash=hash_api_key(key),
                         api_key_prefix=key[:12], scopes=list(scopes), active=True, created_ts=utc_now_iso())
            s.add(d)
            s.flush()
            self._audit(s, actor, "drone.register", "drone", d.drone_id, None, {"name": name, "sites": list(site_ids), "scopes": list(scopes)})
            row = _row(d)
            row.pop("api_key_hash")
            return {**row, "api_key": key}

    def drone_by_key(self, key: str) -> Optional[Dict[str, Any]]:
        with self.read() as s:
            d = s.scalar(select(db.Drone).where(db.Drone.api_key_prefix == key[:12]))
            if d is None or not d.active or not secrets.compare_digest(d.api_key_hash, hash_api_key(key)):
                return None
            row = _row(d)
            row.pop("api_key_hash")
            return row

    def drone_secret(self, drone_id: str) -> Optional[str]:
        """HMAC secret for signed uploads: sha256(api_key), known to the drone and to the server."""
        with self.read() as s:
            d = s.get(db.Drone, drone_id)
            return d.api_key_hash if d else None

    def touch_drone(self, drone_id: str) -> None:
        with self.txn() as s:
            d = s.get(db.Drone, drone_id)
            if d is not None:
                d.last_seen_ts = utc_now_iso()

    def list_drones(self) -> List[Dict[str, Any]]:
        with self.read() as s:
            out = []
            for d in s.scalars(select(db.Drone).order_by(db.Drone.created_ts)):
                row = _row(d)
                row.pop("api_key_hash")
                out.append(row)
            return out

    def set_drone_active(self, drone_id: str, active: bool, actor: str) -> Dict[str, Any]:
        with self.txn() as s:
            d = s.get(db.Drone, drone_id)
            if d is None:
                raise KeyError(drone_id)
            d.active = bool(active)
            self._audit(s, actor, "drone.active" if active else "drone.revoke", "drone", drone_id, None, {})
            row = _row(d)
            row.pop("api_key_hash")
            return row
