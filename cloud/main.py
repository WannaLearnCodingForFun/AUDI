"""AUDI Cloud DB service (separate process / host).

Stores the authoritative copy of every site's memory. Clients authenticate
with JWT (§6.5); deploy behind TLS or pass a cert/key to serve HTTPS directly.

  GET  /cloud/health                      liveness (no auth) — FR-1 probe
  POST /cloud/auth/token                  username/password → JWT
  GET  /cloud/sites/{site_id}/pull        latest site memory — FR-2
  POST /cloud/push                        idempotent change batch — FR-6/FR-7
  GET  /cloud/stats                       record counts per entity
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy import JSON, Column, String, func, select
from sqlalchemy.orm import declarative_base

from auth.credentials import get_credential
from auth.security import TokenError, create_token, decode_token, hash_password, load_secret, verify_password
from memory.db import make_engine, make_session_factory
from ml.config import cfg, load_config, resolve_path
from ml.schemas import utc_now_iso

logger = logging.getLogger("audi.cloud")

CloudBase = declarative_base()

LWW_FIELDS = {"asset": "updated_ts", "defect": "updated_ts", "session": "started_ts"}
SESSION_STATUS_ORDER = {"active": 0, "interrupted": 1, "ended": 2}


class CloudUser(CloudBase):
    __tablename__ = "cloud_users"
    username = Column(String, primary_key=True)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False)


class CloudRecord(CloudBase):
    __tablename__ = "cloud_records"
    entity = Column(String, primary_key=True)
    entity_id = Column(String, primary_key=True)
    site_id = Column(String, nullable=False, index=True)
    payload = Column(JSON, nullable=False)
    received_ts = Column(String, nullable=False)


class AppliedChange(CloudBase):
    __tablename__ = "applied_changes"
    change_id = Column(String, primary_key=True)
    received_ts = Column(String, nullable=False)


class Change(BaseModel):
    change_id: str
    entity: str
    entity_id: str
    site_id: str
    payload: Dict[str, Any]


class PushRequest(BaseModel):
    changes: List[Change]


def _newer(entity: str, incoming: Dict[str, Any], current: Dict[str, Any]) -> bool:
    if entity == "session":
        a = SESSION_STATUS_ORDER.get(incoming.get("status"), 0)
        b = SESSION_STATUS_ORDER.get(current.get("status"), 0)
        return a >= b
    key = LWW_FIELDS[entity]
    return str(incoming.get(key, "")) > str(current.get(key, ""))


def create_app(config: Optional[Dict[str, Any]] = None, db_url: Optional[str] = None, secret: Optional[str] = None) -> FastAPI:
    config = config or load_config()
    scfg = cfg(config, "cloud", "server")
    url = db_url or scfg["db_url"]
    if url.startswith("sqlite:///") and not url.startswith("sqlite:////"):
        path = resolve_path(url[len("sqlite:///"):])
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{path}"
    engine = make_engine(url, int(cfg(config, "storage", "busy_timeout_ms")))
    CloudBase.metadata.create_all(engine)
    Session = make_session_factory(engine)
    secret = secret or load_secret(scfg["secret_env"], scfg["secret_file"])
    algorithm = cfg(config, "auth", "algorithm")
    minutes = int(scfg["access_token_minutes"])

    def seed_user(username: str, password: str, role: str = "sync") -> None:
        with Session() as s:
            u = s.get(CloudUser, username)
            if u is None:
                s.add(CloudUser(username=username, password_hash=hash_password(password), role=role))
            elif not verify_password(password, u.password_hash):
                u.password_hash = hash_password(password)
            s.commit()

    creds_file = cfg(config, "auth", "credentials_file")
    user = get_credential(cfg(config, "cloud", "username_env"), creds_file)
    pw = get_credential(cfg(config, "cloud", "password_env"), creds_file)
    if user and pw:
        seed_user(user, pw)
    else:
        logger.warning("No cloud sync credentials configured; every authenticated route will reject requests")

    app = FastAPI(title="AUDI Cloud DB", version="2.0.0")
    app.state.seed_user = seed_user
    app.state.engine = engine
    oauth2 = OAuth2PasswordBearer(tokenUrl="/cloud/auth/token")

    def current_user(token: str = Depends(oauth2)) -> Dict[str, Any]:
        try:
            claims = decode_token(token, secret, algorithm)
        except TokenError as exc:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"Invalid token: {exc}") from exc
        with Session() as s:
            if s.get(CloudUser, claims["sub"]) is None:
                raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Unknown user")
        return claims

    @app.get("/cloud/health")
    def health():
        return {"status": "ok", "ts": utc_now_iso()}

    @app.post("/cloud/auth/token")
    def token(form: OAuth2PasswordRequestForm = Depends()):
        with Session() as s:
            u = s.get(CloudUser, form.username)
            if u is None or not verify_password(form.password, u.password_hash):
                raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials")
            return {"access_token": create_token(u.username, u.role, secret, algorithm, minutes), "token_type": "bearer"}

    @app.get("/cloud/sites/{site_id}/pull")
    def pull(site_id: str, _: Dict = Depends(current_user)):
        with Session() as s:
            rows = s.scalars(select(CloudRecord).where(CloudRecord.site_id == site_id))
            out: Dict[str, List[Dict[str, Any]]] = {}
            for r in rows:
                out.setdefault(r.entity, []).append(r.payload)
            return {"site_id": site_id, "records": out}

    @app.post("/cloud/push")
    def push(req: PushRequest, _: Dict = Depends(current_user)):
        applied = duplicates = 0
        acked: List[str] = []
        with Session() as s:
            for ch in req.changes:
                if s.get(AppliedChange, ch.change_id) is not None:
                    duplicates += 1
                    acked.append(ch.change_id)
                    continue
                cur = s.get(CloudRecord, (ch.entity, ch.entity_id))
                if cur is None:
                    s.add(CloudRecord(entity=ch.entity, entity_id=ch.entity_id, site_id=ch.site_id, payload=ch.payload, received_ts=utc_now_iso()))
                elif ch.entity in LWW_FIELDS and _newer(ch.entity, ch.payload, cur.payload):
                    cur.payload = ch.payload
                    cur.received_ts = utc_now_iso()
                s.add(AppliedChange(change_id=ch.change_id, received_ts=utc_now_iso()))
                s.flush()
                applied += 1
                acked.append(ch.change_id)
            s.commit()
        return {"applied": applied, "duplicates": duplicates, "acked": acked}

    @app.get("/cloud/stats")
    def stats(_: Dict = Depends(current_user)):
        with Session() as s:
            rows = s.execute(select(CloudRecord.entity, func.count()).group_by(CloudRecord.entity)).all()
            changes = s.scalar(select(func.count()).select_from(AppliedChange))
            return {"records": {e: int(n) for e, n in rows}, "applied_changes": int(changes)}

    return app


def main() -> None:
    import uvicorn

    config = load_config()
    scfg = cfg(config, "cloud", "server")
    logging.basicConfig(level=cfg(config, "logging", "level"))
    cert = os.environ.get(scfg["ssl_certfile_env"])
    key = os.environ.get(scfg["ssl_keyfile_env"])
    uvicorn.run(
        create_app(config),
        host=scfg["host"],
        port=int(scfg["port"]),
        ssl_certfile=cert if cert and key else None,
        ssl_keyfile=key if cert and key else None,
    )


if __name__ == "__main__":
    main()
