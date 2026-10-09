"""SQLAlchemy schema for the Local DB (SQLite, WAL, foreign keys) — D11, SRS §7."""

from __future__ import annotations

from typing import Any, Dict

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

DEFECT_STATUSES = ("Open", "In Progress", "Resolved")
ROLES = ("operator", "maintenance", "admin")


class Site(Base):
    __tablename__ = "sites"
    site_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    latitude = Column(Float)
    longitude = Column(Float)
    created_ts = Column(String, nullable=False)


class User(Base):
    __tablename__ = "users"
    user_id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False)
    email = Column(String)
    active = Column(Boolean, nullable=False, default=True)
    created_ts = Column(String, nullable=False)


class SiteSubscription(Base):
    """Maintenance users designated to receive escalations for a site (FR-19)."""

    __tablename__ = "site_subscriptions"
    user_id = Column(Integer, ForeignKey("users.user_id", ondelete="CASCADE"), primary_key=True)
    site_id = Column(String, ForeignKey("sites.site_id", ondelete="CASCADE"), primary_key=True)


class Session(Base):
    __tablename__ = "sessions"
    session_id = Column(String, primary_key=True)
    site_id = Column(String, ForeignKey("sites.site_id"), nullable=False)
    operator_id = Column(String, nullable=False)
    mission_date = Column(String, nullable=False)
    started_ts = Column(String, nullable=False)
    ended_ts = Column(String)
    status = Column(String, nullable=False)          # active | ended | interrupted
    mode = Column(String, nullable=False)            # online | offline
    assets_processed = Column(Integer, nullable=False, default=0)
    simulated = Column(Boolean, nullable=False, default=False)
    start_sync = Column(JSON)
    end_sync = Column(JSON)
    report = Column(JSON)


class Asset(Base):
    __tablename__ = "assets"
    __table_args__ = (UniqueConstraint("site_id", "marker_id", name="uq_site_marker"),)
    asset_id = Column(String, primary_key=True)
    site_id = Column(String, ForeignKey("sites.site_id"), nullable=False, index=True)
    asset_type = Column(String, nullable=False, index=True)
    marker_id = Column(Integer)
    latitude = Column(Float)
    longitude = Column(Float)
    altitude = Column(Float)
    location_samples = Column(Integer, nullable=False, default=0)
    first_observed_ts = Column(String, nullable=False)
    last_checked_ts = Column(String, nullable=False)
    current_condition = Column(String, nullable=False)
    current_is_defect = Column(Boolean, nullable=False, default=False)
    reference_condition = Column(JSON, nullable=False)   # last CONFIRMED condition (label, scores, …)
    reference_proto_id = Column(Integer)
    criticality = Column(Float, nullable=False)
    revisit_priority_score = Column(Float, nullable=False, default=0.0)
    revisit_components = Column(JSON)
    simulated = Column(Boolean, nullable=False, default=False)
    meta = Column(JSON)
    updated_ts = Column(String, nullable=False)


class Observation(Base):
    __tablename__ = "observations"
    observation_id = Column(String, primary_key=True)
    asset_id = Column(String, ForeignKey("assets.asset_id", ondelete="CASCADE"), nullable=False, index=True)
    session_id = Column(String, ForeignKey("sessions.session_id"), nullable=False, index=True)
    observation_ts = Column(String, nullable=False)
    condition_description = Column(String, nullable=False)
    condition_scores = Column(JSON, nullable=False)
    is_defect = Column(Boolean, nullable=False)
    defect_type = Column(String)
    image_reference = Column(String)
    heatmap_reference = Column(String)
    severity_score = Column(Float, nullable=False)
    severity_level = Column(String, nullable=False)
    change_magnitude = Column(Float, nullable=False)
    condition_delta = Column(Float)
    visual_delta = Column(Float)
    observation_confidence = Column(Float, nullable=False)
    detector_confidence = Column(Float, nullable=False)
    classifier_confidence = Column(Float, nullable=False)
    match_stage = Column(String, nullable=False)
    match_confidence = Column(Float, nullable=False)
    status = Column(String, nullable=False)           # new | confirmed | changed
    latitude = Column(Float)
    longitude = Column(Float)
    altitude = Column(Float)
    bbox = Column(JSON)
    simulated = Column(Boolean, nullable=False, default=False)
    meta = Column(JSON)


class Defect(Base):
    __tablename__ = "defects"
    defect_id = Column(String, primary_key=True)
    asset_id = Column(String, ForeignKey("assets.asset_id", ondelete="CASCADE"), nullable=False, index=True)
    defect_type = Column(String, nullable=False)
    severity_score = Column(Float, nullable=False)
    max_severity = Column(Float, nullable=False)
    first_detected_ts = Column(String, nullable=False)
    last_seen_ts = Column(String, nullable=False)
    status = Column(String, nullable=False)
    normal_streak = Column(Integer, nullable=False, default=0)
    last_normal_session_id = Column(String)
    resolution_suggested = Column(Boolean, nullable=False, default=False)
    notes = Column(Text)
    updated_ts = Column(String, nullable=False)
    resolved_ts = Column(String)


class DefectTrend(Base):
    __tablename__ = "defect_trend"
    trend_id = Column(String, primary_key=True)
    defect_id = Column(String, ForeignKey("defects.defect_id", ondelete="CASCADE"), nullable=False, index=True)
    observation_id = Column(String, nullable=False)
    session_id = Column(String, nullable=False)
    ts = Column(String, nullable=False)
    severity_score = Column(Float, nullable=False)


class Notification(Base):
    __tablename__ = "notifications"
    notification_id = Column(String, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True)
    site_id = Column(String, nullable=False)
    session_id = Column(String)
    asset_id = Column(String)
    defect_id = Column(String)
    severity_score = Column(Float, nullable=False)
    message = Column(Text, nullable=False)
    created_ts = Column(String, nullable=False)
    read = Column(Boolean, nullable=False, default=False)
    email_status = Column(String)


class SyncOutbox(Base):
    __tablename__ = "sync_outbox"
    seq = Column(Integer, primary_key=True, autoincrement=True)
    change_id = Column(String, unique=True, nullable=False)
    entity = Column(String, nullable=False)
    entity_id = Column(String, nullable=False)
    site_id = Column(String, nullable=False)
    payload = Column(JSON, nullable=False)
    created_ts = Column(String, nullable=False)
    sent_ts = Column(String)
    attempts = Column(Integer, nullable=False, default=0)


class AssetPrototype(Base):
    __tablename__ = "asset_prototypes"
    proto_id = Column(Integer, primary_key=True, autoincrement=True)
    proto_uid = Column(String, unique=True, nullable=False)
    asset_id = Column(String, ForeignKey("assets.asset_id", ondelete="CASCADE"), nullable=False, index=True)
    observation_id = Column(String)
    embedding = Column(LargeBinary, nullable=False)
    dim = Column(Integer, nullable=False)
    image_reference = Column(String, nullable=False)
    created_ts = Column(String, nullable=False)


class ConfigVersion(Base):
    __tablename__ = "config_versions"
    version = Column(Integer, primary_key=True, autoincrement=True)
    created_ts = Column(String, nullable=False)
    username = Column(String, nullable=False)
    changes = Column(JSON, nullable=False)
    snapshot = Column(JSON, nullable=False)


class Meta(Base):
    __tablename__ = "meta"
    key = Column(String, primary_key=True)
    value = Column(JSON)


def make_engine(db_url: str, busy_timeout_ms: int) -> Engine:
    connect_args: Dict[str, Any] = {}
    if db_url.startswith("sqlite"):
        connect_args = {"check_same_thread": False, "timeout": busy_timeout_ms / 1000.0}
    engine = create_engine(db_url, connect_args=connect_args, future=True)

    if db_url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _):
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
            cur.close()

    return engine


def make_session_factory(engine: Engine) -> sessionmaker:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)
