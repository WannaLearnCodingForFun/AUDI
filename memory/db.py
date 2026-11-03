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
WORK_ORDER_STATUSES = ("Open", "Assigned", "Fixed", "Verified")
FEEDBACK_LABELS = ("confirmed", "false_alarm", "wrong_asset")


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
    drone_id = Column(String, index=True)            # 6.3: which registered drone flew it
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
    footprint = Column(JSON)                          # 1.2: [[lat, lon], ...] polygon on the site map
    health_index = Column(Float)                      # 5.5: 0–100


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


class SiteMap(Base):
    """Orthomosaic used as the site's spatial memory backbone (1.2)."""

    __tablename__ = "site_maps"
    map_id = Column(String, primary_key=True)
    site_id = Column(String, ForeignKey("sites.site_id", ondelete="CASCADE"), nullable=False, index=True)
    image_path = Column(String, nullable=False)       # GeoTIFF / PNG path (absolute or repo-relative)
    geotransform = Column(JSON, nullable=False)       # GDAL order, lat/lon degrees
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    active = Column(Boolean, nullable=False, default=True)
    created_ts = Column(String, nullable=False)
    meta = Column(JSON)


class AuditLog(Base):
    """Append-only record of administrative actions (1.4 merge/split, 6.5)."""

    __tablename__ = "audit_log"
    audit_id = Column(String, primary_key=True)
    ts = Column(String, nullable=False, index=True)
    actor = Column(String, nullable=False)
    action = Column(String, nullable=False)
    entity = Column(String, nullable=False)
    entity_id = Column(String, nullable=False)
    site_id = Column(String)
    details = Column(JSON)


class DefectInstance(Base):
    """One physical defect (one crack, one rust patch) tracked across visits (2.2)."""

    __tablename__ = "defect_instances"
    instance_id = Column(String, primary_key=True)
    asset_id = Column(String, ForeignKey("assets.asset_id", ondelete="CASCADE"), nullable=False, index=True)
    defect_type = Column(String, nullable=False)
    status = Column(String, nullable=False, default="active")   # active | resolved
    polygon = Column(JSON, nullable=False)            # normalised [0,1] coords on the asset reference view
    centroid_x = Column(Float, nullable=False)
    centroid_y = Column(Float, nullable=False)
    area_cm2 = Column(Float)
    length_cm = Column(Float)
    area_norm = Column(Float, nullable=False)         # fraction of the asset view covered
    first_seen_ts = Column(String, nullable=False)
    last_seen_ts = Column(String, nullable=False)
    missed_visits = Column(Integer, nullable=False, default=0)
    forecast = Column(JSON)                           # 2.3 fitted growth model
    predicted_critical_ts = Column(String)
    updated_ts = Column(String, nullable=False)


class DefectMeasurement(Base):
    """Timestamped size of a defect instance (2.2 time series)."""

    __tablename__ = "defect_measurements"
    measurement_id = Column(String, primary_key=True)
    instance_id = Column(String, ForeignKey("defect_instances.instance_id", ondelete="CASCADE"), nullable=False, index=True)
    observation_id = Column(String, nullable=False)
    session_id = Column(String, nullable=False)
    ts = Column(String, nullable=False)
    area_norm = Column(Float, nullable=False)
    area_px = Column(Float, nullable=False)
    area_cm2 = Column(Float)
    length_px = Column(Float)
    length_cm = Column(Float)
    polygon = Column(JSON)


class AssetRelation(Base):
    """Knowledge-graph edge parent → child, e.g. inverter → string → panel (5.1)."""

    __tablename__ = "asset_relations"
    __table_args__ = (UniqueConstraint("parent_id", "child_id", "relation", name="uq_relation"),)
    relation_id = Column(String, primary_key=True)
    site_id = Column(String, nullable=False, index=True)
    parent_id = Column(String, ForeignKey("assets.asset_id", ondelete="CASCADE"), nullable=False, index=True)
    child_id = Column(String, ForeignKey("assets.asset_id", ondelete="CASCADE"), nullable=False, index=True)
    relation = Column(String, nullable=False)          # feeds | contains | supports | part_of
    created_ts = Column(String, nullable=False)


class AlertFeedback(Base):
    """Inspector verdict on an alert / observation (5.4)."""

    __tablename__ = "alert_feedback"
    feedback_id = Column(String, primary_key=True)
    observation_id = Column(String, nullable=False, index=True)
    asset_id = Column(String, nullable=False)
    site_id = Column(String, nullable=False, index=True)
    session_id = Column(String)
    username = Column(String, nullable=False)
    label = Column(String, nullable=False)
    correct_asset_id = Column(String)
    correct_condition = Column(String)
    notes = Column(Text)
    ts = Column(String, nullable=False)
    used_in_training = Column(Boolean, nullable=False, default=False)


class WorkOrder(Base):
    """Repair ticket: Open → Assigned → Fixed → Verified (6.2)."""

    __tablename__ = "work_orders"
    work_order_id = Column(String, primary_key=True)
    site_id = Column(String, nullable=False, index=True)
    asset_id = Column(String, ForeignKey("assets.asset_id", ondelete="CASCADE"), nullable=False, index=True)
    defect_id = Column(String)
    title = Column(String, nullable=False)
    status = Column(String, nullable=False)
    priority = Column(String, nullable=False)
    assignee = Column(String)
    created_ts = Column(String, nullable=False)
    updated_ts = Column(String, nullable=False)
    fixed_ts = Column(String)
    verified_ts = Column(String)
    verified_session_id = Column(String)
    history = Column(JSON, nullable=False)


class Drone(Base):
    """Registered drone with a scoped API key (6.3, 6.5)."""

    __tablename__ = "drones"
    drone_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    model = Column(String)
    site_ids = Column(JSON, nullable=False)
    api_key_hash = Column(String, nullable=False)
    api_key_prefix = Column(String, nullable=False, index=True)
    scopes = Column(JSON, nullable=False)
    active = Column(Boolean, nullable=False, default=True)
    created_ts = Column(String, nullable=False)
    last_seen_ts = Column(String)


def migrate_columns(engine: Engine) -> list:
    """Add columns that exist in the models but not yet in an older database file.

    `create_all` creates missing tables but never alters existing ones; every column added by
    an upgrade is nullable, so ALTER TABLE ADD COLUMN is enough. Returns the added columns.
    """
    from sqlalchemy import inspect, text

    insp = inspect(engine)
    added = []
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if not insp.has_table(table.name):
                continue
            have = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name in have:
                    continue
                ddl = col.type.compile(dialect=engine.dialect)
                conn.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" {ddl}'))
                added.append(f"{table.name}.{col.name}")
    return added


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
