import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, Index,
    UniqueConstraint, text,
)


def sa_false() -> text:
    return text("false")
from sqlalchemy.dialects.postgresql import JSONB
from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return uuid.uuid4().hex[:12]


class DBApiKey(Base):
    __tablename__ = "api_keys"

    id = Column(String(32), primary_key=True, default=lambda: f"key-{_uuid()}")
    name = Column(String(255), nullable=False)
    key = Column(String(255), nullable=True, unique=True, index=True)
    key_mask = Column(String(64), nullable=False, default="", server_default="")
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    request_count = Column(Integer, default=0, nullable=False)
    key_hash = Column(String(64), nullable=False, index=True, unique=True)
    combo_id = Column(String(32), nullable=True, index=True)
    rate_limit_per_minute = Column(Integer, default=60, nullable=False)
    # Monthly token cap per key (0 = unlimited); aggregates request_logs.
    monthly_token_budget = Column(Integer, nullable=False, server_default=text("0"), default=0)


class DBUser(Base):
    __tablename__ = "users"

    id = Column(String(32), primary_key=True, default=lambda: f"user-{_uuid()}")
    email = Column(String(255), nullable=False, unique=True, index=True)
    name = Column(String(255), nullable=False)
    password_hash = Column(String(255), nullable=True)
    role = Column(String(16), nullable=False, default="user")
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    last_login_at = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    api_key_count = Column(Integer, default=0, nullable=False)
    request_count = Column(Integer, default=0, nullable=False)
    password_changed = Column(Boolean, default=False, nullable=True)


class DBSettings(Base):
    __tablename__ = "settings"

    key = Column(String(255), primary_key=True)
    value = Column(Text, nullable=False)


class DBAuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(String(32), primary_key=True, default=lambda: f"audit-{_uuid()}")
    timestamp = Column(DateTime(timezone=True), default=_utcnow, nullable=False, index=True)
    action = Column(String(255), nullable=False, index=True)
    actor = Column(String(255), nullable=False)
    target = Column(String(255), nullable=False)
    details = Column(Text, nullable=False, default="")
    ip_address = Column(String(64), nullable=False, default="")

    __table_args__ = (
        Index("ix_audit_logs_timestamp_action", "timestamp", "action"),
    )


class DBRequestLog(Base):
    __tablename__ = "request_logs"

    id = Column(String(32), primary_key=True, default=lambda: f"log-{_uuid()}")
    timestamp = Column(DateTime(timezone=True), default=_utcnow, nullable=False, index=True)
    method = Column(String(16), nullable=False)
    path = Column(String(512), nullable=False)
    status_code = Column(Integer, nullable=False)
    model = Column(String(128), nullable=False, default="")
    duration_ms = Column(Integer, nullable=False, default=0)
    user_agent = Column(String(512), nullable=True)
    ip_address = Column(String(64), nullable=True)
    api_key_id = Column(String(32), nullable=True)
    tokens_used = Column(Integer, nullable=True)
    # Redacted + truncated snapshots for replay/debugging of failed requests.
    request_payload = Column(JSONB, nullable=True)
    response_payload = Column(Text, nullable=True)

    __table_args__ = (
        Index("ix_request_logs_timestamp_status", "timestamp", "status_code"),
        Index("ix_request_logs_model", "model"),
    )


class DBRouterCombo(Base):
    __tablename__ = "router_combos"

    id = Column(String(32), primary_key=True, default=lambda: f"combo-{_uuid()}")
    # The name is a routable model id (resolve_combo_ref exact-matches it) —
    # uniqueness is a routing invariant, not a cosmetic preference.
    name = Column(String(255), nullable=False)
    router_type = Column(String(50), nullable=False, index=True)
    models = Column(JSONB, nullable=False, default=list)
    is_active = Column(Boolean, default=False, nullable=False)
    # Monotonic optimistic-concurrency counter; incremented on every update
    # so concurrent edits are detectable (If-Match / expected_version -> 409).
    version = Column(Integer, nullable=False, default=1, server_default=text("1"))
    # How a request bound to this combo is executed: 'agentic' runs the
    # multi-iteration tool loop, 'group' runs a single-shot strategy over
    # models[]. agentic_enabled mirrors routing_mode == 'agentic' for
    # pre-011 clients.
    routing_mode = Column(String(20), default="group", nullable=False)
    routing_strategy = Column(String(20), default="fallback", nullable=False)
    judge_model = Column(String(255), nullable=True)
    agentic_enabled = Column(Boolean, default=False, nullable=False)
    max_iterations = Column(Integer, default=5, nullable=False)
    tool_calling = Column(Boolean, default=True, nullable=False)
    deterministic = Column(Boolean, default=True, nullable=False)
    loop_strategy = Column(String(20), default="sequential", nullable=False)
    # Momentum-lock override: true restores legacy per-iteration round-robin.
    load_spread = Column(Boolean, nullable=False, server_default=sa_false(), default=False)
    compression_target_tokens = Column(Integer, default=4000, nullable=False)
    monthly_token_budget = Column(Integer, default=0, nullable=False)
    tools_allowed = Column(JSONB, default=list, nullable=False)
    strategy_preset = Column(String(50), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)

    __table_args__ = (
        Index("ix_router_combos_type_active", "router_type", "is_active"),
        UniqueConstraint("name", name="uq_router_combos_name"),
        Index(
            "uq_router_combos_one_active_per_type",
            "router_type",
            unique=True,
            postgresql_where=text("is_active"),
        ),
    )

    @property
    def is_agentic(self) -> bool:
        """True when this combo routes through the agentic loop engine."""
        return (self.routing_mode or "group") == "agentic"

    @property
    def group_strategy(self) -> str:
        """Group routing strategy; falls back to ordered failover."""
        return self.routing_strategy or "fallback"


class DBTokenBudget(Base):
    __tablename__ = "token_budgets"

    id = Column(String(32), primary_key=True, default=lambda: f"budget-{_uuid()}")
    combo_id = Column(String(32), nullable=False, index=True)
    month_year = Column(String(7), nullable=False, index=True)  # "YYYY-MM"
    tokens_used = Column(Integer, default=0, nullable=False)
    tokens_budget = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)

    __table_args__ = (
        Index("ix_token_budgets_combo_month", "combo_id", "month_year", unique=True),
        Index("ix_token_budgets_month", "month_year"),
    )


class DBAgenticSession(Base):
    __tablename__ = "agentic_sessions"

    id = Column(String(32), primary_key=True, default=lambda: f"asess-{_uuid()}")
    session_id = Column(String(64), nullable=False, index=True, unique=True)
    combo_id = Column(String(32), nullable=False, index=True)
    messages = Column(JSONB, nullable=False, default=list)
    iteration_index = Column(Integer, default=0, nullable=False)
    step_history = Column(JSONB, nullable=False, default=list)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)
    closed_at = Column(DateTime(timezone=True), nullable=True)


class DBAgenticAnalytics(Base):
    __tablename__ = "agentic_analytics"

    id = Column(String(32), primary_key=True, default=lambda: f"aanal-{_uuid()}")
    combo_id = Column(String(32), nullable=False, index=True)
    month_year = Column(String(7), nullable=False, index=True)
    requests_count = Column(Integer, default=0, nullable=False)
    total_iterations = Column(Integer, default=0, nullable=False)
    total_tokens = Column(Integer, default=0, nullable=False)
    estimated_cost_usd = Column(Float, default=0.0, nullable=False)
    stop_reason_counts = Column(JSONB, nullable=False, default=dict)
    tool_call_counts = Column(JSONB, nullable=False, default=dict)
    model_stats = Column(JSONB, nullable=False, default=list)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)

    __table_args__ = (
        Index("ix_agentic_analytics_combo_month", "combo_id", "month_year", unique=True),
    )


class DBBudgetAlert(Base):
    """Historical budget alerts for analytics."""
    __tablename__ = "budget_alerts"

    id = Column(String(32), primary_key=True, default=lambda: f"alert-{_uuid()}")
    combo_id = Column(String(32), nullable=False, index=True)
    month_year = Column(String(7), nullable=False, index=True)
    level = Column(String(20), nullable=False)
    percentage = Column(Integer, nullable=False)
    tokens_used = Column(Integer, nullable=False)
    tokens_budget = Column(Integer, nullable=False)
    triggered_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    resolved_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_budget_alerts_combo_month_level", "combo_id", "month_year", "level"),
    )
