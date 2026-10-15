import json
import uuid
import hashlib
from datetime import datetime, timezone, timedelta
from sqlalchemy import select, update as sa_update, func, update, delete, and_, Integer
from sqlalchemy.ext.asyncio import AsyncSession
from app.db_models import DBApiKey, DBUser, DBSettings, DBAuditLog, DBRequestLog, DBRouterCombo, DBTokenBudget, DBBudgetAlert, DBAgenticSession, DBAgenticAnalytics, _uuid
from app.models import ROUTING_MODES, ROUTING_STRATEGIES


# ── API Keys ─────────────────────────────────────────────────────────────────

def hash_api_key(key: str) -> str:
    """Hash an API key using SHA-256 for storage."""
    return hashlib.sha256(key.encode()).hexdigest()


def mask_api_key(key: str) -> str:
    """Non-reversible display mask: a short prefix plus fixed asterisks."""
    if not key:
        return ""
    return f"{key[:11]}******"


async def create_api_key(db: AsyncSession, name: str) -> tuple[DBApiKey, str]:
    """Create an API key. The raw key is returned once and never persisted.

    Returns (row, raw_key); the caller must surface raw_key to the client
    immediately — only key_hash and key_mask are stored.
    """
    raw_key = f"sk-{uuid.uuid4().hex}"
    api_key = DBApiKey(
        name=name,
        key=None,
        key_mask=mask_api_key(raw_key),
        key_hash=hash_api_key(raw_key),
    )
    db.add(api_key)
    await db.flush()
    return api_key, raw_key


async def verify_api_key(db: AsyncSession, raw_key: str) -> DBApiKey | None:
    """Verify an API key by comparing its hash against the database."""
    key_hash = hash_api_key(raw_key)
    result = await db.execute(
        select(DBApiKey).where(DBApiKey.key_hash == key_hash, DBApiKey.is_active == True)
    )
    return result.scalar_one_or_none()


async def get_api_keys(db: AsyncSession) -> list[DBApiKey]:
    result = await db.execute(select(DBApiKey).order_by(DBApiKey.created_at.desc()))
    return list(result.scalars().all())


async def revoke_api_key(db: AsyncSession, key_id: str) -> bool:
    result = await db.execute(select(DBApiKey).where(DBApiKey.id == key_id))
    api_key = result.scalar_one_or_none()
    if not api_key:
        return False
    api_key.is_active = False
    await db.flush()
    return True


async def record_api_key_usage(db: AsyncSession, key_id: str) -> None:
    result = await db.execute(select(DBApiKey).where(DBApiKey.id == key_id))
    api_key = result.scalar_one_or_none()
    if api_key:
        api_key.last_used_at = datetime.now(timezone.utc)
        api_key.request_count += 1
        await db.flush()


# ── Users ────────────────────────────────────────────────────────────────────

async def create_user(db: AsyncSession, email: str, name: str, role: str = "user") -> DBUser:
    user = DBUser(email=email, name=name, role=role)
    db.add(user)
    await db.flush()
    return user


async def get_users(db: AsyncSession) -> list[DBUser]:
    result = await db.execute(select(DBUser).order_by(DBUser.created_at.desc()))
    return list(result.scalars().all())


async def get_user(db: AsyncSession, user_id: str) -> DBUser | None:
    result = await db.execute(select(DBUser).where(DBUser.id == user_id))
    return result.scalar_one_or_none()


async def update_user(db: AsyncSession, user_id: str, updates: dict) -> DBUser | None:
    result = await db.execute(select(DBUser).where(DBUser.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        return None
    for k, v in updates.items():
        if v is not None and hasattr(user, k):
            setattr(user, k, v)
    await db.flush()
    return user


async def delete_user(db: AsyncSession, user_id: str) -> bool:
    result = await db.execute(select(DBUser).where(DBUser.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        return False
    await db.delete(user)
    await db.flush()
    return True


# ── Settings ─────────────────────────────────────────────────────────────────

DEFAULT_SETTINGS = {
    "rate_limit_per_minute": 60,
    "rate_limit_per_hour": 1000,
    "max_tokens_per_request": 4096,
    "allowed_models": [
        "claude-sonnet-5", "gpt-5.2-high", "o3-2025-04-16", "gemini-3.1-pro-preview", "qwen3.7-max",
        "gpt-5.5-instant", "deepseek-v4-pro-thinking", "mistral-large-3", "kimi-k2.6", "grok-4.3/text",
        "gpt-5.4-mini-high", "gpt-5.4-nano-high", "gemini-3.6-flash", "gemini-3.1-flash-lite", "mistral-small-2603",
        "qwen3.7-plus", "mimo-v2.5", "glm-5.1", "minimax-m3", "claude-haiku-4-5-20251001",
    ],
    "maintenance_mode": False,
    "cors_origins": ["*"],
}


async def get_settings(db: AsyncSession) -> dict:
    result = await db.execute(select(DBSettings))
    rows = {r.key: r.value for r in result.scalars().all()}
    if not rows:
        # Seed defaults
        for k, v in DEFAULT_SETTINGS.items():
            db.add(DBSettings(key=k, value=json.dumps(v)))
        await db.flush()
        return DEFAULT_SETTINGS.copy()
    # Reconstruct dict
    merged = {}
    for k, default in DEFAULT_SETTINGS.items():
        raw = rows.get(k)
        if raw is not None:
            merged[k] = json.loads(raw)
        else:
            merged[k] = default
            db.add(DBSettings(key=k, value=json.dumps(default)))
    await db.flush()
    return merged


async def update_settings(db: AsyncSession, updates: dict) -> dict:
    for k, v in updates.items():
        result = await db.execute(select(DBSettings).where(DBSettings.key == k))
        row = result.scalar_one_or_none()
        if row:
            row.value = json.dumps(v)
        else:
            db.add(DBSettings(key=k, value=json.dumps(v)))
    await db.flush()
    return await get_settings(db)


# ── Request Logging ──────────────────────────────────────────────────────────

async def get_key_token_usage(db: AsyncSession, key_id: str, months_back: int = 0) -> int:
    """Sum of tokens billed against an API key this month (0 when unknown)."""
    result = await db.execute(text(
        "SELECT COALESCE(SUM(tokens_used), 0) FROM request_logs "
        "WHERE api_key_id = :id AND tokens_used IS NOT NULL "
        "AND timestamp >= date_trunc('month', now())"
    ), {"id": key_id})
    return int(result.scalar() or 0)


async def log_request(
    db: AsyncSession,
    method: str,
    path: str,
    status_code: int,
    model: str,
    duration_ms: int,
    user_agent: str | None = None,
    ip_address: str | None = None,
    api_key_id: str | None = None,
    tokens_used: int | None = None,
    request_payload: dict | None = None,
    response_payload: str | None = None,
) -> DBRequestLog:
    entry = DBRequestLog(
        method=method,
        path=path,
        status_code=status_code,
        model=model,
        duration_ms=duration_ms,
        user_agent=user_agent,
        ip_address=ip_address,
        api_key_id=api_key_id,
        tokens_used=tokens_used,
        request_payload=request_payload,
        response_payload=response_payload,
    )
    db.add(entry)
    await db.flush()
    return entry


async def get_request_logs(
    db: AsyncSession, limit: int = 100, offset: int = 0
) -> tuple[list[DBRequestLog], int]:
    total_result = await db.execute(select(func.count(DBRequestLog.id)))
    total = total_result.scalar() or 0

    result = await db.execute(
        select(DBRequestLog)
        .order_by(DBRequestLog.timestamp.desc())
        .offset(offset)
        .limit(limit)
    )
    return list(result.scalars().all()), total


async def get_request_stats(db: AsyncSession) -> dict:
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    hour_start = now.replace(minute=0, second=0, microsecond=0)

    # Today's stats
    today_result = await db.execute(
        select(
            func.count(DBRequestLog.id),
            func.coalesce(func.sum(DBRequestLog.duration_ms), 0),
            func.coalesce(
                func.sum(func.cast(DBRequestLog.status_code >= 400, Integer)), 0
            ),
            func.coalesce(func.sum(DBRequestLog.tokens_used), 0),
        ).where(DBRequestLog.timestamp >= today_start)
    )
    row = today_result.one()
    today_total = row[0] or 0
    today_duration = row[1] or 0
    today_errors = row[2] or 0
    today_tokens = row[3] or 0

    # This hour's stats
    hour_result = await db.execute(
        select(func.count(DBRequestLog.id)).where(
            DBRequestLog.timestamp >= hour_start
        )
    )
    hour_total = hour_result.scalar() or 0

    avg_duration = today_duration // max(today_total, 1)
    error_rate = (today_errors / max(today_total, 1)) * 100

    # Count active users and api keys
    users_result = await db.execute(
        select(func.count(DBUser.id)).where(DBUser.is_active == True)
    )
    active_users = users_result.scalar() or 0

    keys_result = await db.execute(
        select(func.count(DBApiKey.id)).where(DBApiKey.is_active == True)
    )
    active_keys = keys_result.scalar() or 0

    return {
        "total_requests": today_total,
        "active_users": active_users,
        "active_api_keys": active_keys,
        "avg_response_time": avg_duration,
        "error_rate": round(error_rate, 2),
        "requests_today": today_total,
        "requests_this_hour": hour_total,
        "token_usage_today": int(today_tokens),
    }


async def get_chart_data(db: AsyncSession, hours: int = 24) -> list[dict]:
    now = datetime.now(timezone.utc)
    data = []

    for i in range(hours - 1, -1, -1):
        h = now - timedelta(hours=i)
        h_start = h.replace(minute=0, second=0, microsecond=0)
        h_end = h_start + timedelta(hours=1)
        time_label = h.strftime("%H:%M")

        result = await db.execute(
            select(
                func.count(DBRequestLog.id),
                func.coalesce(func.sum(DBRequestLog.duration_ms), 0),
                func.coalesce(
                    func.sum(func.cast(DBRequestLog.status_code >= 400, Integer)), 0
                ),
            ).where(
                and_(
                    DBRequestLog.timestamp >= h_start,
                    DBRequestLog.timestamp < h_end,
                )
            )
        )
        row = result.one()
        total = row[0] or 0
        duration = row[1] or 0
        errors = row[2] or 0
        avg_dur = duration // max(total, 1)

        data.append({
            "time": time_label,
            "requests": total,
            "errors": errors,
            "avg_duration": avg_dur,
        })

    return data


# ── Audit Log ────────────────────────────────────────────────────────────────

async def audit(
    db: AsyncSession,
    action: str,
    actor: str,
    target: str,
    details: str,
    ip_address: str = "",
) -> DBAuditLog:
    entry = DBAuditLog(
        action=action,
        actor=actor,
        target=target,
        details=details,
        ip_address=ip_address,
    )
    db.add(entry)
    await db.flush()
    return entry


async def get_audit_logs(
    db: AsyncSession, limit: int = 100, offset: int = 0
) -> tuple[list[DBAuditLog], int]:
    total_result = await db.execute(select(func.count(DBAuditLog.id)))
    total = total_result.scalar() or 0

    result = await db.execute(
        select(DBAuditLog)
        .order_by(DBAuditLog.timestamp.desc())
        .offset(offset)
        .limit(limit)
    )
    return list(result.scalars().all()), total


# ── Router Combos ──────────────────────────────────────────────────────────

class ComboConflict(Exception):
    """A write violated a combo invariant (unique name, one-active-per-type,
    or a stale optimistic-concurrency version). Routes translate this to 409."""

    def __init__(self, message: str, current_version: int | None = None):
        super().__init__(message)
        self.current_version = current_version


async def create_router_combo(
    db: AsyncSession, name: str, router_type: str, models: list[dict],
    routing_mode: str = "group", routing_strategy: str = "fallback",
    judge_model: str | None = None,
    max_iterations: int = 5,
    tool_calling: bool = True, deterministic: bool = True,
    loop_strategy: str = "sequential", compression_target_tokens: int = 4000,
    load_spread: bool = False,
    monthly_token_budget: int = 0, tools_allowed: list[str] | None = None,
    strategy_preset: str | None = None,
) -> DBRouterCombo:
    mode = routing_mode if routing_mode in ROUTING_MODES else "group"
    combo = DBRouterCombo(
        name=name, router_type=router_type, models=models,
        routing_mode=mode,
        routing_strategy=(
            routing_strategy if routing_strategy in ROUTING_STRATEGIES else "fallback"
        ),
        judge_model=judge_model or None,
        # Legacy mirror: pre-011 clients and the backoffice read this flag.
        agentic_enabled=(mode == "agentic"),
        max_iterations=max_iterations,
        tool_calling=tool_calling, deterministic=deterministic,
        loop_strategy=loop_strategy,
        load_spread=load_spread, compression_target_tokens=compression_target_tokens,
        monthly_token_budget=monthly_token_budget,
    )
    if tools_allowed is not None:
        combo.tools_allowed = tools_allowed
    if strategy_preset is not None:
        combo.strategy_preset = strategy_preset
    db.add(combo)
    await db.flush()
    return combo


async def get_router_combos(db: AsyncSession, router_type: str | None = None) -> list[DBRouterCombo]:
    query = select(DBRouterCombo).order_by(DBRouterCombo.created_at.desc())
    if router_type:
        query = query.where(DBRouterCombo.router_type == router_type)
    result = await db.execute(query)
    return list(result.scalars().all())


async def get_router_combo(db: AsyncSession, combo_id: str) -> DBRouterCombo | None:
    result = await db.execute(select(DBRouterCombo).where(DBRouterCombo.id == combo_id))
    return result.scalar_one_or_none()


async def update_router_combo(
    db: AsyncSession, combo_id: str, expected_version: int | None = None,
    name: str | None = None, models: list[dict] | None = None,
    routing_mode: str | None = None, routing_strategy: str | None = None,
    judge_model: str | None = None, clear_judge_model: bool = False,
    max_iterations: int | None = None,
    tool_calling: bool | None = None, deterministic: bool | None = None,
    loop_strategy: str | None = None, compression_target_tokens: int | None = None,
    load_spread: bool | None = None,
    monthly_token_budget: int | None = None, tools_allowed: list[str] | None = None,
    strategy_preset: str | None = None,
) -> DBRouterCombo | None:
    combo = await get_router_combo(db, combo_id)
    if not combo:
        return None
    if expected_version is not None and combo.version != expected_version:
        raise ComboConflict(
            f"Combo was modified concurrently (stored version {combo.version}, "
            f"caller expected {expected_version}).",
            current_version=combo.version,
        )
    if name is not None:
        combo.name = name
    if models is not None:
        combo.models = models
    if routing_mode is not None and routing_mode in ROUTING_MODES:
        combo.routing_mode = routing_mode
        # Keep the legacy mirror in lockstep with the single source of truth.
        combo.agentic_enabled = routing_mode == "agentic"
    if routing_strategy is not None and routing_strategy in ROUTING_STRATEGIES:
        combo.routing_strategy = routing_strategy
    if clear_judge_model:
        combo.judge_model = None
    elif judge_model is not None:
        combo.judge_model = judge_model or None
    if max_iterations is not None:
        combo.max_iterations = max_iterations
    if tool_calling is not None:
        combo.tool_calling = tool_calling
    if deterministic is not None:
        combo.deterministic = deterministic
    if loop_strategy is not None:
        combo.loop_strategy = loop_strategy
    if load_spread is not None:
        combo.load_spread = load_spread
    if compression_target_tokens is not None:
        combo.compression_target_tokens = compression_target_tokens
    if monthly_token_budget is not None:
        combo.monthly_token_budget = monthly_token_budget
    if tools_allowed is not None:
        setattr(combo, "tools_allowed", tools_allowed)
    if strategy_preset is not None:
        setattr(combo, "strategy_preset", strategy_preset)
    combo.updated_at = datetime.now(timezone.utc)
    # SQL-side increment so two concurrent updates can never both land on the
    # same version number.
    combo.version = DBRouterCombo.version + 1
    await db.flush()
    return combo


async def delete_router_combo(db: AsyncSession, combo_id: str) -> bool:
    combo = await get_router_combo(db, combo_id)
    if not combo:
        return False
    await db.delete(combo)
    await db.flush()
    return True


async def activate_router_combo(db: AsyncSession, combo_id: str) -> DBRouterCombo | None:
    """Activate a combo, deactivating all others for the same router_type."""
    combo = await get_router_combo(db, combo_id)
    if not combo:
        return None
    await db.execute(
        sa_update(DBRouterCombo)
        .where(DBRouterCombo.router_type == combo.router_type, DBRouterCombo.is_active == True)
        .values(is_active=False)
    )
    combo.is_active = True
    combo.updated_at = datetime.now(timezone.utc)
    await db.flush()
    return combo


async def deactivate_router_combo(db: AsyncSession, combo_id: str) -> DBRouterCombo | None:
    combo = await get_router_combo(db, combo_id)
    if not combo:
        return None
    combo.is_active = False
    combo.updated_at = datetime.now(timezone.utc)
    await db.flush()
    return combo


async def get_active_combo(db: AsyncSession, router_type: str) -> DBRouterCombo | None:
    """Get the active combo for a router type."""
    result = await db.execute(
        select(DBRouterCombo).where(
            DBRouterCombo.router_type == router_type,
            DBRouterCombo.is_active == True,
        )
    )
    return result.scalar_one_or_none()


# ── Combo Tools Configuration ───────────────────────────────────────────────

async def set_combo_tools(db: AsyncSession, combo_id: str, tools: list[str]) -> bool:
    """Set the allowed tools for a combo (stored in tools_allowed column)."""
    combo = await get_router_combo(db, combo_id)
    if not combo:
        return False
    combo.tools_allowed = tools
    combo.updated_at = datetime.now(timezone.utc)
    await db.flush()
    return True


async def get_combo_tools(db: AsyncSession, combo_id: str) -> list[str]:
    """Get the allowed tools for a combo from tools_allowed column (fallback: models JSONB)."""
    combo = await get_router_combo(db, combo_id)
    if not combo:
        return []
    if hasattr(combo, "tools_allowed") and combo.tools_allowed:
        return list(combo.tools_allowed)
    if combo.models:
        for model in combo.models:
            if isinstance(model, dict) and "tools_allowed" in model:
                return model["tools_allowed"]
    return []


# ── Token Budgets ──────────────────────────────────────────────────────

async def get_or_create_monthly_budget(
    db: AsyncSession, combo_id: str, month_year: str, budget: int = 0,
) -> DBTokenBudget:
    """Get or create a monthly token budget entry (race-safe via ON CONFLICT)."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    stmt = (
        pg_insert(DBTokenBudget.__table__)
        .values(
            id=f"budget-{uuid.uuid4().hex[:12]}",
            combo_id=combo_id,
            month_year=month_year,
            tokens_used=0,
            tokens_budget=budget or 0,
        )
        .on_conflict_do_nothing(index_elements=["combo_id", "month_year"])
    )
    await db.execute(stmt)
    result = await db.execute(
        select(DBTokenBudget)
        .where(DBTokenBudget.combo_id == combo_id, DBTokenBudget.month_year == month_year)
    )
    return result.scalar_one()


async def record_tokens_used(
    db: AsyncSession, combo_id: str, month_year: str, tokens: int,
) -> DBTokenBudget | None:
    """Record tokens used for a combo in a given month; sync budget from combo config.

    The increment is a single atomic upsert so concurrent requests cannot lose
    updates (C17).
    """
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    combo = await get_router_combo(db, combo_id)
    combo_budget = int(getattr(combo, "monthly_token_budget", 0) or 0) if combo else 0

    tokens = int(tokens)
    on_conflict = {
        "tokens_used": DBTokenBudget.__table__.c.tokens_used + tokens,
        "updated_at": datetime.now(timezone.utc),
    }
    if combo_budget:
        on_conflict["tokens_budget"] = combo_budget

    stmt = (
        pg_insert(DBTokenBudget.__table__)
        .values(
            id=f"budget-{uuid.uuid4().hex[:12]}",
            combo_id=combo_id,
            month_year=month_year,
            tokens_used=tokens,
            tokens_budget=combo_budget,
        )
        .on_conflict_do_update(index_elements=["combo_id", "month_year"], set_=on_conflict)
        .returning(DBTokenBudget.__table__.c.id)
    )
    row = await db.execute(stmt)
    row.scalar_one()

    result = await db.execute(
        select(DBTokenBudget)
        .where(DBTokenBudget.combo_id == combo_id, DBTokenBudget.month_year == month_year)
    )
    return result.scalar_one_or_none()


async def get_budget_status(
    db: AsyncSession, combo_id: str, month_year: str,
) -> dict | None:
    """Get budget status: used, remaining, percentage (falls back to combo monthly budget)."""
    result = await db.execute(
        select(DBTokenBudget)
        .where(DBTokenBudget.combo_id == combo_id, DBTokenBudget.month_year == month_year)
    )
    budget = result.scalar_one_or_none()
    combo = await get_router_combo(db, combo_id)
    if not budget:
        if combo and getattr(combo, "monthly_token_budget", 0):
            budget_limit = combo.monthly_token_budget
            return {
                "tokens_used": 0,
                "tokens_budget": budget_limit,
                "tokens_remaining": budget_limit,
                "percentage": 0.0,
                "within_budget": True,
            }
        return None
    tokens_budget = budget.tokens_budget or 0
    if not tokens_budget and combo and getattr(combo, "monthly_token_budget", 0):
        tokens_budget = combo.monthly_token_budget
    remaining = tokens_budget - budget.tokens_used
    percentage = (budget.tokens_used / tokens_budget * 100) if tokens_budget > 0 else 0
    return {
        "tokens_used": budget.tokens_used,
        "tokens_budget": tokens_budget,
        "tokens_remaining": max(remaining, 0),
        "percentage": round(percentage, 1),
        "within_budget": remaining >= 0 if tokens_budget > 0 else True,
    }


async def check_budget_allowed(
    db: AsyncSession, combo_id: str, estimated_tokens: int,
) -> bool:
    """Check if adding estimated_tokens would exceed the monthly budget (0 budget = unlimited)."""
    from datetime import datetime
    month_year = datetime.now().strftime("%Y-%m")
    status = await get_budget_status(db, combo_id, month_year)
    if not status:
        return True  # No budget set, allow
    if not status.get("tokens_budget"):
        return True  # Budget of 0 means unlimited
    return status["tokens_remaining"] >= estimated_tokens


# ── Agentic Sessions ──────────────────────────────────────────────────────

async def create_agentic_session(
    db: AsyncSession, session_id: str, combo_id: str,
    messages: list | None = None, iteration_index: int = 0,
    step_history: list | None = None,
):
    messages = messages if messages is not None else []
    step_history = step_history if step_history is not None else []

    result = await db.execute(
        select(DBAgenticSession).where(DBAgenticSession.session_id == session_id))
    existing = result.scalar_one_or_none()

    now = datetime.now(timezone.utc)
    if existing:
        existing.messages = messages
        existing.iteration_index = iteration_index
        existing.step_history = step_history
        existing.updated_at = now
        await db.flush()
        return existing

    session = DBAgenticSession(
        session_id=session_id, combo_id=combo_id,
        messages=messages, iteration_index=iteration_index,
        step_history=step_history, created_at=now, updated_at=now,
    )
    db.add(session)
    await db.flush()
    return session


async def get_agentic_session(db: AsyncSession, session_id: str):
    result = await db.execute(
        select(DBAgenticSession).where(DBAgenticSession.session_id == session_id))
    return result.scalar_one_or_none()


async def update_agentic_session(
    db: AsyncSession, session_id: str,
    messages: list | None = None, iteration_index: int | None = None,
    step_history: list | None = None,
):
    session = await get_agentic_session(db, session_id)
    if not session:
        return None
    if messages is not None:
        session.messages = messages
    if iteration_index is not None:
        session.iteration_index = iteration_index
    if step_history is not None:
        session.step_history = step_history
    session.updated_at = datetime.now(timezone.utc)
    await db.flush()
    return session


async def close_agentic_session(db: AsyncSession, session_id: str) -> bool:
    session = await get_agentic_session(db, session_id)
    if not session:
        return False
    session.closed_at = datetime.now(timezone.utc)
    await db.flush()
    return True


# ── Budget Alerts ──────────────────────────────────────────────────────

async def resolve_budget_alert(db: AsyncSession, alert_id: str) -> bool:
    result = await db.execute(select(DBBudgetAlert).where(DBBudgetAlert.id == alert_id))
    alert = result.scalar_one_or_none()
    if not alert:
        return False
    alert.resolved_at = datetime.now(timezone.utc)
    await db.flush()
    return True


async def list_budget_alerts(
    db: AsyncSession, combo_id: str | None = None, level: str | None = None,
    limit: int = 50, offset: int = 0,
) -> tuple[list, int]:
    query = select(DBBudgetAlert)
    conditions = []
    if combo_id is not None:
        conditions.append(DBBudgetAlert.combo_id == combo_id)
    if level is not None:
        conditions.append(DBBudgetAlert.level == level)
    if conditions:
        query = query.where(and_(*conditions))

    total_result = await db.execute(select(func.count()).select_from(query.subquery()))
    total = total_result.scalar() or 0

    query = query.order_by(DBBudgetAlert.triggered_at.desc()).offset(offset).limit(limit)
    result = await db.execute(query)
    return list(result.scalars().all()), total


# ── Agentic Analytics ──────────────────────────────────────────────────

async def record_agentic_analytics(
    db: AsyncSession, combo_id: str, request_id: str, month_year: str, record: dict,
):
    result = await db.execute(
        select(DBAgenticAnalytics).where(
            DBAgenticAnalytics.combo_id == combo_id,
            DBAgenticAnalytics.month_year == month_year,
        )
    )
    existing = result.scalar_one_or_none()

    if existing:
        if "requests_count" in record and record["requests_count"] is not None:
            existing.requests_count = (existing.requests_count or 0) + record["requests_count"]
        if "total_iterations" in record and record["total_iterations"] is not None:
            existing.total_iterations = (existing.total_iterations or 0) + record["total_iterations"]
        if "total_tokens" in record and record["total_tokens"] is not None:
            existing.total_tokens = (existing.total_tokens or 0) + record["total_tokens"]
        if "estimated_cost_usd" in record and record["estimated_cost_usd"] is not None:
            existing.estimated_cost_usd = (existing.estimated_cost_usd or 0) + record["estimated_cost_usd"]

        if "stop_reason_counts" in record and record["stop_reason_counts"] is not None:
            merged = dict(existing.stop_reason_counts or {})
            for k, v in record["stop_reason_counts"].items():
                merged[k] = (merged.get(k) or 0) + v
            existing.stop_reason_counts = merged

        if "tool_call_counts" in record and record["tool_call_counts"] is not None:
            merged = dict(existing.tool_call_counts or {})
            for k, v in record["tool_call_counts"].items():
                merged[k] = (merged.get(k) or 0) + v
            existing.tool_call_counts = merged

        if "model_stats" in record and record["model_stats"] is not None:
            existing_list = list(existing.model_stats or [])
            existing_list.extend(record["model_stats"])
            existing.model_stats = existing_list

        existing.updated_at = datetime.now(timezone.utc)
        await db.flush()
        return existing

    row = DBAgenticAnalytics(
        combo_id=combo_id,
        month_year=month_year,
        requests_count=record.get("requests_count") or 0,
        total_iterations=record.get("total_iterations") or 0,
        total_tokens=record.get("total_tokens") or 0,
        estimated_cost_usd=record.get("estimated_cost_usd") or 0,
        stop_reason_counts=record.get("stop_reason_counts") or {},
        tool_call_counts=record.get("tool_call_counts") or {},
        model_stats=record.get("model_stats") or [],
    )
    db.add(row)
    await db.flush()
    return row


def _last_n_months(n: int) -> list[str]:
    months = []
    now = datetime.now(timezone.utc)
    for i in range(n):
        d = now.replace(day=1) - timedelta(days=i * 28)
        months.append(d.strftime("%Y-%m"))
    return list(dict.fromkeys(months))


async def get_agentic_overall_stats(db: AsyncSession, since_months: int = 1) -> dict:
    months = _last_n_months(since_months)
    result = await db.execute(
        select(DBAgenticAnalytics).where(DBAgenticAnalytics.month_year.in_(months))
    )
    rows = list(result.scalars().all())

    total_requests = 0
    total_iterations = 0
    total_tokens = 0
    estimated_cost_usd = 0.0
    stop_count = 0

    for row in rows:
        total_requests += row.requests_count or 0
        total_iterations += row.total_iterations or 0
        total_tokens += row.total_tokens or 0
        estimated_cost_usd += row.estimated_cost_usd or 0
        src = row.stop_reason_counts or {}
        stop_count += src.get("stop", 0) or 0

    avg_iterations_per_request = total_iterations / max(1, total_requests)
    success_rate = stop_count / max(1, total_requests)

    return {
        "total_requests": total_requests,
        "total_iterations": total_iterations,
        "avg_iterations_per_request": avg_iterations_per_request,
        "total_tokens": total_tokens,
        "estimated_cost_usd": estimated_cost_usd,
        "success_rate": success_rate,
    }


async def get_combo_performance(db: AsyncSession, combo_id: str, months: int = 1) -> list[dict]:
    month_list = _last_n_months(months)
    result = await db.execute(
        select(DBAgenticAnalytics).where(
            DBAgenticAnalytics.combo_id == combo_id,
            DBAgenticAnalytics.month_year.in_(month_list),
        )
    )
    rows = list(result.scalars().all())

    out = []
    for row in rows:
        rc = row.requests_count or 0
        ti = row.total_iterations or 0
        out.append({
            "combo_id": row.combo_id,
            "period": row.month_year,
            "requests_count": rc,
            "total_iterations": ti,
            "avg_iterations": ti / max(1, rc),
            "total_tokens": row.total_tokens or 0,
            "estimated_cost_usd": row.estimated_cost_usd or 0,
            "stop_reason_counts": dict(row.stop_reason_counts or {}),
            "tool_call_counts": dict(row.tool_call_counts or {}),
        })
    return out


async def get_model_rankings(db: AsyncSession, combo_id: str, months: int = 1) -> list[dict]:
    month_list = _last_n_months(months)
    result = await db.execute(
        select(DBAgenticAnalytics).where(
            DBAgenticAnalytics.combo_id == combo_id,
            DBAgenticAnalytics.month_year.in_(month_list),
        )
    )
    rows = list(result.scalars().all())

    agg = {}
    for row in rows:
        stats = row.model_stats or []
        for entry in stats:
            key = (entry.get("provider"), entry.get("model_id"))
            if key not in agg:
                agg[key] = {"calls": 0, "success_calls": 0, "tokens_weighted": 0.0, "cost_weighted": 0.0}
            calls = entry.get("calls") or 0
            agg[key]["calls"] += calls
            agg[key]["success_calls"] += entry.get("success_calls") or 0
            agg[key]["tokens_weighted"] += (entry.get("avg_tokens") or 0) * calls
            agg[key]["cost_weighted"] += (entry.get("cost_estimate") or 0) * calls

    rankings = []
    for (provider, model_id), v in agg.items():
        total_calls = v["calls"]
        success_calls = v["success_calls"]
        avg_tokens = v["tokens_weighted"] / max(1, total_calls)
        avg_cost = v["cost_weighted"] / max(1, total_calls)
        success_rate = success_calls / max(1, total_calls)
        cost_efficiency = success_rate / max(0.0001, avg_cost)
        rankings.append({
            "provider": provider,
            "model_id": model_id,
            "success_rate": success_rate,
            "avg_tokens_per_call": avg_tokens,
            "cost_efficiency": cost_efficiency,
            "total_calls": total_calls,
        })

    rankings.sort(key=lambda x: x["success_rate"], reverse=True)
    return rankings
