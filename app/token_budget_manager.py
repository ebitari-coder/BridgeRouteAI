"""Token budget manager for agentic combos.

Tracks token usage per combo per month and enforces monthly budgets.
Prevents combos from exceeding their token allowance.
Provides budget alerts and historical usage tracking.
"""
import logging
import calendar
from datetime import datetime, timezone, timedelta
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, update
from app.crud import (
    get_or_create_monthly_budget, record_tokens_used, check_budget_allowed,
    get_budget_status
)
from app.database import async_session
from app.db_models import DBTokenBudget, DBBudgetAlert

logger = logging.getLogger(__name__)


class TokenBudgetManager:
    """Manages monthly token budgets for router combos."""

    def __init__(self, combo_id: str):
        self.combo_id = combo_id

    async def get_status(self) -> dict | None:
        """Get current budget status for this combo."""
        async with async_session() as db:
            month_year = datetime.now().strftime("%Y-%m")
            return await self._get_status(db, month_year)

    async def _get_status(self, db: AsyncSession, month_year: str) -> dict | None:
        from app.crud import get_budget_status
        return await get_budget_status(db, self.combo_id, month_year)

    async def record_usage(self, tokens: int) -> dict | None:
        """Record tokens used for this combo and return updated status."""
        async with async_session() as db:
            month_year = datetime.now().strftime("%Y-%m")
            budget = await record_tokens_used(db, self.combo_id, month_year, tokens)
            await db.flush()
            try:
                await db.commit()
            except Exception:
                await db.rollback()
                raise
            return {
                "tokens_used": budget.tokens_used,
                "tokens_budget": budget.tokens_budget,
                "tokens_remaining": max(budget.tokens_budget - budget.tokens_used, 0),
            }

    async def can_use_tokens(self, estimated_tokens: int) -> bool:
        """Check if the combo can use estimated_tokens without exceeding budget."""
        async with async_session() as db:
            return await check_budget_allowed(db, self.combo_id, estimated_tokens)

    async def is_within_budget(self) -> bool:
        """Check if current usage is within the monthly budget."""
        status = await self.get_status()
        if not status:
            return True
        return status["within_budget"]

    async def get_historical_usage(self, months: int = 12) -> list[dict]:
        """Get historical token usage for the last N months."""
        async with async_session() as db:
            from app.crud import get_router_combo
            
            combo = await get_router_combo(db, self.combo_id)
            if not combo:
                return []
            
            result = await db.execute(
                select(DBTokenBudget)
                .where(DBTokenBudget.combo_id == self.combo_id)
                .order_by(DBTokenBudget.month_year.desc())
                .limit(months)
            )
            budgets = result.scalars().all()
            
            return [
                {
                    "month_year": b.month_year,
                    "tokens_used": b.tokens_used,
                    "tokens_budget": b.tokens_budget,
                    "tokens_remaining": max(b.tokens_budget - b.tokens_used, 0),
                    "percentage": round((b.tokens_used / b.tokens_budget * 100), 1) if b.tokens_budget > 0 else 0,
                }
                for b in budgets
            ]

    async def get_all_budgets(self, combo_id: str = "", limit_months: int = 12) -> list:
        """Return list of all DBTokenBudget rows for this combo."""
        target_combo = combo_id or self.combo_id
        async with async_session() as db:
            result = await db.execute(
                select(DBTokenBudget)
                .where(DBTokenBudget.combo_id == target_combo)
                .order_by(DBTokenBudget.month_year.desc())
                .limit(limit_months)
            )
            return list(result.scalars().all())

    async def resolve_alert(self, alert_id: str) -> bool:
        """Mark a budget alert as resolved by setting resolved_at timestamp."""
        if not alert_id:
            return False
        async with async_session() as db:
            result = await db.execute(
                select(DBBudgetAlert).where(DBBudgetAlert.id == alert_id)
            )
            alert = result.scalar_one_or_none()
            if not alert:
                return False
            alert.resolved_at = datetime.now(timezone.utc)
            await db.flush()
            try:
                await db.commit()
            except Exception:
                await db.rollback()
                raise
            return True

    async def get_alerts(self, level: Optional[str] = None, limit: int = 50) -> list[dict]:
        """Get budget alerts for this combo."""
        async with async_session() as db:
            query = select(DBBudgetAlert).where(
                DBBudgetAlert.combo_id == self.combo_id
            )
            if level:
                query = query.where(DBBudgetAlert.level == level)
            
            result = await db.execute(
                query.order_by(DBBudgetAlert.triggered_at.desc()).limit(limit)
            )
            alerts = result.scalars().all()
            
            return [
                {
                    "id": a.id,
                    "month_year": a.month_year,
                    "level": a.level,
                    "percentage": a.percentage,
                    "tokens_used": a.tokens_used,
                    "tokens_budget": a.tokens_budget,
                    "triggered_at": a.triggered_at.isoformat() if a.triggered_at else None,
                    "resolved_at": a.resolved_at.isoformat() if a.resolved_at else None,
                }
                for a in alerts
            ]


class BudgetEnforcer:
    """Enforces budget limits during agentic loop execution with alert tracking."""

    def __init__(self, combo_id: str, monthly_budget: int = 0, alert_manager=None):
        self.combo_id = combo_id
        self.monthly_budget = monthly_budget
        self.tokens_used_this_session = 0
        self.alert_manager = alert_manager or BudgetAlertManager(combo_id)
        self._last_warning_pct = 0
        self._last_critical_pct = 0

    def check(self, estimated_tokens: int) -> bool:
        """Check if adding estimated_tokens is within budget."""
        if self.monthly_budget <= 0:
            return True  # No budget set
        return (self.tokens_used_this_session + estimated_tokens) <= self.monthly_budget

    def seed_from_status(self, status: dict | None) -> None:
        """Anchor the session counter at month-to-date usage (C12).

        Without this the enforcer restarts at 0 on every request, so the 80%/95%
        thresholds compare a single request's tokens against a *monthly* budget
        and never fire. Threshold flags are fast-forwarded too, so an already
        breached budget does not re-emit the same alert on every request.
        """
        if not status:
            return
        budget = int(status.get("tokens_budget") or 0)
        used = int(status.get("tokens_used") or 0)
        if budget > 0:
            self.monthly_budget = budget
        self.tokens_used_this_session = used
        if self.monthly_budget > 0:
            percentage = (used / self.monthly_budget) * 100
            if percentage >= 80:
                self._last_warning_pct = 80
            if percentage >= 95:
                self._last_critical_pct = 95

    def record(self, tokens: int):
        """Record tokens used in this session."""
        self.tokens_used_this_session += tokens
        self._check_thresholds(tokens)

    def _enqueue(self, alert: dict) -> None:
        """Publish an alert to the manager's pending queue so SSE streams can diff it (C8)."""
        pending = getattr(self.alert_manager, "_pending_alerts", None)
        if isinstance(pending, list):
            pending.append(alert)

    def remaining(self) -> int:
        """Get remaining budget for this session."""
        if self.monthly_budget <= 0:
            return -1  # Unlimited
        return max(self.monthly_budget - self.tokens_used_this_session, 0)

    def _check_thresholds(self, tokens: int) -> list[dict]:
        """Check and generate alerts when thresholds are crossed. Returns list of new alerts."""
        if self.monthly_budget <= 0:
            return []
        
        percentage = (self.tokens_used_this_session / self.monthly_budget) * 100
        new_alerts = []

        def _schedule_persist(combo_id: str, pct: float, used: int):
            try:
                import asyncio
                coro = self.alert_manager.record_alert(combo_id, pct, used)
                try:
                    loop = asyncio.get_running_loop()
                    if loop.is_running():
                        task = loop.create_task(coro)
                        def _log_error(t):
                            try:
                                t.result()
                            except Exception:
                                logger.warning("Async persist alert failed for combo %s", combo_id, exc_info=True)
                        task.add_done_callback(_log_error)
                        return
                except RuntimeError:
                    pass
                try:
                    asyncio.run(coro)
                except Exception:
                    logger.warning("Sync persist alert failed for combo %s", combo_id, exc_info=True)
            except Exception:
                logger.warning("Failed to dispatch budget alert persist for combo %s", combo_id, exc_info=True)
        
        # Check for warning threshold (80%)
        if percentage >= 80 and self._last_warning_pct < 80:
            alert = {
                "level": "warning",
                "percentage": percentage,
                "message": f"Combo {self.combo_id} at {percentage:.1f}% of monthly budget",
            }
            _schedule_persist(self.combo_id, percentage, self.tokens_used_this_session)
            self._enqueue(alert)
            new_alerts.append(alert)
            self._last_warning_pct = 80
        
        # Check for critical threshold (95%)
        if percentage >= 95 and self._last_critical_pct < 95:
            alert = {
                "level": "critical",
                "percentage": percentage,
                "message": f"Combo {self.combo_id} at {percentage:.1f}% of monthly budget - action required",
            }
            _schedule_persist(self.combo_id, percentage, self.tokens_used_this_session)
            self._enqueue(alert)
            new_alerts.append(alert)
            self._last_critical_pct = 95
        
        return new_alerts


class BudgetAlertManager:
    """Manages budget alerts with configurable thresholds and auto-replenishment."""

    def __init__(self, combo_id: str, warning_threshold: float = 80.0, critical_threshold: float = 95.0):
        self.combo_id = combo_id
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self._pending_alerts: list[dict] = []

    def check_and_alert(self, tokens_used: int, tokens_budget: int) -> Optional[dict]:
        """Check thresholds and generate alerts. Returns alert if threshold crossed."""
        if tokens_budget <= 0:
            return None
        
        percentage = (tokens_used / tokens_budget) * 100
        
        if percentage >= self.critical_threshold:
            alert = {
                "level": "critical",
                "percentage": percentage,
                "message": f"Budget critical: {tokens_used}/{tokens_budget} ({percentage:.1f}%)",
                "combo_id": self.combo_id,
            }
            self._pending_alerts.append(alert)
            return alert
        
        if percentage >= self.warning_threshold:
            alert = {
                "level": "warning",
                "percentage": percentage,
                "message": f"Budget warning: {tokens_used}/{tokens_budget} ({percentage:.1f}%)",
                "combo_id": self.combo_id,
            }
            self._pending_alerts.append(alert)
            return alert
        
        return None

    async def record_alert(self, combo_id: str, percentage: float, tokens_used: int) -> dict:
        """Persist an alert to the database.

        One row per (combo, month, level): re-crossing a threshold updates the
        existing row instead of appending duplicates.
        """
        async with async_session() as db:
            month_year = datetime.now().strftime("%Y-%m")
            
            # Get tokens_budget from combo or default
            from app.crud import get_router_combo
            combo = await get_router_combo(db, combo_id)
            tokens_budget = combo.monthly_token_budget if combo else 0
            level = "warning" if percentage < self.critical_threshold else "critical"

            existing = await db.execute(
                select(DBBudgetAlert).where(
                    DBBudgetAlert.combo_id == combo_id,
                    DBBudgetAlert.month_year == month_year,
                    DBBudgetAlert.level == level,
                )
            )
            alert = existing.scalar_one_or_none()
            if alert is not None:
                alert.percentage = round(percentage, 1)
                alert.tokens_used = tokens_used
                alert.tokens_budget = tokens_budget
            else:
                alert = DBBudgetAlert(
                    combo_id=combo_id,
                    month_year=month_year,
                    level=level,
                    percentage=round(percentage, 1),
                    tokens_used=tokens_used,
                    tokens_budget=tokens_budget,
                )
                db.add(alert)
            await db.flush()
            try:
                await db.commit()
            except Exception:
                await db.rollback()
                raise
            return {
                "id": alert.id,
                "level": alert.level,
                "percentage": alert.percentage,
                "message": f"Budget at {percentage}%",
                "timestamp": alert.triggered_at.isoformat() if alert.triggered_at else None,
            }

    async def should_auto_replenish(self, combo_id: str, usage_days_window: int = 7) -> bool:
        """Determine if auto-replenishment should be triggered.

        - Get at least 7 days usage (if no history, return False)
        - daily_rate = tokens_used_in_window / min(7, actual_days)
        - remaining = tokens_budget - tokens_used_this_month
        - days_left_in_month = start of next month - today
        - Return True if daily_rate > 0 AND (remaining / daily_rate) < 7
        Wrapped in try/except, default False on any error.
        """
        try:
            from app.crud import get_router_combo

            async with async_session() as db:
                today = datetime.now(timezone.utc).date()

                combo = await get_router_combo(db, combo_id)
                tokens_budget = combo.monthly_token_budget if combo and combo.monthly_token_budget else 0
                if tokens_budget <= 0:
                    return False

                month_year = today.strftime("%Y-%m")
                current_status = await get_budget_status(db, combo_id, month_year)
                if not current_status:
                    return False
                tokens_used_this_month = current_status.get("tokens_used", 0)

                start_date = today - timedelta(days=usage_days_window - 1)
                window_months: list[str] = []
                d = start_date
                while d <= today:
                    m = d.strftime("%Y-%m")
                    if m not in window_months:
                        window_months.append(m)
                    d += timedelta(days=1)

                result = await db.execute(
                    select(DBTokenBudget)
                    .where(DBTokenBudget.combo_id == combo_id, DBTokenBudget.month_year.in_(window_months))
                )
                budget_rows = {b.month_year: b for b in result.scalars().all()}

                days_in_window = (today - start_date).days + 1
                if days_in_window < 1:
                    days_in_window = 1

                # Estimate daily run-rate: prefer month-to-date usage over
                # elapsed days of the current month; fall back to window rows.
                day_of_month = today.day
                if tokens_used_this_month > 0 and day_of_month >= 1:
                    daily_rate = tokens_used_this_month / day_of_month
                elif budget_rows:
                    window_used = sum(
                        getattr(r, "tokens_used", 0) or 0 for r in budget_rows.values()
                    )
                    daily_rate = window_used / min(usage_days_window, days_in_window)
                else:
                    return False

                if daily_rate <= 0:
                    return False

                remaining = max(tokens_budget - tokens_used_this_month, 0)

                year = today.year
                month = today.month
                if month == 12:
                    next_month_start = datetime(year + 1, 1, 1, tzinfo=timezone.utc).date()
                else:
                    next_month_start = datetime(year, month + 1, 1, tzinfo=timezone.utc).date()
                days_left_in_month = (next_month_start - today).days

                if days_left_in_month <= 0:
                    return False

                days_buffer = remaining / daily_rate
                return days_buffer < 7
        except Exception:
            logger.warning("should_auto_replenish check failed for combo %s", combo_id, exc_info=True)
            return False

    async def get_daily_rate(self, combo_id: str | None = None) -> float:
        """Compute the current daily token run-rate from historical usage."""
        try:
            target = combo_id or self.combo_id
            today = datetime.now(timezone.utc).date()
            day_of_month = max(1, today.day)
            async with async_session() as db:
                from app.crud import get_budget_status
                status = await get_budget_status(db, target, today.strftime("%Y-%m"))
                if not status:
                    return 0.0
                return float(status.get("tokens_used", 0)) / day_of_month
        except Exception:
            return 0.0

    async def build_replenishment_prompt(self, combo_id: str | None = None) -> dict | None:
        """Build a replenishment prompt from real historical usage, if needed."""
        target = combo_id or self.combo_id
        should = await self.should_auto_replenish(target)
        if not should:
            return None
        daily_rate = await self.get_daily_rate(target)
        async with async_session() as db:
            from app.crud import get_budget_status, get_router_combo
            today = datetime.now(timezone.utc)
            status = await get_budget_status(db, target, today.strftime("%Y-%m"))
            combo = await get_router_combo(db, target)
            tokens_budget = 0
            tokens_used = 0
            if status:
                tokens_budget = status.get("tokens_budget", 0)
                tokens_used = status.get("tokens_used", 0)
            elif combo:
                tokens_budget = combo.monthly_token_budget or 0
            pct = (tokens_used / tokens_budget * 100) if tokens_budget > 0 else 0.0
        return self.get_replenishment_prompt(
            current_usage_pct=pct,
            daily_rate=daily_rate,
            tokens_budget=tokens_budget,
            tokens_used=tokens_used,
        )

    def get_replenishment_prompt(self, current_usage_pct: float, daily_rate: float,
                                  tokens_budget: int = 0, tokens_used: int = 0) -> dict:
        """Generate a prompt recommending upgrade or token replenishment.

        - days_remaining_estimate = max(0, (tokens_budget-tokens_used) / daily_rate) if daily_rate>0 else 30
        - recommended_replenish_tokens = int(max(daily_rate*14, tokens_budget * 0.5))
        """
        try:
            if daily_rate and daily_rate > 0 and tokens_budget > 0:
                days_remaining_estimate = max(0, (tokens_budget - tokens_used) / daily_rate)
            else:
                days_remaining_estimate = 30

            base_from_daily = daily_rate * 14 if daily_rate and daily_rate > 0 else 0
            base_from_budget = tokens_budget * 0.5 if tokens_budget else 0
            recommended_replenish_tokens = int(max(base_from_daily, base_from_budget, 100))

            return {
                "action": "upgrade_or_replenish",
                "message": (
                    f"Token budget at {current_usage_pct:.1f}% usage. "
                    f"Estimated {days_remaining_estimate:.1f} days remaining at current rate. "
                    f"Consider upgrading or replenishing tokens to avoid interruption."
                ),
                "recommended_amount": recommended_replenish_tokens,
                "days_remaining": round(days_remaining_estimate, 1),
                "upgrade_link": "/billing/upgrade",
                "daily_rate": daily_rate or 0,
            }
        except Exception:
            return {
                "action": "upgrade_or_replenish",
                "message": "Budget nearly exhausted. Consider upgrading your plan or replenishing tokens.",
                "recommended_amount": max(100, int(tokens_budget * 0.5)) if tokens_budget else 1000,
                "days_remaining": 7,
                "upgrade_link": "/billing/upgrade",
                "daily_rate": daily_rate or 0,
            }


async def get_all_budgets_for_combo(combo_id: str, limit_months: int = 12) -> list[dict]:
    """Return historical budget rows for a combo (used by history endpoints)."""
    tbm = TokenBudgetManager(combo_id)
    rows = await tbm.get_all_budgets(combo_id, limit_months)
    return [
        {
            "month_year": b.month_year,
            "tokens_used": b.tokens_used,
            "tokens_budget": b.tokens_budget,
            "tokens_remaining": max((b.tokens_budget or 0) - (b.tokens_used or 0), 0),
            "percentage": round((b.tokens_used / b.tokens_budget * 100), 1)
            if b.tokens_budget
            else 0,
        }
        for b in rows
    ]


async def resolve_alert(alert_id: str) -> bool:
    """Mark a budget alert resolved (module-level helper for endpoints)."""
    if not alert_id:
        return False
    async with async_session() as db:
        from app.crud import resolve_budget_alert as _resolve
        ok = await _resolve(db, alert_id)
        await db.commit()
        return ok


async def enforce_budget(combo_id: str, estimated_tokens: int) -> bool:
    """Check if a combo can use estimated_tokens from its monthly budget."""
    async with async_session() as db:
        return await check_budget_allowed(db, combo_id, estimated_tokens)


async def get_combo_budget_status(combo_id: str) -> dict | None:
    """Get budget status for a combo."""
    async with async_session() as db:
        month_year = datetime.now().strftime("%Y-%m")
        from app.crud import get_budget_status
        return await get_budget_status(db, combo_id, month_year)