"""Router combo management endpoints for admin."""
import json

from fastapi import APIRouter, Depends, HTTPException, Header, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timezone, timedelta
from app.database import get_db
from app.db_models import DBApiKey
from app.middleware.auth import verify_admin_key
from app import crud
from app.crud import ComboConflict
from app.providers import refresh_active_combos
from app import agentic_presets
from app.models import (
    ComboCreate,
    ComboUpdate,
    ComboResponse,
    ComboModel,
    AgenticConfig,
    BudgetStatusResponse,
    CostEstimateResponse,
    AgenticStrategyPreset,
    ComboPresetApply,
    BudgetAlertResponse,
    ROUTING_MODES,
    ROUTING_STRATEGIES,
    FUSION_JUDGE_PROVIDER,
    ROUTING_STRATEGY_CATALOG,
    RoutingStrategyInfo,
)

router = APIRouter(prefix="/admin/combos", tags=["Router Combos"])


def _mask_actor(admin_key: str) -> str:
    """Audit actor from the admin credential without storing the raw key."""
    mask = getattr(crud, "mask_api_key", None)
    if callable(mask):
        try:
            return f"admin:{mask(admin_key)}"
        except Exception:
            pass
    return "admin"


async def _audit_combo(
    db: AsyncSession, admin_key: str, action: str,
    combo_id: str, details: str,
) -> None:
    """Append an audit entry for a combo mutation (rows commit with it)."""
    await crud.audit(
        db,
        action=f"combo.{action}",
        actor=_mask_actor(admin_key),
        target=combo_id,
        details=details,
    )


def _dedupe_models(models: list[dict]) -> list[dict]:
    """Reject duplicate (provider, model_id) rows and blank entries.

    Duplicate rows in models[] are a silent routing ambiguity (the group
    strategies would fan out to the same upstream twice); it is the caller's
    mistake and should be a 4xx, not a normalization.
    """
    seen: set[tuple] = set()
    out: list[dict] = []
    for m in models:
        p = (m.get("provider") or "").strip()
        mid = (m.get("model_id") or "").strip()
        if not p or not mid:
            raise HTTPException(400, "Every models[] entry needs a provider and model_id")
        key = (p, mid)
        if key in seen:
            raise HTTPException(400, f"Duplicate models[] entry: {p}/{mid}")
        seen.add(key)
        out.append({"provider": p, "model_id": mid})
    return out


def _expected_version(body: ComboUpdate | None, if_match: str | None) -> int | None:
    """Optimistic-concurrency version from If-Match (wins) or the body field.

    If-Match follows RFC 9110 (may be quoted, may be `W/"5"`); anything that
    is not a positive integer is a 400 rather than silently ignored.
    """
    raw = if_match.strip() if if_match else None
    if raw:
        if raw.startswith("W/"):
            raw = raw[2:]
        raw = raw.strip('"')
        try:
            return int(raw)
        except ValueError:
            raise HTTPException(428, "If-Match must be the combo's version number")
    if body is not None and body.expected_version is not None:
        return body.expected_version
    return None



def _agentic_from_config(config) -> dict:
    if not config:
        return {
            "agentic_enabled": False,
            "max_iterations": 5,
            "tool_calling": True,
            "deterministic": True,
            "loop_strategy": "sequential",
            "compression_target_tokens": 4000,
            "load_spread": False,
        }
    if isinstance(config, dict):
        return {
            "agentic_enabled": config.get("agentic_enabled", False),
            "max_iterations": config.get("max_iterations", 5),
            "tool_calling": config.get("tool_calling", True),
            "deterministic": config.get("deterministic", True),
            "loop_strategy": config.get("loop_strategy", "sequential"),
            "compression_target_tokens": config.get("compression_target_tokens", 4000),
        }
    return {
        "agentic_enabled": config.agentic_enabled,
        "max_iterations": config.max_iterations,
        "tool_calling": config.tool_calling,
        "deterministic": config.deterministic,
        "loop_strategy": config.loop_strategy,
        "compression_target_tokens": config.compression_target_tokens,
    }


def _combo_to_response(c) -> dict:
    models_list = c.models if c.models else []
    cleaned_models = []
    for m in models_list:
        if isinstance(m, dict):
            cleaned = {k: v for k, v in m.items() if k != "tools_allowed"}
            if cleaned.get("provider") and cleaned.get("model_id"):
                cleaned_models.append(cleaned)
        else:
            cleaned_models.append(m)

    tools_allowed = None
    if hasattr(c, "tools_allowed") and c.tools_allowed:
        tools_allowed = list(c.tools_allowed) if isinstance(c.tools_allowed, list) else None
    if not tools_allowed and models_list:
        for m in models_list:
            if isinstance(m, dict) and "tools_allowed" in m and m["tools_allowed"]:
                tools_allowed = list(m["tools_allowed"])
                break

    strategy_preset = None
    if hasattr(c, "strategy_preset"):
        strategy_preset = c.strategy_preset

    monthly_token_budget = c.monthly_token_budget if hasattr(c, "monthly_token_budget") else 0
    version = c.version if hasattr(c, "version") else 1
    load_spread = bool(getattr(c, "load_spread", False))

    return {
        "id": c.id,
        "name": c.name,
        "router_type": c.router_type,
        "version": version,
        "models": cleaned_models,
        "is_active": c.is_active,
        "routing_mode": getattr(c, "routing_mode", None) or "group",
        "routing_strategy": getattr(c, "routing_strategy", None) or "fallback",
        "judge_model": getattr(c, "judge_model", None),
        "agentic_enabled": c.agentic_enabled,
        "max_iterations": c.max_iterations,
        "tool_calling": c.tool_calling,
        "deterministic": c.deterministic,
        "loop_strategy": c.loop_strategy,
        "compression_target_tokens": c.compression_target_tokens,
        "load_spread": load_spread,
        "monthly_token_budget": monthly_token_budget,
        "tools_allowed": tools_allowed,
        "strategy_preset": strategy_preset,
        "created_at": c.created_at.isoformat() if c.created_at else None,
        "updated_at": c.updated_at.isoformat() if c.updated_at else None,
    }


@router.get("")
async def list_combos(
    router_type: str | None = None,
    limit: int | None = Query(default=None, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    _: str = Depends(verify_admin_key),
    db: AsyncSession = Depends(get_db),
):
    """List router combos, optionally filtered by type.

    Backward compatible: without limit/offset every combo is returned.
    With limit set the response carries {data, total, limit, offset} so
    admins can page without losing older clients.
    """
    combos = await crud.get_router_combos(db, router_type)
    total = len(combos)
    if limit is not None:
        combos = combos[offset:offset + limit]
    payload: dict = {"data": [_combo_to_response(c) for c in combos]}
    if limit is not None:
        payload.update({"total": total, "limit": limit, "offset": offset})
    return payload


# NOTE: Static-path routes must be registered before /{combo_id} so they are
# not shadowed by the dynamic path parameter (FastAPI matches in registration order).


@router.get("/strategy-presets")
async def list_strategy_presets(
    _: str = Depends(verify_admin_key),
):
    """List available agentic strategy presets."""
    presets = agentic_presets.list_presets()
    return [
        AgenticStrategyPreset(
            name=p["name"],
            description=p["description"],
            max_iterations=p["max_iterations"],
            loop_strategy=p["loop_strategy"],
            compression_target_tokens=p["compression_target_tokens"],
            recommended_tools=p["recommended_tools"],
        )
        for p in presets
    ]


class ComboPresetApplyBody(BaseModel):
    preset_name: str
    combo_id: str | None = None


@router.get("/routing-strategies")
async def list_routing_strategies(
    _: str = Depends(verify_admin_key),
):
    """Describe the group routing strategies an admin can pick per combo."""
    return [s.model_dump() for s in ROUTING_STRATEGY_CATALOG]


@router.get("/judge-models")
async def list_judge_models(
    _: str = Depends(verify_admin_key),
):
    """Reachable judge candidates for the fusion strategy.

    The judge is the one model that fuses the panel answers, so the picker is
    restricted to the live FreeModels catalog — ids the gateway can actually
    reach right now, not a stale seed list.
    """
    from app.providers import PROVIDER_FREEMODELS, _health, freemodels_pool

    pool = freemodels_pool()
    return {
        "provider": FUSION_JUDGE_PROVIDER,
        "healthy": _health[PROVIDER_FREEMODELS].is_healthy,
        "models": [{"model_id": m} for m in pool],
    }


def _routing_from_body(body) -> tuple[str | None, str | None, str | None, bool]:
    """Extract (mode, strategy, judge, clear_judge) from a create/update body.

    ``routing_mode``/``routing_strategy`` are the source of truth. Clients that
    predate 011 only send ``agentic_config.agentic_enabled``; treat that as a
    request for ``routing_mode='agentic'`` so the old backoffice keeps working.
    Returns None for any part the caller did not explicitly set, letting the
    update path leave the column untouched.
    """
    fields = getattr(body, "model_fields_set", set())
    mode = body.routing_mode if "routing_mode" in fields else None
    strategy = body.routing_strategy if "routing_strategy" in fields else None
    judge = body.judge_model if "judge_model" in fields else None
    clear_judge = "judge_model" in fields and body.judge_model is None

    if mode is None and body.agentic_config is not None and "agentic_config" in fields:
        if body.agentic_config.agentic_enabled:
            mode = "agentic"
        elif "agentic_enabled" in getattr(body.agentic_config, "model_fields_set", set()):
            mode = "group"

    return mode, strategy, judge, clear_judge


def _validate_routing(
    name: str, models_count: int, mode: str, strategy: str, judge_model: str | None = None,
) -> None:
    if mode not in ROUTING_MODES:
        raise HTTPException(400, f"routing_mode must be one of {', '.join(ROUTING_MODES)}")
    if strategy not in ROUTING_STRATEGIES:
        raise HTTPException(400, f"routing_strategy must be one of {', '.join(ROUTING_STRATEGIES)}")
    if mode == "group" and strategy == "fusion":
        if models_count < 2:
            raise HTTPException(
                400,
                "Fusion needs at least two models in the combo: one for each panel "
                "member plus the judge (N+1 calls per request).",
            )
        ref = (judge_model or "").strip()
        if not ref:
            raise HTTPException(
                400,
                "Fusion requires an explicit judge model: set judge_model to "
                f"'{FUSION_JUDGE_PROVIDER}/<model_id>' — the model that fuses the "
                "panel answers into one response.",
            )
        provider = ref.split("/", 1)[0].strip()
        if provider != FUSION_JUDGE_PROVIDER:
            raise HTTPException(
                400,
                f"Fusion judges must come from the {FUSION_JUDGE_PROVIDER} provider "
                f"(got '{provider}'). Use '{FUSION_JUDGE_PROVIDER}/<model_id>'.",
            )
        model_id = ref.split("/", 1)[1].strip() if "/" in ref else ""
        from app.providers import freemodels_pool
        if not model_id or model_id not in freemodels_pool():
            raise HTTPException(
                400,
                f"Fusion judge '{ref}' is not in the reachable "
                f"{FUSION_JUDGE_PROVIDER} catalog. GET "
                "/admin/combos/judge-models lists the ids that work right now.",
            )


@router.post("/apply-preset")
async def apply_preset_to_combo(
    body: ComboPresetApplyBody | None = None,
    preset_name: str | None = None,
    combo_id: str | None = None,
    admin: str = Depends(verify_admin_key),
    db: AsyncSession = Depends(get_db),
):
    """Apply a strategy preset to a combo (preset values overwrite agentic fields)."""
    target_combo_id = combo_id
    target_preset = preset_name
    if body is not None:
        if body.combo_id:
            target_combo_id = body.combo_id
        if body.preset_name:
            target_preset = body.preset_name

    if not target_combo_id:
        raise HTTPException(400, "combo_id is required")
    if not target_preset:
        raise HTTPException(400, "preset_name is required")

    combo = await crud.get_router_combo(db, target_combo_id)
    if not combo:
        raise HTTPException(404, "Combo not found")
    preset = agentic_presets.get_preset(target_preset)
    if not preset:
        raise HTTPException(404, "Preset not found")

    # Apply-preset intentionally overwrites loop fields with the preset values.
    # Keep agentic_enabled/tool_calling/deterministic as-is on the combo.
    updated = await crud.update_router_combo(
        db, target_combo_id,
        max_iterations=preset["max_iterations"],
        loop_strategy=preset["loop_strategy"],
        compression_target_tokens=preset["compression_target_tokens"],
        tools_allowed=list(preset["recommended_tools"]),
        strategy_preset=target_preset,
    )
    if not updated:
        raise HTTPException(404, "Combo not found")
    await _audit_combo(
        db, admin, "preset.apply", target_combo_id,
        json.dumps({"preset": target_preset, **dict(preset)}),
    )
    await db.commit()
    await refresh_active_combos(db)
    updated_combo = await crud.get_router_combo(db, target_combo_id)
    return _combo_to_response(updated_combo)


@router.get("/{combo_id}")
async def get_combo(
    combo_id: str,
    _: str = Depends(verify_admin_key),
    db: AsyncSession = Depends(get_db),
):
    """Get a single router combo by id."""
    combo = await crud.get_router_combo(db, combo_id)
    if not combo:
        raise HTTPException(404, "Combo not found")
    return _combo_to_response(combo)


@router.post("")
async def create_combo(
    body: ComboCreate,
    admin: str = Depends(verify_admin_key),
    db: AsyncSession = Depends(get_db),
):
    """Create a new router combo."""
    if body.router_type not in ("chatbot", "agentic", "coding"):
        raise HTTPException(400, "router_type must be chatbot, agentic, or coding")
    if not body.models:
        raise HTTPException(400, "At least one model is required")

    body_fields = body.model_dump()
    if body_fields.get("strategy_preset"):
        body_fields = agentic_presets.apply_preset_defaults(body_fields)

    agentic = _agentic_from_config(body_fields.get("agentic_config"))
    tools_allowed = body_fields.get("tools_allowed")
    strategy_preset = body_fields.get("strategy_preset")

    mode, strategy, judge, _clear = _routing_from_body(body)
    mode = mode or "group"
    strategy = strategy or "fallback"
    _validate_routing(body.name, len(body.models), mode, strategy, judge)

    cleaned_models = _dedupe_models(
        [m.model_dump() if isinstance(m, ComboModel) else m for m in body_fields["models"]]
    )
    # The unique-name violation surfaces at the crud-internal flush, so the
    # whole write (insert + audit + commit) sits inside the 409 guard.
    try:
        combo = await crud.create_router_combo(
            db, body_fields["name"], body_fields["router_type"], cleaned_models,
            routing_mode=mode,
            routing_strategy=strategy,
            judge_model=judge,
            max_iterations=agentic["max_iterations"],
            tool_calling=agentic["tool_calling"],
            deterministic=agentic["deterministic"],
            loop_strategy=agentic["loop_strategy"],
            compression_target_tokens=agentic["compression_target_tokens"],
            load_spread=agentic.get("load_spread", False),
            monthly_token_budget=body_fields.get("monthly_token_budget") or 0,
            tools_allowed=tools_allowed,
            strategy_preset=strategy_preset,
        )
        await _audit_combo(
            db, admin, "create", combo.id,
            json.dumps({"name": body_fields["name"], "router_type": body_fields["router_type"],
                        "routing_mode": mode, "routing_strategy": strategy,
                        "models": cleaned_models, "tools_allowed": tools_allowed}),
        )
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(409, f"A combo named '{body_fields['name']}' already exists") from exc
    await refresh_active_combos(db)
    combo = await crud.get_router_combo(db, combo.id)
    return _combo_to_response(combo)


@router.put("/{combo_id}")
async def update_combo(
    combo_id: str,
    body: ComboUpdate,
    if_match: str | None = Header(default=None, alias="If-Match"),
    admin: str = Depends(verify_admin_key),
    db: AsyncSession = Depends(get_db),
):
    """Update a router combo.

    Optimistic concurrency: pass the stored combo version via If-Match (or
    expected_version in the body). A mismatch returns 409 with the current
    version so the caller can re-read and retry.
    """
    existing = await crud.get_router_combo(db, combo_id)
    if not existing:
        raise HTTPException(404, "Combo not found")
    expected_version = _expected_version(body, if_match)

    models = _dedupe_models([m.model_dump() for m in body.models]) if body.models is not None else None
    agentic = _agentic_from_config(body.agentic_config)
    mode, strategy, judge, clear_judge = _routing_from_body(body)

    # Validate against the post-update state: routing and model list can both
    # change in one call, and fusion's N+1 requirement depends on the final list.
    eff_mode = mode or (existing.routing_mode or "group")
    eff_strategy = strategy or (existing.routing_strategy or "fallback")
    eff_count = len(models) if models is not None else len(existing.models or [])
    if clear_judge:
        eff_judge = None
    elif judge is not None:
        eff_judge = judge
    else:
        eff_judge = existing.judge_model
    _validate_routing(existing.name, eff_count, eff_mode, eff_strategy, eff_judge)

    changed = {
        k: v for k, v in {
            "name": body.name, "models": models, "routing_mode": mode,
            "routing_strategy": strategy, "judge_model": judge,
            "monthly_token_budget": body.monthly_token_budget,
            "tools_allowed": body.tools_allowed,
        }.items() if v is not None
    }
    try:
        combo = await crud.update_router_combo(
            db, combo_id,
            expected_version=expected_version,
            name=body.name,
            models=models,
            routing_mode=mode,
            routing_strategy=strategy,
            judge_model=judge,
            clear_judge_model=clear_judge,
            max_iterations=agentic["max_iterations"] if body.agentic_config else None,
            tool_calling=agentic["tool_calling"] if body.agentic_config else None,
            deterministic=agentic["deterministic"] if body.agentic_config else None,
            loop_strategy=agentic["loop_strategy"] if body.agentic_config else None,
            compression_target_tokens=agentic["compression_target_tokens"] if body.agentic_config else None,
            load_spread=(body.agentic_config.load_spread if body.agentic_config else body.load_spread),
            monthly_token_budget=body.monthly_token_budget,
            tools_allowed=body.tools_allowed,
            strategy_preset=body.strategy_preset,
        )
    except ComboConflict as exc:
        await db.rollback()
        raise HTTPException(
            409,
            {"message": str(exc), "current_version": exc.current_version},
        ) from exc
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(409, f"A combo named '{body.name}' already exists") from exc
    if not combo:
        raise HTTPException(404, "Combo not found")
    await _audit_combo(db, admin, "update", combo_id, json.dumps({
        "expected_version": expected_version, "changed": changed,
    }))
    await db.commit()
    await refresh_active_combos(db)
    return _combo_to_response(combo)


@router.delete("/{combo_id}")
async def delete_combo(
    combo_id: str,
    admin: str = Depends(verify_admin_key),
    db: AsyncSession = Depends(get_db),
):
    """Delete a router combo.

    Dependent rows cascade (sessions, budgets, alerts, analytics) and scoped
    API keys are unscoped (combo_id -> NULL) by the FK actions, so the count
    of affected keys is reported back to the caller.
    """
    combo = await crud.get_router_combo(db, combo_id)
    if not combo:
        raise HTTPException(404, "Combo not found")
    scoped_keys = (await db.execute(
        select(func.count(DBApiKey.id)).where(DBApiKey.combo_id == combo_id)
    )).scalar() or 0
    deleted = await crud.delete_router_combo(db, combo_id)
    if not deleted:
        raise HTTPException(404, "Combo not found")
    await _audit_combo(
        db, admin, "delete", combo_id,
        json.dumps({"name": combo.name, "router_type": combo.router_type,
                    "unscoped_api_keys": int(scoped_keys)}),
    )
    await db.commit()
    await refresh_active_combos(db)
    return {"ok": True, "unscoped_api_keys": int(scoped_keys)}


@router.post("/{combo_id}/activate")
async def activate_combo(
    combo_id: str,
    admin: str = Depends(verify_admin_key),
    db: AsyncSession = Depends(get_db),
):
    """Activate a combo (deactivates others for same router_type)."""
    # The one-active-per-type violation surfaces at the crud-internal flush,
    # so the whole state change sits inside the 409 guard.
    try:
        combo = await crud.activate_router_combo(db, combo_id)
        if not combo:
            raise HTTPException(404, "Combo not found")
        await _audit_combo(
            db, admin, "activate", combo_id,
            json.dumps({"name": combo.name, "router_type": combo.router_type}),
        )
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(
            409,
            "Another combo for this router_type was activated concurrently; "
            "re-read and retry.",
        ) from exc
    await refresh_active_combos(db)
    return _combo_to_response(combo)


@router.post("/{combo_id}/deactivate")
async def deactivate_combo(
    combo_id: str,
    admin: str = Depends(verify_admin_key),
    db: AsyncSession = Depends(get_db),
):
    """Deactivate a combo."""
    combo = await crud.deactivate_router_combo(db, combo_id)
    if not combo:
        raise HTTPException(404, "Combo not found")
    await _audit_combo(db, admin, "deactivate", combo_id,
                       json.dumps({"name": combo.name}))
    await db.commit()
    await refresh_active_combos(db)
    return _combo_to_response(combo)


@router.post("/{combo_id}/test-agentic")
async def test_agentic_combo(
    combo_id: str,
    _: str = Depends(verify_admin_key),
    db: AsyncSession = Depends(get_db),
):
    """Test agentic combo routing (dry-run)."""
    from app.agentic_engine import AgenticLoop, AgenticResult
    combo = await crud.get_router_combo(db, combo_id)
    if not combo:
        raise HTTPException(404, "Combo not found")
    if not combo.agentic_enabled:
        raise HTTPException(400, "Agentic is not enabled for this combo")
    # Same shape as _combo_config in chat.py: entries missing provider or
    # model_id would produce (None, None) attempts that always fail upstream.
    models = [
        (m.get("provider"), m.get("model_id"))
        for m in (combo.models or [])
        if isinstance(m, dict) and m.get("provider") and m.get("model_id")
    ]
    monthly_token_budget = combo.monthly_token_budget if hasattr(combo, "monthly_token_budget") else 0

    # Build tool schemas (with descriptions) when tool_calling is enabled so the
    # dry-run matches what the real chat path sends to the models. The policy
    # comes from `get_tools_for_combo` — the same combo `tools_allowed`
    # authority the chat path enforces — so the dry-run cannot advertise a
    # tool the live request would refuse. A DB failure yields no tools (the
    # function fails closed), which is why there is no separate all-tools
    # fallback here anymore.
    tools = None
    if combo.tool_calling:
        try:
            from app.tools_registry import get_tools_for_combo
            tools = await get_tools_for_combo(combo.id) or None
        except Exception:
            tools = None

    engine = AgenticLoop(
        models=models,
        max_iterations=combo.max_iterations,
        compression_target_tokens=combo.compression_target_tokens,
        deterministic=combo.deterministic,
        tool_calling=combo.tool_calling,
        combo_id=combo.id,
        monthly_token_budget=monthly_token_budget,
        loop_strategy=combo.loop_strategy,
        tools=tools,
    )
    result = await engine.run([{"role": "user", "content": "test"}])
    return {
        "combo_id": combo.id,
        "name": combo.name,
        "iterations": result.iterations,
        "total_tokens": result.total_tokens,
        "compressed_steps": result.compressed_steps,
        "history": [
            {
                "iteration": s.iteration,
                "model": s.model,
                "provider": s.provider,
                "stop_reason": s.stop_reason,
                "content_length": len(s.content) if s.content else 0,
                "compressed": s.compressed,
            }
            for s in result.history
        ],
    }


@router.get("/{combo_id}/budget")
async def get_combo_budget(
    combo_id: str,
    _: str = Depends(verify_admin_key),
    db: AsyncSession = Depends(get_db),
):
    """Get full monthly token budget status with alerts, history, and replenishment prompt."""
    from app.token_budget_manager import TokenBudgetManager, BudgetAlertManager
    month_year = datetime.now().strftime("%Y-%m")

    status = await crud.get_budget_status(db, combo_id, month_year)
    if not status:
        return BudgetStatusResponse(
            combo_id=combo_id,
            month_year=month_year,
            tokens_used=0,
            tokens_budget=0,
            tokens_remaining=0,
            percentage=0.0,
            within_budget=True,
            no_budget_set=True,
            alerts=[],
            historical_usage=[],
            replenishment_prompt=None,
        )

    alerts_raw, _total = await crud.list_budget_alerts(db, combo_id=combo_id, limit=50, offset=0)
    alerts = []
    for a in alerts_raw:
        alerts.append(BudgetAlertResponse(
            id=a.id,
            combo_id=a.combo_id,
            month_year=a.month_year,
            level=a.level,
            percentage=float(a.percentage),
            tokens_used=a.tokens_used,
            tokens_budget=a.tokens_budget,
            triggered_at=a.triggered_at.isoformat() if a.triggered_at else None,
            resolved_at=a.resolved_at.isoformat() if a.resolved_at else None,
        ))

    tbm = TokenBudgetManager(combo_id)
    historical_usage = await tbm.get_historical_usage(12)

    daily_rate = 0.0
    today = datetime.now(timezone.utc)
    first_of_month = today.replace(day=1)
    days_elapsed_this_month = max(1, (today - first_of_month).days)
    if days_elapsed_this_month > 0:
        daily_rate = status["tokens_used"] / days_elapsed_this_month

    bam = BudgetAlertManager(combo_id)
    should_replenish = await bam.should_auto_replenish(combo_id)
    replenishment_prompt = None
    if should_replenish:
        replenishment_prompt = bam.get_replenishment_prompt(
            current_usage_pct=status["percentage"],
            daily_rate=daily_rate,
            tokens_budget=status["tokens_budget"],
            tokens_used=status["tokens_used"],
        )

    return BudgetStatusResponse(
        combo_id=combo_id,
        month_year=month_year,
        tokens_used=status["tokens_used"],
        tokens_budget=status["tokens_budget"],
        tokens_remaining=status["tokens_remaining"],
        percentage=float(status["percentage"]),
        within_budget=status["within_budget"],
        no_budget_set=False,
        alerts=alerts,
        historical_usage=historical_usage,
        replenishment_prompt=replenishment_prompt,
    )


@router.get("/{combo_id}/budget/alerts")
async def list_budget_alerts(
    combo_id: str,
    level: str | None = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    _: str = Depends(verify_admin_key),
    db: AsyncSession = Depends(get_db),
):
    """List paged budget alerts for a combo."""
    combo = await crud.get_router_combo(db, combo_id)
    if not combo:
        raise HTTPException(404, "Combo not found")
    alerts, total = await crud.list_budget_alerts(
        db, combo_id=combo_id, level=level, limit=limit, offset=offset,
    )
    data = []
    for a in alerts:
        data.append({
            "id": a.id,
            "combo_id": a.combo_id,
            "month_year": a.month_year,
            "level": a.level,
            "percentage": float(a.percentage),
            "tokens_used": a.tokens_used,
            "tokens_budget": a.tokens_budget,
            "triggered_at": a.triggered_at.isoformat() if a.triggered_at else None,
            "resolved_at": a.resolved_at.isoformat() if a.resolved_at else None,
        })
    return {"data": data, "total": total}


@router.post("/{combo_id}/budget/alerts/{alert_id}/resolve")
async def resolve_budget_alert(
    combo_id: str,
    alert_id: str,
    _: str = Depends(verify_admin_key),
    db: AsyncSession = Depends(get_db),
):
    """Resolve a budget alert."""
    combo = await crud.get_router_combo(db, combo_id)
    if not combo:
        raise HTTPException(404, "Combo not found")
    ok = await crud.resolve_budget_alert(db, alert_id)
    if not ok:
        raise HTTPException(404, "Alert not found")
    await db.commit()
    return {"ok": True}


@router.get("/{combo_id}/budget/history")
async def get_budget_history(
    combo_id: str,
    months: int = Query(12, ge=1, le=36),
    _: str = Depends(verify_admin_key),
):
    """Get historical token usage for a combo."""
    from app.token_budget_manager import TokenBudgetManager
    tbm = TokenBudgetManager(combo_id)
    history = await tbm.get_historical_usage(months)
    return {"combo_id": combo_id, "months": months, "history": history}


@router.get("/{combo_id}/estimate")
async def estimate_combo_cost(
    combo_id: str,
    _: str = Depends(verify_admin_key),
    db: AsyncSession = Depends(get_db),
):
    """Estimate agentic cost for a combo."""
    from app.agentic_engine import estimate_agentic_cost
    combo = await crud.get_router_combo(db, combo_id)
    if not combo:
        raise HTTPException(404, "Combo not found")
    models = [(m.get("provider"), m.get("model_id")) for m in combo.models]
    estimate = estimate_agentic_cost(models, combo.max_iterations)
    return {
        "combo_id": combo_id,
        **estimate,
    }


@router.get("/{combo_id}/tools")
async def get_combo_tools(
    combo_id: str,
    _: str = Depends(verify_admin_key),
):
    """Get tools enabled for a combo (OpenAI-compatible schemas)."""
    from app.tools_registry import get_tools_for_combo
    tools = await get_tools_for_combo(combo_id)
    return {"combo_id": combo_id, "tools": tools}


class ComboToolsUpdate(BaseModel):
    tools: list[str]


@router.put("/{combo_id}/tools")
async def update_combo_tools(
    combo_id: str,
    body: ComboToolsUpdate,
    admin: str = Depends(verify_admin_key),
    db: AsyncSession = Depends(get_db),
):
    """Set tools allowed for a combo."""
    combo = await crud.get_router_combo(db, combo_id)
    if not combo:
        raise HTTPException(404, "Combo not found")
    ok = await crud.set_combo_tools(db, combo_id, body.tools)
    if not ok:
        raise HTTPException(404, "Combo not found")
    await _audit_combo(
        db, admin, "tools.update", combo_id,
        json.dumps({"tools_allowed": body.tools}),
    )
    await db.commit()
    return {"ok": True, "combo_id": combo_id, "tools_allowed": body.tools}
