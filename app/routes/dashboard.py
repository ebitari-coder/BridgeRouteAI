from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.middleware.auth import verify_admin_key
from app.models import DashboardStats, ChartDataPoint, AgenticStats, ComboPerformance, ModelRankingEntry
from app.crud import (
    get_request_stats,
    get_chart_data,
    get_agentic_overall_stats,
    get_combo_performance,
    get_model_rankings,
    get_router_combos,
)
from app.providers import (
    get_provider_health,
    get_all_models,
    get_provider,
    ensure_catalogs,
    PROVIDER_KILO,
    PROVIDER_FREEMODELS,
    PROVIDER_CLOUDFLARE,
    PROVIDER_BAZAARLINK,
    PROVIDER_UNOROUTER,
    PROVIDER_KIOSAPI,
    PROVIDER_XKIRO,
    PROVIDER_MODELSCOPE,
    PROVIDER_LLM7,
    PROVIDER_AGNES,
    PROVIDER_CHATBOT,
    PROVIDER_AGENTIC,
    PROVIDER_CODING,
    PROVIDER_MODELROUTER,
    PROVIDER_TIARINA,
    PROVIDER_COMBO,
    CHATBOT_MODEL,
    AGENTIC_MODEL,
    CODING_MODEL,
    CHATBOT_MODELS,
    AGENTIC_MODELS,
    CODING_MODELS,
    get_model_health,
)
from app import model_capability as mc

router = APIRouter(prefix="/admin/dashboard", tags=["Dashboard"])


@router.get("/stats", response_model=DashboardStats)
async def dashboard_stats(
    _: str = Depends(verify_admin_key),
    db: AsyncSession = Depends(get_db),
):
    """Get dashboard overview statistics."""
    stats = await get_request_stats(db)
    return DashboardStats(**stats)


@router.get("/chart", response_model=list[ChartDataPoint])
async def dashboard_chart(
    hours: int = 24,
    _: str = Depends(verify_admin_key),
    db: AsyncSession = Depends(get_db),
):
    """Get chart data for the last N hours."""
    data = await get_chart_data(db, hours)
    return [ChartDataPoint(**d) for d in data]


@router.get("/providers")
async def provider_health(_: str = Depends(verify_admin_key)):
    """Get health status of the upstream providers and their models."""
    await ensure_catalogs()
    health = get_provider_health()
    provider_health_map = health.get("providers", {})

    providers = {}
    for model in get_all_models():
        p = model["owned_by"]
        if p not in providers:
            ph = provider_health_map.get(p, {})
            providers[p] = {
                "name": p,
                "healthy": ph.get("healthy", True),
                "consecutive_failures": ph.get("consecutive_failures", 0),
                "last_error": ph.get("last_error"),
                "total_fallbacks": ph.get("total_fallbacks", 0),
                "recovers_at": ph.get("recovers_at"),
                "models": [],
            }
        providers[p]["models"].append({
            "id": model["id"],
            "healthy": model.get("healthy", True),
            "tools_supported": model.get("tools_supported", False),
        })

    # Build fallback tiers from provider routing hierarchy
    kilo_model_ids = [m["id"] for m in get_all_models() if m["owned_by"] == PROVIDER_KILO]
    cf_model_ids = [m["id"] for m in get_all_models() if m["owned_by"] == PROVIDER_CLOUDFLARE]
    uo_model_ids = [m["id"] for m in get_all_models() if m["owned_by"] == PROVIDER_UNOROUTER]
    ka_model_ids = [m["id"] for m in get_all_models() if m["owned_by"] == PROVIDER_KIOSAPI]
    xk_model_ids = [m["id"] for m in get_all_models() if m["owned_by"] == PROVIDER_XKIRO]
    ms_model_ids = [m["id"] for m in get_all_models() if m["owned_by"] == PROVIDER_MODELSCOPE]
    llm7_model_ids = [m["id"] for m in get_all_models() if m["owned_by"] == PROVIDER_LLM7]
    agnes_model_ids = [m["id"] for m in get_all_models() if m["owned_by"] == PROVIDER_AGNES]
    bl_model_ids = [m["id"] for m in get_all_models() if m["owned_by"] == PROVIDER_BAZAARLINK]
    fm_model_ids = [m["id"] for m in get_all_models() if m["owned_by"] == PROVIDER_FREEMODELS]
    chatbot_model_ids = [m["id"] for m in get_all_models() if m["owned_by"] == PROVIDER_CHATBOT]
    agentic_model_ids = [m["id"] for m in get_all_models() if m["owned_by"] == PROVIDER_AGENTIC]
    coding_model_ids = [m["id"] for m in get_all_models() if m["owned_by"] == PROVIDER_CODING]
    modelrouter_model_ids = [m["id"] for m in get_all_models() if m["owned_by"] == PROVIDER_MODELROUTER]
    tiarina_model_ids = [m["id"] for m in get_all_models() if m["owned_by"] == PROVIDER_TIARINA]

    fallback_tiers = [
        {"tier": 0, "models": kilo_model_ids},
        {"tier": 1, "models": cf_model_ids},
        {"tier": 2, "models": uo_model_ids},
        {"tier": 3, "models": ka_model_ids},
        {"tier": 4, "models": xk_model_ids},
        {"tier": 5, "models": ms_model_ids},
        {"tier": 6, "models": llm7_model_ids},
        {"tier": 7, "models": agnes_model_ids},
        {"tier": 8, "models": bl_model_ids},
        {"tier": 9, "models": fm_model_ids},
        {"tier": 10, "models": chatbot_model_ids},
        {"tier": 11, "models": agentic_model_ids},
        {"tier": 12, "models": coding_model_ids},
        {"tier": 13, "models": modelrouter_model_ids},
        {"tier": 14, "models": tiarina_model_ids},
    ]

    return {"providers": list(providers.values()), "fallback_tiers": fallback_tiers}


def _build_router_models(router_name: str, model_list: list[tuple[str, str]]) -> dict:
    """Build a detailed model list for a dedicated router."""
    all_models = {m["id"]: m for m in get_all_models()}
    models = []
    for provider, model_id in model_list:
        m = all_models.get(model_id, {})
        health = get_model_health(model_id)
        models.append({
            "id": model_id,
            "provider": provider,
            "healthy": health.is_healthy,
            "tools_supported": m.get("tools_supported", provider in (PROVIDER_XKIRO, PROVIDER_KIOSAPI)),
            "context_length": m.get("context_length"),
            "fallback_position": len(models) + 1,
        })
    return {
        "router": router_name,
        "model_count": len(models),
        "models": models,
    }


@router.get("/models/chatbot")
async def chatbot_models(_: str = Depends(verify_admin_key)):
    """List all models under the Chatbot dedicated router."""
    return _build_router_models(CHATBOT_MODEL, CHATBOT_MODELS)


@router.get("/models/agentic")
async def agentic_models(_: str = Depends(verify_admin_key)):
    """List all models under the Agentic AI dedicated router."""
    return _build_router_models(AGENTIC_MODEL, AGENTIC_MODELS)


@router.get("/models/coding")
async def coding_models(_: str = Depends(verify_admin_key)):
    """List all models under the Coding dedicated router."""
    return _build_router_models(CODING_MODEL, CODING_MODELS)


@router.get("/agentic", response_model=AgenticStats)
async def agentic_stats(
    months: int = Query(1, gt=0, le=36),
    _: str = Depends(verify_admin_key),
    db: AsyncSession = Depends(get_db),
):
    """Get agentic engine overall statistics for the last N months."""
    stats = await get_agentic_overall_stats(db, months)
    return AgenticStats(**stats)


@router.get("/agentic/combos")
async def agentic_combos_performance(
    months: int = Query(1, gt=0, le=36),
    _: str = Depends(verify_admin_key),
    db: AsyncSession = Depends(get_db),
):
    """Get agentic performance across all combos for the last N months."""
    combos = await get_router_combos(db)
    results = []
    for combo in combos:
        perf = await get_combo_performance(db, combo.id, months)
        results.extend([ComboPerformance(**p).model_dump() for p in perf])
    return {"data": results}


@router.get("/agentic/combos/{combo_id}")
async def agentic_combo_performance(
    combo_id: str,
    months: int = Query(1, gt=0, le=36),
    _: str = Depends(verify_admin_key),
    db: AsyncSession = Depends(get_db),
):
    """Get agentic performance for a specific combo over the last N months."""
    perf = await get_combo_performance(db, combo_id, months)
    data = [ComboPerformance(**p).model_dump() for p in perf]
    return {"combo_id": combo_id, "data": data}


@router.get("/agentic/combos/{combo_id}/rankings", response_model=list[ModelRankingEntry])
async def agentic_combo_rankings(
    combo_id: str,
    months: int = Query(1, gt=0, le=36),
    _: str = Depends(verify_admin_key),
    db: AsyncSession = Depends(get_db),
):
    """Get model rankings (by success rate) for a specific combo."""
    rankings = await get_model_rankings(db, combo_id, months)
    sorted_rankings = sorted(rankings, key=lambda x: x["success_rate"], reverse=True)
    return [ModelRankingEntry(**r) for r in sorted_rankings]


@router.get("/capabilities")
async def capability_probes(_: str = Depends(verify_admin_key)):
    """Get capability probe state for all known models.

    Shows whether each model has been probed for tool support and the
    observed result (True/False/None=unprobed).
    """
    from app.providers import get_all_models
    results = []
    for m in get_all_models():
        owned_by = m.get("owned_by", "")
        probed = mc.model_supports_tools(owned_by, m["id"])
        results.append({
            "id": m["id"],
            "provider": owned_by,
            "catalog_claimed_tools": m.get("tools_supported", False),
            "probed_supports_tools": probed,
            "probe_matches_catalog": probed is None or probed == m.get("tools_supported", False),
        })
    # Sort by provider then model id
    results.sort(key=lambda x: (x["provider"], x["id"]))
    return {"capabilities": results}


@router.post("/capabilities/warmup")
async def warmup_capabilities(_: str = Depends(verify_admin_key)):
    """Trigger a manual capability probe warmup for all free models."""
    from app.providers import get_all_models
    models = [(m.get("owned_by", ""), m["id"]) for m in get_all_models() if m.get("owned_by")]
    await mc.warm_capability_cache(models)
    return {"probed_count": len(models), "status": "complete"}


@router.get("/provider-tokens")
async def provider_token_totals(
    _: str = Depends(verify_admin_key),
    db: AsyncSession = Depends(get_db),
):
    """Token totals per provider, restricted to providers whose models are
    stable and tool-capable — freemodels is excluded (policy layer only
    publishes mirror-catalogs, and its stability signal is not ours)."""
    from sqlalchemy import text as sa_text
    from app.providers import stream_health

    rows = (await db.execute(
        sa_text(
            "SELECT model, COALESCE(SUM(tokens_used),0) AS tokens "
            "FROM request_logs WHERE tokens_used IS NOT NULL AND model <> '' "
            "GROUP BY model"
        )
    )).all()

    catalog = {m["id"]: m for m in get_all_models()}
    per_provider: dict[str, int] = {}
    for r in rows:
        m_id, tokens = r[0], int(r[1] or 0)
        entry = catalog.get(m_id)
        if not entry:
            continue
        per_provider[entry["owned_by"]] = per_provider.get(entry["owned_by"], 0) + tokens

    # Provider 'stable + tool-capable' filter:
    #  - freemodels excluded by policy
    #  - tool-capable: catalog claims tools OR a probe confirmed it
    #  - stable: not degraded in stream telemetry (>=3 samples, success<0.5)
    providers_out: list[dict] = []
    for m in get_all_models():
        p = m["owned_by"]
        if p in (PROVIDER_FREEMODELS, PROVIDER_COMBO, PROVIDER_CHATBOT, "freemodels"):
            continue
        claimed = m.get("tools_supported", False)
        probed = mc.model_supports_tools(p, m["id"])
        if not (claimed or probed is True):
            continue
        h = stream_health(p, m["id"])
        stable = not (h and h["samples"] >= 3 and h["success_rate"] < 0.5)
        providers_out.append({
            "provider": p,
            "model": m["id"],
            "tool_capable": True,
            "probed": probed,
            "stable": stable,
            "tokens_total": per_provider.get(p, 0),
        })

    # dedupe to provider level, keep the best-known model + aggregate tokens
    by_provider: dict[str, dict] = {}
    for row in providers_out:
        cur = by_provider.get(row["provider"])
        if cur is None or (row["probed"] and not cur["probed"]):
            by_provider[row["provider"]] = row
    totals = sorted(
        ({"provider": p, "tokens_total": per_provider.get(p, 0), **row}
         for p, row in by_provider.items()),
        key=lambda x: x["tokens_total"], reverse=True,
    )
    for t in totals: t.pop("model", None); t.pop("probed", None)
    return {"data": totals}
