"""Upstream provider routing with intelligent failover.

Primary upstream: the Kilo AI Gateway (see ``app.kilo``).
Fallback upstream: FreeModels.

Requests for the virtual ``bridgeroute/auto`` model are routed across every
discovered free Kilo model — ranked by capability and skipped when their
circuit breaker is open — and fall back to FreeModels if Kilo is exhausted.
"""
import uuid
import time
import logging
import json
from typing import AsyncIterator
from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession
import httpx

from app.config import settings
from app import kilo
from app import cloudflare as cf
from app import bazaarlink as bl
from app import unorouter as uo
from app import kiosapi as ka
from app import xkiro as xk
from app import modelscope as ms
from app import llm7
from app import agnes
from app import freemodels as fm
from app import composite as cx
from app import septorlabs as sx
from app import routeway as rw
from app import toptools as tt
from app import dreamprompting as dp
from app import cleanapis as cp
from app import modelrouter as mr
from app import tiarina as tr
from app import zyloai as zy
from app import model_capability as mc
from app.provider_utils import (
    QuotaError,
    SamplingParams,
    apply_sampling,
    sse_chunk,
    to_openai_messages,
    usage_chunk,
)

logger = logging.getLogger(__name__)

PROVIDER_KILO = "kilo"
PROVIDER_FREEMODELS = "freemodels"
PROVIDER_CLOUDFLARE = "cloudflare"
PROVIDER_BAZAARLINK = "bazaarlink"
PROVIDER_UNOROUTER = "unorouter"
PROVIDER_KIOSAPI = "kiosapi"
PROVIDER_XKIRO = "xkiro"
PROVIDER_MODELSCOPE = "modelscope"
PROVIDER_LLM7 = "llm7"
PROVIDER_AGNES = "agnes"
PROVIDER_COMPOSITE = "composite"
PROVIDER_SEPTORLABS = "septorlabs"
PROVIDER_ROUTEWAY = "routeway"
PROVIDER_TOPTOOLS = "toptools"
PROVIDER_DREAMPROMPTING = "dreamprompting"
PROVIDER_CLEANAPIS = "cleanapis"
PROVIDER_MODELROUTER = "modelrouter"
PROVIDER_TIARINA = "tiarina"
PROVIDER_ZYLOAI = "zyloai"
PROVIDER_CHATBOT = "chatbot"
PROVIDER_AGENTIC = "agentic"
PROVIDER_CODING = "coding"
# Pseudo-provider: a combo name resolved as a model id (no single upstream).
PROVIDER_COMBO = "combo"

# Cap on upstream attempts per request so a fully-exhausted pool cannot stall
# a request indefinitely.
MAX_ATTEMPTS = 8

# Virtual model identifiers for dedicated use-case routing
CHATBOT_MODEL = "chatbot"
AGENTIC_MODEL = "agentic"
CODING_MODEL = "coding"

# In-memory cache for active combos (router_type -> list of (provider, model_id))
_active_combos: dict[str, list[tuple[str, str]]] = {}

# Active-combo resolution index, rebuilt by refresh_active_combos(). Entries are
# ordered agentic-first then newest; see get_combo_index().
_combo_index: list[dict] = []

# Curated model lists for dedicated use-case routing.
# Chatbot: fast, high-quality conversational models (no tools needed).
CHATBOT_MODELS = [
    (PROVIDER_FREEMODELS, "claude-sonnet-5"),
    (PROVIDER_FREEMODELS, "gpt-5.2-high"),
    (PROVIDER_FREEMODELS, "gpt-5.5-instant"),
    (PROVIDER_FREEMODELS, "gemini-3.6-flash"),
    (PROVIDER_FREEMODELS, "mistral-small-2603"),
    (PROVIDER_FREEMODELS, "qwen3.7-plus"),
    (PROVIDER_FREEMODELS, "mimo-v2.5"),
    (PROVIDER_FREEMODELS, "glm-5.1"),
    (PROVIDER_FREEMODELS, "minimax-m3"),
    (PROVIDER_FREEMODELS, "claude-haiku-4-5-20251001"),
    (PROVIDER_AGNES, "agnes-3.0-flash"),
    (PROVIDER_AGNES, "agnes-2.5-flash"),
    (PROVIDER_CLOUDFLARE, "@cf/meta/llama-4-scout-17b-16e-instruct"),
    (PROVIDER_CLOUDFLARE, "@cf/meta/gemma-4-26b-a4b-it"),
    (PROVIDER_CLOUDFLARE, "@cf/zai-org/glm-5.3-flash"),
    (PROVIDER_CLOUDFLARE, "@cf/meta/kimi-k2.6"),
    (PROVIDER_LLM7, "laguna-s-2.1"),
    (PROVIDER_LLM7, "lfm-2.5-2.6b"),
]


# Static seed for the FreeModels upstream (see ``app.freemodels``). The live
# catalog is scraped from freemodels.pro and merged with this seed; any other
# model name is still forwarded verbatim via the ``modelId`` field.
FREEMODELS_MODELS = [m["id"] for m in fm.STATIC_MODELS]

# Agentic: tool-calling + reasoning models for autonomous agents.
AGENTIC_MODELS = [
    (PROVIDER_AGNES, "agnes-3.0-flash"),
    (PROVIDER_AGNES, "agnes-2.5-flash"),
    (PROVIDER_CLOUDFLARE, "@cf/zai-org/glm-5.3"),
    (PROVIDER_CLOUDFLARE, "@cf/deepseek/deepseek-v4-pro-0813"),
    (PROVIDER_CLOUDFLARE, "@cf/meta/kimi-k2.7-code"),
    (PROVIDER_CLOUDFLARE, "@cf/deepseek/deepseek-v4-flash-0731"),
    (PROVIDER_FREEMODELS, "deepseek-v4-pro-thinking"),
    (PROVIDER_FREEMODELS, "claude-sonnet-5"),
    (PROVIDER_FREEMODELS, "gpt-5.2-high"),
    (PROVIDER_FREEMODELS, "o3-2025-04-16"),
    (PROVIDER_FREEMODELS, "qwen3.7-max"),
]

# Coding: code generation + tool-calling for Agentic Development Environments.
CODING_MODELS = [
    (PROVIDER_FREEMODELS, "claude-sonnet-5"),
    (PROVIDER_FREEMODELS, "deepseek-v4-pro-thinking"),
    (PROVIDER_FREEMODELS, "gpt-5.2-high"),
    (PROVIDER_FREEMODELS, "o3-2025-04-16"),
    (PROVIDER_FREEMODELS, "qwen3.7-max"),
    (PROVIDER_AGNES, "agnes-3.0-flash"),
    (PROVIDER_AGNES, "agnes-2.5-flash"),
    (PROVIDER_CLOUDFLARE, "@cf/meta/kimi-k2.7-code"),
    (PROVIDER_CLOUDFLARE, "@cf/deepseek/deepseek-v4-pro-0813"),
    (PROVIDER_CLOUDFLARE, "@cf/deepseek/deepseek-v4-flash-0731"),
    (PROVIDER_CLOUDFLARE, "@cf/zai-org/glm-5.3"),
    (PROVIDER_CLOUDFLARE, "@cf/qwen/qwen3.8-27b"),
    (PROVIDER_XKIRO, "mistralai/codestral-2508"),
    (PROVIDER_XKIRO, "mistralai/devstral-medium"),
    (PROVIDER_KIOSAPI, "north-mini-code"),
    (PROVIDER_LLM7, "codestral-latest"),
]


def get_active_combo_models(router_type: str) -> list[tuple[str, str]] | None:
    """Return the active combo models for a router type, or None if no active combo."""
    return _active_combos.get(router_type)


def get_combo_index() -> list[dict]:
    """Active combos, ordered for address resolution.

    Ordered agentic-first, then newest: when a name or model id is claimed by
    more than one active combo the first entry wins, so an agentic combo
    shadows a plain group combo for the same reference.
    """
    return _combo_index


def resolve_combo_ref(ref: str) -> dict | None:
    """Resolve a combo by its name — the combo name is a routable model id.

    Exact match first, then case-insensitive. ``ref`` may be ``None``/empty,
    in which case no combo matches.
    """
    if not ref:
        return None
    lowered = ref.lower()
    ci = None
    for entry in _combo_index:
        if entry["name"] == ref:
            return entry
        if ci is None and entry["name"].lower() == lowered:
            ci = entry
    return ci


def find_combo_for_model(model: str) -> dict | None:
    """Resolve a combo that lists ``model`` among its models[] entries."""
    if not model:
        return None
    for entry in _combo_index:
        if model in entry["model_ids"]:
            return entry
    return None


def get_combo_index_entry(combo_id: str) -> dict | None:
    """Look up a cached active combo by id (combo-scoped API keys)."""
    if not combo_id:
        return None
    for entry in _combo_index:
        if entry["id"] == combo_id:
            return entry
    return None


async def refresh_active_combos(db_session: AsyncSession | None = None) -> None:
    """Refresh the active combo cache. Call after combo changes.

    Builds two views: ``_active_combos`` (router_type -> models) for the legacy
    virtual chatbot/agentic/coding model ids, and ``_combo_index`` (list of
    active combo records) which is what request routing resolves against —
    combo name, combo-scoped API key, and models[] membership.
    """
    global _active_combos, _combo_index
    _active_combos = {}
    _combo_index = []
    try:
        from app import crud as app_crud

        async def _load(session: AsyncSession) -> None:
            combos = await app_crud.get_router_combos(session)
            for combo in combos:
                if not (combo.is_active and combo.models):
                    continue

                pairs = [
                    (m.get("provider"), m.get("model_id"))
                    for m in combo.models
                    if m.get("provider") and m.get("model_id")
                ]
                if not pairs:
                    continue

                # Newest active combo per router_type wins (rows come back
                # created_at DESC, so only fill a bucket once).
                if combo.router_type not in _active_combos:
                    _active_combos[combo.router_type] = pairs

                mode = combo.routing_mode or "group"
                _combo_index.append({
                    "id": combo.id,
                    "name": combo.name,
                    "router_type": combo.router_type,
                    "models": pairs,
                    "model_ids": [m for _p, m in pairs],
                    "routing_mode": mode,
                    "routing_strategy": combo.routing_strategy or "fallback",
                    "judge_model": combo.judge_model,
                    "is_agentic": mode == "agentic",
                    "created_at": combo.created_at,
                })

            # Agentic first, then newest — resolution tie-break for a shared
            # name or a model id claimed by several active combos.
            _combo_index.sort(
                key=lambda e: (not e["is_agentic"], -(e["created_at"].timestamp() if e["created_at"] else 0)),
            )

        if db_session is not None:
            await _load(db_session)
        else:
            from app.database import async_session
            async with async_session() as session:
                await _load(session)
    except Exception as e:
        logger.warning("Failed to refresh active combos: %s", e)


def freemodels_pool() -> list[str]:
    """Current FreeModels candidate ids: live catalog first, seed as fallback."""
    return fm.get_model_ids()


# ── Provider Health Tracking ────────────────────────────────────────────────

@dataclass
class ProviderHealth:
    """Track provider health with circuit breaker pattern."""
    exhausted_until: float = 0  # timestamp when provider recovers
    consecutive_failures: int = 0
    last_error: str = ""
    total_fallbacks: int = 0

    @property
    def is_healthy(self) -> bool:
        return time.time() >= self.exhausted_until

    def mark_exhausted(self, error: str, backoff_seconds: float = 30):
        """Mark provider as exhausted with exponential backoff."""
        self.consecutive_failures += 1
        self.last_error = error
        # Exponential backoff: 30s, 60s, 120s, max 300s
        backoff = min(backoff_seconds * (2 ** (self.consecutive_failures - 1)), 300)
        self.exhausted_until = time.time() + backoff
        logger.warning("Provider marked exhausted for %.0fs: %s", backoff, error[:100])

    def mark_healthy(self):
        """Reset health status on successful request."""
        if self.consecutive_failures > 0:
            logger.info("Provider recovered after %d failures", self.consecutive_failures)
        self.consecutive_failures = 0
        self.exhausted_until = 0
        self.last_error = ""

    def record_fallback(self):
        self.total_fallbacks += 1


# Global health tracking per provider AND per model
_health: dict[str, ProviderHealth] = {
    PROVIDER_KILO: ProviderHealth(),
    PROVIDER_FREEMODELS: ProviderHealth(),
    PROVIDER_CLOUDFLARE: ProviderHealth(),
    PROVIDER_BAZAARLINK: ProviderHealth(),
    PROVIDER_UNOROUTER: ProviderHealth(),
    PROVIDER_KIOSAPI: ProviderHealth(),
    PROVIDER_XKIRO: ProviderHealth(),
    PROVIDER_MODELSCOPE: ProviderHealth(),
    PROVIDER_LLM7: ProviderHealth(),
    PROVIDER_AGNES: ProviderHealth(),
    PROVIDER_COMPOSITE: ProviderHealth(),
    PROVIDER_SEPTORLABS: ProviderHealth(),
    PROVIDER_ROUTEWAY: ProviderHealth(),
    PROVIDER_TOPTOOLS: ProviderHealth(),
    PROVIDER_DREAMPROMPTING: ProviderHealth(),
    PROVIDER_CLEANAPIS: ProviderHealth(),
    PROVIDER_MODELROUTER: ProviderHealth(),
    PROVIDER_TIARINA: ProviderHealth(),
    PROVIDER_ZYLOAI: ProviderHealth(),
    PROVIDER_CHATBOT: ProviderHealth(),
    PROVIDER_AGENTIC: ProviderHealth(),
    PROVIDER_CODING: ProviderHealth(),
}
_model_health: dict[str, ProviderHealth] = {}


# ── Rolling latency / success telemetry per (provider, model) ───────────────
# Bounded sample deque with a 5-minute window, feeding the ordering logic:
# candidates whose recent success rate collapsed sink below their siblings
# ("Anthropic is slow -> route to Bedrock", provider-parity edition).
_STREAM_SAMPLES: dict[tuple[str, str], "deque"] = {}
_STREAM_SAMPLE_MAXLEN = 60
_STREAM_WINDOW_S = 300.0


def record_stream_latency(provider: str, model: str, latency_ms: float, ok: bool) -> None:
    """Append a (ts, latency_ms, ok) sample for a completed upstream stream."""
    from collections import deque
    key = (provider, model)
    samples = _STREAM_SAMPLES.setdefault(key, deque(maxlen=_STREAM_SAMPLE_MAXLEN))
    samples.append((time.time(), latency_ms, ok))


def stream_health(provider: str, model: str) -> dict | None:
    """5-minute rolling health for a (provider, model): success rate and p50 latency."""
    from collections import deque
    samples = _STREAM_SAMPLES.get((provider, model), deque(maxlen=_STREAM_SAMPLE_MAXLEN))
    now = time.time()
    recent = [(lat, ok) for ts, lat, ok in samples if now - ts <= _STREAM_WINDOW_S]
    if not recent:
        return None
    oks = sum(1 for _, ok in recent if ok)
    lats = sorted(lat for lat, _ in recent)
    p50 = lats[len(lats) // 2]
    return {
        "success_rate": round(oks / len(recent), 3),
        "p50_latency_ms": round(p50),
        "max_latency_ms": round(max(lats)),
        "samples": len(recent),
    }


def sink_degraded(plan: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Stable reordering: candidates with degraded recent health move last.

    A candidate is degraded when it has >= 3 recent samples and a success
    rate below 0.5. Unsampled candidates are untouched, so cold-start
    behaviour is identical to the declared ranking. Explicitly requested
    models keep their first attempt via _candidate_list's protect_first.
    """
    def key(entry: tuple[str, str]) -> int:
        h = stream_health(entry[0], entry[1])
        if h and h["samples"] >= 3 and h["success_rate"] < 0.5:
            return 1
        return 0
    return sorted(plan, key=key)


def get_model_health(model: str) -> ProviderHealth:
    """Get or create health tracker for a specific model."""
    if model not in _model_health:
        _model_health[model] = ProviderHealth()
    return _model_health[model]


def get_provider_health() -> dict:
    """Return health status for all providers and tracked models."""
    # Rolling stream telemetry per (provider, model) — the routing layer's
    # view of who is actually slow/failing right now.
    stream: dict = {}
    for (p, m), _samples in _STREAM_SAMPLES.items():
        h = stream_health(p, m)
        if h:
            stream[f"{p}:{m}"] = h

    return {
        "stream": stream,
        "providers": {
            name: {
                "healthy": h.is_healthy,
                "consecutive_failures": h.consecutive_failures,
                "last_error": h.last_error[:200] if h.last_error else None,
                "total_fallbacks": h.total_fallbacks,
                "recovers_at": h.exhausted_until if h.exhausted_until > 0 else None,
            }
            for name, h in _health.items()
        },
        "models": {
            name: {
                "healthy": h.is_healthy,
                "consecutive_failures": h.consecutive_failures,
                "last_error": h.last_error[:200] if h.last_error else None,
                "recovers_at": h.exhausted_until if h.exhausted_until > 0 else None,
            }
            for name, h in _model_health.items()
        },
    }


def get_provider(model: str) -> str:
    """Return the primary upstream provider for a model."""
    if model == CHATBOT_MODEL:
        return PROVIDER_CHATBOT
    if model == CODING_MODEL:
        return PROVIDER_CODING
    # A combo name is a routable id but is not a single upstream: the request
    # fans out over the combo's declared model list.
    if resolve_combo_ref(model) is not None:
        return PROVIDER_COMBO
    # ZyloAI's confirmed-working list is checked before the broad gateway
    # catalogs (Tiarina/ModelRouter publish colliding ids): the admin-verified
    # ZyloAI models are the intended upstream for these ids.
    if zy.is_zyloai_model(model):
        return PROVIDER_ZYLOAI
    if tr.is_tiarina_model(model):
        return PROVIDER_TIARINA
    # ModelRouter model ids can overlap with other gateway catalogs. When the
    # ModelRouter catalog claims an id, prefer the explicitly configured
    # ModelRouter upstream for that direct request.
    if mr.is_modelrouter_model(model):
        return PROVIDER_MODELROUTER
    # Composite claims ids it publishes before Kilo: the catalogs overlap
    # (OpenRouter-style `provider/model:free` ids).
    if cx.is_composite_model(model):
        return PROVIDER_COMPOSITE
    if sx.is_septorlabs_model(model):
        return PROVIDER_SEPTORLABS
    if rw.is_routeway_model(model):
        return PROVIDER_ROUTEWAY
    if tt.is_toptools_model(model):
        return PROVIDER_TOPTOOLS
    if dp.is_dreamprompting_model(model):
        return PROVIDER_DREAMPROMPTING
    if cp.is_cleanapis_model(model):
        return PROVIDER_CLEANAPIS
    if kilo.is_kilo_model(model):
        return PROVIDER_KILO
    if cf.is_cloudflare_model(model):
        return PROVIDER_CLOUDFLARE
    if uo.is_unorouter_model(model):
        return PROVIDER_UNOROUTER
    if agnes.is_agnes_model(model):
        return PROVIDER_AGNES
    if ka.is_kiosapi_model(model):
        return PROVIDER_KIOSAPI
    if xk.is_xkiro_model(model):
        return PROVIDER_XKIRO
    if ms.is_modelscope_model(model):
        return PROVIDER_MODELSCOPE
    if llm7.is_llm7_model(model):
        return PROVIDER_LLM7
    if bl.is_bazaarlink_model(model):
        return PROVIDER_BAZAARLINK
    return PROVIDER_FREEMODELS


async def ensure_catalogs() -> None:
    """Refresh upstream model catalogs (best effort)."""
    try:
        await kilo.ensure_models()
    except Exception as exc:
        logger.warning("Kilo catalog refresh failed: %s", exc)
    # Cloudflare uses a static model list; no refresh needed
    try:
        await ka.ensure_models()
    except Exception as exc:
        logger.warning("KiosAPI catalog refresh failed: %s", exc)
    try:
        await xk.ensure_models()
    except Exception as exc:
        logger.warning("Xkiro catalog refresh failed: %s", exc)
    try:
        await ms.ensure_models()
    except Exception as exc:
        logger.warning("ModelScope catalog refresh failed: %s", exc)
    try:
        await llm7.ensure_models()
    except Exception as exc:
        logger.warning("LLM7 catalog refresh failed: %s", exc)
    try:
        await agnes.ensure_models()
    except Exception as exc:
        logger.warning("Agnes AI catalog refresh failed: %s", exc)
    try:
        await fm.ensure_models()
    except Exception as exc:
        logger.warning("FreeModels catalog refresh failed: %s", exc)
    try:
        await cx.ensure_models()
    except Exception as exc:
        logger.warning("Composite catalog refresh failed: %s", exc)
    try:
        await sx.ensure_models()
    except Exception as exc:
        logger.warning("Septor Labs catalog refresh failed: %s", exc)
    try:
        await rw.ensure_models()
    except Exception as exc:
        logger.warning("Routeway catalog refresh failed: %s", exc)
    try:
        await tt.ensure_models()
    except Exception as exc:
        logger.warning("Top Tools AI catalog refresh failed: %s", exc)
    try:
        await dp.ensure_models()
    except Exception as exc:
        logger.warning("DreamPrompting catalog refresh failed: %s", exc)
    try:
        await cp.ensure_models()
    except Exception as exc:
        logger.warning("Clean APIs catalog refresh failed: %s", exc)
    try:
        await mr.ensure_models()
    except Exception as exc:
        logger.warning("ModelRouter catalog refresh failed: %s", exc)
    try:
        await tr.ensure_models()
    except Exception as exc:
        logger.warning("Tiarina catalog refresh failed: %s", exc)
    try:
        await zy.ensure_models()
    except Exception as exc:
        logger.warning("ZyloAI catalog refresh failed: %s", exc)
    try:
        await cf.ensure_models()
    except Exception as exc:
        logger.warning("Cloudflare catalog refresh failed: %s", exc)


def _dedupe(plan: list[tuple[str, str]]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen = set()
    for entry in plan:
        if entry in seen:
            continue
        seen.add(entry)
        out.append(entry)
    return out


# Per-provider slots taken in the first pass of ``_finalize``. Keeps the
# attempt list diverse even when one upstream publishes dozens of free
# models (Kilo alone currently lists 17).
_PER_PROVIDER_ATTEMPTS = 2


def _finalize(plan: list[tuple[str, str]], add_modelrouter: bool = True) -> list[tuple[str, str]]:
    """Dedupe, sink degraded candidates, and cap with provider diversity.

    A plain head slice (``_dedupe(plan)[:MAX_ATTEMPTS]``) would fill every
    slot with a single provider whenever one catalog is large, so a provider
    outage (e.g. a daily rate limit) would exhaust the whole plan. Take at
    most ``_PER_PROVIDER_ATTEMPTS`` entries per provider in plan order first,
    then fill any remaining slots from the leftovers, preserving order.

    Before capping, candidates with degraded recent stream health (>= 3
    samples, success rate < 0.5 over 5 minutes) sink below their siblings, so
    a latency-spiking upstream routes around itself without leaving the
    candidate set entirely.
    """
    # ModelRouter is another free-model pool. Append it before the diversity
    # pass so it participates in normal auto/failover routing without
    # displacing an explicitly requested model.
    if add_modelrouter:
        plan = list(plan) + [(PROVIDER_MODELROUTER, m) for m in mr.rank_free_models()]
        plan += [(PROVIDER_TIARINA, m) for m in tr.rank_models()]
    unique = sink_degraded(_dedupe(plan))

    out: list[tuple[str, str]] = []
    counts: dict[str, int] = {}
    for entry in unique:
        provider = entry[0]
        if counts.get(provider, 0) >= _PER_PROVIDER_ATTEMPTS:
            continue
        counts[provider] = counts.get(provider, 0) + 1
        out.append(entry)
        if len(out) >= MAX_ATTEMPTS:
            return out

    if len(out) < MAX_ATTEMPTS:
        chosen = set(out)
        for entry in unique:
            if entry in chosen:
                continue
            out.append(entry)
            if len(out) >= MAX_ATTEMPTS:
                break
    return out


def _get_stable_unorouter_models(need_tools: bool = False) -> list[str]:
    """Return UnoRouter models that pass stability checks (uptime/success)."""
    if not uo.STABLE_MODELS:
        return []
    models = uo.rank_free_models(need_tools)
    return [m for m in models if m in uo.STABLE_MODELS]


def _get_stable_xkiro_models(need_tools: bool = False) -> list[str]:
    """Return Xkiro models that pass stability checks (health probe).

    Cold-start allowance: a model the stability worker has never vetted
    (no recorded outcome) is kept at its ranked position — unknown is not
    failed. Only models with at least one recorded probe failure are
    dropped, so brand-new catalog entries (e.g. xkiro's fresh qwen3.x
    free-tool models) still route while the worker catches up.
    """
    # Cold start/total-refresh: if the worker has vetted nothing yet (e.g. a
    # probe burst rate-limited an entire pass), do NOT drop the provider —
    # fall back to the ranked order minus models with recorded failures.
    models = xk.rank_free_models(need_tools)
    failed = {m for m, h in xk._model_health.items() if (h.get("fail") or 0) > 0}
    return [m for m in models if m in xk.STABLE_MODELS or m not in failed]


def _get_stable_modelscope_models(need_tools: bool = False) -> list[str]:
    """Return ModelScope models that pass stability checks (health probe)."""
    if not ms.STABLE_MODELS:
        return []
    models = ms.rank_free_models(need_tools)
    return [m for m in models if m in ms.STABLE_MODELS]


def _get_stable_llm7_models(need_tools: bool = False) -> list[str]:
    """Return LLM7 models that pass stability checks (health probe)."""
    if not llm7.STABLE_MODELS:
        return []
    models = llm7.rank_free_models(need_tools)
    return [m for m in models if m in llm7.STABLE_MODELS]


_TOOLS_POOLS = {
    "tiarina": lambda: tr.rank_models(True),
    "modelrouter": lambda: mr.rank_free_models(True),
    "septorlabs": lambda: sx.rank_free_models(True),
    "routeway": lambda: rw.rank_free_models(True),
    "composite": lambda: cx.rank_free_models(True),
    "kiosapi": lambda: ka.rank_free_models(True),
    "agnes": lambda: agnes.rank_free_models(True),
    "bazaarlink": lambda: bl.rank_free_models(True),
    "cloudflare": lambda: cf.rank_free_models(True),
    "xkiro": lambda: _get_stable_xkiro_models(True),
    "modelscope": lambda: _get_stable_modelscope_models(True),
    "llm7": lambda: _get_stable_llm7_models(True),
    "unorouter": lambda: _get_stable_unorouter_models(True),
    # FreeModels is text-only: no model in its catalog can emit tool calls
    # (README: "FreeModels models are text-only — they cannot emit tool calls").
    # The pool must therefore be empty so `_first_tools_attempt` never leads
    # with FreeModels for a tool-bearing request; the plan falls through to the
    # tool-capable providers (Kilo et al.), which is the documented behavior.
    "freemodels": lambda: [],
}


def _first_tools_attempt(model: str) -> tuple[str, str] | None:
    """``(provider, model)`` when the explicitly requested model can serve tools.

    Tool-bearing requests prefer the tool-strong providers, but when the caller
    names a model that is itself tool-capable in its own catalog, that model
    gets the first attempt instead of being silently skipped.
    """
    if not model or model in (kilo.AUTO_MODEL, cf.CLOUDFLARE_AUTO_MODEL, bl.BAZAARLINK_AUTO_MODEL):
        return None
    if kilo.is_kilo_model(model):
        return (PROVIDER_KILO, model)
    if zy.is_zyloai_model(model):
        return (PROVIDER_ZYLOAI, model) if model in zy.rank_free_models(True) else None
    if mr.is_modelrouter_model(model):
        return (PROVIDER_MODELROUTER, model) if model in mr.rank_free_models(True) else None
    if tr.is_tiarina_model(model):
        return (PROVIDER_TIARINA, model) if model in tr.rank_models(True) else None
    try:
        provider = get_provider(model)
    except Exception:
        return None
    # Check probed capability first (live evidence), fall back to catalog pool
    probed = mc.model_supports_tools(provider, model)
    if probed is True:
        return (provider, model)
    pool = _TOOLS_POOLS.get(provider or "")
    if pool and model in pool():
        return (provider, model)
    return None


def get_attempt_plan(model: str, need_tools: bool = False) -> list[tuple[str, str]]:
    """Return the ordered ``(provider, model)`` attempts for a request.

    Kilo is preferred, then Cloudflare Workers AI free tier, then UnoRouter
    (only stable models), then KiosAPI, then Xkiro, then ModelScope (stable),
    then BazaarLink, then Composite (free models), then FreeModels.
    The requested model is always attempted first — tool-bearing (agentic)
    requests otherwise prefer Kilo, Composite, Septor, KiosAPI and Xkiro, and
    the requested model leads only when it is itself tool-capable.

    A combo name is a routable model id: when the model matches an active
    combo, that combo's declared list is the plan, in the admin's order.
    """
    combo = resolve_combo_ref(model)
    if combo is not None:
        return _dedupe(list(combo["models"]))[:MAX_ATTEMPTS]

    if need_tools:
        plan = []
        first = _first_tools_attempt(model)
        if first:
            plan.append(first)
        # Merge each provider's full ranking with its catalog tool-claim, then
        # let a live probe override the claim (see mc.ranked_tool_models). This
        # recovers models whose catalog omits `tools` but which the probe
        # observed emitting `tool_calls` — the provider's own
        # rank_free_models(need_tools=True) has already dropped them.
        plan += [(PROVIDER_KILO, m) for m in mc.ranked_tool_models(PROVIDER_KILO, kilo.rank_free_models(False), kilo.rank_free_models(True)) if m != model]
        plan += [(PROVIDER_MODELROUTER, m) for m in mc.ranked_tool_models(PROVIDER_MODELROUTER, mr.rank_free_models(False), mr.rank_free_models(True)) if m != model]
        plan += [(PROVIDER_TIARINA, m) for m in mc.ranked_tool_models(PROVIDER_TIARINA, tr.rank_models(False), tr.rank_models(True)) if m != model]
        plan += [(PROVIDER_COMPOSITE, m) for m in mc.ranked_tool_models(PROVIDER_COMPOSITE, cx.rank_free_models(False), cx.rank_free_models(True)) if m != model]
        plan += [(PROVIDER_SEPTORLABS, m) for m in mc.ranked_tool_models(PROVIDER_SEPTORLABS, sx.rank_free_models(False), sx.rank_free_models(True)) if m != model]
        plan += [(PROVIDER_ROUTEWAY, m) for m in mc.ranked_tool_models(PROVIDER_ROUTEWAY, rw.rank_free_models(False), rw.rank_free_models(True)) if m != model]
        plan += [(PROVIDER_KIOSAPI, m) for m in mc.ranked_tool_models(PROVIDER_KIOSAPI, ka.rank_free_models(False), ka.rank_free_models(True)) if m != model]
        plan += [(PROVIDER_XKIRO, m) for m in mc.ranked_tool_models(PROVIDER_XKIRO, _get_stable_xkiro_models(False), _get_stable_xkiro_models(True)) if m != model]
        plan += [(PROVIDER_MODELSCOPE, m) for m in mc.ranked_tool_models(PROVIDER_MODELSCOPE, _get_stable_modelscope_models(False), _get_stable_modelscope_models(True)) if m != model]
        plan += [(PROVIDER_LLM7, m) for m in mc.ranked_tool_models(PROVIDER_LLM7, _get_stable_llm7_models(False), _get_stable_llm7_models(True)) if m != model]
        plan += [(PROVIDER_AGNES, m) for m in mc.ranked_tool_models(PROVIDER_AGNES, agnes.rank_free_models(False), agnes.rank_free_models(True)) if m != model]
        plan += [(PROVIDER_ZYLOAI, m) for m in mc.ranked_tool_models(PROVIDER_ZYLOAI, zy.rank_free_models(False), zy.rank_free_models(True)) if m != model]
        plan += [(PROVIDER_DREAMPROMPTING, m) for m in mc.ranked_tool_models(PROVIDER_DREAMPROMPTING, dp.rank_free_models(False), dp.rank_free_models(True)) if m != model]
        plan += [(PROVIDER_TOPTOOLS, m) for m in mc.ranked_tool_models(PROVIDER_TOPTOOLS, tt.rank_free_models(False), tt.rank_free_models(True)) if m != model]
        plan += [(PROVIDER_CLEANAPIS, m) for m in mc.ranked_tool_models(PROVIDER_CLEANAPIS, cp.rank_free_models(False), cp.rank_free_models(True)) if m != model]
        plan += [(PROVIDER_UNOROUTER, m) for m in mc.ranked_tool_models(PROVIDER_UNOROUTER, uo.rank_free_models(False), uo.rank_free_models(True)) if m != model]
        # Probe-confirmed models lead the plan: live probe evidence outranks a
        # catalog claim, and the MAX_ATTEMPTS head-slice would otherwise fill
        # every slot with unprobed candidates. Stable sort keeps provider
        # priority within each tier.
        plan = (
            [(p, m) for (p, m) in plan if mc.model_supports_tools(p, m) is True]
            + [(p, m) for (p, m) in plan if mc.model_supports_tools(p, m) is not True]
        )
        return _finalize(plan, add_modelrouter=False)

    if model == kilo.AUTO_MODEL:
        plan = [(PROVIDER_KILO, m) for m in mc.filter_by_probed_capability(kilo.rank_free_models(need_tools), PROVIDER_KILO, need_tools, kilo.get_free_models)]
        plan += [(PROVIDER_MODELROUTER, m) for m in mc.filter_by_probed_capability(mr.rank_free_models(need_tools), PROVIDER_MODELROUTER, need_tools, mr.get_free_models)]
        plan += [(PROVIDER_TIARINA, m) for m in mc.filter_by_probed_capability(tr.rank_models(need_tools), PROVIDER_TIARINA, need_tools, tr.get_free_models)]
        plan += [(PROVIDER_COMPOSITE, m) for m in mc.filter_by_probed_capability(cx.rank_free_models(need_tools), PROVIDER_COMPOSITE, need_tools, cx.get_free_models)]
        plan += [(PROVIDER_SEPTORLABS, m) for m in mc.filter_by_probed_capability(sx.rank_free_models(need_tools), PROVIDER_SEPTORLABS, need_tools, sx.get_free_models)]
        plan += [(PROVIDER_ROUTEWAY, m) for m in mc.filter_by_probed_capability(rw.rank_free_models(need_tools), PROVIDER_ROUTEWAY, need_tools, rw.get_free_models)]
        plan += [(PROVIDER_CLOUDFLARE, m) for m in mc.filter_by_probed_capability(cf.rank_free_models(need_tools), PROVIDER_CLOUDFLARE, need_tools, cf.get_free_models)]
        plan += [(PROVIDER_UNOROUTER, m) for m in mc.filter_by_probed_capability(_get_stable_unorouter_models(need_tools), PROVIDER_UNOROUTER, need_tools, lambda: [])]
        plan += [(PROVIDER_KIOSAPI, m) for m in mc.filter_by_probed_capability(ka.rank_free_models(need_tools), PROVIDER_KIOSAPI, need_tools, ka.get_free_models)]
        plan += [(PROVIDER_XKIRO, m) for m in mc.filter_by_probed_capability(_get_stable_xkiro_models(need_tools), PROVIDER_XKIRO, need_tools, lambda: [])]
        plan += [(PROVIDER_MODELSCOPE, m) for m in mc.filter_by_probed_capability(_get_stable_modelscope_models(need_tools), PROVIDER_MODELSCOPE, need_tools, lambda: [])]
        plan += [(PROVIDER_LLM7, m) for m in mc.filter_by_probed_capability(_get_stable_llm7_models(need_tools), PROVIDER_LLM7, need_tools, lambda: [])]
        plan += [(PROVIDER_AGNES, m) for m in mc.filter_by_probed_capability(agnes.rank_free_models(need_tools), PROVIDER_AGNES, need_tools, agnes.get_free_models)]
        plan += [(PROVIDER_BAZAARLINK, m) for m in mc.filter_by_probed_capability(bl.rank_free_models(need_tools), PROVIDER_BAZAARLINK, need_tools, bl.get_free_models)]
        plan += [(PROVIDER_FREEMODELS, m) for m in freemodels_pool()]
        return _finalize(plan)

    if model == cf.CLOUDFLARE_AUTO_MODEL:
        plan = [(PROVIDER_CLOUDFLARE, m) for m in cf.rank_free_models(need_tools)]
        plan += [(PROVIDER_UNOROUTER, m) for m in _get_stable_unorouter_models(need_tools)]
        plan += [(PROVIDER_KIOSAPI, m) for m in ka.rank_free_models(need_tools)]
        plan += [(PROVIDER_XKIRO, m) for m in _get_stable_xkiro_models(need_tools)]
        plan += [(PROVIDER_MODELSCOPE, m) for m in _get_stable_modelscope_models(need_tools)]
        plan += [(PROVIDER_LLM7, m) for m in _get_stable_llm7_models(need_tools)]
        plan += [(PROVIDER_AGNES, m) for m in agnes.rank_free_models(need_tools)]
        plan += [(PROVIDER_ZYLOAI, m) for m in zy.rank_free_models(need_tools)]
        plan += [(PROVIDER_BAZAARLINK, m) for m in bl.rank_free_models(need_tools)]
        plan += [(PROVIDER_KILO, m) for m in kilo.rank_free_models(need_tools)]
        plan += [(PROVIDER_COMPOSITE, m) for m in cx.rank_free_models(need_tools)]
        plan += [(PROVIDER_SEPTORLABS, m) for m in sx.rank_free_models(need_tools)]
        plan += [(PROVIDER_ROUTEWAY, m) for m in rw.rank_free_models(need_tools)]
        plan += [(PROVIDER_FREEMODELS, m) for m in freemodels_pool()]
        return _finalize(plan)

    if model == bl.BAZAARLINK_AUTO_MODEL:
        plan = [(PROVIDER_BAZAARLINK, m) for m in bl.rank_free_models(need_tools)]
        plan += [(PROVIDER_XKIRO, m) for m in _get_stable_xkiro_models(need_tools)]
        plan += [(PROVIDER_MODELSCOPE, m) for m in _get_stable_modelscope_models(need_tools)]
        plan += [(PROVIDER_LLM7, m) for m in _get_stable_llm7_models(need_tools)]
        plan += [(PROVIDER_AGNES, m) for m in agnes.rank_free_models(need_tools)]
        plan += [(PROVIDER_ZYLOAI, m) for m in zy.rank_free_models(need_tools)]
        plan += [(PROVIDER_KIOSAPI, m) for m in ka.rank_free_models(need_tools)]
        plan += [(PROVIDER_UNOROUTER, m) for m in _get_stable_unorouter_models(need_tools)]
        plan += [(PROVIDER_CLOUDFLARE, m) for m in cf.rank_free_models(need_tools)]
        plan += [(PROVIDER_KILO, m) for m in kilo.rank_free_models(need_tools)]
        plan += [(PROVIDER_COMPOSITE, m) for m in cx.rank_free_models(need_tools)]
        plan += [(PROVIDER_SEPTORLABS, m) for m in sx.rank_free_models(need_tools)]
        plan += [(PROVIDER_ROUTEWAY, m) for m in rw.rank_free_models(need_tools)]
        plan += [(PROVIDER_FREEMODELS, m) for m in freemodels_pool()]
        return _finalize(plan)

    if mr.is_modelrouter_model(model):
        plan = [(PROVIDER_MODELROUTER, model)]
        plan += [(PROVIDER_MODELROUTER, m) for m in mr.rank_free_models(need_tools) if m != model]
        plan += [(PROVIDER_KILO, m) for m in kilo.rank_free_models(need_tools)]
        plan += [(PROVIDER_CLOUDFLARE, m) for m in cf.rank_free_models(need_tools)]
        plan += [(PROVIDER_KIOSAPI, m) for m in ka.rank_free_models(need_tools)]
        plan += [(PROVIDER_XKIRO, m) for m in _get_stable_xkiro_models(need_tools)]
        plan += [(PROVIDER_AGNES, m) for m in agnes.rank_free_models(need_tools)]
        plan += [(PROVIDER_ZYLOAI, m) for m in zy.rank_free_models(need_tools)]
        plan += [(PROVIDER_FREEMODELS, m) for m in freemodels_pool()]
        return _finalize(plan)

    if tr.is_tiarina_model(model):
        plan = [(PROVIDER_TIARINA, model)]
        plan += [(PROVIDER_TIARINA, m) for m in tr.rank_models(need_tools) if m != model]
        plan += [(PROVIDER_ZYLOAI, m) for m in zy.rank_free_models(need_tools)]
        plan += [(PROVIDER_KILO, m) for m in kilo.rank_free_models(need_tools)]
        plan += [(PROVIDER_MODELROUTER, m) for m in mr.rank_free_models(need_tools)]
        plan += [(PROVIDER_CLOUDFLARE, m) for m in cf.rank_free_models(need_tools)]
        plan += [(PROVIDER_FREEMODELS, m) for m in freemodels_pool()]
        return _finalize(plan)

    if zy.is_zyloai_model(model):
        plan = [(PROVIDER_ZYLOAI, model)]
        plan += [(PROVIDER_ZYLOAI, m) for m in zy.rank_free_models(need_tools) if m != model]
        # Tiarina publishes colliding ids (glm-5.3, kimi-k3, ...) and serves as
        # the graceful fallback while ZYLOAI_API_KEY is unset or exhausted.
        plan += [(PROVIDER_TIARINA, m) for m in tr.rank_models(need_tools) if m != model]
        plan += [(PROVIDER_KILO, m) for m in kilo.rank_free_models(need_tools)]
        plan += [(PROVIDER_MODELROUTER, m) for m in mr.rank_free_models(need_tools)]
        plan += [(PROVIDER_CLOUDFLARE, m) for m in cf.rank_free_models(need_tools)]
        plan += [(PROVIDER_FREEMODELS, m) for m in freemodels_pool()]
        return _finalize(plan)

    if cx.is_composite_model(model):
        plan = [(PROVIDER_COMPOSITE, model)]
        plan += [(PROVIDER_COMPOSITE, m) for m in cx.rank_free_models(need_tools) if m != model]
        plan += [(PROVIDER_SEPTORLABS, m) for m in sx.rank_free_models(need_tools) if m != model]
        plan += [(PROVIDER_ROUTEWAY, m) for m in rw.rank_free_models(need_tools) if m != model]
        plan += [(PROVIDER_KILO, m) for m in kilo.rank_free_models(need_tools)]
        plan += [(PROVIDER_CLOUDFLARE, m) for m in cf.rank_free_models(need_tools)]
        plan += [(PROVIDER_UNOROUTER, m) for m in _get_stable_unorouter_models(need_tools)]
        plan += [(PROVIDER_KIOSAPI, m) for m in ka.rank_free_models(need_tools)]
        plan += [(PROVIDER_XKIRO, m) for m in _get_stable_xkiro_models(need_tools)]
        plan += [(PROVIDER_MODELSCOPE, m) for m in _get_stable_modelscope_models(need_tools)]
        plan += [(PROVIDER_LLM7, m) for m in _get_stable_llm7_models(need_tools)]
        plan += [(PROVIDER_AGNES, m) for m in agnes.rank_free_models(need_tools)]
        plan += [(PROVIDER_ZYLOAI, m) for m in zy.rank_free_models(need_tools)]
        plan += [(PROVIDER_BAZAARLINK, m) for m in bl.rank_free_models(need_tools)]
        plan += [(PROVIDER_FREEMODELS, m) for m in freemodels_pool()]
        return _finalize(plan)


    if sx.is_septorlabs_model(model):
        plan = [(PROVIDER_SEPTORLABS, model)]
        plan += [(PROVIDER_SEPTORLABS, m) for m in sx.rank_free_models(need_tools) if m != model]
        plan += [(PROVIDER_ROUTEWAY, m) for m in rw.rank_free_models(need_tools) if m != model]
        plan += [(PROVIDER_COMPOSITE, m) for m in cx.rank_free_models(need_tools) if m != model]
        plan += [(PROVIDER_KILO, m) for m in kilo.rank_free_models(need_tools)]
        plan += [(PROVIDER_CLOUDFLARE, m) for m in cf.rank_free_models(need_tools)]
        plan += [(PROVIDER_UNOROUTER, m) for m in _get_stable_unorouter_models(need_tools)]
        plan += [(PROVIDER_KIOSAPI, m) for m in ka.rank_free_models(need_tools)]
        plan += [(PROVIDER_XKIRO, m) for m in _get_stable_xkiro_models(need_tools)]
        plan += [(PROVIDER_MODELSCOPE, m) for m in _get_stable_modelscope_models(need_tools)]
        plan += [(PROVIDER_LLM7, m) for m in _get_stable_llm7_models(need_tools)]
        plan += [(PROVIDER_AGNES, m) for m in agnes.rank_free_models(need_tools)]
        plan += [(PROVIDER_ZYLOAI, m) for m in zy.rank_free_models(need_tools)]
        plan += [(PROVIDER_BAZAARLINK, m) for m in bl.rank_free_models(need_tools)]
        plan += [(PROVIDER_FREEMODELS, m) for m in freemodels_pool()]
        return _finalize(plan)


    if rw.is_routeway_model(model):
        plan = [(PROVIDER_ROUTEWAY, model)]
        plan += [(PROVIDER_ROUTEWAY, m) for m in rw.rank_free_models(need_tools) if m != model]
        plan += [(PROVIDER_SEPTORLABS, m) for m in sx.rank_free_models(need_tools) if m != model]
        plan += [(PROVIDER_COMPOSITE, m) for m in cx.rank_free_models(need_tools) if m != model]
        plan += [(PROVIDER_KILO, m) for m in kilo.rank_free_models(need_tools)]
        plan += [(PROVIDER_CLOUDFLARE, m) for m in cf.rank_free_models(need_tools)]
        plan += [(PROVIDER_UNOROUTER, m) for m in _get_stable_unorouter_models(need_tools)]
        plan += [(PROVIDER_KIOSAPI, m) for m in ka.rank_free_models(need_tools)]
        plan += [(PROVIDER_XKIRO, m) for m in _get_stable_xkiro_models(need_tools)]
        plan += [(PROVIDER_MODELSCOPE, m) for m in _get_stable_modelscope_models(need_tools)]
        plan += [(PROVIDER_LLM7, m) for m in _get_stable_llm7_models(need_tools)]
        plan += [(PROVIDER_AGNES, m) for m in agnes.rank_free_models(need_tools)]
        plan += [(PROVIDER_ZYLOAI, m) for m in zy.rank_free_models(need_tools)]
        plan += [(PROVIDER_BAZAARLINK, m) for m in bl.rank_free_models(need_tools)]
        plan += [(PROVIDER_FREEMODELS, m) for m in freemodels_pool()]
        return _finalize(plan)

    if dp.is_dreamprompting_model(model):
        plan = [(PROVIDER_DREAMPROMPTING, model)]
        plan += [(PROVIDER_DREAMPROMPTING, m) for m in dp.rank_free_models(need_tools) if m != model]
        plan += [(PROVIDER_KILO, m) for m in kilo.rank_free_models(need_tools)]
        plan += [(PROVIDER_CLOUDFLARE, m) for m in cf.rank_free_models(need_tools)]
        plan += [(PROVIDER_UNOROUTER, m) for m in _get_stable_unorouter_models(need_tools)]
        plan += [(PROVIDER_KIOSAPI, m) for m in ka.rank_free_models(need_tools)]
        plan += [(PROVIDER_XKIRO, m) for m in _get_stable_xkiro_models(need_tools)]
        plan += [(PROVIDER_MODELSCOPE, m) for m in _get_stable_modelscope_models(need_tools)]
        plan += [(PROVIDER_LLM7, m) for m in _get_stable_llm7_models(need_tools)]
        plan += [(PROVIDER_AGNES, m) for m in agnes.rank_free_models(need_tools)]
        plan += [(PROVIDER_ZYLOAI, m) for m in zy.rank_free_models(need_tools)]
        plan += [(PROVIDER_BAZAARLINK, m) for m in bl.rank_free_models(need_tools)]
        plan += [(PROVIDER_FREEMODELS, m) for m in freemodels_pool()]
        return _finalize(plan)

    if cp.is_cleanapis_model(model):
        plan = [(PROVIDER_CLEANAPIS, model)]
        plan += [(PROVIDER_CLEANAPIS, m) for m in cp.rank_free_models(need_tools) if m != model]
        plan += [(PROVIDER_KILO, m) for m in kilo.rank_free_models(need_tools)]
        plan += [(PROVIDER_CLOUDFLARE, m) for m in cf.rank_free_models(need_tools)]
        plan += [(PROVIDER_UNOROUTER, m) for m in _get_stable_unorouter_models(need_tools)]
        plan += [(PROVIDER_KIOSAPI, m) for m in ka.rank_free_models(need_tools)]
        plan += [(PROVIDER_XKIRO, m) for m in _get_stable_xkiro_models(need_tools)]
        plan += [(PROVIDER_MODELSCOPE, m) for m in _get_stable_modelscope_models(need_tools)]
        plan += [(PROVIDER_LLM7, m) for m in _get_stable_llm7_models(need_tools)]
        plan += [(PROVIDER_AGNES, m) for m in agnes.rank_free_models(need_tools)]
        plan += [(PROVIDER_ZYLOAI, m) for m in zy.rank_free_models(need_tools)]
        plan += [(PROVIDER_BAZAARLINK, m) for m in bl.rank_free_models(need_tools)]
        plan += [(PROVIDER_FREEMODELS, m) for m in freemodels_pool()]
        return _finalize(plan)

    if tt.is_toptools_model(model):
        plan = [(PROVIDER_TOPTOOLS, model)]
        plan += [(PROVIDER_TOPTOOLS, m) for m in tt.rank_free_models(need_tools) if m != model]
        plan += [(PROVIDER_COMPOSITE, m) for m in cx.rank_free_models(need_tools) if m != model]
        plan += [(PROVIDER_KILO, m) for m in kilo.rank_free_models(need_tools)]
        plan += [(PROVIDER_CLOUDFLARE, m) for m in cf.rank_free_models(need_tools)]
        plan += [(PROVIDER_UNOROUTER, m) for m in _get_stable_unorouter_models(need_tools)]
        plan += [(PROVIDER_KIOSAPI, m) for m in ka.rank_free_models(need_tools)]
        plan += [(PROVIDER_XKIRO, m) for m in _get_stable_xkiro_models(need_tools)]
        plan += [(PROVIDER_MODELSCOPE, m) for m in _get_stable_modelscope_models(need_tools)]
        plan += [(PROVIDER_LLM7, m) for m in _get_stable_llm7_models(need_tools)]
        plan += [(PROVIDER_AGNES, m) for m in agnes.rank_free_models(need_tools)]
        plan += [(PROVIDER_ZYLOAI, m) for m in zy.rank_free_models(need_tools)]
        plan += [(PROVIDER_BAZAARLINK, m) for m in bl.rank_free_models(need_tools)]
        plan += [(PROVIDER_FREEMODELS, m) for m in freemodels_pool()]
        return _finalize(plan)

    if kilo.is_kilo_model(model):
        plan = [(PROVIDER_KILO, model)]
        plan += [(PROVIDER_KILO, m) for m in kilo.rank_free_models(need_tools) if m != model]
        plan += [(PROVIDER_CLOUDFLARE, m) for m in cf.rank_free_models(need_tools) if m != model]
        plan += [(PROVIDER_UNOROUTER, m) for m in _get_stable_unorouter_models(need_tools) if m != model]
        plan += [(PROVIDER_KIOSAPI, m) for m in ka.rank_free_models(need_tools) if m != model]
        plan += [(PROVIDER_XKIRO, m) for m in _get_stable_xkiro_models(need_tools) if m != model]
        plan += [(PROVIDER_MODELSCOPE, m) for m in _get_stable_modelscope_models(need_tools) if m != model]
        plan += [(PROVIDER_LLM7, m) for m in _get_stable_llm7_models(need_tools) if m != model]
        plan += [(PROVIDER_AGNES, m) for m in agnes.rank_free_models(need_tools) if m != model]
        plan += [(PROVIDER_BAZAARLINK, m) for m in bl.rank_free_models(need_tools) if m != model]
        plan += [(PROVIDER_COMPOSITE, m) for m in cx.rank_free_models(need_tools)]
        plan += [(PROVIDER_SEPTORLABS, m) for m in sx.rank_free_models(need_tools)]
        plan += [(PROVIDER_ROUTEWAY, m) for m in rw.rank_free_models(need_tools)]
        plan += [(PROVIDER_FREEMODELS, m) for m in freemodels_pool()]
        return _finalize(plan)

    if cf.is_cloudflare_model(model):
        plan = [(PROVIDER_CLOUDFLARE, model)]
        plan += [(PROVIDER_CLOUDFLARE, m) for m in cf.rank_free_models(need_tools) if m != model]
        plan += [(PROVIDER_UNOROUTER, m) for m in _get_stable_unorouter_models(need_tools) if m != model]
        plan += [(PROVIDER_KIOSAPI, m) for m in ka.rank_free_models(need_tools) if m != model]
        plan += [(PROVIDER_XKIRO, m) for m in _get_stable_xkiro_models(need_tools) if m != model]
        plan += [(PROVIDER_MODELSCOPE, m) for m in _get_stable_modelscope_models(need_tools) if m != model]
        plan += [(PROVIDER_LLM7, m) for m in _get_stable_llm7_models(need_tools) if m != model]
        plan += [(PROVIDER_AGNES, m) for m in agnes.rank_free_models(need_tools) if m != model]
        plan += [(PROVIDER_BAZAARLINK, m) for m in bl.rank_free_models(need_tools) if m != model]
        plan += [(PROVIDER_KILO, m) for m in kilo.rank_free_models(need_tools) if m != model]
        plan += [(PROVIDER_COMPOSITE, m) for m in cx.rank_free_models(need_tools)]
        plan += [(PROVIDER_SEPTORLABS, m) for m in sx.rank_free_models(need_tools)]
        plan += [(PROVIDER_ROUTEWAY, m) for m in rw.rank_free_models(need_tools)]
        plan += [(PROVIDER_FREEMODELS, m) for m in freemodels_pool()]
        return _finalize(plan)

    if uo.is_unorouter_model(model):
        plan = [(PROVIDER_UNOROUTER, model)]
        plan += [(PROVIDER_UNOROUTER, m) for m in _get_stable_unorouter_models(need_tools) if m != model]
        plan += [(PROVIDER_CLOUDFLARE, m) for m in cf.rank_free_models(need_tools) if m != model]
        plan += [(PROVIDER_KIOSAPI, m) for m in ka.rank_free_models(need_tools) if m != model]
        plan += [(PROVIDER_XKIRO, m) for m in _get_stable_xkiro_models(need_tools) if m != model]
        plan += [(PROVIDER_MODELSCOPE, m) for m in _get_stable_modelscope_models(need_tools) if m != model]
        plan += [(PROVIDER_LLM7, m) for m in _get_stable_llm7_models(need_tools) if m != model]
        plan += [(PROVIDER_AGNES, m) for m in agnes.rank_free_models(need_tools) if m != model]
        plan += [(PROVIDER_BAZAARLINK, m) for m in bl.rank_free_models(need_tools) if m != model]
        plan += [(PROVIDER_KILO, m) for m in kilo.rank_free_models(need_tools) if m != model]
        plan += [(PROVIDER_COMPOSITE, m) for m in cx.rank_free_models(need_tools)]
        plan += [(PROVIDER_SEPTORLABS, m) for m in sx.rank_free_models(need_tools)]
        plan += [(PROVIDER_ROUTEWAY, m) for m in rw.rank_free_models(need_tools)]
        plan += [(PROVIDER_FREEMODELS, m) for m in freemodels_pool()]
        return _finalize(plan)

    if ka.is_kiosapi_model(model):
        plan = [(PROVIDER_KIOSAPI, model)]
        plan += [(PROVIDER_KIOSAPI, m) for m in ka.rank_free_models(need_tools) if m != model]
        plan += [(PROVIDER_XKIRO, m) for m in _get_stable_xkiro_models(need_tools) if m != model]
        plan += [(PROVIDER_MODELSCOPE, m) for m in _get_stable_modelscope_models(need_tools) if m != model]
        plan += [(PROVIDER_LLM7, m) for m in _get_stable_llm7_models(need_tools) if m != model]
        plan += [(PROVIDER_UNOROUTER, m) for m in _get_stable_unorouter_models(need_tools) if m != model]
        plan += [(PROVIDER_CLOUDFLARE, m) for m in cf.rank_free_models(need_tools) if m != model]
        plan += [(PROVIDER_KILO, m) for m in kilo.rank_free_models(need_tools) if m != model]
        plan += [(PROVIDER_BAZAARLINK, m) for m in bl.rank_free_models(need_tools) if m != model]
        plan += [(PROVIDER_COMPOSITE, m) for m in cx.rank_free_models(need_tools)]
        plan += [(PROVIDER_SEPTORLABS, m) for m in sx.rank_free_models(need_tools)]
        plan += [(PROVIDER_ROUTEWAY, m) for m in rw.rank_free_models(need_tools)]
        plan += [(PROVIDER_FREEMODELS, m) for m in freemodels_pool()]
        return _finalize(plan)

    if xk.is_xkiro_model(model):
        plan = [(PROVIDER_XKIRO, model)]
        plan += [(PROVIDER_XKIRO, m) for m in _get_stable_xkiro_models(need_tools) if m != model]
        plan += [(PROVIDER_MODELSCOPE, m) for m in _get_stable_modelscope_models(need_tools) if m != model]
        plan += [(PROVIDER_LLM7, m) for m in _get_stable_llm7_models(need_tools) if m != model]
        plan += [(PROVIDER_KIOSAPI, m) for m in ka.rank_free_models(need_tools) if m != model]
        plan += [(PROVIDER_UNOROUTER, m) for m in _get_stable_unorouter_models(need_tools) if m != model]
        plan += [(PROVIDER_CLOUDFLARE, m) for m in cf.rank_free_models(need_tools) if m != model]
        plan += [(PROVIDER_KILO, m) for m in kilo.rank_free_models(need_tools) if m != model]
        plan += [(PROVIDER_BAZAARLINK, m) for m in bl.rank_free_models(need_tools) if m != model]
        plan += [(PROVIDER_COMPOSITE, m) for m in cx.rank_free_models(need_tools)]
        plan += [(PROVIDER_SEPTORLABS, m) for m in sx.rank_free_models(need_tools)]
        plan += [(PROVIDER_ROUTEWAY, m) for m in rw.rank_free_models(need_tools)]
        plan += [(PROVIDER_FREEMODELS, m) for m in freemodels_pool()]
        return _finalize(plan)

    if agnes.is_agnes_model(model):
        plan = [(PROVIDER_AGNES, model)]
        plan += [(PROVIDER_AGNES, m) for m in agnes.rank_free_models(need_tools) if m != model]
        plan += [(PROVIDER_KIOSAPI, m) for m in ka.rank_free_models(need_tools) if m != model]
        plan += [(PROVIDER_XKIRO, m) for m in _get_stable_xkiro_models(need_tools) if m != model]
        plan += [(PROVIDER_MODELSCOPE, m) for m in _get_stable_modelscope_models(need_tools) if m != model]
        plan += [(PROVIDER_LLM7, m) for m in _get_stable_llm7_models(need_tools) if m != model]
        plan += [(PROVIDER_UNOROUTER, m) for m in _get_stable_unorouter_models(need_tools) if m != model]
        plan += [(PROVIDER_CLOUDFLARE, m) for m in cf.rank_free_models(need_tools) if m != model]
        plan += [(PROVIDER_KILO, m) for m in kilo.rank_free_models(need_tools) if m != model]
        plan += [(PROVIDER_BAZAARLINK, m) for m in bl.rank_free_models(need_tools) if m != model]
        plan += [(PROVIDER_COMPOSITE, m) for m in cx.rank_free_models(need_tools)]
        plan += [(PROVIDER_SEPTORLABS, m) for m in sx.rank_free_models(need_tools)]
        plan += [(PROVIDER_ROUTEWAY, m) for m in rw.rank_free_models(need_tools)]
        plan += [(PROVIDER_FREEMODELS, m) for m in freemodels_pool()]
        return _finalize(plan)

    if model == CHATBOT_MODEL:
        combo_models = get_active_combo_models(CHATBOT_MODEL)
        if combo_models:
            return _dedupe(combo_models)[:MAX_ATTEMPTS]
        plan = list(CHATBOT_MODELS)
        return _finalize(plan)

    if model == CODING_MODEL:
        combo_models = get_active_combo_models(CODING_MODEL)
        if combo_models:
            return _dedupe(combo_models)[:MAX_ATTEMPTS]
        plan = list(CODING_MODELS)
        return _finalize(plan)

    # FreeModels-style model: try it first, then the rest of its catalog.
    plan = [(PROVIDER_FREEMODELS, model)]
    plan += [(PROVIDER_FREEMODELS, m) for m in freemodels_pool() if m != model]
    plan += [(PROVIDER_SEPTORLABS, m) for m in sx.rank_free_models(need_tools) if m != model]
    plan += [(PROVIDER_ROUTEWAY, m) for m in rw.rank_free_models(need_tools) if m != model]
    plan += [(PROVIDER_COMPOSITE, m) for m in cx.rank_free_models(need_tools) if m != model]
    plan += [(PROVIDER_UNOROUTER, m) for m in _get_stable_unorouter_models(need_tools) if m != model]
    plan += [(PROVIDER_KIOSAPI, m) for m in ka.rank_free_models(need_tools) if m != model]
    plan += [(PROVIDER_XKIRO, m) for m in _get_stable_xkiro_models(need_tools) if m != model]
    plan += [(PROVIDER_MODELSCOPE, m) for m in _get_stable_modelscope_models(need_tools) if m != model]
    plan += [(PROVIDER_LLM7, m) for m in _get_stable_llm7_models(need_tools) if m != model]
    plan += [(PROVIDER_AGNES, m) for m in agnes.rank_free_models(need_tools) if m != model]
    plan += [(PROVIDER_BAZAARLINK, m) for m in bl.rank_free_models(need_tools) if m != model]
    plan += [(PROVIDER_CLOUDFLARE, m) for m in cf.rank_free_models(need_tools) if m != model]
    return _finalize(plan)


def is_transient_error(error_body: str) -> bool:
    """Detect retryable upstream failures (blips, not client mistakes)."""
    indicators = [
        "502", "503", "504", "bad gateway", "service unavailable",
        "service_unavailable", "overloaded", "internal server error",
        "internal_error", "server_error", "temporarily unavailable",
        "unable to process your request",
    ]
    lower = error_body.lower()
    return any(ind in lower for ind in indicators)


def is_quota_error(error_body: str) -> bool:
    """Detect if an error is a quota/rate-limit exhaustion."""
    indicators = [
        "429", "too many requests", "rate limit", "quota exhausted",
        "free quota has been exhausted", "All providers exhausted",
        "retry in", "providers exhausted",
    ]
    lower = error_body.lower()
    return any(ind in lower for ind in indicators)


def _provider_wide_quota(error_body: str, status_code: int | None) -> bool:
    """True when a failure should open the provider breaker, not just the model's.

    Rate limits and account-level errors (401/402/429) affect every model
    behind the provider; a 403/404/5xx scoped to one model (dead model,
    gateway blip) must not circuit-break its healthy siblings.
    """
    if status_code in (401, 402, 429):
        return True
    if status_code is not None:
        return False
    return is_quota_error(error_body)


def get_all_models() -> list[dict]:
    """Return all advertised models across providers."""
    models: list[dict] = []
    seen: set[str] = set()

    # Virtual auto-router
    models.append({
        "id": kilo.AUTO_MODEL,
        "object": "model",
        "created": 1700000000,
        "owned_by": PROVIDER_KILO,
        "healthy": _health[PROVIDER_KILO].is_healthy,
        "tools_supported": True,
    })
    seen.add(kilo.AUTO_MODEL)

    # Virtual Cloudflare auto-router
    models.append({
        "id": cf.CLOUDFLARE_AUTO_MODEL,
        "object": "model",
        "created": 1700000000,
        "owned_by": PROVIDER_CLOUDFLARE,
        "healthy": _health[PROVIDER_CLOUDFLARE].is_healthy,
        "tools_supported": False,
    })
    seen.add(cf.CLOUDFLARE_AUTO_MODEL)

    # Virtual Chatbot model
    models.append({
        "id": CHATBOT_MODEL,
        "object": "model",
        "created": 1700000000,
        "owned_by": PROVIDER_CHATBOT,
        "healthy": True,
        "tools_supported": False,
    })
    seen.add(CHATBOT_MODEL)

    # Active combos are advertised as models under their own name, so an admin
    # can address a group (or an agentic loop) without a special virtual id.
    # Registered before the provider catalogs: a combo name that collides with
    # a catalog id resolves to the combo, matching get_attempt_plan().
    for entry in _combo_index:
        if entry["name"] in seen:
            continue
        seen.add(entry["name"])
        models.append({
            "id": entry["name"],
            "object": "model",
            "created": int(entry["created_at"].timestamp()) if entry["created_at"] else 1700000000,
            "owned_by": PROVIDER_COMBO,
            "healthy": True,
            "tools_supported": entry["is_agentic"],
            "combo_id": entry["id"],
            "routing_mode": entry["routing_mode"],
            "routing_strategy": entry["routing_strategy"],
        })

    # Virtual Coding model
    models.append({
        "id": CODING_MODEL,
        "object": "model",
        "created": 1700000000,
        "owned_by": PROVIDER_CODING,
        "healthy": True,
        "tools_supported": True,
    })
    seen.add(CODING_MODEL)

    # Routeway free models (key-gated; `:free` ids only)
    if rw.is_configured():
        for m in rw.get_free_models():
            if m["id"] in seen:
                continue
            seen.add(m["id"])
            models.append({
                "id": m["id"],
                "object": "model",
                "created": 1700000000,
                "owned_by": PROVIDER_ROUTEWAY,
                "healthy": _health[PROVIDER_ROUTEWAY].is_healthy,
                "tools_supported": "tools" in (m.get("supported_parameters") or []),
                "context_length": m.get("context_length"),
            })

    # Top Tools AI free models (key-gated)
    if tt.is_configured():
        for m in tt.get_free_models():
            if m["id"] in seen:
                continue
            seen.add(m["id"])
            models.append({
                "id": m["id"],
                "object": "model",
                "created": 1700000000,
                "owned_by": PROVIDER_TOPTOOLS,
                "healthy": _health[PROVIDER_TOPTOOLS].is_healthy,
                "tools_supported": "tools" in (m.get("supported_parameters") or []),
                "context_length": m.get("context_length"),
            })

    # DreamPrompting models (key-gated)
    if dp.is_configured():
        for m in dp.get_free_models():
            if m["id"] in seen:
                continue
            seen.add(m["id"])
            models.append({
                "id": m["id"],
                "object": "model",
                "created": 1700000000,
                "owned_by": PROVIDER_DREAMPROMPTING,
                "healthy": _health[PROVIDER_DREAMPROMPTING].is_healthy,
                "tools_supported": "tools" in (m.get("supported_parameters") or []),
                "context_length": m.get("context_length"),
            })

    # ZyloAI models (registered before the broad gateway catalogs: its 5
    # confirmed ids collide with Tiarina/ModelRouter entries, and the confirmed
    # ZyloAI upstream is the intended owner of these ids).
    for m in zy.get_free_models():
        if m["id"] in seen:
            continue
        seen.add(m["id"])
        models.append({
            "id": m["id"],
            "object": "model",
            "created": 1700000000,
            "owned_by": PROVIDER_ZYLOAI,
            "healthy": _health[PROVIDER_ZYLOAI].is_healthy,
            "tools_supported": "tools" in (m.get("supported_parameters") or []),
            "context_length": m.get("context_length"),
        })

    # Clean APIs models (key-gated)
    if cp.is_configured():
        for m in cp.get_free_models():
            if m["id"] in seen:
                continue
            seen.add(m["id"])
            models.append({
                "id": m["id"],
                "object": "model",
                "created": 1700000000,
                "owned_by": PROVIDER_CLEANAPIS,
                "healthy": _health[PROVIDER_CLEANAPIS].is_healthy,
                "tools_supported": "tools" in (m.get("supported_parameters") or []),
                "context_length": m.get("context_length"),
            })

    # ModelRouter free text-chat models (key-gated catalog and chat)
    if mr.is_configured():
        for m in mr.get_free_models():
            if m["id"] in seen:
                continue
            seen.add(m["id"])
            models.append({
                "id": m["id"],
                "object": "model",
                "created": 1700000000,
                "owned_by": PROVIDER_MODELROUTER,
                "healthy": _health[PROVIDER_MODELROUTER].is_healthy,
                "tools_supported": "tools" in (m.get("supported_parameters") or []),
                "context_length": m.get("context_length"),
            })

    # Tiarina Cloud models (key-gated catalog and chat)
    if tr.is_configured():
        for m in tr.get_available_models():
            if m["id"] in seen:
                continue
            seen.add(m["id"])
            models.append({
                "id": m["id"],
                "object": "model",
                "created": 1700000000,
                "owned_by": PROVIDER_TIARINA,
                "healthy": _health[PROVIDER_TIARINA].is_healthy,
                "tools_supported": "tools" in (m.get("supported_parameters") or []),
                "context_length": m.get("context_length"),
            })

    # Septor Labs free models (key-gated; `-free` ids only)
    if sx.is_configured():
        for m in sx.get_free_models():
            if m["id"] in seen:
                continue
            seen.add(m["id"])
            models.append({
                "id": m["id"],
                "object": "model",
                "created": 1700000000,
                "owned_by": PROVIDER_SEPTORLABS,
                "healthy": _health[PROVIDER_SEPTORLABS].is_healthy,
                "tools_supported": "tools" in (m.get("supported_parameters") or []),
                "context_length": m.get("context_length"),
            })

    # Composite free models (key-gated; `:free` ids only)
    if cx.is_configured():
        for m in cx.get_models():
            if m["id"] in seen:
                continue
            seen.add(m["id"])
            models.append({
                "id": m["id"],
                "object": "model",
                "created": 1700000000,
                "owned_by": PROVIDER_COMPOSITE,
                "healthy": _health[PROVIDER_COMPOSITE].is_healthy,
                "tools_supported": "tools" in (m.get("supported_parameters") or []),
                "context_length": m.get("context_length"),
            })

    # Free Kilo models (live catalog)
    for m in kilo.get_free_models():
        mid = m.get("id")
        if not mid or mid in seen:
            continue
        seen.add(mid)
        models.append({
            "id": mid,
            "object": "model",
            "created": m.get("created") or 1700000000,
            "owned_by": PROVIDER_KILO,
            "healthy": _health[PROVIDER_KILO].is_healthy,
            "tools_supported": "tools" in (m.get("supported_parameters") or []),
        })

    # Free Cloudflare models
    for m in cf.get_free_models():
        mid = m.get("id")
        if not mid or mid in seen:
            continue
        seen.add(mid)
        tools_supported = m.get("supported_parameters") is not None and "tools" in m.get("supported_parameters", [])
        models.append({
            "id": mid,
            "object": "model",
            "created": 1700000000,
            "owned_by": PROVIDER_CLOUDFLARE,
            "healthy": _health[PROVIDER_CLOUDFLARE].is_healthy,
            "tools_supported": tools_supported,
        })

    # Free UnoRouter models
    for m in uo.get_free_models():
        mid = m.get("id")
        if not mid or mid in seen:
            continue
        seen.add(mid)
        tools_supported = m.get("supported_parameters") is not None and "tools" in m.get("supported_parameters", [])
        models.append({
            "id": mid,
            "object": "model",
            "created": 1700000000,
            "owned_by": PROVIDER_UNOROUTER,
            "healthy": _health[PROVIDER_UNOROUTER].is_healthy,
            "tools_supported": tools_supported,
        })

    # Free BazaarLink models
    for m in bl.get_free_models():
        mid = m.get("id")
        if not mid or mid in seen:
            continue
        seen.add(mid)
        tools_supported = m.get("supported_parameters") is not None and "tools" in m.get("supported_parameters", [])
        models.append({
            "id": mid,
            "object": "model",
            "created": 1700000000,
            "owned_by": PROVIDER_BAZAARLINK,
            "healthy": _health[PROVIDER_BAZAARLINK].is_healthy,
            "tools_supported": tools_supported,
        })

    # KiosAPI free models (OpenAI-compatible, tool calling support)
    for m in ka.get_free_models():
        mid = m.get("id")
        if not mid or mid in seen:
            continue
        if mid.startswith("agnes-"):
            continue
        seen.add(mid)
        models.append({
            "id": mid,
            "object": "model",
            "created": 1700000000,
            "owned_by": PROVIDER_KIOSAPI,
            "healthy": _health[PROVIDER_KIOSAPI].is_healthy,
            "tools_supported": "tools" in (m.get("supported_parameters") or []),
        })

    # Xkiro free models (OpenAI-compatible, 37+ free models with tool calling)
    for m in xk.get_free_models():
        mid = m.get("id")
        if not mid or mid in seen:
            continue
        seen.add(mid)
        models.append({
            "id": mid,
            "object": "model",
            "created": 1700000000,
            "owned_by": PROVIDER_XKIRO,
            "healthy": _health[PROVIDER_XKIRO].is_healthy,
            "tools_supported": "tools" in (m.get("supported_parameters") or []),
            "context_length": m.get("context_length"),
        })

    # ModelScope free models (API-Inference, 34+ free open-source LLMs)
    for m in ms.get_free_models():
        mid = m.get("id")
        if not mid or mid in seen:
            continue
        seen.add(mid)
        models.append({
            "id": mid,
            "object": "model",
            "created": 1700000000,
            "owned_by": PROVIDER_MODELSCOPE,
            "healthy": _health[PROVIDER_MODELSCOPE].is_healthy,
            "tools_supported": False,
        })

    # LLM7 free models (AI gateway, 6+ free turbo models)
    for m in llm7.get_free_models():
        mid = m.get("id")
        if not mid or mid in seen:
            continue
        seen.add(mid)
        models.append({
            "id": mid,
            "object": "model",
            "created": 1700000000,
            "owned_by": PROVIDER_LLM7,
            "healthy": _health[PROVIDER_LLM7].is_healthy,
            "tools_supported": "tools" in (m.get("supported_parameters") or []),
        })

    # Agnes AI free models (2 free models, $0/M tokens, 512K context)
    for m in agnes.get_free_models():
        mid = m.get("id")
        if not mid or mid in seen:
            continue
        seen.add(mid)
        models.append({
            "id": mid,
            "object": "model",
            "created": 1700000000,
            "owned_by": PROVIDER_AGNES,
            "healthy": _health[PROVIDER_AGNES].is_healthy,
            "tools_supported": "tools" in (m.get("supported_parameters") or []),
            "context_length": m.get("context_length"),
        })

    # FreeModels catalog (text only — no tool-calling support)
    for m in freemodels_pool():
        if m in seen:
            continue
        seen.add(m)
        models.append({
            "id": m,
            "object": "model",
            "created": 1700000000,
            "owned_by": PROVIDER_FREEMODELS,
            "healthy": _health[PROVIDER_FREEMODELS].is_healthy,
            "tools_supported": False,
        })

    return models


# ── FreeModels Provider ─────────────────────────────────────────────────────

async def _freemodels_stream_single(
    model: str,
    messages: list,
    include_usage: bool = False,
    tools: list = None,
    sampling: SamplingParams | None = None,
) -> AsyncIterator[str]:
    """Stream from FreeModels for a single model attempt."""
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    fm_messages = to_openai_messages(messages)

    payload = {
        "messages": fm_messages,
        "modelId": model,
        "thinking": False,
        "deepSearch": False,
        "stream": True,
    }
    if tools:
        payload["tools"] = tools
    apply_sampling(payload, sampling)

    prompt_tokens_est = max(1, len(json.dumps(fm_messages, ensure_ascii=False)) // 4)
    accumulated_text = ""
    yielded_role = False

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            settings.freemodels_url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                # Upstream worker rejects requests without a freemodels.pro referrer
                "Referer": "https://freemodels.pro/",
                "Origin": "https://freemodels.pro",
            },
        ) as response:
            if response.status_code != 200:
                body_bytes = await response.aread()
                error_msg = body_bytes.decode(errors="replace")[:500]
                raise QuotaError(
                    f"FreeModels {response.status_code}: {error_msg}",
                    provider=PROVIDER_FREEMODELS,
                    model=model,
                    status_code=response.status_code,
                )

            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                line_data = line[6:].strip()
                if line_data == "[DONE]":
                    if include_usage:
                        yield usage_chunk(chat_id, created, model, prompt_tokens_est, accumulated_text)
                    yield "data: [DONE]\n\n"
                    return

                try:
                    parsed = json.loads(line_data)
                except json.JSONDecodeError:
                    continue

                choices = parsed.get("choices") or []
                if not choices:
                    continue

                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                tool_calls = delta.get("tool_calls")
                finish_reason = choices[0].get("finish_reason")

                if content:
                    accumulated_text += content

                if content or tool_calls:
                    if not yielded_role:
                        yield sse_chunk(chat_id, created, model, role="assistant")
                        yielded_role = True
                    yield sse_chunk(chat_id, created, model, content=content, tool_calls=tool_calls)

                if finish_reason and finish_reason != "null":
                    if include_usage:
                        yield usage_chunk(chat_id, created, model, prompt_tokens_est, accumulated_text)
                    if yielded_role:
                        yield sse_chunk(chat_id, created, model, finish_reason=finish_reason)
                    yield "data: [DONE]\n\n"
                    return

            # Stream ended without an explicit finish signal
            if include_usage:
                yield usage_chunk(chat_id, created, model, prompt_tokens_est, accumulated_text)
            if yielded_role:
                yield sse_chunk(chat_id, created, model, finish_reason="stop")
            yield "data: [DONE]\n\n"


# ── Intelligent Failover Stream ─────────────────────────────────────────────

async def _measured_stream(
    provider: str,
    model: str,
    generator: AsyncIterator[str],
) -> AsyncIterator[str]:
    """Measure a full upstream stream and record latency/success telemetry.

   _wraps every provider dispatch so smart-stream, smart-collect and the
    agentic loop all feed the same rolling health window. GeneratorExit
    (client disconnect / early break) counts as success: the stream
    delivered. Anything raised (QuotaError, timeouts, HTTP failures from the
    provider module) counts as failure before re-raising.
    """
    start = time.monotonic()
    ok = True
    try:
        async for chunk in generator:
            yield chunk
    except Exception:
        ok = False
        raise
    finally:
        latency_ms = (time.monotonic() - start) * 1000.0
        try:
            record_stream_latency(provider, model, latency_ms, ok)
        except Exception:
            pass


def _stream_for(
    provider: str,
    model: str,
    messages: list,
    include_usage: bool,
    tools: list,
    sampling: SamplingParams | None = None,
):
    """Dispatch an upstream stream with measured telemetry (public entry)."""
    return _measured_stream(
        provider,
        model,
        _dispatch_stream(provider, model, messages, include_usage, tools, sampling),
    )


def _dispatch_stream(
    provider: str,
    model: str,
    messages: list,
    include_usage: bool,
    tools: list,
    sampling: SamplingParams | None = None,
):
    if provider == PROVIDER_KILO:
        return kilo.kilo_stream_single(model, messages, include_usage, tools, sampling)
    if provider == PROVIDER_CLOUDFLARE:
        return cf.cloudflare_stream_single(model, messages, include_usage, tools, sampling)
    if provider == PROVIDER_UNOROUTER:
        return uo.unorouter_stream_single(model, messages, include_usage, tools, sampling)
    if provider == PROVIDER_KIOSAPI:
        return ka.kiosapi_stream_single(model, messages, include_usage, tools, sampling)
    if provider == PROVIDER_XKIRO:
        return xk.xkiro_stream_single(model, messages, include_usage, tools, sampling)
    if provider == PROVIDER_MODELSCOPE:
        return ms.modelscope_stream_single(model, messages, include_usage, tools, sampling)
    if provider == PROVIDER_LLM7:
        return llm7.llm7_stream_single(model, messages, include_usage, tools, sampling)
    if provider == PROVIDER_AGNES:
        return agnes.agnes_stream_single(model, messages, include_usage, tools, sampling)
    if provider == PROVIDER_BAZAARLINK:
        return bl.bazaarlink_stream_single(model, messages, include_usage, tools, sampling)
    if provider == PROVIDER_COMPOSITE:
        return cx.composite_stream_single(model, messages, include_usage, tools, sampling)
    if provider == PROVIDER_SEPTORLABS:
        return sx.septorlabs_stream_single(model, messages, include_usage, tools, sampling)
    if provider == PROVIDER_ROUTEWAY:
        return rw.routeway_stream_single(model, messages, include_usage, tools, sampling)
    if provider == PROVIDER_TOPTOOLS:
        return tt.toptools_stream_single(model, messages, include_usage, tools, sampling)
    if provider == PROVIDER_DREAMPROMPTING:
        return dp.dreamprompting_stream_single(model, messages, include_usage, tools, sampling)
    if provider == PROVIDER_CLEANAPIS:
        return cp.cleanapis_stream_single(model, messages, include_usage, tools, sampling)
    if provider == PROVIDER_MODELROUTER:
        return mr.modelrouter_stream_single(model, messages, include_usage, tools, sampling)
    if provider == PROVIDER_TIARINA:
        return tr.tiarina_stream_single(model, messages, include_usage, tools, sampling)
    if provider == PROVIDER_ZYLOAI:
        return zy.zyloai_stream_single(model, messages, include_usage, tools, sampling)
    return _freemodels_stream_single(model, messages, include_usage, tools, sampling)


def _error_frame(message: str, type_: str = "server_error", code: str | None = None) -> str:
    """Build an OpenAI-style SSE error frame."""
    return "data: " + json.dumps({
        "error": {"message": message, "type": type_, "code": code},
    }, ensure_ascii=False) + "\n\n"


def _candidate_list(
    model: str,
    plan: list[tuple[str, str]],
) -> list[tuple[int, tuple[str, str]]]:
    """Attempts to try, honouring the model- and provider-level breakers.

    An explicitly requested model (i.e. not a virtual auto-router) is always
    tried first even when its breaker is open, so a single model's breaker can
    never hide a user's explicit choice. If the filtering empties the list the
    full plan is used rather than failing outright.
    """
    protect_first = model not in (kilo.AUTO_MODEL, cf.CLOUDFLARE_AUTO_MODEL)
    candidates = [
        (i, entry) for i, entry in enumerate(plan)
        if (
            get_model_health(entry[1]).is_healthy
            and _health[entry[0]].is_healthy
        ) or (i == 0 and protect_first)
    ]
    if not candidates:
        # Everything is circuit-broken — try anyway rather than fail outright.
        candidates = list(enumerate(plan))
    return candidates


async def smart_stream(
    model: str,
    messages: list,
    include_usage: bool = False,
    tools: list = None,
    sampling: SamplingParams | None = None,
    plan_override: list | None = None,
    response_id: str | None = None,
    created_ts: int | None = None,
) -> AsyncIterator[str]:
    """Stream with intelligent failover across providers and models.

    Tries the requested model first, then the ranked fallback plan (Kilo free
    models, then FreeModels). On a 429/quota error the next attempt is used.

    A single ``id``/``created`` pair is stamped on every chunk of the response,
    and output is buffered until the current attempt produces its first token:
    a failed attempt therefore contributes nothing to the client stream (no
    duplicated partial text, no second response id). Once output has been
    emitted, failover is refused — the failure is reported as an error frame
    instead so the client never receives two models' text.
    """
    if kilo.is_kilo_model(model):
        await kilo.ensure_models()

    # An explicit plan (combo routing) replaces catalog ranking entirely: the
    # admin's declared order/fan-out is the contract.
    plan = plan_override if plan_override is not None else get_attempt_plan(model, need_tools=bool(tools))
    if plan and plan_override is None:
        from app import triage
        try:
            plan = triage.reorder_plan(plan, triage.analyze(messages, tools))
        except Exception:
            logger.warning("Intent triage reorder failed", exc_info=True)
    try:
        from app.otel_tracing import span as otel_span, set_attrs as otel_attrs
        with otel_span("routing.plan", {
            "model": model, "need_tools": bool(tools),
            "binding_combo": plan_override is not None, "entries": len(plan),
        }) as _rp:
            otel_attrs(_rp, {"plan": str(plan)[:400]})
    except Exception:
        pass
    if not plan:
        yield _error_frame(
            "No upstream models available for this request.",
            "server_error",
            "no_models_available",
        )
        yield "data: [DONE]\n\n"
        return

    response_id = response_id or f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created_ts = created_ts or int(time.time())
    candidates = _candidate_list(model, plan)
    tried: list[str] = []
    _LAST_FALLBACK_CHAIN = []

    for i, (provider, try_model) in candidates:
        model_h = get_model_health(try_model)
        provider_h = _health[provider]

        # Breakers can open mid-request (an earlier attempt in this very
        # request may have tripped them). Honour them immediately so a
        # rate-limited provider is not charged several attempts' worth of
        # quota; the explicitly requested model keeps its first attempt.
        explicit_first = i == 0 and model not in (kilo.AUTO_MODEL, cf.CLOUDFLARE_AUTO_MODEL)
        if not explicit_first and (not provider_h.is_healthy or not model_h.is_healthy):
            tried.append(f"{provider}:{try_model} (skipped: breaker open)")
            _LAST_FALLBACK_CHAIN.append({"model": try_model, "provider": provider, "status": "skipped_breaker"})
            continue

        tried.append(f"{provider}:{try_model}")

        buffer: list[str] = []
        emitted = False
        saw_finish = False
        transient_abort = False

        try:
            async for chunk in _stream_for(provider, try_model, messages, include_usage, tools, sampling):
                if not chunk.startswith("data:"):
                    if emitted:
                        yield chunk
                    else:
                        buffer.append(chunk)
                    continue

                data_str = chunk[6:].strip()
                if data_str == "[DONE]":
                    if emitted:
                        yield chunk
                    else:
                        buffer.append(chunk)
                    continue

                try:
                    parsed = json.loads(data_str)
                except json.JSONDecodeError:
                    if emitted:
                        yield chunk
                    else:
                        buffer.append(chunk)
                    continue

                if isinstance(parsed, dict) and "error" in parsed:
                    if emitted:
                        # Already streaming to the client: failover would mean a
                        # second model's text appended to the first.
                        yield "data: " + data_str + "\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    error_msg = parsed["error"].get("message", "")
                    if is_quota_error(error_msg) or is_quota_error(data_str):
                        raise QuotaError(
                            error_msg or "quota exhausted",
                            provider=provider,
                            model=try_model,
                        )
                    if is_transient_error(error_msg) or is_transient_error(data_str):
                        # Upstream blip before any output — try the next candidate.
                        logger.info(
                            "Transient upstream error from %s:%s, trying next model...",
                            provider, try_model,
                        )
                        model_h.mark_exhausted(error_msg or "transient upstream error")
                        model_h.record_fallback()
                        transient_abort = True
                        break
                    # Non-quota error: forward it and stop
                    yield "data: " + data_str + "\n\n"
                    yield "data: [DONE]\n\n"
                    return

                delta: dict = {}
                if isinstance(parsed, dict) and parsed.get("choices"):
                    first = parsed["choices"][0] or {}
                    delta = first.get("delta") or {}
                    if first.get("finish_reason"):
                        saw_finish = True
                    # One response id for the whole request, including any
                    # fallback attempt that ends up serving it.
                    parsed["id"] = response_id
                    parsed["created"] = created_ts
                    chunk = "data: " + json.dumps(parsed, ensure_ascii=False) + "\n\n"

                is_token = bool(delta.get("content")) or bool(delta.get("tool_calls"))
                if emitted:
                    yield chunk
                else:
                    buffer.append(chunk)
                    if is_token:
                        for pending in buffer:
                            yield pending
                        buffer.clear()
                        emitted = True

            if transient_abort:
                continue

            # Upstream finished without raising and produced output.
            if emitted:
                model_h.mark_healthy()
                provider_h.mark_healthy()
                if i > 0:
                    logger.info("Fallback success: %s served by %s (requested: %s)", try_model, provider, model)
                return
            # Finish with no content (or no frames at all) — not usable output.
            logger.info("Empty upstream response from %s:%s, trying next model...", provider, try_model)
            model_h.mark_exhausted("empty upstream response")
            model_h.record_fallback()
            continue

        except QuotaError as e:
            model_h.mark_exhausted(str(e))
            model_h.record_fallback()
            if _provider_wide_quota(str(e), getattr(e, "status_code", None)):
                provider_h.mark_exhausted(str(e))
                provider_h.record_fallback()
            logger.info("Quota error on %s:%s, trying next model...", provider, try_model)
            if emitted:
                yield _error_frame(str(e) or "quota exhausted", "rate_limit_error", "quota_exhausted")
                yield "data: [DONE]\n\n"
                return
            continue
        except Exception as e:
            logger.error("Unexpected error on %s:%s: %s", provider, try_model, e)
            model_h.mark_exhausted(str(e))
            model_h.record_fallback()
            if _provider_wide_quota(str(e), None):
                provider_h.mark_exhausted(str(e))
                provider_h.record_fallback()
            if emitted:
                yield _error_frame(str(e) or "upstream error", "server_error", "upstream_error")
                yield "data: [DONE]\n\n"
                return
            continue

    # All models exhausted
    yield _error_frame(
        f"All providers exhausted. Tried: {', '.join(tried)}. Retry in 30s.",
        "rate_limit_error",
        "all_providers_exhausted",
    )
    yield "data: [DONE]\n\n"


async def smart_collect(
    model: str,
    messages: list,
    tools: list = None,
    sampling: SamplingParams | None = None,
    plan_override: list | None = None,
) -> tuple[str, str, dict | None]:
    """Collect a full non-streaming response with failover.

    Returns (full_text, actual_model_used, error_or_none).
    When the model returns tool calls, ``full_text`` is a JSON document of
    ``{"content": ..., "tool_calls": [...]}``.

    The attempt loop is private to this function: every candidate accumulates
    into its own buffers, so a failed attempt can never leak partial text or
    tool-call fragments into the attempt that finally succeeds.
    """
    if kilo.is_kilo_model(model):
        await kilo.ensure_models()

    plan = plan_override if plan_override is not None else get_attempt_plan(model, need_tools=bool(tools))
    if plan and plan_override is None:
        from app import triage
        try:
            plan = triage.reorder_plan(plan, triage.analyze(messages, tools))
        except Exception:
            logger.warning("Intent triage reorder failed", exc_info=True)
    try:
        from app.otel_tracing import span as otel_span, set_attrs as otel_attrs
        with otel_span("routing.plan", {
            "model": model, "need_tools": bool(tools),
            "binding_combo": plan_override is not None, "entries": len(plan),
        }) as _rp:
            otel_attrs(_rp, {"plan": str(plan)[:400]})
    except Exception:
        pass
    logger.info("Attempt plan for %s (tools=%s): %s", model, bool(tools), plan[:8])
    if not plan:
        return "", model, {
            "message": "No upstream models available for this request.",
            "type": "server_error",
            "code": "no_models_available",
        }

    candidates = _candidate_list(model, plan)
    tried: list[str] = []

    for i, (provider, try_model) in candidates:
        model_h = get_model_health(try_model)
        provider_h = _health[provider]

        # Breakers can open mid-request (an earlier attempt in this very
        # request may have tripped them). Honour them immediately so a
        # rate-limited provider is not charged several attempts' worth of
        # quota; the explicitly requested model keeps its first attempt.
        explicit_first = i == 0 and model not in (kilo.AUTO_MODEL, cf.CLOUDFLARE_AUTO_MODEL)
        if not explicit_first and (not provider_h.is_healthy or not model_h.is_healthy):
            tried.append(f"{provider}:{try_model} (skipped: breaker open)")
            continue

        tried.append(f"{provider}:{try_model}")

        # Per-attempt accumulators.
        full_text = ""
        actual_model = try_model
        tool_call_acc: dict[int, dict] = {}
        error: dict | None = None
        error_status: int | None = None
        retryable = False

        try:
            async for chunk in _stream_for(provider, try_model, messages, False, tools, sampling):
                if not chunk.startswith("data:"):
                    continue
                data_str = chunk[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    parsed = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                if not isinstance(parsed, dict):
                    continue

                if "error" in parsed:
                    error = parsed["error"]
                    err_text = str(error.get("message", ""))
                    retryable = (
                        is_quota_error(err_text) or is_quota_error(data_str)
                        or is_transient_error(err_text) or is_transient_error(data_str)
                    )
                    break

                if parsed.get("model"):
                    actual_model = parsed["model"]

                choices = parsed.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    full_text += content
                for tc in (delta.get("tool_calls") or []):
                    idx = tc.get("index", 0)
                    slot = tool_call_acc.setdefault(idx, {
                        "id": None,
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    })
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    if tc.get("type"):
                        slot["type"] = tc["type"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["function"]["name"] = fn["name"]
                    args = fn.get("arguments")
                    if args:
                        slot["function"]["arguments"] += args

        except QuotaError as e:
            error = {"message": str(e), "type": "rate_limit_error", "code": "quota_exhausted"}
            error_status = getattr(e, "status_code", None)
            retryable = True
        except Exception as e:
            logger.error("Unexpected error on %s:%s: %s", provider, try_model, e)
            error = {"message": str(e), "type": "server_error", "code": "upstream_error"}
            retryable = True

        if error is None and not full_text and not tool_call_acc:
            logger.info("Empty upstream response from %s:%s, trying next model...", provider, try_model)
            model_h.mark_exhausted("empty upstream response")
            model_h.record_fallback()
            continue

        if error is None:
            model_h.mark_healthy()
            provider_h.mark_healthy()
            if i > 0:
                logger.info("Fallback success: %s served by %s (requested: %s)", try_model, provider, model)
            if tool_call_acc:
                tool_calls = []
                for idx in sorted(tool_call_acc):
                    slot = tool_call_acc[idx]
                    if not slot["id"]:
                        slot["id"] = f"call_{uuid.uuid4().hex[:24]}"
                    tool_calls.append(slot)
                return json.dumps({"content": full_text, "tool_calls": tool_calls}), actual_model, None
            return full_text, actual_model, None

        if retryable:
            err_msg = str(error.get("message", "")) if error else ""
            model_h.mark_exhausted(err_msg)
            model_h.record_fallback()
            if _provider_wide_quota(err_msg, error_status):
                provider_h.mark_exhausted(err_msg)
                provider_h.record_fallback()
            logger.info("Quota error on %s:%s, trying next model...", provider, try_model)
            continue

        # Non-retryable upstream error: report it as-is.
        return "", actual_model, error

    return "", model, {
        "message": f"All providers exhausted. Tried: {', '.join(tried)}. Retry in 30s.",
        "type": "rate_limit_error",
        "code": "all_providers_exhausted",
    }
