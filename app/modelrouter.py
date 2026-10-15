"""ModelRouter provider — free text-chat models only.

ModelRouter exposes an OpenAI-compatible gateway at
``https://modelrouter.id/v1``. Its catalog contains paid, VIP, image, TTS,
and free models, so this adapter discovers ``GET /v1/models`` and keeps only
models that are explicitly free. A ModelRouter API key is required for both
catalog access and chat completions.
"""

import asyncio
import json
import logging
import re
import time
import uuid
from typing import AsyncIterator

import httpx

from app.config import settings
from app.provider_utils import (
    QuotaError,
    SamplingParams,
    apply_sampling,
    msg_get,
    sse_chunk,
    to_openai_messages,
    usage_chunk,
)

logger = logging.getLogger(__name__)

PROVIDER_NAME = "modelrouter"
DEFAULT_BASE_URL = "https://modelrouter.id/v1"
FREE_MARKERS = {"free", "free tier", "free_tier", "free limited", "free_limited"}
NON_TEXT_MARKERS = ("image", "tts", "speech", "audio")

# Offline seed for the free text models currently shown in ModelRouter's
# public catalog. The live catalog replaces/extends this list when available.
STATIC_FREE_MODELS: list[dict] = [
    {"id": "gemini-3.6-flash-high", "name": "Gemini 3.6 Flash High (Free)", "context_length": 1048576, "supported_parameters": []},
    {"id": "ling-3.1-flash", "name": "Ling 3.1 Flash (Free)", "context_length": 131072, "supported_parameters": []},
    {"id": "fledge-alpha", "name": "Fledge Alpha (Free)", "context_length": 131072, "supported_parameters": []},
    {"id": "godmode", "name": "Godmode (Free)", "context_length": 131072, "supported_parameters": []},
    {"id": "longcat-2.5-preview", "name": "LongCat 2.5 Preview (Free)", "context_length": 131072, "supported_parameters": []},
    {"id": "nemotron-3-ultra", "name": "Nemotron 3 Ultra (Free)", "context_length": 131072, "supported_parameters": []},
    {"id": "nemotron-3.5-lightning", "name": "Nemotron 3.5 Lightning (Free)", "context_length": 131072, "supported_parameters": []},
    {"id": "ling-3.0-flash-fin", "name": "Ling 3.0 Flash Fin (Free)", "context_length": 131072, "supported_parameters": []},
    {"id": "space-bunny", "name": "Space Bunny (Free)", "context_length": 131072, "supported_parameters": []},
    {"id": "mimo-v2.6-flash", "name": "MiMo V2.6 Flash (Free)", "context_length": 1048576, "supported_parameters": []},
    {"id": "muse-spark-1.2-contributor", "name": "Muse Spark 1.2 Contributor (Free)", "context_length": 1048576, "supported_parameters": []},
    {"id": "muse-spark-1.3-contributor", "name": "Muse Spark 1.3 Contributor (Free)", "context_length": 1048576, "supported_parameters": []},
    {"id": "mr-vip", "name": "MR VIP (Free)", "context_length": 1048576, "supported_parameters": []},
    {"id": "mimo-v2.5", "name": "MiMo V2.5 (Free)", "context_length": 262144, "supported_parameters": []},
]

_CACHE: list[dict] = []
_CACHE_TIME: float = 0.0
_DEAD: dict[str, float] = {}
_DEAD_TTL_SECONDS = 6 * 3600.0


def _base_url() -> str:
    return (settings.modelrouter_base_url or DEFAULT_BASE_URL).rstrip("/")


def _ttl() -> float:
    return float(getattr(settings, "modelrouter_catalog_refresh_seconds", 300) or 300)


def _headers(accept: str = "application/json") -> dict[str, str]:
    headers = {"Accept": accept, "Content-Type": "application/json"}
    if settings.modelrouter_api_key:
        headers["Authorization"] = f"Bearer {settings.modelrouter_api_key}"
    return headers


def _mr_enabled() -> bool:
    """Kill switch — MODELROUTER_ENABLED=false removes ModelRouter from
    routing/catalogs/plans (default: disabled). The account has exhausted
    its 50-request free tier (403 free_tier_limit_exceeded), so probes and
    requests fail until quota resets or the balance is topped up.
    """
    try:
        return bool(settings.modelrouter_enabled)
    except Exception:
        return False


def is_configured() -> bool:
    return bool(settings.modelrouter_api_key)


def _is_zero_price(value) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value is False
    if isinstance(value, (int, float)):
        return value == 0
    text = str(value).strip().lower().replace("$", "")
    if text in FREE_MARKERS or text in {"0", "0.0", "0.00", "free / 1m", "free/1m"}:
        return True
    try:
        return float(text) == 0
    except ValueError:
        return False


def _pricing_values(model: dict) -> tuple[object, object]:
    pricing = model.get("pricing") or {}
    if not isinstance(pricing, dict):
        pricing = {}
    input_price = next(
        (pricing.get(name) for name in ("prompt", "input", "input_price", "input_cost_per_token", "input_per_token") if pricing.get(name) is not None),
        None,
    )
    output_price = next(
        (pricing.get(name) for name in ("completion", "output", "output_price", "output_cost_per_token", "output_per_token") if pricing.get(name) is not None),
        None,
    )
    input_price = input_price if input_price is not None else next(
        (model.get(name) for name in ("input_price", "input_cost_per_token", "prompt_price") if model.get(name) is not None),
        None,
    )
    output_price = output_price if output_price is not None else next(
        (model.get(name) for name in ("output_price", "output_cost_per_token", "completion_price") if model.get(name) is not None),
        None,
    )
    return input_price, output_price


def _is_free_model(model: dict) -> bool:
    """Recognize ModelRouter's free entries across catalog field variants."""
    if model.get("is_free") is True or model.get("free") is True:
        return True
    tier = str(model.get("tier") or model.get("access_tier") or "").strip().lower()
    if tier in FREE_MARKERS:
        return True
    input_price, output_price = _pricing_values(model)
    return _is_zero_price(input_price) and _is_zero_price(output_price)


def _is_text_model(model: dict) -> bool:
    model_id = str(model.get("id") or "").lower()
    if any(marker in model_id for marker in NON_TEXT_MARKERS):
        return False
    for key in ("type", "task", "modality", "category"):
        value = str(model.get(key) or "").lower()
        if any(marker in value for marker in NON_TEXT_MARKERS):
            return False
    return True


def _normalize_model(model: dict) -> dict | None:
    model_id = model.get("id")
    if not model_id or not _is_text_model(model):
        return None
    capabilities = model.get("capabilities") or {}
    if not isinstance(capabilities, dict):
        capabilities = {}
    supported = list(model.get("supported_parameters") or [])
    if capabilities.get("tools") or capabilities.get("function_call"):
        if "tools" not in supported:
            supported.append("tools")
    return {
        "id": str(model_id),
        "name": model.get("name") or model.get("display_name") or model_id,
        "context_length": model.get("context_length") or model.get("context_window") or model.get("max_context_tokens"),
        "supported_parameters": supported,
        "vision": bool(model.get("vision") or capabilities.get("vision")),
        "reasoning": bool(model.get("reasoning") or capabilities.get("reasoning")),
    }


async def _fetch_public_free_ids(client: httpx.AsyncClient) -> set[str]:
    """Read free pricing from ModelRouter's public catalog page.

    The OpenAI-compatible ``/v1/models`` response currently exposes model IDs
    but not pricing/access metadata. The public catalog is therefore the
    authoritative free/paid filter; the API catalog remains authoritative for
    which IDs are callable through the gateway.
    """
    response = await client.get("https://modelrouter.id/models")
    if response.status_code != 200:
        raise RuntimeError(f"public catalog HTTP {response.status_code}")

    free_ids: set[str] = set()
    for card in re.split(r'(?=<a href="/models/)', response.text):
        id_match = re.search(r'<span class="truncate">([^<]+)</span>', card)
        if not id_match:
            continue
        # Require both input and output to be explicitly FREE. This excludes
        # models with only free output, promotional discounts, or VIP-only
        # paid input while retaining free/VIP-free catalog entries.
        text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", card)).strip()
        if re.search(r"Input\s+FREE(?:\s|/)", text, re.I) and re.search(r"Output\s+FREE(?:\s|/)", text, re.I):
            free_ids.add(id_match.group(1).strip())
    return free_ids


async def fetch_models(force_refresh: bool = False) -> list[dict]:
    """Fetch and cache the free text-chat catalog from ModelRouter."""
    global _CACHE, _CACHE_TIME
    if not _mr_enabled():
        return []
    if _CACHE and not force_refresh and (time.time() - _CACHE_TIME) < _ttl():
        return _CACHE
    if not is_configured():
        return _CACHE

    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            response = await client.get(f"{_base_url()}/models", headers=_headers())
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
            payload = response.json()
            raw_models = payload.get("data") if isinstance(payload, dict) else payload
            raw_models = raw_models or []
            public_free_ids = set()
            try:
                public_free_ids = await _fetch_public_free_ids(client)
            except Exception as exc:
                logger.warning("ModelRouter public free catalog unavailable: %s", exc)

        models = []
        for raw in raw_models:
            if not isinstance(raw, dict):
                continue
            if not _is_free_model(raw) and str(raw.get("id") or "") not in public_free_ids:
                continue
            normalized = _normalize_model(raw)
            if normalized is not None:
                models.append(normalized)
        if not models:
            raise RuntimeError("no free text-chat models in catalog")
        _CACHE = models
        _CACHE_TIME = time.time()
        logger.info("Discovered %d ModelRouter free models: %s", len(models), ", ".join(m["id"] for m in models))
    except Exception as exc:
        logger.warning("ModelRouter catalog refresh failed (%s); keeping %d cached model(s)", exc, len(_CACHE))
    return _CACHE


async def ensure_models() -> list[dict]:
    return await fetch_models()


async def catalog_refresh_loop(interval: float | None = None) -> None:
    seconds = interval or _ttl()
    while True:
        await asyncio.sleep(seconds)
        try:
            await fetch_models(force_refresh=True)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("ModelRouter catalog refresh failed: %s", exc)


def get_models() -> list[dict]:
    # Once the live API catalog has loaded, do not re-add stale seed IDs that
    # are no longer callable. The seed is only an offline fallback.
    if _CACHE:
        return list(_CACHE)
    seen: dict[str, dict] = {}
    for model in (*_CACHE, *STATIC_FREE_MODELS):
        seen.setdefault(model["id"], model)
    return list(seen.values())


def is_modelrouter_model(model: str) -> bool:
    if not _mr_enabled():
        return False
    # Do not claim overlapping model IDs when the provider has no usable key;
    # another configured upstream should retain ownership in that case.
    return is_configured() and bool(model) and model in {m["id"] for m in get_models()}


def quarantine(model: str, reason: str = "") -> None:
    _DEAD[model] = time.time() + _DEAD_TTL_SECONDS
    logger.warning("ModelRouter quarantining %s for %.0fh%s", model, _DEAD_TTL_SECONDS / 3600, f": {reason}" if reason else "")


def _drop_expired_quarantine() -> None:
    now = time.time()
    for model, expiry in list(_DEAD.items()):
        if expiry <= now:
            del _DEAD[model]


def get_free_models() -> list[dict]:
    if not _mr_enabled():
        return []
    if not is_configured():
        return []
    _drop_expired_quarantine()
    return [m for m in get_models() if m["id"] not in _DEAD]


def rank_free_models(need_tools: bool = False) -> list[str]:
    if not _mr_enabled():
        return []
    models = get_free_models()
    if need_tools:
        models = [m for m in models if "tools" in (m.get("supported_parameters") or [])]

    def score(model: dict) -> tuple:
        params = model.get("supported_parameters") or []
        return (0 if "tools" in params else 1, 0 if model.get("reasoning") else 1, -(model.get("context_length") or 0), model["id"])

    return [m["id"] for m in sorted(models, key=score)]


async def modelrouter_stream_single(
    model: str,
    messages: list,
    include_usage: bool = False,
    tools: list = None,
    sampling: SamplingParams | None = None,
) -> AsyncIterator[str]:
    """Stream an OpenAI-compatible chat completion from ModelRouter."""
    if not is_configured():
        raise QuotaError("ModelRouter API key not configured (set MODELROUTER_API_KEY)", provider=PROVIDER_NAME, model=model, status_code=401)

    chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    payload = {"model": model, "messages": to_openai_messages(messages), "stream": True}
    if tools:
        payload["tools"] = tools
    apply_sampling(payload, sampling)
    prompt_tokens_est = max(1, sum(len(str(msg_get(message, "content") or "")) for message in messages) // 4)
    accumulated_text = ""
    yielded_role = False

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", f"{_base_url()}/chat/completions", json=payload, headers=_headers("text/event-stream")) as response:
            if response.status_code != 200:
                body = (await response.aread()).decode(errors="replace")[:500]
                if response.status_code == 400 and "model" in body.lower() and "not" in body.lower():
                    quarantine(model, f"HTTP {response.status_code}")
                raise QuotaError(f"ModelRouter {response.status_code}: {body}", provider=PROVIDER_NAME, model=model, status_code=response.status_code)

            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    if include_usage:
                        yield usage_chunk(chat_id, created, model, prompt_tokens_est, accumulated_text)
                    if yielded_role:
                        yield sse_chunk(chat_id, created, model, finish_reason="stop")
                    yield "data: [DONE]\n\n"
                    return
                if not data:
                    continue
                try:
                    parsed = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if parsed.get("error"):
                    error = parsed["error"]
                    if not isinstance(error, dict):
                        error = {"message": str(error), "type": "api_error"}
                    yield "data: " + json.dumps({"error": error}) + "\n\n"
                    yield "data: [DONE]\n\n"
                    return
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

            if include_usage:
                yield usage_chunk(chat_id, created, model, prompt_tokens_est, accumulated_text)
            if yielded_role:
                yield sse_chunk(chat_id, created, model, finish_reason="stop")
            yield "data: [DONE]\n\n"
