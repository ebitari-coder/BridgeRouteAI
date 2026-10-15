"""Septor Labs provider — free models only.

Septor Labs is an OpenAI-compatible gateway (``https://api.septorlabs.com/v1``)
hosting a large multi-modal catalog. Only the models labelled free — ids ending
in ``-free`` on https://septorlabs.com/models — are used here.

The catalog page is public, but every API call (``/v1/models`` included) needs
a Bearer key (``SEPTOR_API_KEY``), so the free list is scraped from the website
and chat stays key-gated.

Free-tier limits (upstream): 50 requests/day and 500K tokens/day on the free
catalog; signup grants $1 of credits.

This module provides:
- live discovery of the ``-free`` catalog (scraped, cached, TTL refresh),
- capability-ranked candidate ordering for the auto-router,
- an OpenAI-compatible streaming client.
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

PROVIDER_NAME = "septorlabs"

FREE_SUFFIX = "-free"

# Seed used until the first successful catalog scrape (and as an offline fallback).
STATIC_FREE_MODELS: list[dict] = [
    {"id": "llama-3.3-70b-instruct-free", "name": "Llama 3.3 70B Instruct (free)",
     "supported_parameters": ["tools", "temperature", "top_p", "max_tokens"], "context_length": 131072},
    {"id": "qwen3-next-80b-a3b-instruct-free", "name": "Qwen3 Next 80B A3B Instruct (free)",
     "supported_parameters": ["tools", "temperature", "top_p", "max_tokens"], "context_length": 131072},
    {"id": "qwen3-coder-free", "name": "Qwen3 Coder (free)",
     "supported_parameters": ["tools", "temperature", "top_p", "max_tokens"], "context_length": 131072},
    {"id": "minimax-m2.5-free", "name": "MiniMax M2.5 (free)",
     "supported_parameters": ["tools", "temperature", "top_p", "max_tokens"], "context_length": 131072},
    {"id": "llama-3.2-3b-instruct-free", "name": "Llama 3.2 3B Instruct (free)",
     "supported_parameters": ["tools", "temperature", "top_p", "max_tokens"], "context_length": 33792},
    {"id": "glm-4.5-air-free", "name": "GLM 4.5 Air (free)",
     "supported_parameters": ["tools", "temperature", "top_p", "max_tokens"], "context_length": 131072},
    {"id": "hermes-3-llama-3.1-405b-free", "name": "Hermes 3 Llama 3.1 405B (free)",
     "supported_parameters": ["tools", "temperature", "top_p", "max_tokens"], "context_length": 131072},
    {"id": "dolphin-mistral-24b-venice-edition-free", "name": "Dolphin Mistral 24B Venice (free)",
     "supported_parameters": ["tools", "temperature", "top_p", "max_tokens"], "context_length": 131072},
    {"id": "gemma-4-31b-it-free", "name": "Gemma 4 31B IT (free)",
     "supported_parameters": ["temperature", "top_p", "max_tokens"], "context_length": 131072},
    {"id": "step-3.5-flash-free", "name": "Step 3.5 Flash (free)",
     "supported_parameters": ["temperature", "top_p", "max_tokens"], "context_length": 131072},
    {"id": "minimax-m2-free", "name": "MiniMax M2 (free)",
     "supported_parameters": ["temperature", "top_p", "max_tokens"], "context_length": 131072},
    {"id": "mistral-nemo-instruct-free", "name": "Mistral Nemo Instruct (free)",
     "supported_parameters": ["temperature", "top_p", "max_tokens"], "context_length": 131072},
    {"id": "nemotron-3-nano-30b-a3b-free", "name": "Nemotron 3 Nano 30B A3B (free)",
     "supported_parameters": ["temperature", "top_p", "max_tokens"], "context_length": 131072},
    {"id": "nemotron-nano-9b-v2-free", "name": "Nemotron Nano 9B v2 (free)",
     "supported_parameters": ["temperature", "top_p", "max_tokens"], "context_length": 131072},
    {"id": "llama-3.1-8b-instruct-free", "name": "Llama 3.1 8B Instruct (free)",
     "supported_parameters": ["temperature", "top_p", "max_tokens"], "context_length": 131072},
    {"id": "llama-3.2-1b-instruct-free", "name": "Llama 3.2 1B Instruct (free)",
     "supported_parameters": ["temperature", "top_p", "max_tokens"], "context_length": 131072},
    {"id": "ling-2.6-flash-free", "name": "Ling 2.6 Flash (free)",
     "supported_parameters": ["temperature", "top_p", "max_tokens"], "context_length": 32768},
]

_CACHE: list[dict] = []
_CACHE_TIME: float = 0.0

# Models whose upstream returned "model missing" are parked here (id -> expiry)
# so the router stops burning attempts on them; they rejoin after the TTL.
_DEAD: dict[str, float] = {}
_DEAD_TTL_SECONDS = 6 * 3600.0

# Card opening in the server-rendered catalog page: each free model is one
# `<a class="group block d-card ...">` element whose href is its id.
_CARD_RE = re.compile(
    r'<a class="group block d-card d-card--hover p-5" href="/models/([a-z0-9][a-z0-9._-]*)">'
)
_CTX_RE = re.compile(r"([\d.]+)K ctx")


def _base_url() -> str:
    return (settings.septor_base_url or "https://api.septorlabs.com/v1").rstrip("/")


def _site_url() -> str:
    return (settings.septor_site_url or "https://septorlabs.com").rstrip("/")


def _ttl() -> float:
    return float(getattr(settings, "septor_catalog_refresh_seconds", 300) or 300)


def _headers() -> dict:
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if settings.septor_api_key:
        headers["Authorization"] = f"Bearer {settings.septor_api_key}"
    return headers


def is_configured() -> bool:
    """Chat calls need a key; the catalog does not."""
    return bool(settings.septor_api_key)


def _parse_catalog_page(html: str) -> list[dict]:
    """Extract the ``-free`` cards from the public models page.

    Cards are rendered as consecutive ``<a class="... d-card ..." href="/models/<id>">``
    elements, so each card's text runs from its opening tag to the next card's.
    That window carries the ``Tools`` badge, the ``Nnk ctx`` line and the price.
    """
    matches = list(_CARD_RE.finditer(html))
    if not matches:
        return []

    models: list[dict] = []
    for i, m in enumerate(matches):
        slug = m.group(1)
        if not slug.endswith(FREE_SUFFIX):
            continue

        segment = html[m.end(): matches[i + 1].start() if i + 1 < len(matches) else m.end() + 4000]

        ctx_hits = _CTX_RE.findall(segment)
        context_length = int(float(ctx_hits[-1]) * 1024) if ctx_hits else None
        has_tools = ">Tools<" in segment or ">Tools" in segment

        name = slug[: -len(FREE_SUFFIX)].replace("-", " ").strip()
        models.append({
            "id": slug,
            "name": f"{name} (free)" if name else slug,
            "supported_parameters": (
                ["tools", "temperature", "top_p", "max_tokens"]
                if has_tools else ["temperature", "top_p", "max_tokens"]
            ),
            "context_length": context_length,
        })
    return models


async def _fetch_models_api() -> list[dict]:
    """Authoritative catalog from ``GET /v1/models`` (needs the API key)."""
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(f"{_base_url()}/models", headers=_headers())
        if resp.status_code != 200:
            raise RuntimeError(f"/v1/models HTTP {resp.status_code}")
        data = resp.json().get("data") or []

    models: list[dict] = []
    for m in data:
        mid = m.get("id") or ""
        if not mid.endswith(FREE_SUFFIX):
            continue
        if m.get("is_free") is False:
            continue
        if (m.get("status") or "active") != "active":
            continue
        caps = m.get("capabilities") or {}
        params = ["temperature", "top_p", "max_tokens"]
        if caps.get("tools"):
            params.insert(0, "tools")
        models.append({
            "id": mid,
            "name": m.get("display_name") or mid,
            "supported_parameters": params,
            "context_length": m.get("context_window"),
        })
    return models


async def _fetch_models_site() -> list[dict]:
    """Fallback: scrape the public catalog page (works without a key)."""
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        resp = await client.get(f"{_site_url()}/models")
        if resp.status_code != 200:
            raise RuntimeError(f"/models HTTP {resp.status_code}")
        free = _parse_catalog_page(resp.text)
        if not free:
            raise RuntimeError("no `-free` models found on catalog page")
        return free


async def fetch_models(force_refresh: bool = False) -> list[dict]:
    """Refresh the ``-free`` catalog: API first (key), public page as fallback."""
    global _CACHE, _CACHE_TIME

    if _CACHE and not force_refresh and (time.time() - _CACHE_TIME) < _ttl():
        return _CACHE

    try:
        free: list[dict] = []
        if is_configured():
            try:
                free = await _fetch_models_api()
            except Exception as exc:
                logger.warning("Septor Labs /v1/models failed (%s); falling back to site", exc)
        if not free:
            free = await _fetch_models_site()
        if not free:
            raise RuntimeError("no `-free` models found")
        _CACHE = free
        _CACHE_TIME = time.time()
        logger.info(
            "Discovered %d Septor Labs free models: %s",
            len(free),
            ", ".join(m["id"] for m in free),
        )
    except Exception as exc:
        logger.warning(
            "Septor Labs catalog refresh failed (%s); keeping %d cached model(s)",
            exc,
            len(_CACHE),
        )

    return _CACHE


async def ensure_models() -> list[dict]:
    """Refresh the catalog when it is missing or stale (no key required)."""
    if not is_configured():
        # Nothing can be served without a key, so do not scrape for nothing.
        return get_models()
    return await fetch_models()


async def catalog_refresh_loop(interval: float | None = None) -> None:
    """Refresh the Septor Labs catalog on a fixed interval until cancelled."""
    seconds = interval or _ttl()
    while True:
        await asyncio.sleep(seconds)
        try:
            await fetch_models(force_refresh=True)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Septor Labs catalog refresh failed: %s", exc)


def get_models() -> list[dict]:
    """Live catalog first, then the static seed for anything it does not list."""
    seen: dict[str, dict] = {}
    for m in (*_CACHE, *STATIC_FREE_MODELS):
        seen.setdefault(m["id"], m)
    return list(seen.values())


def is_septorlabs_model(model: str) -> bool:
    """Catalog membership decides ownership (bare ``-free`` ids)."""
    if not model:
        return False
    return model in {m["id"] for m in get_models()}


def quarantine(model: str, reason: str = "") -> None:
    """Park a model whose upstream reported it as missing (self-heals via TTL)."""
    _DEAD[model] = time.time() + _DEAD_TTL_SECONDS
    logger.warning(
        "Septor Labs quarantining %s for %.0fh%s",
        model,
        _DEAD_TTL_SECONDS / 3600,
        f": {reason}" if reason else "",
    )


def _drop_expired_quarantine() -> None:
    now = time.time()
    for mid in [mid for mid, expiry in _DEAD.items() if expiry < now]:
        del _DEAD[mid]


def get_free_models() -> list[dict]:
    """Free models available to the auto-router.

    Chat calls need an API key — without one we return nothing so the router
    does not burn attempts on 401s. Quarantined (upstream-missing) models are
    excluded until their TTL lapses.
    """
    if not is_configured():
        return []
    _drop_expired_quarantine()
    return [m for m in get_models() if m["id"] not in _DEAD]


def rank_free_models(need_tools: bool = False) -> list[str]:
    """Free model ids ordered best-first (tool support / larger context first)."""
    models = get_free_models()
    if need_tools:
        models = [m for m in models if "tools" in (m.get("supported_parameters") or [])]

    def _score(m: dict) -> tuple:
        ctx = m.get("context_length") or 0
        return (0 if "tools" in (m.get("supported_parameters") or []) else 1, -ctx, m["id"])

    return [m["id"] for m in sorted(models, key=_score)]


async def septorlabs_stream_single(
    model: str,
    messages: list,
    include_usage: bool = False,
    tools: list = None,
    sampling: SamplingParams | None = None,
) -> AsyncIterator[str]:
    """Stream a chat completion from Septor Labs (OpenAI-compatible)."""
    if not is_configured():
        raise QuotaError(
            "Septor Labs API key not configured (set SEPTOR_API_KEY)",
            provider=PROVIDER_NAME,
            model=model,
            status_code=401,
        )

    chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    openai_messages = to_openai_messages(messages)

    payload = {
        "model": model,
        "messages": openai_messages,
        "stream": True,
    }
    if tools:
        payload["tools"] = tools
    apply_sampling(payload, sampling)

    prompt_tokens_est = max(
        1, sum(len(str(msg_get(msg, "content") or "")) for msg in messages) // 4
    )
    accumulated_text = ""
    yielded_role = False

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{_base_url()}/chat/completions",
            json=payload,
            headers=_headers(),
        ) as response:
            if response.status_code != 200:
                body_bytes = await response.aread()
                error_msg = body_bytes.decode(errors="replace")[:500]
                if (
                    "upstream_model_missing" in error_msg
                    or "model_not_found" in error_msg
                ):
                    quarantine(model, f"HTTP {response.status_code}")
                raise QuotaError(
                    f"Septor Labs {response.status_code}: {error_msg}",
                    provider=PROVIDER_NAME,
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
                    if yielded_role:
                        yield sse_chunk(chat_id, created, model, finish_reason="stop")
                    yield "data: [DONE]\n\n"
                    return
                if not line_data:
                    continue

                try:
                    parsed = json.loads(line_data)
                except json.JSONDecodeError:
                    continue

                if "error" in parsed:
                    err = parsed["error"]
                    if not isinstance(err, dict):
                        err = {"message": str(err), "type": "api_error"}
                    yield "data: " + json.dumps({"error": err}) + "\n\n"
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

            # Stream ended without an explicit finish signal
            if include_usage:
                yield usage_chunk(chat_id, created, model, prompt_tokens_est, accumulated_text)
            if yielded_role:
                yield sse_chunk(chat_id, created, model, finish_reason="stop")
            yield "data: [DONE]\n\n"
