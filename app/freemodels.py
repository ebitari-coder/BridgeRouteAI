"""FreeModels upstream: dynamic model catalog with a static fallback.

FreeModels' worker (``FREE_MODELS_URL``) has no ``/models`` endpoint — it only
accepts chat POSTs. The official catalog lives inside the SPA bundle that
serves https://freemodels.pro (the same bundle the chat UI uses to build its
model picker), so we scrape that bundle on a TTL and merge the result with a
static seed list. That keeps routing working when the site is unreachable and
picks up newly published models (e.g. ``sol``, ``terra``, ``kimi-k3``) without
a code change.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

def _site() -> str:
    return (getattr(settings, "freemodels_site_url", "") or "https://freemodels.pro").rstrip("/")

# Landing page -> hashed Vite asset bundle(s).
_SCRIPT_RE = re.compile(r'<script[^>]+src="(/assets/[^"]+\.js)"')
# Catalog entries look like: {id:"claude-sonnet-5",name:"Claude Sonnet 5",provider:"Anthropic",group:...}
_MODEL_RE = re.compile(
    r'\{id:"(?P<id>[^"\s]{1,80})",name:"(?P<name>[^"]{1,120})",provider:"(?P<provider>[^"]{0,60})"'
)

# Static seed: the last known-good catalog. Used when the site cannot be
# reached (and always kept in the pool so no previously advertised model
# disappears on a transient scrape failure).
STATIC_MODELS: list[dict] = [
    {"id": mid, "name": mid, "provider": "FreeModels"}
    for mid in (
        "claude-sonnet-5",
        "gpt-5.2-high",
        "o3-2025-04-16",
        "gemini-3.1-pro-preview",
        "qwen3.7-max",
        "gpt-5.5-instant",
        "deepseek-v4-pro-thinking",
        "mistral-large-3",
        "kimi-k2.6",
        "grok-4.3/text",
        "gpt-5.4-mini-high",
        "gpt-5.4-nano-high",
        "gemini-3.6-flash",
        "gemini-3.1-flash-lite",
        "mistral-small-2603",
        "qwen3.7-plus",
        "mimo-v2.5",
        "glm-5.1",
        "minimax-m3",
        "claude-haiku-4-5-20251001",
    )
]

_CACHE: list[dict] = []
_CACHE_TIME = 0.0


def _headers() -> dict[str, str]:
    # The upstream worker rejects bare requests (403) unless they claim to come
    # from freemodels.pro; the site itself is the source of this catalog.
    site = _site()
    return {
        "User-Agent": "Mozilla/5.0 (compatible; BridgeRouteAI-catalog/1.0)",
        "Referer": f"{site}/",
        "Origin": site,
        "Accept": "*/*",
    }


def _ttl() -> float:
    return float(getattr(settings, "freemodels_catalog_refresh_seconds", 300) or 300)


def _parse_catalog(bundle: str) -> list[dict]:
    """Extract ``{id, name, provider}`` entries from the SPA bundle."""
    found: dict[str, dict] = {}
    for m in _MODEL_RE.finditer(bundle):
        mid = m.group("id").strip()
        if not mid or mid in found:
            continue
        found[mid] = {
            "id": mid,
            "name": m.group("name").strip() or mid,
            "provider": m.group("provider").strip() or "FreeModels",
        }
    return list(found.values())


async def fetch_models(force_refresh: bool = False) -> list[dict]:
    """Scrape the freemodels.pro bundle for the current model catalog."""
    global _CACHE, _CACHE_TIME

    if _CACHE and not force_refresh and (time.time() - _CACHE_TIME) < _ttl():
        return _CACHE

    try:
        async with httpx.AsyncClient(
            timeout=20.0, headers=_headers(), follow_redirects=True
        ) as client:
            home = await client.get(f"{_site()}/")
            if home.status_code != 200:
                raise RuntimeError(f"{_site()} returned HTTP {home.status_code}")

            scripts = [s for s in _SCRIPT_RE.findall(home.text) if s.endswith(".js")]
            if not scripts:
                raise RuntimeError(f"no JS asset bundle linked from {_site()}")

            bundle = ""
            for src in scripts:
                resp = await client.get(f"{_site()}{src}")
                if resp.status_code == 200 and len(resp.content) > len(bundle):
                    bundle = resp.text
            if not bundle:
                raise RuntimeError(f"could not download any asset bundle from {_site()}")

            models = _parse_catalog(bundle)
            if not models:
                raise RuntimeError(f"model catalog not found in {_site()} bundle")

        _CACHE = models
        _CACHE_TIME = time.time()
        logger.info(
            "Discovered %d FreeModels models from %s: %s",
            len(models),
            _site(),
            ", ".join(m["id"] for m in models),
        )
    except Exception as exc:
        logger.warning(
            "FreeModels catalog refresh failed (%s); keeping %d cached model(s)",
            exc,
            len(_CACHE),
        )

    return _CACHE


async def ensure_models() -> list[dict]:
    """Refresh the catalog when it is missing or stale."""
    return await fetch_models()


async def catalog_refresh_loop(interval: float | None = None) -> None:
    """Refresh the FreeModels catalog on a fixed interval until cancelled."""
    seconds = interval or _ttl()
    while True:
        await asyncio.sleep(seconds)
        try:
            await fetch_models(force_refresh=True)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("FreeModels catalog refresh failed: %s", exc)


def get_models() -> list[dict]:
    """Dynamic catalog first, then any static seed ids it does not contain."""
    seen: dict[str, dict] = {}
    for m in (*_CACHE, *STATIC_MODELS):
        if m["id"] not in seen:
            seen[m["id"]] = m
    return list(seen.values())


def get_model_ids() -> list[str]:
    return [m["id"] for m in get_models()]


def is_freemodels_model(model: str) -> bool:
    return model in {m["id"] for m in get_models()}
