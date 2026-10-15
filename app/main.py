import asyncio
import logging
import json
import uuid
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.routes import chat, models, dashboard, api_keys, users, settings, logs, audit, anthropic, auth, combos, tools as tools_routes, agentic_sessions as sessions_routes, evals as evals_routes
from app.redis_client import get_redis, close_redis
from app.middleware.logging import RequestLoggingMiddleware
from app.middleware.auth import APIKeyMiddleware
from app.middleware.rate_limit import limiter, rate_limit_exceeded_handler
from app.middleware.security import SecurityHeadersMiddleware
from app.config import settings as app_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_migrations():
    """Run Alembic migrations at startup via a subprocess to avoid event loop conflicts."""
    import subprocess, sys, os
    alembic_ini = os.path.join(os.path.dirname(os.path.dirname(__file__)), "alembic.ini")
    logger.info("Running database migrations...")
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "-c", alembic_ini, "upgrade", "head"],
        capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)),
    )
    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            logger.info("alembic: %s", line)
    if result.returncode != 0:
        logger.error("Alembic failed (code %d): %s", result.returncode, result.stderr)
        raise RuntimeError(f"Alembic migration failed: {result.stderr}")
    logger.info("Database migrations complete")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    logger.info("Starting FreeModels Bridge API")

    # Phase 3: OTel tracing (no-op unless OTEL_EXPORTER_OTLP_ENDPOINT set)
    from app.otel_tracing import init_tracing
    init_tracing()

    # Run Postgres migrations
    try:
        run_migrations()
    except Exception as exc:
        logger.error("Failed to run migrations: %s", exc)
        raise

    # Seed superadmin user and API key
    try:
        import asyncio
        from app.seed import seed_superadmin
        await seed_superadmin()
    except Exception:
        # Don't raise — seeding failure shouldn't block startup — but log the full
        # traceback: a swallowed seed failure on a fresh database used to be silent.
        logger.exception("Failed to seed superadmin")

    # Refresh router combo cache from database
    try:
        from app.database import async_session
        from app.providers import refresh_active_combos
        async with async_session() as session:
            await refresh_active_combos(session)
        logger.info("Router combo cache refreshed")
    except Exception as exc:
        logger.warning("Failed to refresh combo cache: %s", exc)

    # Verify Redis
    try:
        r = await get_redis()
        await r.ping()
        logger.info("Redis connection established")
    except Exception as exc:
        logger.warning("Redis not available at startup: %s", exc)

    # Warm the Kilo free-model catalog
    from app import kilo
    logger.info(
        "Kilo API key: %s",
        "configured" if app_settings.kilo_api_key else "not set (anonymous, lower rate limits)",
    )
    try:
        await kilo.fetch_kilo_models()
        logger.info("Kilo catalog loaded: %d free models available", len(kilo.get_free_models()))
    except Exception as exc:
        logger.warning("Kilo catalog discovery failed: %s", exc)

    # Cloudflare Workers AI status
    from app import cloudflare as cf
    if app_settings.cloudflare_account_id and app_settings.cloudflare_api_key:
        logger.info(
            "Cloudflare Workers AI enabled (%d free models available)",
            len(cf.get_free_models()),
        )
    else:
        logger.info("Cloudflare Workers AI not configured (CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_KEY)")

    # BazaarLink status
    from app import bazaarlink as bl
    if app_settings.bazaarlink_api_key:
        logger.info(
            "BazaarLink enabled (%d free models available)",
            len(bl.get_free_models()),
        )
    else:
        logger.info("BazaarLink not configured (BAZAARLINK_API_KEY)")

    # UnoRouter status (catalog is public; API key needed for chat completions)
    from app import unorouter as uo
    await uo.fetch_unorouter_models()
    model_count = len(uo.get_free_models())
    if app_settings.unorouter_api_key:
        logger.info("UnoRouter enabled (%d free models available)", model_count)
    elif model_count > 0:
        logger.info(
            "UnoRouter listed (%d free models, no API key — chat completions unavailable)",
            model_count,
        )
    else:
        logger.info("UnoRouter catalog unavailable (UNOROUTER_API_KEY or network)")

    # Keep the catalog fresh in the background
    refresh_task = asyncio.create_task(kilo.catalog_refresh_loop())
    logger.info(
        "Kilo catalog refresher started (every %ds)",
        app_settings.kilo_catalog_refresh_seconds,
    )

    # Start UnoRouter stability worker (checks uptime/success periodically)
    from app import unorouter as uo
    uo_task = asyncio.create_task(uo.stability_worker())
    logger.info(
        "UnoRouter stability worker started (every %ds)",
        app_settings.unorouter_stability_check_seconds,
    )

    # Start Xkiro stability worker (health-checks free models periodically)
    from app import xkiro as xk
    xk_task = asyncio.create_task(xk.stability_worker())
    logger.info(
        "Xkiro stability worker started (every %ds)",
        app_settings.xkiro_stability_check_seconds,
    )

    # Start ModelScope stability worker (health-checks free models periodically)
    from app import modelscope as ms
    ms_task = asyncio.create_task(ms.stability_worker())
    logger.info(
        "ModelScope stability worker started (every %ds)",
        app_settings.modelscope_stability_check_seconds,
    )

    # Start LLM7 stability worker (health-checks free turbo models periodically)
    from app import llm7
    llm7_task = asyncio.create_task(llm7.stability_worker())
    logger.info(
        "LLM7 stability worker started (every %ds)",
        app_settings.llm7_stability_check_seconds,
    )

    # Pre-fetch Agnes AI catalog
    from app import agnes
    try:
        await agnes.ensure_models()
        logger.info("Agnes AI catalog loaded")
    except Exception as exc:
        logger.warning("Agnes AI catalog refresh failed: %s", exc)

    # Pre-fetch the FreeModels catalog (scraped from freemodels.pro) and keep
    # it fresh so newly published models appear without a restart.
    from app import freemodels as fm
    try:
        await fm.ensure_models()
        logger.info(
            "FreeModels catalog loaded (%d models: %s)",
            len(fm.get_models()),
            ", ".join(m["id"] for m in fm.get_models()),
        )
    except Exception as exc:
        logger.warning("FreeModels catalog refresh failed: %s", exc)
    fm_task = asyncio.create_task(fm.catalog_refresh_loop())
    logger.info(
        "FreeModels catalog refresher started (every %ds)",
        app_settings.freemodels_catalog_refresh_seconds,
    )

    # Pre-fetch the Composite (Lucidity) free-model catalog and keep it fresh.
    from app import composite as cx
    try:
        await cx.ensure_models()
        logger.info(
            "Composite catalog loaded (%d free models: %s)",
            len(cx.get_models()),
            ", ".join(m["id"] for m in cx.get_models()),
        )
    except Exception as exc:
        logger.warning("Composite catalog refresh failed: %s", exc)
    if cx.is_configured():
        logger.info("Composite chat enabled (COMPOSITE_API_KEY set)")
    else:
        logger.warning(
            "Composite chat disabled — set COMPOSITE_API_KEY to use %s",
            ", ".join(m["id"] for m in cx.get_models()),
        )
    cx_task = asyncio.create_task(cx.catalog_refresh_loop())
    logger.info(
        "Composite catalog refresher started (every %ds)",
        app_settings.composite_catalog_refresh_seconds,
    )

    # Pre-fetch the Septor Labs free-model catalog and keep it fresh.
    from app import septorlabs as sx
    try:
        await sx.ensure_models()
        logger.info(
            "Septor Labs catalog loaded (%d free models: %s)",
            len(sx.get_models()),
            ", ".join(m["id"] for m in sx.get_models()),
        )
    except Exception as exc:
        logger.warning("Septor Labs catalog refresh failed: %s", exc)
    if sx.is_configured():
        logger.info("Septor Labs chat enabled (SEPTOR_API_KEY set)")
    else:
        logger.warning(
            "Septor Labs chat disabled — set SEPTOR_API_KEY to use %s",
            ", ".join(m["id"] for m in sx.get_models()),
        )
    sx_task = asyncio.create_task(sx.catalog_refresh_loop())
    logger.info(
        "Septor Labs catalog refresher started (every %ds)",
        app_settings.septor_catalog_refresh_seconds,
    )

    # Pre-fetch the Routeway free-model catalog and keep it fresh.
    from app import routeway as rw
    try:
        await rw.ensure_models()
        logger.info(
            "Routeway catalog loaded (%d free models: %s)",
            len(rw.get_models()),
            ", ".join(m["id"] for m in rw.get_models()),
        )
    except Exception as exc:
        logger.warning("Routeway catalog refresh failed: %s", exc)
    if rw.is_configured():
        logger.info("Routeway chat enabled (ROUTEWAY_API_KEY set)")
    else:
        logger.warning(
            "Routeway chat disabled — set ROUTEWAY_API_KEY to use %s",
            ", ".join(m["id"] for m in rw.get_models()),
        )
    rw_task = asyncio.create_task(rw.catalog_refresh_loop())
    logger.info(
        "Routeway catalog refresher started (every %ds)",
        app_settings.routeway_catalog_refresh_seconds,
    )

    # Pre-fetch the Top Tools AI model catalog and keep it fresh.
    from app import toptools as tt
    try:
        await tt.ensure_models()
        logger.info(
            "Top Tools AI catalog loaded (%d models: %s)",
            len(tt.get_models()),
            ", ".join(m["id"] for m in tt.get_models()),
        )
    except Exception as exc:
        logger.warning("Top Tools AI catalog refresh failed: %s", exc)
    if tt.is_configured():
        logger.info("Top Tools AI chat enabled (TOPTOOLS_API_KEY set)")
    else:
        logger.warning(
            "Top Tools AI chat disabled — set TOPTOOLS_API_KEY to use %s",
            ", ".join(m["id"] for m in tt.get_models()),
        )
    tt_task = asyncio.create_task(tt.catalog_refresh_loop())
    logger.info(
        "Top Tools AI catalog refresher started (every %ds)",
        app_settings.toptools_catalog_refresh_seconds,
    )

    # Pre-fetch the DreamPrompting model catalog and keep it fresh.
    from app import dreamprompting as dp
    try:
        await dp.ensure_models()
        logger.info(
            "DreamPrompting catalog loaded (%d models: %s)",
            len(dp.get_models()),
            ", ".join(m["id"] for m in dp.get_models()),
        )
    except Exception as exc:
        logger.warning("DreamPrompting catalog refresh failed: %s", exc)
    if dp.is_configured():
        logger.info("DreamPrompting chat enabled (DREAMPROMPTING_API_KEY set)")
    else:
        logger.warning(
            "DreamPrompting chat disabled — set DREAMPROMPTING_API_KEY to use %s",
            ", ".join(m["id"] for m in dp.get_models()),
        )
    dp_task = asyncio.create_task(dp.catalog_refresh_loop())
    logger.info(
        "DreamPrompting catalog refresher started (every %ds)",
        app_settings.dreamprompting_catalog_refresh_seconds,
    )

    # Pre-fetch the Clean APIs model catalog and keep it fresh.
    from app import cleanapis as cp
    try:
        await cp.ensure_models()
        logger.info(
            "Clean APIs catalog loaded (%d models: %s)",
            len(cp.get_models()),
            ", ".join(m["id"] for m in cp.get_models()),
        )
    except Exception as exc:
        logger.warning("Clean APIs catalog refresh failed: %s", exc)
    if cp.is_configured():
        logger.info("Clean APIs chat enabled (CLEANAPIS_API_KEY set)")
    else:
        logger.warning(
            "Clean APIs chat disabled — set CLEANAPIS_API_KEY to use %s",
            ", ".join(m["id"] for m in cp.get_models()),
        )
    cp_task = asyncio.create_task(cp.catalog_refresh_loop())
    logger.info(
        "Clean APIs catalog refresher started (every %ds)",
        app_settings.cleanapis_catalog_refresh_seconds,
    )

    # Pre-fetch the ModelRouter free text-chat catalog and keep it fresh.
    from app import modelrouter as mr
    try:
        await mr.ensure_models()
        logger.info(
            "ModelRouter catalog loaded (%d free models: %s)",
            len(mr.get_models()),
            ", ".join(m["id"] for m in mr.get_models()),
        )
    except Exception as exc:
        logger.warning("ModelRouter catalog refresh failed: %s", exc)
    if mr.is_configured():
        logger.info("ModelRouter chat enabled (MODELROUTER_API_KEY set)")
    else:
        logger.warning(
            "ModelRouter chat disabled — set MODELROUTER_API_KEY to use its free models"
        )
    mr_task = asyncio.create_task(mr.catalog_refresh_loop())
    logger.info(
        "ModelRouter catalog refresher started (every %ds)",
        app_settings.modelrouter_catalog_refresh_seconds,
    )

    # Pre-fetch the Tiarina Cloud catalog and keep it fresh.
    from app import tiarina as tr
    try:
        await tr.ensure_models()
        logger.info(
            "Tiarina catalog loaded (%d models: %s)",
            len(tr.get_models()),
            ", ".join(m["id"] for m in tr.get_models()),
        )
    except Exception as exc:
        logger.warning("Tiarina catalog refresh failed: %s", exc)
    if tr.is_configured():
        logger.info("Tiarina chat enabled (TIARINA_API_KEY set)")
    else:
        logger.warning("Tiarina chat disabled — set TIARINA_API_KEY to use its models")
    tr_task = asyncio.create_task(tr.catalog_refresh_loop())
    logger.info(
        "Tiarina catalog refresher started (every %ds)",
        app_settings.tiarina_catalog_refresh_seconds,
    )

    # Capability probe warmup: probe a small set of free models from the
    # tool-strong providers so the first tool-bearing request has live data.
    # This runs in the background and never blocks startup.
    if getattr(app_settings, "capability_probe_enabled", True):
        from app import model_capability as mc
        from app import providers as prov
        from app import kilo, composite as cx, septorlabs as sx, routeway as rw
        from app import kiosapi as ka, agnes, modelrouter as mr
        from app import zyloai as zy, cleanapis as cp, dreamprompting as dp, toptools as tt
        from app import unorouter as uo
        probe_models = []
        try:
            for p_name, getter in [
                (prov.PROVIDER_KILO, kilo.get_free_models),
                (prov.PROVIDER_COMPOSITE, cx.get_free_models),
                (prov.PROVIDER_SEPTORLABS, sx.get_free_models),
                (prov.PROVIDER_ROUTEWAY, rw.get_free_models),
                (prov.PROVIDER_KIOSAPI, ka.get_free_models),
                (prov.PROVIDER_MODELROUTER, mr.get_free_models),
                (prov.PROVIDER_AGNES, agnes.get_free_models),
                (prov.PROVIDER_CLEANAPIS, cp.get_free_models),
                (prov.PROVIDER_DREAMPROMPTING, dp.get_free_models),
                (prov.PROVIDER_TOPTOOLS, tt.get_free_models),
                (prov.PROVIDER_UNOROUTER, uo.get_free_models),
            ]:
                free = getter()
                for m in free[:2]:  # top 2 per provider
                    probe_models.append((p_name, m["id"]))
            # ZyloAI's five confirmed ids are the routing contract for them —
            # probe all of them, not just the top 2.
            for m in zy.get_free_models():
                probe_models.append((prov.PROVIDER_ZYLOAI, m["id"]))
        except Exception:
            pass
        if probe_models:
            asyncio.create_task(mc.warm_capability_cache(probe_models))
            logger.info("Capability probe warmup queued for %d models", len(probe_models))

        # Start the background capability refresher
        mc_task = asyncio.create_task(mc.capability_refresh_loop())
        logger.info(
            "Capability probe refresher started (every %ds)",
            app_settings.capability_probe_refresh_seconds,
        )

    yield

    cx_task.cancel()
    try:
        await cx_task
    except asyncio.CancelledError:
        pass

    sx_task.cancel()
    try:
        await sx_task
    except asyncio.CancelledError:
        pass

    rw_task.cancel()
    try:
        await rw_task
    except asyncio.CancelledError:
        pass

    tt_task.cancel()
    try:
        await tt_task
    except asyncio.CancelledError:
        pass

    dp_task.cancel()
    try:
        await dp_task
    except asyncio.CancelledError:
        pass

    cp_task.cancel()
    try:
        await cp_task
    except asyncio.CancelledError:
        pass

    mr_task.cancel()
    try:
        await mr_task
    except asyncio.CancelledError:
        pass

    tr_task.cancel()
    try:
        await tr_task
    except asyncio.CancelledError:
        pass

    fm_task.cancel()
    try:
        await fm_task
    except asyncio.CancelledError:
        pass

    mc_task.cancel()
    try:
        await mc_task
    except asyncio.CancelledError:
        pass

    refresh_task.cancel()
    try:
        await refresh_task
    except asyncio.CancelledError:
        pass

    uo_task.cancel()
    try:
        await uo_task
    except asyncio.CancelledError:
        pass

    xk_task.cancel()
    try:
        await xk_task
    except asyncio.CancelledError:
        pass

    ms_task.cancel()
    try:
        await ms_task
    except asyncio.CancelledError:
        pass

    llm7_task.cancel()
    try:
        await llm7_task
    except asyncio.CancelledError:
        pass

    await close_redis()
    logger.info("Shutdown complete")


app = FastAPI(
    title="FreeModels Bridge API",
    description="OpenAI-compatible API bridge to FreeModels",
    version="2.0.0",
    lifespan=lifespan,
    openapi_url=None,
    docs_url=None,
    redoc_url=None,
)

app.state.limiter = limiter
app.add_exception_handler(429, rate_limit_exceeded_handler)

# Starlette applies middleware in reverse registration order, so CORS must be
# registered LAST to become the OUTERMOST layer. Registered first (as before) it
# sat inside auth/rate-limit and 401/429 responses never carried CORS headers.
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(APIKeyMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=app_settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "x-api-key", "X-Session-Id", "X-Max-Iterations-Total"],
)


# ── Global OpenAI-format error handler ──────────────────────────────────────

@app.exception_handler(Exception)
async def openai_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions and return OpenAI error format."""
    logger.error("Unhandled exception: %s", type(exc).__name__, exc_info=True)

    # Never expose internal exception details to clients
    err_type = "server_error"
    code = None
    status = 500

    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": "An internal error occurred. Please try again later.",
                "type": err_type,
                "param": None,
                "code": code,
            }
        },
    )


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Return 404 in OpenAI error format for unknown routes."""
    path = request.url.path
    if path.startswith("/v1/"):
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "message": f"The path '{path}' does not exist",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "not_found",
                }
            },
        )
    return JSONResponse(status_code=404, content={"detail": "Not found"})


@app.exception_handler(405)
async def method_not_allowed_handler(request: Request, exc):
    """Return 405 in OpenAI error format."""
    return JSONResponse(
        status_code=405,
        content={
            "error": {
                "message": f"Method {request.method} not allowed for {request.url.path}",
                "type": "invalid_request_error",
                "param": None,
                "code": "method_not_allowed",
            }
        },
    )


# ── OpenAI-compatible routes ────────────────────────────────────────────────

app.include_router(models.router)
app.include_router(chat.router)

# Anthropic Messages API adapter (Claude Code, etc.)
app.include_router(anthropic.router)

# Auth routes (login, JWT)
app.include_router(auth.router)

# Dashboard API routes
app.include_router(dashboard.router)
app.include_router(combos.router)
app.include_router(api_keys.router)
app.include_router(users.router)
app.include_router(settings.router)
app.include_router(logs.router)
app.include_router(audit.router)
app.include_router(tools_routes.router)
app.include_router(sessions_routes.router)
app.include_router(evals_routes.router)


# ── Responses API stub (for AG2, CrewAI, Dify) ─────────────────────────────

@app.post("/v1/responses")
async def create_response(request: Request):
    """Responses API stub.

    Returns a response in the format expected by newer agentic frameworks
    that use the Responses API instead of Chat Completions.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    input_text = body.get("input", "")
    model = body.get("model", "claude-sonnet-5")

    from app.routes.chat import _openai_error
    from app.models import ChatMessage

    # Handle string input
    if isinstance(input_text, str):
        messages = [ChatMessage(role="user", content=input_text)]
    elif isinstance(input_text, list):
        messages = []
        for item in input_text:
            if isinstance(item, dict):
                role = item.get("role", "user")
                content = item.get("content", "")
                messages.append(ChatMessage(role=role, content=str(content)))
            else:
                messages.append(ChatMessage(role="user", content=str(item)))
    else:
        messages = [ChatMessage(role="user", content=str(input_text))]

    if not messages:
        return _openai_error("input is required", "invalid_request_error")

    # Non-streaming only for responses endpoint
    from app.providers import smart_collect

    full_text, _actual_model, error = await smart_collect(model, messages)
    if error:
        status = 429 if "rate" in (error.get("type") or "").lower() else 502
        return _openai_error(error.get("message", "Unknown error"), error.get("type", "server_error"), error.get("code"), status)

    # Tool-call responses are wrapped as JSON; surface only the text here.
    try:
        parsed_text = json.loads(full_text)
        if isinstance(parsed_text, dict) and "tool_calls" in parsed_text:
            full_text = parsed_text.get("content", "")
    except (json.JSONDecodeError, TypeError):
        pass

    resp_id = f"resp_{uuid.uuid4().hex[:24]}"
    input_tokens = sum(max(1, len(m.content or "") // 4) for m in messages)
    output_tokens = max(1, len(full_text) // 4)
    request.state.tokens_used = input_tokens + output_tokens
    return {
        "id": resp_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model,
        "output": [
            {
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex[:24]}",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": full_text,
                    }
                ],
            }
        ],
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


@app.get("/v1/responses/{response_id}")
async def get_response(response_id: str):
    """Responses API GET stub."""
    return JSONResponse(
        status_code=404,
        content={
            "error": {
                "message": f"Response '{response_id}' not found. Responses are not persisted.",
                "type": "invalid_request_error",
                "param": None,
                "code": "not_found",
            }
        },
    )


# ── Health & Root ───────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check endpoint for load balancers and monitoring."""
    redis_status = "unavailable"
    try:
        r = await get_redis()
        await r.ping()
        redis_status = "ok"
    except Exception:
        pass

    pg_status = "unavailable"
    try:
        from app.database import async_session
        async with async_session() as db:
            from sqlalchemy import text
            await db.execute(text("SELECT 1"))
        pg_status = "ok"
    except Exception:
        pass

    return {"status": "ok", "redis": redis_status, "postgres": pg_status}


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "FreeModels Bridge",
        "version": "2.0.0",
        "health": "/health",
        "api_base": "/v1",
    }
