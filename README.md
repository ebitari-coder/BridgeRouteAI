# FreeModels Bridge API

An OpenAI-compatible (and Anthropic-compatible) FastAPI gateway that proxies chat
completion requests to the **Kilo AI Gateway**, with **FreeModels** as a fallback
upstream. An advanced auto-router (`bridgeroute/auto`) spreads requests across every free
Kilo model, ranked by capability. It also adds native tool-calling passthrough,
JWT-authenticated admin/dashboard APIs, and request analytics backed by PostgreSQL.

## Features

- **OpenAI-Compatible**: `/v1/models`, `/v1/chat/completions`, `/v1/completions`.
- **Anthropic Adapter**: `/v1/messages` for Claude Code and other Anthropic-native tools.
- **Streaming Support**: Full SSE streaming for real-time token output.
- **Advanced Auto-Router**: The virtual `bridgeroute/auto` model discovers Kilo's live free
  catalog and routes across it, preferring capable, healthy models.
- **Kilo-first Failover**: Requests prefer Kilo and fall back to FreeModels when
  Kilo's free models are exhausted, with per-model circuit breaking.
- **ModelRouter Free Pool**: When configured, the gateway discovers ModelRouter's
  live catalog, filters it to free text-chat models, and includes those models
  in `/v1/models`, auto-routing, and failover.
- **Tiarina Cloud**: When configured, the gateway discovers Tiarina's live
  OpenAI-compatible model catalog and includes Tiarina models in direct routing,
  auto-routing, and failover.
- **Tool Calling**: OpenAI `tools` are forwarded, and the Anthropic adapter
  translates `tool_use` / `tool_result` end to end (for Claude Code etc.).
- **Agentic-safe routing**: Tool-bearing (agentic) requests are pinned to Kilo's
  tool-capable models and never fall back to FreeModels, which is text-only.
- **Combo routing**: group models under one name and pick a per-combo strategy —
  **Fallback** (ordered failover), **Round Robin** (rotate across requests), or
  **Fusion** (parallel panel + judge, N+1 calls) — or set the combo to
  **Agentic** routing for a multi-iteration tool loop.
- **Multi-tenant**: PostgreSQL-backed users, API keys, settings, request logs,
  and audit logs, with a JWT-authenticated admin API.
- **Dockerized**: One-command deployment with Docker Compose.

## Quick Start

### 1. Clone and configure

```bash
cp .env.example .env
# Edit .env and set ADMIN_API_KEY, JWT_SECRET, and SUPERADMIN_PASSWORD
```

### 2. Run with Docker Compose

```bash
docker compose up -d
```

The API is now running at `http://localhost:8001`.

### 3. Use the API

```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Authorization: Bearer YOUR_ADMIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-5",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

## Authentication

- **`/v1/*` endpoints** require an API key supplied via either
  `Authorization: Bearer <key>` or the `x-api-key` header. Keys are verified
  against the `api_keys` table (SHA-256 hashes); the `ADMIN_API_KEY` env var and
  valid JWTs are also accepted.
- **`/admin/*` endpoints** require an admin key (env var, JWT with the `admin`
  role, or a stored API key).
- **`/auth/*` endpoints** issue JWTs for the dashboard, using email + password.

## API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/health` | None | Health check with Redis + Postgres status |
| `GET` | `/` | None | Service metadata |
| `GET` | `/v1/models` | API Key | List available models with health |
| `GET` | `/v1/models/{id}` | API Key | Get a single model |
| `POST` | `/v1/chat/completions` | API Key | Chat completions (streaming supported) |
| `POST` | `/v1/completions` | API Key | Legacy text completions |
| `POST` | `/v1/messages` | API Key | Anthropic Messages API |
| `POST` | `/v1/embeddings` | API Key | Embeddings (deterministic stub) |
| `GET` | `/v1/provider-health` | API Key | Upstream + per-model health |
| `POST` | `/v1/responses` | API Key | Responses API shim (AG2, CrewAI, Dify) |
| `POST` | `/auth/login` | None | Email + password login (returns JWT) |
| `GET` | `/auth/me` | JWT | Current user |
| `POST` | `/auth/change-password` | JWT | Change password |
| `GET` | `/admin/dashboard/stats` | Admin | Overview statistics |
| `GET` | `/admin/dashboard/chart` | Admin | Requests-per-hour chart data |
| `GET` | `/admin/dashboard/providers` | Admin | Provider + model health |
| `GET/POST` | `/admin/api-keys` | Admin | List / create API keys |
| `DELETE` | `/admin/api-keys/{id}` | Admin | Revoke an API key |
| `GET/POST` | `/admin/users` | Admin | List / create users |
| `GET/PATCH/DELETE` | `/admin/users/{id}` | Admin | Manage a user |
| `GET/PUT` | `/admin/settings` | Admin | System settings |
| `GET` | `/admin/logs` | Admin | Request logs (paginated) |
| `GET` | `/admin/audit` | Admin | Audit logs (paginated) |

## Routing & Failover

Requests prefer the **Kilo** gateway and fall back to **FreeModels** when Kilo is
exhausted. On a quota/rate-limit error the gateway walks an ordered attempt plan
until one succeeds, applying exponential backoff (circuit breaking) to models that
keep failing. A single request is capped at 8 upstream attempts.

### Combo routing

Agentic is **not** a model id. Every combo declares how requests bound to it run
(`routing_mode`), and a group combo additionally declares `routing_strategy`:

| `routing_mode` | What happens |
|----------------|--------------|
| `agentic` | Multi-iteration tool loop (compression, budgets, reflection, resumable sessions) |
| `group` | Single-shot, using the combo's strategy below |

| `routing_strategy` (group only) | Behavior |
|---------------------------------|----------|
| `fallback` | Tries models in order; next on failure |
| `round_robin` | Rotates the starting model across requests to spread load |
| `fusion` | Queries all models in parallel, then a judge you pick synthesizes one answer — **N+1 calls per request**; the judge must be a reachable `freemodels` model |

A request binds to a combo in this order: **combo-scoped API key → `model`
equals the combo's name → `model` is listed in the combo's `models[]`**.
The combo name is itself a routable id and appears in `GET /v1/models` with
`owned_by: "combo"`.

```bash
# group combo named "my-panel" with routing_strategy=fallback
curl http://localhost:8001/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "my-panel", "messages": [{"role": "user", "content": "Hi"}]}'
```

`model: "agentic"` is no longer accepted and returns `400` with
`code: "agentic_model_removed"`.

### Auto-router (`bridgeroute/auto`)

`bridgeroute/auto` is a virtual model that routes across **every free Kilo model**:

- The free catalog is discovered live from `GET https://api.kilo.ai/api/gateway/models`
  and filtered by the `isFree` flag / `:free` suffix. Only text chat models are
  kept (music/image/video generation models are excluded), with a 5-minute cache.
- A background task re-fetches the catalog every `KILO_CATALOG_REFRESH_SECONDS`
  (default 300s), so new free models appear without a restart.
- Candidates are ranked by capability — context window, tool support, benchmark
  scores, and recency — and models with an open circuit breaker are skipped.
- When `tools` are supplied, models that do not advertise tool support are excluded.
- **Tool-bearing requests stay on Kilo entirely** — FreeModels has no tool-calling
  support, so it is never used for them (this is what lets Claude Code, opencode,
  and other agents work).
- For non-tool requests, if every Kilo model fails the router falls back to the
  FreeModels catalog.

Kilo free models are also callable directly by id (e.g. `poolside/laguna-s-2.1:free`).
Any other model name is forwarded to FreeModels verbatim. The fallback FreeModels
list lives in `FREEMODELS_MODELS` (`app/providers.py`).

### Tool calling

FreeModels models are **text-only** — they cannot emit tool calls. To keep agentic
clients working:

- Any request that includes `tools` (OpenAI `/v1/chat/completions`) is routed only to
  Kilo models that advertise tool support; FreeModels is excluded. Requesting a
  FreeModels model with `tools` routes the call to Kilo's tool-capable models instead.
- The Anthropic `/v1/messages` adapter forwards `tools`, converts `tool_use` and
  `tool_result` content blocks to/from OpenAI tool calls, and emits Anthropic
  `tool_use` stream events with `stop_reason: "tool_use"`.

## Running Without Docker

```bash
pip install -r requirements.txt
# Make sure Redis and PostgreSQL are running, then:
export DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/bridgeroute"
export REDIS_URL="redis://localhost:6379/0"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Alembic migrations and superadmin seeding run automatically at startup.

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ADMIN_API_KEY` | **Yes** | — | API key for admin endpoints |
| `JWT_SECRET` | **Yes** | — | JWT signing secret (`openssl rand -hex 32`) |
| `SUPERADMIN_PASSWORD` | **Yes** | — | Superadmin password (min 12 chars, mixed case + digits + symbol) |
| `DATABASE_URL` | **Yes** | — | PostgreSQL connection string |
| `REDIS_URL` | **Yes** | — | Redis connection string |
| `SUPERADMIN_EMAIL` | No | `admin@bridgeroute.local` | Superadmin login email |
| `CORS_ORIGINS` | No | `http://localhost:3000` | Comma-separated allowed origins |
| `KILO_BASE_URL` | No | `https://api.kilo.ai/api/gateway` | Kilo AI Gateway base URL |
| `KILO_API_KEY` | No | — | Kilo API key (optional — free models work anonymously) |
| `KILO_CATALOG_REFRESH_SECONDS` | No | `300` | How often the background task refreshes the Kilo model catalog |
| `FREE_MODELS_URL` | No | `https://freemodels-chat.freemodels.workers.dev` | FreeModels fallback endpoint |
| `MODELROUTER_API_KEY` | No | — | ModelRouter key for its live free-model catalog and chat |
| `MODELROUTER_BASE_URL` | No | `https://modelrouter.id/v1` | ModelRouter OpenAI-compatible base URL |
| `MODELROUTER_CATALOG_REFRESH_SECONDS` | No | `300` | How often the ModelRouter free-model catalog refreshes |
| `TIARINA_API_KEY` | No | — | Tiarina Cloud key for its model catalog and chat |
| `TIARINA_BASE_URL` | No | `https://ai.tiarina.cloud/v1` | Tiarina OpenAI-compatible base URL |
| `TIARINA_CATALOG_REFRESH_SECONDS` | No | `300` | How often the Tiarina model catalog refreshes |
| `HOST` | No | `0.0.0.0` | Server bind address |
| `PORT` | No | `8000` | Server port |

## Using with the OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="YOUR_ADMIN_API_KEY",
)

response = client.chat.completions.create(
    model="claude-sonnet-5",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```
