# Agentic AI Engine Feature Completion Spec

## Problem

The BridgeRouteAI Agentic AI Engine has foundational scaffolding but several feature categories listed in AGENTIC_ENGINE.md are incomplete or not wired through the full stack. Users cannot rely on parallel/best_of strategies, real tool-calling, SSE streaming of iterations, budget intelligence, agentic analytics, or a backoffice that configures agentic combos. The goal is to close every gap in the user-provided missing-feature list so the agentic system is production-complete.

## Users

- **API Consumers** — clients sending `{"model": "agentic", ...}` requests, expecting streaming iterations, tool use, and deterministic budget behavior.
- **Administrators** — operators using the backoffice to create/edit agentic combos, monitor budgets, and review performance analytics.
- **Platform Owners** — stakeholders needing cost visibility, reliability circuit breakers, and usage analytics.

## Goals

1. Fully implement and integrate loop strategies (`sequential`, `parallel`, `best_of`) end-to-end from `/v1/chat/completions` through the agentic loop.
2. Connect `_call_tool()` to a real, schema-validated tool registry with discovery endpoints and safe execution.
3. Expose SSE streaming of intermediate agentic iterations, not just the final content.
4. Deliver budget intelligence: threshold alerts (80%/95%), historical monthly usage, and auto-replenishment prompts.
5. Implement reliability primitives: per-model circuit breaker in combos, retry with next model on failure, pre-execution cost estimation endpoint.
6. Provide analytics endpoints and data structures: agentic usage metrics, per-combo performance stats, and model success-rate ranking.
7. Wire agentic configuration and budget widgets into the backoffice combos page including combo edit forms with agentic fields.
8. Implement advanced features: research/code-review/debug strategy presets, conversation history persistence for resumable loops, per-combo API key rate limiting, and tool argument validation/sanitization.

## Non-Goals

- Rewriting the upstream provider routing (Kilo, Cloudflare, etc.) beyond what is required to route AGENTIC_MODEL requests through the agentic loop correctly.
- Replacing existing failover streaming in `smart_stream` for non-agentic models.
- Implementing a full billing/payment system for auto-replenishment; only the prompt/trigger logic is in scope.
- New UI frameworks or backoffice rewrites; extend the existing backoffice in-place.
- Production-grade alert delivery (email, SMS, webhook calls); persistence of alert records and HTTP-exposed status is sufficient.

## Functional Requirements

### FR-1 Loop Strategies
- The agentic engine must support three strategies selectable per combo: `sequential`, `parallel`, `best_of`.
- For `sequential`: each iteration round-robin through the combo models, exactly one call per iteration.
- For `parallel`: all healthy models in the combo are called concurrently per iteration; the highest-scoring result drives the next iteration and is recorded as the chosen output.
- For `best_of`: top-2 healthy models are called per iteration; the result with the longest substantive content wins.
- All three strategies must respect per-model circuit breakers, compression, tool execution, and budget enforcement.
- Strategy selection propagates from the combo row through `providers.py` into the actual loop entry point for both streaming and non-streaming chat completions.

### FR-2 Tool Integration
- `AgenticLoop._call_tool()` must dispatch through `app.tools_registry.tools_registry`, validate args against the registered JSON schema, sanitize outputs, and return structured results.
- A GET endpoint must expose the list of registered tools with schemas for agentic use.
- Tool execution must log successes/failures into the step history and, on failure, bubble an error payload back to the model as a tool result rather than aborting the loop.
- The combo record stores the list of enabled tool names; the agentic loop only invokes tools on that allow-list.

### FR-3 Streaming
- `POST /v1/chat/completions` with `model=agentic` and `stream=true` must yield Server-Sent Events for each intermediate agentic iteration, not only the final content.
- Stream event types include: `iteration`, `tool_execution`, `compression`, `budget_alert`, `complete`, `error`.
- Every iteration event contains iteration index, provider/model, tokens, stop_reason, and content delta.
- The existing `run_agentic_stream` generator is used (or extended) and wired through `smart_stream`/`providers.py`.
- Non-streaming chat completions for agentic continue to return the final `ChatCompletionResponse`.

### FR-4 Budget Intelligence
- Budget threshold alerts trigger at 80% (warning) and 95% (critical) percent of monthly budget; events are persisted to `budget_alerts` and surfaced via API.
- A combo budget status response includes percentage, tokens used/remaining, alerts fired this month, and auto-replenishment prompt when applicable.
- Historical budget usage endpoint returns per-month records for up to 12 months for a given combo.
- Auto-replenishment / upgrade prompts are returned alongside budget status when usage rate indicates imminent exhaustion.
- Budget enforcement records tokens used per iteration and halts the loop with `stop_reason=budget_exceeded` when exceeded.

### FR-5 Reliability
- Per-model circuit breaker inside `AgenticLoop` tracks consecutive failures and skips open-circuit models for the duration of the recovery window; breaker state is recorded per step.
- On upstream call failure within the combo loop, the next healthy model in the combo is retried before moving to the next iteration.
- A pre-execution cost estimation endpoint estimates total tokens and USD cost based on combo configuration, max_iterations, and average tokens-per-iteration.

### FR-6 Analytics
- A dashboard endpoint aggregates agentic-specific metrics: total agentic requests, total iterations, avg iterations/request, total tokens, total estimated cost, success/failure rates.
- Per-combo performance endpoint returns: iterations per request distribution, total tokens per period, cost, stop_reason breakdown, tool-call counts.
- Model ranking endpoint ranks each model in a combo by success rate (stop_reason=stop vs error), avg tokens, and cost efficiency.

### FR-7 Backoffice
- The combos list page displays agentic fields (enabled, max_iterations, loop_strategy, compression_target, budget) for every combo.
- The combo create/edit forms include agentic configuration inputs and the monthly token budget input.
- A budget monitoring widget on the combo detail page shows current usage, percentage, alerts history, and upgrade/replenishment prompt.
- A test-agentic button invokes the dry-run endpoint and renders the iteration/history preview.
- Monthly budget field is included in ComboCreate/ComboUpdate API and backoffice forms.

### FR-8 Advanced Features
- Strategy presets (research, code-review, debug) — each preset defines a named set of defaults (max_iterations, loop_strategy, compression_target, recommended tool list) that users can apply when creating a combo.
- Conversation/session history persistence: a `agentic_sessions` table (or equivalent store) records conversation messages, iteration history, and metadata keyed by a session ID, enabling resume of an interrupted loop.
- Per-combo API key rate limiting: API keys can be scoped to a combo with a configured requests-per-minute budget; middleware enforces the limit.
- Tool argument validation: strict JSON-schema validation (required fields, types, bounds) plus sanitization (string length caps, field allow-listing) before tool invocation; validation errors are returned as structured tool results.

## Non-Functional Requirements

- **NFR-1 Determinism**: The existing deterministic round-robin and compression guarantees in AGENTIC_ENGINE.md must remain intact for the sequential strategy; parallel/best_of document their non-determinism.
- **NFR-2 Backward Compatibility**: Existing non-agentic routes and models continue to work identically; no changes to OpenAI-compatible response shapes outside the agentic stream extensions.
- **NFR-3 Safety**: Tool argument sanitization and path/command allow-lists remain enforced; no new unsafe default permissions.
- **NFR-4 Observability**: New endpoints and loop branches include structured logger calls with combo_id, strategy, iteration, and model fields.
- **NFR-5 Performance**: Parallel fan-out per iteration is bounded by combo size (expected ≤5); no unbounded concurrency.
- **NFR-6 Database Safety**: All new schema additions are introduced via Alembic migrations; existing tables are not destructively altered.

## Constraints

- Backoffice project resides at `/home/brave-dimaro/source/repos/BridgeRouteAI-backoffice`; if the directory is empty or sparse, add only what is required without inventing unrelated files.
- Python backend uses FastAPI, SQLAlchemy (async), Alembic, Pydantic v2. Follow existing conventions in `app/routes/combos.py`, `app/crud.py`, `app/models.py`.
- Do not introduce new external dependencies unless absolutely necessary; prefer stdlib + libraries already in `requirements.txt`.
- All public endpoints under `/admin/*` require the existing admin-key header middleware.
- All chat endpoints keep the OpenAI-compatible envelope.

## Dependencies

- Alembic migrations for any new columns or tables.
- The existing `DBRouterCombo`, `DBTokenBudget`, `DBBudgetAlert` models (and any new ones) in `app/db_models.py`.
- Existing module layout: `app/agentic_engine.py`, `app/token_budget_manager.py`, `app/tools_registry.py`, `app/providers.py`, `app/routes/combos.py`, `app/routes/chat.py`, `app/routes/dashboard.py`, `app/crud.py`, `app/models.py`.

## Assumptions

- The backoffice is a Next.js/React TypeScript project; components extend existing pages rather than being created from scratch.
- USD cost estimates use the documented $0.002/1K tokens rough rate; per-model fine-grained pricing tables are out of scope.
- `month_year` column format is `YYYY-MM` as already implemented.
- Auto-replenishment does not actually charge a card; it returns a suggested action and prompt only.
- Conversation resume endpoints accept a `session_id` query/header parameter; creating a session is opt-in (by requesting one).

## Open Questions

- (OQ-1) Should budget alerts also create audit log entries, or is the dedicated budget alert table sufficient? → Accept dedicated table + dedicated alerts endpoint.
- (OQ-2) Is there a UI target framework for the backoffice already established? → Extend existing TSX pages; assume Next.js App Router based on modern project layout conventions.
- (OQ-3) Should per-combo rate limiting use Redis or only in-process counters? → Prefer Redis-backed rate limiting via existing `redis_client.py` when available; fall back to in-process when Redis is unreachable.

## Acceptance Criteria

### AC-1 (Loop Strategies End-to-End) — rule
An admin creates an agentic combo with `loop_strategy=parallel`, `parallel` with 3 models, and `best_of` with 2 models respectively, activates each, and sends a non-streaming `POST /v1/chat/completions` with `model=agentic`. The `AgenticResult.history` shows: for sequential, exactly one provider/model per iteration; for parallel, all healthy models attempted and a single winner chosen; for best_of, exactly 2 models attempted per iteration and longest-content result chosen. All three strategies respect the combo's circuit breaker.

### AC-2 (Tool Integration — Real Dispatch) — rule
Register a custom tool `echo_test` via `@register_tool` with a schema requiring a `message` string. Execute an agentic loop whose model issues a tool call to `echo_test` with correct args and with an invalid/missing `message` field. The valid call returns the echo result appended as a `role=tool` message and the loop continues; the invalid call returns a validation error as the tool content (not a loop abort) with fields `{error, tool, params}`. A GET endpoint returns `echo_test` in its registered tool list with schema.

### AC-3 (SSE Streaming of Iterations) — rule
Send `POST /v1/chat/completions` with `model=agentic`, `stream=true`, and a combo configured for 3 iterations. Collect SSE events; assert event types include at least one `iteration` event per round, event fields `iteration`, `model`, `provider`, `content`, `input_tokens`, `output_tokens`, followed by exactly one `complete` event containing `stop_reason`, `total_tokens`, `iterations`. Chunks are media-type `text/event-stream` with correct `data:` framing and trailing `[DONE]`.

### AC-4 (Budget Intelligence — Thresholds + History) — rule
For a combo with `monthly_token_budget=100000`, simulate usage of 80,000 tokens → a warning alert exists in `budget_alerts` for this combo and is returned by `GET /admin/combos/{id}/budget` with `level=warning` at ~80%; simulate 95,000 tokens → a critical alert appears. Calling `GET /admin/combos/{id}/budget/history?months=3` returns the last 3 months ordered descending by `month_year` with `{month_year, tokens_used, tokens_budget, tokens_remaining, percentage}`. When projected daily run-rate exceeds remaining budget, the status response includes a non-empty `replenishment_prompt` object.

### AC-5 (Reliability — Circuit Breaker + Retry + Cost Estimate) — rule
Configure a combo with 2 models; repeatedly fail the first model 3 times in a single loop. The circuit breaker for model 1 transitions to open; subsequent iterations skip it and attempt model 2. The cost estimate endpoint `GET /admin/combos/{id}/estimate` returns `{estimated_total_tokens, estimated_cost_usd, iterations, models_per_iteration, tokens_per_iteration}` with non-negative values and USD rounded to 4 decimals.

### AC-6 (Analytics Endpoints) — rule
After exercising the agentic loop with several combos, the dashboard agentic summary endpoint returns aggregates (`total_requests`, `total_iterations`, `avg_iterations_per_request`, `total_tokens`, `estimated_cost_usd`, `success_rate`). Per-combo endpoint returns stop_reason counts and tool_call counts. Model ranking endpoint ranks combo models by `success_rate` descending and includes numeric scores for `success_rate`, `avg_tokens_per_call`, `cost_efficiency`.

### AC-7 (Backoffice Combo Page) — rubric
Dimension: UI completeness and correctness for agentic combo management. Scale 0-2: `2` = combos list displays every agentic column, create/edit forms include all agentic fields + monthly budget with validation, combo detail includes budget widget and a working test-agentic action that renders iteration history; `1` = forms/list render but a subset of fields/actions are missing or not wired to API; `0` = no agentic UI wiring visible. Pass threshold ≥2.

### AC-8 (Advanced Features) — rule
(8a) Strategy presets endpoint returns `research`, `code-review`, `debug` with concrete preset values; applying a preset via create-combo populates the combo defaults correctly. (8b) Create an agentic session with `session_id`, run 2 iterations, interrupt, then resume with the same `session_id`; resumed loop restarts from the persisted message history and iteration index. (8c) Create an API key scoped to a combo with 5 req/min limit; send 6 requests in rapid succession; the 6th receives HTTP 429 with a `rate_limit_exceeded` error body. (8d) Tool argument validation rejects payloads missing required fields, wrong types, or oversized strings, and returns a structured error; successful invocations have string args truncated to schema maxima before dispatch.

### AC-9 (Database Safety) — rule
All new schema additions are provided as Alembic migration scripts under `alembic/versions/` with unique identifiers, and running `alembic upgrade head` in a clean database creates the new tables/columns without error and without dropping existing `router_combos`, `token_budgets`, `budget_alerts`, `request_logs` data.
