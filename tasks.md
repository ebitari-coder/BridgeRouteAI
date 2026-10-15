# Agentic AI Engine Implementation Tasks

## Dependency Graph (summary)

1. Schema/Migrations (T1) → Core module fixes (T2, T3, T4)
2. Core modules → CRUD + models (T5, T6) → API routes (T7, T8, T9, T10)
3. Provider/chat integration (T11) depends on routes and core
4. Backoffice wiring (T12, T13, T14) depends on API routes
5. Advanced features (T15, T16, T17) layered on top of routes and middleware

---

## Task 1: Alembic migrations for new schema additions

**Priority**: high
**Status**: complete
**Parent ACs**: AC-9, FR-4, FR-6, FR-8

**Scope**
- Create migration for new tables/columns:
  - `agentic_sessions` table: id (PK), combo_id, session_id (unique), messages (JSONB), iteration_index (int), step_history (JSONB), created_at, updated_at, closed_at
  - `api_keys` extension: add `combo_id` (nullable FK to router_combos), `rate_limit_per_minute` (int default 60)
  - `router_combos` extension: add `tools_allowed` JSONB column, `strategy_preset` String column
  - `agentic_analytics` table: id (PK), combo_id, request_id, month_year, requests_count, total_iterations, total_tokens, estimated_cost_usd, stop_reason_counts JSONB, tool_call_counts JSONB, model_stats JSONB
  - `budget_alerts` table consolidation: ensure single definition, fix duplicate class in db_models.py; keep the version with `month_year`, `percentage`, `tokens_used`, `tokens_budget`, `triggered_at`, `resolved_at`
  - Ensure `monthly_token_budget` column is consistently included in ComboUpdate (crud.py) and list/get endpoints

**Test Requirements (TR)**
- TR-1.1 (rule): Running `alembic upgrade head` from empty DB succeeds; tables exist verified via inspect.
- TR-1.2 (rule): Running `alembic downgrade -1` rolls back cleanly without dropping unrelated columns.
- TR-1.3 (rule): `DBRouterCombo.tools_allowed` and `DBRouterCombo.strategy_preset` fields are deserializable from JSONB (None defaults).

---

## Task 2: Fix, consolidate, and enhance agentic_engine.py

**Priority**: high
**Status**: complete
**Parent ACs**: AC-1, AC-2, AC-5, FR-1, FR-2, FR-5

**Scope**
- Remove duplicate `get_consecutive_failures` function floating outside `CircuitBreaker`.
- Fix `DBBudgetAlert` import reference inside `token_budget_manager.py` (DBTokenBudget undefined in `get_historical_usage`).
- `_run_parallel` and `_run_best_of`: ensure circuit breaker per-model skips and `next-model-on-failure` retry inside the iteration: when `_call_upstream` raises, iterate to next model in combo before giving up on iteration.
- Score function for parallel: consider more heuristics (tool-call validity, fewer tokens for same content length).
- `_call_upstream` should be strategy-agnostic and actually honor the provider dispatch for the provider it receives (not hard-code kilo only), using `_stream_for` logic from providers.py or a `_collect_for` helper.
- Add `cost_estimate` helper that accepts combo config and returns estimate before actual run.
- Ensure `run_agentic_stream` produces every event type specified in FR-3: iteration, tool_execution, compression, budget_alert, complete, error; wrap parallel/best_of streaming variants too (today stream is only sequential).
- Add tool allow-list filtering inside `_execute_tools`: only run tools present in `combo.tools_allowed`; return error payload for disallowed tools.

**Test Requirements (TR)**
- TR-2.1 (rule): `python -c "from app.agentic_engine import AgenticLoop"` imports without error.
- TR-2.2 (rule): Parallel strategy with a 2-model combo where model-0 errors 3 times in a mocked `_call_upstream` → circuit breaker opens for model-0, subsequent iterations use model-1; `_model_failures["model-0"]` ≥3.
- TR-2.3 (rule): Best_of strategy returns the longest content result for mocked results of differing lengths; history length equals iteration count.
- TR-2.4 (rubric): _score_result quality. 0 = always returns 0; 1 = simple length-only scoring; 2 = balances content length, tokens, stop_reason, tool_calls. Pass threshold ≥2.
- TR-2.5 (rule): `_execute_tools` with a tool not on `tools_allowed` returns structured error content, loop continues without raising.

---

## Task 3: Enhance tools_registry.py schema discovery & built-ins

**Priority**: high
**Status**: complete
**Parent ACs**: AC-2, FR-2, FR-8d

**Scope**
- Add `sanitize_tool_args(schema, params)` helper that:
  - Truncates strings to `maxLength` when declared in schema properties.
  - Strips keys not declared in `properties` (allow-listing).
  - Coerces numeric types to the declared type.
- Add `register_tools_from_module(module_name)` bulk discover function.
- Improve `get_tools_for_combo(combo_id)`: read `router_combos.tools_allowed` from DB; filter `tool_schemas` to only allowed set; return OpenAI-compatible `{type:"function", function:{name, description, parameters}}` list.
- Add safe `http_request` built-in with allow-listed URL schemes (https only), timeout, and response size caps; schema includes `url`, `method`, `headers`, `body` (optional).
- Add `tool_usage_metrics` dict tracking `{tool_name: {calls, success, error}}`; expose via getter.

**Test Requirements (TR)**
- TR-3.1 (rule): `sanitize_tool_args` for schema `{properties: {msg: {type: string, maxLength: 5}}}` + input `{msg: "123456789", extra: "x"}` returns `{msg: "12345"}`.
- TR-3.2 (rule): `get_tools_for_combo` for a combo with `tools_allowed=["calculate"]` returns only calculate in its list.
- TR-3.3 (rule): `http_request` tool with `url="http://insecure"` returns an error; with `url="https://example.invalid"` returns structured error, no exception raised.

---

## Task 4: Budget intelligence completion in token_budget_manager.py

**Priority**: high
**Status**: complete
**Parent ACs**: AC-4, FR-4

**Scope**
- Fix import of `DBTokenBudget` in `TokenBudgetManager.get_historical_usage` and `DBBudgetAlert` in `get_alerts` (use correct name and import).
- In `BudgetEnforcer._check_thresholds` → actually persist the alert via `alert_manager.record_alert` to DB, not just in-memory.
- `BudgetAlertManager.should_auto_replenish`: compute using last N days of usage from historical records to project run-rate; return True when projected to exceed in <7 days.
- `get_replenishment_prompt`: consume actual historical usage data, not stub calculation.
- Add `get_all_budgets_for_combo(combo_id)` helper used by history endpoint.
- Add `resolve_alert(alert_id)` endpoint helper to mark alerts resolved_at.

**Test Requirements (TR)**
- TR-4.1 (rule): Import `app.token_budget_manager` without `NameError: DBTokenBudget`.
- TR-4.2 (rule): `BudgetEnforcer` with `monthly_budget=100000`, after `record(80001)` → triggers a warning alert in DB that is visible via `get_alerts`.
- TR-4.3 (rule): `get_historical_usage(months=3)` returns records ordered by `month_year` descending when 3 rows exist in DB.
- TR-4.4 (rule): For a combo on track to exhaust within 5 days, `should_auto_replenish` returns True and `get_replenishment_prompt` contains a non-empty `recommended_amount`.

---

## Task 5: Extend Pydantic models in app/models.py

**Priority**: high
**Status**: complete
**Parent ACs**: AC-6, AC-7, AC-8, FR-6, FR-7, FR-8

**Scope**
- Add pydantic models:
  - `AgenticStrategyPreset`: `name: str`, `description: str`, `max_iterations: int`, `loop_strategy: str`, `compression_target_tokens: int`, `recommended_tools: list[str]`
  - `ComboPresetApply`: `preset_name: str`
  - `AgenticStats`: `total_requests: int`, `total_iterations: int`, `avg_iterations_per_request: float`, `total_tokens: int`, `estimated_cost_usd: float`, `success_rate: float`
  - `ComboPerformance`: `combo_id: str`, `period: str`, `requests_count: int`, `total_iterations: int`, `avg_iterations: float`, `total_tokens: int`, `estimated_cost_usd: float`, `stop_reason_counts: dict`, `tool_call_counts: dict`
  - `ModelRankingEntry`: `provider: str`, `model_id: str`, `success_rate: float`, `avg_tokens_per_call: float`, `cost_efficiency: float`, `total_calls: int`
  - `AgenticSessionCreate` / `AgenticSessionResponse` with session_id, messages, iteration_index
  - `BudgetAlertResponse`: id, combo_id, month_year, level, percentage, tokens_used, tokens_budget, triggered_at, resolved_at
  - `BudgetStatusResponse`: add `alerts: list[BudgetAlertResponse]`, `historical_usage: list[dict]`, `replenishment_prompt: dict | None`
  - `CostEstimateResponse`: `estimated_total_tokens: int`, `estimated_cost_usd: float`, `iterations: int`, `models_per_iteration: int`, `tokens_per_iteration: int`
  - Extend `ApiKeyCreate` with optional `combo_id` and `rate_limit_per_minute`; extend `ApiKey` response with them.
  - Extend `ComboCreate` / `ComboUpdate` with `tools_allowed: list[str] | None` and `strategy_preset: str | None`
  - Extend `ComboResponse` with the same

**Test Requirements (TR)**
- TR-5.1 (rule): All new pydantic models instantiate with valid data and reject missing required fields via `model_validate`.
- TR-5.2 (rule): `ComboResponse.model_dump()` round-trips `tools_allowed` list and `strategy_preset` string.

---

## Task 6: CRUD extensions in app/crud.py

**Priority**: high
**Status**: complete
**Parent ACs**: AC-4, AC-6, AC-8, FR-4, FR-6, FR-8

**Scope**
- Add:
  - `create_agentic_session(db, session_id, combo_id, messages, iteration_index)`: upsert by session_id.
  - `get_agentic_session(db, session_id)` → return session or None.
  - `update_agentic_session(db, session_id, messages, iteration_index, step_history)`: atomically update.
  - `close_agentic_session(db, session_id)`.
  - `resolve_budget_alert(db, alert_id)` → set resolved_at = now.
  - `list_budget_alerts(db, combo_id, level, limit, offset)` → pageable list.
  - `record_agentic_analytics(db, combo_id, request_id, month_year, record)` → upsert aggregate row for combo+month.
  - `get_agentic_overall_stats(db, since_months: int = 1)` → aggregate to `AgenticStats`.
  - `get_combo_performance(db, combo_id, months: int = 1)` → list of `ComboPerformance` per month.
  - `get_model_rankings(db, combo_id, months: int = 1)` → list of `ModelRankingEntry` sorted by success_rate desc.
  - Ensure `update_router_combo` passes through `monthly_token_budget`, `tools_allowed`, `strategy_preset`.
  - Ensure `get_router_combos` / `get_router_combo` include `monthly_token_budget`, `tools_allowed`, `strategy_preset` (default empty).

**Test Requirements (TR)**
- TR-6.1 (rule): create → get → update → close session round-trip; `iteration_index` returned equals updated value.
- TR-6.2 (rule): `record_agentic_analytics` called twice for same combo+month sums counters rather than duplicating row.
- TR-6.3 (rule): `update_router_combo` with `monthly_token_budget=1_000_000` → subsequent `get_router_combo` returns that value.
- TR-6.4 (rule): `get_model_rankings` for mocked 2-model combo stats returns list ordered by success_rate desc with floats in [0,1].

---

## Task 7: Extend /admin/combos routes with new endpoints

**Priority**: high
**Status**: complete
**Parent ACs**: AC-1, AC-2, AC-4, AC-5, AC-7, FR-1, FR-2, FR-4, FR-5, FR-7, FR-8

**Scope**
Add to `app/routes/combos.py`:
- `GET /admin/combos/{id}` (ensure all new fields present: budget, tools_allowed, strategy_preset)
- `GET /admin/combos/{id}/budget` → return `BudgetStatusResponse` with alerts + history + replenishment prompt.
- `GET /admin/combos/{id}/budget/alerts` → paged budget alerts list.
- `POST /admin/combos/{id}/budget/alerts/{alert_id}/resolve`.
- `GET /admin/combos/{id}/budget/history?months=12` → historical usage list.
- `GET /admin/combos/{id}/estimate` → cost estimate `CostEstimateResponse`.
- `GET /admin/combos/{id}/tools` → return enabled tools list (OpenAI-compatible) for the combo.
- `PUT /admin/combos/{id}/tools` → set `tools_allowed` for combo.
- `GET /admin/combos/strategy-presets` → return list of `AgenticStrategyPreset` (research, code-review, debug + defaults).
- `POST /admin/combos/apply-preset` → accept `ComboPresetApply` (or via query param on create) and apply defaults.
- Ensure `test-agentic` passes `loop_strategy`, `combo_id`, `monthly_token_budget` into loop constructor.
- Ensure combo create/update accept `tools_allowed`, `strategy_preset`, and persist them.

**Test Requirements (TR)**
- TR-7.1 (rule): `GET /admin/combos/strategy-presets` returns `research`, `code-review`, `debug` each with distinct max_iterations values.
- TR-7.2 (rule): `GET /admin/combos/{id}/budget` returns `percentage`, `tokens_used`, tokens_budget`, `alerts`, `historical_usage`, `replenishment_prompt` keys.
- TR-7.3 (rule): `GET /admin/combos/{id}/estimate` response keys match `CostEstimateResponse` and all numeric fields ≥ 0.
- TR-7.4 (rule): `PUT /admin/combos/{id}/tools` with `["calculate"]` body → subsequent `GET /tools` returns exactly 1 entry with `name=calculate`.

---

## Task 8: New analytics routes under /admin/dashboard

**Priority**: medium
**Status**: complete
**Parent ACs**: AC-6, FR-6, FR-7

**Scope**
Add to `app/routes/dashboard.py` (or new `app/routes/analytics.py` imported by main):
- `GET /admin/dashboard/agentic` → `AgenticStats` aggregate.
- `GET /admin/dashboard/agentic/combos` → list of `ComboPerformance` across all combos (default last month).
- `GET /admin/dashboard/agentic/combos/{id}` → `ComboPerformance` for a single combo, ?months= query.
- `GET /admin/dashboard/agentic/combos/{id}/rankings` → model ranking list.
- Each endpoint accepts `months=N` query param (default 1).
- Analytics record calls: ensure `record_agentic_analytics` is invoked after every agentic run (both stream and non-stream), so stats are accurate.

**Test Requirements (TR)**
- TR-8.1 (rule): `GET /admin/dashboard/agentic` after recording one agentic run returns `total_requests >= 1`.
- TR-8.2 (rule): `GET /admin/dashboard/agentic/combos/{id}/rankings` returns non-empty list sorted by `success_rate` desc.
- TR-8.3 (rule): Invalid months param (0 or negative) is rejected with 422 by Pydantic.

---

## Task 9: Tools registry route + OpenAI-compatible tools endpoint

**Priority**: medium
**Status**: complete
**Parent ACs**: AC-2, FR-2

**Scope**
Create `app/routes/tools.py`:
- `GET /admin/tools` → list all registered tools with schemas (for backoffice browsing).
- `GET /admin/tools/{name}` → single tool schema.
- `POST /admin/tools/exec` (admin-only) → dry-run execute a tool with given params and return result, enforcing safety caps.
- Register route in `app/main.py` include_router.

**Test Requirements (TR)**
- TR-9.1 (rule): `GET /admin/tools` list contains at least `web_search`, `calculate`, `read_file`, `write_file`, `bash_command`, `sleep`, `http_request`.
- TR-9.2 (rule): `POST /admin/tools/exec` with `{name: "calculate", params: {expression: "2+3"}}` returns `{result: 5}`.

---

## Task 10: Agentic session routes

**Priority**: medium
**Status**: complete
**Parent ACs**: AC-8b, FR-8b

**Scope**
Create `app/routes/agentic_sessions.py`:
- `POST /admin/agentic-sessions` → create session (combo_id, optional session_id override), returns `AgenticSessionResponse`.
- `GET /admin/agentic-sessions/{session_id}` → retrieve session.
- `DELETE /admin/agentic-sessions/{session_id}` → close/expire session.
- Wire into chat endpoint: accept `X-Session-Id` header; if present, restore messages and iteration index; after run, persist updated session.

**Test Requirements (TR)**
- TR-10.1 (rule): POST create → GET retrieve returns same session_id and combo_id.
- TR-10.2 (rule): Closed session retrieved via GET has non-null closed_at.

---

## Task 11: Wire AGENTIC_MODEL through chat route for both streaming & non-streaming

**Priority**: high
**Status**: complete
**Parent ACs**: AC-1, AC-3, FR-1, FR-3

**Scope**
Modify `app/routes/chat.py` and `app/providers.py`:
- In `chat_completions`, when `body.model == AGENTIC_MODEL`:
  - Resolve the active agentic combo (or default AGENTIC_MODELS) + combo config.
  - If `stream=true`: use `run_agentic_stream` wrapped as SSE, converting agentic events to OpenAI SSE envelope. Embed an `X-Agentic-Event` header or surface custom events via the SSE JSON body alongside delta chunks; also emit standard delta chunks for the content stream.
  - If `stream=false`: use `run_agentic_loop`; return `ChatCompletionResponse` with final content, usage summing all steps, stop_reason mapped from agentic.
- Ensure `tools` from request body are merged with combo-enabled tools for tool-bearing agentic runs; pass tool schemas to upstream calls where the provider supports it.
- Call `record_agentic_analytics` after each agentic run completes (success or error).
- Ensure `smart_collect` and `smart_stream` for `AGENTIC_MODEL` actually route through the agentic loop (today they route through the same plan as other models, not the loop — this is the core missing wiring).

**Test Requirements (TR)**
- TR-11.1 (rule): Non-streaming request for `model=agentic` returns `ChatCompletionResponse` with usage tokens >0.
- TR-11.2 (rule): Streaming request for `model=agentic` with `stream=true` returns SSE chunks; at least one chunk contains an `agentic_event` field (or sidecar metadata), and final `[DONE]` exists.
- TR-11.3 (rule): After request completes, `record_agentic_analytics` has been invoked (verify via DB row existence for the combo+month).

---

## Task 12: Backoffice — Types, API client, AgenticConfig model

**Priority**: high
**Status**: complete
**Parent ACs**: AC-7, FR-7

**Scope**
Backoffice project: `/home/brave-dimaro/source/repos/BridgeRouteAI-backoffice`
- Ensure `src/types/index.ts` defines:
  - `AgenticConfig`, `BudgetStatusResponse`, `BudgetAlert`, `ComboPerformance`, `ModelRankingEntry`, `StrategyPreset`, `ToolDef`.
  - Extend `Combo` type to include `agentic_enabled`, `max_iterations`, `tool_calling`, `deterministic`, `loop_strategy`, `compression_target_tokens`, `monthly_token_budget`, `tools_allowed`, `strategy_preset`.
- Ensure `src/lib/api-client.ts` methods:
  - `combos.list`, `combos.get`, `combos.create`, `combos.update`, `combos.delete`, `combos.activate`, `combos.deactivate`
  - `combos.testAgentic(id)`, `combos.getBudget(id)`, `combos.getBudgetHistory(id, months?)`, `combos.getBudgetAlerts(id)`, `combos.resolveBudgetAlert(id, alertId)`, `combos.estimateCost(id)`, `combos.getTools(id)`, `combos.setTools(id, tools: string[])`, `combos.getStrategyPresets()`, `combos.applyPreset(id, presetName)`
  - `dashboard.getAgenticStats()`, `dashboard.getAgenticComboPerformances(months?)`, `dashboard.getAgenticComboPerformance(comboId, months?)`, `dashboard.getAgenticModelRankings(comboId, months?)`
  - `tools.list()`, `tools.get(name)`, `tools.exec(name, params)`
  - `agenticSessions.create(comboId)`, `agenticSessions.get(sessionId)`, `agenticSessions.delete(sessionId)`

**Test Requirements (TR)**
- TR-12.1 (rule): All declared types exist; `tsc --noEmit` on backoffice type-checks them.
- TR-12.2 (rule): Every declared api-client method is present on the exported client object (duck-typed reflection test via node REPL import).

---

## Task 13: Backoffice — Combos page with agentic fields, create/edit, budget widget

**Priority**: high
**Status**: complete
**Parent ACs**: AC-7, FR-7

**Scope**
- Combos list page: add columns for agentic_enabled, loop_strategy, max_iterations, monthly_token_budget; highlight rows with budget >80% usage via warning/error badges.
- Combo create dialog/form:
  - Agentic section with toggles (agentic_enabled, tool_calling, deterministic), dropdown loop_strategy (sequential/parallel/best_of), inputs max_iterations, compression_target_tokens, monthly_token_budget.
  - Strategy preset dropdown: selecting research/code-review/debug auto-fills the agentic defaults and recommended tools.
  - Tools selector: multi-select from the registered tools list, initial defaults from preset.
- Combo edit page: same fields, loaded with GET /{id}, save via PUT.
- Combo detail page sections:
  - Budget monitoring widget: progress bar (%), remaining tokens, number of alerts, history sparkline (last 3 months), replenishment prompt banner when applicable, alert list with resolve button.
  - Test agentic action: button triggers POST /test-agentic and renders iteration history table (iteration, model, provider, tokens, stop_reason, content length, compressed toggle).
- `src/app/(dashboard)/routers/combos/page.tsx` (or analogous path) is the main entry; create/edit/detail as subcomponents under the same route or sub-routes.
- Error handling: if budget endpoint 404s due to missing budget row, show "No budget set" fallback card.

**Test Requirements (TR)**
- TR-13.1 (rule): Create combo form with agentic toggles + budget value submits without client-side validation error; response echoes agentic fields back correctly.
- TR-13.2 (rule): Combo detail page budget widget renders progress bar with color change at >=80% and >=95% thresholds.
- TR-13.3 (rubric): Combo page UI fidelity. 0 = agentic fields missing; 1 = present but test-agentic/budget widget non-functional; 2 = every agentic field, budget widget, and test-agentic preview rendered and functional. Pass threshold ≥2.

---

## Task 14: Backoffice — Agentic analytics dashboard views

**Priority**: medium
**Status**: complete
**Parent ACs**: AC-6, FR-6

**Scope**
- Dashboard overview card for agentic usage: total_requests, total_iterations, avg_iterations/request, total_tokens, estimated_cost_usd, success_rate % — KPI card styling matching existing dashboard.
- Agentic performance page (or section within the existing Routers dashboard):
  - Per-combo performance table/list with combo name, requests, avg iterations, tokens, cost, stop_reason distribution (pie/bar).
  - Model ranking view: ranked list of models within each combo by success_rate, tokens/call, cost efficiency.
- Period selector (last 1 month / 3 months / 6 months / 12 months).
- Wire to the api-client analytics methods from Task 12.

**Test Requirements (TR)**
- TR-14.1 (rule): Period selector change re-fetches `months=N` parameter.
- TR-14.2 (rule): Model rankings are sorted descending by success_rate in the rendered list.

---

## Task 15: Per-combo API key rate limiting

**Priority**: medium
**Status**: complete
**Parent ACs**: AC-8c, FR-8c

**Scope**
- Extend `app/middleware/rate_limit.py` to support per-API-key and per-combo limits.
- For an API key with `combo_id` set and `rate_limit_per_minute`: enforce that limit specifically for requests where the combo matching that key would be selected (or the routed model maps to that combo). Also enforce global limits on top; take the minimum of (global, key-level, combo-level).
- Use Redis via `app/redis_client.py` for counters; fallback to in-process dict with TTL if Redis is unavailable.
- Extend `ApiKeyCreate` / api_keys route to accept and persist `combo_id` and `rate_limit_per_minute`.
- Exceeding limit → HTTP 429 with OpenAI-compatible error envelope `{type: "rate_limit_error"}` and Retry-After header.

**Test Requirements (TR)**
- TR-15.1 (rule): Create key scoped to combo with `rate_limit_per_minute=2`; 3rd immediate request receives 429 with `Retry-After` header set.
- TR-15.2 (rule): Redis unavailable → in-process fallback still enforces limits (test by mocking redis client failure).

---

## Task 16: Advanced strategy presets & preset application

**Priority**: medium
**Status**: complete
**Parent ACs**: AC-8a, FR-8a

**Scope**
- Define presets in a single source of truth module (inside `app/providers.py` or new `app/agentic_presets.py`):
  - `research`: `max_iterations=8`, `loop_strategy=parallel`, `compression_target_tokens=6000`, `recommended_tools=["web_search", "http_request", "calculate", "bash_command"]`
  - `code-review`: `max_iterations=6`, `loop_strategy=best_of`, `compression_target_tokens=5000`, `recommended_tools=["read_file", "calculate", "write_file", "bash_command"]`
  - `debug`: `max_iterations=10`, `loop_strategy=sequential`, `compression_target_tokens=4000`, `recommended_tools=["bash_command", "read_file", "calculate", "http_request"]`
- Expose via `GET /admin/combos/strategy-presets` from T7.
- When create combo passes `strategy_preset`: merge preset values with explicit overrides (explicit values win).
- Backoffice Task 13 already includes preset dropdown; ensure preset application does not erase user edits post-apply unless user explicitly re-selects.

**Test Requirements (TR)**
- TR-16.1 (rule): Create combo with `strategy_preset="research"` without explicit agentic_config → combo loaded has `max_iterations=8`, `loop_strategy="parallel"`.
- TR-16.2 (rule): Create combo with `strategy_preset="debug"` AND explicit `max_iterations=20` → `max_iterations=20` (explicit wins), loop_strategy=sequential from preset.

---

## Task 17: Conversation/session resumption end-to-end

**Priority**: medium
**Status**: complete
**Parent ACs**: AC-8b, FR-8b

**Scope**
- Wire `X-Session-Id` handling through the chat route (Task 10, Task 11):
  - If header present and session exists → seed agentic loop with session messages and start iteration index.
  - After run completes, update session with new messages, iteration index, step history.
- `AgenticResult.history` and new messages are merged into session store.
- When client resumes, the iteration counter does not restart from 0; it continues from the stored index.
- Add optional `X-Max-Iterations-Total` header to bound total iterations across resume (vs per-call).

**Test Requirements (TR)**
- TR-17.1 (rule): Create session, run call, close; re-open same session_id via `X-Session-Id` → next run `iteration` index starts at N (previous max).
- TR-17.2 (rule): Resume with `X-Max-Iterations-Total=5` after 3 iterations already consumed → only 2 additional iterations can run before stop_reason=max_iterations.

---

## Task 18: Integration sanity & build verification

**Priority**: high
**Status**: complete
**Parent ACs**: All ACs

**Scope**
- `alembic upgrade head` succeeds on clean DB.
- Backend starts: `uvicorn app.main:app --host 0.0.0.0 --port 8000` (or docker-compose alternative) boots without import errors.
- Backoffice build: `next build` (or `npm run build`) completes without type or compilation errors.
- End-to-end smoke curl sequence documented in the task completion evidence:
  1. Create admin API key.
  2. Create an agentic combo with monthly budget and strategy=parallel.
  3. Activate combo.
  4. Call `/v1/chat/completions` with `model=agentic`, stream=true, collect SSE.
  5. Check budget endpoint reports non-zero usage.
  6. Check dashboard agentic stats non-zero after run.
  7. Run `/test-agentic` on the combo and see iteration history.

**Test Requirements (TR)**
- TR-18.1 (rule): `alembic upgrade head` exit code 0.
- TR-18.2 (rule): Server boot has no Python traceback in stdout for import of agentic/budget/tools modules.
- TR-18.3 (rule): Backoffice `next build` / `tsc --noEmit` exit code 0 (or build command already present in backoffice package.json).
- TR-18.4 (rule): E2E curl sequence above produces all expected non-empty fields (each step response parsed successfully without top-level error key).

---

## Task 19: Reflection / critic repair pass

**Priority**: high
**Status**: complete
**Parent ACs**: Agentic quality loop

**Scope**
- `app/reflection.py`: `critique_draft` scores content with structural checks + `_score_result`-style heuristics; returns score, issues, feedback when below threshold.
- `AgenticLoop._maybe_repair`: after a content-producing iteration (sequential / parallel / best_of + stream paths), run critic once; if score < threshold and an iteration slot remains, inject critic message and allow exactly one repair round.
- Track `reflection_repairs` and `reflection_score` on `AgenticResult` and stream `complete` events.
- Config: `REFLECTION_ENABLED` (default true), `REFLECTION_THRESHOLD` (default 0.55) via `app/config.py`, threaded through `run_agentic_loop` / `run_agentic_stream` / `chat.py`.

**Test Requirements (TR)**
- TR-19.1 (rule): `critique_draft("")` returns `needs_repair=True`, score 0.0.
- TR-19.2 (rule): `critique_draft` of a long structured answer scores ≥ threshold and `needs_repair=False`.
- TR-19.3 (rule): `AgenticResult` includes `reflection_repairs` / `reflection_score` fields.
- TR-19.4 (rule): At most one repair message is injected per engine run (`reflection_repairs <= 1`).

---

## Task 20: Tool result cache

**Priority**: high
**Status**: complete
**Parent ACs**: Tool efficiency

**Scope**
- `app/tool_cache.py`: key = hash(tool + canonical JSON args); allow-list `calculate`, `read_file`, `web_search`, `http_request`, `echo_test`; exclude `bash_command`, `write_file`, `sleep`.
- Redis `SET EX` with in-process TTL fallback (mirror rate_limit pattern). TTL from `TOOL_CACHE_TTL` (default 300s); disable via `TOOL_CACHE_ENABLED=false`.
- Wire into `AgenticLoop._call_tool` and `tools_registry.execute_tool` (cache successful results only).

**Test Requirements (TR)**
- TR-20.1 (rule): `make_cache_key` is stable under key reordering of args.
- TR-20.2 (rule): `should_cache("bash_command")` is False; `should_cache("calculate")` is True when enabled.
- TR-20.3 (rule): Second identical `execute_tool("echo_test", ...)` within TTL returns cached payload (fallback store works with Redis down).

---

## Task 21: Eval harness (golden tasks + scoring + admin route)

**Priority**: high
**Status**: complete
**Parent ACs**: Evaluation / regression

**Scope**
- `evals/golden_tasks.json`: golden tasks with messages, `expect_substrings`, optional `expect_regex`, tool allow-list, max_iterations, loop_strategy.
- `app/eval_harness.py`: load tasks, run suite via `run_agentic_loop`, score (`success` + `score`), aggregate `success_rate`, `cost_efficiency = 1/(1+avg_tokens/2000)`, `score = success_rate * cost_efficiency`; store results in Redis with in-process fallback.
- `app/routes/evals.py` + models: `GET /admin/evals/tasks`, `POST /admin/evals/run`, `GET /admin/evals/results`, `GET /admin/evals/results/{run_id}` (admin key).

**Test Requirements (TR)**
- TR-21.1 (rule): `load_golden_tasks()` returns ≥ 1 task with non-empty `id` and `messages`.
- TR-21.2 (rule): `score_task` marks success only when all `expect_substrings` match (case-insensitive).
- TR-21.3 (rule): `GET /admin/evals/tasks` returns 200 with admin key; `POST /admin/evals/run` returns a `run_id` and `score` field.
- TR-21.4 (rule): Router registered in `app.main` (`/admin/evals` reachable).
