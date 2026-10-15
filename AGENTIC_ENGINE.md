# Agentic AI Engine Implementation

## Overview

Advanced agentic AI system with deterministic looping, token compression, and combo assignment for BridgeRouteAI.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Client Request                               │
│        {"model": "<combo name>", "messages": [...]}              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                 routes/chat.py: combo resolution                 │
│  API key → combo name → model ∈ combo.models                     │
│  combo.routing_mode == 'agentic' ?  → loop engine                │
│                       routing_mode == 'group' → strategy         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                 app/agentic_engine.py                            │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ Loop Engine  │→│ Token        │→│ Upstream Providers   │   │
│  │ Deterministic│  │ Compressor   │  │ (Kilo, FreeModels)   │   │
│  │ Round-Robin  │  │              │  │                      │   │
│  └─────────────┘  └──────────────┘  └──────────────────────┘   │
│       │                                                        │
│       ▼                                                        │
│  ┌─────────────┐                                              │
│  │ History Log │ → request_logs (full audit trail)            │
│  └─────────────┘                                              │
└─────────────────────────────────────────────────────────────────┘
```

## Database Schema

### `router_combos` Table Extension

| Column | Type | Default | Description |
|--------|------|---------|-------------|
| `agentic_enabled` | Boolean | `false` | Enable deterministic looping agentic mode |
| `max_iterations` | Integer | `5` | Max loop iterations before forced response |
| `tool_calling` | Boolean | `true` | Whether agent uses tool-calling loop |
| `deterministic` | Boolean | `true` | Deterministic model selection (round-robin) |
| `loop_strategy` | String | `"sequential"` | `sequential` / `parallel` / `best_of` |
| `compression_target_tokens` | Integer | `4000` | Target token count after compression |

## Core Modules

### 1. Agentic Engine (`app/agentic_engine.py`)

```python
class AgenticLoop:
    """Deterministic looping agentic engine with token compression."""
    
    def __init__(self, models, max_iterations=5, compression_target_tokens=4000):
        self.models = models
        self.max_iterations = max_iterations
        self.compression_target_tokens = compression_target_tokens
        self.history: list[AgenticStep] = []
    
    async def run(self, messages) -> str:
        """Loop: call model → check if done → compress → next model"""
        for i in range(self.max_iterations):
            model = self.models[i % len(self.models)]  # deterministic round-robin
            result = await call_upstream(model, messages)
            self.history.append(AgenticStep(i, model, result))
            if result.stop_reason == "stop" or not result.tool_calls:
                return result.content  # done
            messages = execute_tools(result.tool_calls)
            messages = self._compress(messages)  # token compression
        return result.content  # max iterations reached
    
    def _compress(self, messages) -> list:
        """Compress message history to target token count."""
        # 1. Remove redundant tool outputs
        # 2. Summarize older turns using small model
        # 3. Extract key facts
        # 4. Return compressed history
        return compressed_messages
```

### 2. Token Compression Strategy

**Compression Pipeline:**
```
Raw Messages → Deduplicate → Summarize Old Turns → Extract Facts → Compressed
```

| Step | Action | Token Savings |
|------|--------|---------------|
| Deduplicate | Remove duplicate tool args/results | ~20-30% |
| Summarize | Use small model to condense turns > threshold | ~40-50% |
| Extract Facts | Keep only key facts, actions, results | ~30-40% |
| Target Check | Ensure final token count ≤ target | bounded |

**Deterministic Compression:**
- Same input → same compression output
- No random summarization
- Fixed summarization model selection
- Reproducible results

### 3. Routing Integration (`app/providers.py`, `app/routing_strategies.py`)

Agentic is no longer a virtual model id. Every combo declares how a request
bound to it executes:

| Column | Values | Meaning |
|--------|--------|---------|
| `routing_mode` | `agentic` / `group` | engine: multi-iteration tool loop, or a single-shot strategy |
| `routing_strategy` | `fallback` / `round_robin` / `fusion` | group only — how the models are used |
| `judge_model` | `freemodels/<model_id>` | fusion only — required for new combos; the judge that fuses the panel answers (NULL only on rows created before 011, which fall back to the combo's first model) |

Request binding, in order: **combo-scoped API key → `model` equals a combo's
name → `model` is listed in a combo's `models[]`**. The combo name is itself a
routable model id (it appears in `GET /v1/models` with `owned_by: "combo"`).

```python
def resolve_combo_ref(ref: str) -> dict | None:
    """Resolve a combo by its name — the combo name is a routable model id."""
```

Group strategies (`app/routing_strategies.py`):

- **fallback** — the combo's declared list becomes the attempt plan; the
  existing circuit breakers and first-token buffering apply unchanged.
- **round_robin** — the starting model rotates per request (Redis `INCR`, with
  a local counter if Redis is down); the tail keeps the declared order so
  failover still works.
- **fusion** — every panel model answers in parallel, then the judge
  synthesizes one reply. **N+1 calls per request.** The judge is mandatory and
  must be a reachable FreeModels model (`GET /admin/combos/judge-models`
  lists them), so the synthesizer is always a free, currently-working model.
  If the judge fails but the panel produced answers, the first panel answer is
  returned instead of an error.

### 4. API Endpoints (`app/routes/combos.py`)

- `POST /admin/combos` — Create combo with `agentic_config`
- `PUT /admin/combos/{id}` — Update agentic config
- `POST /admin/combos/{id}/test-agentic` — Dry-run agentic loop
- `GET /admin/combos/{id}` — Get combo with agentic fields
- `GET /admin/combos/routing-strategies` — Strategy catalog (`requires_judge`, `judge_provider`)
- `GET /admin/combos/judge-models` — Reachable FreeModels judges for fusion

### 5. Pydantic Models (`app/models.py`)

```python
class AgenticConfig(BaseModel):
    agentic_enabled: bool = False
    max_iterations: int = 5
    tool_calling: bool = True
    deterministic: bool = True
    loop_strategy: str = "sequential"
    compression_target_tokens: int = 4000

class ComboCreate(BaseModel):
    name: str
    router_type: str
    models: list[ComboModel]
    agentic_config: AgenticConfig | None = None
```

## Token Compression Details

### Why Compression Matters

Without compression, an agentic loop of 5 iterations:
- System prompt: ~200 tokens
- 5 model calls × ~2000 tokens each = ~10,000 tokens
- Tool results: ~3000 tokens per iteration = ~15,000 tokens
- **Total: ~25,000+ tokens per request**

With compression:
- System prompt: ~200 tokens
- Compressed history: ~2000 tokens (after summarization)
- Current turn: ~2000 tokens
- Tool results (current): ~1000 tokens
- **Total: ~5,000 tokens per request**

### Compression Algorithm

```python
def _compress_messages(messages: list[Message], target_tokens: int) -> list[Message]:
    # 1. Strip verbose tool outputs (keep only essential fields)
    messages = strip_verbose_tool_outputs(messages)
    
    # 2. Check if under target
    if count_tokens(messages) <= target_tokens:
        return messages
    
    # 3. Identify compression boundary (last N messages stay intact)
    recent = messages[-2:]  # Keep last 2 turns intact
    old = messages[:-2]
    
    # 4. Summarize old messages deterministically
    summary = summarize(old, model="fast-summarizer")
    
    # 5. Return compressed history
    return [summary] + recent
```

### Deterministic Summarization

- Uses fixed model (e.g., first model in combo) for summarization
- No random seed or variation
- Same input → same summary output
- Reproducible across runs

## Combo Assignment Flow

```
Admin creates combo
    │
    ├── router_type = "agentic"
    │   ├── agentic_enabled: true/false
    │   ├── max_iterations: N
    │   ├── compression_target_tokens: M
    │   └── loop_strategy: sequential/parallel/best_of
    │
    └── Activate combo
        │
        ▼
    GET /v1/chat/completions {"model": "my-agent"}   # combo name
        │
        ▼
    routes/chat.py → resolve combo (key → name → models[])
        │
        ▼
    combo.routing_mode == 'agentic'
        │
        ▼
    agentic_engine.run(messages, combo)
        │
        ├── Loop iteration 1: model call
        ├── Tool result received
        ├── Token compression applied
        ├── Loop iteration 2: next model call
        ├── ...
        └── Max iterations or stop → return response
```

## Files Created/Modified

| File | Action |
|------|--------|
| `alembic/versions/006_add_agentic_combo_fields.py` | Create |
| `app/db_models.py` | Add agentic fields to DBRouterCombo |
| `app/models.py` | Add AgenticConfig Pydantic model |
| `app/crud.py` | Extend combo CRUD with agentic fields |
| `app/agentic_engine.py` | Create new module |
| `app/providers.py` | Combo index + `resolve_combo_ref()`; combo name routing |
| `app/routing_strategies.py` | Create (fallback / round_robin / fusion) |
| `alembic/versions/011_add_routing_mode.py` | Create (`routing_mode`, `routing_strategy`, `judge_model`) |
| `app/routes/combos.py` | Update endpoints for agentic config |
| `backoffice/src/types/index.ts` | Add AgenticConfig type |
| `backoffice/src/lib/api-client.ts` | Add agentic methods |
| `backoffice/src/app/(dashboard)/routers/combos/page.tsx` | Add agentic UI |

## Running the Migration

```bash
cd /home/brave-dimaro/source/repos/BridgeRouteAI
docker compose build api
docker compose up -d api
docker compose run --rm api alembic upgrade head
```

## Testing

1. **Create agentic combo:**
```bash
curl -X POST -H "Authorization: Bearer test-admin-key-12345" \
  -H "Content-Type: application/json" \
  http://localhost:8001/admin/combos \
  -d '{
    "name": "Agentic Reasoning",
    "router_type": "agentic",
    "agentic_enabled": true,
    "max_iterations": 3,
    "compression_target_tokens": 2000,
    "models": [
      {"provider": "agnes", "model_id": "agnes-3.0-flash"},
      {"provider": "freemodels", "model_id": "deepseek-v4-pro-thinking"}
    ]
  }'
```

2. **Activate combo:**
```bash
curl -X POST -H "Authorization: Bearer test-admin-key-12345" \
  http://localhost:8001/admin/combos/{id}/activate
```

3. **Test agentic routing:**
```bash
# routing_mode='agentic' combo named "my-agent"
curl -X POST -H "Authorization: Bearer test-admin-key-12345" \
  -H "Content-Type: application/json" \
  http://localhost:8001/v1/chat/completions \
  -d '{"model": "my-agent", "messages": [{"role": "user", "content": "solve this step by step"}]}'

# model="agentic" is rejected with 400 since migration 011
```

## Token Budget Management

### Overview

Each combo can have a monthly token budget that is enforced automatically. The agentic engine tracks token usage and prevents exceeding the budget.

### How It Works

```
Combo Created with monthly_token_budget
    │
    ▼
Agentic Loop runs → records tokens used → checks budget
    │
    ├── Within budget → continue loop
    └── Budget exceeded → stop with "budget_exceeded" stop_reason
```

### Database Schema

**`token_budgets` table:**
| Column | Type | Description |
|--------|------|-------------|
| `combo_id` | String(32) | Foreign key to router_combos |
| `month_year` | String(7) | "YYYY-MM" format |
| `tokens_used` | Integer | Total tokens consumed this month |
| `tokens_budget` | Integer | Monthly budget limit |

### Budget Enforcement

- **Lazy tracking**: Budget entries are created on first token usage
- **Session-level enforcement**: `BudgetEnforcer` checks before each iteration
- **Monthly reset**: Budget tracking resets each month based on `month_year`
- **Status endpoint**: `GET /admin/combos/{id}/budget` returns usage stats

### API Endpoints

```bash
# Get budget status
curl -H "Authorization: Bearer test-admin-key-12345" \
  http://localhost:8001/admin/combos/{id}/budget

# Create combo with monthly budget
curl -X POST -H "Authorization: Bearer test-admin-key-12345" \
  -H "Content-Type: application/json" \
  http://localhost:8001/admin/combos \
  -d '{
    "name": "Production Agentic",
    "router_type": "agentic",
    "monthly_token_budget": 5000000,
    "agentic_config": {
      "agentic_enabled": true,
      "max_iterations": 3,
      "compression_target_tokens": 2000
    },
    "models": [...]
  }'
```

### Advantages

| Feature | Benefit |
|---------|---------|
| Deterministic | Same input → same output sequence |
| Looping | Multi-step reasoning with tool calls |
| Token Compression | Reduces token usage by 60-80% |
| Monthly Token Budget | Prevents unlimited spending |
| Combo Assignment | Admin-configurable agentic behaviors |
| Cost Control | max_iterations + budget bounds spending |
| Audit Trail | Full history in request_logs |
| Budget Status API | Real-time usage monitoring |
| Reflection | One critic repair pass when draft scores below threshold |
| Tool Cache | Redis TTL cache for idempotent tool results |
| Eval Harness | Golden tasks + score under `/admin/evals` |
| Production Ready | Reproducible, tested, bounded |

## Version History

- `v1.1.0` — Agentic engine with token compression and monthly budget management
  - Migration `006_add_agentic_combo_fields`: agentic config fields
  - Migration `007_add_token_budget_management`: monthly token budget tracking
  - New module `app/agentic_engine.py`: AgenticLoop with TokenCompressor
  - New module `app/token_budget_manager.py`: TokenBudgetManager, BudgetEnforcer
  - New endpoints: `POST /admin/combos/{id}/test-agentic`, `GET /admin/combos/{id}/budget`
  - Backoffice: Combo management page with agentic config and budget monitoring

- `v1.2.0` — Reflection critic, tool cache, eval harness
  - New module `app/reflection.py`: one-shot critic repair pass
  - New module `app/tool_cache.py`: Redis tool result cache + fallback
  - New module `app/eval_harness.py` + `evals/golden_tasks.json`
  - New endpoints: `GET/POST /admin/evals/*`

## Test Results

| Test | Result |
|------|--------|
| Create agentic combo | ✓ |
| Activate agentic combo | ✓ |
| Agentic routing (model returned: agnes-3.0-flash) | ✓ |
| Token compression | ✓ |
| Monthly budget (5M tokens) | ✓ |
| Budget status API | ✓ |
| Test agentic endpoint | ✓ |
| Backoffice build | ✓ |
| Full combo list with all fields | ✓ |
