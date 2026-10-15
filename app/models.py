from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal, Union
from datetime import datetime


# ── OpenAI Chat Completion ───────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "function", "tool"]
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class Tool(BaseModel):
    type: Literal["function"] = "function"
    function: ToolFunction


class ResponseFormat(BaseModel):
    type: Literal["text", "json_object"] = "text"


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    n: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, int]] = None
    user: Optional[str] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    response_format: Optional[ResponseFormat] = None
    seed: Optional[int] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    # Agentic gateway extension (OpenAI-compatible superset): metadata
    # ESCALATION triggers auto-escalation to the combo's strongest model.
    metadata: Optional[Dict[str, Any]] = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str
    permission: List[Any] = []
    root: Optional[str] = None
    parent: Optional[str] = None
    tools_supported: bool = False


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


class ModelDeleteResponse(BaseModel):
    id: str
    object: str = "model"
    deleted: bool = True


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo
    system_fingerprint: Optional[str] = None


class StreamChoice(BaseModel):
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[StreamChoice]
    system_fingerprint: Optional[str] = None
    usage: Optional[UsageInfo] = None


# ── OpenAI Embeddings ───────────────────────────────────────────────────────

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = "text-embedding-ada-002"
    encoding_format: Optional[str] = None
    dimensions: Optional[int] = None
    user: Optional[str] = None


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingUsage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage


# ── OpenAI Error ────────────────────────────────────────────────────────────

class OpenAIError(BaseModel):
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class OpenAIErrorResponse(BaseModel):
    error: OpenAIError


# ── OpenAI Legacy Completions ───────────────────────────────────────────────

class LegacyCompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]] = ""
    suffix: Optional[str] = None
    max_tokens: int = 16
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    user: Optional[str] = None


class LegacyCompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = None


class LegacyCompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[LegacyCompletionChoice]
    usage: UsageInfo


# ── Admin / Dashboard models ────────────────────────────────────────────────

class StatusResponse(BaseModel):
    status: str
    message: str = ""


class HealthResponse(BaseModel):
    status: str
    redis: str
    postgres: str


class DashboardStats(BaseModel):
    total_requests: int = 0
    active_users: int = 0
    active_api_keys: int = 0
    avg_response_time: float = 0
    error_rate: float = 0
    requests_today: int = 0
    requests_this_hour: int = 0
    token_usage_today: int = 0


class ChartDataPoint(BaseModel):
    time: str
    requests: int = 0
    errors: int = 0
    avg_duration: float = 0


class ApiKey(BaseModel):
    id: str
    name: str
    # Masked placeholder (e.g. "sk-1a2b3c4d******"). The raw key is returned only
    # by ApiKeyCreated.full_key on creation and is never stored server-side.
    key: str = ""
    key_mask: str = ""
    created_at: str
    last_used_at: Optional[str] = None
    expires_at: Optional[str] = None
    is_active: bool = True
    request_count: int = 0
    combo_id: str | None = None
    rate_limit_per_minute: int | None = None
    monthly_token_budget: int | None = 0


class ApiKeyCreated(ApiKey):
    """POST /admin/api-keys response — the only place the raw key is disclosed."""
    full_key: str


class ApiKeyCreate(BaseModel):
    name: str = Field(..., min_length=1)
    combo_id: str | None = None
    rate_limit_per_minute: int | None = 60
    # 0 = unlimited; enforced by the rate limiter (aggregate of request_logs)
    monthly_token_budget: int = Field(default=0, ge=0, le=1_000_000_000)


class ApiKeyListResponse(BaseModel):
    data: List[ApiKey]


class User(BaseModel):
    id: str
    email: str
    name: str
    role: Literal["admin", "user"] = "user"
    created_at: str
    last_login_at: Optional[str] = None
    is_active: bool = True
    api_key_count: int = 0
    request_count: int = 0


class UserCreate(BaseModel):
    email: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    role: Literal["admin", "user"] = "user"


class UserUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    role: Optional[Literal["admin", "user"]] = None
    is_active: Optional[bool] = None


class UserListResponse(BaseModel):
    data: List[User]


class RequestLog(BaseModel):
    id: str
    timestamp: str
    method: str
    path: str
    status_code: int
    model: str
    duration_ms: int
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    api_key_id: Optional[str] = None
    tokens_used: Optional[int] = None
    # Replay/debugging snapshots (bounded, redacted)
    request_payload: Optional[dict] = None
    response_payload: Optional[str] = None


class RequestLogListResponse(BaseModel):
    data: List[RequestLog]
    total: int


class SystemSettings(BaseModel):
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000
    max_tokens_per_request: int = 4096
    allowed_models: List[str] = [
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
    ]
    maintenance_mode: bool = False
    cors_origins: List[str] = ["*"]


class AuditLog(BaseModel):
    id: str
    timestamp: str
    action: str
    actor: str
    target: str
    details: str
    ip_address: str


class AuditLogListResponse(BaseModel):
    data: List[AuditLog]
    total: int


class AgenticConfig(BaseModel):
    agentic_enabled: bool = False
    max_iterations: int = Field(default=5, ge=1, le=50)
    tool_calling: bool = True
    deterministic: bool = True
    loop_strategy: str = "sequential"
    compression_target_tokens: int = Field(default=4000, ge=500, le=200_000)
    # true = legacy per-iteration round-robin (load spreading)
    load_spread: bool = False


class AgenticStrategyPreset(BaseModel):
    name: str
    description: str
    max_iterations: int
    loop_strategy: str
    compression_target_tokens: int
    recommended_tools: list[str]


class ComboPresetApply(BaseModel):
    preset_name: str


class AgenticStats(BaseModel):
    total_requests: int = 0
    total_iterations: int = 0
    avg_iterations_per_request: float = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0
    success_rate: float = 0


class ComboPerformance(BaseModel):
    combo_id: str
    period: str
    requests_count: int = 0
    total_iterations: int = 0
    avg_iterations: float = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0
    stop_reason_counts: dict = {}
    tool_call_counts: dict = {}


class ModelRankingEntry(BaseModel):
    provider: str
    model_id: str
    success_rate: float = 0
    avg_tokens_per_call: float = 0
    cost_efficiency: float = 0
    total_calls: int = 0


class AgenticSessionCreate(BaseModel):
    combo_id: str
    session_id: str | None = None


class AgenticSessionResponse(BaseModel):
    id: str
    session_id: str
    combo_id: str
    messages: list = []
    iteration_index: int = 0
    step_history: list = []
    created_at: str | None = None
    updated_at: str | None = None
    closed_at: str | None = None


class BudgetAlertResponse(BaseModel):
    id: str
    combo_id: str
    month_year: str
    level: str
    percentage: float
    tokens_used: int
    tokens_budget: int
    triggered_at: str | None = None
    resolved_at: str | None = None


class BudgetStatusResponse(BaseModel):
    combo_id: str
    month_year: str
    tokens_used: int
    tokens_budget: int
    tokens_remaining: int
    percentage: float
    within_budget: bool
    no_budget_set: bool = False
    alerts: list[BudgetAlertResponse] = []
    historical_usage: list[dict] = []
    replenishment_prompt: dict | None = None


class CostEstimateResponse(BaseModel):
    estimated_total_tokens: int
    estimated_cost_usd: float
    iterations: int
    models_per_iteration: int
    tokens_per_iteration: int


class ComboModel(BaseModel):
    provider: str
    model_id: str


# ── Combo routing (011) ──────────────────────────────────────────────────────
#
# Every combo declares HOW a request bound to it executes. ``agentic`` runs the
# multi-iteration tool loop; ``group`` runs a single-shot strategy over the
# combo's model list under the combo's name.

ROUTING_MODES = ("agentic", "group")
ROUTING_STRATEGIES = ("fallback", "round_robin", "fusion")

# Fusion judges are pinned to one provider: the admin picks the synthesizer
# explicitly, and it always comes from the free catalog rather than an
# arbitrary (possibly paid) panel member.
FUSION_JUDGE_PROVIDER = "freemodels"


class RoutingStrategyInfo(BaseModel):
    """Static description of a group routing strategy (for admin UIs)."""
    name: str
    label: str
    description: str
    cost_note: str | None = None
    min_models: int = 1
    # Strategies that synthesize a single answer must be given an explicit
    # judge model; ``judge_provider`` pins which catalog the judge comes from.
    requires_judge: bool = False
    judge_provider: str | None = None


ROUTING_STRATEGY_CATALOG: list[RoutingStrategyInfo] = [
    RoutingStrategyInfo(
        name="fallback",
        label="Fallback",
        description="Tries models in order (next on failure).",
        min_models=1,
    ),
    RoutingStrategyInfo(
        name="round_robin",
        label="Round Robin",
        description="Rotates models across requests to spread load.",
        min_models=1,
    ),
    RoutingStrategyInfo(
        name="fusion",
        label="Fusion",
        description=(
            "Queries all models in parallel, then a judge model you pick "
            "synthesizes one answer."
        ),
        cost_note=(
            "Best quality, but costs the most: every request bills all panel "
            "models + the judge (N+1 calls)."
        ),
        min_models=2,
        requires_judge=True,
        judge_provider=FUSION_JUDGE_PROVIDER,
    ),
]


class ComboCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    router_type: str
    models: list[ComboModel] = Field(min_length=1)
    routing_mode: Literal["agentic", "group"] = "group"
    routing_strategy: Literal["fallback", "round_robin", "fusion"] = "fallback"
    judge_model: str | None = None
    agentic_config: AgenticConfig | None = None
    monthly_token_budget: int = Field(default=0, ge=0, le=1_000_000_000)
    tools_allowed: list[str] | None = None
    strategy_preset: str | None = None


class ComboUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    # Optimistic concurrency: when set, the stored version must match or the
    # update is rejected with 409. Also settable via the If-Match header.
    expected_version: int | None = None
    models: list[ComboModel] | None = None
    routing_mode: Literal["agentic", "group"] | None = None
    routing_strategy: Literal["fallback", "round_robin", "fusion"] | None = None
    judge_model: str | None = None
    agentic_config: AgenticConfig | None = None
    tools_allowed: list[str] | None = None
    strategy_preset: str | None = None
    monthly_token_budget: int | None = Field(default=None, ge=0, le=1_000_000_000)
    load_spread: bool | None = None


class ComboResponse(BaseModel):
    id: str
    name: str
    router_type: str
    models: list[ComboModel]
    is_active: bool
    routing_mode: str = "group"
    routing_strategy: str = "fallback"
    judge_model: str | None = None
    agentic_enabled: bool = False
    max_iterations: int = 5
    tool_calling: bool = True
    deterministic: bool = True
    loop_strategy: str = "sequential"
    compression_target_tokens: int = 4000
    monthly_token_budget: int = 0
    load_spread: bool = False
    tools_allowed: list[str] | None = None
    strategy_preset: str | None = None
    version: int = 1
    created_at: str | None = None
    updated_at: str | None = None


class EvalTaskInfo(BaseModel):
    id: str
    name: str = ""
    messages: list[dict] = []
    expect_substrings: list[str] = []
    expect_regex: str | None = None
    tools_allowed: list[str] = []
    max_iterations: int = 4
    loop_strategy: str = "sequential"


class EvalRunRequest(BaseModel):
    task_ids: list[str] | None = None
