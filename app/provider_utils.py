"""Shared helpers for upstream providers that speak OpenAI-style chat.

These live outside ``providers``/``kilo`` so both modules can use them
without a circular import.
"""
import json
from dataclasses import dataclass

# Free-tier upstreams reject large ``max_tokens`` with a 400. Requests above
# the cap are clamped rather than failed; ``None`` means "no cap" (rare).
DEFAULT_MAX_TOKENS_CAP = 8192
STRICT_MAX_TOKENS_CAP = 4096


@dataclass(frozen=True)
class SamplingParams:
    """Sampling controls forwarded to upstream OpenAI-style APIs.

    Only non-``None`` fields are written into the payload so every upstream
    default is preserved when the caller does not specify a value.
    """

    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: list[str] | str | None = None
    tool_choice: object | None = None

    def has_any(self) -> bool:
        return any(
            getattr(self, f) is not None
            for f in ("temperature", "top_p", "max_tokens", "stop", "tool_choice")
        )


def apply_sampling(
    payload: dict,
    sampling: "SamplingParams | None",
    max_tokens_cap: int | None = DEFAULT_MAX_TOKENS_CAP,
) -> dict:
    """Merge sampling controls into an upstream payload.

    ``max_tokens`` is clamped to ``max_tokens_cap`` (when set) because the
    free-tier upstreams reject unbounded values. Fields that are ``None`` are
    omitted entirely.
    """
    if sampling is None:
        return payload
    if sampling.temperature is not None:
        payload["temperature"] = sampling.temperature
    if sampling.top_p is not None:
        payload["top_p"] = sampling.top_p
    if sampling.max_tokens is not None:
        mt = sampling.max_tokens
        if max_tokens_cap is not None:
            mt = min(mt, max_tokens_cap)
        if mt > 0:
            payload["max_tokens"] = mt
    if sampling.stop is not None:
        payload["stop"] = sampling.stop
    if sampling.tool_choice is not None:
        payload["tool_choice"] = sampling.tool_choice
    return payload


class QuotaError(Exception):
    """Raised when an upstream returns 429 or a quota-exhausted error."""
    def __init__(self, message: str, provider: str = "", model: str = "", status_code: int = 429):
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.status_code = status_code


def msg_get(msg, key: str, default=None):
    """Read a field from a message that may be a pydantic model or a dict."""
    if hasattr(msg, key):
        return getattr(msg, key)
    if hasattr(msg, "get"):
        return msg.get(key, default)
    return default


def to_openai_messages(messages: list) -> list[dict]:
    """Convert OpenAI-format messages into plain chat messages."""
    out = []
    for msg in messages:
        entry = {
            "role": msg_get(msg, "role", "user"),
            "content": msg_get(msg, "content") or "",
        }
        tool_calls = msg_get(msg, "tool_calls")
        if tool_calls:
            entry["tool_calls"] = tool_calls
        tool_call_id = msg_get(msg, "tool_call_id")
        if tool_call_id:
            entry["tool_call_id"] = tool_call_id
        name = msg_get(msg, "name")
        if name:
            entry["name"] = name
        out.append(entry)
    return out


def usage_chunk(chat_id: str, created: int, model: str, prompt_tokens: int, text: str) -> str:
    """Build an OpenAI streaming usage chunk (empty choices)."""
    ct = max(1, len(text) // 4)
    return "data: " + json.dumps({
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": ct,
            "total_tokens": prompt_tokens + ct,
        },
    }) + "\n\n"


def sse_chunk(
    chat_id: str,
    created: int,
    model: str,
    content: str = None,
    role: str = None,
    finish_reason: str = None,
    tool_calls: list = None,
) -> str:
    """Build an OpenAI-compatible streaming chunk."""
    delta = {}
    if role:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    if tool_calls:
        delta["tool_calls"] = tool_calls

    chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return "data: " + json.dumps(chunk) + "\n\n"
