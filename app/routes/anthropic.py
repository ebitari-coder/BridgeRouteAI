"""Anthropic Messages API adapter for Claude Code and other Anthropic-native tools.

Translates Anthropic Messages API requests (including tool use) into upstream
requests and back into Anthropic's format, using the shared provider failover
in ``app.providers``.
"""
import json
import uuid
import logging
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Literal, Dict, Any
from app.provider_utils import SamplingParams
from app.providers import smart_stream

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Anthropic"])


# ── Anthropic Messages API Models ───────────────────────────────────────────

class AnthropicContentBlock(BaseModel):
    """A content block — text, tool_use, or tool_result.

    Deliberately permissive so unknown block types pass through.
    """
    type: str
    text: Optional[str] = None
    # tool_use
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[Dict[str, Any]] = None
    # tool_result
    tool_use_id: Optional[str] = None
    content: Optional[Any] = None
    is_error: Optional[bool] = None

    model_config = {"extra": "allow"}


class AnthropicMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[AnthropicContentBlock]]


class AnthropicTool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None


class AnthropicMessagesRequest(BaseModel):
    model: str
    messages: List[AnthropicMessage]
    max_tokens: int = Field(..., gt=0)
    system: Optional[Union[str, List[Dict[str, Any]]]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    stream: bool = False
    tools: Optional[List[AnthropicTool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class AnthropicUsage(BaseModel):
    input_tokens: int
    output_tokens: int


class AnthropicResponse(BaseModel):
    id: str
    type: str = "message"
    role: str = "assistant"
    content: List[Dict[str, Any]]
    model: str
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None
    usage: AnthropicUsage


# ── Conversion helpers ──────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _anthropic_error(message: str, err_type: str = "invalid_request_error", status: int = 400):
    return JSONResponse(
        status_code=status,
        content={"type": "error", "error": {"type": err_type, "message": message}},
    )


def _system_to_text(system) -> Optional[str]:
    if system is None:
        return None
    if isinstance(system, str):
        return system or None
    parts = []
    for block in system:
        if isinstance(block, dict) and block.get("text"):
            parts.append(block["text"])
        elif isinstance(block, str):
            parts.append(block)
    return "\n".join(parts) or None


def _tool_result_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return json.dumps(content, ensure_ascii=False)


def _messages_to_openai(messages: list, system=None) -> list[dict]:
    """Convert Anthropic messages (incl. tool_use / tool_result) to OpenAI format."""
    result = []
    sys_text = _system_to_text(system)
    if sys_text:
        result.append({"role": "system", "content": sys_text})

    for msg in messages:
        blocks = msg.content

        if isinstance(blocks, str):
            result.append({"role": msg.role, "content": blocks})
            continue

        if msg.role == "assistant":
            texts, tool_calls = [], []
            for b in blocks:
                if b.type == "text" and b.text:
                    texts.append(b.text)
                elif b.type == "tool_use":
                    tool_calls.append({
                        "id": b.id or f"call_{uuid.uuid4().hex[:24]}",
                        "type": "function",
                        "function": {
                            "name": b.name or "",
                            "arguments": json.dumps(b.input or {}, ensure_ascii=False),
                        },
                    })
            entry = {"role": "assistant", "content": "\n".join(texts)}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            result.append(entry)
        else:  # user
            texts, tool_msgs = [], []
            for b in blocks:
                if b.type == "text" and b.text:
                    texts.append(b.text)
                elif b.type == "tool_result":
                    tool_msgs.append({
                        "role": "tool",
                        "tool_call_id": b.tool_use_id or "",
                        "content": _tool_result_text(b.content),
                    })
            if texts:
                result.append({"role": "user", "content": "\n".join(texts)})
            result.extend(tool_msgs)

    return result


def _tools_to_openai(tools) -> Optional[list[dict]]:
    if not tools:
        return None
    out = []
    for t in tools:
        out.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.input_schema or {"type": "object", "properties": {}},
            },
        })
    return out or None


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _parse_event(raw: str) -> tuple[Optional[str], Optional[dict]]:
    if not raw.startswith("event: "):
        return None, None
    header, _, rest = raw.partition("\ndata: ")
    name = header[len("event: "):].strip()
    try:
        return name, json.loads(rest.strip())
    except json.JSONDecodeError:
        return name, None


# ── Streaming ───────────────────────────────────────────────────────────────

async def _stream_anthropic(
    model: str,
    messages: list,
    tools: list = None,
    sampling: SamplingParams | None = None,
):
    """Translate the upstream OpenAI-style stream into Anthropic SSE events."""
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    yield _sse_event("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })

    next_index = 0
    text_index = None
    tool_slots: dict[int, dict] = {}
    accumulated_text = ""

    async for chunk_str in smart_stream(model, messages, tools=tools, sampling=sampling):
        if not chunk_str.startswith("data:"):
            continue
        data_str = chunk_str[6:].strip()
        if data_str == "[DONE]":
            break
        try:
            parsed = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        if "error" in parsed:
            yield _sse_event("error", {
                "type": "error",
                "error": {
                    "type": parsed["error"].get("type", "api_error"),
                    "message": parsed["error"].get("message", "Unknown error"),
                },
            })
            return

        choices = parsed.get("choices") or []
        if not choices:
            continue

        delta = choices[0].get("delta") or {}

        content = delta.get("content")
        if content:
            if text_index is None:
                text_index = next_index
                next_index += 1
                yield _sse_event("content_block_start", {
                    "type": "content_block_start",
                    "index": text_index,
                    "content_block": {"type": "text", "text": ""},
                })
            accumulated_text += content
            yield _sse_event("content_block_delta", {
                "type": "content_block_delta",
                "index": text_index,
                "delta": {"type": "text_delta", "text": content},
            })

        for tc in (delta.get("tool_calls") or []):
            oi = tc.get("index", 0)
            fn = tc.get("function") or {}
            slot = tool_slots.get(oi)
            if slot is None:
                slot = {
                    "index": next_index,
                    "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                    "name": fn.get("name") or "",
                    "started": False,
                }
                next_index += 1
                tool_slots[oi] = slot
            if fn.get("name"):
                slot["name"] = fn["name"]
            if not slot["started"] and slot["name"]:
                yield _sse_event("content_block_start", {
                    "type": "content_block_start",
                    "index": slot["index"],
                    "content_block": {
                        "type": "tool_use",
                        "id": slot["id"],
                        "name": slot["name"],
                        "input": {},
                    },
                })
                slot["started"] = True
            args = fn.get("arguments")
            if args:
                yield _sse_event("content_block_delta", {
                    "type": "content_block_delta",
                    "index": slot["index"],
                    "delta": {"type": "input_json_delta", "partial_json": args},
                })

        finish_reason = choices[0].get("finish_reason")
        if finish_reason and finish_reason != "null":
            break

    if text_index is not None:
        yield _sse_event("content_block_stop", {"type": "content_block_stop", "index": text_index})

    for slot in tool_slots.values():
        if not slot["started"]:
            yield _sse_event("content_block_start", {
                "type": "content_block_start",
                "index": slot["index"],
                "content_block": {"type": "tool_use", "id": slot["id"], "name": slot["name"], "input": {}},
            })
        yield _sse_event("content_block_stop", {"type": "content_block_stop", "index": slot["index"]})

    stop_reason = "tool_use" if tool_slots else "end_turn"
    yield _sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": _estimate_tokens(accumulated_text)},
    })
    yield _sse_event("message_stop", {"type": "message_stop"})


# ── Routes ──────────────────────────────────────────────────────────────────

@router.post("/v1/messages")
async def anthropic_messages(request: Request, body: AnthropicMessagesRequest):
    """Anthropic Messages API endpoint for Claude Code and other Anthropic-native tools."""
    if not body.messages:
        return _anthropic_error("messages is required")

    chat_messages = _messages_to_openai(body.messages, body.system)
    tools = _tools_to_openai(body.tools)

    if body.stream:
        return StreamingResponse(
            _stream_anthropic(body.model, chat_messages, tools, _sampling_from(body)),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    # Non-streaming: reconstruct content blocks from the event stream
    blocks: dict[int, dict] = {}
    stop_reason = None

    async for event_str in _stream_anthropic(body.model, chat_messages, tools, _sampling_from(body)):
        name, data = _parse_event(event_str)
        if name == "error":
            if data:
                return _anthropic_error(
                    data["error"].get("message", "Unknown error"),
                    data["error"].get("type", "api_error"),
                    502,
                )
            return _anthropic_error("Upstream error", "api_error", 502)
        if not data:
            continue

        etype = data.get("type")
        if etype == "content_block_start":
            cb = data.get("content_block", {})
            idx = data.get("index", 0)
            if cb.get("type") == "tool_use":
                blocks[idx] = {"type": "tool_use", "id": cb.get("id"), "name": cb.get("name"), "_json": ""}
            else:
                blocks[idx] = {"type": "text", "text": ""}
        elif etype == "content_block_delta":
            idx = data.get("index", 0)
            delta = data.get("delta", {})
            if idx not in blocks:
                continue
            if delta.get("type") == "text_delta":
                blocks[idx]["text"] += delta.get("text", "")
            elif delta.get("type") == "input_json_delta":
                blocks[idx]["_json"] += delta.get("partial_json", "")
        elif etype == "message_delta":
            stop_reason = data.get("delta", {}).get("stop_reason")

    content: list[dict] = []
    for idx in sorted(blocks):
        b = blocks[idx]
        if b["type"] == "text":
            if b["text"]:
                content.append({"type": "text", "text": b["text"]})
        else:
            try:
                tool_input = json.loads(b["_json"]) if b["_json"] else {}
            except json.JSONDecodeError:
                tool_input = {}
            content.append({
                "type": "tool_use",
                "id": b["id"],
                "name": b["name"],
                "input": tool_input,
            })

    if not content:
        content = [{"type": "text", "text": ""}]
    if stop_reason is None:
        stop_reason = "tool_use" if any(b["type"] == "tool_use" for b in blocks.values()) else "end_turn"

    input_tokens = sum(
        _estimate_tokens(_tool_result_text(m.get("content")))
        for m in chat_messages
        if m.get("content")
    )
    output_tokens = sum(
        _estimate_tokens(b.get("text", "") or b.get("_json", ""))
        for b in blocks.values()
    )

    return AnthropicResponse(
        id=f"msg_{uuid.uuid4().hex[:24]}",
        content=content,
        model=body.model,
        stop_reason=stop_reason,
        usage=AnthropicUsage(input_tokens=input_tokens, output_tokens=max(1, output_tokens)),
    )


# ── Model metadata for Aider ────────────────────────────────────────────────

@router.get("/.aider.model.metadata.json")
async def aider_model_metadata():
    """Aider model metadata endpoint."""
    return {
        "gpt-5.2-high": {"max_input_tokens": 128000, "max_output_tokens": 16384, "litellm_provider": "openai", "mode": "chat"},
        "gpt-5.5-instant": {"max_input_tokens": 128000, "max_output_tokens": 16384, "litellm_provider": "openai", "mode": "chat"},
        "claude-sonnet-5": {"max_input_tokens": 200000, "max_output_tokens": 8192, "litellm_provider": "anthropic", "mode": "chat"},
        "o3-2025-04-16": {"max_input_tokens": 200000, "max_output_tokens": 4096, "litellm_provider": "openai", "mode": "chat"},
        "gemini-3.1-pro-preview": {"max_input_tokens": 2000000, "max_output_tokens": 8192, "litellm_provider": "vertex_ai", "mode": "chat"},
        "gemini-3.6-flash": {"max_input_tokens": 1000000, "max_output_tokens": 8192, "litellm_provider": "vertex_ai", "mode": "chat"},
        "qwen3.7-max": {"max_input_tokens": 128000, "max_output_tokens": 4096, "litellm_provider": "openai", "mode": "chat"},
        "qwen3.7-plus": {"max_input_tokens": 128000, "max_output_tokens": 4096, "litellm_provider": "openai", "mode": "chat"},
        "mistral-large-3": {"max_input_tokens": 128000, "max_output_tokens": 4096, "litellm_provider": "openai", "mode": "chat"},
        "claude-fable-5": {"max_input_tokens": 200000, "max_output_tokens": 8192, "litellm_provider": "anthropic", "mode": "chat"},
        "claude-fable-5.1": {"max_input_tokens": 200000, "max_output_tokens": 8192, "litellm_provider": "anthropic", "mode": "chat"},
        "sol": {"max_input_tokens": 128000, "max_output_tokens": 8192, "litellm_provider": "openai", "mode": "chat"},
        "terra": {"max_input_tokens": 128000, "max_output_tokens": 8192, "litellm_provider": "openai", "mode": "chat"},
        "glm-5.2": {"max_input_tokens": 128000, "max_output_tokens": 8192, "litellm_provider": "openai", "mode": "chat"},
        "kimi-k3": {"max_input_tokens": 128000, "max_output_tokens": 8192, "litellm_provider": "openai", "mode": "chat"},
    }
