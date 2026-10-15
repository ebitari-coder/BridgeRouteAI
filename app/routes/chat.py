import json
import uuid
import time
import logging
import asyncio
from datetime import datetime, timezone
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, JSONResponse
from app.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
    UsageInfo,
    LegacyCompletionRequest,
    LegacyCompletionResponse,
    LegacyCompletionChoice,
)
from app.provider_utils import SamplingParams
from app.providers import (
    get_all_models,
    ensure_catalogs,
    smart_stream,
    smart_collect,
    get_provider_health,
    AGENTIC_MODEL,
    AGENTIC_MODELS,
)


logger = logging.getLogger(__name__)
router = APIRouter(tags=["Chat"])


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _sampling_from(body) -> SamplingParams | None:
    """Sampling controls to forward upstream for an OpenAI-style request.

    Returns ``None`` when the caller specified nothing, so every upstream
    default is preserved.
    """
    sampling = SamplingParams(
        temperature=body.temperature,
        top_p=body.top_p,
        max_tokens=body.max_tokens or getattr(body, "max_completion_tokens", None),
        stop=body.stop,
        tool_choice=body.tool_choice,
    )
    return sampling if sampling.has_any() else None


def _openai_error(message: str, err_type: str = "invalid_request_error", code: str = None, status: int = 400):
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type, "param": None, "code": code}},
    )


async def _resolve_request_combo(request: Request, body) -> tuple[object | None, str]:
    """Find the combo this request belongs to, and how it was bound.

    Binding order (first match wins):

    1. ``key``   — the API key is scoped to a combo (``api_keys.combo_id``).
    2. ``name``  — ``body.model`` is a combo's name: the name is a routable id.
    3. ``model`` — ``body.model`` appears in an active combo's ``models[]``.

    ``name`` and ``key`` are explicit addresses, so the combo fully controls
    execution. ``model`` is only a hint: it selects the engine (agentic or not)
    but a *group* combo matched this way must not hijack a request that named a
    real model — that request keeps its ordinary provider routing.
    """
    combo_id = getattr(request.state, "combo_id", None)
    if combo_id:
        from app.database import async_session
        from app.crud import get_router_combo
        try:
            async with async_session() as db:
                combo = await get_router_combo(db, combo_id)
            if combo is not None:
                return combo, "key"
        except Exception:
            logger.warning("Failed to load combo %s for API key", combo_id, exc_info=True)

    from app.providers import resolve_combo_ref, find_combo_for_model

    entry = resolve_combo_ref(body.model)
    binding = "name"
    if entry is None:
        entry = find_combo_for_model(body.model)
        binding = "model"
    if entry is None:
        return None, ""

    from app.database import async_session
    from app.crud import get_router_combo
    try:
        async with async_session() as db:
            return await get_router_combo(db, entry["id"]), binding
    except Exception:
        logger.warning("Failed to load combo %s", entry["id"], exc_info=True)
        return None, ""


def _combo_entry(combo) -> dict:
    """Index-shaped view of a combo row for the routing strategies."""
    pairs = [
        (m.get("provider"), m.get("model_id"))
        for m in (combo.models or [])
        if isinstance(m, dict) and m.get("provider") and m.get("model_id")
    ]
    return {
        "id": combo.id,
        "name": combo.name,
        "models": pairs,
        "routing_mode": combo.routing_mode or "group",
        "routing_strategy": combo.routing_strategy or "fallback",
        "judge_model": combo.judge_model,
    }


def _combo_config(combo) -> tuple[list, dict]:
    """(models, engine config) for an agentic combo, with safe defaults."""
    models = [
        (m.get("provider"), m.get("model_id"))
        for m in (combo.models or [])
        if isinstance(m, dict) and m.get("provider") and m.get("model_id")
    ] if combo is not None else []

    # When the combo will run a tool loop, lead with a model that actually emits
    # tool_calls. Agnes' 3.0-flash advertises tool support but ignores the
    # `tools` payload in practice, so a combo listing it first silently drops
    # tool execution. The reordering is capability-driven and only touches Agnes
    # entries — every other provider keeps its declared position.
    tool_calling = (combo.tool_calling if combo.tool_calling is not None else True) if combo is not None else True
    if tool_calling and models:
        try:
            from app.agnes import rank_combo_models
            models = rank_combo_models(models, need_tools=True)
        except Exception:
            logger.warning("Failed to rank combo models for tool use", exc_info=True)

    config = {
        "combo_id": combo.id if combo is not None else "",
        "max_iterations": (combo.max_iterations or 5) if combo is not None else 5,
        "compression_target_tokens": (combo.compression_target_tokens or 4000) if combo is not None else 4000,
        "deterministic": (combo.deterministic if combo.deterministic is not None else True) if combo is not None else True,
        "tool_calling": tool_calling,
        "loop_strategy": (combo.loop_strategy or "sequential") if combo is not None else "sequential",
        "load_spread": bool(getattr(combo, "load_spread", False)) if combo is not None else False,
        "monthly_token_budget": (combo.monthly_token_budget or 0) if combo is not None else 0,
        "agentic_enabled": combo is not None and bool(combo.agentic_enabled),
        "reflection_enabled": _reflection_enabled(),
        "reflection_threshold": _reflection_threshold(),
    }
    if not models:
        models = list(AGENTIC_MODELS)
        config["combo_id"] = ""
    return models, config


def _reflection_enabled() -> bool:
    try:
        from app.config import settings
        return bool(settings.reflection_enabled)
    except Exception:
        return True


def _reflection_threshold() -> float:
    try:
        from app.config import settings
        return float(settings.reflection_threshold)
    except Exception:
        return 0.55


async def _merge_tools(body_tools, combo_id: str, config: dict) -> list[dict] | None:
    """Merge request tools with the combo's ``tools_allowed`` allow-list.

    The combo's ``tools_allowed`` is the authority over what tools the model
    may see. The effective tool set is the intersection of:

    * the combo allow-list (``None`` = unrestricted, ``[]`` = deny all), and
    * the client-supplied ``tools`` (``None`` = not restricted by the request).

    Intersection with ``None`` yields the other set; intersection of two
    lists is their overlap. An empty result means no tools are advertised —
    which is the correct outcome for an explicit empty allow-list, a
    contradictory request, or a DB failure loading the policy (fail closed).

    Returns ``None`` when no tools should be advertised.
    """
    from app.tools_registry import (
        get_tools_for_combo,
        get_tool_description,
        tool_schemas,
    )

    # 1. Resolve the combo's allow-list. ``get_tools_for_combo`` returns ``[]``
    #    both for an explicit empty allow-list and for a DB failure — either
    #    way, no tools should be advertised to the model.
    allowed_names: list[str] | None = None
    if combo_id:
        try:
            combo_schemas = await get_tools_for_combo(combo_id)
            allowed_names = (
                [t["function"]["name"] for t in combo_schemas]
                if combo_schemas
                else []
            )
        except Exception:
            logger.warning(
                "Failed to load tools_allowed for combo %s — denying all tools",
                combo_id, exc_info=True,
            )
            allowed_names = []

    # 2. Resolve the client-supplied tools (None = not restricted by request).
    #    ``body_tools`` may be pydantic Tool objects or already-serialized dicts.
    client_names: list[str] | None = None
    if body_tools:
        client_names = []
        for t in body_tools:
            fn = getattr(t, "function", None)
            if fn is not None:
                name = getattr(fn, "name", None)
            else:
                name = (t.get("function") or {}).get("name")
            if name:
                client_names.append(name)

    # 3. Intersect. None means "all registered"; intersection with None is
    #    the other set; intersection of two lists is their overlap.
    if allowed_names is None and client_names is None:
        effective: list[str] | None = None
    elif allowed_names is None:
        effective = client_names
    elif client_names is None:
        effective = allowed_names
    else:
        effective = [n for n in client_names if n in allowed_names]

    # 4. Build the OpenAI-shaped schema list.
    if not config.get("tool_calling", True):
        return None

    if effective is None:
        schemas = [
            {
                "type": "function",
                "function": {
                    "name": n,
                    "description": get_tool_description(n),
                    "parameters": s,
                },
            }
            for n, s in tool_schemas.items()
        ]
    else:
        schemas = []
        for n in effective:
            s = tool_schemas.get(n)
            if s is None:
                continue
            schemas.append({
                "type": "function",
                "function": {
                    "name": n,
                    "description": get_tool_description(n),
                    "parameters": s,
                },
            })

    return schemas or None


async def _load_session(session_id: str, combo_id: str):
    if not session_id:
        return None
    try:
        from app.database import async_session
        from app.crud import get_agentic_session
        async with async_session() as db:
            return await get_agentic_session(db, session_id)
    except Exception:
        logger.warning("Failed to load session %s", session_id, exc_info=True)
        return None


async def _save_session(session_id: str, combo_id: str, messages: list, iteration_index: int, step_history: list):
    if not session_id:
        return
    try:
        from app.database import async_session
        from app.crud import create_agentic_session, update_agentic_session, get_agentic_session
        async with async_session() as db:
            row = await get_agentic_session(db, session_id)
            if row:
                await update_agentic_session(
                    db, session_id,
                    messages=messages,
                    iteration_index=iteration_index,
                    step_history=step_history,
                )
            else:
                await create_agentic_session(
                    db, session_id=session_id, combo_id=combo_id,
                    messages=messages, iteration_index=iteration_index,
                    step_history=step_history,
                )
            await db.commit()
    except Exception:
        logger.warning("Failed to persist session %s", session_id, exc_info=True)


async def _record_analytics(combo_id: str, request_id: str, result) -> None:
    """Persist agentic analytics after each run (FR-6)."""
    if not combo_id:
        return
    try:
        from app.database import async_session
        from app.crud import record_agentic_analytics
        from app.agentic_engine import estimate_agentic_cost

        stop_counts = {result.stop_reason: 1} if getattr(result, "stop_reason", None) else {}
        tool_counts: dict[str, int] = {}
        model_stats = []
        for step in (result.history or []):
            if step.tool_calls:
                for tc in step.tool_calls:
                    name = (tc.get("function") or {}).get("name") or "unknown"
                    tool_counts[name] = tool_counts.get(name, 0) + 1
            model_stats.append({
                "provider": step.provider,
                "model_id": step.model,
                "calls": 1,
                "success_calls": 1 if step.stop_reason not in ("error", None) else 0,
                "avg_tokens": (step.input_tokens or 0) + (step.output_tokens or 0),
                "cost_estimate": 0.0,
            })
        cost = estimate_agentic_cost(
            [(s.provider, s.model) for s in (result.history or [])] or [("", "")],
            max(1, result.iterations or 1),
            estimated_tokens_per_iteration=max(1, (result.total_tokens or 0) // max(1, result.iterations or 1)),
        )
        month_year = datetime.now(timezone.utc).strftime("%Y-%m")
        record = {
            "requests_count": 1,
            "total_iterations": result.iterations or 0,
            "total_tokens": result.total_tokens or 0,
            "estimated_cost_usd": cost.get("estimated_cost_usd", 0.0),
            "stop_reason_counts": stop_counts,
            "tool_call_counts": tool_counts,
            "model_stats": model_stats,
        }
        async with async_session() as db:
            await record_agentic_analytics(db, combo_id, request_id, month_year, record)
            await db.commit()
    except Exception:
        logger.warning("Failed to record agentic analytics", exc_info=True)


def _escalation_requested(request: Request, body) -> bool:
    """Auto-escalation signal: body.metadata.escalation or X-Escalation header."""
    try:
        meta = getattr(body, "metadata", None) or {}
        if isinstance(meta, dict) and meta.get("escalation"):
            return True
    except Exception:
        pass
    return (request.headers.get("x-escalation", "") or "").strip().lower() in ("1", "true", "yes")


def _prefix_hash(messages: list[dict]) -> str:
    import hashlib
    head = json.dumps(messages[:1], ensure_ascii=False, default=str)[:4000]
    return hashlib.sha1(head.encode()).hexdigest()[:16]


async def _prefix_affinity_reorder(models: list[tuple[str, str]], combo,
                                   messages: list[dict], load_spread: bool) -> tuple[list, str | None]:
    """Prefix-cache affinity: if this combo served an identical conversation
    head recently from one provider, lead with that provider again so
    upstreams with implicit prompt caching reuse KV state (TTFT + cost win).
    Skipped under load_spread (round-robin is the admin's explicit wish).
    """
    if not combo or load_spread or not models:
        return models, None
    try:
        from app.redis_client import get_redis
        r = await get_redis()
        key = f"gw:affinity:{combo.id}:{_prefix_hash(messages)}"
        provider = await r.get(key)
        if provider:
            first = [m for m in models if m[0] == provider]
            rest = [m for m in models if m[0] != provider]
            if first and rest + first != models:
                models = first + rest
        return models, key
    except Exception:
        return models, None


async def _store_prefix_affinity(key: str | None, provider: str | None) -> None:
    if not key or not provider:
        return
    try:
        from app.redis_client import get_redis
        r = await get_redis()
        await r.set(key, provider, ex=600)
    except Exception:
        pass


async def _run_agentic_nonstream(request: Request, body: ChatCompletionRequest, tools_list, combo=None):
    """Non-streaming combo-bound agentic path → ChatCompletionResponse (TR-11.1)."""
    from app.agentic_engine import run_agentic_loop

    models, config = _combo_config(combo)

    session_id = request.headers.get("X-Session-Id") or request.headers.get("x-session-id")
    session_row = await _load_session(session_id, config.get("combo_id", ""))
    messages = [m.model_dump() for m in body.messages]
    start_iteration = 0
    total_budget = None

    if session_row is not None:
        if session_row.messages:
            # Seed with prior history, then append current request messages
            prior = list(session_row.messages)
            # Avoid duplicating identical tail if client re-sends full history
            if prior and messages and prior[-1] == messages[0]:
                messages = prior + messages[1:]
            else:
                messages = prior + messages
        start_iteration = int(session_row.iteration_index or 0)

    max_iter_header = request.headers.get("X-Max-Iterations-Total") or request.headers.get("x-max-iterations-total")
    if max_iter_header:
        try:
            total_budget = int(max_iter_header)
        except ValueError:
            total_budget = None

    request_id = f"req-{uuid.uuid4().hex[:16]}"
    merged_tools = tools_list  # already OpenAI schemas from request; engines also use combo tools
    models, affinity_key = await _prefix_affinity_reorder(
        models, combo, messages, config.get("load_spread", False))

    # If tools_allowed on combo and request tools given, keep request tools; engine filters allow-list
    result = await run_agentic_loop(
            combo_models=models,
            messages=messages,
            max_iterations=config.get("max_iterations", 5),
            compression_target_tokens=config.get("compression_target_tokens", 4000),
            deterministic=config.get("deterministic", True),
            tool_calling=config.get("tool_calling", True),
            combo_id=config.get("combo_id", ""),
            monthly_token_budget=config.get("monthly_token_budget", 0),
            loop_strategy=config.get("loop_strategy", "sequential"),
            escalation=_escalation_requested(request, body),
            load_spread=config.get("load_spread", False),
            tools=merged_tools,
            start_iteration=start_iteration,
            total_iteration_budget=total_budget,
            reflection_enabled=config.get("reflection_enabled", True),
            reflection_threshold=config.get("reflection_threshold", 0.55),
            tool_choice=body.tool_choice,
    )
    await _store_prefix_affinity(
        affinity_key,
        result.history[-1].provider if result.history else None,
    )

    # Persist session after run
    if session_id:
        final_messages = list(messages)
        # Append final assistant content if present
        if result.content:
            final_messages.append({"role": "assistant", "content": result.content})
        new_iteration = start_iteration + (result.iterations or 0)
        step_history = [
            {
                "iteration": s.iteration,
                "model": s.model,
                "provider": s.provider,
                "stop_reason": s.stop_reason,
                "tokens": (s.input_tokens or 0) + (s.output_tokens or 0),
            }
            for s in (result.history or [])
        ]
        await _save_session(session_id, config.get("combo_id", ""), final_messages, new_iteration, step_history)

    await _record_analytics(config.get("combo_id", ""), request_id, result)

    prompt_tokens = sum(_estimate_tokens(m.content or "") for m in body.messages)
    completion_tokens = _estimate_tokens(result.content or "")

    response = ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        model=body.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=result.content or ""),
                finish_reason=result.stop_reason or "stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=max(1, result.total_tokens or (prompt_tokens + completion_tokens)),
        ),
    )
    request.state.tokens_used = response.usage.total_tokens
    return response


async def _agentic_sse_stream(request: Request, body: ChatCompletionRequest, tools_list, include_usage: bool, combo=None):
    """Streaming combo-bound agentic path: OpenAI SSE + agentic_event sidecar (TR-11.2)."""
    from app.agentic_engine import run_agentic_stream

    models, config = _combo_config(combo)

    session_id = request.headers.get("X-Session-Id") or request.headers.get("x-session-id")
    session_row = await _load_session(session_id, config.get("combo_id", ""))
    messages = [m.model_dump() for m in body.messages]
    start_iteration = 0
    total_budget = None

    if session_row is not None:
        if session_row.messages:
            prior = list(session_row.messages)
            if prior and messages and prior[-1] == messages[0]:
                messages = prior + messages[1:]
            else:
                messages = prior + messages
        start_iteration = int(session_row.iteration_index or 0)

    max_iter_header = request.headers.get("X-Max-Iterations-Total") or request.headers.get("x-max-iterations-total")
    if max_iter_header:
        try:
            total_budget = int(max_iter_header)
        except ValueError:
            total_budget = None

    request_id = f"req-{uuid.uuid4().hex[:16]}"
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    final_content_parts: list[str] = []
    final_stop = "max_iterations"
    total_tokens = 0
    iterations_ran = 0
    compressed_steps = 0
    max_iteration_seen = start_iteration

    def _chunk(delta: dict, finish_reason=None, usage=None, agentic_event=None) -> str:
        payload = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": body.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if usage is not None:
            payload["usage"] = usage
        if agentic_event is not None:
            payload["agentic_event"] = agentic_event
        return "data: " + json.dumps(payload) + "\n\n"

    yield _chunk({"role": "assistant"})

    iteration_steps: list = []
    # Emit a short progress line the first time a tool runs, so a slow
    # agentic request still shows visible output before the client-side
    # read timeout expires. Suppressed once real answer content streams.
    progress_sent = False
    true_models, affinity_key = await _prefix_affinity_reorder(
        models, combo, messages, config.get("load_spread", False))
    used_provider = None
    async for ev in run_agentic_stream(
        combo_models=true_models,
        messages=messages,
        max_iterations=config.get("max_iterations", 5),
        compression_target_tokens=config.get("compression_target_tokens", 4000),
        deterministic=config.get("deterministic", True),
        tool_calling=config.get("tool_calling", True),
        combo_id=config.get("combo_id", ""),
        monthly_token_budget=config.get("monthly_token_budget", 0),
        loop_strategy=config.get("loop_strategy", "sequential"),
        escalation=_escalation_requested(request, body),
        tools=tools_list,
        start_iteration=start_iteration,
        total_iteration_budget=total_budget,
        reflection_enabled=config.get("reflection_enabled", True),
        reflection_threshold=config.get("reflection_threshold", 0.55),
        tool_choice=body.tool_choice,
        load_spread=config.get("load_spread", False),
    ):
        etype = ev.get("type")
        if etype == "iteration" and ev.get("provider"):
            used_provider = ev.get("provider")
        if etype == "iteration":
            content = ev.get("content") or ""
            # Only surface an iteration's content when it is the real answer:
            # a final synthesis pass, or an iteration that produced text with
            # no tool calls (nothing more to run). Intermediate tool-calling
            # iterations carry only preamble like "Mapping your workspace…",
            # which must not be streamed to the client.
            is_final_answer = bool(ev.get("synthesis")) or not ev.get("tool_calls")
            if content and is_final_answer:
                final_content_parts.append(content)
                progress_sent = True
                yield _chunk({"content": content}, agentic_event=ev)
            else:
                yield _chunk({}, agentic_event=ev)
            iterations_ran = ev.get("iteration", iterations_ran) + 1 if isinstance(ev.get("iteration"), int) else iterations_ran
            max_iteration_seen = max(max_iteration_seen, ev.get("iteration", 0) or 0)
            total_tokens += (ev.get("input_tokens") or 0) + (ev.get("output_tokens") or 0)
            iteration_steps.append(type("Step", (), dict(ev))())
        elif etype == "complete":
            final_stop = ev.get("stop_reason", final_stop)
            total_tokens = ev.get("total_tokens", total_tokens) or total_tokens
            iterations_ran = ev.get("iterations", iterations_ran) or iterations_ran
            compressed_steps = ev.get("compressed_steps", compressed_steps) or compressed_steps
            yield _chunk({}, agentic_event=ev)
        else:
            # reflection, tool_execution, compression, budget_alert, error, skip
            if etype == "tool_execution" and not progress_sent:
                # First tool run — flush a short progress marker so a slow
                # request isn't a blank bubble if the client times out before
                # the final synthesis answer arrives.
                progress_sent = True
                names = []
                for tc in (ev.get("tool_calls") or []):
                    fn = (tc or {}).get("function") or {}
                    if fn.get("name"):
                        names.append(str(fn["name"]))
                label = ", ".join(names) if names else "tools"
                yield _chunk(
                    {"content": f"_Working — running {label}…_"},
                    agentic_event=ev,
                )
                continue
            yield _chunk({}, agentic_event=ev)

    # Persist session
    if session_id:
        final_messages = list(messages)
        final_text = "".join(final_content_parts)
        if final_text:
            final_messages.append({"role": "assistant", "content": final_text})
        step_history = [{"last_event": final_stop}]
        await _save_session(session_id, config.get("combo_id", ""), final_messages, max_iteration_seen + 1, step_history)

    # Record analytics
    class _FakeResult:
        pass
    fake = type("R", (), {})()
    fake.content = "".join(final_content_parts)
    fake.stop_reason = final_stop
    fake.iterations = iterations_ran
    fake.total_tokens = total_tokens
    fake.history = iteration_steps
    await _record_analytics(config.get("combo_id", ""), request_id, fake)

    await _store_prefix_affinity(affinity_key, used_provider)
    if include_usage:
        prompt_tokens = sum(_estimate_tokens(m.content or "") for m in body.messages)
        completion_tokens = _estimate_tokens("".join(final_content_parts))
        yield _chunk({}, usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": max(1, total_tokens or (prompt_tokens + completion_tokens)),
        })

    yield "data: [DONE]\n\n"


def _text_completion_response(
    request: Request,
    body,
    full_text: str,
    actual_model: str,
    model_field: str | None = None,
) -> ChatCompletionResponse:
    """Build a non-streaming ChatCompletionResponse from collected text.

    ``model_field`` overrides the reported model — combo-routed requests echo
    the combo name the client asked for, since the answer may have come from
    any member of the group (or the fusion judge).
    """
    prompt_tokens = sum(_estimate_tokens(m.content or "") for m in body.messages)

    # The upstream may wrap tool calls as JSON: {"content", "tool_calls"}.
    tool_calls_result = None
    content_text = full_text
    try:
        parsed_text = json.loads(full_text)
        if isinstance(parsed_text, dict) and "tool_calls" in parsed_text:
            content_text = parsed_text.get("content", "")
            tool_calls_result = parsed_text.get("tool_calls")
    except (json.JSONDecodeError, TypeError):
        pass

    completion_tokens = _estimate_tokens(content_text)

    assistant_msg = ChatMessage(role="assistant", content=content_text or None)
    if tool_calls_result:
        assistant_msg.tool_calls = tool_calls_result

    response = ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        model=model_field or actual_model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=assistant_msg,
                finish_reason="tool_calls" if tool_calls_result else "stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
    request.state.tokens_used = prompt_tokens + completion_tokens
    return response


@router.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest):
    """OpenAI-compatible chat completion with intelligent failover.

    A combo is resolved first and its ``routing_mode`` picks the engine:
    ``agentic`` runs the multi-iteration tool loop, ``group`` runs the combo's
    strategy (fallback / round robin / fusion). Models that belong to no combo
    keep ordinary provider failover.
    """
    if not body.messages:
        return _openai_error("messages is required", "invalid_request_error")

    include_usage = body.stream_options and body.stream_options.include_usage
    tools_list = [t.model_dump() for t in body.tools] if body.tools else None

    # 011: "agentic" used to be a virtual model id. Refuse it explicitly —
    # otherwise it would silently fall through to the FreeModels default plan.
    if body.model == AGENTIC_MODEL:
        return _openai_error(
            "'agentic' is no longer a model. Create a combo, set "
            "routing_mode='agentic' (or a group strategy), and request it by "
            "the combo's name.",
            "invalid_request_error",
            "agentic_model_removed",
            400,
        )

    sampling = _sampling_from(body)
    combo, binding = await _resolve_request_combo(request, body)

    # ── Agentic combo: the loop engine (TR-11) ─────────────────────────────
    if combo is not None and combo.is_agentic:
        tool_calling = bool(combo.tool_calling) if combo.tool_calling is not None else True
        # The combo's tools_allowed is the authority over what the model may
        # see. When the request carried no tools we seed the list from the
        # combo allow-list; when it did, we intersect the two so a client can
        # never widen the combo's sandbox (FR-2). A combo with no allow-list
        # configured (None) leaves the client's tools unrestricted.
        if tool_calling:
            combo_config = {"tool_calling": True}
            tools_list = await _merge_tools(body.tools, combo.id, combo_config)
        if body.stream:
            return StreamingResponse(
                _agentic_sse_stream(
                    request, body, tools_list,
                    include_usage or bool(body.stream_options), combo,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "X-Model-Requested": body.model,
                },
            )
        return await _run_agentic_nonstream(request, body, tools_list, combo)

    # ── Group combo, addressed by name or combo-scoped key ─────────────────
    # A bare model-id match only hinted at the engine: the caller named a real
    # model, so that request keeps its own provider routing.
    group_entry = _combo_entry(combo) if (
        combo is not None and binding in ("key", "name")
    ) else None

    if group_entry is not None:
        from app.routing_strategies import group_stream, group_collect

        if body.stream:
            return StreamingResponse(
                group_stream(
                    group_entry, body.messages,
                    bool(include_usage or body.stream_options),
                    tools=tools_list, sampling=sampling,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "X-Model-Requested": body.model,
                },
            )
        full_text, actual_model, error = await group_collect(
            group_entry, body.messages, tools=tools_list, sampling=sampling,
        )
        if error:
            return _openai_error(
                error.get("message", "Unknown error"),
                error.get("type", "server_error"),
                error.get("code"),
                429 if "exhausted" in error.get("message", "").lower() else 502,
            )
        return _text_completion_response(
            request, body, full_text, actual_model, group_entry["name"],
        )
    else:
        if body.stream:
            return StreamingResponse(
                smart_stream(body.model, body.messages, include_usage, tools=tools_list, sampling=sampling),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "X-Model-Requested": body.model,
                },
            )

    # Non-streaming with failover
    full_text, actual_model, error = await smart_collect(body.model, body.messages, tools=tools_list, sampling=sampling)

    if error:
        return _openai_error(
            error.get("message", "Unknown error"),
            error.get("type", "server_error"),
            error.get("code"),
            429 if "exhausted" in error.get("message", "").lower() or "rate" in error.get("type", "").lower() else 502,
        )

    return _text_completion_response(request, body, full_text, actual_model)


# ── Legacy Completions ──────────────────────────────────────────────────────

@router.post("/v1/completions")
async def legacy_completions(request: Request, body: LegacyCompletionRequest):
    """Legacy completions with failover."""
    messages = [ChatMessage(role="user", content=body.prompt)]
    sampling = SamplingParams(
        temperature=body.temperature,
        top_p=body.top_p,
        max_tokens=body.max_tokens,
        stop=body.stop,
    )

    if body.stream:
        return StreamingResponse(
            smart_stream(body.model, messages, sampling=sampling),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    full_text, actual_model, error = await smart_collect(body.model, messages, sampling=sampling)
    if error:
        return _openai_error(error.get("message", "Unknown error"), "server_error", None, 502)

    prompt_tokens = _estimate_tokens(body.prompt)
    completion_tokens = _estimate_tokens(full_text)
    request.state.tokens_used = prompt_tokens + completion_tokens

    return LegacyCompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        model=actual_model,
        choices=[LegacyCompletionChoice(text=full_text, index=0, finish_reason="stop")],
        usage=UsageInfo(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=prompt_tokens + completion_tokens),
    )


# ── Models ──────────────────────────────────────────────────────────────────

@router.get("/v1/models", response_model=None)
async def list_models():
    """List all supported models from all providers with health status."""
    from app.models import ModelListResponse, ModelInfo
    await ensure_catalogs()
    all_models = get_all_models()
    return ModelListResponse(data=[
        ModelInfo(
            id=m["id"],
            created=m["created"],
            owned_by=m["owned_by"],
            tools_supported=m.get("tools_supported", False),
        )
        for m in all_models
    ])


@router.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    from app.models import ModelInfo
    for m in get_all_models():
        if m["id"] == model_id:
            return ModelInfo(
                id=m["id"],
                created=m["created"],
                owned_by=m["owned_by"],
                tools_supported=m.get("tools_supported", False),
            )
    return _openai_error(f"Model '{model_id}' not found", "invalid_request_error", None, 404)


@router.delete("/v1/models/{model_id}")
async def delete_model(model_id: str):
    return _openai_error("Models cannot be deleted.", "invalid_request_error", None, 400)


# ── Provider Health ─────────────────────────────────────────────────────────

@router.get("/v1/provider-health")
async def provider_health():
    """Check health status of all upstream providers."""
    return get_provider_health()
