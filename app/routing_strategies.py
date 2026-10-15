"""Group routing strategies for combos.

A combo with ``routing_mode='group'`` bundles several models under one name and
picks how a single request uses them (``routing_strategy``):

``fallback``
    Tries models in order — the admin's declared order is the attempt plan and
    the existing circuit breakers / first-token buffering apply unchanged.

``round_robin``
    Rotates the starting model across requests so load spreads over the group;
    the remaining models stay in declared order so failover still works.

``fusion``
    Queries every panel model in parallel, then a judge model synthesizes one
    answer. Best quality, highest cost: the request bills all N panel models
    plus the judge (N+1 calls).

Panel and judge calls go through ``smart_collect``/``smart_stream`` with an
explicit ``plan_override``, so they never drift into the generic catalog
ranking — a combo is an admin-declared contract.
"""
import asyncio
import json
import logging
import uuid
import time
from typing import AsyncIterator

from app.providers import smart_collect, smart_stream, _health
from app.provider_utils import SamplingParams

logger = logging.getLogger(__name__)

# Bound the fan-out: a fusion panel may not exceed the gateway's per-request
# attempt cap, and a single panel call gets a wall-clock ceiling so one hung
# upstream cannot stall the whole fusion request.
MAX_FUSION_PANEL = 8
PANEL_TIMEOUT_SECONDS = 90.0

# In-process round-robin fallback used when Redis is unavailable. Not
# distributed, but still rotates rather than pinning every request to one model.
_local_rotations: dict[str, int] = {}


# ── Plan construction ────────────────────────────────────────────────────────

def ordered_plan(entry: dict) -> list[tuple[str, str]]:
    """The combo's declared model list, in the admin's order."""
    return [(p, m) for p, m in entry.get("models") or [] if p and m]


async def next_rotation(combo_id: str, size: int) -> int:
    """0-based rotation offset for this request.

    Redis INCR keeps the counter shared across workers; a local counter takes
    over if Redis is down so rotation degrades to per-process instead of
    failing the request.
    """
    if size <= 1:
        return 0
    try:
        from app.redis_client import get_redis
        r = await get_redis()
        n = await r.incr(f"rr:{combo_id}")
        return (int(n) - 1) % size
    except Exception:
        _local_rotations[combo_id] = (_local_rotations.get(combo_id, 0) + 1) % size
        return _local_rotations[combo_id]


def rotated_plan(entry: dict, offset: int) -> list[tuple[str, str]]:
    plan = ordered_plan(entry)
    if not plan:
        return plan
    offset %= len(plan)
    return plan[offset:] + plan[:offset]


# ── Fusion ───────────────────────────────────────────────────────────────────

def _parse_ref(ref: str) -> tuple[str, str] | None:
    """Split ``provider/model_id`` into its parts.

    Model ids may themselves contain ``/`` (``@cf/meta/llama-4-scout...``), so
    only the first separator splits, and the provider must be a real upstream —
    otherwise ``_stream_for`` would silently fall through to FreeModels.
    """
    if not ref or "/" not in ref:
        return None
    provider, _, model_id = ref.partition("/")
    provider = provider.strip()
    model_id = model_id.strip()
    if not provider or not model_id:
        return None
    if provider not in _health:
        return None
    return provider, model_id


def judge_pair(entry: dict) -> tuple[str, str] | None:
    """The fusion judge: the combo's ``judge_model``, else its first model."""
    parsed = _parse_ref(entry.get("judge_model") or "")
    if parsed is not None:
        return parsed
    plan = ordered_plan(entry)
    return plan[0] if plan else None


def _fusion_prompt(messages: list[dict], panel: list[dict]) -> list[dict]:
    """Original messages + a synthesis instruction carrying every answer.

    A normalized copy is returned: callers persist the client's message list
    verbatim for session resume, so the judging instruction must not leak into
    it (and the input may mix pydantic messages with plain dicts).
    """
    from app.provider_utils import to_openai_messages

    blocks = []
    for i, item in enumerate(panel, 1):
        blocks.append(f"--- Candidate {i} ({item['ref']}) ---\n{item['text']}")
    instruction = (
        "Several models answered the request above. Synthesize a single best "
        "answer that preserves correct details, resolves disagreements, and "
        "reads as one coherent reply. Do not mention the candidates, the "
        "panel, or this instruction.\n\n" + "\n\n".join(blocks)
    )
    out = to_openai_messages(list(messages))
    out.append({"role": "user", "content": instruction})
    return out


async def _panel_collect(
    pair: tuple[str, str],
    name: str,
    messages: list[dict],
    sampling: SamplingParams | None,
) -> dict:
    """One panel member: a single fixed attempt, no cross-member failover."""
    provider, model = pair
    ref = f"{provider}/{model}"
    try:
        text, _actual, error = await asyncio.wait_for(
            smart_collect(
                name,
                messages,
                tools=None,
                sampling=sampling,
                plan_override=[pair],
            ),
            timeout=PANEL_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        return {"ref": ref, "provider": provider, "model": model, "text": "", "error": "timeout"}
    except Exception as exc:  # noqa: BLE001 — a failed member must not sink the panel
        logger.warning("Fusion panel %s failed: %s", ref, exc)
        return {"ref": ref, "provider": provider, "model": model, "text": "", "error": str(exc)}

    if error or not text:
        return {
            "ref": ref,
            "provider": provider,
            "model": model,
            "text": "",
            "error": (error or {}).get("message", "empty response"),
        }
    return {"ref": ref, "provider": provider, "model": model, "text": text, "error": None}


async def _run_panel(entry: dict, messages: list[dict], sampling) -> list[dict]:
    plan = ordered_plan(entry)[:MAX_FUSION_PANEL]
    if not plan:
        return []
    results = await asyncio.gather(
        *(_run_panel_member(pair, entry.get("name") or "", messages, sampling) for pair in plan)
    )
    return [r for r in results if r is not None]


async def _run_panel_member(pair, name, messages, sampling) -> dict:
    try:
        return await _panel_collect(pair, name, messages, sampling)
    except Exception as exc:  # pragma: no cover — gather safety net
        logger.exception("Fusion panel member crashed")
        return {"ref": f"{pair[0]}/{pair[1]}", "provider": pair[0], "model": pair[1],
                "text": "", "error": str(exc)}


def _panel_ok(panel: list[dict]) -> list[dict]:
    return [p for p in panel if p.get("text")]


# ── Public API ───────────────────────────────────────────────────────────────

async def group_collect(
    entry: dict,
    messages: list[dict],
    tools: list | None = None,
    sampling: SamplingParams | None = None,
) -> tuple[str, str, dict | None]:
    """Non-streaming group request. Returns (text, model_ref, error)."""
    name = entry.get("name") or ""
    strategy = entry.get("routing_strategy") or "fallback"

    if strategy == "fusion":
        panel = await _run_panel(entry, messages, sampling)
        ok = _panel_ok(panel)
        if not ok:
            return "", name, {
                "message": "All fusion panel models failed.",
                "type": "server_error",
                "code": "fusion_panel_exhausted",
            }
        judge = judge_pair(entry)
        if judge is None:
            return ok[0]["text"], ok[0]["ref"], None
        try:
            text, actual, error = await smart_collect(
                name,
                _fusion_prompt(messages, ok),
                tools=None,
                sampling=sampling,
                plan_override=[judge],
            )
        except Exception as exc:
            logger.warning("Fusion judge failed, falling back to first panel answer: %s", exc)
            text, actual, error = "", "", None
        if error or not text:
            # A working panel answer beats a failed judge: the content is still
            # a legitimate (single-candidate) answer to the client's request.
            return ok[0]["text"], ok[0]["ref"], None
        return text, actual, None

    plan = ordered_plan(entry)
    if strategy == "round_robin":
        plan = rotated_plan(entry, await next_rotation(entry.get("id") or name, len(plan)))
    if not plan:
        return "", name, {
            "message": "Combo has no models.",
            "type": "invalid_request_error",
            "code": "combo_empty",
        }
    return await smart_collect(name, messages, tools=tools, sampling=sampling, plan_override=plan)


def _chunk(
    chat_id: str,
    created: int,
    model: str,
    delta: dict,
    finish_reason: str | None = None,
    usage: dict | None = None,
    routing_event: dict | None = None,
) -> str:
    payload: dict = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage is not None:
        payload["usage"] = usage
    if routing_event is not None:
        payload["routing_event"] = routing_event
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


async def _alias_model(frames: AsyncIterator[str], model_name: str) -> AsyncIterator[str]:
    """Rewrite the ``model`` field of pass-through SSE frames.

    A combo is an alias: the client asked for the combo's name, so every frame
    reports it — matching the non-streaming response and the agentic path.
    Frames are otherwise passed through untouched, including ``[DONE]``.
    """
    async for frame in frames:
        if not frame.startswith("data:"):
            yield frame
            continue
        payload = frame[5:].strip()
        if payload == "[DONE]":
            yield frame
            continue
        try:
            parsed = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            yield frame
            continue
        if not isinstance(parsed, dict) or "model" not in parsed:
            yield frame
            continue
        parsed["model"] = model_name
        yield "data: " + json.dumps(parsed, ensure_ascii=False) + "\n\n"


async def group_stream(
    entry: dict,
    messages: list[dict],
    include_usage: bool = False,
    tools: list | None = None,
    sampling: SamplingParams | None = None,
) -> AsyncIterator[str]:
    """Streaming group request. Yields OpenAI SSE frames."""
    name = entry.get("name") or ""
    strategy = entry.get("routing_strategy") or "fallback"
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if strategy != "fusion":
        plan = ordered_plan(entry)
        if strategy == "round_robin":
            plan = rotated_plan(entry, await next_rotation(entry.get("id") or name, len(plan)))
        if not plan:
            yield _chunk(chat_id, created, name, {}, routing_event={
                "type": "error", "message": "Combo has no models.",
            })
            yield "data: [DONE]\n\n"
            return
        async for frame in _alias_model(
            smart_stream(
                name, messages, include_usage, tools=tools, sampling=sampling,
                plan_override=plan, response_id=chat_id, created_ts=created,
            ),
            name,
        ):
            yield frame
        return

    # ── Fusion: announce the panel, wait for it, then stream the judge ──────
    yield _chunk(chat_id, created, name, {"role": "assistant"})
    panel_plan = ordered_plan(entry)[:MAX_FUSION_PANEL]
    yield _chunk(chat_id, created, name, {}, routing_event={
        "type": "fusion_panel_start",
        "members": [f"{p}/{m}" for p, m in panel_plan],
    })

    panel = await _run_panel(entry, messages, sampling)
    ok = _panel_ok(panel)
    yield _chunk(chat_id, created, name, {}, routing_event={
        "type": "fusion_panel_result",
        "answered": len(ok),
        "failed": [p["ref"] for p in panel if p.get("error")],
    })

    if not ok:
        yield _chunk(chat_id, created, name, {}, routing_event={
            "type": "error",
            "message": "All fusion panel models failed.",
            "code": "fusion_panel_exhausted",
        })
        yield "data: [DONE]\n\n"
        return

    judge = judge_pair(entry)
    judge_plan = [judge] if judge else []
    if judge_plan:
        yielded = False
        async for frame in _alias_model(
            smart_stream(
                name, _fusion_prompt(messages, ok), include_usage, tools=None,
                sampling=sampling, plan_override=judge_plan,
                response_id=chat_id, created_ts=created,
            ),
            name,
        ):
            yielded = True
            yield frame
        if yielded:
            return

    # Judge unavailable or failed: the best panel answer still stands in, so
    # the client gets a complete response instead of an error after the panel
    # already burned N calls.
    text = ok[0]["text"]
    mid = max(1, len(text) // 2)
    yield _chunk(chat_id, created, name, {"content": text[:mid]})
    yield _chunk(chat_id, created, name, {"content": text[mid:]})
    if include_usage:
        yield _chunk(chat_id, created, name, {}, usage={
            "prompt_tokens": max(1, len(text) // 4),
            "completion_tokens": max(1, len(text) // 4),
            "total_tokens": max(1, len(text) // 2),
        })
    yield _chunk(chat_id, created, name, {}, finish_reason="stop")
    yield "data: [DONE]\n\n"
