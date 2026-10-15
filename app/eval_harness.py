"""Golden-task eval harness for the agentic engine.

Loads evals/golden_tasks.json, runs each task through run_agentic_loop,
scores outputs (substring + optional regex), and reports
score ≈ success_rate * cost_efficiency. Results stored in Redis with
in-process fallback.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_EVALS_DIR = Path(__file__).resolve().parent.parent / "evals"
_GOLDEN_PATH = _EVALS_DIR / "golden_tasks.json"

_results_memory: dict[str, dict] = {}
_results_order: list[str] = []


@dataclass
class EvalTask:
    id: str
    name: str = ""
    messages: list[dict] = field(default_factory=list)
    expect_substrings: list[str] = field(default_factory=list)
    expect_regex: str | None = None
    tools_allowed: list[str] = field(default_factory=list)
    max_iterations: int = 4
    loop_strategy: str = "sequential"


@dataclass
class EvalTaskResult:
    task_id: str
    name: str
    success: bool
    score: float
    stop_reason: str
    iterations: int
    total_tokens: int
    content: str
    missing: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class EvalRunResult:
    run_id: str
    status: str
    started_at: float
    finished_at: float | None
    task_ids: list[str]
    results: list[EvalTaskResult]
    success_rate: float
    cost_efficiency: float
    score: float
    combo_id: str = ""
    error: str | None = None


def load_golden_tasks(path: str | Path | None = None) -> list[EvalTask]:
    p = Path(path) if path else _GOLDEN_PATH
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("tasks", data if isinstance(data, list) else [])
    tasks: list[EvalTask] = []
    for item in raw:
        tasks.append(
            EvalTask(
                id=str(item.get("id") or f"task-{len(tasks)}"),
                name=item.get("name") or item.get("id") or "",
                messages=list(item.get("messages") or []),
                expect_substrings=list(item.get("expect_substrings") or []),
                expect_regex=item.get("expect_regex"),
                tools_allowed=list(item.get("tools_allowed") or []),
                max_iterations=int(item.get("max_iterations") or 4),
                loop_strategy=str(item.get("loop_strategy") or "sequential"),
            )
        )
    return tasks


def score_task(task: EvalTask, content: str, stop_reason: str = "", total_tokens: int = 0) -> tuple[bool, float, list[str]]:
    """Return (success, score, missing_expectations)."""
    text = content or ""
    missing: list[str] = []
    lower = text.lower()

    for needle in task.expect_substrings:
        if needle.lower() not in lower:
            missing.append(needle)

    if task.expect_regex:
        try:
            if not re.search(task.expect_regex, text, re.IGNORECASE | re.DOTALL):
                missing.append(f"regex:{task.expect_regex}")
        except re.error:
            missing.append(f"invalid_regex:{task.expect_regex}")

    if not task.expect_substrings and not task.expect_regex:
        success = bool(text.strip()) and stop_reason not in ("error", "budget_exceeded")
        if not text.strip():
            missing.append("empty_content")
    else:
        success = not missing

    # Soft score: 1.0 when all expectations met; partial credit for substrings.
    if task.expect_substrings:
        total = len(task.expect_substrings)
        hit = sum(1 for n in task.expect_substrings if n.lower() in lower)
        base = hit / total if total else 0.0
        if task.expect_regex and f"regex:{task.expect_regex}" in missing:
            base = min(base, 0.5)
        success = success and base >= 1.0
        score = base if success else base * 0.8
    else:
        score = 1.0 if success else 0.0

    if not success:
        score = min(score, 0.75)

    return success, round(score, 4), missing


def _cost_efficiency(total_tokens_list: list[int]) -> float:
    """1.0 when free, decays as average tokens grow. Clamped to (0, 1]."""
    if not total_tokens_list:
        return 1.0
    avg = sum(total_tokens_list) / len(total_tokens_list)
    # 0 tokens → 1.0; 2000 tokens → ~0.5; 8000 → ~0.2
    return round(1.0 / (1.0 + avg / 2000.0), 4)


def _task_to_dict(task: EvalTask) -> dict:
    return asdict(task)


def _result_to_dict(result: EvalTaskResult) -> dict:
    return asdict(result)


async def _resolve_models() -> tuple[list[tuple[str, str]], str]:
    try:
        from app.providers import get_active_combo_models
        from app.database import async_session
        from app.crud import get_active_combo

        combo_models = get_active_combo_models("agentic")
        if combo_models:
            combo_id = ""
            try:
                async with async_session() as db:
                    combo = await get_active_combo(db, "agentic")
                    if combo is not None:
                        combo_id = str(combo.id)
            except Exception:
                pass
            return list(combo_models), combo_id
    except Exception:
        logger.warning("Failed to resolve active agentic combo for eval", exc_info=True)

    try:
        from app.routes.chat import AGENTIC_MODELS
        return [(p, m) for p, m in AGENTIC_MODELS], ""
    except Exception:
        return [], ""


async def run_eval_suite(
    task_ids: list[str] | None = None,
    tasks_path: str | Path | None = None,
) -> dict:
    """Run golden tasks and persist a run result. Returns the result dict."""
    from app.agentic_engine import run_agentic_loop

    started = time.time()
    run_id = f"eval-{uuid.uuid4().hex[:12]}"

    try:
        all_tasks = load_golden_tasks(tasks_path)
    except Exception as exc:
        result = EvalRunResult(
            run_id=run_id,
            status="failed",
            started_at=started,
            finished_at=time.time(),
            task_ids=[],
            results=[],
            success_rate=0.0,
            cost_efficiency=0.0,
            score=0.0,
            error=f"Failed to load golden tasks: {exc}",
        )
        payload = _run_to_dict(result)
        await _store_result(run_id, payload)
        return payload

    if task_ids:
        wanted = set(task_ids)
        tasks = [t for t in all_tasks if t.id in wanted]
    else:
        tasks = all_tasks

    models, combo_id = await _resolve_models()
    task_results: list[EvalTaskResult] = []
    tokens_list: list[int] = []
    success_count = 0

    for task in tasks:
        try:
            if not models:
                task_results.append(
                    EvalTaskResult(
                        task_id=task.id,
                        name=task.name,
                        success=False,
                        score=0.0,
                        stop_reason="error",
                        iterations=0,
                        total_tokens=0,
                        content="",
                        error="No agentic models configured",
                    )
                )
                tokens_list.append(0)
                continue

            tools = None
            if task.tools_allowed:
                try:
                    from app.tools_registry import tool_schemas
                    tools = [
                        {"type": "function", "function": {"name": n, "parameters": s}}
                        for n, s in tool_schemas.items()
                        if n in task.tools_allowed
                    ] or None
                except Exception:
                    tools = None

            result = await run_agentic_loop(
                combo_models=models,
                messages=list(task.messages),
                max_iterations=task.max_iterations,
                compression_target_tokens=4000,
                deterministic=True,
                tool_calling=True,
                combo_id=combo_id,
                monthly_token_budget=0,
                loop_strategy=task.loop_strategy,
                tools=tools,
            )
            content = result.content or ""
            success, score, missing = score_task(
                task, content, result.stop_reason or "", result.total_tokens or 0
            )
            if success:
                success_count += 1
            tokens_list.append(int(result.total_tokens or 0))
            task_results.append(
                EvalTaskResult(
                    task_id=task.id,
                    name=task.name,
                    success=success,
                    score=score,
                    stop_reason=result.stop_reason or "",
                    iterations=int(result.iterations or 0),
                    total_tokens=int(result.total_tokens or 0),
                    content=content[:4000],
                    missing=missing,
                )
            )
        except Exception as exc:
            logger.warning("Eval task %s failed: %s", task.id, exc, exc_info=True)
            task_results.append(
                EvalTaskResult(
                    task_id=task.id,
                    name=task.name,
                    success=False,
                    score=0.0,
                    stop_reason="error",
                    iterations=0,
                    total_tokens=0,
                    content="",
                    error=str(exc),
                )
            )
            tokens_list.append(0)

    n = max(1, len(tasks))
    success_rate = round(success_count / n, 4)
    cost_efficiency = _cost_efficiency(tokens_list)
    score = round(success_rate * cost_efficiency, 4)

    run = EvalRunResult(
        run_id=run_id,
        status="complete",
        started_at=started,
        finished_at=time.time(),
        task_ids=[t.id for t in tasks],
        results=task_results,
        success_rate=success_rate,
        cost_efficiency=cost_efficiency,
        score=score,
        combo_id=combo_id,
    )
    payload = _run_to_dict(run)
    await _store_result(run_id, payload)
    return payload


def _run_to_dict(run: EvalRunResult) -> dict:
    return {
        "run_id": run.run_id,
        "status": run.status,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "task_ids": run.task_ids,
        "results": [_result_to_dict(r) for r in run.results],
        "success_rate": run.success_rate,
        "cost_efficiency": run.cost_efficiency,
        "score": run.score,
        "combo_id": run.combo_id,
        "error": run.error,
        "task_count": len(run.results),
    }


async def _store_result(run_id: str, payload: dict) -> None:
    _results_memory[run_id] = payload
    _results_order.append(run_id)
    if len(_results_order) > 50:
        old = _results_order.pop(0)
        _results_memory.pop(old, None)
    try:
        from app.redis_client import get_redis
        r = await get_redis()
        await r.set(f"eval:result:{run_id}", json.dumps(payload, default=str), ex=86400)
        await r.lpush("eval:result:ids", run_id)
        await r.ltrim("eval:result:ids", 0, 49)
    except Exception:
        logger.debug("Eval result Redis store unavailable", exc_info=True)


async def get_eval_result(run_id: str) -> dict | None:
    if run_id in _results_memory:
        return _results_memory[run_id]
    try:
        from app.redis_client import get_redis
        r = await get_redis()
        raw = await r.get(f"eval:result:{run_id}")
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return None


async def list_eval_results(limit: int = 20) -> list[dict]:
    ids: list[str] = list(reversed(_results_order))
    try:
        from app.redis_client import get_redis
        r = await get_redis()
        remote = await r.lrange("eval:result:ids", 0, max(0, limit - 1))
        seen = set(ids)
        for rid in remote:
            if rid not in seen:
                ids.append(rid)
                seen.add(rid)
    except Exception:
        pass

    out: list[dict] = []
    for rid in ids[:limit]:
        item = await get_eval_result(rid)
        if item:
            out.append(
                {
                    "run_id": item.get("run_id", rid),
                    "status": item.get("status"),
                    "started_at": item.get("started_at"),
                    "finished_at": item.get("finished_at"),
                    "task_count": item.get("task_count", len(item.get("results") or [])),
                    "success_rate": item.get("success_rate", 0),
                    "cost_efficiency": item.get("cost_efficiency", 0),
                    "score": item.get("score", 0),
                    "error": item.get("error"),
                }
            )
    return out
