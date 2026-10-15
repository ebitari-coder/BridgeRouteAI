"""Deterministic agentic loop engine with token compression.

The engine runs a loop of model calls and tool executions until the task
is complete or max_iterations is reached. Token compression reduces
context window usage before each upstream call.
"""
import asyncio
import time
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any
from datetime import datetime, timezone
from app.providers import PROVIDER_FREEMODELS
from app.token_budget_manager import TokenBudgetManager, BudgetEnforcer, get_combo_budget_status
from app.db_models import DBTokenBudget, DBBudgetAlert

logger = logging.getLogger(__name__)


def _truncate_tool_content(content: str, limit: int = 500) -> str:
    """Truncate a tool-result body while keeping JSON documents valid.

    Tool results are JSON documents (``_call_tool``/``execute_tool`` wrap
    outputs as ``{"success": ..., "result": ..., "tool": ...}``). Raw
    string-slicing such a document severs it mid-key/mid-string and produces
    malformed JSON the model cannot parse — and the same corrupted payload is
    forwarded verbatim to strict upstream providers, which can reject the
    whole request with a 400. So JSON bodies are truncated *structurally*:
    long string values are clamped in place and the document is re-serialized
    with balanced structure. Plain-text bodies keep the simple slice.

    Returns ``content`` unchanged when it already fits within ``limit``.
    """
    if len(content) <= limit:
        return content

    import json

    try:
        parsed = json.loads(content)
    except (ValueError, TypeError):
        # Not JSON — a plain-text body; slice it as before.
        return content[:limit] + "...[truncated]"

    def _clamp(node):
        if isinstance(node, str):
            if len(node) > limit:
                return node[:limit] + "...[truncated]"
            return node
        if isinstance(node, dict):
            return {k: _clamp(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_clamp(v) for v in node]
        return node

    try:
        return json.dumps(_clamp(parsed), ensure_ascii=False)
    except (ValueError, TypeError):
        return content[:limit] + "...[truncated]"


class CircuitBreaker:
    """Tracks failures per model and skips them when the breaker is open."""

    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._failures: dict[str, list[float]] = {}

    @property
    def failures(self) -> dict[str, list[float]]:
        """Return the failures tracking dict per model."""
        return self._failures

    def record_failure(self, model: str) -> None:
        """Record a failure for the given model."""
        if model not in self._failures:
            self._failures[model] = []
        self._failures[model].append(time.time())
        # Keep only failures within the recovery window
        cutoff = time.time() - self.recovery_timeout
        self._failures[model] = [t for t in self._failures[model] if t > cutoff]

    def record_success(self, model: str) -> None:
        """Clear failures for the given model on success."""
        if model in self._failures:
            self._failures[model] = []

    def is_open(self, model: str) -> bool:
        """Check if the circuit breaker is open for the given model."""
        if model not in self._failures or not self._failures[model]:
            return False
        cutoff = time.time() - self.recovery_timeout
        recent = [t for t in self._failures[model] if t > cutoff]
        return len(recent) >= self.failure_threshold

    def get_consecutive_failures(self, model: str) -> int:
        """Return the number of recent consecutive failures for a model."""
        if model not in self._failures:
            return 0
        cutoff = time.time() - self.recovery_timeout
        return len([t for t in self._failures[model] if t > cutoff])


@dataclass
class AgenticStep:
    iteration: int
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str | None = None
    tool_calls: list[dict] | None = None
    content: str | None = None
    compressed: bool = False


@dataclass
class AgenticResult:
    content: str | None
    stop_reason: str
    iterations: int
    history: list[AgenticStep]
    total_tokens: int = 0
    compressed_steps: int = 0
    reflection_repairs: int = 0
    reflection_score: float | None = None


@dataclass
class BudgetAlertConfig:
    """Configuration for budget alert thresholds."""
    warning_threshold_percent: float = 80.0
    critical_threshold_percent: float = 95.0
    notify_email: str | None = None
    notify_webhook: str | None = None


@dataclass
class BudgetAlert:
    """A budget alert record."""
    level: str  # "warning" or "critical"
    message: str
    combo_id: str
    percentage: float
    tokens_used: int
    tokens_budget: int
    timestamp: str


class BudgetAlertManager:
    """Manages budget alerts for combos."""

    def __init__(self, config: BudgetAlertConfig | None = None):
        self.config = config or BudgetAlertConfig()
        self.alerts: list[BudgetAlert] = []

    def check(self, combo_id: str, tokens_used: int, tokens_budget: int) -> list[BudgetAlert]:
        """Check if budget thresholds are crossed and generate alerts."""
        new_alerts = []
        if tokens_budget <= 0:
            return new_alerts

        percentage = (tokens_used / tokens_budget) * 100

        if percentage >= self.config.critical_threshold_percent:
            alert = BudgetAlert(
                level="critical",
                message=f"Critical: Combo {combo_id} at {percentage:.1f}% budget ({tokens_used}/{tokens_budget})",
                combo_id=combo_id,
                percentage=percentage,
                tokens_used=tokens_used,
                tokens_budget=tokens_budget,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            new_alerts.append(alert)
        elif percentage >= self.config.warning_threshold_percent:
            alert = BudgetAlert(
                level="warning",
                message=f"Warning: Combo {combo_id} at {percentage:.1f}% budget ({tokens_used}/{tokens_budget})",
                combo_id=combo_id,
                percentage=percentage,
                tokens_used=tokens_used,
                tokens_budget=tokens_budget,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            new_alerts.append(alert)

        self.alerts.extend(new_alerts)
        return new_alerts

    def get_alerts(self, combo_id: str | None = None) -> list[BudgetAlert]:
        """Get all alerts, optionally filtered by combo."""
        if combo_id:
            return [a for a in self.alerts if a.combo_id == combo_id]
        return self.alerts


class TokenCompressor:
    """Deterministic token compressor for agentic message history."""

    def __init__(self, target_tokens: int = 4000):
        self.target_tokens = target_tokens

    async def compress(self, messages: list[dict], loop_step: int) -> list[dict]:
        """Compress message history to target token count.

        Returns the *same list object* when nothing was compressed, so callers can
        detect "did this call compress" with an identity check instead of scanning
        for a stale `compressed` marker left over from an earlier iteration.

        Leading `system` messages are never summarised away (C11) — they carry the
        agent's standing instructions and dropping them silently changes behaviour.
        """
        if not messages:
            return messages
        estimated = self._estimate_tokens(messages)
        if estimated <= self.target_tokens:
            return messages

        messages = self._strip_verbose_tool_outputs(messages)
        if self._estimate_tokens(messages) <= self.target_tokens:
            return messages

        # Keep every leading system message verbatim.
        sys_count = 0
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "system":
                sys_count += 1
            else:
                break
        prefix = messages[:sys_count]
        rest = messages[sys_count:]

        recent = rest[-2:] if len(rest) >= 2 else rest
        old = rest[:-2] if len(rest) > 2 else []

        if old:
            summary = await self._summarize(old, loop_step)
            return prefix + [summary] + recent

        return messages

    def _estimate_tokens(self, messages: list[dict]) -> int:
        total = 0
        for m in messages:
            content = m.get("content", "") or ""
            role = m.get("role", "")
            total += len(content) // 4 + 10
            tools = m.get("tools")
            if tools:
                total += len(str(tools)) // 4
        return max(total, 1)

    def _strip_verbose_tool_outputs(self, messages: list[dict]) -> list[dict]:
        stripped = []
        changed = False
        for m in messages:
            if m.get("role") == "tool" and m.get("content"):
                content = m["content"]
                if isinstance(content, str) and len(content) > 500:
                    # Structural (JSON-safe) truncation: tool bodies are JSON
                    # documents, and raw slicing severs them mid-key, leaving
                    # malformed JSON the model cannot parse and strict
                    # upstreams reject with a 400.
                    m = {**m, "content": _truncate_tool_content(content, 500)}
                    changed = True
            stripped.append(m)
        # Identity must only change when content actually changed — callers use
        # `result is not original` to decide whether this call compressed.
        return stripped if changed else messages

    async def _summarize(self, messages: list[dict], loop_step: int) -> dict:
        summary_content = f"[Compressed summary at step {loop_step}] "
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "") or ""
            if role == "assistant":
                summary_content += f"Assistant said: {content[:200]} "
            elif role == "tool":
                summary_content += f"Tool result: {content[:200]} "
            elif role == "user":
                summary_content += f"User asked: {content[:200]} "
        return {"role": "assistant", "content": summary_content[:2000], "compressed": True}


class AgenticLoop:
    """Deterministic looping agentic engine with token compression, circuit breaker, and parallel/best_of strategies."""

    def __init__(self, models: list[tuple[str, str]], max_iterations: int = 5,
                 compression_target_tokens: int = 4000, deterministic: bool = True,
                 tool_calling: bool = True, combo_id: str = "",
                 monthly_token_budget: int = 0, loop_strategy: str = "sequential",
                 tools: list[dict] | None = None, start_iteration: int = 0,
                 total_iteration_budget: int | None = None,
                 reflection_enabled: bool = True, reflection_threshold: float = 0.55,
                 tool_choice: object | None = None, escalation: bool = False,
                 load_spread: bool = False):
        self.escalation = escalation
        # Load-spreading restores the legacy round-robin: every iteration
        # takes the next model in the combo instead of holding the momentum
        # lock. Admin opt-in per combo — the lock exists to protect reasoning
        # continuity, and the two modes are mutually exclusive.
        self.load_spread = load_spread
        if escalation:
            # Auto-escalation: strongest model leads (context window first,
            # frontier families as tie-break), and the momentum lock below
            # pins the loop to it. Same failure-context idea as the critic
            # pass, but decided at routing time instead of after a bad draft.
            models = escalation_order(models)
        self.models = models
        # Contextual momentum: the sequential loop holds one model for the
        # entire run and rotates only on failure (quota/error/breaker).
        # self.models[global_idx % n] used to round-robin mid-thought, which
        # is exactly the "switching models mid-loop" hazard this avoids.
        self._lock_pos = 0
        self._ctx_map = None  # lazy: provider catalog context windows
        self.max_iterations = max_iterations
        self.compression_target_tokens = compression_target_tokens
        self.deterministic = deterministic
        self.tool_calling = tool_calling
        self.combo_id = combo_id
        self.monthly_token_budget = monthly_token_budget
        self.loop_strategy = loop_strategy
        self.tools = tools
        self.start_iteration = start_iteration
        self.total_iteration_budget = total_iteration_budget
        self.reflection_enabled = reflection_enabled
        self.reflection_threshold = reflection_threshold
        self.tool_choice = tool_choice
        self.reflection_repairs = 0
        self.reflection_score: float | None = None
        self._reflection_used = False
        self._last_repair_messages: list[dict] = []
        self.history: list[AgenticStep] = []
        self.compressor = TokenCompressor(compression_target_tokens)
        self.enforcer = BudgetEnforcer(combo_id, monthly_token_budget)
        self.budget_manager = TokenBudgetManager(combo_id)
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
        self._model_failures: dict[str, int] = {}

    def _locked_model(self, fallback_idx: int = 0) -> tuple[str, str]:
        """Pick the momentum-locked model, skipping breaker-open entries.

        The lock starts at position 0 (post-escalation ordering: strongest
        first) and advances one position per real failure; successful
        iterations never rotate. Falls back to the legacy round-robin index
        only if every model is breaker-open.

        With ``load_spread`` set, this is the plain legacy round-robin —
        every call takes the next model in the combo (brearker-open entries
        still skipped) to spread load, accepting the mid-loop model-switch
        that the lock was built to avoid.
        """
        models = self.models
        if not models:
            return ("", "")
        if self.load_spread:
            n = len(models)
            for off in range(n):
                cand = models[(fallback_idx + off) % n]
                if not self.circuit_breaker.is_open(f"{cand[0]}:{cand[1]}"):
                    return cand
            return models[fallback_idx % n]
        n = len(models)
        for off in range(n):
            cand = models[(self._lock_pos + off) % n]
            if not self.circuit_breaker.is_open(f"{cand[0]}:{cand[1]}"):
                return cand
        return models[fallback_idx % n]

    def _rotate_lock(self) -> None:
        """Advance the momentum lock after a failure (quota/error/breaker)."""
        self._lock_pos += 1

    def _maybe_scale_context(self, model_provider: str, model_id: str,
                             messages: list[dict]) -> tuple[str, str]:
        """Dynamic context scaling: move to the combo's biggest window when the
        conversation outgrows the current leader's. Within-combo only — the
        admin's declared list is the contract; nothing is injected.
        """
        if self._ctx_map is None:
            try:
                from app.providers import get_all_models
                self._ctx_map = {m["id"]: (m.get("context_length") or 0)
                                 for m in get_all_models()}
            except Exception:
                self._ctx_map = {}
        ctx = self._ctx_map.get(model_id)
        if not ctx:
            return model_provider, model_id
        est = sum(len(str(m.get("content") or "")) for m in messages) // 4
        if est <= 0 or est * 1.2 <= ctx:
            return model_provider, model_id
        best = None
        for p, m in self.models:
            c = self._ctx_map.get(m, 0)
            if est * 1.1 <= c and (best is None or c > best[2]):
                best = (p, m, c)
        if best and best[1] != model_id:
            logger.info(
                "Context scaling: est %d tokens > %s window %d — switching to %s (%d)",
                est, model_id, ctx, best[1], best[2],
            )
            return best[0], best[1]
        return model_provider, model_id

    def _iterations_remaining(self, offset: int = 0) -> int:
        """Effective remaining iterations, honoring resume budget (TR-17.2)."""
        if self.total_iteration_budget is None:
            return max(0, self.max_iterations - offset)
        consumed = self.start_iteration + offset
        return max(0, self.total_iteration_budget - consumed)

    def _begin_index(self) -> int:
        return self.start_iteration

    async def run(self, messages: list[dict]) -> AgenticResult:
        """Run the deterministic agentic loop.

        Args:
            messages: Initial conversation messages.

        Returns:
            AgenticResult with final content, stop reason, and full history.
        """
        total_tokens = 0
        compressed_steps = 0
        last_content = None
        stop_reason = "max_iterations"

        # Check monthly budget status
        budget_status = await get_combo_budget_status(self.combo_id)
        # Anchor the enforcer at month-to-date usage so 80%/95% thresholds and
        # blocking reflect the real monthly position (C12).
        self.enforcer.seed_from_status(budget_status)
        if budget_status and not budget_status["within_budget"]:
            return AgenticResult(
                content="[Budget limit reached] Monthly token budget exceeded. No further processing.",
                stop_reason="budget_exceeded",
                iterations=0,
                history=[],
                total_tokens=0,
                compressed_steps=0,
            )

        if self.loop_strategy == "parallel":
            return await self._run_parallel(self._with_agent_system_prompt(messages), total_tokens, compressed_steps, stop_reason)
        elif self.loop_strategy == "best_of":
            return await self._run_best_of(self._with_agent_system_prompt(messages), total_tokens, compressed_steps, stop_reason)

        return await self._run_sequential(self._with_agent_system_prompt(messages), total_tokens, compressed_steps, stop_reason)

    def _with_agent_system_prompt(self, messages: list[dict]) -> list[dict]:
        """Prepend a system prompt that instructs tool use when tool_calling is on.

        Without this, models receive the tools but have no instruction to use
        them, so they reply conversationally (\"I don't have the tools\") or
        describe intent (\"Mapping your workspace now\") without emitting a
        tool_call. Only injects when no leading system message already exists,
        so a caller-supplied system prompt always wins.
        """
        if self.escalation:
            messages = list(messages)
            directive = {
                "role": "system",
                "content": (
                    "ESCALATION: a previous attempt on a smaller model failed this "
                    "task. You are the escalation target — be strictly correct, "
                    "emit only valid JSON when calling tools, and re-verify "
                    "constraints before answering."
                ),
            }
            # Insert after any leading system message so the caller's own
            # system prompt is still respected.
            if messages and messages[0].get("role") == "system":
                messages.insert(1, directive)
            else:
                messages.insert(0, directive)
        if not (self.tool_calling and self.tools):
            return messages
        # Respect an existing leading system message.
        if any(isinstance(m, dict) and m.get("role") == "system" for m in messages[:1]):
            return messages

        tool_names = []
        for t in self.tools:
            try:
                tool_names.append(t["function"]["name"])
            except (KeyError, TypeError):
                continue
        tools_line = ", ".join(tool_names) if tool_names else "(none)"
        system_prompt = (
            "You are an autonomous coding agent with access to tools. "
            "You MUST use the provided tools to inspect and modify the user's project — "
            "do not claim you cannot access files or run commands; the tools give you that ability. "
            "To read, write, or list files and run commands, emit a tool_call using the available tools "
            f"({tools_line}). "
            "When the user asks you to look at, list, read, edit, or run something in their project, "
            "call the relevant tool immediately instead of describing what you would do. "
            "After receiving tool results, continue until the task is complete, then give a concise final answer."
        )
        return [{"role": "system", "content": system_prompt}] + list(messages)

    async def _run_sequential(self, messages: list[dict], total_tokens: int,
                               compressed_steps: int, stop_reason: str) -> AgenticResult:
        """Sequential loop strategy (existing behavior)."""
        last_content = None
        stop_reason = "max_iterations"
        iterations = self._iterations_remaining()

        for i in range(iterations):
            global_idx = self.start_iteration + i
            model_provider, model_id = self._locked_model(global_idx)
            from app.otel_tracing import span as otel_span, set_attrs as otel_attrs
            _iter_cm = otel_span("agentic.iteration", {
                "iteration": global_idx, "provider": model_provider,
                "model": model_id, "combo_id": self.combo_id or None,
                "strategy": self.loop_strategy, "load_spread": self.load_spread,
                "escalation": self.escalation,
            }).__enter__() if True else None

            if self.circuit_breaker.is_open(f"{model_provider}:{model_id}"):
                self._rotate_lock()
                logger.warning("Circuit breaker open for %s:%s, skipping", model_provider, model_id)
                self._model_failures[f"{model_provider}:{model_id}"] = self._model_failures.get(f"{model_provider}:{model_id}", 0) + 1
                continue

            step = AgenticStep(
                iteration=global_idx,
                model=model_id,
                provider=model_provider,
            )
            self.history.append(step)

            # Check budget before each iteration
            if not self.enforcer.check(1000):
                stop_reason = "budget_exceeded"
                break

            try:
                from app.providers import get_attempt_plan
                from app.provider_utils import QuotaError

                raw_messages = messages
                messages = await self.compressor.compress(messages, i)
                step.compressed = messages is not raw_messages
                if step.compressed:
                    compressed_steps += 1

                model_provider, model_id = self._maybe_scale_context(model_provider, model_id, messages)
                plan = [(model_provider, model_id)]
                result = await self._call_upstream(plan, messages)

                step.input_tokens = result.get("input_tokens", 0)
                step.output_tokens = result.get("output_tokens", 0)
                iteration_tokens = step.input_tokens + step.output_tokens
                total_tokens += iteration_tokens
                self.enforcer.record(iteration_tokens)

                # Record to monthly budget
                if self.combo_id:
                    try:
                        await self.budget_manager.record_usage(iteration_tokens)
                    except Exception:
                        logger.warning("Failed to record token budget for combo %s", self.combo_id)

                last_content = result.get("content")
                step.content = last_content
                step.stop_reason = result.get("stop_reason")

                try:
                    otel_attrs(_iter_cm, {
                        "input_tokens": step.input_tokens,
                        "output_tokens": step.output_tokens,
                        "stop_reason": step.stop_reason,
                        "tool_calls": len(result.get("tool_calls") or []),
                    })
                    _iter_cm.__exit__(None, None, None)
                    _iter_cm = None
                except Exception:
                    _iter_cm = None

                # Schema-adjacent repair: when the draft carries detectable
                # structural damage (fences / prose / truncation), apply the
                # deterministic repair before it reaches the agent.
                if last_content and result.get("stop_reason") in ("stop", "length"):
                    issues = _structural_issues(last_content)
                    if issues:
                        try:
                            from app.reflection import repair_json_content
                            repaired, changed = repair_json_content(last_content)
                            if changed:
                                step.content = repaired
                                last_content = repaired
                                logger.info(
                                    "JSON repair on combo %s iteration %d: %s",
                                    self.combo_id or "-", global_idx, issues,
                                )
                        except Exception:
                            logger.warning("JSON repair failed silently", exc_info=True)

                if result.get("stop_reason") == "error":
                    logger.warning("Upstream error on iteration %d model %s, trying next model", i, model_id)
                    self.circuit_breaker.record_failure(f"{model_provider}:{model_id}")
                    self._model_failures[f"{model_provider}:{model_id}"] = self._model_failures.get(f"{model_provider}:{model_id}", 0) + 1
                    self._rotate_lock()
                    continue

                if result.get("stop_reason"):
                    stop_reason = result["stop_reason"]

                tool_calls = result.get("tool_calls")
                if tool_calls:
                    step.tool_calls = tool_calls
                    if self.tool_calling:
                        messages = await self._execute_tools(tool_calls, messages)
                        self.circuit_breaker.record_success(f"{model_provider}:{model_id}")
                        continue
                    else:
                        stop_reason = "tool_calls"
                        break

                if last_content and last_content.strip():
                    stop_reason = "stop"
                    self.circuit_breaker.record_success(f"{model_provider}:{model_id}")
                    # Reflection critic: one repair round if draft scores below threshold
                    if await self._maybe_repair(last_content, result, messages, global_idx, i, iterations):
                        messages = self._last_repair_messages
                        continue
                    break

            except QuotaError:
                logger.warning("Quota exceeded on iteration %d, model %s", i, model_id)
                self.circuit_breaker.record_failure(f"{model_provider}:{model_id}")
                self._model_failures[f"{model_provider}:{model_id}"] = self._model_failures.get(f"{model_provider}:{model_id}", 0) + 1
                continue
            except Exception as exc:
                logger.warning("Agentic loop error on iteration %d: %s", i, exc)
                self.circuit_breaker.record_failure(f"{model_provider}:{model_id}")
                self._model_failures[f"{model_provider}:{model_id}"] = self._model_failures.get(f"{model_provider}:{model_id}", 0) + 1
                continue

        return AgenticResult(
            content=last_content,
            stop_reason=stop_reason,
            iterations=len(self.history),
            history=self.history,
            total_tokens=total_tokens,
            compressed_steps=compressed_steps,
            reflection_repairs=self.reflection_repairs,
            reflection_score=self.reflection_score,
        )

    async def _maybe_repair(self, content: str, result: dict, messages: list[dict],
                            global_idx: int, loop_i: int, iterations: int) -> bool:
        """Run reflection critic; if score < threshold and budget remains, inject feedback for one repair.

        Returns True when a repair round should continue (messages on self._last_repair_messages).
        """
        self._last_repair_messages = messages
        if not self.reflection_enabled or self._reflection_used:
            return False
        # Need at least one remaining loop slot after the current iteration
        if loop_i + 1 >= iterations:
            return False

        try:
            from app.reflection import critique_draft, critic_message
            outcome = critique_draft(content, result, threshold=self.reflection_threshold)
            self.reflection_score = outcome.score
            if not outcome.needs_repair:
                return False
            if not self.enforcer.check(500):
                return False
            self._reflection_used = True
            self.reflection_repairs += 1
            self._last_repair_messages = messages + [critic_message(outcome.feedback)]
            logger.info(
                "Reflection repair on combo %s: score=%.2f issues=%s",
                self.combo_id or "-", outcome.score, outcome.issues,
            )
            return True
        except Exception as exc:
            logger.warning("Reflection critic failed: %s", exc)
            return False

    async def _run_parallel(self, messages: list[dict], total_tokens: int,
                             compressed_steps: int, stop_reason: str) -> AgenticResult:
        """Parallel loop strategy: call all models concurrently, fall back on failure."""
        last_content = None
        stop_reason = "max_iterations"
        MAX_FALLBACK_ATTEMPTS = 2
        ended_mid_tool_loop = False
        iterations = self._iterations_remaining()

        for i in range(iterations):
            global_idx = self.start_iteration + i
            available_models = [
                (mp, mid) for mp, mid in self.models
                if not self.circuit_breaker.is_open(f"{mp}:{mid}")
            ]
            if not available_models:
                logger.warning("All models circuit-broken, skipping iteration %d", i)
                continue

            if not self.enforcer.check(1000):
                stop_reason = "budget_exceeded"
                break

            try:
                raw_messages = messages
                messages = await self.compressor.compress(messages, i)
                step_compressed_flag = messages is not raw_messages
                if step_compressed_flag:
                    compressed_steps += 1

                tasks = []
                for model_provider, model_id in available_models:
                    plan = [(model_provider, model_id)]
                    tasks.append(self._call_upstream(plan, messages))

                results = await asyncio.gather(*tasks, return_exceptions=True)

                successful_results: list[tuple[int, dict]] = []
                failed_indices = []
                for j, result in enumerate(results):
                    if isinstance(result, Exception):
                        mp, mid = available_models[j]
                        self.circuit_breaker.record_failure(f"{mp}:{mid}")
                        self._model_failures[f"{mp}:{mid}"] = self._model_failures.get(f"{mp}:{mid}", 0) + 1
                        failed_indices.append(j)
                    elif isinstance(result, dict) and result.get("stop_reason") == "error":
                        mp, mid = available_models[j]
                        self.circuit_breaker.record_failure(f"{mp}:{mid}")
                        self._model_failures[f"{mp}:{mid}"] = self._model_failures.get(f"{mp}:{mid}", 0) + 1
                        failed_indices.append(j)
                    else:
                        successful_results.append((j, result))

                used_model_indices = {j for j, _ in successful_results} | set(failed_indices)
                fallback_attempts = 0
                while (not successful_results) and fallback_attempts < MAX_FALLBACK_ATTEMPTS:
                    next_idx = None
                    for idx in range(len(available_models)):
                        if idx not in used_model_indices:
                            next_idx = idx
                            break
                    if next_idx is None:
                        break
                    mp, mid = available_models[next_idx]
                    used_model_indices.add(next_idx)
                    fallback_attempts += 1
                    plan = [(mp, mid)]
                    try:
                        fb_result = await self._call_upstream(plan, messages)
                    except Exception:
                        fb_result = {"content": "", "tool_calls": [], "input_tokens": 0, "output_tokens": 0, "stop_reason": "error"}
                    if isinstance(fb_result, dict) and fb_result.get("stop_reason") != "error":
                        successful_results.append((next_idx, fb_result))
                    else:
                        self.circuit_breaker.record_failure(f"{mp}:{mid}")
                        self._model_failures[f"{mp}:{mid}"] = self._model_failures.get(f"{mp}:{mid}", 0) + 1

                best_result = None
                best_score = -float("inf")
                best_j = 0
                for j, result in successful_results:
                    score = self._score_result(result)
                    if score > best_score:
                        best_score = score
                        best_result = result
                        best_j = j

                if best_result:
                    model_provider, model_id = available_models[best_j]
                    step = AgenticStep(
                        iteration=global_idx,
                        model=model_id,
                        provider=model_provider,
                    )
                    self.history.append(step)

                    step.input_tokens = best_result.get("input_tokens", 0)
                    step.output_tokens = best_result.get("output_tokens", 0)
                    iteration_tokens = step.input_tokens + step.output_tokens
                    total_tokens += iteration_tokens
                    self.enforcer.record(iteration_tokens)

                    if self.combo_id:
                        try:
                            await self.budget_manager.record_usage(iteration_tokens)
                        except Exception:
                            logger.warning("Failed to record token budget for combo %s", self.combo_id)

                    last_content = best_result.get("content")
                    step.content = last_content
                    step.stop_reason = best_result.get("stop_reason")

                    if best_result.get("stop_reason") and best_result["stop_reason"] != "error":
                        stop_reason = best_result["stop_reason"]

                    # Only clear failures for models that actually succeeded (TR-2.2)
                    self.circuit_breaker.record_success(f"{model_provider}:{model_id}")

                    tool_calls = best_result.get("tool_calls")
                    if tool_calls:
                        step.tool_calls = tool_calls
                        if self.tool_calling:
                            messages = await self._execute_tools(tool_calls, messages)
                            ended_mid_tool_loop = True
                            continue
                        else:
                            stop_reason = "tool_calls"
                            break

                    if last_content and last_content.strip():
                        stop_reason = "stop"
                        if await self._maybe_repair(last_content, best_result, messages, global_idx, i, iterations):
                            messages = self._last_repair_messages
                            continue
                        break

            except Exception as exc:
                logger.warning("Agentic parallel loop error on iteration %d: %s", i, exc)
                continue

        # Synthesis pass: exhausted max_iterations while still tool-calling, so no
        # clean final answer exists — force one tools-disabled call to synthesize.
        # Try models in order until one succeeds (some reject tool-call history).
        if ended_mid_tool_loop and self.tool_calling and self.models:
            synth_messages = _trim_tool_results_for_synthesis(list(messages)) + [{
                "role": "user",
                "content": (
                    "You have gathered enough information from the tool results above. "
                    "Give your FINAL answer now as plain text: summarize what the project "
                    "contains and what it does. Do NOT call any more tools and do NOT emit "
                    "tool-call syntax — just write the complete answer."
                ),
            }]
            for synth_provider, synth_model in self.models:
                if self.circuit_breaker.is_open(f"{synth_provider}:{synth_model}"):
                    continue
                try:
                    synth = await self._call_upstream([(synth_provider, synth_model)], synth_messages, disable_tools=True)
                except Exception as exc:
                    logger.warning("Agentic synthesis pass failed for %s:%s: %s", synth_provider, synth_model, exc)
                    continue
                synth_content = (synth.get("content") or "").strip()
                if synth.get("stop_reason") != "error" and synth_content:
                    last_content = synth_content
                    stop_reason = "stop"
                    total_tokens += (synth.get("input_tokens", 0) or 0) + (synth.get("output_tokens", 0) or 0)
                    break

        return AgenticResult(
            content=last_content,
            stop_reason=stop_reason,
            iterations=len(self.history),
            history=self.history,
            total_tokens=total_tokens,
            compressed_steps=compressed_steps,
            reflection_repairs=self.reflection_repairs,
            reflection_score=self.reflection_score,
        )

    async def _run_best_of(self, messages: list[dict], total_tokens: int,
                            compressed_steps: int, stop_reason: str) -> AgenticResult:
        """Best_of loop strategy: call top models, score and pick best, fall back on failure."""
        last_content = None
        stop_reason = "max_iterations"
        MAX_FALLBACK_ATTEMPTS = 2
        iterations = self._iterations_remaining()

        for i in range(iterations):
            global_idx = self.start_iteration + i
            available_models = [
                (mp, mid) for mp, mid in self.models
                if not self.circuit_breaker.is_open(f"{mp}:{mid}")
            ]
            top_models = available_models[:2]
            if not top_models:
                logger.warning("No models available for best_of on iteration %d", i)
                continue

            if not self.enforcer.check(1000):
                stop_reason = "budget_exceeded"
                break

            try:
                raw_messages = messages
                messages = await self.compressor.compress(messages, i)
                step_compressed_flag = messages is not raw_messages
                if step_compressed_flag:
                    compressed_steps += 1

                tasks = []
                for model_provider, model_id in top_models:
                    plan = [(model_provider, model_id)]
                    tasks.append(self._call_upstream(plan, messages))

                results = await asyncio.gather(*tasks, return_exceptions=True)

                successful_results: list[tuple[int, dict]] = []
                failed_indices = []
                for j, result in enumerate(results):
                    if isinstance(result, Exception):
                        mp, mid = top_models[j]
                        self.circuit_breaker.record_failure(f"{mp}:{mid}")
                        self._model_failures[f"{mp}:{mid}"] = self._model_failures.get(f"{mp}:{mid}", 0) + 1
                        failed_indices.append(j)
                    elif isinstance(result, dict) and result.get("stop_reason") == "error":
                        mp, mid = top_models[j]
                        self.circuit_breaker.record_failure(f"{mp}:{mid}")
                        self._model_failures[f"{mp}:{mid}"] = self._model_failures.get(f"{mp}:{mid}", 0) + 1
                        failed_indices.append(j)
                    else:
                        successful_results.append((j, result))

                used_top_indices = {j for j, _ in successful_results} | set(failed_indices)
                remaining_pool = [
                    (idx, m) for idx, m in enumerate(available_models)
                    if idx >= len(top_models) or idx not in used_top_indices
                ]
                fallback_attempts = 0
                while (not successful_results) and fallback_attempts < MAX_FALLBACK_ATTEMPTS and remaining_pool:
                    pool_idx, (mp, mid) = remaining_pool.pop(0)
                    fallback_attempts += 1
                    plan = [(mp, mid)]
                    try:
                        fb_result = await self._call_upstream(plan, messages)
                    except Exception:
                        fb_result = {"content": "", "tool_calls": [], "input_tokens": 0, "output_tokens": 0, "stop_reason": "error"}
                    if isinstance(fb_result, dict) and fb_result.get("stop_reason") != "error":
                        effective_j = len(top_models) + (pool_idx - len(top_models)) if pool_idx >= len(top_models) else pool_idx
                        successful_results.append((effective_j, fb_result))
                    else:
                        self.circuit_breaker.record_failure(f"{mp}:{mid}")
                        self._model_failures[f"{mp}:{mid}"] = self._model_failures.get(f"{mp}:{mid}", 0) + 1

                best_result = None
                best_j = 0
                if successful_results:
                    best_score = -float("inf")
                    for j, result in successful_results:
                        score = self._score_result(result)
                        if score > best_score:
                            best_score = score
                            best_result = result
                            best_j = j

                if best_result:
                    if best_j < len(top_models):
                        model_provider, model_id = top_models[best_j]
                    else:
                        rel = best_j - len(top_models)
                        extra = [m for idx, m in enumerate(available_models) if idx >= len(top_models)]
                        if rel < len(extra):
                            model_provider, model_id = extra[rel]
                        else:
                            model_provider, model_id = top_models[0] if top_models else available_models[0]

                    step = AgenticStep(
                        iteration=global_idx,
                        model=model_id,
                        provider=model_provider,
                    )
                    self.history.append(step)

                    step.input_tokens = best_result.get("input_tokens", 0)
                    step.output_tokens = best_result.get("output_tokens", 0)
                    iteration_tokens = step.input_tokens + step.output_tokens
                    total_tokens += iteration_tokens
                    self.enforcer.record(iteration_tokens)

                    if self.combo_id:
                        try:
                            await self.budget_manager.record_usage(iteration_tokens)
                        except Exception:
                            logger.warning("Failed to record token budget for combo %s", self.combo_id)

                    last_content = best_result.get("content")
                    step.content = last_content
                    step.stop_reason = best_result.get("stop_reason")

                    if best_result.get("stop_reason") and best_result["stop_reason"] != "error":
                        stop_reason = best_result["stop_reason"]

                    self.circuit_breaker.record_success(f"{model_provider}:{model_id}")

                    tool_calls = best_result.get("tool_calls")
                    if tool_calls:
                        step.tool_calls = tool_calls
                        if self.tool_calling:
                            messages = await self._execute_tools(tool_calls, messages)
                            continue
                        else:
                            stop_reason = "tool_calls"
                            break

                    if last_content and last_content.strip():
                        stop_reason = "stop"
                        if await self._maybe_repair(last_content, best_result, messages, global_idx, i, iterations):
                            messages = self._last_repair_messages
                            continue
                        break

            except Exception as exc:
                logger.warning("Agentic best_of loop error on iteration %d: %s", i, exc)
                continue

        return AgenticResult(
            content=last_content,
            stop_reason=stop_reason,
            iterations=len(self.history),
            history=self.history,
            total_tokens=total_tokens,
            compressed_steps=compressed_steps,
            reflection_repairs=self.reflection_repairs,
            reflection_score=self.reflection_score,
        )

    def _score_result(self, result: dict) -> float:
        """Score a model result for parallel/best_of selection.

        Balances content length, token cost, stop_reason, and tool_calls (TR-2.4).
        """
        content = result.get("content", "") or ""
        input_tokens = result.get("input_tokens", 0) or 0
        output_tokens = result.get("output_tokens", 0) or 0
        stop_reason = result.get("stop_reason", "") or ""
        tool_calls = result.get("tool_calls") or []

        score = len(content) * 2
        # Prefer fewer tokens for the same content length
        score -= (input_tokens + output_tokens) * 0.01
        if stop_reason == "stop":
            score += 15
        if stop_reason == "tool_calls":
            score += 8
        if stop_reason == "error":
            score -= 50
        if tool_calls:
            score += 5
            # Validity: tool calls with parsed function names earn a little more
            valid_tcs = sum(1 for tc in tool_calls if isinstance(tc, dict) and tc.get("function", {}).get("name"))
            score += valid_tcs * 2

        # When tool_calling is enabled, a result that emits tool_calls is the
        # correct next action in the loop — it must decisively outrank a
        # conversational refusal, whose length-based score would otherwise win.
        if self.tool_calling and tool_calls and stop_reason != "error":
            score += 1000

        reasoning_keywords = ["step", "therefore", "because", "first", "next", "conclude"]
        content_lower = content.lower()
        reasoning_count = sum(content_lower.count(kw) for kw in reasoning_keywords)
        score += reasoning_count * 2

        # Mild efficiency: higher content-per-token wins ties
        if output_tokens > 0:
            score += (len(content) / max(output_tokens, 1)) * 0.5

        denom = 1 + (output_tokens * 0.0001)
        return score / denom

    async def _call_upstream(self, plan: list[tuple[str, str]], messages: list[dict], disable_tools: bool = False) -> dict:
        """Strategy-agnostic upstream call dispatching via providers._stream_for (FR-1).

        ``disable_tools=True`` forces a tool-free call (used by the synthesis pass
        so the model must emit a final text answer, not another tool_call).
        """
        provider, model_id = plan[0]
        import json
        from app.provider_utils import QuotaError, to_openai_messages

        # The caller's tool_choice (e.g. "required"/"none"/a named function) is a
        # per-request control the loop must honor, but the engine only threads it
        # into calls that actually carry tools: forwarding tool_choice on a
        # tool-free call (synthesis pass, tool_calling off) makes strict upstreams
        # reject the request with a 400, so it is gated here exactly like `tools`.
        sampling = None
        if self.tool_choice is not None and not disable_tools and self.tool_calling:
            from app.provider_utils import SamplingParams
            sampling = SamplingParams(tool_choice=self.tool_choice)

        async def _collect(provider_str: str, model_str: str, msgs: list[dict]) -> dict:
            from app.providers import _stream_for
            fm_messages = to_openai_messages(msgs)
            tools = None if disable_tools else (self.tools if self.tool_calling else None)
            content_parts: list[str] = []
            tool_call_acc: dict[int, dict] = {}
            input_tokens = 0
            output_tokens = 0
            finish_reason = "stop"
            saw_done = False
            async for chunk in _stream_for(provider_str, model_str, fm_messages, include_usage=True, tools=tools, sampling=sampling):
                if not isinstance(chunk, str) or not chunk.startswith("data:"):
                    continue
                data_str = chunk[6:].strip()
                if data_str == "[DONE]":
                    saw_done = True
                    break
                try:
                    parsed = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict) and "error" in parsed and parsed.get("error"):
                    err = parsed["error"]
                    msg = err.get("message", "") if isinstance(err, dict) else str(err)
                    if "429" in msg or "quota" in msg.lower() or "rate" in msg.lower():
                        raise QuotaError(msg or "quota exhausted", provider=provider_str, model=model_str)
                    return {"content": "", "tool_calls": [], "input_tokens": 0, "output_tokens": 0, "stop_reason": "error", "error": msg}
                if parsed.get("model"):
                    model_str_actual = parsed["model"]
                usage = parsed.get("usage")
                if usage:
                    input_tokens = usage.get("prompt_tokens", input_tokens) or input_tokens
                    output_tokens = usage.get("completion_tokens", output_tokens) or output_tokens
                choices = parsed.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    if delta.get("content"):
                        content_parts.append(delta["content"])
                    for tc in (delta.get("tool_calls") or []):
                        idx = tc.get("index", 0)
                        slot = tool_call_acc.setdefault(idx, {
                            "id": None,
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        })
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["function"]["name"] = fn["name"]
                        if fn.get("arguments"):
                            slot["function"]["arguments"] += fn["arguments"]
                    fr = choices[0].get("finish_reason")
                    if fr:
                        finish_reason = fr
            content = "".join(content_parts)
            if not input_tokens and not output_tokens:
                input_tokens = max(1, len(to_openai_messages(messages)[0].get("content") or "") // 4) if messages else 0
                output_tokens = max(1, len(content) // 4) if content else 0
            tool_calls = []
            for idx in sorted(tool_call_acc):
                slot = tool_call_acc[idx]
                if not slot["id"]:
                    slot["id"] = f"call_{uuid.uuid4().hex[:24]}"
                tool_calls.append(slot)
            if tool_calls:
                finish_reason = "tool_calls" if finish_reason in (None, "stop") else finish_reason
            return {
                "content": content,
                "tool_calls": tool_calls,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "stop_reason": finish_reason or ("tool_calls" if tool_calls else "stop"),
            }

        try:
            return await _collect(provider, model_id, messages)
        except QuotaError:
            raise
        except Exception as exc:
            logger.warning("Upstream call failed for %s:%s: %s", provider, model_id, exc)
            return {"content": "", "tool_calls": [], "input_tokens": 0, "output_tokens": 0, "stop_reason": "error", "error": str(exc)}

    async def _get_combo_allowed_tools(self) -> list[str] | None:
        """Query DB via crud for combo-specific allowed tools list.

        Returns None if no combo_id is set (means all tools allowed), or when the
        combo carries an explicitly empty allow-list (also unrestricted).
        Returns TOOLS_POLICY_UNAVAILABLE — which denies every tool — when the
        policy could not be read (S3: a DB blip must not widen the sandbox).
        """
        if not self.combo_id:
            return None
        try:
            from app.crud import get_combo_tools
            from app.database import async_session
            from app.tools_registry import TOOLS_POLICY_UNAVAILABLE
            async with async_session() as db:
                tools = await get_combo_tools(db, self.combo_id)
            return tools or None
        except Exception:
            from app.tools_registry import TOOLS_POLICY_UNAVAILABLE
            logger.error(
                "Failed to fetch allowed tools for combo %s — denying all tools",
                self.combo_id, exc_info=True,
            )
            return TOOLS_POLICY_UNAVAILABLE

    async def _execute_tools(self, tool_calls: list[dict], messages: list[dict]) -> list[dict]:
        """Execute tool calls with allow-list filtering; append assistant+tool messages (FR-2)."""
        import json
        allowed_tools = await self._get_combo_allowed_tools()

        # Include the assistant turn that requested the tool calls (multi-turn correctness)
        assistant_content_parts = []
        for tc in tool_calls:
            fn = tc.get("function", {}) or {}
            if fn.get("name"):
                assistant_content_parts.append(fn.get("arguments") or "")
        assistant_msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls,
        }
        # Only append if last message isn't already this assistant tool_calls turn
        if not (messages and messages[-1].get("role") == "assistant" and messages[-1].get("tool_calls")):
            messages = list(messages) + [assistant_msg]

        results = []
        for tc in tool_calls:
            function_name = tc.get("function", {}).get("name", "")
            if allowed_tools is not None and function_name not in allowed_tools:
                tool_result = json.dumps({
                    "success": False,
                    "error": f"Tool '{function_name}' not allowed for this combo",
                    "allowed_tools": allowed_tools,
                })
            else:
                args = tc.get("function", {}).get("arguments", "{}")
                from contextlib import contextmanager as _cm
                from app.otel_tracing import span as otel_span, set_attrs as otel_attrs

                @contextmanager
                def _noop():
                    yield None

                tool_span_cm = (otel_span("tool.execution", {"tool.name": function_name, "tool.args": args}) if True else _noop())
                with tool_span_cm as _ts:
                    tool_result = await self._call_tool(function_name, args)
                    otel_attrs(_ts, {"success": bool(isinstance(tool_result, str) and '"error"' not in tool_result[:200])})
            results.append({
                "role": "tool",
                "tool_call_id": tc.get("id"),
                "content": tool_result,
            })
        return messages + results

    async def _call_tool(self, function_name: str, args: str) -> str:
        """Execute a tool call by dynamically loading and invoking the tool function.
        
        Supports registered tools in app.tools_registry module.
        Args are validated and sanitized against tool schema before execution.
        """
        import json
        from app.tools_registry import get_tool_schema, tools_registry, sanitize_tool_args, tools_usage_metrics
        
        try:
            params = json.loads(args) if isinstance(args, str) else args

            tools_usage_metrics[function_name]["calls"] += 1
            
            # Check if tool is registered
            if function_name not in tools_registry:
                tools_usage_metrics[function_name]["error"] += 1
                return json.dumps({
                    "error": f"Tool '{function_name}' not found. Available tools: {list(tools_registry.keys())}",
                    "tool": function_name,
                    "params": params,
                })
            
            # Get tool schema for validation
            tool_schema = get_tool_schema(function_name)

            # Sanitize arguments against schema
            if tool_schema:
                params = sanitize_tool_args(tool_schema, params)
            
            # Validate arguments against schema
            if tool_schema:
                validation_result = self._validate_tool_args(tool_schema, params)
                if not validation_result["valid"]:
                    tools_usage_metrics[function_name]["error"] += 1
                    return json.dumps({
                        "error": validation_result["error"],
                        "tool": function_name,
                        "params": params,
                    })
            
            # Get and execute the tool function
            # Tool result cache (idempotent tools only)
            from app.tool_cache import get_or_none, put as cache_put

            cached = await get_or_none(function_name, params)
            if cached is not None:
                tools_usage_metrics[function_name]["success"] += 1
                return cached

            tool_func = tools_registry[function_name]
            result = await tool_func(params)

            tools_usage_metrics[function_name]["success"] += 1

            # Sanitize result for safe JSON serialization
            if hasattr(result, 'model_dump'):
                result = result.model_dump()
            elif hasattr(result, '__dict__'):
                result = {k: v for k, v in result.__dict__.items() if not k.startswith('_')}

            payload = json.dumps({
                "success": True,
                "result": result,
                "tool": function_name,
            })
            # Only cache successful results
            if isinstance(result, dict) and result.get("success") is False:
                return payload
            await cache_put(function_name, params, payload)
            return payload

        except json.JSONDecodeError as e:
            tools_usage_metrics[function_name]["error"] += 1
            return json.dumps({
                "error": f"Invalid JSON arguments: {str(e)}",
                "tool": function_name,
                "args": str(args)[:200],
            })
        except Exception as exc:
            tools_usage_metrics[function_name]["error"] += 1
            return json.dumps({
                "error": str(exc),
                "tool": function_name,
                "params": str(params)[:200] if 'params' in dir() else str(args)[:200],
            })
    
    def _validate_tool_args(self, schema: dict, params: dict) -> dict:
        """Validate tool arguments against schema.
        
        Returns {'valid': bool, 'error': str | None}
        """
        try:
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            
            # Check required fields
            for field in required:
                if field not in params:
                    return {"valid": False, "error": f"Missing required field: {field}"}
            
            # Validate types
            for key, value in params.items():
                if key in properties:
                    expected_type = properties[key].get("type")
                    if expected_type:
                        if not self._check_type(value, expected_type):
                            return {
                                "valid": False, 
                                "error": f"Invalid type for {key}: expected {expected_type}, got {type(value).__name__}"
                            }
            
            return {"valid": True, "error": None}
        except Exception as e:
            return {"valid": False, "error": f"Validation error: {str(e)}"}
    
    def _check_type(self, value, expected_type: str) -> bool:
        """Check if value matches expected type."""
        type_mapping = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        
        if expected_type not in type_mapping:
            return True  # Unknown type, allow
        
        expected = type_mapping[expected_type]
        if expected_type == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        return isinstance(value, expected)


# BudgetEnforcer lives in token_budget_manager; keep _pending_alerts accessor helper here.
def _pending_alerts_of(engine_or_enforcer) -> list[dict]:
    """Return pending budget alerts list from a BudgetEnforcer's alert manager."""
    am = getattr(getattr(engine_or_enforcer, "alert_manager", None), "_pending_alerts", None)
    if am is None:
        am = getattr(engine_or_enforcer, "_pending_alerts", None)
    return am if am is not None else []


def escalation_order(models: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Strongest-model-first ordering for escalated requests.

    Strength = live-catalog context window (agents mid-loop are usually
    context-bound), then a small frontier-family tie-break (claude/gpt/grok/
    gemini), then the admin's declared order. Parse errors or missing catalog
    entries count as 0 — the admin's order decides those.
    """
    try:
        from app.providers import get_all_models
        ctx = {m["id"]: (m.get("context_length") or 0) for m in get_all_models()}
    except Exception:
        ctx = {}
    frontier = ("claude", "gpt", "grok", "gemini", "opus", "sonnet")

    def strength(item: tuple[str, str], idx: int) -> tuple:
        _p, m = item
        low = m.lower()
        return (
            ctx.get(m, 0),
            any(f in low for f in frontier),
            -idx,
        )

    return [
        m for _, m in sorted(
            enumerate(models),
            key=lambda t: strength(t[1], t[0]),
            reverse=True,
        )
    ]


async def run_agentic_loop(combo_models: list[tuple[str, str]], messages: list[dict],
                           max_iterations: int = 5, compression_target_tokens: int = 4000,
                           deterministic: bool = True, tool_calling: bool = True,
                           combo_id: str = "", monthly_token_budget: int = 0,
                           loop_strategy: str = "sequential",
                           tools: list[dict] | None = None,
                           start_iteration: int = 0,
                           total_iteration_budget: int | None = None,
                           reflection_enabled: bool = True,
                           reflection_threshold: float = 0.55,
                           tool_choice: object | None = None, escalation: bool = False,
                           load_spread: bool = False) -> AgenticResult:
    """Convenience function to run an agentic loop."""
    engine = AgenticLoop(
        models=combo_models,
        max_iterations=max_iterations,
        compression_target_tokens=compression_target_tokens,
        deterministic=deterministic,
        tool_calling=tool_calling,
        combo_id=combo_id,
        monthly_token_budget=monthly_token_budget,
        loop_strategy=loop_strategy,
        tools=tools,
        start_iteration=start_iteration,
        total_iteration_budget=total_iteration_budget,
        reflection_enabled=reflection_enabled,
        reflection_threshold=reflection_threshold,
        tool_choice=tool_choice,
        escalation=escalation,
        load_spread=load_spread,
    )
    from app.otel_tracing import span as otel_span, set_attrs as otel_attrs
    with otel_span("agentic.completion", {
        "combo_id": combo_id or None, "strategy": loop_strategy,
        "max_iterations": max_iterations, "tools": len(tools or []),
        "escalation": escalation, "load_spread": load_spread,
    }) as _root:
        result = await engine.run(messages)
        otel_attrs(_root, {
            "iterations": result.iterations, "total_tokens": result.total_tokens,
            "stop_reason": result.stop_reason, "compressed_steps": result.compressed_steps,
        })
        return result


async def run_agentic_stream(combo_models: list[tuple[str, str]], messages: list[dict],
                             max_iterations: int = 5, compression_target_tokens: int = 4000,
                             deterministic: bool = True, tool_calling: bool = True,
                             combo_id: str = "", monthly_token_budget: int = 0,
                             loop_strategy: str = "sequential",
                             tools: list[dict] | None = None,
                             start_iteration: int = 0,
                             total_iteration_budget: int | None = None,
                             reflection_enabled: bool = True,
                             reflection_threshold: float = 0.55,
                             tool_choice: object | None = None,
                             escalation: bool = False,
                             load_spread: bool = False):
    """Run agentic loop with streaming - yields iteration, tool, compression, budget, and complete events."""
    engine = AgenticLoop(
        models=combo_models,
        max_iterations=max_iterations,
        compression_target_tokens=compression_target_tokens,
        deterministic=deterministic,
        tool_calling=tool_calling,
        combo_id=combo_id,
        monthly_token_budget=monthly_token_budget,
        loop_strategy=loop_strategy,
        tools=tools,
        start_iteration=start_iteration,
        total_iteration_budget=total_iteration_budget,
        reflection_enabled=reflection_enabled,
        reflection_threshold=reflection_threshold,
        tool_choice=tool_choice,
        escalation=escalation,
        load_spread=load_spread,
    )
    from app.otel_tracing import span as otel_span, set_attrs as otel_attrs
    _root = otel_span("agentic.completion.stream", {
        "combo_id": combo_id or None, "strategy": loop_strategy,
        "max_iterations": max_iterations, "tools": len(tools or []),
        "escalation": escalation, "load_spread": load_spread,
    })
    _root.__enter__()
    total_tokens = 0
    compressed_steps = 0
    last_content = None
    stop_reason = "max_iterations"

    budget_status = await get_combo_budget_status(combo_id)
    # Month-to-date baseline for the enforcer (C12) — see BudgetEnforcer.seed_from_status.
    engine.enforcer.seed_from_status(budget_status)
    if budget_status and not budget_status["within_budget"]:
        yield {"type": "error", "content": "[Budget limit reached] Monthly token budget exceeded."}
        yield {
            "type": "complete",
            "content": "[Budget limit reached] Monthly token budget exceeded.",
            "stop_reason": "budget_exceeded",
            "total_tokens": 0,
            "compressed_steps": 0,
            "iterations": 0,
        }
        try:
            _root.__exit__(None, None, None)
        except Exception:
            pass
        return

    # Inject the agent system prompt here (streaming path bypasses run()).
    messages = engine._with_agent_system_prompt(messages)

    if loop_strategy == "parallel":
        async for ev in _stream_parallel(engine, messages, max_iterations, combo_id, tool_calling):
            yield ev
            if ev["type"] == "complete":
                return
        return

    if loop_strategy == "best_of":
        async for ev in _stream_best_of(engine, messages, max_iterations, combo_id, tool_calling):
            yield ev
            if ev["type"] == "complete":
                return
        return

    final_i = 0
    steps_run = 0
    stream_iterations = engine._iterations_remaining()
    for i in range(stream_iterations):
        final_i = i
        steps_run += 1
        global_idx = engine.start_iteration + i
        model_provider, model_id = engine._locked_model(global_idx)
        if engine.circuit_breaker.is_open(f"{model_provider}:{model_id}"):
            engine._rotate_lock()
            yield {"type": "skip", "model": model_id, "provider": model_provider, "reason": "circuit_breaker_open"}
            continue

        if not engine.enforcer.check(1000):
            yield {"type": "error", "content": "[Budget limit reached] Per-session budget exceeded."}
            break

        try:
            pre_alerts_len = len(getattr(engine.enforcer.alert_manager, "_pending_alerts", []))
            raw_messages = messages
            messages = await engine.compressor.compress(messages, i)
            step_compressed = messages is not raw_messages
            if step_compressed:
                compressed_steps += 1
                yield {
                    "type": "compression",
                    "iteration": global_idx,
                    "compressed": True,
                    "target_tokens": compression_target_tokens,
                }

            plan = [(model_provider, model_id)]
            result = await engine._call_upstream(plan, messages)

            input_tokens = result.get("input_tokens", 0)
            output_tokens = result.get("output_tokens", 0)
            iteration_tokens = input_tokens + output_tokens
            total_tokens += iteration_tokens
            engine.enforcer.record(iteration_tokens)

            post_alerts = getattr(engine.enforcer.alert_manager, "_pending_alerts", [])
            if len(post_alerts) > pre_alerts_len:
                for alert in post_alerts[pre_alerts_len:]:
                    yield {
                        "type": "budget_alert",
                        "level": alert.get("level", "warning"),
                        "percentage": alert.get("percentage", 0),
                        "message": alert.get("message", ""),
                        "combo_id": combo_id,
                    }

            if combo_id:
                try:
                    await engine.budget_manager.record_usage(iteration_tokens)
                except Exception:
                    pass

            content = result.get("content")
            last_content = content

            if result.get("stop_reason") and result["stop_reason"] != "error":
                stop_reason = result["stop_reason"]

            tool_calls = result.get("tool_calls")

            yield {
                "type": "iteration",
                "iteration": global_idx,
                "model": model_id,
                "provider": model_provider,
                "content": content,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "stop_reason": stop_reason,
                "tool_calls": tool_calls,
                "compressed": step_compressed,
            }

            if tool_calls:
                if engine.tool_calling:
                    messages = await engine._execute_tools(tool_calls, messages)
                    engine.circuit_breaker.record_success(f"{model_provider}:{model_id}")
                    yield {"type": "tool_execution", "tool_calls": tool_calls}
                    continue
                else:
                    stop_reason = "tool_calls"
                    break

            if content and content.strip():
                stop_reason = "stop"
                engine.circuit_breaker.record_success(f"{model_provider}:{model_id}")
                if await engine._maybe_repair(content, result, messages, global_idx, i, stream_iterations):
                    messages = engine._last_repair_messages
                    yield {
                        "type": "reflection",
                        "score": engine.reflection_score,
                        "repair_round": engine.reflection_repairs,
                        "repaired": True,
                    }
                    continue
                break

        except Exception as exc:
            logger.warning("Agentic stream error on iteration %d: %s", global_idx, exc)
            engine.circuit_breaker.record_failure(f"{model_provider}:{model_id}")
            yield {"type": "error", "content": f"Error: {exc}"}
            continue

    yield {
        "type": "complete",
        "content": last_content,
        "stop_reason": stop_reason,
        "total_tokens": total_tokens,
        "compressed_steps": compressed_steps,
        "iterations": steps_run,
        "reflection_repairs": engine.reflection_repairs,
        "reflection_score": engine.reflection_score,
    }
    try:
        _root.__exit__(None, None, None)
    except Exception:
        pass


def _trim_tool_results_for_synthesis(messages: list[dict], max_chars: int = 1500) -> list[dict]:
    """Clamp each tool-result body before the synthesis call.

    Tool outputs (directory listings, file dumps) dominate the prompt and make
    the synthesis round-trip slow. Keeping a bounded head per result preserves
    the structure the model needs to answer while cutting latency. Bodies are
    clamped JSON-safely (see ``_truncate_tool_content``) so a truncated tool
    result stays parseable when it is re-serialized and forwarded upstream.
    """
    out = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "tool":
            content = m.get("content")
            if isinstance(content, str) and len(content) > max_chars:
                m = {**m, "content": _truncate_tool_content(content, max_chars)}
        out.append(m)
    return out


async def _stream_parallel(engine: AgenticLoop, messages: list[dict], max_iterations: int,
                            combo_id: str, tool_calling_enabled: bool):
    """Yield stream events for the parallel strategy."""
    total_tokens = 0
    compressed_steps = 0
    last_content = None
    stop_reason = "max_iterations"
    MAX_FALLBACK_ATTEMPTS = 2
    final_i = 0
    steps_run = 0
    ended_mid_tool_loop = False
    stream_iterations = engine._iterations_remaining()

    for i in range(stream_iterations):
        final_i = i
        steps_run += 1
        global_idx = engine.start_iteration + i
        available_models = [
            (mp, mid) for mp, mid in engine.models
            if not engine.circuit_breaker.is_open(f"{mp}:{mid}")
        ]
        if not available_models:
            yield {"type": "skip", "iteration": global_idx, "reason": "all_models_circuit_broken"}
            continue

        if not engine.enforcer.check(1000):
            yield {"type": "error", "content": "[Budget limit reached] Per-session budget exceeded."}
            break

        try:
            pre_alerts_len = len(getattr(engine.enforcer.alert_manager, "_pending_alerts", []))
            raw_messages = messages
            messages = await engine.compressor.compress(messages, i)
            step_compressed = messages is not raw_messages
            if step_compressed:
                compressed_steps += 1
                yield {
                    "type": "compression",
                    "iteration": global_idx,
                    "compressed": True,
                    "target_tokens": engine.compression_target_tokens,
                }

            tasks = []
            for model_provider, model_id in available_models:
                plan = [(model_provider, model_id)]
                tasks.append(engine._call_upstream(plan, messages))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            successful_results: list[tuple[int, dict]] = []
            for j, result in enumerate(results):
                if isinstance(result, Exception):
                    mp, mid = available_models[j]
                    engine.circuit_breaker.record_failure(f"{mp}:{mid}")
                elif isinstance(result, dict) and result.get("stop_reason") == "error":
                    mp, mid = available_models[j]
                    engine.circuit_breaker.record_failure(f"{mp}:{mid}")
                else:
                    successful_results.append((j, result))

            used_model_indices = {j for j, _ in successful_results} | {
                j for j, r in enumerate(results)
                if isinstance(r, Exception) or (isinstance(r, dict) and r.get("stop_reason") == "error")
            }
            fallback_attempts = 0
            while (not successful_results) and fallback_attempts < MAX_FALLBACK_ATTEMPTS:
                next_idx = None
                for idx in range(len(available_models)):
                    if idx not in used_model_indices:
                        next_idx = idx
                        break
                if next_idx is None:
                    break
                mp, mid = available_models[next_idx]
                used_model_indices.add(next_idx)
                fallback_attempts += 1
                plan = [(mp, mid)]
                fb_result = await engine._call_upstream(plan, messages)
                if isinstance(fb_result, dict) and fb_result.get("stop_reason") != "error":
                    successful_results.append((next_idx, fb_result))
                else:
                    engine.circuit_breaker.record_failure(f"{mp}:{mid}")

            best_result = None
            best_score = -float("inf")
            best_j = 0
            for j, result in successful_results:
                score = engine._score_result(result)
                if score > best_score:
                    best_score = score
                    best_result = result
                    best_j = j

            if best_result:
                model_provider, model_id = available_models[best_j]
                step = AgenticStep(
                    iteration=global_idx, model=model_id, provider=model_provider,
                )
                engine.history.append(step)

                step.input_tokens = best_result.get("input_tokens", 0)
                step.output_tokens = best_result.get("output_tokens", 0)
                iteration_tokens = step.input_tokens + step.output_tokens
                total_tokens += iteration_tokens
                engine.enforcer.record(iteration_tokens)

                post_alerts = getattr(engine.enforcer.alert_manager, "_pending_alerts", [])
                if len(post_alerts) > pre_alerts_len:
                    for alert in post_alerts[pre_alerts_len:]:
                        yield {
                            "type": "budget_alert",
                            "level": alert.get("level", "warning"),
                            "percentage": alert.get("percentage", 0),
                            "message": alert.get("message", ""),
                            "combo_id": combo_id,
                        }

                if combo_id:
                    try:
                        await engine.budget_manager.record_usage(iteration_tokens)
                    except Exception:
                        logger.warning("Failed to record token budget for combo %s", combo_id)

                last_content = best_result.get("content")
                step.content = last_content

                if best_result.get("stop_reason") and best_result["stop_reason"] != "error":
                    stop_reason = best_result["stop_reason"]

                tool_calls = best_result.get("tool_calls")
                if step_compressed:
                    step.compressed = True

                yield {
                    "type": "iteration",
                    "iteration": global_idx,
                    "model": model_id,
                    "provider": model_provider,
                    "content": last_content,
                    "input_tokens": step.input_tokens,
                    "output_tokens": step.output_tokens,
                    "stop_reason": stop_reason,
                    "tool_calls": tool_calls,
                    "compressed": step_compressed,
                    "candidates": len(successful_results),
                }

                if tool_calls:
                    step.tool_calls = tool_calls
                    if tool_calling_enabled:
                        messages = await engine._execute_tools(tool_calls, messages)
                        for mp, mid in available_models:
                            engine.circuit_breaker.record_success(f"{mp}:{mid}")
                        yield {"type": "tool_execution", "tool_calls": tool_calls}
                        ended_mid_tool_loop = True
                        continue
                    else:
                        stop_reason = "tool_calls"
                        break

                if last_content and last_content.strip():
                    stop_reason = "stop"
                    for mp, mid in available_models:
                        engine.circuit_breaker.record_success(f"{mp}:{mid}")
                    if await engine._maybe_repair(last_content, best_result, messages, global_idx, i, stream_iterations):
                        messages = engine._last_repair_messages
                        yield {
                            "type": "reflection",
                            "score": engine.reflection_score,
                            "repair_round": engine.reflection_repairs,
                            "repaired": True,
                        }
                        continue
                    break

        except Exception as exc:
            logger.warning("Agentic parallel stream error on iteration %d: %s", i, exc)
            yield {"type": "error", "content": f"Error: {exc}"}
            continue

    # Synthesis pass: the loop exhausted max_iterations while still emitting
    # tool_calls, so no model ever produced a clean final answer — the client
    # would otherwise see only the last "Mapping your workspace..." preamble.
    # One final call with tools disabled forces a real, tool-free answer built
    # from the tool results already in `messages`. Try models in order until one
    # succeeds — some (e.g. agnes) reject tool-call history with a 400.
    if ended_mid_tool_loop and tool_calling_enabled and engine.models:
        # Directive: force a final, tool-free answer. Without it the model keeps
        # trying to call tools (and, with tools disabled, writes the call as text).
        synth_messages = _trim_tool_results_for_synthesis(list(messages)) + [{
            "role": "user",
            "content": (
                "You have gathered enough information from the tool results above. "
                "Give your FINAL answer now as plain text: summarize what the project "
                "contains and what it does. Do NOT call any more tools and do NOT emit "
                "tool-call syntax — just write the complete answer."
            ),
        }]
        for synth_provider, synth_model in engine.models:
            if engine.circuit_breaker.is_open(f"{synth_provider}:{synth_model}"):
                continue
            try:
                synth = await engine._call_upstream([(synth_provider, synth_model)], synth_messages, disable_tools=True)
            except Exception as exc:
                logger.warning("Agentic synthesis pass failed for %s:%s: %s", synth_provider, synth_model, exc)
                continue
            synth_content = (synth.get("content") or "").strip()
            if synth.get("stop_reason") != "error" and synth_content:
                last_content = synth_content
                stop_reason = "stop"
                total_tokens += (synth.get("input_tokens", 0) or 0) + (synth.get("output_tokens", 0) or 0)
                yield {
                    "type": "iteration",
                    "iteration": engine.start_iteration + steps_run,
                    "model": synth_model,
                    "provider": synth_provider,
                    "content": last_content,
                    "input_tokens": synth.get("input_tokens", 0),
                    "output_tokens": synth.get("output_tokens", 0),
                    "stop_reason": "stop",
                    "tool_calls": [],
                    "compressed": False,
                    "synthesis": True,
                }
                break

    yield {
        "type": "complete",
        "content": last_content,
        "stop_reason": stop_reason,
        "total_tokens": total_tokens,
        "compressed_steps": compressed_steps,
        "iterations": steps_run,
        "reflection_repairs": engine.reflection_repairs,
        "reflection_score": engine.reflection_score,
    }


async def _stream_best_of(engine: AgenticLoop, messages: list[dict], max_iterations: int,
                           combo_id: str, tool_calling_enabled: bool):
    """Yield stream events for the best_of strategy."""
    total_tokens = 0
    compressed_steps = 0
    last_content = None
    stop_reason = "max_iterations"
    MAX_FALLBACK_ATTEMPTS = 2
    final_i = 0
    steps_run = 0
    stream_iterations = engine._iterations_remaining()

    for i in range(stream_iterations):
        final_i = i
        steps_run += 1
        global_idx = engine.start_iteration + i
        available_models = [
            (mp, mid) for mp, mid in engine.models
            if not engine.circuit_breaker.is_open(f"{mp}:{mid}")
        ]
        top_models = available_models[:2]
        if not top_models:
            yield {"type": "skip", "iteration": global_idx, "reason": "no_models_available"}
            continue

        if not engine.enforcer.check(1000):
            yield {"type": "error", "content": "[Budget limit reached] Per-session budget exceeded."}
            break

        try:
            pre_alerts_len = len(getattr(engine.enforcer.alert_manager, "_pending_alerts", []))
            raw_messages = messages
            messages = await engine.compressor.compress(messages, i)
            step_compressed = messages is not raw_messages
            if step_compressed:
                compressed_steps += 1
                yield {
                    "type": "compression",
                    "iteration": global_idx,
                    "compressed": True,
                    "target_tokens": engine.compression_target_tokens,
                }

            tasks = []
            for model_provider, model_id in top_models:
                plan = [(model_provider, model_id)]
                tasks.append(engine._call_upstream(plan, messages))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            successful_results: list[tuple[int, dict]] = []
            for j, result in enumerate(results):
                if isinstance(result, Exception):
                    mp, mid = top_models[j]
                    engine.circuit_breaker.record_failure(f"{mp}:{mid}")
                elif isinstance(result, dict) and result.get("stop_reason") == "error":
                    mp, mid = top_models[j]
                    engine.circuit_breaker.record_failure(f"{mp}:{mid}")
                else:
                    successful_results.append((j, result))

            used_top_indices = {j for j, _ in successful_results} | {
                j for j, r in enumerate(results)
                if isinstance(r, Exception) or (isinstance(r, dict) and r.get("stop_reason") == "error")
            }
            remaining_pool = [
                (idx, m) for idx, m in enumerate(available_models)
                if idx >= len(top_models) or idx not in used_top_indices
            ]
            fallback_attempts = 0
            while (not successful_results) and fallback_attempts < MAX_FALLBACK_ATTEMPTS and remaining_pool:
                pool_idx, (mp, mid) = remaining_pool.pop(0)
                fallback_attempts += 1
                plan = [(mp, mid)]
                fb_result = await engine._call_upstream(plan, messages)
                if isinstance(fb_result, dict) and fb_result.get("stop_reason") != "error":
                    effective_j = len(top_models) + (pool_idx - len(top_models)) if pool_idx >= len(top_models) else pool_idx
                    successful_results.append((effective_j, fb_result))
                else:
                    engine.circuit_breaker.record_failure(f"{mp}:{mid}")

            best_result = None
            best_j = 0
            if successful_results:
                best_score = -float("inf")
                for j, result in successful_results:
                    score = engine._score_result(result)
                    if score > best_score:
                        best_score = score
                        best_result = result
                        best_j = j

            if best_result:
                if best_j < len(top_models):
                    model_provider, model_id = top_models[best_j]
                else:
                    rel = best_j - len(top_models)
                    extra = [m for idx, m in enumerate(available_models) if idx >= len(top_models)]
                    if rel < len(extra):
                        model_provider, model_id = extra[rel]
                    else:
                        model_provider, model_id = top_models[0] if top_models else available_models[0]

                step = AgenticStep(
                    iteration=global_idx, model=model_id, provider=model_provider,
                )
                engine.history.append(step)

                step.input_tokens = best_result.get("input_tokens", 0)
                step.output_tokens = best_result.get("output_tokens", 0)
                iteration_tokens = step.input_tokens + step.output_tokens
                total_tokens += iteration_tokens
                engine.enforcer.record(iteration_tokens)

                post_alerts = getattr(engine.enforcer.alert_manager, "_pending_alerts", [])
                if len(post_alerts) > pre_alerts_len:
                    for alert in post_alerts[pre_alerts_len:]:
                        yield {
                            "type": "budget_alert",
                            "level": alert.get("level", "warning"),
                            "percentage": alert.get("percentage", 0),
                            "message": alert.get("message", ""),
                            "combo_id": combo_id,
                        }

                if combo_id:
                    try:
                        await engine.budget_manager.record_usage(iteration_tokens)
                    except Exception:
                        logger.warning("Failed to record token budget for combo %s", combo_id)

                last_content = best_result.get("content")
                step.content = last_content

                if best_result.get("stop_reason") and best_result["stop_reason"] != "error":
                    stop_reason = best_result["stop_reason"]

                tool_calls = best_result.get("tool_calls")
                if step_compressed:
                    step.compressed = True

                yield {
                    "type": "iteration",
                    "iteration": global_idx,
                    "model": model_id,
                    "provider": model_provider,
                    "content": last_content,
                    "input_tokens": step.input_tokens,
                    "output_tokens": step.output_tokens,
                    "stop_reason": stop_reason,
                    "tool_calls": tool_calls,
                    "compressed": step_compressed,
                    "candidates": len(successful_results),
                }

                if tool_calls:
                    step.tool_calls = tool_calls
                    if tool_calling_enabled:
                        messages = await engine._execute_tools(tool_calls, messages)
                        for mp, mid in top_models:
                            engine.circuit_breaker.record_success(f"{mp}:{mid}")
                        yield {"type": "tool_execution", "tool_calls": tool_calls}
                        continue
                    else:
                        stop_reason = "tool_calls"
                        break

                if last_content and last_content.strip():
                    stop_reason = "stop"
                    for mp, mid in top_models:
                        engine.circuit_breaker.record_success(f"{mp}:{mid}")
                    if await engine._maybe_repair(last_content, best_result, messages, global_idx, i, stream_iterations):
                        messages = engine._last_repair_messages
                        yield {
                            "type": "reflection",
                            "score": engine.reflection_score,
                            "repair_round": engine.reflection_repairs,
                            "repaired": True,
                        }
                        continue
                    break

        except Exception as exc:
            logger.warning("Agentic best_of stream error on iteration %d: %s", i, exc)
            yield {"type": "error", "content": f"Error: {exc}"}
            continue

    yield {
        "type": "complete",
        "content": last_content,
        "stop_reason": stop_reason,
        "total_tokens": total_tokens,
        "compressed_steps": compressed_steps,
        "iterations": steps_run,
        "reflection_repairs": engine.reflection_repairs,
        "reflection_score": engine.reflection_score,
    }


def estimate_agentic_cost(combo_models: list[tuple[str, str]], max_iterations: int,
                          estimated_tokens_per_iteration: int = 2000) -> dict:
    """Estimate cost for running an agentic loop."""
    estimated_total_tokens = estimated_tokens_per_iteration * max_iterations * len(combo_models)
    # Rough estimate: $0.002 per 1K tokens for input, $0.002 for output
    estimated_cost_usd = (estimated_total_tokens / 1000) * 0.002
    return {
        "estimated_total_tokens": estimated_total_tokens,
        "estimated_cost_usd": round(estimated_cost_usd, 4),
        "iterations": max_iterations,
        "models_per_iteration": len(combo_models),
        "tokens_per_iteration": estimated_tokens_per_iteration,
    }