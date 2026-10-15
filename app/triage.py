"""Deterministic intent triage for the auto-router (Phase 2).

Routes the `bridgeroute/*` auto paths by what the request *is*, not just by
name: cheap, zero-latency heuristics tag each request and reorder the ranked
candidate plan. No SLM, no embeddings — the tags below cover the signals that
actually change model choice for free-tier gateways:

  needs_tool_call — request carries tools (already gated upstream; echoed)
  long_context    — estimated tokens exceed ~30k
  code_gen        — code markers in the conversation
"""
import re

_LONG_CONTEXT_TOKENS = 30_000

_CODE_MARKERS = re.compile(
    r"```|\bdef \b|\bclass \b|^\s*import \b|\bfunction \b|=>|<\/?[a-z]+>|console\.\w+\(",
    re.IGNORECASE | re.MULTILINE,
)

_CODER_MODEL_HINTS = ("coder", "code", "devstral", "codestral", "gpt-oss", "inkling")


def _content_of(m) -> str:
    """Messages arrive as dicts in the engine and pydantic models in routes."""
    if isinstance(m, dict):
        return str(m.get("content") or "")
    return str(getattr(m, "content", "") or "")


def analyze(messages: list, tools: list | None = None) -> dict:
    est_chars = sum(len(_content_of(m)) for m in messages)
    est_tokens = est_chars // 4
    joined = "\n".join(_content_of(m) for m in messages)[-8000:]
    return {
        "needs_tool_call": bool(tools),
        "long_context": est_tokens >= _LONG_CONTEXT_TOKENS,
        "code_gen": bool(_CODE_MARKERS.search(joined)),
        "est_tokens": est_tokens,
    }


def reorder_plan(plan: list[tuple[str, str]], tags: dict,
                 context_lengths: dict[str, int] | None = None) -> list[tuple[str, str]]:
    """Apply tag-conditioned preferences to the ranked plan (stable).

    long_context: candidates with a known window sort by window size desc
    (unknown-context candidates keep relative order after known small ones —
    "unknown" is treated as their catalog priority, not as zero).
    code_gen: coder-hinted ids get a stable boost within their provider group.
    """
    if context_lengths is None:
        try:
            from app.providers import get_all_models
            context_lengths = {m["id"]: (m.get("context_length") or 0)
                               for m in get_all_models()}
        except Exception:
            context_lengths = {}

    out = list(plan)
    if tags.get("long_context"):

        def ctx_key(entry):
            return context_lengths.get(entry[1], 0)
        # Stable sort desc by known context; ids with no catalog window stay
        # in place relative to each other.
        known = [e for e in out if context_lengths.get(e[1], 0) > 0]
        unknown = [e for e in out if context_lengths.get(e[1], 0) <= 0]
        known = sorted(known, key=ctx_key, reverse=True)
        out = known + unknown

    if tags.get("code_gen"):
        boosted = [e for e in out if any(h in e[1].lower() for h in _CODER_MODEL_HINTS)]
        rest = [e for e in out if not any(h in e[1].lower() for h in _CODER_MODEL_HINTS)]
        out = boosted + rest
    return out
