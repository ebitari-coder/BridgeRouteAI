"""Reflection/critic pass for agentic drafts.

Scores content-producing drafts with structural checks and `_score_result`-style
heuristics (length, reasoning signals, stop_reason). When score is below the
configured threshold, returns repair feedback for one follow-up iteration.
"""
from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_THRESHOLD = 0.55

_REASONING_KEYWORDS = (
    "step", "therefore", "because", "first", "next", "conclude",
    "answer", "result", "because", "summary",
)

_PLACEHOLDER_PATTERNS = (
    "lorem ipsum",
    "placeholder",
    "todo:",
    "fixme",
    "as an ai language model",
    "i cannot assist",
    "tbd",
)


@dataclass
class ReflectionOutcome:
    score: float
    needs_repair: bool
    feedback: str
    issues: list[str] = field(default_factory=list)


def repair_json_content(text: str) -> tuple[str, bool]:
    """Deterministically repair the JSON damage the critic detects.

    Handles the three failure modes free models actually produce: code fences
    around JSON, prose before/after a JSON document, and truncated tails
    (missing closers) from short max_tokens. Returns (text_or_repaired,
    changed). A candidate is only accepted when json.loads succeeds, so the
    repair can never make valid output worse.
    """
    if not text or not text.strip():
        return text, False
    body = text.strip()

    # 1. Strip markdown fences.
    if body.startswith("```"):
        first_newline = body.find("\n")
        if first_newline != -1:
            body = body[first_newline + 1:]
        if body.rstrip().endswith("```"):
            body = body.rstrip()[:-3].rstrip()

    # 2. Slice to the outermost JSON document.
    j_start = min(
        (i for i in (body.find("{"), body.find("[")) if i != -1),
        default=-1,
    )
    if j_start == -1:
        return text, False
    opener = body[j_start]
    closer = "}" if opener == "{" else "]"
    j_end = body.rfind(closer)
    candidates = []
    if j_end > j_start:
        candidates.append(body[j_start:j_end + 1])
    # 3. Truncated tail: rebuild the very tail would be guesswork, but a
    # missing closer on an otherwise intact document is recoverable.
    if opener == "{":
        candidates.append(body[j_start:] + "}")
    else:
        candidates.append(body[j_start:] + "]")

    import json as _json
    for cand in candidates:
        try:
            parsed = _json.loads(cand)
            repaired = _json.dumps(parsed, ensure_ascii=False)
            return (repaired, repaired.strip() != text.strip())
        except Exception:
            continue
    return text, False


def _structural_issues(text: str) -> list[str]:
    issues: list[str] = []
    stripped = text.strip()
    if not stripped:
        issues.append("empty_content")
        return issues
    if len(stripped) < 40:
        issues.append("too_short")
    if stripped.count("```") % 2 != 0:
        issues.append("unclosed_code_fence")
    lower = stripped.lower()
    for pat in _PLACEHOLDER_PATTERNS:
        if pat in lower:
            issues.append(f"placeholder:{pat}")
    if stripped.endswith(("...", "…")) and len(stripped) < 120:
        issues.append("possibly_truncated")
    return issues


def _heuristic_score(text: str, result: dict | None) -> float:
    """Normalize `_score_result`-style signals into a 0..1 draft quality score."""
    stripped = (text or "").strip()
    if not stripped:
        return 0.0

    score = 0.35
    # Length (mirrors len(content)*2 signal, soft-capped)
    score += min(0.25, len(stripped) / 800.0 * 0.25)

    # Reasoning / structure keywords
    lower = stripped.lower()
    hits = sum(1 for kw in _REASONING_KEYWORDS if kw in lower)
    score += min(0.15, hits * 0.04)

    # Markdown structure bonus
    if any(tok in stripped for tok in ("- ", "* ", "1. ", "##", "```", "|")):
        score += 0.05

    if result:
        stop_reason = str(result.get("stop_reason") or "")
        if stop_reason == "stop":
            score += 0.05
        elif stop_reason == "error":
            score -= 0.4
        tool_calls = result.get("tool_calls") or []
        if tool_calls:
            score += 0.03

    return max(0.0, min(1.0, score))


def critique_draft(
    content: str,
    result: dict | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> ReflectionOutcome:
    """Critique a draft. Returns score, issues, and repair feedback when below threshold."""
    text = content or ""
    issues = _structural_issues(text)
    score = _heuristic_score(text, result)

    if "empty_content" in issues:
        score = 0.0
    if any(i.startswith("placeholder:") for i in issues):
        score = min(score, 0.25)
    if "unclosed_code_fence" in issues:
        score = min(score, 0.45)
    if "possibly_truncated" in issues:
        score = min(score, 0.5)
    if "too_short" in issues:
        score = min(score, 0.45)

    needs_repair = score < float(threshold)
    if not needs_repair:
        return ReflectionOutcome(score=round(score, 4), needs_repair=False, feedback="", issues=issues)

    parts: list[str] = []
    if "empty_content" in issues:
        parts.append("The previous response was empty.")
    if "too_short" in issues:
        parts.append("The previous response was too short and likely incomplete.")
    if "unclosed_code_fence" in issues:
        parts.append("A markdown code fence was left unclosed.")
    if "possibly_truncated" in issues:
        parts.append("The previous response appears truncated mid-thought.")
    for issue in issues:
        if issue.startswith("placeholder:"):
            parts.append(f"The previous response contained a placeholder ({issue.split(':', 1)[1]}).")
    if not parts:
        parts.append(
            "The previous response scored below the quality threshold "
            f"({score:.2f} < {threshold:.2f})."
        )
    parts.append(
        "Revise the answer: address every issue, give a complete self-contained response, "
        "and close any code fences. Output only the improved final answer."
    )
    return ReflectionOutcome(
        score=round(score, 4),
        needs_repair=True,
        feedback=" ".join(parts),
        issues=issues,
    )


def critic_message(feedback: str) -> dict:
    """User-role message injected into the loop for a single repair round."""
    return {
        "role": "user",
        "content": (
            "Critic feedback on your previous draft:\n"
            f"{feedback}"
        ),
    }
