"""Shared helper: ask the LLM to write text, then check every number with code.

Used by the comparison agent, portfolio mode and chat. If the LLM is missing,
fails, or invents a number, the caller gets None and uses its own template text.
"""

from __future__ import annotations

import logging

from maple_etf.grounding import ungrounded_numbers

log = logging.getLogger(__name__)

PLAIN_ENGLISH = (
    "Write for a regular Canadian investor whose first language may not be English. "
    "Use short, simple sentences. Use ONLY facts and numbers from the data you are given. "
    "Never invent numbers. Never promise returns. Plain text, no markdown."
)


def grounded_llm_text(llm, system: str, context: str, max_tokens: int = 600) -> tuple[str | None, dict]:
    """Return (text, trace). text is None when the LLM output cannot be trusted."""
    if llm is None:
        return None, {}
    try:
        resp = llm.complete(f"{system}\n\n{PLAIN_ENGLISH}", context, max_tokens=max_tokens)
    except Exception as exc:  # network error, auth error...
        log.warning("LLM call failed: %s", exc)
        return None, {"llm_error": str(exc)}
    bad = ungrounded_numbers(resp.text, context)
    trace = {"llm_latency_ms": resp.latency_ms, "input_tokens": resp.input_tokens,
             "output_tokens": resp.output_tokens, "ungrounded_numbers": bad}
    if not resp.text or bad:
        log.warning("LLM text rejected; ungrounded numbers: %s", bad)
        return None, trace
    return resp.text, trace
