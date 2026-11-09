"""Groundedness check: every number the LLM writes must come from the agents' facts.

LLMs sometimes invent numbers ("the MER is 0.15%" when it is 0.20%). For a
finance tool that is the most dangerous kind of error, so we check it with code
instead of trusting the model.

Matching rule: a number written with d decimals is grounded if some number in the
context, rounded to d decimals, is equal to it. So "24.8%" matches 24.83 in the
context, but "0.15%" does NOT match 0.20.
"""

from __future__ import annotations

import re

NUMBER_RE = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?%?")


def _tokens(text: str) -> list[tuple[float, int]]:
    """(value, decimals_written) for every number in the text."""
    out = []
    for token in NUMBER_RE.findall(text):
        clean = token.replace("$", "").replace(",", "").replace("%", "")
        try:
            value = float(clean)
        except ValueError:
            continue
        decimals = len(clean.split(".")[1]) if "." in clean else 0
        out.append((value, decimals))
    return out


def extract_numbers(text: str) -> list[float]:
    return [v for v, _ in _tokens(text)]


def ungrounded_numbers(answer: str, context: str, ignore_small_ints: int = 10) -> list[float]:
    """Numbers in `answer` that cannot be found in `context`.

    Small whole numbers (0..10) are ignored: they are usually counts like "3 agents".
    """
    allowed = extract_numbers(context)
    bad = []
    for value, decimals in _tokens(answer):
        if decimals == 0 and value.is_integer() and 0 <= abs(value) <= ignore_small_ints:
            continue
        if not any(_matches(value, decimals, a) for a in allowed):
            bad.append(value)
    return bad


def _matches(value: float, decimals: int, candidate: float) -> bool:
    for c in (candidate, -candidate):  # "fell 7.0%" may be written for -7.0
        if abs(round(c, decimals) - value) < 1e-9:
            return True
    return False
