"""Base class for analyst agents.

Every analyst is deterministic: same input -> same output. That makes the system
testable and lets the evals measure it. The (optional) LLM only works on top of
these reports, in the portfolio manager.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from maple_etf.models import AgentReport, ETFData, InvestorProfile, Signal

BULLISH_AT = 0.2
BEARISH_AT = -0.2


def clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def pct(x: float | None, digits: int = 1) -> str:
    """Format a fraction (0.123) as a percent string ('12.3%')."""
    return "n/a" if x is None else f"{x * 100:.{digits}f}%"


class Agent(ABC):
    name: str = "agent"
    description: str = ""

    @abstractmethod
    def analyze(self, etf: ETFData, profile: InvestorProfile) -> AgentReport: ...

    def _report(
        self, score: float, confidence: int, reasons: list[str], metrics: dict[str, Any] | None = None
    ) -> AgentReport:
        score = round(clamp(score), 3)
        if score >= BULLISH_AT:
            signal = Signal.BULLISH
        elif score <= BEARISH_AT:
            signal = Signal.BEARISH
        else:
            signal = Signal.NEUTRAL
        return AgentReport(
            agent=self.name,
            signal=signal,
            score=score,
            confidence=max(0, min(100, int(confidence))),
            reasons=reasons,
            metrics=metrics or {},
        )
