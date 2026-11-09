"""Shared data models (Pydantic) used by every part of the system."""

from __future__ import annotations

from enum import Enum
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class Signal(str, Enum):
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"


class Action(str, Enum):
    BUY = "BUY"
    HOLD = "HOLD"
    AVOID = "AVOID"


class AccountType(str, Enum):
    TFSA = "TFSA"
    RRSP = "RRSP"
    FHSA = "FHSA"
    NON_REGISTERED = "NON_REGISTERED"


class RiskTolerance(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Goal(str, Enum):
    GROWTH = "growth"
    INCOME = "income"
    BALANCED = "balanced"


class Structure(str, Enum):
    """How the ETF is built. This matters for Canadian taxes (see the tax agent)."""

    CANADIAN_EQUITY = "canadian_equity"            # Canadian-listed, holds Canadian stocks
    CDN_LISTED_HOLDS_FOREIGN = "cdn_listed_holds_foreign"  # Canadian-listed, holds US/intl stocks directly
    CDN_LISTED_WRAPS_US_ETF = "cdn_listed_wraps_us_etf"    # Canadian-listed, holds US-listed ETFs
    US_LISTED = "us_listed"                        # Listed on a US exchange (USD)
    FIXED_INCOME = "fixed_income"                  # Bond ETF


class InvestorProfile(BaseModel):
    account: AccountType = AccountType.TFSA
    risk: RiskTolerance = RiskTolerance.MEDIUM
    goal: Goal = Goal.GROWTH


class ETFProfile(BaseModel):
    """Static facts about an ETF, loaded from data/catalog.json."""

    ticker: str
    name: str
    issuer: str
    category: str
    mer: float = Field(description="Management expense ratio, in percent (0.20 = 0.20%)")
    holdings_count: int
    top10_weight: float = Field(description="Weight of top 10 holdings, in percent")
    dividend_yield: float = Field(description="Trailing distribution yield, in percent")
    distribution_frequency: str
    equity_pct: float
    fixed_income_pct: float
    region_mix: dict[str, float] = Field(description="Percent by region: canada, us, intl, em")
    structure: Structure
    currency: str = "CAD"
    hedged: bool = False


class ETFData(BaseModel):
    """Everything an agent needs: static profile + price history up to the analysis date."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    profile: ETFProfile
    prices: pd.Series
    synthetic: bool = False


class AgentReport(BaseModel):
    agent: str
    signal: Signal
    score: float = Field(ge=-1.0, le=1.0, description="-1 very bearish ... +1 very bullish")
    confidence: int = Field(ge=0, le=100)
    reasons: list[str]
    metrics: dict[str, Any] = Field(default_factory=dict)


class RiskLimits(BaseModel):
    max_allocation_pct: float
    reasons: list[str]


class Decision(BaseModel):
    ticker: str
    action: Action
    weighted_score: float
    confidence: int
    suggested_allocation_pct: float
    vetoes: list[str] = Field(default_factory=list)
    explanation: str
    explanation_source: str = "template"  # "template" | "llm" | "llm_rejected"
    decided_by: str = "rules"             # "rules" | "llm"


class AnalysisResult(BaseModel):
    ticker: str
    as_of: str
    synthetic_data: bool
    profile: InvestorProfile
    reports: list[AgentReport]
    risk: RiskLimits
    decision: Decision

    def report(self, agent: str) -> AgentReport:
        for r in self.reports:
            if r.agent == agent:
                return r
        raise KeyError(agent)
