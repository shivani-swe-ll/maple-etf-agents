"""Portfolio mode: check everything you own together, not one ETF at a time.

Example: 60% XEQT.TO + 40% ZAG.TO in a TFSA.
It "looks through" each ETF to find the real mix (regions, stocks vs bonds, fees),
measures the risk of the combined portfolio from prices, runs the agent team on
each holding, and lists problems in plain English with a health score.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from maple_etf.agents.risk import VOL_LIMIT
from maple_etf.explain import grounded_llm_text
from maple_etf.graph import AgentGraph
from maple_etf.metrics import TRADING_DAYS
from maple_etf.models import Action, AnalysisResult, ETFData, Goal, InvestorProfile, RiskTolerance

MAX_EQUITY = {RiskTolerance.LOW: 50, RiskTolerance.MEDIUM: 85, RiskTolerance.HIGH: 100}
SINGLE_COUNTRY_LIMIT = 70
FEE_HIGH, FEE_GOOD = 0.40, 0.25
TOO_MANY_ETFS = 6

SYSTEM_PORTFOLIO = (
    "You review a whole ETF portfolio for one investor. In 4 to 6 sentences, say what is good, "
    "what the biggest problems are, and one simple next step."
)


class Finding(BaseModel):
    level: str            # "good" | "note" | "warning"
    topic: str
    message: str


class PortfolioReport(BaseModel):
    profile: InvestorProfile
    holdings: dict[str, float] = Field(description="ticker -> weight in percent (normalized to 100)")
    weighted_mer_pct: float
    annual_fee_per_10k: float
    equity_pct: float
    fixed_income_pct: float
    region_mix: dict[str, float]
    weighted_yield_pct: float
    est_withholding_drag_pct: float
    volatility_1y: float | None
    max_drawdown_1y: float | None
    health_score: int
    grade: str
    findings: list[Finding]
    summary: str
    summary_source: str = "template"
    results: list[AnalysisResult]


def parse_holdings(items: list[str]) -> dict[str, float]:
    """['XEQT.TO=60', 'ZAG.TO=40'] -> {'XEQT.TO': 60.0, 'ZAG.TO': 40.0}"""
    out: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Use TICKER=PERCENT, for example XEQT.TO=60 (got '{item}')")
        t, w = item.split("=", 1)
        out[t.strip().upper()] = out.get(t.strip().upper(), 0.0) + float(w)
    return out


def _normalize(holdings: dict[str, float]) -> tuple[dict[str, float], float]:
    total = sum(holdings.values())
    if total <= 0:
        raise ValueError("Portfolio weights must add up to more than 0")
    return {t: w * 100 / total for t, w in holdings.items()}, total


def analyze_portfolio(
    etfs: dict[str, ETFData], holdings: dict[str, float], profile: InvestorProfile, llm=None
) -> PortfolioReport:
    weights, raw_total = _normalize(holdings)
    w = {t: v / 100 for t, v in weights.items()}
    findings: list[Finding] = []

    if abs(raw_total - 100) > 0.5:
        findings.append(Finding(level="note", topic="weights",
                                message=f"Your weights added up to {raw_total:.0f}%, so they were scaled to 100%."))

    profiles = {t: etfs[t].profile for t in weights}
    mer = sum(w[t] * profiles[t].mer for t in w)
    equity = sum(w[t] * profiles[t].equity_pct for t in w)
    bonds = sum(w[t] * profiles[t].fixed_income_pct for t in w)
    regions: dict[str, float] = {}
    for t in w:
        for region, share in profiles[t].region_mix.items():
            regions[region] = regions.get(region, 0.0) + w[t] * share
    yld = sum(w[t] * profiles[t].dividend_yield for t in w)

    graph = AgentGraph()
    results = [graph.run(etfs[t], profile) for t in weights]
    by_ticker = {r.ticker: r for r in results}
    drag = sum(w[t] * by_ticker[t].report("tax_account").metrics["est_withholding_drag_pct"] for t in w)

    # Risk of the combined portfolio (daily-rebalanced approximation)
    prices = pd.DataFrame({t: etfs[t].prices for t in weights}).ffill().dropna()
    vol = dd = None
    if len(prices) > 30:
        rets = prices.pct_change().dropna().iloc[-TRADING_DAYS:]
        port = rets.mul(pd.Series(w)).sum(axis=1)
        vol = float(port.std() * np.sqrt(TRADING_DAYS))
        value = (1 + port).cumprod()
        dd = float((value / value.cummax() - 1).min())

    # ---- checks ----
    if mer > FEE_HIGH:
        findings.append(Finding(level="warning", topic="fees",
                                message=f"Average MER is {mer:.2f}%. That is high; low-cost index ETFs are often under 0.25%."))
    elif mer <= FEE_GOOD:
        findings.append(Finding(level="good", topic="fees", message=f"Average MER is {mer:.2f}%, which is low."))
    else:
        findings.append(Finding(level="note", topic="fees", message=f"Average MER is {mer:.2f}%, which is okay."))

    top_country, top_share = max(((r, regions.get(r, 0.0)) for r in ("canada", "us")), key=lambda kv: kv[1])
    if top_share > SINGLE_COUNTRY_LIMIT:
        findings.append(Finding(level="warning", topic="country",
                                message=f"{top_share:.0f}% is in {top_country}. One country is a big part of your money."))
    elif regions.get("intl", 0) + regions.get("em", 0) >= 15 and top_share <= 60:
        findings.append(Finding(level="good", topic="country", message="Your money is spread across several parts of the world."))

    max_eq = MAX_EQUITY[profile.risk]
    if equity > max_eq + 5:
        findings.append(Finding(level="warning", topic="stocks_vs_bonds",
                                message=f"{equity:.0f}% is in stocks. For a {profile.risk.value}-risk investor, about {max_eq}% or less is common."))
    else:
        findings.append(Finding(level="good", topic="stocks_vs_bonds",
                                message=f"{equity:.0f}% stocks and {bonds:.0f}% bonds fits a {profile.risk.value}-risk investor."))

    limit = VOL_LIMIT[profile.risk]
    if vol is not None and vol > limit:
        findings.append(Finding(level="warning", topic="volatility",
                                message=f"Portfolio volatility is {vol * 100:.1f}%, above the {limit * 100:.0f}% comfort limit."))

    for r in results:
        for v in r.decision.vetoes:
            findings.append(Finding(level="warning", topic="holding", message=f"{r.ticker}: {v}"))
        if r.decision.action == Action.AVOID:
            findings.append(Finding(level="warning", topic="holding",
                                    message=f"{r.ticker}: the agents say AVOID ({r.decision.weighted_score:+.2f})."))

    findings.extend(_overlap_findings(profiles, weights))

    if len(weights) > TOO_MANY_ETFS:
        findings.append(Finding(level="note", topic="simplicity",
                                message=f"You hold {len(weights)} ETFs. Fewer, broader ETFs are often easier to manage."))
    if profile.goal == Goal.INCOME and yld < 2.5:
        findings.append(Finding(level="note", topic="income",
                                message=f"Average yield is {yld:.1f}%, which is low for an income goal."))

    warnings = sum(f.level == "warning" for f in findings)
    notes = sum(f.level == "note" for f in findings)
    score = max(0, 100 - 15 * warnings - 4 * notes)
    grade = "A" if score >= 85 else "B" if score >= 70 else "C" if score >= 50 else "D"

    summary = _template(score, grade, findings)
    source = "template"
    context = json.dumps({
        "investor": profile.model_dump(mode="json"),
        "holdings_pct": {t: round(v, 1) for t, v in weights.items()},
        "average_mer_pct": round(mer, 3), "stocks_pct": round(equity, 1), "bonds_pct": round(bonds, 1),
        "regions_pct": {k: round(v, 1) for k, v in regions.items()}, "average_yield_pct": round(yld, 2),
        "health_score": score, "grade": grade,
        "findings": [f.model_dump() for f in findings],
    }, indent=2)
    text, _ = grounded_llm_text(llm, SYSTEM_PORTFOLIO, context)
    if text:
        summary, source = text, "llm"

    return PortfolioReport(
        profile=profile, holdings={t: round(v, 2) for t, v in weights.items()},
        weighted_mer_pct=round(mer, 3), annual_fee_per_10k=round(mer / 100 * 10_000, 2),
        equity_pct=round(equity, 1), fixed_income_pct=round(bonds, 1),
        region_mix={k: round(v, 1) for k, v in regions.items()}, weighted_yield_pct=round(yld, 2),
        est_withholding_drag_pct=round(drag, 3), volatility_1y=vol, max_drawdown_1y=dd,
        health_score=score, grade=grade, findings=findings, summary=summary, summary_source=source,
        results=results,
    )


def _overlap_findings(profiles, weights) -> list[Finding]:
    out = []
    # Same exposure twice: e.g. VFV + ZSP + VOO all track the S&P 500
    groups: dict[tuple, list[str]] = {}
    for t, p in profiles.items():
        key = (p.category, tuple(sorted((k, round(v)) for k, v in p.region_mix.items())))
        groups.setdefault(key, []).append(t)
    for tickers in groups.values():
        if len(tickers) > 1:
            out.append(Finding(level="note", topic="overlap",
                               message=f"{' and '.join(sorted(tickers))} hold almost the same thing. You may only need one."))
    # An all-in-one ETF plus extra stock ETFs on top
    all_in_one = [t for t, p in profiles.items() if "portfolio" in p.category]
    extra_equity = [t for t, p in profiles.items() if "portfolio" not in p.category and p.equity_pct >= 90]
    if all_in_one and extra_equity:
        out.append(Finding(level="note", topic="overlap",
                           message=f"{', '.join(all_in_one)} is already a complete portfolio. "
                                   f"{', '.join(extra_equity)} adds more of the same kind of stocks on top."))
    return out


def _template(score: int, grade: str, findings: list[Finding]) -> str:
    warnings = [f.message for f in findings if f.level == "warning"]
    goods = [f.message for f in findings if f.level == "good"]
    parts = [f"Portfolio health: {score}/100 (grade {grade})."]
    if goods:
        parts.append("Good: " + " ".join(goods[:2]))
    if warnings:
        parts.append("Fix first: " + warnings[0])
        if len(warnings) > 1:
            parts.append(f"There are {len(warnings) - 1} more warnings below.")
    else:
        parts.append("No big problems found.")
    return " ".join(parts)
