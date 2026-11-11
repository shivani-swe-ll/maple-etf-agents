"""ETF Comparison Agent: which of two ETFs fits THIS investor better, and why?

It runs the full agent team on both ETFs, then compares agent by agent.
The biggest score gaps become the "key differences". An optional LLM writes
the explanation; its numbers are checked with code (see explain.py).
"""

from __future__ import annotations

import json

from pydantic import BaseModel

from maple_etf.explain import grounded_llm_text
from maple_etf.graph import AgentGraph
from maple_etf.models import Action, AnalysisResult, ETFData, InvestorProfile

ACTION_RANK = {Action.BUY: 2, Action.HOLD: 1, Action.AVOID: 0}
TIE_GAP = 0.05

SYSTEM_COMPARE = (
    "You compare two ETFs for one investor. Say which one fits better and the 2 or 3 biggest reasons, "
    "in 3 to 5 sentences. If the result is a tie, say they are close and what could decide it."
)


class AgentDiff(BaseModel):
    agent: str
    score_a: float
    score_b: float
    edge: str          # ticker with the higher score, or "tie"
    reason_a: str
    reason_b: str


class ComparisonResult(BaseModel):
    ticker_a: str
    ticker_b: str
    profile: InvestorProfile
    winner: str        # a ticker, or "tie"
    score_a: float
    score_b: float
    action_a: Action
    action_b: Action
    diffs: list[AgentDiff]
    key_differences: list[AgentDiff]
    explanation: str
    explanation_source: str = "template"
    result_a: AnalysisResult
    result_b: AnalysisResult


def _winner(a: AnalysisResult, b: AnalysisResult) -> str:
    # 1) better action wins, 2) fewer vetoes wins (a vetoed ETF has a real problem for this investor),
    # 3) otherwise the higher weighted score, unless the gap is tiny.
    ra = (ACTION_RANK[a.decision.action], -len(a.decision.vetoes))
    rb = (ACTION_RANK[b.decision.action], -len(b.decision.vetoes))
    if ra != rb:
        return a.ticker if ra > rb else b.ticker
    gap = a.decision.weighted_score - b.decision.weighted_score
    if abs(gap) < TIE_GAP:
        return "tie"
    return a.ticker if gap > 0 else b.ticker


def compare_results(a: AnalysisResult, b: AnalysisResult, llm=None) -> ComparisonResult:
    diffs = []
    for rep_a in a.reports:
        rep_b = b.report(rep_a.agent)
        gap = rep_a.score - rep_b.score
        edge = "tie" if abs(gap) < TIE_GAP else (a.ticker if gap > 0 else b.ticker)
        diffs.append(AgentDiff(agent=rep_a.agent, score_a=rep_a.score, score_b=rep_b.score, edge=edge,
                               reason_a=rep_a.reasons[0], reason_b=rep_b.reasons[0]))
    key = sorted([d for d in diffs if d.edge != "tie"], key=lambda d: abs(d.score_a - d.score_b), reverse=True)[:3]
    winner = _winner(a, b)

    explanation = _template(a, b, winner, key)
    source = "template"
    context = json.dumps({
        "investor": a.profile.model_dump(mode="json"),
        "winner_by_rules": winner,
        a.ticker: {"decision": a.decision.action.value, "weighted_score": a.decision.weighted_score,
                   "agents": {r.agent: {"score": r.score, "reasons": r.reasons} for r in a.reports}},
        b.ticker: {"decision": b.decision.action.value, "weighted_score": b.decision.weighted_score,
                   "agents": {r.agent: {"score": r.score, "reasons": r.reasons} for r in b.reports}},
    }, indent=2)
    text, _ = grounded_llm_text(llm, SYSTEM_COMPARE, context)
    if text:
        explanation, source = text, "llm"

    return ComparisonResult(
        ticker_a=a.ticker, ticker_b=b.ticker, profile=a.profile, winner=winner,
        score_a=a.decision.weighted_score, score_b=b.decision.weighted_score,
        action_a=a.decision.action, action_b=b.decision.action,
        diffs=diffs, key_differences=key, explanation=explanation, explanation_source=source,
        result_a=a, result_b=b,
    )


def compare_etfs(etf_a: ETFData, etf_b: ETFData, profile: InvestorProfile, llm=None) -> ComparisonResult:
    graph = AgentGraph()  # rules for the per-ETF decisions; the LLM is only used once, for the comparison text
    return compare_results(graph.run(etf_a, profile), graph.run(etf_b, profile), llm=llm)


def _template(a: AnalysisResult, b: AnalysisResult, winner: str, key: list[AgentDiff]) -> str:
    p = a.profile
    who = f"For a {p.account.value} account, {p.risk.value} risk and a {p.goal.value} goal"
    scores = f"({a.ticker} {a.decision.action.value} {a.decision.weighted_score:+.2f} vs " \
             f"{b.ticker} {b.decision.action.value} {b.decision.weighted_score:+.2f})"
    if winner == "tie":
        parts = [f"{who}, {a.ticker} and {b.ticker} are very close {scores}."]
    else:
        loser = b.ticker if winner == a.ticker else a.ticker
        parts = [f"{who}, {winner} fits better than {loser} {scores}."]
    for v in a.decision.vetoes:
        parts.append(f"{a.ticker} has a problem: {v}")
    for v in b.decision.vetoes:
        parts.append(f"{b.ticker} has a problem: {v}")
    for d in key[:2]:
        parts.append(f"{d.agent} favours {d.edge}. {a.ticker}: {d.reason_a} {b.ticker}: {d.reason_b}")
    return " ".join(parts)
