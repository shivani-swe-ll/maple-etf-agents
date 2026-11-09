"""The agent workflow.

    data ──► [cost, momentum, risk, diversification, income, tax_account]  (run in parallel)
                              │
                              ▼
                        risk manager (position cap)
                              │
                              ▼
                     portfolio manager (decision)

Plain Python on purpose: easy to read, easy to test, no framework lock-in.
Every run can write a JSONL trace (one line per step) for debugging and evals.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd

from maple_etf.agents import Agent, PortfolioManager, RiskManager, default_analysts
from maple_etf.data import Catalog, PriceProvider
from maple_etf.llm import LLMClient
from maple_etf.models import AnalysisResult, ETFData, InvestorProfile

LOOKBACK_DAYS = 550  # calendar days of history pulled for each analysis (> 1 trading year)


class AgentGraph:
    def __init__(
        self,
        analysts: list[Agent] | None = None,
        llm: LLMClient | None = None,
        llm_mode: str = "explain",
        trace_path: str | Path | None = None,
    ):
        self.analysts = analysts or default_analysts()
        self.risk_manager = RiskManager()
        self.pm = PortfolioManager(llm=llm, mode=llm_mode)
        self.trace_path = Path(trace_path) if trace_path else None

    def run(self, etf: ETFData, profile: InvestorProfile) -> AnalysisResult:
        events = []

        def timed(agent: Agent):
            t0 = time.perf_counter()
            report = agent.analyze(etf, profile)
            events.append({"step": agent.name, "ms": round((time.perf_counter() - t0) * 1000, 2),
                           "signal": report.signal.value, "score": report.score})
            return report

        with ThreadPoolExecutor(max_workers=len(self.analysts)) as pool:
            reports = list(pool.map(timed, self.analysts))

        risk = self.risk_manager.limits(etf, profile)
        events.append({"step": "risk_manager", "max_allocation_pct": risk.max_allocation_pct})

        t0 = time.perf_counter()
        decision = self.pm.decide(etf.profile.ticker, reports, risk, profile)
        events.append({"step": "portfolio_manager", "ms": round((time.perf_counter() - t0) * 1000, 2),
                       "action": decision.action.value, "decided_by": decision.decided_by,
                       "explanation_source": decision.explanation_source, **self.pm.last_trace})

        result = AnalysisResult(
            ticker=etf.profile.ticker,
            as_of=str(etf.prices.index[-1].date()),
            synthetic_data=etf.synthetic,
            profile=profile,
            reports=reports,
            risk=risk,
            decision=decision,
        )
        self._write_trace(result, events)
        return result

    def _write_trace(self, result: AnalysisResult, events: list[dict]) -> None:
        if not self.trace_path:
            return
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        line = {"ts": datetime.now().isoformat(timespec="seconds"), "ticker": result.ticker,
                "as_of": result.as_of, "profile": result.profile.model_dump(mode="json"), "events": events}
        with self.trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, default=str) + "\n")


def load_etf(
    ticker: str, catalog: Catalog, provider: PriceProvider, as_of: str | pd.Timestamp | None = None
) -> ETFData:
    end = pd.Timestamp(as_of) if as_of else pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=LOOKBACK_DAYS)
    prices = provider.get_prices(ticker, str(start.date()), str(end.date()))
    if len(prices) < 30:
        raise ValueError(f"Too little price history for {ticker} ({len(prices)} days)")
    return ETFData(profile=catalog.get(ticker), prices=prices, synthetic=provider.synthetic)


def analyze_tickers(
    tickers: list[str],
    profile: InvestorProfile,
    catalog: Catalog,
    provider: PriceProvider,
    graph: AgentGraph | None = None,
    as_of: str | None = None,
) -> list[AnalysisResult]:
    graph = graph or AgentGraph()
    return [graph.run(load_etf(t, catalog, provider, as_of), profile) for t in tickers]
