"""Simple monthly backtester.

At each month end the agents see ONLY data up to that day (no look-ahead),
the portfolio is rebalanced to the BUY decisions, and the rest stays in cash.
Results are compared with buying and holding a benchmark ETF.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from maple_etf.data import Catalog, PriceProvider
from maple_etf.graph import LOOKBACK_DAYS, AgentGraph
from maple_etf.metrics import TRADING_DAYS, cagr, max_drawdown
from maple_etf.models import Action, ETFData, InvestorProfile


@dataclass
class BacktestResult:
    equity: pd.DataFrame          # one column per strategy, daily portfolio value
    stats: pd.DataFrame           # summary metrics per strategy
    allocations: pd.DataFrame     # target weights (%) at each rebalance date


def _stats(series: pd.Series) -> dict[str, float]:
    rets = series.pct_change().dropna()
    vol = float(rets.std() * np.sqrt(TRADING_DAYS)) if len(rets) > 1 else float("nan")
    growth = cagr(series) or 0.0
    return {
        "total_return_pct": round((series.iloc[-1] / series.iloc[0] - 1) * 100, 2),
        "cagr_pct": round(growth * 100, 2),
        "volatility_pct": round(vol * 100, 2),
        "max_drawdown_pct": round((max_drawdown(series) or 0.0) * 100, 2),
        "sharpe": round((growth - 0.03) / vol, 2) if vol and vol > 0 else float("nan"),
    }


def run_backtest(
    tickers: list[str],
    profile: InvestorProfile,
    catalog: Catalog,
    provider: PriceProvider,
    start: str,
    end: str,
    benchmark: str = "XEQT.TO",
    initial_cash: float = 10_000.0,
    cost_bps: float = 5.0,
) -> BacktestResult:
    universe = sorted(set(tickers) | {benchmark})
    fetch_start = str((pd.Timestamp(start) - pd.Timedelta(days=LOOKBACK_DAYS)).date())
    prices = pd.DataFrame({t: provider.get_prices(t, fetch_start, end) for t in universe}).ffill().dropna()
    trading = prices.loc[pd.Timestamp(start) : pd.Timestamp(end)]
    if len(trading) < 40:
        raise ValueError("Backtest window is too short; use at least 2 months.")

    month_ends = set(trading.groupby(trading.index.to_period("M")).tail(1).index)
    rebalance_days = {trading.index[0]} | month_ends

    graph = AgentGraph()  # rules only: fast and reproducible
    cash, shares = initial_cash, {t: 0.0 for t in tickers}
    values, alloc_rows = [], []

    for day, row in trading.iterrows():
        if day in rebalance_days:
            total = cash + sum(shares[t] * row[t] for t in tickers)
            targets = {}
            for t in tickers:
                hist = prices[t].loc[day - pd.Timedelta(days=LOOKBACK_DAYS) : day]
                etf = ETFData(profile=catalog.get(t), prices=hist, synthetic=provider.synthetic)
                d = graph.run(etf, profile).decision
                if d.action == Action.BUY:
                    targets[t] = d.suggested_allocation_pct
            scale = 100 / sum(targets.values()) if sum(targets.values()) > 100 else 1.0
            targets = {t: w * scale for t, w in targets.items()}

            turnover = 0.0
            new_shares = {}
            for t in tickers:
                target_value = total * targets.get(t, 0.0) / 100
                turnover += abs(target_value - shares[t] * row[t])
                new_shares[t] = target_value / row[t]
            fees = turnover * cost_bps / 10_000
            shares = new_shares
            cash = total - sum(shares[t] * row[t] for t in tickers) - fees
            alloc_rows.append({"date": day.date(), **{t: round(targets.get(t, 0.0), 1) for t in tickers},
                               "cash": round(100 - sum(targets.values()), 1)})

        values.append(cash + sum(shares[t] * row[t] for t in tickers))

    equity = pd.DataFrame(
        {
            "agents": values,
            f"buy_hold_{benchmark}": initial_cash * trading[benchmark] / trading[benchmark].iloc[0],
            "equal_weight_buy_hold": initial_cash * (trading[tickers] / trading[tickers].iloc[0]).mean(axis=1),
        },
        index=trading.index,
    )
    stats = pd.DataFrame({c: _stats(equity[c]) for c in equity.columns}).T
    return BacktestResult(equity=equity, stats=stats, allocations=pd.DataFrame(alloc_rows).set_index("date"))
