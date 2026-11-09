"""Pure functions that turn a price series into numbers the agents can reason about."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
RISK_FREE_RATE = 0.03  # rough Canadian T-bill rate, used for the Sharpe ratio


def period_return(prices: pd.Series, days: int) -> float | None:
    if len(prices) <= days:
        return None
    return float(prices.iloc[-1] / prices.iloc[-1 - days] - 1)


def annualized_volatility(prices: pd.Series, window: int = TRADING_DAYS) -> float | None:
    rets = prices.pct_change().dropna().iloc[-window:]
    if len(rets) < 20:
        return None
    return float(rets.std() * np.sqrt(TRADING_DAYS))


def max_drawdown(prices: pd.Series, window: int | None = None) -> float | None:
    """Largest peak-to-trough fall, as a negative number (e.g. -0.25 = -25%)."""
    p = prices if window is None else prices.iloc[-window:]
    if len(p) < 2:
        return None
    return float((p / p.cummax() - 1).min())


def sma(prices: pd.Series, days: int) -> float | None:
    if len(prices) < days:
        return None
    return float(prices.iloc[-days:].mean())


def cagr(prices: pd.Series) -> float | None:
    if len(prices) < 2:
        return None
    years = (prices.index[-1] - prices.index[0]).days / 365.25
    if years <= 0:
        return None
    return float((prices.iloc[-1] / prices.iloc[0]) ** (1 / years) - 1)


def sharpe(prices: pd.Series, window: int = TRADING_DAYS) -> float | None:
    rets = prices.pct_change().dropna().iloc[-window:]
    if len(rets) < 20 or rets.std() == 0:
        return None
    excess = rets.mean() * TRADING_DAYS - RISK_FREE_RATE
    return float(excess / (rets.std() * np.sqrt(TRADING_DAYS)))


def fee_drag(mer_pct: float, amount: float = 10_000, years: int = 25, gross_return: float = 0.06) -> float:
    """Dollars lost to fees over `years`, versus a zero-fee fund with the same gross return."""
    no_fee = amount * (1 + gross_return) ** years
    with_fee = amount * (1 + gross_return - mer_pct / 100) ** years
    return float(no_fee - with_fee)


def price_metrics(prices: pd.Series) -> dict[str, float | None]:
    last = float(prices.iloc[-1])
    sma50, sma200 = sma(prices, 50), sma(prices, 200)
    return {
        "last_price": last,
        "return_1m": period_return(prices, 21),
        "return_3m": period_return(prices, 63),
        "return_6m": period_return(prices, 126),
        "return_12m": period_return(prices, 252),
        "volatility_1y": annualized_volatility(prices),
        "max_drawdown_1y": max_drawdown(prices, TRADING_DAYS),
        "sma50": sma50,
        "sma200": sma200,
        "price_vs_sma200": (last / sma200 - 1) if sma200 else None,
        "sharpe_1y": sharpe(prices),
    }
