"""Price data providers.

* SampleDataProvider -> synthetic prices. Works offline, no API keys. NOT real market data.
* YFinanceProvider   -> real daily prices from Yahoo Finance (needs `pip install yfinance`).
"""

from __future__ import annotations

import zlib
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# Rough (return, volatility) per category, only used to make the synthetic demo look sensible.
_SAMPLE_PARAMS: dict[str, tuple[float, float]] = {
    "all_equity_portfolio": (0.09, 0.13),
    "growth_portfolio": (0.075, 0.11),
    "balanced_portfolio": (0.06, 0.08),
    "us_equity": (0.11, 0.16),
    "canadian_equity": (0.08, 0.14),
    "canadian_dividend": (0.075, 0.15),
    "canadian_sector": (0.07, 0.22),
    "canadian_bonds": (0.025, 0.06),
    "global_equity": (0.10, 0.14),
    "international_equity": (0.07, 0.14),
    "covered_call": (0.06, 0.12),
}


def make_price_path(
    annual_return: float = 0.07,
    annual_vol: float = 0.12,
    days: int = 504,
    seed: int = 0,
    start_price: float = 100.0,
    end_date: str | pd.Timestamp = "2026-10-30",
    crash_pct: float = 0.0,
    crash_days_from_end: int = 0,
) -> pd.Series:
    """Build a reproducible random-walk price series (geometric Brownian motion).

    crash_pct / crash_days_from_end let eval scenarios add a sharp fall near the end,
    e.g. crash_pct=0.30 spread over the last 40 days.
    """
    rng = np.random.default_rng(seed)
    daily_mu = annual_return / TRADING_DAYS
    daily_sigma = annual_vol / np.sqrt(TRADING_DAYS)
    rets = rng.normal(daily_mu - 0.5 * daily_sigma**2, daily_sigma, size=days)
    if crash_pct > 0 and crash_days_from_end > 0:
        per_day = np.log(1 - crash_pct) / crash_days_from_end
        rets[-crash_days_from_end:] += per_day
    prices = start_price * np.exp(np.cumsum(rets))
    index = pd.bdate_range(end=pd.Timestamp(end_date), periods=days)
    return pd.Series(prices, index=index, name="close")


class PriceProvider(ABC):
    synthetic: bool = False

    @abstractmethod
    def get_prices(self, ticker: str, start: str, end: str) -> pd.Series:
        """Daily closing prices (adjusted for distributions when possible), indexed by date."""


class SampleDataProvider(PriceProvider):
    """Deterministic synthetic prices so the project runs with zero setup."""

    synthetic = True

    def __init__(self, category_lookup: dict[str, str] | None = None, history_start: str = "2018-01-02"):
        self.category_lookup = {k.upper(): v for k, v in (category_lookup or {}).items()}
        self.history_start = pd.Timestamp(history_start)
        self._cache: dict[str, pd.Series] = {}

    def _full_history(self, ticker: str) -> pd.Series:
        key = ticker.upper()
        if key not in self._cache:
            category = self.category_lookup.get(key, "all_equity_portfolio")
            ret, vol = _SAMPLE_PARAMS.get(category, (0.07, 0.12))
            end = pd.Timestamp("2026-10-30")
            days = len(pd.bdate_range(self.history_start, end))
            self._cache[key] = make_price_path(
                annual_return=ret, annual_vol=vol, days=days, seed=zlib.crc32(key.encode()), end_date=end
            )
        return self._cache[key]

    def get_prices(self, ticker: str, start: str, end: str) -> pd.Series:
        s = self._full_history(ticker)
        return s.loc[pd.Timestamp(start) : pd.Timestamp(end)]


class YFinanceProvider(PriceProvider):
    """Real prices from Yahoo Finance. Canadian ETFs use the .TO suffix (e.g. XEQT.TO)."""

    synthetic = False

    def __init__(self) -> None:
        try:
            import yfinance  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError("Live data needs yfinance. Run: pip install yfinance") from exc
        self._cache: dict[tuple[str, str, str], pd.Series] = {}

    def get_prices(self, ticker: str, start: str, end: str) -> pd.Series:  # pragma: no cover - network
        import yfinance as yf

        key = (ticker.upper(), start, end)
        if key not in self._cache:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            if df is None or df.empty:
                raise ValueError(f"No price data returned for {ticker} between {start} and {end}")
            close = df["Close"]
            if isinstance(close, pd.DataFrame):  # newer yfinance returns one column per ticker
                close = close.iloc[:, 0]
            self._cache[key] = close.dropna().rename("close")
        return self._cache[key]
