import pandas as pd
import pytest

from maple_etf import metrics
from maple_etf.data import make_price_path


def series(values):
    return pd.Series(values, index=pd.bdate_range("2025-01-01", periods=len(values)), dtype=float)


def test_period_return():
    s = series([100] * 21 + [110])
    assert metrics.period_return(s, 21) == pytest.approx(0.10)
    assert metrics.period_return(series([1, 2]), 21) is None


def test_max_drawdown_is_negative_peak_to_trough():
    s = series([100, 120, 90, 130])
    assert metrics.max_drawdown(s) == pytest.approx(90 / 120 - 1)


def test_fee_drag_grows_with_mer():
    assert metrics.fee_drag(0.0) == pytest.approx(0.0)
    assert metrics.fee_drag(1.0) > metrics.fee_drag(0.2) > 0


def test_price_path_is_reproducible():
    a = make_price_path(seed=7)
    b = make_price_path(seed=7)
    assert a.equals(b)
    assert len(a) == 504


def test_crash_makes_price_fall():
    calm = make_price_path(seed=3, annual_vol=0.05)
    crash = make_price_path(seed=3, annual_vol=0.05, crash_pct=0.3, crash_days_from_end=20)
    assert crash.iloc[-1] < calm.iloc[-1] * 0.75


def test_price_metrics_keys():
    m = metrics.price_metrics(make_price_path())
    for key in ("return_12m", "volatility_1y", "max_drawdown_1y", "sma50", "sma200", "sharpe_1y"):
        assert m[key] is not None
