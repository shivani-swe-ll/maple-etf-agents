import pytest

from maple_etf.backtest import run_backtest
from maple_etf.cli import main
from maple_etf.data import SampleDataProvider
from maple_etf.evals import run_evals
from maple_etf.models import InvestorProfile


@pytest.fixture
def provider(catalog):
    return SampleDataProvider({t: catalog.get(t).category for t in catalog.tickers()})


def test_backtest_runs_without_lookahead(catalog, provider):
    res = run_backtest(["XEQT.TO", "ZAG.TO", "ZEB.TO"], InvestorProfile(), catalog, provider,
                       "2025-01-01", "2026-06-30")
    assert {"agents", "buy_hold_XEQT.TO", "equal_weight_buy_hold"} <= set(res.equity.columns)
    assert res.equity["agents"].iloc[0] == pytest.approx(10_000, rel=0.01)
    weights = res.allocations.drop(columns="cash").sum(axis=1)
    assert (weights <= 100.01).all()


def test_golden_evals_pass(tmp_path):
    summary = run_evals(out_dir=tmp_path, verbose=False)
    assert summary["decision_accuracy"] >= 0.9
    assert summary["agent_accuracy"] >= 0.9
    assert summary["groundedness"] == 1.0
    assert (tmp_path / "latest.md").exists()


def test_cli_analyze(capsys):
    assert main(["analyze", "XEQT.TO", "VOO", "--brief", "--llm", "off", "--trace", ""]) == 0
    out = capsys.readouterr().out
    assert "XEQT.TO" in out and "SYNTHETIC" in out


def test_cli_unknown_ticker():
    with pytest.raises(KeyError):
        main(["analyze", "NOPE.TO", "--llm", "off", "--trace", ""])
