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


def test_csv_export_has_one_row_per_etf(catalog, provider):
    import io

    import pandas as pd

    from maple_etf.graph import analyze_tickers
    from maple_etf.report import results_to_csv, results_to_json

    results = analyze_tickers(["XEQT.TO", "ZWC.TO", "VOO"], InvestorProfile(), catalog, provider)
    df = pd.read_csv(io.StringIO(results_to_csv(results)))
    assert list(df["ticker"]) == ["XEQT.TO", "ZWC.TO", "VOO"]
    for col in ("decision", "weighted_score", "tax_account_signal", "explanation"):
        assert col in df.columns
    assert '"reasons"' in results_to_json(results)


def test_cli_csv_option(tmp_path):
    out = tmp_path / "out" / "analysis.csv"
    assert main(["analyze", "XEQT.TO", "XBB.TO", "--brief", "--llm", "off", "--trace", "", "--csv", str(out)]) == 0
    assert out.read_text().count("\n") == 3  # header + 2 ETFs
