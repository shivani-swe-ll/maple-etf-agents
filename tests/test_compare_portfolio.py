import pytest

from maple_etf.compare import compare_etfs
from maple_etf.data import make_price_path
from maple_etf.models import AccountType, ETFData, InvestorProfile, RiskTolerance
from maple_etf.portfolio import analyze_portfolio, parse_holdings


def same_prices(catalog, *tickers, **kw):
    prices = make_price_path(seed=kw.get("seed", 5), annual_vol=kw.get("vol", 0.10), annual_return=0.09)
    return [ETFData(profile=catalog.get(t), prices=prices, synthetic=True) for t in tickers]


# ---------------- comparison ----------------
def test_tfsa_prefers_canadian_listed(catalog):
    voo, zsp = same_prices(catalog, "VOO", "ZSP.TO")
    res = compare_etfs(voo, zsp, InvestorProfile(account=AccountType.TFSA))
    assert res.winner == "ZSP.TO"
    assert "VOO has a problem" in res.explanation


def test_rrsp_prefers_us_listed(catalog):
    voo, zsp = same_prices(catalog, "VOO", "ZSP.TO")
    assert compare_etfs(voo, zsp, InvestorProfile(account=AccountType.RRSP)).winner == "VOO"


def test_near_identical_etfs_tie(catalog):
    zag, xbb = same_prices(catalog, "ZAG.TO", "XBB.TO", vol=0.05)
    res = compare_etfs(zag, xbb, InvestorProfile(account=AccountType.RRSP, risk=RiskTolerance.LOW))
    assert res.winner == "tie"
    assert "very close" in res.explanation


def test_comparison_uses_grounded_llm_text(catalog, fake_llm):
    xeqt, zeb = same_prices(catalog, "XEQT.TO", "ZEB.TO")
    good = compare_etfs(xeqt, zeb, InvestorProfile(), llm=fake_llm("XEQT.TO fits better. ZEB.TO holds only 6 banks."))
    bad = compare_etfs(xeqt, zeb, InvestorProfile(), llm=fake_llm("XEQT.TO returns 31.4% a year."))
    assert good.explanation_source == "llm"
    assert bad.explanation_source == "template"


# ---------------- portfolio ----------------
def test_parse_holdings():
    assert parse_holdings(["xeqt.to=60", "ZAG.TO=40"]) == {"XEQT.TO": 60.0, "ZAG.TO": 40.0}
    with pytest.raises(ValueError):
        parse_holdings(["XEQT.TO"])


def test_simple_two_fund_portfolio_is_healthy(catalog):
    etfs = dict(zip(["XEQT.TO", "ZAG.TO"], same_prices(catalog, "XEQT.TO", "ZAG.TO", vol=0.07)))
    rep = analyze_portfolio(etfs, {"XEQT.TO": 60, "ZAG.TO": 40}, InvestorProfile())
    assert rep.weighted_mer_pct == 0.16  # 0.6 * 0.20 + 0.4 * 0.09 = 0.156, rounded once
    assert rep.annual_fee_per_10k == 16.0
    assert rep.equity_pct == 60 and rep.fixed_income_pct == 40
    assert rep.grade in ("A", "B")
    assert not [f for f in rep.findings if f.level == "warning"]


def test_fee_is_the_same_number_everywhere(catalog):
    # 50/50 XEQT + ZAG = 0.145%: the app once showed 0.14% at the top and 0.15% in the text.
    etfs = dict(zip(["XEQT.TO", "ZAG.TO"], same_prices(catalog, "XEQT.TO", "ZAG.TO", vol=0.07)))
    rep = analyze_portfolio(etfs, {"XEQT.TO": 50, "ZAG.TO": 50}, InvestorProfile())
    shown = f"{rep.weighted_mer_pct:.2f}%"
    assert shown == "0.15%"
    assert shown in rep.summary and any(shown in f.message for f in rep.findings)


def test_bad_portfolio_gets_warnings(catalog):
    tickers = ["VFV.TO", "ZSP.TO", "VOO", "ZEB.TO"]
    etfs = dict(zip(tickers, same_prices(catalog, *tickers, vol=0.18)))
    rep = analyze_portfolio(etfs, {"VFV.TO": 40, "ZSP.TO": 30, "VOO": 20, "ZEB.TO": 10},
                            InvestorProfile(risk=RiskTolerance.LOW))
    topics = {f.topic for f in rep.findings if f.level == "warning"}
    assert {"country", "stocks_vs_bonds", "holding"} <= topics
    assert any(f.topic == "overlap" for f in rep.findings)
    assert rep.grade == "D"


def test_weights_are_normalized(catalog):
    etfs = dict(zip(["XEQT.TO", "ZAG.TO"], same_prices(catalog, "XEQT.TO", "ZAG.TO")))
    rep = analyze_portfolio(etfs, {"XEQT.TO": 3, "ZAG.TO": 1}, InvestorProfile())
    assert rep.holdings == {"XEQT.TO": 75.0, "ZAG.TO": 25.0}
    assert any(f.topic == "weights" for f in rep.findings)


def test_all_in_one_plus_extra_equity_is_flagged(catalog):
    etfs = dict(zip(["XEQT.TO", "VFV.TO"], same_prices(catalog, "XEQT.TO", "VFV.TO")))
    rep = analyze_portfolio(etfs, {"XEQT.TO": 70, "VFV.TO": 30}, InvestorProfile(risk=RiskTolerance.HIGH))
    assert any("already a complete portfolio" in f.message for f in rep.findings)
