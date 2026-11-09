from maple_etf.agents import (
    CostAgent,
    DiversificationAgent,
    IncomeAgent,
    MomentumAgent,
    RiskAgent,
    RiskManager,
    TaxAccountAgent,
)
from maple_etf.models import AccountType, Goal, InvestorProfile, RiskTolerance, Signal


def test_cost_agent_rewards_low_fees(make_etf, profile):
    assert CostAgent().analyze(make_etf("ZAG.TO"), profile).signal == Signal.BULLISH
    assert CostAgent().analyze(make_etf("ZEB.TO"), profile).signal == Signal.BEARISH


def test_momentum_needs_history(make_etf, profile):
    r = MomentumAgent().analyze(make_etf(days=60), profile)
    assert r.signal == Signal.NEUTRAL and r.confidence <= 10


def test_momentum_sees_crash(make_etf, profile, catalog):
    from maple_etf.data import make_price_path
    from maple_etf.models import ETFData

    prices = make_price_path(annual_vol=0.08, seed=4, crash_pct=0.3, crash_days_from_end=40)
    etf = ETFData(profile=catalog.get("VFV.TO"), prices=prices)
    assert MomentumAgent().analyze(etf, profile).signal == Signal.BEARISH


def test_risk_depends_on_investor(make_etf):
    etf = make_etf("VFV.TO", annual_vol=0.20, seed=12)
    low = RiskAgent().analyze(etf, InvestorProfile(risk=RiskTolerance.LOW))
    high = RiskAgent().analyze(etf, InvestorProfile(risk=RiskTolerance.HIGH))
    assert low.score < high.score
    assert low.signal == Signal.BEARISH


def test_diversification(make_etf, profile):
    assert DiversificationAgent().analyze(make_etf("XEQT.TO"), profile).signal == Signal.BULLISH
    assert DiversificationAgent().analyze(make_etf("ZEB.TO"), profile).signal == Signal.BEARISH


def test_income_follows_goal(make_etf):
    etf = make_etf("VDY.TO")
    assert IncomeAgent().analyze(etf, InvestorProfile(goal=Goal.INCOME)).signal == Signal.BULLISH
    assert IncomeAgent().analyze(etf, InvestorProfile(goal=Goal.GROWTH)).signal == Signal.NEUTRAL


def test_tax_agent_us_listed_by_account(make_etf):
    etf = make_etf("VOO")
    agent = TaxAccountAgent()
    assert agent.analyze(etf, InvestorProfile(account=AccountType.TFSA)).signal == Signal.BEARISH
    assert agent.analyze(etf, InvestorProfile(account=AccountType.RRSP)).signal == Signal.BULLISH
    assert agent.analyze(etf, InvestorProfile(account=AccountType.NON_REGISTERED)).signal == Signal.NEUTRAL


def test_tax_agent_bonds_and_canadian_dividends(make_etf):
    agent = TaxAccountAgent()
    nonreg = InvestorProfile(account=AccountType.NON_REGISTERED)
    assert agent.analyze(make_etf("ZAG.TO"), nonreg).signal == Signal.BEARISH
    assert agent.analyze(make_etf("VDY.TO"), nonreg).signal == Signal.BULLISH


def test_risk_manager_caps_concentrated_etf(make_etf, profile):
    assert RiskManager().limits(make_etf("ZEB.TO"), profile).max_allocation_pct <= 10
    assert RiskManager().limits(make_etf("XEQT.TO"), profile).max_allocation_pct == 100


def test_scores_stay_in_range(make_etf, profile, catalog):
    agents = [CostAgent(), MomentumAgent(), RiskAgent(), DiversificationAgent(), IncomeAgent(), TaxAccountAgent()]
    for t in catalog.tickers():
        for a in agents:
            r = a.analyze(make_etf(t, annual_vol=0.3, seed=99), profile)
            assert -1 <= r.score <= 1 and r.reasons
