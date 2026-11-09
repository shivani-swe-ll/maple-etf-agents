from maple_etf.agents.base import Agent, pct
from maple_etf.metrics import annualized_volatility, max_drawdown
from maple_etf.models import AgentReport, ETFData, InvestorProfile, RiskTolerance

# Max volatility / drawdown each investor type is comfortable with
VOL_LIMIT = {RiskTolerance.LOW: 0.08, RiskTolerance.MEDIUM: 0.15, RiskTolerance.HIGH: 0.25}
DD_LIMIT = {RiskTolerance.LOW: -0.10, RiskTolerance.MEDIUM: -0.20, RiskTolerance.HIGH: -0.35}


class RiskAgent(Agent):
    """Checks if the ETF's ups and downs fit the investor's risk tolerance."""

    name = "risk"
    description = "Risk analyst: volatility and drawdown vs the investor's tolerance"

    def analyze(self, etf: ETFData, profile: InvestorProfile) -> AgentReport:
        vol = annualized_volatility(etf.prices)
        dd = max_drawdown(etf.prices, 252)
        if vol is None or dd is None:
            return self._report(0.0, 10, ["Not enough price history to measure risk."])

        vol_limit, dd_limit = VOL_LIMIT[profile.risk], DD_LIMIT[profile.risk]
        vol_ratio = vol / vol_limit          # < 1 means inside the comfort zone
        dd_ratio = dd / dd_limit             # both negative -> positive ratio, < 1 is fine

        score = 0.0
        score += 0.5 if vol_ratio <= 0.8 else 0.1 if vol_ratio <= 1.0 else -0.5 if vol_ratio <= 1.5 else -0.8
        # Inside the limit (even near it) is a small plus. Eval case "balanced-low-risk" found
        # that scoring it 0.0 made a fund that passes BOTH limits look neutral.
        score += 0.4 if dd_ratio <= 0.8 else 0.1 if dd_ratio <= 1.0 else -0.4

        reasons = [
            f"1-year volatility is {pct(vol)}; limit for a {profile.risk.value}-risk investor is {pct(vol_limit, 0)}.",
            f"Worst fall in the last year was {pct(dd)}; comfort limit is {pct(dd_limit, 0)}.",
        ]
        metrics = {"volatility_1y": vol, "max_drawdown_1y": dd, "vol_limit": vol_limit, "drawdown_limit": dd_limit}
        return self._report(score, 80, reasons, metrics)
