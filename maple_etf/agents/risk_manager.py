"""Risk manager: sets the MAXIMUM share of a portfolio one ETF may take.

It does not say buy or sell. It puts a ceiling on position size so that one bad
ETF cannot sink the whole portfolio.
"""

from maple_etf.agents.risk import VOL_LIMIT
from maple_etf.metrics import annualized_volatility
from maple_etf.models import ETFData, InvestorProfile, RiskLimits, RiskTolerance

BASE_CAP = {RiskTolerance.LOW: 20.0, RiskTolerance.MEDIUM: 30.0, RiskTolerance.HIGH: 40.0}


class RiskManager:
    name = "risk_manager"

    def limits(self, etf: ETFData, profile: InvestorProfile) -> RiskLimits:
        p = etf.profile
        reasons = []

        if "portfolio" in p.category:
            cap = 100.0
            reasons.append("All-in-one portfolio ETF: designed to be a whole portfolio, so up to 100% is allowed.")
        elif p.structure.value == "fixed_income" and profile.risk == RiskTolerance.LOW:
            cap = 60.0
            reasons.append("Broad bond ETF for a low-risk investor: up to 60%.")
        else:
            cap = BASE_CAP[profile.risk]
            reasons.append(f"Base cap for a {profile.risk.value}-risk investor: {cap:.0f}%.")

        if p.holdings_count < 30:
            cap = min(cap, 10.0)
            reasons.append(f"Only {p.holdings_count} holdings (concentrated), so capped at 10%.")

        vol = annualized_volatility(etf.prices)
        limit = VOL_LIMIT[profile.risk]
        if vol is not None and vol > limit:
            scaled = cap * limit / vol
            reasons.append(f"Volatility {vol * 100:.1f}% is above the {limit * 100:.0f}% limit, so cap is cut to {scaled:.0f}%.")
            cap = scaled

        return RiskLimits(max_allocation_pct=round(cap, 1), reasons=reasons)
