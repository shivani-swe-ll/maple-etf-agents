from maple_etf.agents.base import Agent
from maple_etf.metrics import fee_drag
from maple_etf.models import AgentReport, ETFData, InvestorProfile


class CostAgent(Agent):
    """Looks at fees (MER). Fees are one of the few things an investor fully controls."""

    name = "cost"
    description = "Fee analyst: MER level and long-term fee drag"

    def analyze(self, etf: ETFData, profile: InvestorProfile) -> AgentReport:
        mer = etf.profile.mer
        drag = fee_drag(mer)

        if mer <= 0.10:
            score, label = 0.9, "very low"
        elif mer <= 0.25:
            score, label = 0.5, "low"
        elif mer <= 0.50:
            score, label = 0.0, "average"
        elif mer <= 1.00:
            score, label = -0.5, "high"
        else:
            score, label = -0.9, "very high"

        reasons = [
            f"MER is {mer:.2f}%, which is {label} for an ETF.",
            f"On $10,000 over 25 years (6% gross return), fees cost about ${drag:,.0f}.",
        ]
        if etf.profile.hedged:
            score -= 0.1
            reasons.append(
                "It is currency-hedged: this removes USD/CAD swings but has a small extra cost that is not in the MER."
            )
        metrics = {"mer_pct": mer, "fee_drag_25y_on_10k": round(drag, 2), "hedged": etf.profile.hedged}
        return self._report(score, 90, reasons, metrics)
