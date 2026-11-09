from maple_etf.agents.base import Agent
from maple_etf.models import AgentReport, ETFData, Goal, InvestorProfile


class IncomeAgent(Agent):
    """Distribution yield, judged against what the investor wants (income vs growth)."""

    name = "income"
    description = "Income analyst: distribution yield and frequency vs the investor's goal"

    def analyze(self, etf: ETFData, profile: InvestorProfile) -> AgentReport:
        y = etf.profile.dividend_yield
        freq = etf.profile.distribution_frequency
        reasons = [f"Distribution yield is about {y:.1f}%, paid {freq}."]

        if profile.goal == Goal.INCOME:
            score = 0.7 if y >= 3.5 else 0.2 if y >= 2.5 else -0.5
            if freq == "monthly":
                score += 0.2
                reasons.append("Monthly payments suit an income investor.")
            reasons.append("Goal is income, so yield matters a lot.")
            confidence = 80
        elif profile.goal == Goal.BALANCED:
            score = 0.3 if y >= 2.5 else 0.0
            reasons.append("Goal is balanced, so yield matters a little.")
            confidence = 60
        else:  # growth
            score = 0.0
            reasons.append("Goal is growth, so yield is not important here.")
            confidence = 40

        return self._report(score, confidence, reasons, {"dividend_yield_pct": y, "frequency": freq})
