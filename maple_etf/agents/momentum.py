from maple_etf.agents.base import Agent, clamp, pct
from maple_etf.metrics import price_metrics
from maple_etf.models import AgentReport, ETFData, InvestorProfile


class MomentumAgent(Agent):
    """Trend follower: 12-month return and moving averages."""

    name = "momentum"
    description = "Trend analyst: 12-month return, 50/200-day moving averages"

    def analyze(self, etf: ETFData, profile: InvestorProfile) -> AgentReport:
        m = price_metrics(etf.prices)
        r12, vs200 = m["return_12m"], m["price_vs_sma200"]
        sma50, sma200 = m["sma50"], m["sma200"]

        if r12 is None or vs200 is None:
            return self._report(0.0, 10, ["Not enough price history (need about 1 year)."], m)

        score = 0.0
        reasons = []

        # 12-month return: +/-20% maps to +/-0.5
        score += clamp(r12 / 0.20, -1, 1) * 0.5
        reasons.append(f"12-month return is {pct(r12)}.")

        # Price above/below the 200-day average
        score += 0.3 if vs200 > 0 else -0.3
        side = "above" if vs200 > 0 else "below"
        reasons.append(f"Price is {pct(abs(vs200))} {side} its 200-day average.")

        # Golden cross / death cross
        if sma50 > sma200:
            score += 0.2
            reasons.append("50-day average is above the 200-day average (uptrend).")
        else:
            score -= 0.2
            reasons.append("50-day average is below the 200-day average (downtrend).")

        confidence = int(50 + min(abs(score), 1) * 40)
        return self._report(score, confidence, reasons, m)
