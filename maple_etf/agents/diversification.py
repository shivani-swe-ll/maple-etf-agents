from maple_etf.agents.base import Agent
from maple_etf.models import AgentReport, ETFData, InvestorProfile


class DiversificationAgent(Agent):
    """Is the money spread across many companies and countries, or concentrated?"""

    name = "diversification"
    description = "Diversification analyst: number of holdings, top-10 weight, regions"

    def analyze(self, etf: ETFData, profile: InvestorProfile) -> AgentReport:
        p = etf.profile
        score = 0.0
        reasons = []

        if p.holdings_count >= 1000:
            score += 0.4
        elif p.holdings_count >= 200:
            score += 0.2
        elif p.holdings_count < 30:
            score -= 0.5
        reasons.append(f"Holds about {p.holdings_count:,} securities.")

        if p.top10_weight <= 20:
            score += 0.3
        elif p.top10_weight >= 50:
            score -= 0.4
        reasons.append(f"Top 10 holdings are {p.top10_weight:.0f}% of the fund.")

        top_region, top_share = max(p.region_mix.items(), key=lambda kv: kv[1])
        regions_used = sum(1 for v in p.region_mix.values() if v >= 5)
        if top_share >= 95:
            score -= 0.3
            reasons.append(f"Only one region: {top_share:.0f}% in {top_region}.")
        elif regions_used >= 3:
            score += 0.3
            reasons.append(f"Spread across {regions_used} regions (largest: {top_region} at {top_share:.0f}%).")
        else:
            reasons.append(f"Largest region is {top_region} at {top_share:.0f}%.")

        metrics = {
            "holdings_count": p.holdings_count,
            "top10_weight_pct": p.top10_weight,
            "largest_region": top_region,
            "largest_region_pct": top_share,
            "regions_used": regions_used,
        }
        return self._report(score, 85, reasons, metrics)
