"""NEW agent: Canadian tax & account-fit analyst.

The same ETF can be a good or bad choice depending on WHERE you hold it
(TFSA, RRSP, FHSA or a non-registered account). This agent encodes common,
simplified Canadian rules of thumb. It is general information, not tax advice.
"""

from maple_etf.agents.base import Agent
from maple_etf.models import AccountType, AgentReport, ETFData, InvestorProfile, Structure

US_WITHHOLDING = 0.15  # US tax withheld on dividends paid to Canadians (treaty rate)


class TaxAccountAgent(Agent):
    name = "tax_account"
    description = "Canadian tax & account-fit analyst: TFSA / RRSP / FHSA / non-registered"

    def analyze(self, etf: ETFData, profile: InvestorProfile) -> AgentReport:
        p = etf.profile
        acct = profile.account
        mix = p.region_mix
        foreign_share = (mix.get("us", 0) + mix.get("intl", 0) + mix.get("em", 0)) / 100
        withholding_drag = foreign_share * p.dividend_yield * US_WITHHOLDING  # % per year, rough

        score = 0.0
        reasons: list[str] = []

        if p.structure == Structure.US_LISTED:
            if acct == AccountType.RRSP:
                score = 0.5
                reasons.append("US-listed ETF in an RRSP: US dividends are usually exempt from the 15% withholding tax (Canada-US treaty).")
                reasons.append("You need US dollars to buy it, so watch currency conversion costs.")
            elif acct in (AccountType.TFSA, AccountType.FHSA):
                score = -0.7
                reasons.append(f"US-listed ETF in a {acct.value}: about 15% of US dividends are withheld and cannot be recovered.")
                reasons.append("A Canadian-listed version of the same index is usually a better fit for this account.")
            else:
                score = -0.1
                reasons.append("US-listed ETF in a non-registered account: withholding can usually be claimed as a foreign tax credit.")
                reasons.append("It adds currency conversion and paperwork (T1135 if foreign property costs over $100,000).")

        elif p.structure in (Structure.CDN_LISTED_HOLDS_FOREIGN, Structure.CDN_LISTED_WRAPS_US_ETF):
            score = 0.2
            reasons.append("Canadian-listed and traded in CAD, so it is simple to hold in any account.")
            if withholding_drag > 0:
                reasons.append(
                    f"About {foreign_share * 100:.0f}% is foreign, so roughly {withholding_drag:.2f}% a year may be lost to foreign withholding tax."
                )
            if acct == AccountType.NON_REGISTERED:
                reasons.append("In a non-registered account, part of that withholding can be claimed back as a foreign tax credit.")

        elif p.structure == Structure.CANADIAN_EQUITY:
            if acct == AccountType.NON_REGISTERED:
                score = 0.6
                reasons.append("Canadian dividends in a non-registered account get the dividend tax credit, which lowers your tax.")
            else:
                score = 0.3
                reasons.append(f"Canadian stocks have no foreign withholding tax, so they fit well in a {acct.value}.")

        elif p.structure == Structure.FIXED_INCOME:
            if acct == AccountType.NON_REGISTERED:
                score = -0.5
                reasons.append("Bond interest is taxed at your full rate in a non-registered account. A TFSA or RRSP is usually a better home.")
            else:
                score = 0.4
                reasons.append(f"Bond interest grows tax-sheltered in a {acct.value}, which is a good fit.")

        if acct == AccountType.FHSA and p.equity_pct >= 90:
            score -= 0.2
            reasons.append("FHSA money is often needed within a few years for a home; a 100% stock ETF can fall a lot in that time.")

        metrics = {
            "account": acct.value,
            "structure": p.structure.value,
            "foreign_share_pct": round(foreign_share * 100, 1),
            "est_withholding_drag_pct": round(withholding_drag, 3),
        }
        return self._report(score, 75, reasons, metrics)
