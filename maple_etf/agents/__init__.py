from maple_etf.agents.base import Agent
from maple_etf.agents.cost import CostAgent
from maple_etf.agents.diversification import DiversificationAgent
from maple_etf.agents.income import IncomeAgent
from maple_etf.agents.momentum import MomentumAgent
from maple_etf.agents.portfolio_manager import PortfolioManager
from maple_etf.agents.risk import RiskAgent
from maple_etf.agents.risk_manager import RiskManager
from maple_etf.agents.tax_account import TaxAccountAgent


def default_analysts() -> list[Agent]:
    return [CostAgent(), MomentumAgent(), RiskAgent(), DiversificationAgent(), IncomeAgent(), TaxAccountAgent()]


ANALYST_NAMES = [a.name for a in default_analysts()]

__all__ = [
    "Agent",
    "CostAgent",
    "MomentumAgent",
    "RiskAgent",
    "DiversificationAgent",
    "IncomeAgent",
    "TaxAccountAgent",
    "RiskManager",
    "PortfolioManager",
    "default_analysts",
    "ANALYST_NAMES",
]
