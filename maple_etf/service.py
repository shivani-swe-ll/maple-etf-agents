"""Service layer: one place that runs every feature from plain request objects.

The FastAPI app (api.py) is a thin wrapper around this class, so all the real
logic is tested without needing a web server.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from maple_etf import __version__
from maple_etf.chat import ChatSession
from maple_etf.compare import compare_etfs
from maple_etf.data import SampleDataProvider, YFinanceProvider, load_catalog
from maple_etf.graph import AgentGraph, analyze_tickers, load_etf
from maple_etf.llm import LLMClient
from maple_etf.models import AccountType, Goal, InvestorProfile, RiskTolerance
from maple_etf.portfolio import analyze_portfolio


class ProfileIn(BaseModel):
    account: AccountType = AccountType.TFSA
    risk: RiskTolerance = RiskTolerance.MEDIUM
    goal: Goal = Goal.GROWTH
    live: bool = Field(False, description="use real Yahoo Finance prices instead of synthetic sample prices")
    as_of: str | None = Field(None, description="analysis date YYYY-MM-DD (default: today)")

    def profile(self) -> InvestorProfile:
        return InvestorProfile(account=self.account, risk=self.risk, goal=self.goal)


class AnalyzeIn(ProfileIn):
    tickers: list[str] = Field(..., min_length=1, max_length=20, examples=[["XEQT.TO", "VOO"]])


class CompareIn(ProfileIn):
    a: str = Field(..., examples=["VOO"])
    b: str = Field(..., examples=["ZSP.TO"])


class PortfolioIn(ProfileIn):
    holdings: dict[str, float] = Field(..., examples=[{"XEQT.TO": 60, "ZAG.TO": 40}])


class ChatIn(ProfileIn):
    question: str = Field(..., min_length=1, max_length=1000, examples=["Why HOLD for VOO?"])


class MapleService:
    def __init__(self, llm: LLMClient | None = None, use_env_llm: bool = True):
        self.catalog = load_catalog()
        self.llm = llm if llm is not None else (LLMClient.from_env() if use_env_llm else None)
        self._sample = SampleDataProvider({t: self.catalog.get(t).category for t in self.catalog.tickers()})
        self._live = None

    def provider(self, live: bool):
        if not live:
            return self._sample
        if self._live is None:
            self._live = YFinanceProvider()
        return self._live

    def _ticker(self, t: str) -> str:
        t = t.strip().upper()
        if t not in self.catalog and f"{t}.TO" in self.catalog:
            t = f"{t}.TO"
        self.catalog.get(t)  # raises KeyError with a helpful message
        return t

    def health(self) -> dict:
        return {"status": "ok", "version": __version__, "etfs": len(self.catalog.tickers()),
                "llm": f"{self.llm.provider}/{self.llm.model}" if self.llm else None}

    def list_etfs(self) -> list[dict]:
        return [self.catalog.get(t).model_dump(mode="json") for t in self.catalog.tickers()]

    def analyze(self, req: AnalyzeIn) -> list[dict]:
        tickers = [self._ticker(t) for t in req.tickers]
        graph = AgentGraph(llm=self.llm)
        results = analyze_tickers(tickers, req.profile(), self.catalog, self.provider(req.live), graph, req.as_of)
        return [r.model_dump(mode="json") for r in results]

    def compare(self, req: CompareIn) -> dict:
        prov = self.provider(req.live)
        a = load_etf(self._ticker(req.a), self.catalog, prov, req.as_of)
        b = load_etf(self._ticker(req.b), self.catalog, prov, req.as_of)
        return compare_etfs(a, b, req.profile(), llm=self.llm).model_dump(mode="json", exclude={"result_a", "result_b"})

    def portfolio(self, req: PortfolioIn) -> dict:
        prov = self.provider(req.live)
        holdings = {self._ticker(t): w for t, w in req.holdings.items()}
        etfs = {t: load_etf(t, self.catalog, prov, req.as_of) for t in holdings}
        rep = analyze_portfolio(etfs, holdings, req.profile(), llm=self.llm)
        return rep.model_dump(mode="json", exclude={"results"})

    def chat(self, req: ChatIn) -> dict:
        session = ChatSession(self.catalog, self.provider(req.live), req.profile(), llm=self.llm)
        r = session.ask(req.question)
        return {"answer": r.text, "source": r.source, "tool_calls": r.tool_calls}
