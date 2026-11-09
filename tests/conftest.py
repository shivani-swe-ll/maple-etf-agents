import pytest

from maple_etf.data import load_catalog, make_price_path
from maple_etf.llm import LLMResponse
from maple_etf.models import ETFData, InvestorProfile


@pytest.fixture(scope="session")
def catalog():
    return load_catalog()


@pytest.fixture
def make_etf(catalog):
    def _make(ticker="XEQT.TO", annual_return=0.08, annual_vol=0.10, seed=1, days=504, **overrides):
        base = catalog.get(ticker).model_copy(update=overrides)
        prices = make_price_path(annual_return=annual_return, annual_vol=annual_vol, seed=seed, days=days)
        return ETFData(profile=base, prices=prices, synthetic=True)

    return _make


@pytest.fixture
def profile():
    return InvestorProfile()


class FakeLLM:
    """Stands in for a real model so tests are fast, free and offline."""

    provider, model = "fake", "fake-model"

    def __init__(self, reply: str | Exception):
        self.reply = reply
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
        self.prompts = []

    def complete(self, system, user, max_tokens=600):
        self.prompts.append(user)
        self.usage["calls"] += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return LLMResponse(self.reply, 100, 50, 5)


@pytest.fixture
def fake_llm():
    return FakeLLM
