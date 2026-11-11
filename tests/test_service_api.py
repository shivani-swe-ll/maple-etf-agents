import pytest

from maple_etf.service import AnalyzeIn, ChatIn, CompareIn, MapleService, PortfolioIn


@pytest.fixture(scope="module")
def service():
    return MapleService(use_env_llm=False)


def test_health_and_list(service):
    assert service.health()["status"] == "ok"
    assert len(service.list_etfs()) == 17


def test_analyze_accepts_short_tickers(service):
    out = service.analyze(AnalyzeIn(tickers=["xeqt", "VOO"]))
    assert [r["ticker"] for r in out] == ["XEQT.TO", "VOO"]
    assert out[1]["decision"]["action"] == "HOLD"


def test_compare_portfolio_chat(service):
    assert service.compare(CompareIn(a="VOO", b="ZSP.TO"))["winner"] in ("ZSP.TO", "VOO", "tie")
    rep = service.portfolio(PortfolioIn(holdings={"XEQT.TO": 60, "ZAG.TO": 40}))
    assert rep["grade"] in "ABCD" and "results" not in rep
    assert service.chat(ChatIn(question="Why HOLD for VOO?"))["source"] == "rules"


def test_unknown_ticker_raises_keyerror(service):
    with pytest.raises(KeyError):
        service.analyze(AnalyzeIn(tickers=["NOPE"]))


# ---- HTTP layer: runs where fastapi + httpx are installed (CI installs them via .[dev]) ----
def test_http_endpoints(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    from fastapi.testclient import TestClient

    from maple_etf.api import app

    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    assert len(client.get("/etfs").json()) == 17
    r = client.post("/analyze", json={"tickers": ["XEQT.TO"], "account": "TFSA"})
    assert r.status_code == 200 and r.json()[0]["decision"]["action"] in ("BUY", "HOLD", "AVOID")
    assert client.post("/analyze", json={"tickers": ["NOPE"]}).status_code == 404
    assert client.post("/analyze", json={"tickers": ["XEQT.TO"], "account": "BANK"}).status_code == 422
    assert client.post("/compare", json={"a": "VOO", "b": "ZSP.TO"}).status_code == 200
    assert client.post("/portfolio", json={"holdings": {"XEQT.TO": 60, "ZAG.TO": 40}}).status_code == 200
    assert "answer" in client.post("/chat", json={"question": "xeqt or veqt?"}).json()
