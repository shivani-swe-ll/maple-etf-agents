"""REST API (FastAPI).

Run locally:   uvicorn maple_etf.api:app --reload
Docs:          http://localhost:8000/docs   (interactive Swagger page)
"""

from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from maple_etf import __version__
from maple_etf.service import AnalyzeIn, ChatIn, CompareIn, MapleService, PortfolioIn

load_dotenv()

app = FastAPI(
    title="Maple ETF Agents API",
    version=__version__,
    description="Multi-agent AI analyst for Canadian ETFs. Educational project, not financial or tax advice.",
)
service = MapleService()


def _run(fn, req):
    try:
        return fn(req)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc).strip('"')) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict:
    return service.health()


@app.get("/etfs")
def etfs() -> list[dict]:
    return service.list_etfs()


@app.post("/analyze")
def analyze(req: AnalyzeIn) -> list[dict]:
    return _run(service.analyze, req)


@app.post("/compare")
def compare(req: CompareIn) -> dict:
    return _run(service.compare, req)


@app.post("/portfolio")
def portfolio(req: PortfolioIn) -> dict:
    return _run(service.portfolio, req)


@app.post("/chat")
def chat(req: ChatIn) -> dict:
    return _run(service.chat, req)
