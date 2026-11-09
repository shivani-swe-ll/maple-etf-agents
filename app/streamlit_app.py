"""Web UI.  Run:  pip install streamlit  &&  streamlit run app/streamlit_app.py"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maple_etf.backtest import run_backtest  # noqa: E402
from maple_etf.data import SampleDataProvider, YFinanceProvider, load_catalog  # noqa: E402
from maple_etf.graph import AgentGraph, analyze_tickers  # noqa: E402
from maple_etf.llm import LLMClient  # noqa: E402
from maple_etf.models import AccountType, Goal, InvestorProfile, RiskTolerance  # noqa: E402

load_dotenv()
st.set_page_config(page_title="Maple ETF Agents", page_icon="🍁", layout="wide")
st.title("🍁 Maple ETF Agents")
st.caption("Multi-agent AI analyst for Canadian ETFs. Educational project, not financial or tax advice.")

catalog = load_catalog()

with st.sidebar:
    st.header("Investor")
    account = st.selectbox("Account", [a.value for a in AccountType])
    risk = st.selectbox("Risk tolerance", [r.value for r in RiskTolerance], index=1)
    goal = st.selectbox("Goal", [g.value for g in Goal])
    st.header("Data")
    live = st.toggle("Live Yahoo Finance prices", value=False)
    llm = LLMClient.from_env()
    use_llm = st.toggle("LLM explanations", value=llm is not None, disabled=llm is None,
                        help="Set LLM_PROVIDER and LLM_MODEL in .env to enable")
    tickers = st.multiselect("ETFs", catalog.tickers(), default=["XEQT.TO", "VFV.TO", "ZAG.TO", "VOO"])

profile = InvestorProfile(account=AccountType(account), risk=RiskTolerance(risk), goal=Goal(goal))
provider = YFinanceProvider() if live else SampleDataProvider({t: catalog.get(t).category for t in catalog.tickers()})
if provider.synthetic:
    st.warning("Using synthetic sample prices. Turn on live prices for real market data.")

tab_analyze, tab_backtest = st.tabs(["Analyze", "Backtest"])

with tab_analyze:
    if st.button("Run agents", type="primary", disabled=not tickers):
        graph = AgentGraph(llm=llm if use_llm else None)
        with st.spinner("Agents are working..."):
            results = analyze_tickers(tickers, profile, catalog, provider, graph)
        st.dataframe(
            pd.DataFrame(
                [{"ETF": r.ticker, "Decision": r.decision.action.value, "Score": r.decision.weighted_score,
                  "Confidence %": r.decision.confidence, "Max allocation %": r.decision.suggested_allocation_pct,
                  **{x.agent: x.signal.value for x in r.reports}} for r in results]
            ),
            hide_index=True,
            use_container_width=True,
        )
        for r in results:
            with st.expander(f"{r.ticker}: {r.decision.action.value}"):
                st.write(r.decision.explanation)
                for v in r.decision.vetoes:
                    st.error(v)
                for rep in r.reports:
                    st.markdown(f"**{rep.agent}** · {rep.signal.value} · score {rep.score:+.2f} · conf {rep.confidence}%")
                    for reason in rep.reasons:
                        st.markdown(f"- {reason}")
                st.markdown(f"**Risk manager:** max {r.risk.max_allocation_pct:.0f}%")
                start_hist = str((pd.Timestamp(r.as_of) - pd.Timedelta(days=550)).date())
                st.line_chart(provider.get_prices(r.ticker, start_hist, r.as_of))

with tab_backtest:
    c1, c2, c3 = st.columns(3)
    start = c1.date_input("Start", pd.Timestamp("2024-01-01"))
    end = c2.date_input("End", pd.Timestamp("2026-10-30"))
    bench = c3.selectbox("Benchmark", catalog.tickers(), index=catalog.tickers().index("XEQT.TO"))
    if st.button("Run backtest", disabled=not tickers):
        with st.spinner("Backtesting month by month..."):
            res = run_backtest(tickers, profile, catalog, provider, str(start), str(end), benchmark=bench)
        st.line_chart(res.equity)
        st.dataframe(res.stats, use_container_width=True)
        st.subheader("Target weights at each rebalance (%)")
        st.dataframe(res.allocations, use_container_width=True)
