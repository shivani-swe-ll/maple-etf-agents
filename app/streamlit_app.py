"""Web UI.  Run:  pip install -r requirements.txt  &&  streamlit run app/streamlit_app.py"""

import sys
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from maple_etf.backtest import run_backtest  # noqa: E402
from maple_etf.chat import ChatSession  # noqa: E402
from maple_etf.compare import compare_etfs  # noqa: E402
from maple_etf.data import SampleDataProvider, YFinanceProvider, load_catalog  # noqa: E402
from maple_etf.graph import AgentGraph, analyze_tickers, load_etf  # noqa: E402
from maple_etf.llm import LLMClient  # noqa: E402
from maple_etf.models import AccountType, Goal, InvestorProfile, RiskTolerance  # noqa: E402
from maple_etf.portfolio import analyze_portfolio  # noqa: E402
from maple_etf.rag import ask, build_index, extract_facts  # noqa: E402
from maple_etf.report import results_to_csv, results_to_json, results_to_rows  # noqa: E402

load_dotenv()
st.set_page_config(page_title="Maple ETF Agents", page_icon="🍁", layout="wide")
st.title("🍁 Maple ETF Agents")
st.caption("Multi-agent AI analyst for Canadian ETFs. Educational project, not financial or tax advice.")

catalog = load_catalog()
TICKERS = catalog.tickers()
today = date.today().isoformat()

with st.sidebar:
    st.header("Investor")
    account = st.selectbox("Account", [a.value for a in AccountType])
    risk = st.selectbox("Risk tolerance", [r.value for r in RiskTolerance], index=1)
    goal = st.selectbox("Goal", [g.value for g in Goal])
    st.header("Data")
    live = st.toggle("Live Yahoo Finance prices", value=False)
    llm = LLMClient.from_env()
    use_llm = st.toggle("Use LLM", value=llm is not None, disabled=llm is None,
                        help="Set LLM_PROVIDER and LLM_MODEL in .env (or Streamlit secrets) to enable")
    st.caption("LLM: " + (f"{llm.provider} / {llm.model}" if llm else "off (rules only)"))

profile = InvestorProfile(account=AccountType(account), risk=RiskTolerance(risk), goal=Goal(goal))
active_llm = llm if use_llm else None


@st.cache_resource
def sample_provider():
    return SampleDataProvider({t: catalog.get(t).category for t in TICKERS})


provider = YFinanceProvider() if live else sample_provider()
if provider.synthetic:
    st.warning("Using synthetic sample prices. Turn on live prices for real market data.")

# Streamlit re-runs this script on every click, so results live in session_state.
for key in ("results", "backtest", "comparison", "portfolio", "chat_log", "chat_key", "rag_dir"):
    st.session_state.setdefault(key, None)

tab_analyze, tab_compare, tab_portfolio, tab_chat, tab_docs, tab_backtest = st.tabs(
    ["Analyze", "Compare", "Portfolio", "Chat", "Fact sheets (RAG)", "Backtest"]
)

# ---------------- Analyze ----------------
with tab_analyze:
    tickers = st.multiselect("ETFs", TICKERS, default=["XEQT.TO", "VFV.TO", "ZAG.TO", "VOO"])
    if st.button("Run agents", type="primary", disabled=not tickers):
        with st.spinner("Agents are working..."):
            st.session_state.results = analyze_tickers(tickers, profile, catalog, provider, AgentGraph(llm=active_llm))

    results = st.session_state.results
    if results:
        summary = pd.DataFrame(results_to_rows(results))
        cols = ["ticker", "decision", "weighted_score", "confidence_pct", "suggested_allocation_pct",
                *[c for c in summary.columns if c.endswith("_signal")]]
        st.dataframe(summary[cols], hide_index=True, use_container_width=True)
        d1, d2, _ = st.columns([1, 1, 3])
        d1.download_button("⬇️ Download CSV", results_to_csv(results), file_name=f"maple-etf-analysis-{today}.csv",
                           mime="text/csv")
        d2.download_button("⬇️ Download JSON", results_to_json(results), file_name=f"maple-etf-analysis-{today}.json",
                           mime="application/json")
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

# ---------------- Compare ----------------
with tab_compare:
    c1, c2 = st.columns(2)
    a = c1.selectbox("ETF A", TICKERS, index=TICKERS.index("XEQT.TO"))
    b = c2.selectbox("ETF B", TICKERS, index=TICKERS.index("VEQT.TO"))
    if st.button("Compare", type="primary", disabled=a == b):
        with st.spinner("Comparing..."):
            st.session_state.comparison = compare_etfs(load_etf(a, catalog, provider), load_etf(b, catalog, provider),
                                                       profile, llm=active_llm)
    res = st.session_state.comparison
    if res is not None:
        if res.winner == "tie":
            st.info(f"🤝 {res.ticker_a} and {res.ticker_b} are very close for you.")
        else:
            st.success(f"🏆 {res.winner} fits you better.")
        st.write(res.explanation)
        st.dataframe(pd.DataFrame([{"agent": d.agent, res.ticker_a: d.score_a, res.ticker_b: d.score_b,
                                    "better": d.edge} for d in res.diffs]), hide_index=True, use_container_width=True)

# ---------------- Portfolio ----------------
with tab_portfolio:
    st.write("Enter what you own (percent of your money in each ETF).")
    held = st.multiselect("Your ETFs", TICKERS, default=["XEQT.TO", "ZAG.TO"], key="pf_tickers")
    weights = {}
    cols = st.columns(max(1, min(4, len(held))))
    for i, t in enumerate(held):
        weights[t] = cols[i % len(cols)].number_input(f"{t} %", min_value=0.0, max_value=100.0,
                                                      value=round(100 / len(held), 1), step=5.0, key=f"w_{t}")
    if st.button("Check my portfolio", type="primary", disabled=not held or sum(weights.values()) <= 0):
        with st.spinner("Checking the whole portfolio..."):
            etfs = {t: load_etf(t, catalog, provider) for t in weights}
            st.session_state.portfolio = analyze_portfolio(etfs, weights, profile, llm=active_llm)
    rep = st.session_state.portfolio
    if rep is not None:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Health", f"{rep.health_score}/100", rep.grade)
        m2.metric("Average MER", f"{rep.weighted_mer_pct:.2f}%", f"${rep.annual_fee_per_10k:.0f}/yr per $10k",
                  delta_color="off")
        m3.metric("Stocks / bonds", f"{rep.equity_pct:.0f}% / {rep.fixed_income_pct:.0f}%")
        m4.metric("Yield", f"{rep.weighted_yield_pct:.1f}%")
        st.write(rep.summary)
        for f in rep.findings:
            {"good": st.success, "note": st.info, "warning": st.warning}[f.level](f.message)
        st.bar_chart(pd.Series(rep.region_mix, name="% of portfolio"))

# ---------------- Chat ----------------
with tab_chat:
    st.write("Ask things like **Why HOLD for VOO?**, **XEQT or VEQT?**, or **XEQT 60% ZAG 40%**.")
    chat_key = (profile.model_dump_json(), live, use_llm)
    if st.session_state.chat_key != chat_key or st.session_state.chat_log is None:
        st.session_state.chat_key = chat_key
        st.session_state.chat_log = []
        st.session_state.chat_session = ChatSession(catalog, provider, profile, llm=active_llm)
    for role, text in st.session_state.chat_log:
        st.chat_message(role).write(text)
    question = st.chat_input("Ask the agents...")
    if question:
        st.chat_message("user").write(question)
        with st.spinner("Thinking..."):
            reply = st.session_state.chat_session.ask(question)
        tools = ", ".join(c["name"] for c in reply.tool_calls) or "none"
        answer = f"{reply.text}\n\n_source: {reply.source} · tools used: {tools}_"
        st.chat_message("assistant").write(answer)
        st.session_state.chat_log += [("user", question), ("assistant", answer)]

# ---------------- Fact sheets (RAG) ----------------
with tab_docs:
    st.write("Upload ETF fact sheets (PDF, TXT or MD). The system reads MER, holdings, yield and frequency, "
             "with the page and exact quote for every value.")
    files = st.file_uploader("Fact sheets", type=["pdf", "txt", "md"], accept_multiple_files=True)
    u1, u2 = st.columns(2)
    if u1.button("Read uploaded files", disabled=not files):
        folder = Path(tempfile.mkdtemp(prefix="maple-docs-"))
        for f in files:
            (folder / Path(f.name).name).write_bytes(f.getvalue())
        st.session_state.rag_dir = str(folder)
    if u2.button("Try the sample fact sheets"):
        st.session_state.rag_dir = str(ROOT / "evals" / "rag_samples")

    if st.session_state.rag_dir:
        index = build_index(st.session_state.rag_dir)
        for src in sorted({c.source for c in index.chunks}):
            facts = extract_facts(index, source=src, llm=active_llm)
            st.subheader(src)
            st.dataframe(pd.DataFrame([{"field": f.field, "value": f.value, "method": f.method,
                                        "page": f.page or None, "quote": f.quote} for f in facts.values()]),
                         hide_index=True, use_container_width=True)
        q = st.text_input("Ask a question about these documents", placeholder="What is the MER of the bond ETF?")
        if q:
            ans = ask(index, q, llm=active_llm)
            st.write(ans.text)
            with st.expander("Sources"):
                for c in ans.citations:
                    st.markdown(f"**[{c['n']}] {c['source']}, page {c['page']}**")
                    st.text(c["text"])

# ---------------- Backtest ----------------
with tab_backtest:
    bt_tickers = st.multiselect("ETFs to choose from", TICKERS, default=["XEQT.TO", "VFV.TO", "ZAG.TO", "XIU.TO"],
                                key="bt_tickers")
    c1, c2, c3 = st.columns(3)
    start = c1.date_input("Start", pd.Timestamp("2024-01-01"))
    end = c2.date_input("End", pd.Timestamp("2026-10-30"))
    bench = c3.selectbox("Benchmark", TICKERS, index=TICKERS.index("XEQT.TO"))
    if st.button("Run backtest", disabled=not bt_tickers):
        with st.spinner("Backtesting month by month..."):
            st.session_state.backtest = run_backtest(bt_tickers, profile, catalog, provider, str(start), str(end),
                                                     benchmark=bench)
    res = st.session_state.backtest
    if res is not None:
        st.line_chart(res.equity)
        st.dataframe(res.stats, use_container_width=True)
        st.subheader("Target weights at each rebalance (%)")
        st.dataframe(res.allocations, use_container_width=True)
        b1, b2, b3, _ = st.columns([1, 1, 1, 2])
        b1.download_button("⬇️ Equity curve (CSV)", res.equity.to_csv(),
                           file_name=f"maple-etf-backtest-equity-{today}.csv", mime="text/csv")
        b2.download_button("⬇️ Summary stats (CSV)", res.stats.to_csv(),
                           file_name=f"maple-etf-backtest-stats-{today}.csv", mime="text/csv")
        b3.download_button("⬇️ Allocations (CSV)", res.allocations.to_csv(),
                           file_name=f"maple-etf-backtest-allocations-{today}.csv", mime="text/csv")
