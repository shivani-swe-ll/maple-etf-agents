"""Command-line interface.

    python -m maple_etf list
    python -m maple_etf analyze XEQT.TO VFV.TO VOO --account TFSA --risk medium --goal growth
    python -m maple_etf backtest XEQT.TO VFV.TO ZAG.TO XIU.TO --start 2024-01-01 --end 2026-10-30
    python -m maple_etf eval
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from maple_etf.data import SampleDataProvider, YFinanceProvider, load_catalog
from maple_etf.graph import AgentGraph, analyze_tickers
from maple_etf.llm import LLMClient
from maple_etf.models import AccountType, Goal, InvestorProfile, RiskTolerance
from maple_etf import report

SYNTHETIC_WARNING = "⚠  Using SYNTHETIC sample prices (offline demo). Add --live for real Yahoo Finance data.\n"


def _provider(args, catalog):
    if args.live:
        return YFinanceProvider()
    return SampleDataProvider({t: catalog.get(t).category for t in catalog.tickers()})


def _profile(args) -> InvestorProfile:
    return InvestorProfile(
        account=AccountType(args.account.upper()), risk=RiskTolerance(args.risk), goal=Goal(args.goal)
    )


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--account", default="TFSA", choices=[a.value for a in AccountType], type=str.upper)
    p.add_argument("--risk", default="medium", choices=[r.value for r in RiskTolerance])
    p.add_argument("--goal", default="growth", choices=[g.value for g in Goal])
    p.add_argument("--live", action="store_true", help="real prices from Yahoo Finance (pip install yfinance)")


def cmd_list(args) -> int:
    catalog = load_catalog()
    rows = [[t, (e := catalog.get(t)).name, e.category, f"{e.mer:.2f}%", e.structure.value] for t in catalog.tickers()]
    print(report.table(rows, ["Ticker", "Name", "Category", "MER", "Structure"]))
    return 0


def cmd_analyze(args) -> int:
    catalog = load_catalog()
    provider = _provider(args, catalog)
    llm = None if args.llm == "off" else LLMClient.from_env()
    if args.llm != "off" and llm is None and args.llm != "auto":
        print("LLM not configured: set LLM_PROVIDER and LLM_MODEL in .env (see .env.example).", file=sys.stderr)
        return 2
    mode = "decide" if args.llm == "decide" else "explain"
    graph = AgentGraph(llm=llm, llm_mode=mode, trace_path=args.trace)
    results = analyze_tickers(args.tickers, _profile(args), catalog, provider, graph, as_of=args.as_of)

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        Path(args.csv).write_text(report.results_to_csv(results), encoding="utf-8")
    if args.json:
        print(report.results_to_json(results))
        return 0
    if provider.synthetic:
        print(SYNTHETIC_WARNING)
    p = results[0].profile
    print(f"Investor: {p.account.value} account, {p.risk.value} risk, {p.goal.value} goal")
    print(f"LLM: {'off (rules only)' if llm is None else f'{llm.provider}/{llm.model} ({mode})'}\n")
    print(report.summary(results))
    if not args.brief:
        for r in results:
            print(report.detail(r))
    if llm is not None:
        print(f"\nLLM usage: {llm.usage}")
    if args.trace:
        print(f"\nTrace written to {args.trace}")
    if args.csv:
        print(f"Results saved to {args.csv}")
    print("\nEducational project only. Not financial or tax advice.")
    return 0


def _llm_or_none(args):
    return None if getattr(args, "llm", "auto") == "off" else LLMClient.from_env()


def cmd_compare(args) -> int:
    from maple_etf.compare import compare_etfs
    from maple_etf.graph import load_etf

    catalog = load_catalog()
    provider = _provider(args, catalog)
    if provider.synthetic:
        print(SYNTHETIC_WARNING)
    a, b = (load_etf(t, catalog, provider, args.as_of) for t in (args.a, args.b))
    res = compare_etfs(a, b, _profile(args), llm=_llm_or_none(args))
    rows = [[d.agent, f"{d.score_a:+.2f}", f"{d.score_b:+.2f}", d.edge] for d in res.diffs]
    rows.append(["WEIGHTED", f"{res.score_a:+.2f}", f"{res.score_b:+.2f}", res.winner])
    print(report.table(rows, ["Agent", res.ticker_a, res.ticker_b, "Better"]))
    print(f"\nWinner: {res.winner}   ({res.ticker_a}: {res.action_a.value}, {res.ticker_b}: {res.action_b.value})")
    print(f"\n{res.explanation}")
    return 0


def cmd_portfolio(args) -> int:
    from maple_etf.graph import load_etf
    from maple_etf.portfolio import analyze_portfolio, parse_holdings

    catalog = load_catalog()
    provider = _provider(args, catalog)
    if provider.synthetic:
        print(SYNTHETIC_WARNING)
    holdings = parse_holdings(args.holdings)
    etfs = {t: load_etf(t, catalog, provider, args.as_of) for t in holdings}
    rep = analyze_portfolio(etfs, holdings, _profile(args), llm=_llm_or_none(args))
    rows = [[r.ticker, f"{rep.holdings[r.ticker]:.0f}%", r.decision.action.value, f"{catalog.get(r.ticker).mer:.2f}%"]
            for r in rep.results]
    print(report.table(rows, ["ETF", "Weight", "Agents say", "MER"]))
    print(f"\nAverage MER {rep.weighted_mer_pct:.2f}% (about ${rep.annual_fee_per_10k:.0f}/year per $10,000) | "
          f"stocks {rep.equity_pct:.0f}% / bonds {rep.fixed_income_pct:.0f}% | yield {rep.weighted_yield_pct:.1f}%")
    print("Regions: " + ", ".join(f"{k} {v:.0f}%" for k, v in rep.region_mix.items() if v > 0))
    if rep.volatility_1y is not None:
        print(f"1-year volatility {rep.volatility_1y * 100:.1f}% | worst fall {rep.max_drawdown_1y * 100:.1f}%")
    print(f"\nHealth score: {rep.health_score}/100 (grade {rep.grade})")
    icon = {"good": "✅", "note": "ℹ️ ", "warning": "⚠️ "}
    for f in rep.findings:
        print(f"  {icon[f.level]} {f.message}")
    print(f"\n{rep.summary}")
    return 0


def cmd_chat(args) -> int:
    from maple_etf.chat import ChatSession

    catalog = load_catalog()
    provider = _provider(args, catalog)
    llm = _llm_or_none(args)
    session = ChatSession(catalog, provider, _profile(args), llm=llm)
    mode = "rules only (no LLM set in .env)" if llm is None else f"{llm.provider}/{llm.model} with tools"
    print(f"Chat with the agents [{mode}]. Type 'quit' to stop.")
    if provider.synthetic:
        print(SYNTHETIC_WARNING)
    questions = [args.question] if args.question else None
    while True:
        q = questions.pop(0) if questions else (None if questions is not None else input("\nYou: ").strip())
        if q is None or q.lower() in ("quit", "exit", "q"):
            return 0
        if not q:
            continue
        reply = session.ask(q)
        tools = ", ".join(c["name"] for c in reply.tool_calls) or "none"
        print(f"\nAgents: {reply.text}\n   [source: {reply.source} | tools used: {tools}]")


def cmd_rag(args) -> int:
    from maple_etf.data.catalog import DEFAULT_CATALOG
    from maple_etf.rag import apply_to_catalog, ask, build_index, catalog_diff, extract_facts

    llm = _llm_or_none(args)
    if args.rag_cmd == "eval":
        from maple_etf.rag.evals import run_rag_evals

        run_rag_evals(args.path or "evals/rag_samples", llm=llm)
        return 0

    index = build_index(args.path)
    if args.rag_cmd == "ask":
        ans = ask(index, args.question, llm=llm)
        print(ans.text)
        print("\nSources:")
        for c in ans.citations:
            print(f"  [{c['n']}] {c['source']} page {c['page']}")
        return 0

    # extract
    sources = sorted({c.source for c in index.chunks})
    for src in sources:
        facts = extract_facts(index, source=src, llm=llm)
        print(f"\n=== {src} ===")
        rows = [[f.field, "-" if f.value is None else f.value, f.method, f"p.{f.page}" if f.page else "", f.quote[:60]]
                for f in facts.values()]
        print(report.table(rows, ["Field", "Value", "Method", "Page", "Quote"]))
        if args.ticker:
            catalog = load_catalog()
            current = catalog.get(args.ticker).model_dump()
            diff = catalog_diff(facts, current)
            changed = [r for r in diff if r["changed"]]
            print(f"\nCompared with catalog entry {args.ticker.upper()}: {len(changed)} difference(s)")
            for r in changed:
                print(f"  {r['field']}: catalog {r['catalog']} -> document {r['document']}   ({r['citation']})")
            if args.apply and changed:
                fields = apply_to_catalog(DEFAULT_CATALOG, args.ticker, facts)
                print(f"Updated catalog.json: {', '.join(fields)}")
    return 0


def cmd_judge(args) -> int:
    from maple_etf.judge import HeuristicJudge, LLMJudge, calibrate, label_interactively, load_items, print_calibration

    if args.judge_cmd == "label":
        n = label_interactively(load_items(args.labels), args.labels, who=args.who)
        print(f"Saved {n} new label(s) to {args.labels}")
        return 0
    items = load_items(args.labels)
    print_calibration(calibrate(HeuristicJudge(), items))
    llm = LLMClient.from_env()
    if llm is None:
        print("\nLLM judge skipped: set LLM_PROVIDER and LLM_MODEL in .env (Ollama works for free).")
    else:
        print_calibration(calibrate(LLMJudge(llm), items))
    return 0


def cmd_backtest(args) -> int:
    from maple_etf.backtest import run_backtest

    catalog = load_catalog()
    provider = _provider(args, catalog)
    if provider.synthetic:
        print(SYNTHETIC_WARNING)
    res = run_backtest(args.tickers, _profile(args), catalog, provider, args.start, args.end,
                       benchmark=args.benchmark, initial_cash=args.cash)
    print(report.table([[i, *map(str, row)] for i, row in res.stats.iterrows()],
                       ["Strategy", *res.stats.columns]))
    print("\nLast 3 rebalances (target weights %):")
    print(res.allocations.tail(3).to_string())
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        res.equity.to_csv(args.out)
        print(f"\nDaily equity curve saved to {args.out}")
    return 0


def cmd_eval(args) -> int:
    from maple_etf.evals import run_evals

    llm = LLMClient.from_env() if args.with_llm else None
    if args.with_llm and llm is None:
        print("--with-llm needs LLM_PROVIDER and LLM_MODEL in .env", file=sys.stderr)
        return 2
    summary = run_evals(cases_path=args.cases, out_dir=args.out_dir, llm=llm, llm_mode=args.llm_mode)
    return 0 if summary["decision_accuracy"] >= args.min_accuracy else 1


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(prog="maple_etf", description="Multi-agent AI analyst for Canadian ETFs")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="show ETFs in the catalog").set_defaults(func=cmd_list)

    a = sub.add_parser("analyze", help="run all agents on one or more ETFs")
    a.add_argument("tickers", nargs="+")
    _add_common(a)
    a.add_argument("--llm", default="auto", choices=["auto", "off", "explain", "decide"],
                   help="auto = use LLM for explanations if .env is set")
    a.add_argument("--as-of", help="analysis date, YYYY-MM-DD (default: today)")
    a.add_argument("--json", action="store_true", help="print raw JSON")
    a.add_argument("--brief", action="store_true", help="summary table only")
    a.add_argument("--csv", help="also save the results to a CSV file, e.g. results/analysis.csv")
    a.add_argument("--trace", default="traces/runs.jsonl", help="JSONL trace file ('' to disable)")
    a.set_defaults(func=cmd_analyze)

    c = sub.add_parser("compare", help="which of two ETFs fits you better")
    c.add_argument("a")
    c.add_argument("b")
    _add_common(c)
    c.add_argument("--llm", default="auto", choices=["auto", "off"])
    c.add_argument("--as-of")
    c.set_defaults(func=cmd_compare)

    pf = sub.add_parser("portfolio", help="check a whole portfolio, e.g. XEQT.TO=60 ZAG.TO=40")
    pf.add_argument("holdings", nargs="+", help="TICKER=PERCENT")
    _add_common(pf)
    pf.add_argument("--llm", default="auto", choices=["auto", "off"])
    pf.add_argument("--as-of")
    pf.set_defaults(func=cmd_portfolio)

    ch = sub.add_parser("chat", help="ask the agents questions in plain English")
    _add_common(ch)
    ch.add_argument("--llm", default="auto", choices=["auto", "off"])
    ch.add_argument("-q", "--question", help="ask one question and exit")
    ch.set_defaults(func=cmd_chat)

    rg = sub.add_parser("rag", help="read ETF fact sheets (PDF/TXT/MD): extract facts or ask questions")
    rg_sub = rg.add_subparsers(dest="rag_cmd", required=True)
    rx = rg_sub.add_parser("extract", help="read MER, holdings, yield and frequency from documents")
    rx.add_argument("path", help="a file or a folder of fact sheets")
    rx.add_argument("--ticker", help="compare with this catalog entry, e.g. XEQT.TO (use with one file)")
    rx.add_argument("--apply", action="store_true", help="write the new values into catalog.json")
    rq = rg_sub.add_parser("ask", help="ask a question; the answer cites its sources")
    rq.add_argument("path")
    rq.add_argument("question")
    re_ = rg_sub.add_parser("eval", help="extraction + retrieval accuracy on the sample fact sheets")
    re_.add_argument("path", nargs="?")
    for sp in (rx, rq, re_):
        sp.add_argument("--llm", default="auto", choices=["auto", "off"])
    rg.set_defaults(func=cmd_rag)

    jg = sub.add_parser("judge", help="LLM-as-judge: grade explanations and check agreement with human labels")
    jg_sub = jg.add_subparsers(dest="judge_cmd", required=True)
    je = jg_sub.add_parser("eval", help="compare the judges with the human labels")
    jl = jg_sub.add_parser("label", help="add your own grades in the terminal")
    jl.add_argument("--who", default="you")
    for sp in (je, jl):
        sp.add_argument("--labels", default="evals/judge_labels.json")
    jg.set_defaults(func=cmd_judge)

    b = sub.add_parser("backtest", help="monthly backtest vs a benchmark")
    b.add_argument("tickers", nargs="+")
    _add_common(b)
    b.add_argument("--start", default="2024-01-01")
    b.add_argument("--end", default="2026-10-30")
    b.add_argument("--benchmark", default="XEQT.TO")
    b.add_argument("--cash", type=float, default=10_000)
    b.add_argument("--out", help="save daily equity curve to CSV")
    b.set_defaults(func=cmd_backtest)

    e = sub.add_parser("eval", help="run the golden-set evaluation")
    e.add_argument("--cases", default="evals/golden_cases.json")
    e.add_argument("--out-dir", default="evals/results")
    e.add_argument("--with-llm", action="store_true", help="also evaluate LLM explanations")
    e.add_argument("--llm-mode", default="explain", choices=["explain", "decide"])
    e.add_argument("--min-accuracy", type=float, default=0.0, help="exit code 1 if decision accuracy is lower")
    e.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    if getattr(args, "trace", None) == "":
        args.trace = None
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
