"""Command-line interface.

    python -m maple_etf list
    python -m maple_etf analyze XEQT.TO VFV.TO VOO --account TFSA --risk medium --goal growth
    python -m maple_etf backtest XEQT.TO VFV.TO ZAG.TO XIU.TO --start 2024-01-01 --end 2026-10-30
    python -m maple_etf eval
"""

from __future__ import annotations

import argparse
import json
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

    if args.json:
        print(json.dumps([r.model_dump(mode="json") for r in results], indent=2))
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
    print("\nEducational project only. Not financial or tax advice.")
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
    a.add_argument("--trace", default="traces/runs.jsonl", help="JSONL trace file ('' to disable)")
    a.set_defaults(func=cmd_analyze)

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
