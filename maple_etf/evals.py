"""Evaluation harness.

Measures three things on a golden set of scenarios (evals/golden_cases.json):

1. Agent accuracy     - does each analyst give the signal a human expert expects?
2. Decision accuracy  - is the final BUY/HOLD/AVOID in the allowed set?
3. Groundedness       - does every number in the explanation come from the facts?

Every failure is tagged with a type (failure taxonomy) so you can see WHAT breaks,
not just how often. Results are saved to evals/results/latest.json and latest.md.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from maple_etf.agents import PortfolioManager
from maple_etf.compare import compare_etfs
from maple_etf.data import load_catalog, make_price_path
from maple_etf.graph import AgentGraph
from maple_etf.grounding import ungrounded_numbers
from maple_etf.llm import LLMClient
from maple_etf.models import ETFData, ETFProfile, InvestorProfile


def build_case(case: dict, catalog) -> tuple[ETFData, InvestorProfile]:
    base = catalog.get(case["base"]).model_dump()
    base.update(case.get("overrides", {}))
    price_args = {"days": 504, **case["prices"]}
    prices = make_price_path(**price_args)
    return ETFData(profile=ETFProfile(**base), prices=prices, synthetic=True), InvestorProfile(**case["profile"])


def build_etf(ticker: str, price_args: dict, catalog, overrides: dict | None = None) -> ETFData:
    base = catalog.get(ticker).model_dump()
    base.update(overrides or {})
    prices = make_price_path(**{"days": 504, **price_args})
    return ETFData(profile=ETFProfile(**base), prices=prices, synthetic=True)


def run_comparison_cases(cases: list[dict], catalog, llm=None) -> tuple[list[bool], list[dict]]:
    hits, failures = [], []
    for case in cases:
        a = build_etf(case["a"], case.get("prices_a", case.get("prices", {})), catalog)
        b = build_etf(case["b"], case.get("prices_b", case.get("prices", {})), catalog)
        res = compare_etfs(a, b, InvestorProfile(**case["profile"]), llm=llm)
        ok = res.winner in case["expected_winner"]
        hits.append(ok)
        if not ok:
            failures.append({"case": case["id"], "type": "comparison_mismatch", "expected": case["expected_winner"],
                             "got": res.winner, "scores": [res.score_a, res.score_b]})
    return hits, failures


def run_evals(
    cases_path: str | Path = "evals/golden_cases.json",
    out_dir: str | Path | None = "evals/results",
    llm: LLMClient | None = None,
    llm_mode: str = "explain",
    verbose: bool = True,
) -> dict:
    catalog = load_catalog()
    golden = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    cases = golden["cases"]
    graph = AgentGraph(llm=llm, llm_mode=llm_mode)

    agent_hits: dict[str, list[bool]] = defaultdict(list)
    decision_hits: list[bool] = []
    grounded: list[bool] = []
    failures: list[dict] = []
    rows = []

    for case in cases:
        etf, profile = build_case(case, catalog)
        result = graph.run(etf, profile)
        exp = case["expected"]
        case_ok = True

        for agent, want in exp.get("agents", {}).items():
            got = result.report(agent).signal.value
            ok = got == want
            agent_hits[agent].append(ok)
            if not ok:
                case_ok = False
                failures.append({"case": case["id"], "type": "agent_signal_mismatch", "agent": agent,
                                 "expected": want, "got": got, "reasons": result.report(agent).reasons})

        if "decision" in exp:
            got = result.decision.action.value
            ok = got in exp["decision"]
            decision_hits.append(ok)
            if not ok:
                case_ok = False
                failures.append({"case": case["id"], "type": "decision_mismatch", "expected": exp["decision"],
                                 "got": got, "score": result.decision.weighted_score})

        context = PortfolioManager._context(etf.profile.ticker, result.reports, result.risk, profile,
                                            result.decision.weighted_score, result.decision.action,
                                            result.decision.vetoes)
        bad = ungrounded_numbers(result.decision.explanation, context)
        grounded.append(not bad)
        if bad:
            case_ok = False
            failures.append({"case": case["id"], "type": "ungrounded_explanation", "numbers": bad})
        if result.decision.explanation_source == "llm_rejected":
            failures.append({"case": case["id"], "type": "llm_rejected_fallback_used",
                             "detail": graph.pm.last_trace})

        rows.append({"case": case["id"], "pass": case_ok, "decision": result.decision.action.value,
                     "score": result.decision.weighted_score, "explanation_source": result.decision.explanation_source})

    comparison_hits, comparison_failures = run_comparison_cases(golden.get("comparisons", []), catalog, llm)
    failures.extend(comparison_failures)

    rate = lambda xs: round(sum(xs) / len(xs), 3) if xs else None  # noqa: E731
    all_agent = [x for xs in agent_hits.values() for x in xs]
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "llm": f"{llm.provider}/{llm.model} ({llm_mode})" if llm else "off (rules + template explanations)",
        "cases": len(cases),
        "cases_passed": sum(r["pass"] for r in rows),
        "agent_accuracy": rate(all_agent),
        "agent_accuracy_by_agent": {a: rate(h) for a, h in sorted(agent_hits.items())},
        "decision_accuracy": rate(decision_hits),
        "groundedness": rate(grounded),
        "comparison_accuracy": rate(comparison_hits),
        "comparison_cases": len(comparison_hits),
        "failure_types": dict(Counter(f["type"] for f in failures)),
        "llm_usage": llm.usage if llm else None,
        "rows": rows,
        "failures": failures,
    }

    md = to_markdown(summary)
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "latest.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        (out / "latest.md").write_text(md, encoding="utf-8")
    if verbose:
        print(md)
    return summary


def to_markdown(s: dict) -> str:
    lines = [
        "# Eval results",
        "",
        f"- Run: {s['timestamp']}",
        f"- LLM: {s['llm']}",
        f"- Cases passed: **{s['cases_passed']} / {s['cases']}**",
        "",
        "| Metric | Score |",
        "|---|---|",
        f"| Agent signal accuracy | {s['agent_accuracy']:.0%} |",
        f"| Decision accuracy | {s['decision_accuracy']:.0%} |",
        f"| Explanation groundedness | {s['groundedness']:.0%} |",
        *([f"| Comparison winner accuracy ({s['comparison_cases']} pairs) | {s['comparison_accuracy']:.0%} |"]
          if s.get("comparison_cases") else []),
        "",
        "| Agent | Accuracy |",
        "|---|---|",
        *[f"| {a} | {v:.0%} |" for a, v in s["agent_accuracy_by_agent"].items()],
        "",
        "## Failures",
        "",
    ]
    if not s["failures"]:
        lines.append("None. 🎉")
    for f in s["failures"]:
        detail = {k: v for k, v in f.items() if k not in ("case", "type")}
        lines.append(f"- **{f['case']}** · `{f['type']}` · {json.dumps(detail, default=str)}")
    return "\n".join(lines) + "\n"
