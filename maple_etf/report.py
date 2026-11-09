"""Plain-text output for the terminal (no extra libraries needed)."""

from __future__ import annotations

from maple_etf.models import AnalysisResult

ICON = {"bullish": "▲", "neutral": "●", "bearish": "▼", "BUY": "✅", "HOLD": "⏸ ", "AVOID": "⛔"}


def table(rows: list[list[str]], headers: list[str]) -> str:
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
    line = "  ".join("-" * w for w in widths)
    fmt = lambda r: "  ".join(str(x).ljust(w) for x, w in zip(r, widths))  # noqa: E731
    return "\n".join([fmt(headers), line, *(fmt(r) for r in rows)])


def summary(results: list[AnalysisResult]) -> str:
    rows = []
    for r in results:
        d = r.decision
        signals = " ".join(f"{x.agent[:4]}{ICON[x.signal.value]}" for x in r.reports)
        rows.append([r.ticker, f"{ICON[d.action.value]} {d.action.value}", f"{d.weighted_score:+.2f}",
                     f"{d.confidence}%", f"{d.suggested_allocation_pct:.0f}%", signals])
    return table(rows, ["ETF", "Decision", "Score", "Conf.", "Max alloc.", "Agent signals"])


def detail(r: AnalysisResult) -> str:
    out = [f"\n=== {r.ticker} — {r.decision.action.value} (as of {r.as_of}) ==="]
    for rep in r.reports:
        out.append(f"  {ICON[rep.signal.value]} {rep.agent:<16} score {rep.score:+.2f}  conf {rep.confidence}%")
        out.extend(f"      - {reason}" for reason in rep.reasons)
    out.append(f"  Risk manager: max {r.risk.max_allocation_pct:.0f}% of portfolio")
    out.extend(f"      - {reason}" for reason in r.risk.reasons)
    src = r.decision.explanation_source
    out.append(f"  Portfolio manager ({r.decision.decided_by}, explanation: {src}):")
    out.append(f"      {r.decision.explanation}")
    return "\n".join(out)
