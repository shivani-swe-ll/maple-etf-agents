"""LLM-as-judge for explanations, calibrated against human labels.

Rubric (each is pass/fail):
  correct - agrees with the decision and the facts; no wrong numbers or wrong claims
  clear   - a reader can tell what the decision is and the main reason for it
  simple  - short sentences and plain words (no finance jargon)

Two judges:
  HeuristicJudge - rules only, free, runs everywhere (a baseline)
  LLMJudge       - asks an LLM to grade with the rubric (needs .env)

A judge is only useful if it agrees with people. `calibrate()` compares a judge
with human labels (evals/judge_labels.json) and reports agreement and Cohen's kappa
(agreement corrected for luck: 1.0 = perfect, 0 = no better than chance).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from maple_etf.grounding import ungrounded_numbers
from maple_etf.llm import extract_json

CRITERIA = ("correct", "clear", "simple")
LABELS_PATH = Path("evals/judge_labels.json")

JARGON = ["alpha", "beta", "sharpe", "standard deviation", "sortino", "basis points", "bps", "idiosyncratic",
          "convexity", "duration risk", "heteroskedastic", "drawdown", "tracking error", "factor exposure",
          "mean reversion", "volatility-adjusted", "risk-adjusted", "covariance"]

SYSTEM_JUDGE = (
    "You grade a short explanation written for a regular Canadian investor whose first language may not be English. "
    "You get the true decision, the facts, and the explanation. Grade each criterion true or false:\n"
    "- correct: agrees with the true decision and the facts. Any wrong number, wrong claim or wrong decision = false.\n"
    "- clear: a reader can tell what the decision is and the main reason for it.\n"
    "- simple: short sentences and everyday words. Finance jargon (e.g. Sharpe, drawdown, basis points) = false.\n"
    'Reply with JSON only: {"correct": true|false, "clear": true|false, "simple": true|false, "why": "one short sentence"}'
)


@dataclass
class Item:
    id: str
    ticker: str
    decision: str
    facts: str
    explanation: str
    human: dict[str, bool] = field(default_factory=dict)
    label_source: str = ""


@dataclass
class Grade:
    correct: bool
    clear: bool
    simple: bool
    why: str = ""

    def get(self, c: str) -> bool:
        return getattr(self, c)


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


class HeuristicJudge:
    name = "heuristic"

    def grade(self, item: Item) -> Grade:
        text = item.explanation
        words = text.split()
        actions = {a for a in ("BUY", "HOLD", "AVOID") if re.search(rf"\b{a}\b", text, re.I)}
        wrong_numbers = ungrounded_numbers(text, item.facts)
        correct = not wrong_numbers and (not actions or actions == {item.decision})
        clear = 12 <= len(words) <= 120 and (item.decision in actions or item.ticker.split(".")[0] in text)
        sents = _sentences(text)
        avg_len = (len(words) / len(sents)) if sents else len(words)
        jargon = sum(1 for j in JARGON if j in text.lower())
        simple = avg_len <= 24 and jargon == 0
        return Grade(correct, clear, simple, f"wrong numbers={wrong_numbers}, actions={sorted(actions)}, "
                                             f"avg sentence={avg_len:.0f} words, jargon={jargon}")


class LLMJudge:
    name = "llm"

    def __init__(self, llm):
        self.llm = llm

    def grade(self, item: Item) -> Grade:
        user = (f"True decision for {item.ticker}: {item.decision}\n\nFacts:\n{item.facts}\n\n"
                f"Explanation to grade:\n{item.explanation}")
        data = extract_json(self.llm.complete(SYSTEM_JUDGE, user, max_tokens=200).text)
        return Grade(bool(data["correct"]), bool(data["clear"]), bool(data["simple"]), str(data.get("why", "")))


def load_items(path: str | Path = LABELS_PATH) -> list[Item]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Item(**x) for x in raw["items"]]


def cohen_kappa(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else round((po - pe) / (1 - pe), 3)


def calibrate(judge, items: list[Item]) -> dict:
    labeled = [it for it in items if all(c in it.human for c in CRITERIA)]
    grades = []
    for it in labeled:
        try:
            grades.append(judge.grade(it))
        except Exception as exc:  # LLM down or bad JSON: count as a failed grade
            grades.append(None)
            print(f"  judge failed on {it.id}: {exc}")
    out = {"judge": judge.name, "items": len(labeled), "criteria": {}, "disagreements": []}
    for c in CRITERIA:
        pairs = [(it.human[c], g.get(c)) for it, g in zip(labeled, grades) if g is not None]
        human = [h for h, _ in pairs]
        machine = [m for _, m in pairs]
        agree = sum(h == m for h, m in pairs) / len(pairs) if pairs else None
        out["criteria"][c] = {"agreement": round(agree, 3) if agree is not None else None,
                              "kappa": cohen_kappa(human, machine)}
    for it, g in zip(labeled, grades):
        if g is None:
            continue
        diff = [c for c in CRITERIA if it.human[c] != g.get(c)]
        if diff:
            out["disagreements"].append({"id": it.id, "criteria": diff, "judge_said": {c: g.get(c) for c in diff},
                                         "why": g.why})
    return out


def print_calibration(result: dict) -> None:
    print(f"\nJudge: {result['judge']}  ({result['items']} labeled explanations)")
    print("| Criterion | Agreement with human | Cohen's kappa |")
    print("|---|---|---|")
    for c, v in result["criteria"].items():
        agree = "n/a" if v["agreement"] is None else f"{v['agreement']:.0%}"
        print(f"| {c} | {agree} | {v['kappa']} |")
    for d in result["disagreements"]:
        print(f"  disagree on {d['id']}: {d['criteria']} (judge said {d['judge_said']})")


def label_interactively(items: list[Item], path: str | Path = LABELS_PATH, who: str = "you") -> int:
    """Ask a person to grade unlabeled items in the terminal. Returns how many were labeled."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    done = 0
    for entry in raw["items"]:
        if all(c in entry.get("human", {}) for c in CRITERIA):
            continue
        print(f"\n--- {entry['id']} | {entry['ticker']} | true decision: {entry['decision']} ---")
        print(f"Facts: {entry['facts']}\n\nExplanation: {entry['explanation']}\n")
        human = {}
        for c in CRITERIA:
            ans = ""
            while ans not in ("y", "n", "q"):
                ans = input(f"Is it {c}? [y/n, q=quit] ").strip().lower()
            if ans == "q":
                Path(path).write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
                return done
            human[c] = ans == "y"
        entry["human"], entry["label_source"] = human, who
        done += 1
    Path(path).write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    return done
