"""Chat with the agents (tool calling).

The LLM does NOT answer from its own memory. It gets tools that run the real
agents (analyze, compare, portfolio, ETF facts), decides which ones to call,
reads their JSON results, and then answers. Every number in the final answer is
checked against the tool results (grounding.py).

Without an LLM, a simple rules-based router handles common questions, so the
chat also works on the free live website with no API key.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from maple_etf.compare import compare_etfs
from maple_etf.data import Catalog, PriceProvider
from maple_etf.graph import AgentGraph, load_etf
from maple_etf.grounding import ungrounded_numbers
from maple_etf.models import InvestorProfile
from maple_etf.portfolio import analyze_portfolio

log = logging.getLogger(__name__)
MAX_TOOL_ROUNDS = 5

SYSTEM_CHAT = (
    "You are the assistant of Maple ETF Agents, a team of analyst agents for Canadian ETFs. "
    "Always use the tools to get facts; never answer from memory. "
    "Only these ETFs are supported: {tickers}. If asked about another one, say it is not in the catalog. "
    "The investor: {profile}. "
    "Answer in short, simple sentences for someone whose first language may not be English. "
    "Use only numbers that appear in tool results. Never promise returns. "
    "This is educational, not financial or tax advice."
)

TOOLS = [
    {"name": "list_etfs", "description": "List every ETF the system knows, with name and MER.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "get_etf_facts", "description": "Static facts about one ETF: MER, holdings, yield, regions, structure.",
     "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}},
    {"name": "analyze_etf",
     "description": "Run all analyst agents on one ETF for this investor. Returns BUY/HOLD/AVOID with every agent's reasons.",
     "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}},
    {"name": "compare_etfs", "description": "Compare two ETFs for this investor and say which fits better and why.",
     "parameters": {"type": "object", "properties": {"ticker_a": {"type": "string"}, "ticker_b": {"type": "string"}},
                    "required": ["ticker_a", "ticker_b"]}},
    {"name": "check_portfolio",
     "description": "Check a whole portfolio. holdings maps ticker to percent, e.g. {\"XEQT.TO\": 60, \"ZAG.TO\": 40}.",
     "parameters": {"type": "object",
                    "properties": {"holdings": {"type": "object", "additionalProperties": {"type": "number"}}},
                    "required": ["holdings"]}},
]


@dataclass
class ChatReply:
    text: str
    source: str                     # "llm" | "rules" | "llm_rejected" | "llm_error"
    tool_calls: list[dict] = field(default_factory=list)
    trace: dict = field(default_factory=dict)


class AgentTools:
    """Python functions the LLM can call. Each returns a JSON string."""

    def __init__(self, catalog: Catalog, provider: PriceProvider, profile: InvestorProfile):
        self.catalog, self.provider, self.profile = catalog, provider, profile
        self.graph = AgentGraph()

    def resolve(self, ticker: str) -> str:
        t = str(ticker).strip().upper()
        if t in self.catalog:
            return t
        if f"{t}.TO" in self.catalog:
            return f"{t}.TO"
        raise KeyError(f"{ticker} is not in the catalog")

    def run(self, name: str, args: dict) -> str:
        try:
            fn = getattr(self, f"tool_{name}")
        except AttributeError:
            return json.dumps({"error": f"unknown tool {name}"})
        try:
            return json.dumps(fn(**args), default=str)
        except Exception as exc:  # bad ticker, bad arguments...
            return json.dumps({"error": str(exc)})

    def tool_list_etfs(self) -> list[dict]:
        return [{"ticker": t, "name": self.catalog.get(t).name, "mer_pct": self.catalog.get(t).mer}
                for t in self.catalog.tickers()]

    def tool_get_etf_facts(self, ticker: str) -> dict:
        return self.catalog.get(self.resolve(ticker)).model_dump(mode="json")

    def tool_analyze_etf(self, ticker: str) -> dict:
        r = self.graph.run(load_etf(self.resolve(ticker), self.catalog, self.provider), self.profile)
        return {"ticker": r.ticker, "decision": r.decision.action.value, "weighted_score": r.decision.weighted_score,
                "vetoes": r.decision.vetoes, "max_allocation_pct": r.risk.max_allocation_pct,
                "agents": {x.agent: {"signal": x.signal.value, "score": x.score, "reasons": x.reasons} for x in r.reports},
                "synthetic_prices": r.synthetic_data}

    def tool_compare_etfs(self, ticker_a: str, ticker_b: str) -> dict:
        a = load_etf(self.resolve(ticker_a), self.catalog, self.provider)
        b = load_etf(self.resolve(ticker_b), self.catalog, self.provider)
        res = compare_etfs(a, b, self.profile)
        return {"winner": res.winner, res.ticker_a: {"decision": res.action_a.value, "score": res.score_a},
                res.ticker_b: {"decision": res.action_b.value, "score": res.score_b},
                "key_differences": [d.model_dump() for d in res.key_differences], "summary": res.explanation}

    def tool_check_portfolio(self, holdings: dict) -> dict:
        h = {self.resolve(t): float(w) for t, w in holdings.items()}
        etfs = {t: load_etf(t, self.catalog, self.provider) for t in h}
        rep = analyze_portfolio(etfs, h, self.profile)
        return rep.model_dump(mode="json", exclude={"results"})


class ChatSession:
    def __init__(self, catalog: Catalog, provider: PriceProvider, profile: InvestorProfile, llm=None):
        self.tools = AgentTools(catalog, provider, profile)
        self.catalog, self.profile, self.llm = catalog, profile, llm
        self.messages: list[dict] = []   # provider-native history (LLM mode only)

    def ask(self, question: str) -> ChatReply:
        if self.llm is None:
            return RuleBasedChat(self.tools).answer(question)
        return self._ask_llm(question)

    def _ask_llm(self, question: str) -> ChatReply:
        system = SYSTEM_CHAT.format(tickers=", ".join(self.catalog.tickers()),
                                    profile=self.profile.model_dump_json())
        history = [*self.messages, self.llm.user_message(question)]
        calls_made, evidence = [], [question, system]
        try:
            for _ in range(MAX_TOOL_ROUNDS):
                step = self.llm.chat_step(system, history, TOOLS)
                history.append(step.assistant_message)
                if not step.tool_calls:
                    break
                results = []
                for call in step.tool_calls:
                    output = self.tools.run(call.name, call.arguments)
                    calls_made.append({"name": call.name, "arguments": call.arguments})
                    evidence.append(output)
                    results.append((call, output))
                history.extend(self.llm.tool_result_messages(results))
            else:
                step.text = step.text or "Sorry, I needed too many steps for that question. Please ask it more simply."
        except Exception as exc:
            log.warning("chat LLM failed: %s", exc)
            reply = RuleBasedChat(self.tools).answer(question)
            reply.source, reply.trace = "llm_error", {"error": str(exc)}
            return reply

        bad = ungrounded_numbers(step.text, "\n".join(evidence))
        if bad:
            reply = RuleBasedChat(self.tools).answer(question)
            reply.text = "(The AI answer used numbers I could not verify, so here is the rules-based answer.) " + reply.text
            reply.source, reply.trace = "llm_rejected", {"ungrounded_numbers": bad, "tool_calls": calls_made}
            return reply

        self.messages = history  # keep memory of the conversation only for trusted turns
        return ChatReply(step.text, "llm", calls_made, {"tool_rounds": len(calls_made)})


class RuleBasedChat:
    """No-LLM fallback: find tickers in the question and pick a tool by simple rules."""

    HOLDING_RE = re.compile(r"\b([A-Za-z]{2,5}(?:\.TO)?)\s*(?:=|:)?\s*(\d+(?:\.\d+)?)\s*%")

    def __init__(self, tools: AgentTools):
        self.tools = tools

    def find_tickers(self, text: str) -> list[str]:
        found = []
        for word in re.findall(r"[A-Za-z]{2,5}(?:\.TO)?", text):
            try:
                t = self.tools.resolve(word)
            except KeyError:
                continue
            if t not in found:
                found.append(t)
        return found

    def answer(self, question: str) -> ChatReply:
        q = question.lower()
        holdings = {}
        for sym, pct in self.HOLDING_RE.findall(question):
            try:
                holdings[self.tools.resolve(sym)] = float(pct)
            except KeyError:
                continue
        if len(holdings) >= 2:
            data = json.loads(self.tools.run("check_portfolio", {"holdings": holdings}))
            if "error" in data:
                return ChatReply(f"Sorry: {data['error']}", "rules")
            return ChatReply(data["summary"], "rules", [{"name": "check_portfolio", "arguments": {"holdings": holdings}}])

        tickers = self.find_tickers(question)
        if len(tickers) >= 2:
            a, b = tickers[:2]
            data = json.loads(self.tools.run("compare_etfs", {"ticker_a": a, "ticker_b": b}))
            return ChatReply(data.get("summary", data.get("error", "")), "rules",
                             [{"name": "compare_etfs", "arguments": {"ticker_a": a, "ticker_b": b}}])
        if len(tickers) == 1:
            t = tickers[0]
            data = json.loads(self.tools.run("analyze_etf", {"ticker": t}))
            if "error" in data:
                return ChatReply(f"Sorry: {data['error']}", "rules")
            lines = [f"For your profile, the agents say {data['decision']} for {t} (score {data['weighted_score']:+.2f})."]
            lines.extend(data["vetoes"])
            ranked = sorted(data["agents"].items(), key=lambda kv: abs(kv[1]["score"]), reverse=True)
            for name, info in ranked[:3]:
                lines.append(f"{name} ({info['signal']}): {info['reasons'][0]}")
            return ChatReply(" ".join(lines), "rules", [{"name": "analyze_etf", "arguments": {"ticker": t}}])
        if any(w in q for w in ("list", "which etf", "what etf", "available")):
            names = ", ".join(e["ticker"] for e in self.tools.tool_list_etfs())
            return ChatReply(f"I know these ETFs: {names}.", "rules", [{"name": "list_etfs", "arguments": {}}])
        return ChatReply(
            "I can help with: one ETF (\"Why HOLD for VOO?\"), two ETFs (\"XEQT or VEQT?\"), "
            "or a portfolio (\"XEQT 60% ZAG 40%\"). Ask \"list ETFs\" to see what I know.",
            "rules",
        )
