"""Portfolio manager: combines every analyst report into one decision.

Default ("rules" mode):
    confidence-weighted average of the analyst scores -> BUY / HOLD / AVOID,
    then hard guardrails (vetoes) are applied.

With an LLM:
    mode="explain" -> rules decide, the LLM writes a plain-English explanation.
    mode="decide"  -> the LLM decides (JSON, validated). Guardrails still apply after it.
In both LLM modes every number the LLM writes is checked against the facts
(see grounding.py). Ungrounded text is rejected and the template is used instead.
"""

from __future__ import annotations

import json
import logging

from maple_etf.grounding import ungrounded_numbers
from maple_etf.llm import LLMClient, extract_json
from maple_etf.models import Action, AgentReport, Decision, Goal, InvestorProfile, RiskLimits, Signal

log = logging.getLogger(__name__)

WEIGHTS: dict[Goal, dict[str, float]] = {
    Goal.GROWTH: {"cost": 0.20, "momentum": 0.15, "risk": 0.20, "diversification": 0.20, "income": 0.05, "tax_account": 0.20},
    Goal.INCOME: {"cost": 0.15, "momentum": 0.10, "risk": 0.20, "diversification": 0.10, "income": 0.30, "tax_account": 0.15},
    Goal.BALANCED: {"cost": 0.20, "momentum": 0.15, "risk": 0.20, "diversification": 0.20, "income": 0.10, "tax_account": 0.15},
}

BUY_AT = 0.25
AVOID_AT = -0.15
VETO_AT = -0.6

SYSTEM_EXPLAIN = (
    "You are a careful portfolio manager writing for a regular Canadian investor. "
    "Explain the decision in 3 to 5 short, simple sentences. "
    "Use ONLY facts and numbers from the analyst reports. Do not invent numbers. "
    "Do not promise returns. Plain text, no markdown."
)

SYSTEM_DECIDE = (
    "You are a careful portfolio manager for a regular Canadian investor. "
    "Read the analyst reports and decide BUY, HOLD or AVOID. "
    "Use ONLY facts and numbers from the reports. "
    'Reply with JSON only: {"action": "BUY|HOLD|AVOID", "confidence": 0-100, "explanation": "3-5 simple sentences"}'
)


class PortfolioManager:
    name = "portfolio_manager"

    def __init__(self, llm: LLMClient | None = None, mode: str = "explain"):
        if mode not in ("explain", "decide"):
            raise ValueError("mode must be 'explain' or 'decide'")
        self.llm = llm
        self.mode = mode
        self.last_trace: dict = {}

    # ---------- rules ----------
    @staticmethod
    def weighted_score(reports: list[AgentReport], goal: Goal) -> float:
        weights = WEIGHTS[goal]
        num = den = 0.0
        for r in reports:
            w = weights.get(r.agent, 0.0) * r.confidence / 100
            num += w * r.score
            den += w
        return round(num / den, 3) if den else 0.0

    @staticmethod
    def vetoes(reports: list[AgentReport]) -> list[str]:
        out = []
        by_name = {r.agent: r for r in reports}
        if "tax_account" in by_name and by_name["tax_account"].score <= VETO_AT:
            out.append("Tax/account veto: this ETF is a poor fit for this account type.")
        if "risk" in by_name and by_name["risk"].score <= VETO_AT:
            out.append("Risk veto: too volatile for this investor's risk tolerance.")
        return out

    @staticmethod
    def apply_guardrails(action: Action, vetoes: list[str]) -> Action:
        if len(vetoes) >= 2:
            return Action.AVOID
        if vetoes and action == Action.BUY:
            return Action.HOLD
        return action

    @staticmethod
    def agreement(reports: list[AgentReport], action: Action) -> float:
        target = {Action.BUY: Signal.BULLISH, Action.AVOID: Signal.BEARISH, Action.HOLD: Signal.NEUTRAL}[action]
        return sum(r.signal == target for r in reports) / len(reports) if reports else 0.0

    def decide(self, ticker: str, reports: list[AgentReport], risk: RiskLimits, profile: InvestorProfile) -> Decision:
        score = self.weighted_score(reports, profile.goal)
        action = Action.BUY if score >= BUY_AT else Action.AVOID if score <= AVOID_AT else Action.HOLD
        vetoes = self.vetoes(reports)
        action = self.apply_guardrails(action, vetoes)
        rules_action = action
        confidence = rules_confidence = int(40 + 60 * self.agreement(reports, action))
        decided_by = "rules"
        explanation = self._template(ticker, action, score, reports, vetoes)
        source = "template"
        self.last_trace = {}

        if self.llm is not None:
            context = self._context(ticker, reports, risk, profile, score, action, vetoes)
            try:
                if self.mode == "decide":
                    resp = self.llm.complete(SYSTEM_DECIDE, context)
                    data = extract_json(resp.text)
                    llm_action = Action(str(data["action"]).upper())
                    llm_action = self.apply_guardrails(llm_action, vetoes)  # LLM cannot override hard rules
                    text = str(data.get("explanation", "")).strip()
                    action, decided_by = llm_action, "llm"
                    confidence = max(0, min(100, int(data.get("confidence", confidence))))
                else:
                    resp = self.llm.complete(SYSTEM_EXPLAIN, context)
                    text = resp.text

                bad = ungrounded_numbers(text, context)
                self.last_trace = {
                    "llm_latency_ms": resp.latency_ms,
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                    "ungrounded_numbers": bad,
                }
                if text and not bad:
                    explanation, source = text, "llm"
                else:
                    # Don't trust a decision whose explanation invents facts: fall back to rules.
                    source = "llm_rejected"
                    action, confidence, decided_by = rules_action, rules_confidence, "rules"
                    log.warning("LLM explanation for %s rejected; ungrounded numbers: %s", ticker, bad)
            except Exception as exc:  # network error, bad JSON, invalid action...
                log.warning("LLM step failed for %s, using rules: %s", ticker, exc)
                self.last_trace = {"llm_error": str(exc)}
                action, confidence, decided_by = rules_action, rules_confidence, "rules"

            if source != "llm":
                explanation = self._template(ticker, action, score, reports, vetoes)

        allocation = 0.0
        if action == Action.BUY:
            allocation = round(risk.max_allocation_pct * min(1.0, 0.5 + score), 1)

        return Decision(
            ticker=ticker,
            action=action,
            weighted_score=score,
            confidence=confidence,
            suggested_allocation_pct=allocation,
            vetoes=vetoes,
            explanation=explanation,
            explanation_source=source,
            decided_by=decided_by,
        )

    # ---------- text ----------
    @staticmethod
    def _template(ticker: str, action: Action, score: float, reports: list[AgentReport], vetoes: list[str]) -> str:
        best = max(reports, key=lambda r: r.score)
        worst = min(reports, key=lambda r: r.score)
        parts = [f"{action.value} {ticker} (weighted score {score:+.2f})."]
        good = f"({best.agent}): {best.reasons[0]}" if best.score > 0 else ""
        bad = f"({worst.agent}): {worst.reasons[0]}" if worst.score < 0 else ""
        # Lead with what matters for the decision. Judge labels showed that an AVOID starting with
        # "Strongest point" confused readers.
        if action == Action.AVOID:
            if bad:
                parts.append(f"Main reason {bad}")
            if good:
                parts.append(f"On the plus side {good}")
        else:
            if good:
                parts.append(f"Strongest point {good}")
            if bad:
                parts.append(f"Biggest concern {bad}")
        parts.extend(vetoes)
        return " ".join(parts)

    @staticmethod
    def _context(ticker, reports, risk, profile, score, action, vetoes) -> str:
        payload = {
            "ticker": ticker,
            "investor": profile.model_dump(mode="json"),
            "rules_engine": {"weighted_score": score, "suggested_action": action.value, "vetoes": vetoes},
            "risk_manager": risk.model_dump(),
            "analysts": [
                {"agent": r.agent, "signal": r.signal.value, "score": r.score, "confidence": r.confidence, "reasons": r.reasons}
                for r in reports
            ],
        }
        return json.dumps(payload, indent=2)
