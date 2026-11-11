"""Ask questions about the fact sheets and get answers WITH citations.

Without an LLM: returns the best matching passages (extractive answer).
With an LLM:    the LLM answers using only the retrieved passages, cites them as [1], [2],
                and every number is checked against those passages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from maple_etf.grounding import ungrounded_numbers
from maple_etf.rag.documents import BM25Index

SYSTEM_QA = (
    "Answer the question using ONLY the numbered excerpts from ETF documents. "
    "Cite excerpts like [1] or [2] after each fact. If the answer is not in the excerpts, say you could not find it. "
    "Use short, simple sentences. Never invent numbers."
)


@dataclass
class Answer:
    text: str
    citations: list[dict] = field(default_factory=list)
    source: str = "extractive"     # "extractive" | "llm" | "llm_rejected"


def ask(index: BM25Index, question: str, llm=None, k: int = 3) -> Answer:
    hits = index.search(question, k=k)
    if not hits:
        return Answer("I could not find anything about that in the documents.")
    citations = [{"n": i + 1, "source": c.source, "page": c.page, "text": c.text} for i, (c, _) in enumerate(hits)]
    context = "\n\n".join(f"[{c['n']}] ({c['source']}, page {c['page']})\n{c['text']}" for c in citations)

    if llm is not None:
        try:
            resp = llm.complete(SYSTEM_QA, f"Question: {question}\n\nExcerpts:\n{context}", max_tokens=400)
            if not ungrounded_numbers(resp.text, context):
                return Answer(resp.text, citations, "llm")
            fallback_note = "(The AI answer used numbers not found in the documents, so here are the passages.) "
            source = "llm_rejected"
        except Exception:
            fallback_note, source = "", "extractive"
    else:
        fallback_note, source = "", "extractive"

    best = citations[0]
    return Answer(f"{fallback_note}{_best_lines(best['text'], question)} [1]", citations, source)


def _best_lines(text: str, question: str, n: int = 2) -> str:
    """Pick the sentences that share the most words with the question (simple extractive answer)."""
    from maple_etf.rag.documents import expand_query, tokenize

    stop = {"the", "a", "an", "of", "is", "what", "doe", "do", "how", "much", "for", "to", "in", "it", "fund", "etf"}
    q = {t for t in expand_query(question) if t not in stop}
    parts = [s.strip(" #") for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip(" #")) > 3]
    parts = [s for s in parts if not s.upper().startswith("SAMPLE DOCUMENT")]
    ranked = sorted(parts, key=lambda s: len(q & set(tokenize(s))), reverse=True)
    best = [s for s in ranked[:n] if q & set(tokenize(s))] or parts[:1]
    return " ".join(best)
