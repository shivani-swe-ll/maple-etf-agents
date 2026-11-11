"""Read ETF facts (MER, holdings, yield, distribution frequency) from fact sheets.

Pipeline for each field:
  1. RETRIEVE  the best chunks for that field with BM25
  2. EXTRACT   with regex first (free, fast, exact); if regex finds nothing and an
               LLM is set, ask the LLM for the value AND an exact quote
  3. VERIFY    the quote must really exist in the chunk, and the value must be in
               the quote. If not, the value is rejected (no made-up facts).
Every accepted value keeps a citation: file name, page and quote.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from maple_etf.llm import extract_json
from maple_etf.rag.documents import BM25Index, Chunk

FIELDS = {
    "mer": "management expense ratio MER fees",
    "holdings_count": "number of holdings securities",
    "dividend_yield": "distribution yield dividend yield",
    "distribution_frequency": "distribution frequency paid monthly quarterly",
}

# Order matters: the first pattern that matches in the best chunk wins.
PATTERNS: dict[str, list[re.Pattern]] = {
    "mer": [
        re.compile(r"\b(?:management expense ratio|MER)\b(?!\s*\(%\))[^0-9\n%]{0,40}?(\d{1,2}\.\d{1,3})\s*%", re.I),
        re.compile(r"\bMER\s*\(%\)\s*[:\-]?\s*(\d{1,2}\.\d{1,3})", re.I),
    ],
    "holdings_count": [
        re.compile(r"\b(?:number of holdings|number of securities|total holdings|holdings)\b[^0-9\n]{0,25}?(\d{1,3}(?:,\d{3})+|\d+)", re.I),
    ],
    "dividend_yield": [
        re.compile(r"\b(?:distribution yield|dividend yield|trailing yield)\b[^0-9\n%]{0,45}?(\d{1,2}\.\d{1,2})\s*%", re.I),
    ],
    "distribution_frequency": [
        re.compile(r"\b(?:distribution frequency|distributions?(?: are)? paid|pays distributions)\b[^A-Za-z\n]{0,10}"
                   r"(?:[A-Za-z]+\s){0,2}?(monthly|quarterly|semi-annual(?:ly)?|annual(?:ly)?)", re.I),
        re.compile(r"\bdistributions?\b[^A-Za-z\n]{0,10}(monthly|quarterly|semi-annual(?:ly)?|annual(?:ly)?)\b", re.I),
    ],
}

SYSTEM_EXTRACT = (
    "You read ETF fact sheet excerpts. For the requested field, return JSON only: "
    '{"value": <number or word or null>, "chunk_id": "<id of the excerpt>", "quote": "<exact text copied from that excerpt>"}. '
    "The quote must be copied exactly and must contain the value. If the field is not in the excerpts, return value null."
)


@dataclass
class ExtractedFact:
    field: str
    value: float | int | str | None
    source: str = ""
    page: int = 0
    quote: str = ""
    method: str = "none"      # "regex" | "llm" | "none" | "llm_rejected"

    def to_dict(self) -> dict:
        return asdict(self)


def _normalize(field: str, raw: str) -> float | int | str:
    if field == "holdings_count":
        return int(raw.replace(",", ""))
    if field == "distribution_frequency":
        word = raw.lower()
        return "semi-annual" if word.startswith("semi") else "annual" if word.startswith("annual") else word
    return float(raw)


def _squash(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def regex_extract(field: str, chunk: Chunk) -> ExtractedFact | None:
    for pattern in PATTERNS[field]:
        m = pattern.search(chunk.text)
        if m:
            return ExtractedFact(field, _normalize(field, m.group(1)), chunk.source, chunk.page,
                                 m.group(0).strip(), "regex")
    return None


def llm_extract(field: str, hits: list[Chunk], llm) -> ExtractedFact:
    context = "\n\n".join(f"[{c.id}]\n{c.text}" for c in hits)
    try:
        resp = llm.complete(SYSTEM_EXTRACT, f"Field: {field} ({FIELDS[field]})\n\nExcerpts:\n{context}", max_tokens=200)
        data = extract_json(resp.text)
    except Exception:
        return ExtractedFact(field, None, method="llm_rejected")
    value, quote = data.get("value"), str(data.get("quote") or "")
    chunk = next((c for c in hits if c.id == data.get("chunk_id")), None)
    if value is None:
        return ExtractedFact(field, None, method="none")
    # VERIFY: the quote must exist in the cited chunk, and the value must appear in the quote.
    if chunk is None or not quote or _squash(quote) not in _squash(chunk.text):
        return ExtractedFact(field, None, quote=quote, method="llm_rejected")
    value_text = str(value).replace(",", "").lower()
    quote_text = quote.replace(",", "").lower()
    if field != "distribution_frequency":
        try:
            value = _normalize(field, value_text)
        except ValueError:
            return ExtractedFact(field, None, quote=quote, method="llm_rejected")
        if not any(abs(float(n) - float(value)) < 1e-9 for n in re.findall(r"\d+(?:\.\d+)?", quote_text)):
            return ExtractedFact(field, None, quote=quote, method="llm_rejected")
    elif value_text[:4] not in quote_text:
        return ExtractedFact(field, None, quote=quote, method="llm_rejected")
    else:
        value = _normalize(field, value_text)
    return ExtractedFact(field, value, chunk.source, chunk.page, quote, "llm")


def extract_facts(index: BM25Index, source: str | None = None, llm=None, k: int = 3) -> dict[str, ExtractedFact]:
    """Extract every field from one document (source=file name) or from all indexed documents."""
    out = {}
    for field, query in FIELDS.items():
        hits = [c for c, _ in index.search(query, k=k, source=source)]
        fact = None
        for c in hits:
            fact = regex_extract(field, c)
            if fact:
                break
        if fact is None:
            fact = llm_extract(field, hits, llm) if (llm is not None and hits) else ExtractedFact(field, None)
        out[field] = fact
    return out


def catalog_diff(facts: dict[str, ExtractedFact], current: dict) -> list[dict]:
    """Compare extracted facts with the catalog entry for the same ETF."""
    rows = []
    for field, fact in facts.items():
        if fact.value is None:
            continue
        old = current.get(field)
        changed = old != fact.value if not isinstance(old, float) else abs(old - float(fact.value)) > 1e-9
        rows.append({"field": field, "catalog": old, "document": fact.value, "changed": changed,
                     "citation": f"{fact.source} p.{fact.page}: \"{fact.quote}\""})
    return rows


def apply_to_catalog(catalog_path, ticker: str, facts: dict[str, ExtractedFact]) -> list[str]:
    """Write accepted values into catalog.json. Returns the fields that changed."""
    from pathlib import Path

    path = Path(catalog_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    entry = next((e for e in raw["etfs"] if e["ticker"].upper() == ticker.upper()), None)
    if entry is None:
        raise KeyError(f"{ticker} is not in {path.name}. Add a full entry first, then re-run.")
    changed = []
    for field, fact in facts.items():
        if fact.value is not None and entry.get(field) != fact.value:
            entry[field] = fact.value
            changed.append(field)
    path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    return changed
