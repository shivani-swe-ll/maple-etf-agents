"""Load documents (PDF, TXT, MD), split them into chunks, and search them with BM25.

BM25 is a classic keyword-search ranking (it is what search engines used before
embeddings). It needs no model, no GPU and no API key, and for short, number-heavy
documents like ETF fact sheets it works very well.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

SUPPORTED = {".pdf", ".txt", ".md"}


@dataclass
class Page:
    source: str
    page: int
    text: str


@dataclass
class Chunk:
    id: str
    source: str
    page: int
    text: str


def load_document(path: str | Path) -> list[Page]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"Unsupported file type {suffix}. Use PDF, TXT or MD.")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Reading PDFs needs pypdf. Run: pip install pypdf") from exc
        reader = PdfReader(str(path))
        return [Page(path.name, i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]
    return [Page(path.name, 1, path.read_text(encoding="utf-8", errors="replace"))]


def load_folder(folder: str | Path) -> list[Page]:
    pages: list[Page] = []
    for f in sorted(Path(folder).iterdir()):
        if f.suffix.lower() in SUPPORTED:
            pages.extend(load_document(f))
    return pages


def chunk_pages(pages: list[Page], size: int = 600, overlap: int = 150) -> list[Chunk]:
    """Split each page into overlapping windows of about `size` characters, on line breaks when possible."""
    chunks = []
    for p in pages:
        text = re.sub(r"[ \t]+", " ", p.text).strip()
        if not text:
            continue
        start, n = 0, 0
        while start < len(text):
            end = min(len(text), start + size)
            if end < len(text):
                cut = text.rfind("\n", start + size // 2, end)
                end = cut if cut > start else end
            chunks.append(Chunk(f"{p.source}#p{p.page}-{n}", p.source, p.page, text[start:end].strip()))
            n += 1
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks


TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?%?")

# Words people use in questions -> words that appear in fact sheets.
SYNONYMS = {
    "mer": ["management", "expense", "ratio"],
    "fee": ["mer", "management", "expense"],
    "cost": ["mer", "management", "expense"],
    "often": ["frequency"],
    "pay": ["distribution", "paid"],
    "dividend": ["distribution", "yield"],
    "income": ["distribution", "yield"],
    "stock": ["holding"],
    "company": ["holding"],
}


def _stem(token: str) -> str:
    """Very small stemmer: 'distributions' -> 'distribution', 'holdings' -> 'holding'."""
    if len(token) > 4 and token.isalpha():
        if token.endswith("ies"):
            return token[:-3] + "y"
        if token.endswith("s") and not token.endswith("ss"):
            return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in TOKEN_RE.findall(text.lower())]


def expand_query(text: str) -> list[str]:
    terms = tokenize(text)
    extra = [x for t in terms for x in SYNONYMS.get(t, [])]
    return terms + [_stem(x) for x in extra]


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(c.text)) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0
        df = Counter(term for d in self.docs for term in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 4, source: str | None = None) -> list[tuple[Chunk, float]]:
        terms = expand_query(query)
        scored = []
        for i, (doc, length) in enumerate(zip(self.docs, self.lengths)):
            if source and self.chunks[i].source != source:
                continue
            score = 0.0
            for t in terms:
                f = doc.get(t, 0)
                if f:
                    denom = f + self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1))
                    score += self.idf.get(t, 0) * f * (self.k1 + 1) / denom
            if score > 0:
                scored.append((self.chunks[i], score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
