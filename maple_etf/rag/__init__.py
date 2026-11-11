"""RAG on ETF fact sheets: load -> chunk -> BM25 retrieve -> extract/answer -> verify with citations."""

from maple_etf.rag.documents import BM25Index, Chunk, Page, chunk_pages, load_document, load_folder
from maple_etf.rag.extract import ExtractedFact, apply_to_catalog, catalog_diff, extract_facts
from maple_etf.rag.qa import Answer, ask


def build_index(path) -> BM25Index:
    """Index one file or every PDF/TXT/MD file in a folder."""
    from pathlib import Path

    p = Path(path)
    pages = load_folder(p) if p.is_dir() else load_document(p)
    return BM25Index(chunk_pages(pages))


__all__ = [
    "BM25Index", "Chunk", "Page", "chunk_pages", "load_document", "load_folder", "build_index",
    "ExtractedFact", "extract_facts", "catalog_diff", "apply_to_catalog", "Answer", "ask",
]
