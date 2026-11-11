import json
import shutil

import pytest

from maple_etf.rag import BM25Index, Chunk, apply_to_catalog, ask, build_index, catalog_diff, extract_facts
from maple_etf.rag.evals import run_rag_evals

SAMPLES = "evals/rag_samples"


@pytest.fixture(scope="module")
def index():
    return build_index(SAMPLES)


def test_pdf_txt_and_md_are_indexed(index):
    sources = {c.source for c in index.chunks}
    assert sources == {"msge-global-equity.pdf", "mscb-canadian-bond.txt", "msdi-dividend-income.md"}
    assert {c.page for c in index.chunks if c.source.endswith(".pdf")} == {1, 2}


def test_bm25_ranks_the_right_document(index):
    top = index.search("covered call options", k=1)[0][0]
    assert top.source == "msdi-dividend-income.md"


def test_regex_reads_facts_with_citations(index):
    facts = extract_facts(index, source="msge-global-equity.pdf")
    assert facts["mer"].value == 0.20            # not the 0.18% management fee
    assert facts["holdings_count"].value == 9412
    assert facts["distribution_frequency"].value == "quarterly"
    assert facts["dividend_yield"].page == 2
    assert "0.20%" in facts["mer"].quote


def test_regex_misses_table_yield_without_llm(index):
    assert extract_facts(index, source="msdi-dividend-income.md")["dividend_yield"].value is None


def test_llm_fills_the_gap_with_a_verified_quote(index, fake_llm):
    llm = fake_llm(json.dumps({"value": 5.10, "chunk_id": "msdi-dividend-income.md#p1-0", "quote": "Yield (%) 5.10"}))
    fact = extract_facts(index, source="msdi-dividend-income.md", llm=llm)["dividend_yield"]
    assert fact.value == 5.10 and fact.method == "llm"


def test_llm_made_up_quote_is_rejected(index, fake_llm):
    llm = fake_llm(json.dumps({"value": 6.2, "chunk_id": "msdi-dividend-income.md#p1-0", "quote": "Yield (%) 6.2"}))
    fact = extract_facts(index, source="msdi-dividend-income.md", llm=llm)["dividend_yield"]
    assert fact.value is None and fact.method == "llm_rejected"


def test_llm_value_not_in_quote_is_rejected(index, fake_llm):
    llm = fake_llm(json.dumps({"value": 9.9, "chunk_id": "msdi-dividend-income.md#p1-0", "quote": "Yield (%) 5.10"}))
    assert extract_facts(index, source="msdi-dividend-income.md", llm=llm)["dividend_yield"].method == "llm_rejected"


def test_catalog_diff_and_apply(tmp_path, index):
    cat = tmp_path / "catalog.json"
    shutil.copy("maple_etf/data/catalog.json", cat)
    facts = extract_facts(index, source="mscb-canadian-bond.txt")
    current = next(e for e in json.loads(cat.read_text())["etfs"] if e["ticker"] == "ZAG.TO")
    diff = {r["field"]: r for r in catalog_diff(facts, current)}
    assert diff["mer"]["changed"] is False and diff["holdings_count"]["changed"] is True
    changed = apply_to_catalog(cat, "ZAG.TO", facts)
    assert "holdings_count" in changed
    entry = next(e for e in json.loads(cat.read_text())["etfs"] if e["ticker"] == "ZAG.TO")
    assert entry["holdings_count"] == 1512


def test_ask_extractive_and_llm(index, fake_llm):
    ans = ask(index, "What is the MER of the bond ETF?")
    assert ans.source == "extractive" and ans.citations[0]["source"] == "mscb-canadian-bond.txt"
    good = ask(index, "MER of the bond ETF?", llm=fake_llm("The MER is 0.09% [1]."))
    bad = ask(index, "MER of the bond ETF?", llm=fake_llm("The MER is 0.07% [1]."))
    assert good.source == "llm" and bad.source == "llm_rejected"


def test_chunking_overlaps_long_text():
    from maple_etf.rag import Page, chunk_pages

    text = "\n".join(f"line {i} " + "x" * 50 for i in range(40))
    chunks = chunk_pages([Page("t.txt", 1, text)], size=300, overlap=80)
    assert len(chunks) > 5
    assert all(len(c.text) <= 300 for c in chunks)


def test_rag_eval_numbers():
    s = run_rag_evals(verbose=False)
    assert s["extraction_accuracy"] >= 0.9 and s["retrieval_hit_rate"] == 1.0


def test_empty_index_search():
    assert BM25Index([Chunk("a", "a.txt", 1, "hello world")]).search("bond yield") == []
