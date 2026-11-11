"""RAG eval: on sample fact sheets with known answers, measure
* extraction accuracy  (did we read the right MER / holdings / yield / frequency?)
* retrieval hit rate   (is the right document the top search result for a question?)
"""

from __future__ import annotations

import json
from pathlib import Path

from maple_etf.rag import build_index, extract_facts

DEFAULT_DIR = Path("evals/rag_samples")


def run_rag_evals(folder: str | Path = DEFAULT_DIR, llm=None, verbose: bool = True) -> dict:
    folder = Path(folder)
    expected = json.loads((folder / "expected.json").read_text(encoding="utf-8"))
    index = build_index(folder)

    field_hits, misses = [], []
    for doc, fields in expected["documents"].items():
        facts = extract_facts(index, source=doc, llm=llm)
        for field, want in fields.items():
            got = facts[field].value
            ok = (abs(float(got) - float(want)) < 1e-9) if isinstance(want, (int, float)) and got is not None else got == want
            field_hits.append(ok)
            if not ok:
                misses.append({"doc": doc, "field": field, "expected": want, "got": got, "method": facts[field].method})

    retrieval_hits = []
    for q in expected["questions"]:
        top = index.search(q["question"], k=1)
        hit = bool(top) and top[0][0].source == q["expected_source"]
        retrieval_hits.append(hit)
        if not hit:
            misses.append({"question": q["question"], "expected_source": q["expected_source"],
                           "got": top[0][0].source if top else None, "type": "retrieval_miss"})

    summary = {
        "documents": len(expected["documents"]),
        "extraction_accuracy": round(sum(field_hits) / len(field_hits), 3),
        "extraction_checks": len(field_hits),
        "retrieval_hit_rate": round(sum(retrieval_hits) / len(retrieval_hits), 3),
        "retrieval_checks": len(retrieval_hits),
        "llm": "on" if llm else "off (regex only)",
        "misses": misses,
    }
    if verbose:
        print(f"RAG eval ({summary['llm']}): extraction {summary['extraction_accuracy']:.0%} "
              f"({sum(field_hits)}/{len(field_hits)}), retrieval {summary['retrieval_hit_rate']:.0%} "
              f"({sum(retrieval_hits)}/{len(retrieval_hits)})")
        for m in misses:
            print("  miss:", json.dumps(m))
    return summary
