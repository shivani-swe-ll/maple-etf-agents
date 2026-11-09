"""Load ETF facts (MER, holdings, structure...) from catalog.json."""

from __future__ import annotations

import json
from pathlib import Path

from maple_etf.models import ETFProfile

DEFAULT_CATALOG = Path(__file__).with_name("catalog.json")


class Catalog:
    def __init__(self, etfs: list[ETFProfile]):
        self._etfs = {e.ticker.upper(): e for e in etfs}

    def get(self, ticker: str) -> ETFProfile:
        key = ticker.upper()
        if key not in self._etfs:
            known = ", ".join(sorted(self._etfs))
            raise KeyError(
                f"'{ticker}' is not in the ETF catalog. Add it to maple_etf/data/catalog.json. "
                f"Known tickers: {known}"
            )
        return self._etfs[key]

    def tickers(self) -> list[str]:
        return sorted(self._etfs)

    def __contains__(self, ticker: str) -> bool:
        return ticker.upper() in self._etfs


def load_catalog(path: str | Path | None = None) -> Catalog:
    raw = json.loads(Path(path or DEFAULT_CATALOG).read_text(encoding="utf-8"))
    return Catalog([ETFProfile(**item) for item in raw["etfs"]])
