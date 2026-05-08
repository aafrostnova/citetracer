from __future__ import annotations

import os
import threading
import time
from typing import Any

from packages.core.models import CitationRecord

from .base import BaseConnector, RequestPolicy


# Semantic Scholar's free tier (with or without API key) caps individual
# clients at roughly 1 request per second; bursts beyond that get 429ed
# regardless of how many threads we run. Without coordination across our
# 200+ concurrent paper / citation workers we melt down into synchronized
# exponential-backoff retry waves and the whole pool stalls. The class-level
# token-bucket below serializes all SemanticScholarConnector.search calls
# across the process so the API sees a steady ~1 req/s, not a burst of 320.
# Tunable via SEMANTIC_SCHOLAR_MIN_INTERVAL_S env var (default 1.05s).
_S2_LOCK = threading.Lock()
_S2_LAST_CALL_TS: list[float] = [0.0]  # mutable holder
_S2_MIN_INTERVAL_S = float(os.getenv("SEMANTIC_SCHOLAR_MIN_INTERVAL_S", "1.05"))


def _s2_throttle() -> None:
    """Block until at least _S2_MIN_INTERVAL_S has passed since the last
    Semantic Scholar request started by ANY thread in this process."""
    with _S2_LOCK:
        now = time.monotonic()
        wait = _S2_LAST_CALL_TS[0] + _S2_MIN_INTERVAL_S - now
        if wait > 0:
            time.sleep(wait)
        _S2_LAST_CALL_TS[0] = time.monotonic()


class SemanticScholarConnector(BaseConnector):
    name = "semantic_scholar"
    ttl_s = 60 * 60 * 24

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key

    def search(self, citation: CitationRecord, policy: RequestPolicy) -> list[dict[str, Any]]:
        query = citation.doi or citation.title or citation.raw_text
        if not query:
            return []
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        _s2_throttle()
        payload = self._request_json(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            {
                "query": query,
                "limit": 5,
                "fields": "title,authors,year,venue,externalIds,url,journal,publicationVenue",
            },
            policy,
            headers=headers,
        )
        records = []
        for paper in payload.get("data", []):
            external_ids = paper.get("externalIds", {}) or {}
            journal = paper.get("journal") or {}
            pub_venue = paper.get("publicationVenue") or {}
            volume = str(journal.get("volume", "") or "")
            pages = str(journal.get("pages", "") or "")
            publisher = str(pub_venue.get("publisher", "") or "")
            # S2 encodes arXiv preprint IDs in the volume field as "abs/<arxiv_id>",
            # "arXiv:<id>", or "corr/abs/<id>". These are NOT real journal volumes —
            # treat them as empty so downstream field comparison doesn't see a fake
            # non-empty value.
            _vol_lower = volume.strip().lower()
            if (_vol_lower.startswith("abs/")
                    or _vol_lower.startswith("arxiv:")
                    or _vol_lower.startswith("corr/")):
                volume = ""
            records.append(
                {
                    "title": str(paper.get("title", "") or ""),
                    "authors": [str(author.get("name", "") or "") for author in paper.get("authors", []) if author.get("name")],
                    "venue": str(paper.get("venue", "") or ""),
                    "year": paper.get("year"),
                    "doi": str(external_ids.get("DOI", "") or "").lower(),
                    "arxiv_id": str(external_ids.get("ArXiv", "") or "").lower(),
                    "url": str(paper.get("url", "") or ""),
                    "volume": volume,
                    "pages": pages,
                    "publisher": publisher,
                }
            )
        return records

    def fetch_paper_details(self, paper_id: str, policy: RequestPolicy) -> dict[str, Any]:
        """Fetch detailed paper info including publication types for preprint linking."""
        if not paper_id:
            return {}
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        _s2_throttle()
        try:
            payload = self._request_json(
                f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}",
                {"fields": "title,authors,year,venue,externalIds,publicationTypes,journal"},
                policy,
                headers=headers,
            )
        except Exception:
            return {}
        external_ids = payload.get("externalIds", {}) or {}
        return {
            "title": str(payload.get("title", "") or ""),
            "authors": [str(a.get("name", "") or "") for a in payload.get("authors", []) if a.get("name")],
            "year": payload.get("year"),
            "venue": str(payload.get("venue", "") or ""),
            "doi": str(external_ids.get("DOI", "") or "").lower(),
            "arxiv_id": str(external_ids.get("ArXiv", "") or "").lower(),
            "publication_types": payload.get("publicationTypes") or [],
            "journal": payload.get("journal") or {},
        }
