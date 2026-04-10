from __future__ import annotations

from typing import Any

from packages.core.models import CitationRecord

from .base import BaseConnector, RequestPolicy


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
