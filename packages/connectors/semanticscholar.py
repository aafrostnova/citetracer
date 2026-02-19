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
                "fields": "title,authors,year,venue,externalIds,url",
            },
            policy,
            headers=headers,
        )
        records = []
        for paper in payload.get("data", []):
            external_ids = paper.get("externalIds", {}) or {}
            records.append(
                {
                    "title": str(paper.get("title", "") or ""),
                    "authors": [str(author.get("name", "") or "") for author in paper.get("authors", []) if author.get("name")],
                    "venue": str(paper.get("venue", "") or ""),
                    "year": paper.get("year"),
                    "doi": str(external_ids.get("DOI", "") or "").lower(),
                    "arxiv_id": str(external_ids.get("ArXiv", "") or "").lower(),
                    "url": str(paper.get("url", "") or ""),
                }
            )
        return records
