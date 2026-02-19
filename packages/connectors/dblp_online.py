from __future__ import annotations

from typing import Any

from packages.core.models import CitationRecord

from .base import BaseConnector, RequestPolicy


class DBLPOnlineConnector(BaseConnector):
    name = "dblp_online"
    ttl_s = 60 * 60 * 24 * 3

    def search(self, citation: CitationRecord, policy: RequestPolicy) -> list[dict[str, Any]]:
        query = citation.title or citation.raw_text
        if not query:
            return []
        payload = self._request_json(
            "https://dblp.org/search/publ/api",
            {"q": query, "h": 5, "format": "json"},
            policy,
        )
        hits = payload.get("result", {}).get("hits", {}).get("hit", [])
        if isinstance(hits, dict):
            hits = [hits]
        records = []
        for hit in hits:
            info = hit.get("info", {})
            authors = []
            author_value = info.get("authors", {}).get("author")
            if isinstance(author_value, list):
                authors = [str(item.get("text", "") if isinstance(item, dict) else item) for item in author_value]
            elif isinstance(author_value, dict):
                authors = [str(author_value.get("text", "") or "")]
            elif isinstance(author_value, str):
                authors = [author_value]
            year = info.get("year")
            try:
                year = int(year) if year else None
            except (TypeError, ValueError):
                year = None
            records.append(
                {
                    "title": str(info.get("title", "") or ""),
                    "authors": [author for author in authors if author],
                    "venue": str(info.get("venue", "") or ""),
                    "year": year,
                    "doi": str(info.get("doi", "") or "").lower(),
                    "arxiv_id": "",
                    "url": str(info.get("url", "") or ""),
                }
            )
        return records
