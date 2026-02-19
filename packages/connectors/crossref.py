from __future__ import annotations

from typing import Any

from packages.core.models import CitationRecord

from .base import BaseConnector, RequestPolicy


class CrossrefConnector(BaseConnector):
    name = "crossref"
    ttl_s = 60 * 60 * 24 * 7

    def search(self, citation: CitationRecord, policy: RequestPolicy) -> list[dict[str, Any]]:
        query = citation.doi or citation.title or citation.raw_text
        if not query:
            return []
        payload = self._request_json(
            "https://api.crossref.org/works",
            {"query.bibliographic": query, "rows": 5},
            policy,
        )
        items = payload.get("message", {}).get("items", [])
        return [self._normalize_item(item) for item in items]

    @staticmethod
    def _normalize_item(item: dict[str, Any]) -> dict[str, Any]:
        title = ""
        title_values = item.get("title") or []
        if title_values:
            title = title_values[0]

        authors = []
        for author in item.get("author", []):
            given = (author.get("given") or "").strip()
            family = (author.get("family") or "").strip()
            full_name = f"{given} {family}".strip()
            if full_name:
                authors.append(full_name)

        venue_values = item.get("container-title") or []
        year = None
        issued = item.get("issued", {}).get("date-parts", [])
        if issued and issued[0]:
            try:
                year = int(issued[0][0])
            except (TypeError, ValueError):
                year = None

        return {
            "title": title,
            "authors": authors,
            "venue": venue_values[0] if venue_values else "",
            "year": year,
            "doi": str(item.get("DOI", "") or "").lower(),
            "arxiv_id": "",
            "url": str(item.get("URL", "") or ""),
        }
