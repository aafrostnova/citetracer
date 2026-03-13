from __future__ import annotations

from packages.core.models import CitationRecord

from .base import BaseConnector, RequestPolicy


class SearxNGConnector(BaseConnector):
    name = "searxng_search"
    ttl_s = 60 * 60 * 6

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or "").rstrip("/")

    def search(self, citation: CitationRecord, policy: RequestPolicy) -> list[dict[str, object]]:
        if not self.base_url:
            return []
        query = citation.title or citation.raw_text
        if not query:
            return []
        payload = self._request_json(
            f"{self.base_url}/search",
            {
                "q": query,
                "format": "json",
                "categories": "general",
            },
            policy,
        )
        records = []
        for item in payload.get("results", [])[:5]:
            records.append(
                {
                    "title": str(item.get("title", "") or ""),
                    "authors": [],
                    "venue": str(item.get("engine", "") or "Web Search"),
                    "year": _extract_year_from_content(str(item.get("content", "") or "")),
                    "doi": "",
                    "arxiv_id": "",
                    "url": str(item.get("url", "") or ""),
                }
            )
        return records


def _extract_year_from_content(content: str) -> int | None:
    for token in content.replace("/", " ").replace("-", " ").split():
        if token.isdigit() and len(token) == 4 and 1800 <= int(token) <= 2100:
            return int(token)
    return None
