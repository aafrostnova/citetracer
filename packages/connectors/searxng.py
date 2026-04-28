from __future__ import annotations

from typing import Any

from packages.core.models import CitationRecord

from .base import BaseConnector, RequestPolicy
from .google_search import _build_normalized_search_record, _build_query, _source_from_url


class SearxNGConnector(BaseConnector):
    name = "searxng_search"
    ttl_s = 60 * 60 * 6

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or "").rstrip("/")

    def search(self, citation: CitationRecord, policy: RequestPolicy) -> list[dict[str, Any]]:
        if not self.base_url:
            return []
        query = _build_query(citation)
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
            headers={"X-Forwarded-For": "127.0.0.1"},
        )
        return [
            _normalize_searxng_item(item, query=query)
            for item in payload.get("results", [])[:5]
        ]


def _normalize_searxng_item(item: dict[str, Any], query: str) -> dict[str, Any]:
    title = str(item.get("title", "") or "")
    snippet = str(item.get("content", "") or "")
    link = str(item.get("url", "") or "")
    source = _source_from_url(link)
    author_text = ""
    raw_authors = item.get("author") or item.get("authors")
    if isinstance(raw_authors, str):
        author_text = raw_authors
    elif isinstance(raw_authors, list):
        author_text = ", ".join(str(a) for a in raw_authors if a)
    date_text = str(item.get("publishedDate") or item.get("pubdate") or "")

    record = _build_normalized_search_record(
        query=query,
        title=title,
        snippet=snippet,
        link=link,
        source=source,
        author_text=author_text,
        date_text=date_text,
        raw_item=item,
    )
    blob_parts = []
    if title:
        blob_parts.append(f"Title: {title}")
    if author_text:
        blob_parts.append(f"Authors: {author_text}")
    if date_text:
        blob_parts.append(f"Published: {date_text}")
    if source:
        blob_parts.append(f"Source: {source}")
    if link:
        blob_parts.append(f"URL: {link}")
    if snippet:
        blob_parts.append(f"Abstract: {snippet}")
    if blob_parts:
        record["raw_content"] = "\n".join(blob_parts)
    if item.get("score") is not None:
        record["search_score"] = item.get("score")
    return record
