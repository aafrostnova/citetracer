from __future__ import annotations

import json
import os
import re
from typing import Any
from urllib.parse import urlparse

from packages.core.models import CitationRecord
from packages.core.normalize import extract_identifier

from .base import BaseConnector, RequestPolicy


def _query_field_mode() -> str:
    """Decide which fields to send in a web search query.

    Returns one of:
      'extreme'  — title + authors only (CITATION_CHECKER_EXTREME_RELAXED=1)
      'relaxed'  — title + authors + venue + year
                   (CITATION_CHECKER_RELAXED_FIELDS=1)
      'full'     — original behaviour: prefer raw_text; fall back to all fields
    """
    truthy = {"1", "true", "yes", "on"}
    if os.getenv("CITATION_CHECKER_EXTREME_RELAXED", "0").strip().lower() in truthy:
        return "extreme"
    if os.getenv("CITATION_CHECKER_RELAXED_FIELDS", "0").strip().lower() in truthy:
        return "relaxed"
    return "full"


class WebSearchConnector(BaseConnector):
    name = "web_search"
    ttl_s = 60 * 60 * 6

    def __init__(
        self,
        provider: str | None = None,
        api_key: str | None = None,
        cse_id: str | None = None,
        serpapi_key: str | None = None,
        tavily_api_key: str | None = None,
    ) -> None:
        self.provider = _normalize_provider(provider)
        self.api_key = (api_key or "").strip()
        self.cse_id = (cse_id or "").strip()
        self.serpapi_key = (serpapi_key or "").strip()
        self.tavily_api_key = (tavily_api_key or "").strip()
        self.cache_identity = f"{self.name}:{self._resolve_provider() or 'disabled'}"

    def search(self, citation: CitationRecord, policy: RequestPolicy) -> list[dict[str, Any]]:
        query = _build_query(citation)
        if not query:
            return []
        provider = self._resolve_provider()
        if provider == "google_cse":
            return self._search_google_cse(query, policy)
        if provider == "serpapi":
            return self._search_serpapi(query, policy)
        if provider == "tavily":
            return self._search_tavily(query, policy)
        return []

    def _resolve_provider(self) -> str:
        if self.provider in {"google_cse", "serpapi", "tavily"}:
            return self.provider
        if self.api_key and self.cse_id:
            return "google_cse"
        if self.serpapi_key:
            return "serpapi"
        if self.tavily_api_key:
            return "tavily"
        return ""

    def _search_google_cse(self, query: str, policy: RequestPolicy) -> list[dict[str, Any]]:
        payload = self._request_json(
            "https://www.googleapis.com/customsearch/v1",
            {
                "key": self.api_key,
                "cx": self.cse_id,
                "q": query,
                "num": 5,
            },
            policy,
        )
        return [
            self._normalize_google_like_item(item, query=query)
            for item in payload.get("items", [])[:5]
        ]

    def _search_serpapi(self, query: str, policy: RequestPolicy) -> list[dict[str, Any]]:
        payload = self._request_json(
            "https://serpapi.com/search.json",
            {
                "engine": "google",
                "q": query,
                "hl": "en",
                "num": 5,
                "api_key": self.serpapi_key,
            },
            policy,
        )
        return [
            self._normalize_google_like_item(item, query=query)
            for item in payload.get("organic_results", [])[:5]
        ]

    def _search_tavily(self, query: str, policy: RequestPolicy) -> list[dict[str, Any]]:
        payload = self._request_json_post(
            "https://api.tavily.com/search",
            {
                "query": query,
                "topic": "general",
                "search_depth": "basic",
                "max_results": 5,
                "include_answer": False,
                "include_raw_content": "markdown",
                "include_images": False,
                "include_favicon": True,
            },
            policy,
            headers={"Authorization": f"Bearer {self.tavily_api_key}"},
        )
        return [
            self._normalize_tavily_item(item, query=query)
            for item in payload.get("results", [])[:5]
        ]

    @staticmethod
    def _normalize_google_like_item(item: dict[str, Any], query: str) -> dict[str, Any]:
        title = str(item.get("title", "") or "")
        snippet = str(item.get("snippet", "") or "")
        link = str(item.get("link", "") or "")
        source = str(item.get("source", "") or "")
        author_text = str(item.get("author", "") or "")
        date_text = str(item.get("date", "") or "")
        return _build_normalized_search_record(
            query=query,
            title=title,
            snippet=snippet,
            link=link,
            source=source,
            author_text=author_text,
            date_text=date_text,
            raw_item=item,
        )

    @staticmethod
    def _normalize_tavily_item(item: dict[str, Any], query: str) -> dict[str, Any]:
        title = str(item.get("title", "") or "")
        snippet = str(item.get("content", "") or "")
        raw_content = str(item.get("raw_content", "") or "")
        link = str(item.get("url", "") or "")
        source = _source_from_url(link)
        # Use raw_content (full page markdown) as the snippet if available,
        # falling back to the short content summary.
        effective_snippet = raw_content[:5000] if raw_content else snippet
        record = _build_normalized_search_record(
            query=query,
            title=title,
            snippet=effective_snippet,
            link=link,
            source=source,
            author_text="",
            date_text="",
            raw_item=item,
        )
        if raw_content:
            record["raw_content"] = raw_content
        favicon = str(item.get("favicon", "") or "").strip()
        if favicon:
            record["search_favicon"] = favicon
        if item.get("score") is not None:
            record["search_score"] = item.get("score")
        return record


class GoogleSearchConnector(WebSearchConnector):
    pass


def _normalize_provider(provider: str | None) -> str:
    value = str(provider or "").strip().lower().replace("-", "_")
    if value in {"google", "google_custom_search", "cse"}:
        return "google_cse"
    if value in {"serpapi", "tavily", "auto"}:
        return value
    return ""


def _build_normalized_search_record(
    query: str,
    title: str,
    snippet: str,
    link: str,
    source: str,
    author_text: str,
    date_text: str,
    raw_item: dict[str, Any],
) -> dict[str, Any]:
    blob = " ".join(part for part in [title, snippet, source, author_text, date_text] if part).strip()
    doi, arxiv_id = extract_identifier(" ".join(part for part in [blob, link] if part))
    search_result_text = _stringify_search_item(
        item={
            "title": title,
            "link": link,
            "source": source,
            "author": author_text,
            "date": date_text,
            "snippet": snippet,
        },
        query=query,
    )
    return {
        "title": "",
        "authors": [],
        "venue": None,
        "year": None,
        "doi": "",
        "arxiv_id": "",
        "url": link,
        "snippet": snippet,
        "search_source": source,
        "search_author": author_text,
        "search_date": date_text,
        "search_query": query,
        "search_result_text": search_result_text,
        "search_result_json": json.dumps(raw_item, ensure_ascii=False, sort_keys=True),
        "heuristic_doi": doi,
        "heuristic_arxiv_id": arxiv_id,
    }


def _build_query(citation: CitationRecord) -> str:
    # Mode-aware query:
    #   - 'extreme'  → title + authors only (matches the extreme-relaxed
    #                  verifier which only checks those two fields).
    #   - 'relaxed'  → title + authors + venue + year (matches the relaxed
    #                  verifier which masks all other fields).
    #   - 'full'     → original behaviour: prefer raw_text; fall back to all
    #                  structured fields.
    mode = _query_field_mode()
    title = (citation.title or "").strip()
    all_authors = _extract_all_authors(citation.authors)
    venue = (citation.venue or "").strip()
    year = str(citation.year).strip() if citation.year is not None else ""

    if mode == "extreme":
        parts = [p for p in (title, all_authors) if p]
        if parts:
            return " ".join(parts)
        return (citation.doi or "").strip()

    if mode == "relaxed":
        parts = [p for p in (title, all_authors, venue, year) if p]
        if parts:
            return " ".join(parts)
        return (citation.doi or "").strip()

    # 'full' (default): raw_text wins when available.
    raw = (citation.raw_text or "").strip()
    if raw and len(raw) > 20:
        return raw[:500]

    parts = [p for p in (title, all_authors, venue, year) if p]
    if parts:
        return " ".join(parts)
    return (citation.doi or "").strip()


def _normalize_query_author(author: str) -> str:
    value = str(author or "").strip()
    if not value:
        return ""
    if "," in value:
        left, _, right = value.partition(",")
        left = left.strip()
        right = right.strip()
        if left and right:
            return f"{left} {right}"
        return left or right
    return value


def _extract_all_authors(authors: list[str]) -> str:
    normalized = [_normalize_query_author(author) for author in authors]
    normalized = [author for author in normalized if author]
    return " ".join(normalized)


def _extract_year(text: str) -> int | None:
    match = re.search(r"(19|20)\d{2}", text)
    return int(match.group(0)) if match else None


def _source_from_url(url: str) -> str:
    netloc = urlparse(str(url or "")).netloc.lower().strip()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def _stringify_search_item(item: dict[str, Any], query: str) -> str:
    # NOTE: query is intentionally excluded from the result text to prevent
    # downstream LLM from confusing query content (which contains citation
    # metadata like authors/venue) with actual search result evidence.
    # The query is stored separately in the "search_query" field.
    lines = [
        f"title: {str(item.get('title', '') or '').strip()}",
        f"link: {str(item.get('link', '') or '').strip()}",
        f"source: {str(item.get('source', '') or '').strip()}",
        f"author: {str(item.get('author', '') or '').strip()}",
        f"date: {str(item.get('date', '') or '').strip()}",
        f"snippet: {str(item.get('snippet', '') or '').strip()}",
    ]
    return "\n".join(line for line in lines if line.split(": ", 1)[1])


def _extract_authors(blob: str, author_text: str) -> list[str]:
    if author_text.strip():
        cleaned = re.sub(r"^\s*by\s+", "", author_text.strip(), flags=re.IGNORECASE)
        parts = [part.strip() for part in re.split(r",| and ", cleaned) if part.strip()]
        if parts:
            return _unique_preserve(parts)

    blocked = {
        "university",
        "college",
        "school",
        "department",
        "institute",
        "laboratory",
        "center",
        "centre",
        "faculty",
    }
    out: list[str] = []
    for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", blob):
        candidate = match.group(1).strip()
        parts = candidate.lower().split()
        if any(part in blocked for part in parts):
            continue
        out.append(candidate)
        if len(out) >= 8:
            break
    return _unique_preserve(out)


def _extract_venue(item: dict[str, Any], default_venue: str) -> str:
    source = str(item.get("source", "") or "").strip()
    if source:
        return source

    title = str(item.get("title", "") or "")
    snippet = str(item.get("snippet", "") or "")
    blob = f"{title} {snippet}"
    link = str(item.get("link", "") or "")
    netloc = urlparse(link).netloc.lower()

    venue_patterns = [
        (r"\bneurips\b|\bneural information processing systems\b", "NeurIPS"),
        (r"\bicml\b|\binternational conference on machine learning\b", "ICML"),
        (r"\biclr\b|\binternational conference on learning representations\b", "ICLR"),
        (r"\bcvpr\b|\bcomputer vision and pattern recognition\b", "CVPR"),
        (r"\bwacv\b|\bwinter conference on applications of computer vision\b", "WACV"),
        (r"\barxiv\b", "arXiv"),
        (r"\bopenreview\b", "OpenReview"),
    ]
    lowered_blob = blob.lower()
    for pattern, label in venue_patterns:
        if re.search(pattern, lowered_blob):
            return label

    if "arxiv.org" in netloc:
        return "arXiv"
    if "openreview.net" in netloc:
        return "OpenReview"
    if "aclanthology.org" in netloc:
        return "ACL Anthology"
    if "huggingface.co" in netloc:
        return "Hugging Face"
    if "researchgate.net" in netloc:
        return "ResearchGate"
    return default_venue


def _unique_preserve(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = (item or "").strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out
