from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlparse

from packages.core.models import CitationRecord
from packages.core.normalize import extract_identifier

from .base import BaseConnector, RequestPolicy


class GoogleSearchConnector(BaseConnector):
    name = "google_search"
    ttl_s = 60 * 60 * 6

    def __init__(
        self,
        api_key: str | None = None,
        cse_id: str | None = None,
        serpapi_key: str | None = None,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.cse_id = (cse_id or "").strip()
        self.serpapi_key = (serpapi_key or "").strip()

    def search(self, citation: CitationRecord, policy: RequestPolicy) -> list[dict[str, Any]]:
        query = _build_query(citation)
        if not query:
            return []
        if self.api_key and self.cse_id:
            return self._search_google_cse(query, policy)
        if self.serpapi_key:
            return self._search_serpapi(query, policy)
        return []

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
            self._normalize_item(item, venue="Google Search", query=query)
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
            self._normalize_item(item, query=query)
            for item in payload.get("organic_results", [])[:5]
        ]

    @staticmethod
    def _normalize_item(item: dict[str, Any], query: str) -> dict[str, Any]:
        title = str(item.get("title", "") or "")
        snippet = str(item.get("snippet", "") or "")
        link = str(item.get("link", "") or "")
        source = str(item.get("source", "") or "")
        author_text = str(item.get("author", "") or "")
        date_text = str(item.get("date", "") or "")
        blob = " ".join(part for part in [title, snippet, source, author_text, date_text] if part).strip()
        doi, arxiv_id = extract_identifier(" ".join(part for part in [blob, link] if part))
        search_result_text = _stringify_search_item(item=item, query=query)
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
            "search_result_json": json.dumps(item, ensure_ascii=False, sort_keys=True),
            "heuristic_doi": doi,
            "heuristic_arxiv_id": arxiv_id,
        }


def _build_query(citation: CitationRecord) -> str:
    title = (citation.title or "").strip()
    all_authors = _extract_all_authors(citation.authors)
    venue = (citation.venue or "").strip()
    year = str(citation.year).strip() if citation.year is not None else ""

    parts: list[str] = []
    if title:
        parts.append(title)
    if all_authors:
        parts.append(all_authors)
    if venue:
        parts.append(venue)
    if year:
        parts.append(year)

    if parts:
        return " ".join(parts)
    return (citation.raw_text or citation.doi or "").strip()


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


def _stringify_search_item(item: dict[str, Any], query: str) -> str:
    lines = [
        f"query: {query}",
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
