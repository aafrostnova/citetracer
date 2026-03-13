from __future__ import annotations

import re
from typing import Any

from packages.core.models import CitationRecord
from packages.core.normalize import extract_identifier

from .base import BaseConnector, RequestPolicy


class GoogleScholarConnector(BaseConnector):
    name = "google_scholar"
    ttl_s = 60 * 60 * 6

    def __init__(self, serpapi_key: str | None = None) -> None:
        self.serpapi_key = (serpapi_key or "").strip()

    def search(self, citation: CitationRecord, policy: RequestPolicy) -> list[dict[str, Any]]:
        if not self.serpapi_key:
            return []
        query = citation.title or citation.raw_text or citation.doi
        if not query:
            return []

        payload = self._request_json(
            "https://serpapi.com/search.json",
            {
                "engine": "google_scholar",
                "q": query,
                "hl": "en",
                "num": 5,
                "api_key": self.serpapi_key,
            },
            policy,
        )
        records: list[dict[str, Any]] = []
        for item in payload.get("organic_results", [])[:5]:
            record = self._normalize_item(item, policy)
            if any(record.values()):
                records.append(record)
        return records

    def _normalize_item(self, item: dict[str, Any], policy: RequestPolicy) -> dict[str, Any]:
        title = str(item.get("title", "") or "").strip()
        link = str(item.get("link", "") or "").strip()
        publication_info = item.get("publication_info") or {}
        summary = str(publication_info.get("summary", "") or "").strip()
        snippet = str(item.get("snippet", "") or summary).strip()
        cite_payload = self._fetch_cite_payload(item, policy)

        harvard_record = _record_from_harvard_citation(cite_payload)
        if harvard_record is not None:
            harvard_record.setdefault("title", title)
            harvard_record.setdefault("url", link)
            if not harvard_record.get("venue"):
                harvard_record["venue"] = _extract_venue_from_summary(summary) or ""
            if not harvard_record.get("doi") or not harvard_record.get("arxiv_id"):
                doi, arxiv_id = extract_identifier(" ".join([title, snippet, link]))
                harvard_record["doi"] = str(harvard_record.get("doi") or doi or "").lower()
                harvard_record["arxiv_id"] = str(harvard_record.get("arxiv_id") or arxiv_id or "").lower()
            return harvard_record

        bibtex_record = self._record_from_bibtex_link(cite_payload, policy)
        if bibtex_record is not None:
            bibtex_record.setdefault("title", title)
            bibtex_record.setdefault("url", link)
            if not bibtex_record.get("arxiv_id") or not bibtex_record.get("doi"):
                doi, arxiv_id = extract_identifier(" ".join([title, snippet, link]))
                bibtex_record["doi"] = str(bibtex_record.get("doi") or doi or "").lower()
                bibtex_record["arxiv_id"] = str(bibtex_record.get("arxiv_id") or arxiv_id or "").lower()
            if not bibtex_record.get("venue"):
                bibtex_record["venue"] = "Google Scholar"
            return bibtex_record

        authors = _parse_authors_from_summary(summary)
        year = _extract_year(f"{summary} {snippet}")
        doi, arxiv_id = extract_identifier(" ".join([title, snippet, link]))
        return {
            "title": title,
            "authors": authors,
            "venue": _extract_venue_from_summary(summary) or "Google Scholar",
            "year": year,
            "doi": doi,
            "arxiv_id": arxiv_id,
            "url": link,
        }

    def _fetch_cite_payload(self, item: dict[str, Any], policy: RequestPolicy) -> dict[str, Any]:
        result_id = str(item.get("result_id", "") or "").strip()
        cite_link = str(item.get("serpapi_cite_link", "") or "").strip()
        if result_id:
            return self._request_json(
                "https://serpapi.com/search.json",
                {
                    "engine": "google_scholar_cite",
                    "q": result_id,
                    "hl": "en",
                    "api_key": self.serpapi_key,
                },
                policy,
            )
        if cite_link:
            return self._request_json(
                cite_link,
                {},
                policy,
            )
        return {}

    def _record_from_bibtex_link(self, cite_payload: dict[str, Any], policy: RequestPolicy) -> dict[str, Any] | None:
        links = cite_payload.get("links", []) or []
        bibtex_link = ""
        for item in links:
            if not isinstance(item, dict):
                continue
            if str(item.get("name", "")).strip().lower() == "bibtex":
                bibtex_link = str(item.get("link", "") or "").strip()
                break
        if not bibtex_link:
            return None

        try:
            bibtex_text = self._request_text(bibtex_link, {}, policy)
        except Exception:
            return None
        return _parse_bibtex_record(bibtex_text)


def _extract_year(text: str) -> int | None:
    match = re.search(r"\b(19\d{2}|20\d{2})\b", text or "")
    return int(match.group(1)) if match else None


def _parse_authors_from_summary(summary: str) -> list[str]:
    if not summary:
        return []
    parts = [part.strip() for part in summary.split("-")]
    if not parts:
        return []
    left = parts[0]
    authors = []
    for chunk in re.split(r",| and | & ", left):
        value = chunk.strip()
        if value:
            authors.append(value)
    return _unique_preserve(authors)


def _extract_venue_from_summary(summary: str) -> str:
    if not summary:
        return ""
    parts = [part.strip() for part in summary.split("-") if part.strip()]
    if len(parts) >= 2:
        # Common Scholar summary layout: authors - venue, year
        return parts[1].rstrip(",")
    return ""


def _parse_bibtex_record(bibtex_text: str) -> dict[str, Any]:
    title = _extract_bibtex_field(bibtex_text, "title")
    author_text = _extract_bibtex_field(bibtex_text, "author")
    year = _extract_year(_extract_bibtex_field(bibtex_text, "year"))
    venue = (
        _extract_bibtex_field(bibtex_text, "journal")
        or _extract_bibtex_field(bibtex_text, "booktitle")
        or _extract_bibtex_field(bibtex_text, "publisher")
    )
    doi = _extract_bibtex_field(bibtex_text, "doi").lower()
    url = _extract_bibtex_field(bibtex_text, "url")
    doi_from_text, arxiv_id_from_text = extract_identifier(" ".join([bibtex_text, url]))
    authors = []
    if author_text:
        authors = _unique_preserve([part.strip() for part in author_text.split(" and ") if part.strip()])
    return {
        "title": title,
        "authors": authors,
        "venue": venue,
        "year": year,
        "doi": doi or doi_from_text,
        "arxiv_id": arxiv_id_from_text,
        "url": url,
    }


def _record_from_harvard_citation(cite_payload: dict[str, Any]) -> dict[str, Any] | None:
    citations = cite_payload.get("citations", []) or []
    for item in citations:
        if not isinstance(item, dict):
            continue
        if str(item.get("title", "") or "").strip().lower() != "harvard":
            continue
        snippet = str(item.get("snippet", "") or "").strip()
        if not snippet:
            return None
        return _parse_harvard_citation(snippet)
    return None


def _parse_harvard_citation(snippet: str) -> dict[str, Any] | None:
    text = " ".join(str(snippet or "").split()).strip()
    if not text:
        return None

    year_match = re.search(
        r",\s*(?P<year>(?:19|20)\d{2})(?:,\s*(?P<month>[A-Za-z]+))?\.",
        text,
    )
    year = int(year_match.group("year")) if year_match else None

    author_part = text
    citation_body = ""
    if year_match:
        author_part = text[: year_match.start()].strip(" ,.;")
        citation_body = text[year_match.end() :].strip(" ,.;")
    else:
        title_start = re.search(r"\.\s+(?=[A-Z][a-z])", text)
        if title_start:
            author_part = text[: title_start.start()].strip(" ,.;")
            citation_body = text[title_start.end() :].strip(" ,.;")

    authors = _parse_harvard_authors(author_part)
    title, venue = _split_harvard_title_and_venue(citation_body)

    doi, arxiv_id = extract_identifier(text)
    return {
        "title": title,
        "authors": authors,
        "venue": venue or "Google Scholar",
        "year": year,
        "doi": doi,
        "arxiv_id": arxiv_id,
        "url": "",
    }


def _parse_harvard_authors(author_part: str) -> list[str]:
    if not author_part:
        return []
    text = author_part.replace(" and ", ", ")
    text = re.sub(r"\bet al\b\.?", "", text, flags=re.IGNORECASE)
    matches = re.finditer(r"([A-Z][A-Za-z'`\-]+,\s*(?:[A-Z]\.)+)", text)
    authors = [match.group(1).strip(" ,.;") for match in matches]
    return _unique_preserve(authors)


def _split_harvard_title_and_venue(citation_body: str) -> tuple[str, str]:
    body = citation_body.strip(" ,.;")
    if not body:
        return "", ""

    in_match = re.search(r"\.\s+In\s+(?P<venue>.+)$", body, flags=re.IGNORECASE)
    if in_match:
        title = body[: in_match.start()].strip(" ,.;")
        venue = in_match.group("venue").strip(" ,.;")
        return title, venue

    parts = re.split(r"\.\s+", body, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(" ,.;"), parts[1].strip(" ,.;")
    return body, ""


def _extract_bibtex_field(bibtex_text: str, field: str) -> str:
    match = re.search(
        rf"{re.escape(field)}\s*=\s*[\{{\"](?P<value>.*?)[\}}\"]\s*,?\s*(?:\n|$)",
        bibtex_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return ""
    return " ".join(match.group("value").replace("\n", " ").split()).strip()


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
