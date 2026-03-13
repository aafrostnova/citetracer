from __future__ import annotations

import re
from html import unescape
from urllib.parse import urlsplit

from packages.core.models import CitationRecord
from packages.core.normalize import extract_identifier

from .base import BaseConnector, RequestPolicy


META_RE = re.compile(
    r"<meta[^>]+(?:name|property)\s*=\s*[\"'](?P<key>[^\"']+)[\"'][^>]+content\s*=\s*[\"'](?P<value>[^\"']*)[\"'][^>]*>",
    flags=re.IGNORECASE,
)
TITLE_RE = re.compile(r"<title[^>]*>(?P<title>.*?)</title>", flags=re.IGNORECASE | re.DOTALL)
YEAR_RE = re.compile(r"(19|20)\d{2}")


class URLDirectConnector(BaseConnector):
    name = "url_direct"
    ttl_s = 60 * 60 * 24
    def search(self, citation: CitationRecord, policy: RequestPolicy) -> list[dict[str, object]]:
        url = str(citation.url or "").strip()
        if not url or not urlsplit(url).scheme:
            return []

        body = self._request_text(
            url,
            {},
            policy,
            headers={"Accept": "text/html,application/xhtml+xml"},
        )
        body = body[:512_000]
        record = self._extract_record(url, body)
        return [record] if any(record.values()) else []

    def _extract_record(self, url: str, body: str) -> dict[str, object]:
        meta: dict[str, list[str]] = {}
        for match in META_RE.finditer(body):
            key = match.group("key").strip().lower()
            value = _clean(match.group("value"))
            if not key or not value:
                continue
            meta.setdefault(key, []).append(value)

        title = (
            _first_meta(meta, "citation_title")
            or _first_meta(meta, "dc.title")
            or _first_meta(meta, "og:title")
            or _extract_title_tag(body)
        )
        authors = (
            meta.get("citation_author", [])
            or meta.get("dc.creator", [])
            or meta.get("author", [])
        )
        venue = (
            _first_meta(meta, "citation_journal_title")
            or _first_meta(meta, "citation_conference_title")
            or _first_meta(meta, "og:site_name")
            or _first_meta(meta, "dc.source")
            or ""
        )
        date_text = (
            _first_meta(meta, "citation_publication_date")
            or _first_meta(meta, "citation_date")
            or _first_meta(meta, "dc.date")
            or ""
        )
        year = _extract_year(date_text) or _extract_year(body[:4000])
        doi = (
            _first_meta(meta, "citation_doi")
            or _first_meta(meta, "dc.identifier")
            or extract_identifier(url)[0]
            or extract_identifier(body[:8000])[0]
        )
        arxiv_id = extract_identifier(url)[1] or extract_identifier(body[:8000])[1]
        return {
            "title": title,
            "authors": list(authors),
            "venue": venue,
            "year": year,
            "doi": str(doi or "").lower(),
            "arxiv_id": str(arxiv_id or "").lower(),
            "url": url,
        }


def _clean(value: str) -> str:
    return " ".join(unescape(str(value or "")).split()).strip()


def _first_meta(meta: dict[str, list[str]], key: str) -> str:
    values = meta.get(key.lower(), [])
    return values[0] if values else ""


def _extract_title_tag(body: str) -> str:
    match = TITLE_RE.search(body)
    return _clean(match.group("title")) if match else ""


def _extract_year(text: str) -> int | None:
    match = YEAR_RE.search(str(text or ""))
    return int(match.group(0)) if match else None
