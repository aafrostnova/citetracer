from __future__ import annotations

import re
from html import unescape
from typing import Any
from urllib.parse import urlsplit

from packages.core.models import CitationRecord
from packages.core.normalize import extract_identifier

from .base import BaseConnector, RequestPolicy


META_TAG_RE = re.compile(r"<meta\b[^>]*>", flags=re.IGNORECASE)
META_ATTR_RE = re.compile(
    r"(?P<attr>[a-zA-Z_:][\w:.-]*)\s*=\s*(?:(?P<dq>\"[^\"]*\")|(?P<sq>'[^']*')|(?P<bare>[^\s>]+))",
    flags=re.IGNORECASE,
)
TITLE_RE = re.compile(r"<title[^>]*>(?P<title>.*?)</title>", flags=re.IGNORECASE | re.DOTALL)
YEAR_RE = re.compile(r"(19|20)\d{2}")

# Patterns to locate inline citation blocks on academic sites.
# ACL Anthology: id=citeACL (quoted or unquoted)
CITE_ACL_RE = re.compile(
    r"(?:id|class)\s*=\s*[\"']?citeACL[\"']?[^>]*>",
    flags=re.IGNORECASE,
)
# PMLR and others: class="citecode" containing BibTeX
CITE_BIBTEX_RE = re.compile(
    r"class\s*=\s*[\"']?citecode[\"']?[^>]*>\s*(@\w+\{.*?\})\s*<",
    flags=re.IGNORECASE | re.DOTALL,
)

# ACL-style citation:  Authors. Year. Title. In Venue, pages X–Y, Location. Publisher.
ACL_CITE_RE = re.compile(
    r"^(?P<authors>.+?)\.\s+"
    r"(?P<year>\d{4})\.\s+"
    r"(?P<title>.+?)\.\s+"
    r"(?:In\s+)?(?P<venue>.+?)"
    r"(?:,\s*pages?\s+(?P<pages>[\d]+[–\-]+[\d]+))?"
    r"(?:,\s*(?P<location>[^.]+?))?\.\s*"
    r"(?P<publisher>[^.]+?)\.\s*$",
    flags=re.DOTALL,
)


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
        # --- Try structured cite block first (e.g. ACL Anthology) ---
        cite_record = _extract_cite_block(body)
        if cite_record:
            # Supplement DOI / arXiv ID from meta tags or URL if cite block lacks them
            if not cite_record.get("doi"):
                cite_record["doi"] = _extract_doi_from_meta_or_url(body, url)
            if not cite_record.get("arxiv_id"):
                cite_record["arxiv_id"] = extract_identifier(url)[1] or extract_identifier(body[:8000])[1] or ""
            cite_record.setdefault("url", url)
            cite_record["doi"] = str(cite_record.get("doi") or "").lower()
            cite_record["arxiv_id"] = str(cite_record.get("arxiv_id") or "").lower()
            return cite_record

        # --- Fallback: meta tag extraction ---
        return self._extract_from_meta(url, body)

    def _extract_from_meta(self, url: str, body: str) -> dict[str, object]:
        meta: dict[str, list[str]] = {}
        for tag_match in META_TAG_RE.finditer(body):
            attributes = _parse_meta_attributes(tag_match.group(0))
            key = str(attributes.get("name") or attributes.get("property") or "").strip().lower()
            value = _clean(str(attributes.get("content") or ""))
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


def _extract_cite_block(body: str) -> dict[str, Any] | None:
    """Try to extract a structured citation from an inline cite block.

    Supports:
    - ACL Anthology (id=citeACL): ACL-style plain-text citation
    - PMLR (class=citecode): BibTeX block
    """
    # --- Try ACL Anthology cite block ---
    acl_match = CITE_ACL_RE.search(body)
    if acl_match:
        start = acl_match.end()
        chunk = body[start:start + 1000]
        informal_idx = chunk.find("Cite (Informal)")
        if informal_idx > 0:
            chunk = chunk[:informal_idx]
        text = unescape(re.sub(r"<[^>]+>", "", chunk)).strip()
        text = " ".join(text.split())
        if len(text) >= 20:
            result = _parse_acl_cite(text)
            if result:
                return result

    # --- Try BibTeX cite block (PMLR, etc.) ---
    bib_match = CITE_BIBTEX_RE.search(body)
    if bib_match:
        bibtex = bib_match.group(1)
        result = _parse_bibtex_cite(bibtex)
        if result:
            return result

    return None


def _parse_acl_cite(text: str) -> dict[str, Any] | None:
    """Parse an ACL-style citation string into structured fields.

    Format: Authors. Year. Title. In Venue, pages X-Y, Location. Publisher.
    """
    m = ACL_CITE_RE.match(text)
    if m:
        raw_authors = m.group("authors").strip()
        authors = _split_author_string(raw_authors)
        venue = _clean(m.group("venue") or "")
        return {
            "title": _clean(m.group("title") or ""),
            "authors": authors,
            "venue": venue,
            "year": _extract_year(m.group("year") or ""),
            "pages": _clean(m.group("pages") or ""),
            "publisher": _clean(m.group("publisher") or ""),
            "doi": "",
            "arxiv_id": "",
        }

    # Regex didn't match perfectly — try a simpler split
    # Pattern: Authors. Year. Title. Rest...
    simple = re.match(
        r"^(?P<authors>.+?)\.\s+(?P<year>\d{4})\.\s+(?P<title>.+?)\.\s+(?P<rest>.+)$",
        text,
        re.DOTALL,
    )
    if not simple:
        return None

    raw_authors = simple.group("authors").strip()
    authors = _split_author_string(raw_authors)
    title = _clean(simple.group("title") or "")
    rest = _clean(simple.group("rest") or "")

    # Try to extract venue, pages, publisher from rest
    venue = rest
    pages = ""
    publisher = ""
    pages_match = re.search(r"pages?\s+([\d]+[–\-]+[\d]+)", rest)
    if pages_match:
        pages = pages_match.group(1)
        # Venue is everything before "pages"
        venue_end = rest.find(pages_match.group(0))
        venue = rest[:venue_end].rstrip(", ").strip()
        # Publisher is the last sentence
        after_pages = rest[pages_match.end():].strip(", .")
        parts = [p.strip() for p in after_pages.split(".") if p.strip()]
        if parts:
            publisher = parts[-1]

    # Strip leading "In " from venue
    if venue.lower().startswith("in "):
        venue = venue[3:].strip()

    return {
        "title": title,
        "authors": authors,
        "venue": venue,
        "year": _extract_year(simple.group("year") or ""),
        "pages": pages,
        "publisher": publisher,
        "doi": "",
        "arxiv_id": "",
    }


def _parse_bibtex_cite(bibtex: str) -> dict[str, Any] | None:
    """Parse a BibTeX entry into structured fields."""
    def _bib_field(field_name: str) -> str:
        m = re.search(
            rf"{field_name}\s*=\s*\{{(.*?)\}}",
            bibtex,
            re.IGNORECASE | re.DOTALL,
        )
        if not m:
            return ""
        return " ".join(m.group(1).split()).strip()

    title = _bib_field("title")
    if not title:
        return None

    raw_authors = _bib_field("author")
    # BibTeX uses "and" to separate authors, names are "Last, First"
    authors: list[str] = []
    if raw_authors:
        for part in raw_authors.split(" and "):
            part = part.strip()
            if not part:
                continue
            # Convert "Last, First" → "First Last"
            if ", " in part:
                last, first = part.split(", ", 1)
                authors.append(f"{first.strip()} {last.strip()}")
            else:
                authors.append(part)

    venue = _bib_field("booktitle") or _bib_field("journal")
    year_str = _bib_field("year")
    year = int(year_str) if year_str and year_str.isdigit() else None
    pages = _bib_field("pages").replace("--", "–")
    publisher = _bib_field("publisher")
    doi = _bib_field("doi")

    return {
        "title": title,
        "authors": authors,
        "venue": venue,
        "year": year,
        "pages": pages,
        "publisher": publisher,
        "doi": doi.lower() if doi else "",
        "arxiv_id": "",
    }


def _split_author_string(raw: str) -> list[str]:
    """Split 'A, B, and C' or 'A, B, C' into a list of author names."""
    # Remove trailing period
    raw = raw.rstrip(".")
    # Replace " and " with comma
    raw = re.sub(r",?\s+and\s+", ", ", raw)
    return [a.strip() for a in raw.split(",") if a.strip()]


def _extract_doi_from_meta_or_url(body: str, url: str) -> str:
    """Extract DOI from meta tags or URL."""
    meta: dict[str, list[str]] = {}
    for tag_match in META_TAG_RE.finditer(body):
        attributes = _parse_meta_attributes(tag_match.group(0))
        key = str(attributes.get("name") or attributes.get("property") or "").strip().lower()
        value = _clean(str(attributes.get("content") or ""))
        if key and value:
            meta.setdefault(key, []).append(value)
    return (
        _first_meta(meta, "citation_doi")
        or _first_meta(meta, "dc.identifier")
        or extract_identifier(url)[0]
        or extract_identifier(body[:8000])[0]
        or ""
    )


def _parse_meta_attributes(tag_text: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for match in META_ATTR_RE.finditer(tag_text):
        attr = str(match.group("attr") or "").strip().lower()
        raw_value = match.group("dq") or match.group("sq") or match.group("bare") or ""
        value = str(raw_value).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if attr:
            attrs[attr] = value
    return attrs


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
