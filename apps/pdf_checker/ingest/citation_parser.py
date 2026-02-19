from __future__ import annotations

import re

from packages.core.models import CitationRecord
from packages.core.normalize import extract_identifier

YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
TITLE_QUOTE_RE = re.compile(r"[\"“](.+?)[\"”]")



def _parse_authors(text: str, year_pos: int) -> list[str]:
    prefix = text[:year_pos].strip(" .,")
    if not prefix:
        return []
    prefix = prefix.replace(";", " and ")
    return [piece.strip() for piece in prefix.split(" and ") if piece.strip()]


def _parse_title(text: str, year_pos: int) -> str:
    quoted = TITLE_QUOTE_RE.search(text)
    if quoted:
        return quoted.group(1).strip()
    suffix = text[year_pos + 4 :].strip(" .,;")
    # Heuristic: title often terminates at first period after year.
    parts = suffix.split(".")
    if parts and parts[0].strip():
        return parts[0].strip()
    return suffix


def _parse_venue(text: str, title: str, year: int | None) -> str:
    candidate = text
    if year:
        candidate = candidate.split(str(year), maxsplit=1)[-1]
    if title and title in candidate:
        candidate = candidate.split(title, maxsplit=1)[-1]
    candidate = candidate.strip(" .,")
    pieces = [piece.strip() for piece in candidate.split(".") if piece.strip()]
    return pieces[0] if pieces else ""


def parse_reference_entry(entry: str, citation_id: str) -> CitationRecord:
    year_match = YEAR_RE.search(entry)
    year = int(year_match.group(0)) if year_match else None
    year_pos = year_match.start() if year_match else 0

    authors = _parse_authors(entry, year_pos) if year_match else []
    title = _parse_title(entry, year_pos) if year_match else ""
    venue = _parse_venue(entry, title, year)
    doi, arxiv_id = extract_identifier(entry)

    return CitationRecord(
        citation_id=citation_id,
        raw_text=entry,
        title=title,
        authors=authors,
        venue=venue,
        year=year,
        doi=doi,
        arxiv_id=arxiv_id,
        parsed_fields={
            "parsed_from": "pdf_reference",
            "raw_year": year,
            "raw_authors": authors,
        },
    )


def parse_reference_entries(entries: list[str]) -> list[CitationRecord]:
    records = []
    for idx, entry in enumerate(entries, start=1):
        records.append(parse_reference_entry(entry, f"pdf-ref:{idx}"))
    return records
