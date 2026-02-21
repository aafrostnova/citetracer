from __future__ import annotations

import re

from packages.core.models import CitationRecord
from packages.core.normalize import extract_identifier

YEAR_RE = re.compile(r"\b(19|20)\d{2}[a-z]?\b")
TITLE_QUOTE_RE = re.compile(r"[\"“](.+?)[\"”]")
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)


def _split_reference_segments(entry: str) -> list[str]:
    # Avoid splitting on initials like "T. J. M." while still splitting sentence boundaries.
    protected = re.sub(r"\b([A-Z])\.", r"\1§", entry)
    protected = re.sub(r"\s+", " ", protected).strip()
    parts = [part.strip(" .;") for part in protected.split(". ") if part.strip(" .;")]
    segments = [part.replace("§", ".").strip(" .;") for part in parts]
    return [segment for segment in segments if segment]


def _parse_authors_from_segment(first_segment: str) -> list[str]:
    if not first_segment:
        return []
    text = first_segment.replace(";", ",")
    text = re.sub(r"\s+(?:and|&)\s+", ", ", text, flags=re.IGNORECASE)
    parts = [piece.strip(" ,.") for piece in text.split(",") if piece.strip(" ,.")]
    return parts


def _parse_title(segments: list[str], raw_entry: str) -> tuple[str, int | None]:
    if not segments:
        return "", None

    # Prefer explicit quoted title if present.
    if raw_entry:
        quoted = TITLE_QUOTE_RE.search(raw_entry)
        if quoted:
            return quoted.group(1).strip(), None

    # Usually segment[0] = authors, segment[1] = title.
    for idx in range(1, len(segments)):
        candidate = segments[idx].strip(" .;")
        lowered = candidate.lower()
        if not candidate:
            continue
        if lowered.startswith(("in ", "doi:", "url:", "http://", "https://")):
            continue
        if len(candidate) < 6:
            continue
        return candidate, idx

    return "", None


def _parse_venue(segments: list[str], title_index: int | None) -> str:
    if not segments:
        return ""

    start = 1
    if title_index is not None:
        start = title_index + 1
    if start >= len(segments):
        return ""

    venue_parts: list[str] = []
    for seg in segments[start:]:
        lowered = seg.lower().strip()
        if not lowered:
            continue
        if lowered.startswith(("doi:", "url:", "http://", "https://")):
            break
        venue_parts.append(seg.strip(" .;"))

    return ". ".join(part for part in venue_parts if part).strip(" .;")


def _extract_url(raw_text: str) -> str:
    match = URL_RE.search(raw_text)
    if not match:
        return ""
    return match.group(0).rstrip(".,);]")


def _extract_year(raw_text: str) -> int | None:
    year_match = YEAR_RE.search(raw_text)
    if not year_match:
        return None
    try:
        return int(year_match.group(0)[:4])
    except (TypeError, ValueError):
        return None


def parse_reference_entry(entry: str, citation_id: str) -> CitationRecord:
    normalized = re.sub(r"\s+", " ", entry).strip()
    segments = _split_reference_segments(normalized)

    authors = _parse_authors_from_segment(segments[0]) if segments else []
    title, title_index = _parse_title(segments, normalized)
    venue = _parse_venue(segments, title_index)
    year = _extract_year(normalized)
    doi, arxiv_id = extract_identifier(entry)
    doi = doi.rstrip(".,;")
    url = _extract_url(normalized)

    return CitationRecord(
        citation_id=citation_id,
        raw_text=normalized,
        title=title,
        authors=authors,
        venue=venue,
        year=year,
        doi=doi,
        arxiv_id=arxiv_id,
        url=url,
        parsed_fields={
            "parsed_from": "pdf_reference",
            "raw_year": year,
            "raw_authors": authors,
            "segments": segments,
        },
    )


def parse_reference_entries(entries: list[str]) -> list[CitationRecord]:
    records = []
    for idx, entry in enumerate(entries, start=1):
        records.append(parse_reference_entry(entry, f"pdf-ref:{idx}"))
    return records
