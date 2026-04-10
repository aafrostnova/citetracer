from __future__ import annotations

from typing import Any

from .models import CandidateMatch, CitationRecord
from .normalize import (
    author_overlap_score,
    normalize_arxiv_id,
    normalize_title,
    similarity,
    venue_match_score,
    year_consistency,
)


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _normalize_authors(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(a).strip() for a in value if str(a).strip()]
    if isinstance(value, str):
        parts = [piece.strip() for piece in value.replace(";", " and ").split(" and ")]
        return [part for part in parts if part]
    return []


def _extract_version_years(record: dict[str, Any]) -> list[int]:
    raw = record.get("version_years")
    if not isinstance(raw, list):
        return []
    years: list[int] = []
    for item in raw:
        year = _to_int(item)
        if year is not None and year not in years:
            years.append(year)
    return years


def _edit_distance_at_most(left: str, right: str, max_distance: int) -> bool:
    if max_distance < 0:
        return False
    if left == right:
        return True
    if abs(len(left) - len(right)) > max_distance:
        return False

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        row_min = current[0]
        for j, right_char in enumerate(right, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (0 if left_char == right_char else 1)
            value = min(insertion, deletion, substitution)
            current.append(value)
            if value < row_min:
                row_min = value
        if row_min > max_distance:
            return False
        previous = current
    return previous[-1] <= max_distance


def _title_is_candidate_eligible(citation_title: str, record_title: str) -> bool:
    if not citation_title or not record_title:
        return False
    return _edit_distance_at_most(citation_title, record_title, max_distance=10)


def _year_is_candidate_eligible(citation_year: int | None, record: dict[str, Any]) -> bool:
    if citation_year is None:
        return True
    record_year = _to_int(record.get("year"))
    if record_year is None:
        return True  # Record has no year info — don't filter it out
    if record_year == citation_year:
        return True
    version_years = _extract_version_years(record)
    return citation_year in version_years


def build_candidate_match(citation: CitationRecord, connector: str, record: dict[str, Any]) -> CandidateMatch:
    title = str(record.get("title", "") or "")
    authors = _normalize_authors(record.get("authors"))
    venue = str(record.get("venue", "") or "")
    year = _to_int(record.get("year"))
    doi = str(record.get("doi", "") or "").lower()
    arxiv_id = normalize_arxiv_id(str(record.get("arxiv_id", "") or ""))
    url = str(record.get("url", "") or "")
    volume = str(record.get("volume", "") or "")
    pages = str(record.get("pages", "") or "")
    publisher = str(record.get("publisher", "") or "")
    version_years = _extract_version_years(record)

    title_score = similarity(citation.title, title)
    author_score = author_overlap_score(citation.authors, authors)
    venue_score = venue_match_score(citation.venue, venue)
    year_score = 1.0 if citation.year and citation.year in version_years else year_consistency(citation.year, year)

    identifier_boost = 0.0
    matched_fields: list[str] = []
    conflicts: list[str] = []

    if citation.doi and doi:
        if citation.doi.lower() == doi.lower():
            identifier_boost += 0.3
            matched_fields.append("doi")
        else:
            conflicts.append("doi_mismatch")

    citation_arxiv_id = normalize_arxiv_id(citation.arxiv_id)
    if citation_arxiv_id and arxiv_id:
        if citation_arxiv_id == arxiv_id:
            identifier_boost += 0.25
            matched_fields.append("arxiv_id")
        else:
            conflicts.append("arxiv_id_mismatch")

    if title_score >= 0.85:
        matched_fields.append("title")
    elif title_score < 0.45 and citation.title and title:
        conflicts.append("title_mismatch")

    if author_score >= 0.6:
        matched_fields.append("authors")
    elif author_score < 0.25 and citation.authors and authors:
        conflicts.append("author_mismatch")

    if venue_score >= 0.7:
        matched_fields.append("venue")
    elif venue_score < 0.3 and citation.venue and venue:
        conflicts.append("venue_mismatch")

    if year_score >= 1.0:
        matched_fields.append("year")
    elif year_score == 0.0 and citation.year and (year or version_years):
        conflicts.append("year_mismatch")

    return CandidateMatch(
        connector=connector,
        title=title,
        authors=authors,
        venue=venue,
        year=year,
        doi=doi,
        arxiv_id=arxiv_id,
        url=url,
        volume=volume,
        pages=pages,
        publisher=publisher,
        matched_fields=matched_fields,
        conflicts=conflicts,
        raw_record=record,
    )


def collect_candidates(citation: CitationRecord, connector_records: dict[str, list[dict[str, Any]]]) -> list[CandidateMatch]:
    matches: list[CandidateMatch] = []
    citation_title = normalize_title(citation.title)
    raw_web_connectors = {"google_search", "web_search"}
    for connector, records in connector_records.items():
        for record in records:
            if connector in raw_web_connectors:
                matches.append(build_candidate_match(citation, connector, record))
                continue
            record_title = normalize_title(str(record.get("title", "") or ""))
            if not _title_is_candidate_eligible(citation_title, record_title):
                continue
            if not _year_is_candidate_eligible(citation.year, record):
                continue
            matches.append(build_candidate_match(citation, connector, record))
    return matches


def rank_candidates(citation: CitationRecord, connector_records: dict[str, list[dict[str, Any]]]) -> list[CandidateMatch]:
    return collect_candidates(citation, connector_records)
