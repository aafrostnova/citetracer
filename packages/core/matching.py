from __future__ import annotations

from typing import Any

from .models import CandidateMatch, CitationRecord
from .normalize import author_overlap_score, similarity, venue_match_score, year_consistency


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


def build_candidate_match(citation: CitationRecord, connector: str, record: dict[str, Any]) -> CandidateMatch:
    title = str(record.get("title", "") or "")
    authors = _normalize_authors(record.get("authors"))
    venue = str(record.get("venue", "") or "")
    year = _to_int(record.get("year"))
    doi = str(record.get("doi", "") or "").lower()
    arxiv_id = str(record.get("arxiv_id", "") or "").lower()
    url = str(record.get("url", "") or "")

    title_score = similarity(citation.title, title)
    author_score = author_overlap_score(citation.authors, authors)
    venue_score = venue_match_score(citation.venue, venue)
    year_score = year_consistency(citation.year, year)

    identifier_boost = 0.0
    matched_fields: list[str] = []
    conflicts: list[str] = []

    if citation.doi and doi:
        if citation.doi.lower() == doi.lower():
            identifier_boost += 0.3
            matched_fields.append("doi")
        else:
            conflicts.append("doi_mismatch")

    if citation.arxiv_id and arxiv_id:
        if citation.arxiv_id.lower() == arxiv_id.lower():
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
    elif year_score == 0.0 and citation.year and year:
        conflicts.append("year_mismatch")

    score = (
        0.45 * title_score
        + 0.25 * author_score
        + 0.15 * venue_score
        + 0.15 * year_score
        + identifier_boost
    )
    score = max(0.0, min(1.0, score))

    return CandidateMatch(
        connector=connector,
        score=score,
        title=title,
        authors=authors,
        venue=venue,
        year=year,
        doi=doi,
        arxiv_id=arxiv_id,
        url=url,
        matched_fields=matched_fields,
        conflicts=conflicts,
        raw_record=record,
    )


def rank_candidates(citation: CitationRecord, connector_records: dict[str, list[dict[str, Any]]]) -> list[CandidateMatch]:
    matches: list[CandidateMatch] = []
    for connector, records in connector_records.items():
        for record in records:
            matches.append(build_candidate_match(citation, connector, record))
    matches.sort(key=lambda match: match.score, reverse=True)
    return matches
