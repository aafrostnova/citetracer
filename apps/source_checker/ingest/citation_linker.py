from __future__ import annotations

import re

from packages.core.models import CitationRecord


YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _authors_from_bib(author_field: str) -> list[str]:
    if not author_field:
        return []
    return [author.strip() for author in author_field.split(" and ") if author.strip()]


def _extract_year(value: str) -> int | None:
    if not value:
        return None
    match = YEAR_RE.search(value)
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def build_citation_records(
    key_to_locations: dict[str, list[str]],
    bib_entries: dict[str, dict[str, str]],
) -> list[CitationRecord]:
    records: list[CitationRecord] = []
    for key in sorted(key_to_locations):
        entry = bib_entries.get(key, {})
        title = entry.get("title", "")
        authors = _authors_from_bib(entry.get("author", ""))
        venue = entry.get("journal", "") or entry.get("booktitle", "")
        year = _extract_year(entry.get("year", ""))
        doi = entry.get("doi", "").lower()
        arxiv_id = entry.get("eprint", "")
        url = entry.get("url", "")

        raw_text = " ".join(
            [
                piece
                for piece in [
                    "; ".join(authors),
                    f"({year})" if year else "",
                    title,
                    venue,
                    f"doi:{doi}" if doi else "",
                ]
                if piece
            ]
        )
        records.append(
            CitationRecord(
                citation_id=f"bib:{key}",
                raw_text=raw_text,
                title=title,
                authors=authors,
                venue=venue,
                year=year,
                doi=doi,
                arxiv_id=arxiv_id,
                url=url,
                source_span=key,
                provenance={
                    "citation_key": key,
                    "locations": key_to_locations.get(key, []),
                },
                parsed_fields=entry,
            )
        )
    return records
