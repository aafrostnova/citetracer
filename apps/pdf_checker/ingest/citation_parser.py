from __future__ import annotations

import re

from packages.core.models import CitationRecord
from packages.core.normalize import extract_identifier

YEAR_RE = re.compile(r"\b(19|20)\d{2}[a-z]?\b")
TITLE_QUOTE_RE = re.compile(r"[\"“](.+?)[\"”]")
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
TITLE_STOP_PREFIXES = ("in ", "doi:", "doi ", "url:", "url ", "http://", "https://")
ENTRY_PREFIX_RE = re.compile(r"^\s*(?:\[\d{1,3}\]|\(\d{1,3}\)|\d{1,3}\.)\s*")
INITIALS_TOKEN_RE = re.compile(r"^(?:[A-Z](?:\.)?){1,4}$")
APA_RE = re.compile(
    r"^(?P<authors>.+?)\(\s*(?:19|20)\d{2}[a-z]?\s*\)\.\s*(?P<title>.+?)(?:\.\s+|$)",
    re.IGNORECASE,
)
HARVARD_RE = re.compile(
    r"^(?P<authors>.+?),\s*(?:19|20)\d{2}[a-z]?\.\s*(?P<title>.+?)(?:\.\s+|$)",
    re.IGNORECASE,
)
COLON_STYLE_RE = re.compile(r"^(?P<authors>[^:]{6,280}):\s*(?P<title>.+)$")
TITLE_CUT_RE = re.compile(
    r"\s+(?:In:|in:|Proceedings of|Lecture Notes in Computer Science|LNCS|Springer|"
    r"IEEE Transactions|Nature|Journal|arXiv preprint|arXiv|doi:|doi |URL|https?://)",
    re.IGNORECASE,
)


def _split_reference_segments(entry: str) -> list[str]:
    # Avoid splitting on initials like "T. J. M." while still splitting sentence boundaries.
    protected = re.sub(r"\b([A-Z])\.", r"\1§", entry)
    protected = re.sub(r"\s+", " ", protected).strip()
    parts = [part.strip(" .;") for part in protected.split(". ") if part.strip(" .;")]
    segments = [part.replace("§", ".").strip(" .;") for part in parts]
    return [segment for segment in segments if segment]


def _strip_entry_prefix(text: str) -> str:
    return ENTRY_PREFIX_RE.sub("", text).strip()


def _normalize_match_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _find_title_segment_index(segments: list[str], title: str) -> int | None:
    title_norm = _normalize_match_text(title)
    if not title_norm:
        return None
    for idx, segment in enumerate(segments):
        seg_norm = _normalize_match_text(segment)
        if not seg_norm:
            continue
        if title_norm == seg_norm or title_norm in seg_norm or seg_norm in title_norm:
            return idx
    return None


def _extract_title_from_colon_style(raw_entry: str) -> str:
    body = _strip_entry_prefix(raw_entry)
    match = COLON_STYLE_RE.match(body)
    if not match:
        return ""
    author_side = match.group("authors")
    title_side = match.group("title").strip(" .;")
    if "," not in author_side:
        return ""
    if not re.search(r"\b[A-Z][A-Za-z'`\-]+,\s*[A-Z]", author_side):
        return ""
    title_side = TITLE_CUT_RE.split(title_side, maxsplit=1)[0].strip(" .;")
    if not re.search(r"[A-Za-z]", title_side):
        return ""
    return title_side if len(title_side) >= 6 else ""


def _extract_title_from_apa_style(raw_entry: str) -> str:
    body = _strip_entry_prefix(raw_entry)
    match = APA_RE.match(body)
    if not match:
        return ""
    candidate = match.group("title").strip(" .;")
    return candidate if len(candidate) >= 6 else ""


def _extract_title_from_harvard_style(raw_entry: str) -> str:
    body = _strip_entry_prefix(raw_entry)
    match = HARVARD_RE.match(body)
    if not match:
        return ""
    candidate = match.group("title").strip(" .;")
    if candidate and not candidate.lower().startswith(TITLE_STOP_PREFIXES):
        return candidate
    return ""


def _detect_reference_style(raw_entry: str) -> str:
    body = _strip_entry_prefix(raw_entry)
    if TITLE_QUOTE_RE.search(raw_entry):
        return "mla_or_chicago"
    if _extract_title_from_colon_style(raw_entry):
        return "lncs_or_springer_numeric"
    if APA_RE.match(body):
        return "apa"
    if HARVARD_RE.match(body):
        return "harvard"
    if ENTRY_PREFIX_RE.match(raw_entry):
        return "vancouver_or_numeric"
    return "unknown"


def _parse_authors_from_segment(first_segment: str) -> list[str]:
    if not first_segment:
        return []
    text = _strip_entry_prefix(first_segment)
    if ":" in text:
        before_colon, after_colon = text.split(":", 1)
        if "," in before_colon and re.search(r"[A-Za-z]", after_colon):
            text = before_colon
    text = text.replace(";", ",")
    text = re.sub(r"\s+(?:and|&)\s+", ", ", text, flags=re.IGNORECASE)
    parts = [piece.strip(" ,") for piece in text.split(",") if piece.strip(" ,")]
    merged_parts: list[str] = []
    index = 0
    while index < len(parts):
        token = parts[index]
        if index + 1 < len(parts):
            initials = re.sub(r"\s+", "", parts[index + 1])
            # LNCS/APA-like author formatting: "Surname, A." or "Surname, D.S."
            if INITIALS_TOKEN_RE.fullmatch(initials):
                merged_parts.append(f"{token}, {parts[index + 1]}")
                index += 2
                continue
        merged_parts.append(token)
        index += 1
    merged_parts = [piece for piece in merged_parts if not re.fullmatch(r"(?:19|20)\d{2}[a-z]?", piece)]
    return merged_parts


def _extract_title_from_first_segment(first_segment: str) -> str:
    text = _strip_entry_prefix(first_segment).strip(" .;")
    if not text:
        return ""

    # Pattern: "Organization, 2024"
    org_match = re.match(r"^(?P<title>.+?),\s*(?:19|20)\d{2}[a-z]?$", text)
    if org_match:
        candidate = org_match.group("title").strip(" .;")
        if candidate:
            return candidate

    # If it does not look like an author list, treat first segment as title.
    if not re.search(r",[ ](?:[A-Z]\.|[A-Z][a-z]{1,20})", text[:120]):
        if not text.lower().startswith(TITLE_STOP_PREFIXES):
            return text

    return ""


def _parse_title(segments: list[str], raw_entry: str) -> tuple[str, int | None]:
    if not segments:
        return "", None

    colon_title = _extract_title_from_colon_style(raw_entry)
    if colon_title:
        return colon_title, _find_title_segment_index(segments, colon_title)

    apa_title = _extract_title_from_apa_style(raw_entry)
    if apa_title:
        return apa_title, _find_title_segment_index(segments, apa_title)

    harvard_title = _extract_title_from_harvard_style(raw_entry)
    if harvard_title:
        return harvard_title, _find_title_segment_index(segments, harvard_title)

    # Prefer explicit quoted title if present.
    if raw_entry:
        quoted = TITLE_QUOTE_RE.search(raw_entry)
        if quoted:
            title = quoted.group(1).strip()
            return title, _find_title_segment_index(segments, title)

    # Usually segment[0] = authors, segment[1] = title.
    for idx in range(1, len(segments)):
        candidate = segments[idx].strip(" .;")
        lowered = candidate.lower()
        if not candidate:
            continue
        if lowered.startswith(TITLE_STOP_PREFIXES):
            continue
        if len(candidate) < 6:
            continue
        return candidate, idx

    if segments:
        first_segment_title = _extract_title_from_first_segment(segments[0])
        if first_segment_title:
            return first_segment_title, 0

    return "", None


def _parse_venue(segments: list[str], title: str, title_index: int | None) -> str:
    if not segments:
        return ""

    start = 1
    if title_index is not None:
        start = title_index + 1
    if start >= len(segments):
        return ""

    title_norm = _normalize_match_text(title)
    venue_parts: list[str] = []
    for seg in segments[start:]:
        seg_norm = _normalize_match_text(seg)
        if title_norm and seg_norm and (title_norm == seg_norm or title_norm in seg_norm):
            continue
        lowered = seg.lower().strip()
        if not lowered:
            continue
        if lowered.startswith(("doi:", "doi ", "url:", "url ", "http://", "https://")):
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
    style_hint = _detect_reference_style(normalized)

    authors = _parse_authors_from_segment(segments[0]) if segments else []
    title, title_index = _parse_title(segments, normalized)
    venue = _parse_venue(segments, title, title_index)
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
            "style_hint": style_hint,
        },
    )


def parse_reference_entries(entries: list[str]) -> list[CitationRecord]:
    records = []
    for idx, entry in enumerate(entries, start=1):
        records.append(parse_reference_entry(entry, f"pdf-ref:{idx}"))
    return records
