from __future__ import annotations

import json
import os
import re
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any, Mapping

from packages.core.models import CitationRecord
from packages.core.normalize import extract_identifier

YEAR_RE = re.compile(r"\b(?:19|20)\d{2}[a-z]?\b")
TITLE_QUOTE_RE = re.compile(r"[\"“](.+?)[\"”]")
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)

# Identifier validation. Rejects truncated arxiv IDs ("2511" without
# .NNNNN suffix, e.g. when BibTeX shipped an incomplete `arXiv–2511.`)
# and partial DOIs that would otherwise trigger spurious H5 / DOI mismatch
# verdicts downstream.
_ARXIV_PREFIX_RE = re.compile(r"^\s*arxiv\s*[:–—\-]\s*", re.IGNORECASE)
_ARXIV_ID_RE = re.compile(
    r"^(?:\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})$"
)
_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")
TITLE_STOP_PREFIXES = (
    "in proceedings",
    "in international conference",
    "in advances in",
    "findings of",
    "proceedings of",
    "doi:",
    "doi ",
    "url:",
    "url ",
    "http://",
    "https://",
    "//",
    "issn ",
)
# Matches a leading entry/reference marker that should be stripped before parsing.
# Handles:
#   [12]            - bracketed numeric (Vancouver/IEEE)
#   (12)            - parenthesized numeric
#   12.             - numeric + period (numbered list)
#   374             - bare PDF line number (1-4 digits, no period)
#                      Common in OCR'd journal pages with line numbering.
ENTRY_PREFIX_RE = re.compile(
    r"^\s*(?:\[\d{1,4}\]|\(\d{1,4}\)|\d{1,4}\.|\d{1,4}(?=\s+[A-Z]))\s*"
)
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
    r"\s+(?:In:|in:|Proceedings of|Lecture Notes in Computer Science|LNCS|Springer|CoRR|"
    r"IEEE Transactions|Nature|Journal|arXiv preprint|arXiv|doi:|doi |URL|https?://)",
    re.IGNORECASE,
)
VENUE_CUE_RE = re.compile(
    r"\b("
    r"proceedings|conference|workshop|symposium|journal|transactions|letters|review|annals|"
    r"icml|iclr|neurips|nips|cvpr|iccv|eccv|acl|naacl|emnlp|aaai|ijcai|kdd|uai|aistats|"
    r"pmlr|springer|wiley|elsevier|acm|ieee|auai press|mit press|association for computational linguistics"
    r")\b",
    re.IGNORECASE,
)
VENUE_NAME_HINT_RE = re.compile(
    r"\b("
    r"advances in neural information processing systems|neural information processing systems|"
    r"international conference on machine learning|icml|iclr|aaai|ijcai|"
    r"conference on empirical methods in natural language processing|emnlp|"
    r"association for computational linguistics|proceedings of machine learning research|pmlr"
    r")\b",
    re.IGNORECASE,
)
VENUE_HEAD_RE = re.compile(
    r"\b(?:"
    r"In\s+Proceedings(?:\s+of)?|Proceedings(?:\s+of)?|Findings(?:\s+of)?|"
    r"International Conference(?:\s+on)?|Conference(?:\s+on)?|Workshop(?:\s+on)?|Symposium(?:\s+on)?|"
    r"Journal(?:\s+of)?|IEEE Transactions(?:\s+on)?|Transactions(?:\s+on)?|"
    r"Advances in Neural Information Processing Systems|Neural Information Processing Systems|"
    r"Machine Learning Research|PMLR|ACL|EMNLP|NAACL|ICML|ICLR|CVPR|ICCV|ECCV|AAAI|IJCAI|KDD|UAI|AISTATS"
    r")\b",
    re.IGNORECASE,
)

_LLM_REPARSE_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}
_LLM_REPARSE_MODEL_CACHE_LOCK = threading.Lock()
_QWEN3_THINK_END_TOKEN_ID = 151668


def _build_bedrock_client(region: str, bearer_token: str | None = None):
    """Backwards-compatible: routes through packages.llm.client.build_chat_client
    so LLM_PROVIDER=openai/azure_openai swap to those backends without touching
    the converse() callsites further down."""
    from packages.llm.client import build_chat_client
    return build_chat_client(region=region, bearer_token=bearer_token)


def _extract_text_from_converse_response(response: dict[str, Any]) -> str:
    content = response.get("output", {}).get("message", {}).get("content", [])
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict) and "text" in item:
            parts.append(str(item["text"]))
    return "\n".join(parts).strip()


def _split_reference_segments(entry: str) -> list[str]:
    # Avoid splitting on initials like "T. J. M." while still splitting sentence boundaries.
    protected = re.sub(r"\b([A-Z])\.", r"\1§", entry)
    # Avoid splitting on version numbers like "Llada2. 0", "v2. 0", "GPT-3. 5"
    # — any digit followed by ". " followed by another digit is a version separator,
    # not a sentence boundary.
    protected = re.sub(r"(\d)\.(\s+\d)", r"\1§\2", protected)
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
    colon_pos = body.find(":")
    if colon_pos <= 0:
        return ""
    # Reject false positives where the first ":" is likely pagination, e.g. "32(10):6700-6713".
    prev_char = body[colon_pos - 1] if colon_pos - 1 >= 0 else ""
    near = body[max(0, colon_pos - 20) : min(len(body), colon_pos + 20)]
    if prev_char.isdigit() or prev_char == ")" or re.search(r"\d+\s*(?:\(\d+\))?\s*:\s*\d+", near):
        return ""
    match = COLON_STYLE_RE.match(body)
    if not match:
        return ""
    author_side = match.group("authors")
    title_side = match.group("title").strip(" .;")
    # True LNCS-style "Author, A.: Title" should not contain sentence boundary before ":".
    if re.search(r"\.\s+[A-Za-z]", author_side):
        return ""
    if re.search(r"\b(arxiv|doi|url)\b", author_side, flags=re.IGNORECASE):
        return ""
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
    author_side = match.group("authors").strip()
    if len(author_side) > 400:
        return ""
    # Reject over-matched harvard parse where year appears far after authors block.
    if re.search(
        r"\b(in|proceedings|conference|workshop|symposium|journal|volume|pp\.?|ieee|springer)\b",
        author_side,
        flags=re.IGNORECASE,
    ):
        return ""
    candidate = match.group("title").strip(" .;")
    if candidate and not candidate.lower().startswith(TITLE_STOP_PREFIXES):
        return candidate
    return ""


def _parse_authors_from_initial_style(author_block: str) -> list[str]:
    text = author_block.strip().strip(" .;")
    text = re.sub(r"\s+(?:and|&)\s+", ", ", text, flags=re.IGNORECASE)
    parts = [piece.strip(" ,") for piece in text.split(",") if piece.strip(" ,")]
    if not parts:
        return []

    parsed: list[str] = []
    idx = 0
    while idx < len(parts):
        token = parts[idx]
        if idx + 1 < len(parts):
            initials = re.sub(r"\s+", "", parts[idx + 1])
            if INITIALS_TOKEN_RE.fullmatch(initials):
                parsed.append(f"{token}, {parts[idx + 1]}")
                idx += 2
                continue
            # OCR may keep lowercase name particles with initials, e.g. "J. v."
            if re.fullmatch(r"[A-Z](?:\.)?(?:\s+[a-z](?:\.)?){1,2}", parts[idx + 1].strip()):
                parsed.append(f"{token}, {parts[idx + 1]}")
                idx += 2
                continue
        parsed.append(token)
        idx += 1
    return [item for item in parsed if item]


def _looks_like_author_list_left(text: str) -> bool:
    if "," not in text:
        return False
    lowered = text.lower()
    if any(marker in lowered for marker in ("http://", "https://", "doi", "arxiv", "proceedings", "conference", "journal", "pp.")):
        return False
    normalized = re.sub(r"\s+(?:and|&)\s+", ", ", text, flags=re.IGNORECASE)
    parts = [piece.strip(" ,") for piece in normalized.split(",") if piece.strip(" ,")]
    if len(parts) < 2:
        return False
    name_like = 0
    for part in parts:
        if not re.search(r"[A-Za-z]", part):
            continue
        words = [w for w in part.split() if w]
        if 1 <= len(words) <= 4:
            name_like += 1
    return name_like >= 2


def _looks_like_single_author_with_initials(text: str) -> bool:
    candidate = text.strip(" .;")
    return bool(re.match(r"^[A-Z][A-Za-z'`\-]+,\s*(?:[A-Z](?:\.|\b)\s*){1,4}$", candidate))


def _looks_like_publisher_segment(text: str) -> bool:
    cleaned = text.strip(" .;")
    lowered = cleaned.lower()
    if re.search(
        r"\b(press|publications|publisher|publishing|university press|verlag|society|association|institute|inc|ltd|llc|corporation|corp|photonics)\b",
        lowered,
    ):
        if re.search(r"\b(?:19|20)\d{2}[a-z]?\b", lowered):
            return True
    if re.match(
        r"^(?:[A-Z][A-Za-z'`\-]+|for|of|and|the|on|in|at|to|&|\-|\s)+,\s*(?:19|20)\d{2}[a-z]?$",
        cleaned,
    ):
        return True
    if re.match(r"^(?:[A-Z][A-Za-z'`\-]+(?:\s+[A-Z][A-Za-z'`\-]+){0,4}),\s*(?:19|20)\d{2}[a-z]?$", cleaned):
        return True
    return False


def _starts_with_venue_cue(text: str) -> bool:
    lowered = text.strip(" .;").lower()
    if not lowered:
        return False
    if lowered.startswith(("in ", "proceedings", "findings", "journal", "transactions", "ieee", "annals of")):
        return True
    if re.match(r"^in\s+(?:19|20)\d{2}\b", lowered):
        return True
    if re.match(r"^\d+\(\d+\)\s*:\s*\d", lowered):
        return True
    return False


def _split_initial_style_author_title(body: str) -> tuple[str, str]:
    candidates: list[tuple[int, int, str, str]] = []
    for boundary in re.finditer(r"\.\s+", body):
        left = body[: boundary.start()].strip(" .;")
        right = body[boundary.end() :].strip(" .;")
        if not left or not right:
            continue
        if not _looks_like_author_list_left(left):
            continue

        if re.match(r"^(?:and|&)\b", right, flags=re.IGNORECASE):
            continue
        if re.match(r"^[A-Z]\.,\s*", right):
            continue
        if re.match(r"^[A-Z]\.\s+", right):
            continue

        authors_candidate = _parse_authors_from_initial_style(left)
        if len(authors_candidate) < 1:
            continue
        if len(authors_candidate) == 1 and not _looks_like_single_author_with_initials(left):
            continue

        title_probe = re.split(
            r"\.\s+(?=(?:URL\b|doi\b|https?://|arXiv\b|arxiv\b))",
            right,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip(" .;")
        if not title_probe:
            continue

        score = len(authors_candidate) * 3
        if len(authors_candidate) >= 3:
            score += 4
        if len(authors_candidate) == 1 and _looks_like_single_author_with_initials(left):
            score += 2
        if _authors_look_contaminated(authors_candidate):
            score -= 12
        if _starts_with_venue_cue(title_probe):
            score -= 8
        if re.match(r"^[A-Z][A-Za-z'`\-]+,\s*(?:[A-Z](?:\.|\b))(?:\s*[A-Z](?:\.|\b))*", title_probe):
            score -= 6
        if title_probe.count(",") >= 12:
            score -= 4
        if len(title_probe.split()) <= 2:
            score -= 3
        if title_probe.lower().startswith(("url ", "http://", "https://", "doi:", "doi ", "arxiv")):
            score -= 5

        candidates.append((score, boundary.start(), left, right))

    if not candidates:
        return "", ""

    # Pick the highest score; if tied, prefer the later boundary in the sentence.
    best = max(candidates, key=lambda item: (item[0], item[1]))
    if best[0] < 4:
        return "", ""
    return best[2], best[3]


def _title_looks_like_author_overcapture(title: str) -> bool:
    text = (title or "").strip()
    if not text:
        return False
    if text.count(",") >= 10 and re.search(r"\band\b", text, flags=re.IGNORECASE):
        return True
    if text.count(",") >= 8 and re.search(r"[A-Z][A-Za-z'`\-]+,\s*[A-Z](?:\.|\b)", text):
        return True
    if text.count(",") >= 5 and re.search(r"\band\b", text, flags=re.IGNORECASE) and not _looks_like_venue_text(text):
        return True
    return False


def _extract_authors_and_title_from_initial_style(raw_entry: str) -> tuple[list[str], str]:
    body = _strip_entry_prefix(raw_entry)
    author_block, title = _split_initial_style_author_title(body)
    if not author_block or not title:
        return [], ""

    if not title or len(title) < 6:
        return [], ""
    if title.lower().startswith(TITLE_STOP_PREFIXES):
        return [], ""
    title = re.split(
        r"\.\s+(?=(?:In\b|Proceedings\b|Findings\b|CoRR\b|arXiv\b|IEEE\b|Nature\b|Journal\b|Math\b|Association\b|URL\b|doi\b))",
        title,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" .;")
    title = TITLE_CUT_RE.split(title, maxsplit=1)[0].strip(" .;")
    if not title or len(title) < 6:
        return [], ""

    authors = _parse_authors_from_initial_style(author_block)
    return authors, title


def _authors_look_contaminated(authors: list[str]) -> bool:
    if not authors:
        return True
    for author in authors:
        text = author.strip()
        lowered = text.lower()
        if len(text.split()) >= 6:
            return True
        if ". " in text and len(text.split()) >= 4:
            return True
        if re.match(r"^[A-Z]\.[ ]+[A-Za-z]", text):
            return True
        if re.search(r"\d", text) and len(text.split()) >= 2:
            return True
        if any(marker in lowered for marker in ("http://", "https://", "arxiv", "proceedings", "conference", "journal", "volume", "pages")):
            return True
    return False


def _title_looks_bad(title: str) -> bool:
    text = (title or "").strip()
    if not text:
        return True
    lowered = text.lower()
    if lowered.startswith(TITLE_STOP_PREFIXES):
        return True
    if text.startswith("//"):
        return True
    if re.fullmatch(r"\d+(?:[-–]\d+)?", text):
        return True
    if re.match(r"^\d{1,6}\s*[-–]\s*\d{1,6}\b", text):
        return True
    if "issn" in lowered:
        return True
    if re.search(r"\barxiv\s*:\s*\d{4}\.\d{4,5}\b", lowered):
        return True
    if lowered.startswith(("corr,", "arxiv:", "arxiv ")):
        return True
    if "abs/" in lowered and re.search(r"\b\d{4}\.\d{4,5}\b", lowered):
        return True
    if re.search(r"\bpp\.\s*\d", lowered):
        return True
    if re.search(r"\bassociation for computational linguistics\b", lowered):
        return True
    if re.search(r"\bassociation for computing machinery\b", lowered):
        return True
    if lowered.startswith(("in proceedings", "proceedings of", "findings of", "in international conference")):
        return True
    if re.match(r"^in\s+(?:19|20)\d{2}\b", lowered):
        return True
    if lowered.startswith(("program.,", "journal", "ieee transactions", "transactions on")):
        return True
    if re.search(r"\b\d+\(\d+\)\s*:\s*\d{2,6}[-–]\d{2,6}\b", lowered):
        return True
    if re.search(r",\s*\d{1,4}\s*:\s*\d{1,6}(?:[-–]\d{1,6})?,\s*(?:19|20)\d{2}[a-z]?\b", lowered):
        return True
    if _looks_like_publisher_segment(text):
        return True
    if _title_looks_like_author_overcapture(text):
        return True
    return False


def _looks_like_venue_segment_for_title(candidate: str) -> bool:
    text = candidate.strip(" .;")
    if not text:
        return False
    lowered = text.lower()
    if lowered.startswith(("in ", "proceedings", "findings", "journal", "transactions", "ieee", "annals of")):
        return True
    if re.match(r"^in\s+(?:19|20)\d{2}\b", lowered):
        return True
    if re.search(r"\bpp\.?\b", lowered):
        return True
    if _looks_like_publisher_segment(text):
        return True
    if re.search(r"\b\d+\(\d+\)\s*:\s*\d{1,6}(?:[-–]\d{1,6})?\b", lowered):
        return True
    if re.search(r"\b\d+\(\d+\)\s*:\s*e\d{2,8}(?:[-–]e?\d{2,8})?\b", lowered):
        return True
    if re.search(r"\b\d{1,4}\s*:\s*e\d{2,8}\b", lowered):
        return True
    if re.search(r",\s*\d{1,4}\s*:\s*\d{1,6}(?:[-–]\d{1,6})?,\s*(?:19|20)\d{2}[a-z]?\b", lowered):
        return True
    if _looks_like_venue_text(text) and re.search(r"\b(?:19|20)\d{2}[a-z]?\b", lowered):
        return True
    return False


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
            if re.fullmatch(r"[A-Z](?:\.)?(?:\s+[a-z](?:\.)?){1,2}", parts[index + 1].strip()):
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
        if candidate and not candidate.lower().startswith(TITLE_STOP_PREFIXES):
            return candidate

    # If it does not look like an author list, treat first segment as title.
    if text.count(",") >= 3 and re.search(r"\s(?:and|&)\s", text, flags=re.IGNORECASE):
        return ""
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
        if _looks_like_venue_segment_for_title(candidate):
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
        if lowered.startswith(("doi:", "doi ", "url:", "url ", "http://", "https://", "issn ")):
            break
        venue_parts.append(seg.strip(" .;"))

    venue = ". ".join(part for part in venue_parts if part).strip(" .;")
    venue = re.sub(r"\s+ISSN\s+\S.*$", "", venue, flags=re.IGNORECASE).strip(" .;")
    if venue:
        return venue

    # Fallback: when title itself starts with venue cue (e.g., "In International Conference ..."),
    # treat it as venue-only fragment instead of a true title.
    if title and title.lower().startswith(("in proceedings", "in international conference", "proceedings of", "findings of")):
        return title.strip(" .;")
    return ""


def _strip_trailing_metadata(text: str) -> str:
    cleaned = text.strip(" .;")
    for _ in range(4):
        before = cleaned
        cleaned = re.sub(r"\bdoi\s*:?\s*\S+$", "", cleaned, flags=re.IGNORECASE).strip(" .;")
        cleaned = re.sub(r"\burl\s*:?\s*\S+$", "", cleaned, flags=re.IGNORECASE).strip(" .;")
        cleaned = re.sub(r"https?://\S+$", "", cleaned, flags=re.IGNORECASE).strip(" .;")
        if not re.match(
            r"^\s*arxiv(?:\s+preprint)?\s*:?[\sA-Za-z0-9\.\[\],:\-]*$",
            cleaned,
            flags=re.IGNORECASE,
        ):
            cleaned = re.sub(
                r"\barxiv\s*:?\s*\d{4}\.\d{4,5}(?:v\d+)?(?:\s+\[[^\]]+\])?$",
                "",
                cleaned,
                flags=re.IGNORECASE,
            ).strip(" .;")
        if cleaned == before:
            break
    return cleaned


def _looks_like_venue_text(text: str) -> bool:
    cleaned = _strip_trailing_metadata(text)
    if not cleaned:
        return False
    lowered = cleaned.lower()
    if lowered.startswith(("in proceedings", "proceedings of", "in international conference", "findings of")):
        return True
    if lowered.startswith(("arxiv preprint", "arxiv:", "corr,")):
        return True
    if VENUE_CUE_RE.search(cleaned):
        return True
    if re.search(r"\barxiv\b", cleaned, flags=re.IGNORECASE) and re.search(r"\b(?:19|20)\d{2}[a-z]?\b", cleaned):
        return True
    if re.search(r"\b\d+\(\d+\)\s*:\s*\d{1,6}(?:[-–]\d{1,6})?\b", cleaned):
        return True
    if re.search(r"\b\d+\(\d+\)\s*:\s*e\d{2,8}(?:[-–]e?\d{2,8})?\b", cleaned, flags=re.IGNORECASE):
        return True
    if re.search(r"\b\d{1,4}\s*:\s*e\d{2,8}\b", cleaned, flags=re.IGNORECASE):
        return True
    if re.search(r"\bvol(?:ume)?\.?\s*\d+\b", cleaned, flags=re.IGNORECASE):
        return True
    if re.search(r"\bpp\.?\s*\d", cleaned, flags=re.IGNORECASE):
        return True
    if re.search(r"\bissn\b", cleaned, flags=re.IGNORECASE):
        return True
    if _looks_like_publisher_segment(cleaned):
        return True
    if VENUE_NAME_HINT_RE.search(cleaned):
        return True
    if re.search(r",\s*\d{1,4}\s*:\s*\d{1,6}(?:[-–]\d{1,6})?,\s*(?:19|20)\d{2}[a-z]?\b", cleaned):
        return True
    return False


def _split_title_and_venue(title: str) -> tuple[str, str]:
    cleaned = _strip_trailing_metadata(title)
    parts = [part.strip(" .;") for part in re.split(r"\.\s+", cleaned) if part.strip(" .;")]
    if len(parts) < 2:
        return "", ""

    for idx in range(1, len(parts)):
        head = ". ".join(parts[:idx]).strip(" .;")
        tail = ". ".join(parts[idx:]).strip(" .;")
        if not head or not tail:
            continue
        if _looks_like_venue_text(head):
            continue
        if _looks_like_venue_text(tail):
            return head, tail
    return "", ""


def _split_title_and_venue_by_cue(title: str) -> tuple[str, str]:
    cleaned = _strip_trailing_metadata(title)
    if not cleaned:
        return "", ""
    match = VENUE_HEAD_RE.search(cleaned)
    if not match:
        return "", ""
    head = cleaned[: match.start()].strip(" .;,:")
    tail = cleaned[match.start() :].strip(" .;")
    if not head or not tail:
        return "", ""
    if len(head) < 12:
        return "", ""
    if not _looks_like_venue_text(tail):
        return "", ""
    return head, tail


def _extract_venue_from_raw_text(raw_text: str, title: str) -> str:
    title_norm = _normalize_match_text(title)
    candidates: list[str] = []
    for segment in _split_reference_segments(raw_text):
        seg = _strip_trailing_metadata(segment)
        if not seg:
            continue
        lowered = seg.lower()
        if lowered.startswith(("doi:", "doi ", "url:", "url ", "http://", "https://")):
            continue
        seg_norm = _normalize_match_text(seg)
        if title_norm and (seg_norm == title_norm or title_norm in seg_norm):
            split_title, split_venue = _split_title_and_venue_by_cue(seg)
            if split_title and split_venue:
                candidates.append(split_venue.strip(" .;"))
            continue
        if _looks_like_venue_text(seg):
            candidates.append(seg.strip(" .;"))
    if not candidates:
        return ""
    return candidates[0]


def _backfill_venue_from_title_and_raw(title: str, raw_text: str) -> tuple[str, str, str]:
    title_clean = title.strip(" .;")
    split_title, split_venue = _split_title_and_venue(title_clean)
    if split_title and split_venue:
        return split_title, split_venue, "title_split"
    split_title, split_venue = _split_title_and_venue_by_cue(title_clean)
    if split_title and split_venue:
        return split_title, split_venue, "title_cue_split"
    if title_clean and _looks_like_venue_text(title_clean):
        return "", _strip_trailing_metadata(title_clean), "title_as_venue"
    venue_from_raw = _extract_venue_from_raw_text(raw_text, title_clean)
    if venue_from_raw:
        return title_clean, venue_from_raw, "raw_text"
    return title_clean, "", ""


# Page-range regexes, in priority order.
#  - With explicit "pages"/"pp." prefix.
#  - "X(Y), AAA-BBB" — IEEE/ACM journal shorthand: vol(issue), page-range.
#  - "X:AAA-BBB" — JMLR-style "20:1-67".
#  - Bare "AAA-BBB" near the end of the string (last segment).
_PAGES_PREFIXED_RE = re.compile(
    r",?\s*(?:pages?|pp\.?)\s+([\d]+\s*[-–—]+\s*[\d]+)",
    re.IGNORECASE,
)
_PAGES_VOL_ISSUE_RE = re.compile(
    r"\b\d+\s*\(\s*\d+\s*\)\s*[,:]\s*([\d]+\s*[-–—]+\s*[\d]+)"
)
_PAGES_COLON_RE = re.compile(
    r"\b\d{1,4}\s*:\s*([\d]+\s*[-–—]+\s*[\d]+)"
)
_PAGES_BARE_RE = re.compile(
    r"(?:^|[,;.\s])([\d]{1,5}\s*[-–—]+\s*[\d]{1,5})\s*[.,]?\s*$"
)
# Volume regexes, in priority order.
#  - "volume X" / "vol. X" / "Vol X".
#  - "X(Y)" with vol(issue) shorthand — capture X.
#  - JMLR-style "X:" before a page range — capture X.
_VOLUME_EXPLICIT_RE = re.compile(
    r"\b(?:volume|vol\.?)\s*(\d+)\b",
    re.IGNORECASE,
)
_VOLUME_PAREN_RE = re.compile(r"\b(\d+)\s*\(\s*\d+\s*\)\s*[,:]\s*\d+\s*[-–—]+\s*\d+")
_VOLUME_COLON_RE = re.compile(r"\b(\d{1,4})\s*:\s*\d+\s*[-–—]+\s*\d+")

_PUBLISHER_PATTERNS = [
    "Association for Computational Linguistics",
    "Springer-Verlag",
    "Springer Nature",
    "Springer International Publishing",
    "Springer",
    "IEEE Computer Society",
    "IEEE Press",
    "IEEE",
    "ACM Press",
    "ACM",
    "PMLR",
    "Elsevier",
    "MIT Press",
    "Morgan Kaufmann",
    "AAAI Press",
    "Cambridge University Press",
    "Oxford University Press",
    "Wiley-Blackwell",
    "Wiley",
    "World Scientific",
    "IOS Press",
    "Nature Publishing Group",
    "Taylor & Francis",
    "SAGE Publications",
    "OpenReview.net",
    "Curran Associates",
    "JMLR.org",
    "Association for the Advancement of Artificial Intelligence (AAAI)",
    "Association for the Advancement of Artificial Intelligence",
]
# Wildcard patterns — matched after the literal list to catch
# "X University Press", "X Publishing", "X Publishers", etc.
_PUBLISHER_WILDCARD_RES = [
    re.compile(r"\b([A-Z][A-Za-z&]*(?:\s+[A-Z][A-Za-z&]*){0,3}\s+University\s+Press)\b"),
    re.compile(r"\b([A-Z][A-Za-z&]*(?:\s+[A-Z][A-Za-z&]*){0,3}\s+Publishing(?:\s+Co\.?)?)\b"),
    re.compile(r"\b([A-Z][A-Za-z&]*(?:\s+[A-Z][A-Za-z&]*){0,3}\s+Publishers?)\b"),
]
# Country names are typically capitalized words ("United Kingdom", "France",
# "Iran") or 2–5 letter all-caps acronyms ("USA", "UK", "PRC"). The word form
# requires ≥3 lowercase chars after the leading capital so corporate suffixes
# like "Inc" / "Ltd" / "Co" don't get mis-matched as country names by
# constructions like "Dover Publications, Inc, 1956".
_COUNTRY = r"(?:[A-Z][a-z]{3,}(?:\s+[A-Z][a-z]+)*|[A-Z]{2,5})"
_CITY_COUNTRY = r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*" + _COUNTRY
_CITY_REGION_COUNTRY = (
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*"   # City
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*"   # Region/State word form
    + _COUNTRY                                # Country (word or acronym)
)
_CITY_STATE = r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2}"
# Anchor on `,` OR `.` OR start-of-string so we can still match a location
# even after the publisher splice removed the leading comma.
_LOCATION_RE = re.compile(
    r"(?:^|[,.])\s*("
    r"(?:(?:Online|Virtual)\s+and\s+" + _CITY_REGION_COUNTRY + r")"  # Online and City, Region, Country
    r"|(?:(?:Online|Virtual)\s+and\s+" + _CITY_COUNTRY + r")"        # Online and City, Country
    r"|(?:" + _CITY_REGION_COUNTRY + r")"                            # City, Region, Country
    r"|(?:Online|Virtual)"                                            # Online / Virtual
    r"|(?:" + _CITY_COUNTRY + r")"                                    # City, Country
    r"|(?:" + _CITY_STATE + r")"                                      # City, STATE
    r")\s*[.,]?"
)


def _normalize_pages(pages: str) -> str:
    """Collapse internal whitespace; convert en/em dashes to hyphen."""
    s = re.sub(r"\s+", "", pages or "")
    s = re.sub(r"[-–—]+", "-", s)
    return s


def _find_pages(text: str) -> tuple[str, int, int] | None:
    """Try the three pages patterns in priority order; return (pages, start, end)."""
    if not text:
        return None
    for rx in (_PAGES_PREFIXED_RE, _PAGES_VOL_ISSUE_RE, _PAGES_COLON_RE, _PAGES_BARE_RE):
        m = rx.search(text)
        if m:
            return _normalize_pages(m.group(1)), m.start(), m.end()
    return None


def _find_volume(text: str) -> str:
    if not text:
        return ""
    for rx in (_VOLUME_EXPLICIT_RE, _VOLUME_PAREN_RE, _VOLUME_COLON_RE):
        m = rx.search(text)
        if m:
            return m.group(1).strip()
    return ""


def _find_publisher(text: str) -> tuple[str, int, int]:
    """Find a publisher name in `text`. Literal patterns first (longest first),
    then wildcard regexes for "X University Press" / "X Publishing".

    For each pattern we scan ALL occurrences and pick the first one that is
    standalone (preceded by ", " / ". " / ";"), so a venue that EMBEDS a
    publisher name (e.g. "ACL Findings ... Association for Computational
    Linguistics") still finds the trailing standalone publisher mention.

    Returns (publisher, start, end). Empty publisher means none found.
    The (start, end) span is the literal occurrence in `text` so the caller
    can strip the EXACT standalone occurrence (not the embedded one)."""
    if not text:
        return "", -1, -1
    for pub in _PUBLISHER_PATTERNS:
        start = 0
        while True:
            idx = text.find(pub, start)
            if idx < 0:
                break
            start = idx + 1
            if idx == 0:
                continue
            before_char = text[idx - 1] if idx > 0 else ""
            before_2 = text[max(0, idx - 2):idx]
            is_standalone = before_2 in (". ", ", ") or before_char in (".", ",", ";")
            if not is_standalone:
                continue
            return pub, idx, idx + len(pub)
    for rx in _PUBLISHER_WILDCARD_RES:
        m = rx.search(text)
        if m:
            return m.group(1), m.start(1), m.end(1)
    return "", -1, -1


def _split_venue_fields(venue: str) -> tuple[str, str, str, str, str]:
    """Split a raw venue string into (venue, pages, publisher, location, volume)."""
    if not venue:
        return "", "", "", "", ""

    remaining = venue
    pages = ""
    volume = ""
    publisher = ""
    location = ""

    # Extract volume FIRST. The "X(Y), AAA-BBB" and "X:AAA-BBB" shorthands
    # encode both volume and pages — if pages strips them first, volume is
    # gone. _find_volume only reads the patterns; it never mutates `remaining`.
    volume = _find_volume(remaining)

    # Extract pages
    pgr = _find_pages(remaining)
    if pgr:
        pages, ps, pe = pgr
        remaining = remaining[:ps] + remaining[pe:]

    # Strip "volume X" / "vol. X" tokens (parenthetical / colon shorthands
    # were already removed when we stripped pages).
    if volume:
        remaining = _VOLUME_EXPLICIT_RE.sub("", remaining, count=1)

    # Extract publisher (known patterns) — only if it appears as a standalone
    # trailing/separated part, not embedded in the venue name itself. We use
    # the position returned by _find_publisher (NOT a fresh str.find) so we
    # strip the standalone occurrence, not an embedded one. Note: we keep the
    # leading comma in the `after` slice so a trailing "Publisher, City, State,
    # Country" still has the comma anchor that _LOCATION_RE needs.
    publisher, ps, pe = _find_publisher(remaining)
    if publisher and ps >= 0:
        before = remaining[:ps].rstrip(", .")
        after = remaining[pe:].lstrip(" .")  # keep leading "," for location anchor
        remaining = (before + " " + after).strip() if after else before

    # Extract location
    loc_match = _LOCATION_RE.search(remaining)
    if loc_match:
        location = loc_match.group(1).strip(" .,")
        remaining = remaining[:loc_match.start()] + remaining[loc_match.end():]

    # Clean up remaining venue: collapse whitespace, then dedupe successive
    # commas / semicolons that the various strip steps may have left behind
    # (e.g. "Ann. Stat, , 2016" after volume + pages were pulled out).
    remaining = re.sub(r"\s+", " ", remaining)
    remaining = re.sub(r"(?:[,;]\s*){2,}", ", ", remaining)
    remaining = re.sub(r"\s*,\s*$", "", remaining)
    remaining = remaining.strip(" ,;.")

    return remaining, pages, publisher, location, volume


def _backfill_meta_fields_from_raw(
    raw_text: str,
    title: str,
    venue: str,
    pages: str,
    publisher: str,
    location: str,
    volume: str,
) -> tuple[str, str, str, str]:
    """For pages/volume/publisher/location not yet found in venue, scan the
    rest of raw_text (everything after the title and venue). Citations like
    "Authors. 2024. Title. Venue. pp. 12-34. Springer." put metadata in
    trailing sentences that the venue string never captured."""
    if not raw_text:
        return pages, publisher, location, volume

    # Build the search corpus = raw_text minus the title and venue substrings,
    # so we don't accidentally re-extract things that were already part of
    # title/venue text.
    corpus = raw_text
    for blob in (title, venue):
        if blob and blob in corpus:
            corpus = corpus.replace(blob, " ", 1)

    if not pages:
        pgr = _find_pages(corpus)
        if pgr:
            pages = pgr[0]
    if not volume:
        volume = _find_volume(corpus)
    if not publisher:
        pub_name, _, _ = _find_publisher(corpus)
        if pub_name:
            publisher = pub_name
    # Note: we deliberately do NOT backfill `location` from raw_text. The
    # _LOCATION_RE pattern (City, Country) collides with author-list commas
    # (e.g. "Author One, Author Two") and produces false positives. Location
    # extraction stays scoped to the parsed venue string.

    return pages, publisher, location, volume


def _extract_url(raw_text: str) -> str:
    match = URL_RE.search(raw_text)
    if not match:
        return ""
    return match.group(0).rstrip(".,);]")


_PARENS_YEAR_RE = re.compile(r"\((19|20)\d{2}\)")


def _extract_year(raw_text: str, paper_style: str = "") -> int | None:
    # Parens-year preference: APA / numbered styles always parenthesize the
    # publication year ("Foo (2024). Title..."), so when a parenthesized year
    # is present, prefer it over title-embedded numerals like "Healthy China
    # 2030". This must run on raw_text BEFORE the URL/DOI scrubbing so the
    # original parens are preserved.
    parens_match = _PARENS_YEAR_RE.search(raw_text or "")
    if parens_match:
        try:
            y = int(parens_match.group(0)[1:5])
        except (TypeError, ValueError):
            y = None
        if y is not None and 1800 <= y <= 2035:
            return y

    cleaned = raw_text
    cleaned = re.sub(r"https?://\S+", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bdoi\s*:\s*\S+", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"10\.\d{4,9}/\S+", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"arxiv\s*:\s*\d{4}\.\d{4,5}(?:v\d+)?", " ", cleaned, flags=re.IGNORECASE)
    # Page-range scrub: "1964-1978" / "20-25" page or date spans contain
    # year-shaped digits that YEAR_RE would otherwise capture as the trailing
    # publication year. Wipe these spans before scanning for years.
    cleaned = re.sub(r"\b\d{1,5}\s*[-–—]+\s*\d{1,5}\b", " ", cleaned)

    candidates: list[int] = []
    for match in YEAR_RE.finditer(cleaned):
        token = match.group(0)
        start, end = match.start(), match.end()
        # Ignore arXiv-like numeric IDs such as 1904.10509.
        if end + 1 < len(cleaned) and cleaned[end] == "." and cleaned[end + 1].isdigit():
            continue
        if start - 1 >= 0 and cleaned[start - 1].isdigit():
            continue
        try:
            year = int(token[:4])
        except (TypeError, ValueError):
            continue
        if 1800 <= year <= 2035:
            candidates.append(year)
    if not candidates:
        return None
    # CCS / ACM Reference Format: "Authors. YEAR. Title. Venue, vol, pages."
    # The year is the FIRST 4-digit number (right after authors), and any
    # later digit clusters (volume, page range) can confuse a "trailing year"
    # heuristic into picking e.g. 1954 from "pp. 1954-1990". When the style
    # hint says ccs/acm, prefer the leading year.
    style_key = (paper_style or "").strip().lower()
    if style_key in {"ccs", "acm"}:
        return candidates[0]
    # Default: prefer the trailing publication year over earlier numbers
    # (e.g., arXiv IDs that escaped the prefix strip, or page spans).
    return candidates[-1]


def _citation_has_missing_core_fields(record: CitationRecord) -> bool:
    return (
        not record.title.strip()
        or not record.authors
        or not record.venue.strip()
        or record.year is None
    )


def _extract_llm_json_payload(text: str) -> Mapping[str, Any] | None:
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, Mapping) else None


def _normalize_llm_reparse_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    title = str(payload.get("title", "") or "").strip(" .;")
    venue = str(payload.get("venue", "") or "").strip(" .;")
    doi = str(payload.get("doi", "") or "").strip().rstrip(".,;")
    arxiv_id = str(payload.get("arxiv_id", "") or "").strip(" .;")
    url = str(payload.get("url", "") or "").strip().rstrip(".,);]")

    authors_value = payload.get("authors", [])
    authors: list[str] = []
    if isinstance(authors_value, list):
        authors = [str(item).strip(" .;") for item in authors_value if str(item).strip(" .;")]
    elif isinstance(authors_value, str):
        text = authors_value.strip()
        if text:
            authors = [item.strip(" .;") for item in re.split(r"\s*;\s*", text) if item.strip(" .;")]

    year_value = payload.get("year")
    year: int | None = None
    if isinstance(year_value, int):
        year = year_value
    elif isinstance(year_value, str):
        match = re.search(r"(?:19|20)\d{2}", year_value)
        if match:
            year = int(match.group(0))
    if year is not None and not (1800 <= year <= 2035):
        year = None

    volume = str(payload.get("volume", "") or "").strip(" .;")
    pages = str(payload.get("pages", "") or "").strip(" .;")
    publisher = str(payload.get("publisher", "") or "").strip(" .;")
    location = str(payload.get("location", "") or "").strip(" .;")

    return {
        "title": title,
        "authors": authors,
        "venue": venue,
        "year": year,
        "volume": volume,
        "pages": pages,
        "publisher": publisher,
        "location": location,
        "doi": doi,
        "arxiv_id": arxiv_id,
        "url": url,
    }


def _load_llm_reparse_bundle(model_path: str) -> tuple[Any, Any]:
    path = (model_path or "").strip()
    if not path:
        raise RuntimeError("ocr_vlm_extract.local.model_path is empty.")

    with _LLM_REPARSE_MODEL_CACHE_LOCK:
        cached = _LLM_REPARSE_MODEL_CACHE.get(path)
    if cached is not None:
        return cached

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "LLM reparse requires transformers and torch. Install them in the current environment."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            trust_remote_code=True,
            torch_dtype="auto",
        )
        model = model.to("cpu")
    model.eval()

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bundle = (tokenizer, model)
    with _LLM_REPARSE_MODEL_CACHE_LOCK:
        _LLM_REPARSE_MODEL_CACHE[path] = bundle
    return bundle


def _run_llm_reparse_on_raw(
    raw_text: str,
    model_path: str,
    max_new_tokens: int,
    temperature: float,
) -> dict[str, Any] | None:
    tokenizer, model = _load_llm_reparse_bundle(model_path)

    system_prompt = (
        "You are a strict bibliography parser. "
        "Extract structured fields from a raw reference string. "
        "Return JSON only."
    )
    user_prompt = (
        "Reference:\n"
        f"{raw_text}\n\n"
        "Return only valid JSON with exact keys:\n"
        '{"title": "", "authors": [], "venue": "", "year": null, "volume": "", "pages": "", "publisher": "", "location": "", "doi": "", "arxiv_id": "", "url": ""}\n'
        "Rules:\n"
        "- authors must be an array of strings.\n"
        "- IMPORTANT: If the reference uses 'et al.', 'et al', 'others', or 'and others' "
        "to indicate that the author list was truncated, you MUST include that marker as "
        "the LAST item in the authors array verbatim (e.g., ['Tom Brown', 'Benjamin Mann', "
        "..., 'et al.']). Do NOT silently drop it — downstream code uses this marker to "
        "detect that the author list is intentionally partial.\n"
        "- year must be integer or null.\n"
        "- venue is ONLY the journal or conference name (e.g., 'NeurIPS', 'Nature', 'Proceedings of EMNLP 2021'). Do NOT include pages, volume, issue, location, or publisher in venue.\n"
        "- volume is the volume number (e.g., '104', '15'). Empty if not present.\n"
        "- pages is the page range (e.g., '1234-1245'). Recognize 'pp.' as pages. Empty if not present.\n"
        "- publisher is the publishing organization (e.g., 'Association for Computational Linguistics', 'Springer'). Empty if not present.\n"
        "- location is the conference location (e.g., 'Online', 'Seoul, Korea'). Empty if not present.\n"
        "- doi/arxiv_id/url should be plain strings (empty if unknown).\n"
        "- if uncertain, leave empty string / [] / null.\n"
        "- do not output explanations."
    )

    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for LLM reparse.") from exc

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    model_device = getattr(model, "device", None)
    if model_device is not None:
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max(32, int(max_new_tokens)),
        "do_sample": bool(temperature and temperature > 0.0),
    }
    if generate_kwargs["do_sample"]:
        generate_kwargs["temperature"] = float(temperature)
        generate_kwargs["top_p"] = 0.95
    if tokenizer.eos_token_id is not None:
        generate_kwargs["eos_token_id"] = tokenizer.eos_token_id
    if tokenizer.pad_token_id is not None:
        generate_kwargs["pad_token_id"] = tokenizer.pad_token_id

    with torch.no_grad():
        outputs = model.generate(**inputs, **generate_kwargs)

    generated = outputs[0][input_ids.shape[1] :]
    output_ids = generated.tolist()

    think_split_index = 0
    try:
        think_split_index = len(output_ids) - output_ids[::-1].index(_QWEN3_THINK_END_TOKEN_ID)
    except ValueError:
        think_split_index = 0

    if think_split_index > 0:
        thinking_content = tokenizer.decode(generated[:think_split_index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(generated[think_split_index:], skip_special_tokens=True).strip("\n")
    else:
        thinking_content = ""
        content = tokenizer.decode(generated, skip_special_tokens=True).strip("\n")

    # Prefer final answer content in Qwen think mode.
    payload = _extract_llm_json_payload(content)
    if payload is None and thinking_content:
        merged_text = f"{thinking_content}\n{content}".strip()
        payload = _extract_llm_json_payload(merged_text)
    if payload is None and not content:
        payload = _extract_llm_json_payload(tokenizer.decode(generated, skip_special_tokens=True).strip())
    if payload is None:
        return None
    return _normalize_llm_reparse_result(payload)


# Paper-style hints for the reparse LLM. Each block names the canonical
# bibliography format the venue uses, with a focus on whichever fields the
# format places in non-default slots (most importantly: where YEAR sits).
# When a citation gets reparsed against the right style hint, the LLM is
# much less likely to swap year with a page number or a volume — the
# pdf-ref:20 case ("Year 1954 in citation vs 2021 in candidate") came from
# a CCS-style reference where year follows authors but the heuristic
# parser grabbed a stray digit further down the entry.
_PAPER_STYLE_GUIDES = {
    "ccs": (
        "PAPER STYLE: ACM CCS / ACM Reference Format. Canonical layout per entry:\n"
        "  [N] Authors. <YEAR>. Title. Venue, volume, pages. doi/url.\n"
        "Year ALWAYS appears RIGHT AFTER the trailing period of the author list "
        "and BEFORE the title. It is a 4-digit number standing alone, not part "
        "of a page range or a volume. If you see two 4-digit numbers in the "
        "entry (e.g. one after authors, another inside 'pages 1954-1990'), the "
        "year is the one IMMEDIATELY after authors.\n"
    ),
    "acm": (
        "PAPER STYLE: ACM Reference Format. Canonical layout per entry:\n"
        "  [N] Authors. <YEAR>. Title. Venue, volume, pages. doi/url.\n"
        "Year ALWAYS appears right after authors, before the title.\n"
    ),
    "ieee": (
        "PAPER STYLE: IEEE. Canonical layout:\n"
        "  [N] Authors, \"Title,\" Venue, vol. V, no. M, pp. P-Q, <YEAR>.\n"
        "Year is at the END of the entry, after pages. Volume is preceded "
        "by 'vol.', issue by 'no.', pages by 'pp.'.\n"
    ),
    "lncs": (
        "PAPER STYLE: Springer LNCS. Canonical layout:\n"
        "  N. Authors: Title. In: Editors (eds.) Venue. LNCS, vol. V, pp. P-Q. "
        "Springer, Location (<YEAR>).\n"
        "Year sits in parentheses near the END of the entry. Volume is "
        "preceded by 'vol.'.\n"
    ),
    "iclr": (
        "PAPER STYLE: ICLR / arXiv author-year. Canonical layout:\n"
        "  Authors. Title. Venue, <YEAR>.\n"
        "Year appears at the end of the entry, after the venue.\n"
    ),
    "neurips": (
        "PAPER STYLE: NeurIPS author-year. Canonical layout:\n"
        "  Authors. Title. Venue, volume:pages, <YEAR>.\n"
        "Year appears at the end, after the page range.\n"
    ),
    "icml": (
        "PAPER STYLE: ICML author-year. Canonical layout similar to NeurIPS.\n"
        "Year sits at the end of the entry.\n"
    ),
    "plain": (
        "PAPER STYLE: plain (BibTeX plain). Canonical layout:\n"
        "  N. Authors. Title. Venue, volume(issue):pages, <YEAR>.\n"
        "Year is at the END of the entry, just before the closing period.\n"
    ),
}


def _build_paper_style_hint(paper_style: str) -> str:
    """Return a paper-style guidance block for the reparse prompt."""
    key = (paper_style or "").strip().lower()
    if not key:
        return ""
    guide = _PAPER_STYLE_GUIDES.get(key)
    if not guide:
        return ""
    return guide + "\n"


def _run_llm_reparse_on_raw_chat_client(
    raw_text: str,
    provider: str,
    model_id: str,
    region: str,
    bearer_token: str | None,
    max_new_tokens: int,
    temperature: float,
    reference_page_images: list[bytes] | None = None,
    paper_style: str = "",
) -> dict[str, Any] | None:
    """Run the cropped-block VLM reparse against any cloud provider that
    `packages.llm.client.build_chat_client` supports: ``bedrock``,
    ``openai``, or ``azure_openai``. All three accept the same multimodal
    message envelope thanks to the OpenAIChatShim image-block translation.
    The local-vLLM path is handled by ``_run_llm_reparse_on_raw`` below."""
    from packages.llm.client import build_chat_client
    client = build_chat_client(
        provider=provider,
        region=region,
        bearer_token=bearer_token,
    )

    n_images = len(reference_page_images) if reference_page_images else 0
    common_image_block = (
        "★★★ THE OCR TEXT IS NOISY AND HAS PROBABLY MIS-READ AT LEAST ONE "
        "AUTHOR NAME OR TITLE WORD. ★★★\n\n"
        "Mandatory verification procedure for AUTHORS and TITLE:\n"
        "  1. For each author in the OCR text, locate the same name in the image.\n"
        "  2. Read every character in the image left-to-right.\n"
        "  3. If the image shows even ONE different letter (extra letter, missing "
        "letter, wrong letter, swapped letters, dropped diacritic), you MUST "
        "output the spelling AS IT APPEARS IN THE IMAGE, not as it appears in "
        "the OCR text.\n"
        "  4. Apply the same procedure to TITLE words.\n\n"
        "Concrete OCR errors that must be fixed when seen:\n"
        "  - Extra letter:      OCR 'Schreck' (7 chars), image 'Schrek' (6 chars).\n"
        "  - Phantom letter:    OCR 'Polychroniadou' read as 'Golyprohniadou'.\n"
        "  - Missing letter:    OCR 'Polychroniadou' loses an internal 'i'.\n"
        "  - Adjacent swap:     OCR 'Thainyi' → image 'Tihanyi'; "
        "OCR 'Akocora' → image 'Akcora'; OCR 'Haignet' → image 'HiAgent'.\n"
        "  - Mis-capitalized acronyms: OCR 'Sw-bench' → image 'SWE-bench'; "
        "OCR 'Codeelo' → image 'CodeElo'.\n"
        "  - Dropped diacritics: OCR 'Bugueno' → image 'Bugueño'.\n\n"
        "Do NOT trust the OCR spelling just because it 'looks plausible'. "
        "The IMAGE is ground truth; the OCR text is a flawed hypothesis.\n\n"
        "ANTI-HALLUCINATION RULES (do not violate even once):\n"
        "  - SCOPE GUARD: If you cannot locate the OCR text's title or first "
        "author surname ANYWHERE in the image, this means the crop covers a "
        "DIFFERENT reference than the OCR text describes. In that case, output "
        "the OCR-derived values as best you can parse them — do NOT substitute "
        "fields from any other reference visible in the image, and do NOT "
        "invent a new reference from your training knowledge.\n"
        "  - If a field is not visible in the image AND not present in the OCR "
        "text, output empty string / [] / null. Do NOT fill it from your "
        "training knowledge.\n"
        "  - For arxiv_id and doi specifically: output ONLY what is printed on "
        "the page. If you see 'arXiv–2511.' (truncated, no number after dot), "
        "leave arxiv_id empty — do NOT guess the missing digits, do NOT invent "
        "a plausible-looking ID. Same for DOIs.\n\n"
    )
    image_instruction = ""
    if n_images == 1:
        image_instruction = (
            "IMPORTANT: An image is attached. It shows ONLY the cited "
            "reference (cropped from the original page), so you do not need "
            "to search for it — read the fields directly from the image.\n\n"
            + common_image_block
        )
    elif n_images >= 2:
        image_instruction = (
            f"IMPORTANT: {n_images} images are attached. The reference SPANS "
            "ACROSS COLUMNS OR PAGES — the start is in the first crop and "
            "continues in the next. Read fields directly from the crops.\n\n"
            + common_image_block
        )

    style_hint = _build_paper_style_hint(paper_style)

    prompt = (
        "You are a strict bibliography parser. "
        "Extract structured fields from a raw reference string.\n"
        + image_instruction
        + style_hint +
        "Return only valid JSON with exact keys:\n"
        '{"title": "", "authors": [], "venue": "", "year": null, "volume": "", "pages": "", "publisher": "", "location": "", "doi": "", "arxiv_id": "", "url": ""}\n'
        "Rules:\n"
        "- authors must be an array of strings.\n"
        "- IMPORTANT: If the reference uses 'et al.', 'et al', 'others', or 'and others' "
        "to indicate that the author list was truncated, you MUST include that marker as "
        "the LAST item in the authors array verbatim (e.g., ['Tom Brown', 'Benjamin Mann', "
        "..., 'et al.']). Do NOT silently drop it — downstream code uses this marker to "
        "detect that the author list is intentionally partial.\n"
        "- year must be integer or null.\n"
        "- venue is ONLY the journal or conference name (e.g., 'NeurIPS', 'Nature', 'Proceedings of the Indian Academy of Sciences'). Do NOT include volume, pages, issue, location, or publisher in venue.\n"
        "- volume is the volume number (e.g., '104', '15'). Empty if not present.\n"
        "- pages is the page range (e.g., '483-494', '1234-1245'). Recognize 'pp.' as pages. Empty if not present.\n"
        "- publisher is the publishing organization (e.g., 'Association for Computational Linguistics', 'Springer'). Empty if not present.\n"
        "- location is the conference location (e.g., 'Online', 'Seoul, Korea'). Empty if not present.\n"
        "- doi/arxiv_id/url should be plain strings (empty if unknown).\n"
        "- if uncertain, leave empty string / [] / null.\n"
        "- do not output explanations.\n\n"
        f"Reference:\n{raw_text}"
    )

    # Build message content blocks — images first, then text prompt
    content_blocks: list[dict] = []
    for img_bytes in (reference_page_images or []):
        content_blocks.append({
            "image": {
                "format": "png",
                "source": {"bytes": img_bytes},
            }
        })
    content_blocks.append({"text": prompt})

    response = client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": content_blocks}],
        inferenceConfig={
            "temperature": max(0.0, float(temperature)),
            "maxTokens": max(64, int(max_new_tokens)),
        },
    )
    text = _extract_text_from_converse_response(response)
    payload = _extract_llm_json_payload(text)
    if payload is None:
        return None
    return _normalize_llm_reparse_result(payload)


# Markers that indicate "et al." truncation in an author list (case-insensitive)
_ET_AL_MARKER_RE = re.compile(
    r"\b(?:et\s*\.?\s*al\s*\.?|and\s+others|others)\b",
    re.IGNORECASE,
)


def _is_et_al_token(text: str) -> bool:
    """True if a single string is an et al. / others marker."""
    cleaned = re.sub(r"[\s.]+", "", str(text or "")).lower()
    return cleaned in {"etal", "andothers", "others"}


def _ensure_et_al_preserved(
    new_authors: list[str],
    raw_text: str,
    original_authors: list[str],
) -> list[str]:
    """Re-add 'et al.' to author list if the LLM dropped it.

    LLMs frequently ignore 'preserve et al.' prompt instructions and silently
    strip the marker. This is a problem because downstream R3 detection relies
    on the marker to know the author list is intentionally partial.

    Detection: if the original raw_text contains 'et al.' / 'others' / 'and
    others' AND the new author list doesn't end with such a marker, append
    'et al.' as the last item.
    """
    if not new_authors:
        return new_authors
    # Already has marker?
    if any(_is_et_al_token(a) for a in new_authors):
        return new_authors
    # Was the marker present in the raw text or the rule-parser output?
    raw_has_marker = bool(_ET_AL_MARKER_RE.search(raw_text or ""))
    orig_has_marker = any(_is_et_al_token(a) for a in (original_authors or []))
    if raw_has_marker or orig_has_marker:
        return list(new_authors) + ["et al."]
    return new_authors


def _apply_llm_reparse_if_needed(
    record: CitationRecord,
    provider: str,
    model_path: str,
    bedrock_model_id: str,
    bedrock_region: str,
    bedrock_bearer_token: str | None,
    max_new_tokens: int,
    temperature: float,
    reference_page_images: list[bytes] | None = None,
    paper_style: str = "",
) -> None:
    parsed_fields = dict(record.parsed_fields or {})
    parsed_fields["llm_reparse_attempted"] = True
    if paper_style:
        parsed_fields["llm_reparse_paper_style"] = paper_style
    try:
        # bedrock / openai / azure_openai all share the same chat-client
        # interface via build_chat_client(); a single code path handles them.
        # The legacy local-HF path was removed because the OCR vLLM bundle
        # and a second local LLM tend to fight for GPU memory.
        parsed = _run_llm_reparse_on_raw_chat_client(
            raw_text=record.raw_text,
            provider=provider,
            model_id=bedrock_model_id,
            region=bedrock_region,
            bearer_token=bedrock_bearer_token,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            reference_page_images=reference_page_images,
            paper_style=paper_style,
        )
    except Exception as exc:
        parsed_fields["llm_reparse_error"] = str(exc)
        record.parsed_fields = parsed_fields
        return

    if not parsed:
        parsed_fields["llm_reparse_error"] = "LLM returned no valid JSON payload."
        record.parsed_fields = parsed_fields
        return

    new_title = str(parsed.get("title", "") or "").strip()
    new_authors = [str(item).strip() for item in parsed.get("authors", []) if str(item).strip()]
    # Post-process: re-add 'et al.' marker if it was in the raw text but the LLM
    # dropped it (LLMs often ignore "preserve et al" prompt instructions).
    new_authors = _ensure_et_al_preserved(new_authors, record.raw_text or "", record.authors or [])
    new_venue = str(parsed.get("venue", "") or "").strip()
    new_year = parsed.get("year")
    new_volume = str(parsed.get("volume", "") or "").strip()
    new_pages = str(parsed.get("pages", "") or "").strip()
    new_publisher = str(parsed.get("publisher", "") or "").strip()
    new_location = str(parsed.get("location", "") or "").strip()
    new_doi = str(parsed.get("doi", "") or "").strip().rstrip(".,;")
    new_arxiv_id = str(parsed.get("arxiv_id", "") or "").strip(" .;")
    new_url = str(parsed.get("url", "") or "").strip().rstrip(".,);]")

    # Identifier sanitization. The LLM occasionally returns half-truncated
    # arxiv IDs (e.g. "arXiv:2511" or "2511" when the source BibTeX shipped
    # an incomplete "arXiv–2511.") or malformed DOIs. Both confuse Phase 0:
    # downstream connectors try to resolve the identifier, fail, and emit
    # H5 ("identifier not reachable") for what is really an OK paper missing
    # an arxiv ID. Strip "arXiv:" / "arxiv–" prefixes, then drop the field
    # if it does not match the canonical ID shape.
    rejected_ids: list[str] = []
    if new_arxiv_id:
        cleaned = _ARXIV_PREFIX_RE.sub("", new_arxiv_id).strip(" .;")
        if cleaned and _ARXIV_ID_RE.match(cleaned):
            new_arxiv_id = cleaned
        else:
            rejected_ids.append(f"arxiv_id={new_arxiv_id!r}")
            new_arxiv_id = ""
    if new_doi and not _DOI_RE.match(new_doi):
        rejected_ids.append(f"doi={new_doi!r}")
        new_doi = ""
    if rejected_ids:
        parsed_fields["llm_reparse_rejected_identifiers"] = rejected_ids

    # Avoid destructive overwrite when model yields a syntactically valid but empty answer.
    if not (
        new_title
        or new_authors
        or new_venue
        or new_year is not None
        or new_doi
        or new_arxiv_id
        or new_url
    ):
        parsed_fields["llm_reparse_error"] = "LLM returned an empty structured result."
        record.parsed_fields = parsed_fields
        return

    before = {
        "title": record.title,
        "authors": list(record.authors),
        "venue": record.venue,
        "year": record.year,
        "volume": record.volume,
        "pages": record.pages,
        "publisher": record.publisher,
        "location": record.location,
        "doi": record.doi,
        "arxiv_id": record.arxiv_id,
        "url": record.url,
    }

    # Overwrite core fields + identifiers for this citation once reparse is triggered.
    record.title = new_title
    record.authors = new_authors
    record.venue = new_venue
    # NEVER let the LLM rewrite the year. The H4 mutation in our taxonomy
    # deliberately introduces a wrong year so that the verifier can flag it
    # downstream; an LLM (especially the image-aware reparse) tends to
    # silently "correct" the printed year against the conference year in the
    # venue name or against its own training knowledge, which destroys the
    # H4 detection signal. Keep the heuristic-extracted year as ground truth
    # for what is on the page.
    parsed_fields["llm_reparse_year_skipped"] = True
    # record.year stays whatever the heuristic parser produced.
    record.volume = new_volume
    record.pages = new_pages
    record.publisher = new_publisher
    record.location = new_location
    record.doi = new_doi
    record.arxiv_id = new_arxiv_id
    record.url = new_url

    applied: list[str] = []
    if before["title"] != record.title:
        applied.append("title")
    if before["authors"] != record.authors:
        applied.append("authors")
    if before["venue"] != record.venue:
        applied.append("venue")
    if before["year"] != record.year:
        applied.append("year")
    if before["volume"] != record.volume:
        applied.append("volume")
    if before["pages"] != record.pages:
        applied.append("pages")
    if before["publisher"] != record.publisher:
        applied.append("publisher")
    if before["location"] != record.location:
        applied.append("location")
    if before["doi"] != record.doi:
        applied.append("doi")
    if before["arxiv_id"] != record.arxiv_id:
        applied.append("arxiv_id")
    if before["url"] != record.url:
        applied.append("url")

    parsed_fields["llm_reparse_mode"] = "overwrite_core_and_identifier_fields"
    parsed_fields["llm_reparse_applied_fields"] = applied
    record.parsed_fields = parsed_fields


def parse_reference_entry(entry: str, citation_id: str, paper_style: str = "") -> CitationRecord:
    normalized = re.sub(r"\s+", " ", entry).strip()
    segments = _split_reference_segments(normalized)
    style_hint = _detect_reference_style(normalized)

    authors = _parse_authors_from_segment(segments[0]) if segments else []
    title, title_index = _parse_title(segments, normalized)
    fallback_authors, fallback_title = _extract_authors_and_title_from_initial_style(normalized)
    if fallback_authors and (
        _authors_look_contaminated(authors)
        or len(authors) <= 1
        or (len(fallback_authors) >= 2 and len(authors) < len(fallback_authors))
    ):
        authors = fallback_authors
    if fallback_title and _title_looks_bad(title):
        title = fallback_title
        title_index = _find_title_segment_index(segments, title)

    # If parser picked a later segment as title but first segment clearly contains author+title,
    # prefer recovered title and treat later segment as venue-like metadata.
    if title and title_index is not None and title_index > 0:
        recovered_authors, recovered_title = _extract_authors_and_title_from_initial_style(normalized)
        if recovered_title and (
            _looks_like_venue_segment_for_title(title)
            or re.search(r",\s*\d{1,4}\s*:\s*\d{2,6}(?:[-–]\d{2,6})?,\s*(?:19|20)\d{2}[a-z]?\b", title)
            or VENUE_NAME_HINT_RE.search(title)
        ):
            if recovered_authors:
                authors = recovered_authors
            title = recovered_title
            title_index = _find_title_segment_index(segments, title)

    # Recovery path for long-author citations where boundary split collapsed and title captures authors.
    if _title_looks_like_author_overcapture(title):
        recovered_authors, recovered_title = _extract_authors_and_title_from_initial_style(normalized)
        if recovered_title and not _title_looks_like_author_overcapture(recovered_title):
            if recovered_authors:
                authors = recovered_authors
            title = recovered_title
            title_index = _find_title_segment_index(segments, title)
    venue = _parse_venue(segments, title, title_index)

    # If title still embeds a trailing venue fragment, split and reconcile.
    split_title, split_venue = _split_title_and_venue(title)
    if split_title and split_venue:
        title = split_title
        if not venue:
            venue = split_venue
        else:
            split_norm = _normalize_match_text(split_venue)
            venue_norm = _normalize_match_text(venue)
            if split_norm and venue_norm and (split_norm.endswith(venue_norm) or venue_norm.endswith(split_norm)):
                venue = split_venue if len(split_venue) >= len(venue) else venue

    year = _extract_year(normalized, paper_style=paper_style)
    doi, arxiv_id = extract_identifier(entry)
    doi = doi.rstrip(".,;")
    url = _extract_url(normalized)

    if (
        not title
        and len(authors) == 1
        and authors[0].lower().startswith(("in ", "proceedings of", "findings of"))
    ):
        if not venue:
            venue = authors[0]
        authors = []

    if title and _title_looks_bad(title) and authors and all("," not in a for a in authors):
        split_title_dot, split_venue_dot = _split_title_and_venue(title)
        split_title_cue, split_venue_cue = _split_title_and_venue_by_cue(title)
        has_recoverable_title = (split_title_dot and split_venue_dot) or (split_title_cue and split_venue_cue)
        if not has_recoverable_title:
            venue = title
            title = ""
            authors = []

    if len(authors) == 1 and title:
        if _normalize_match_text(authors[0]) == _normalize_match_text(title):
            authors = []

    lowered_raw = normalized.lower()
    if _title_looks_bad(title):
        if lowered_raw.startswith(("in proceedings", "proceedings of", "in international conference", "pp.")):
            title = ""
            authors = []
            first_sentence = re.split(r"\.(?:\s+|$)", normalized, maxsplit=1)[0].strip(" .;")
            venue = first_sentence or venue

    venue_backfill_source = ""
    if not venue:
        title, venue_backfilled, venue_backfill_source = _backfill_venue_from_title_and_raw(title, normalized)
        if venue_backfilled:
            venue = venue_backfilled

    # Split venue into clean venue + pages + publisher + location + volume.
    venue, pages, publisher, location, volume = _split_venue_fields(venue)
    # Backfill anything still missing by scanning the raw_text after title+venue.
    pages, publisher, location, volume = _backfill_meta_fields_from_raw(
        normalized, title, venue, pages, publisher, location, volume,
    )

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
        volume=volume,
        pages=pages,
        publisher=publisher,
        location=location,
        parsed_fields={
            "parsed_from": "pdf_reference",
            "raw_year": year,
            "raw_authors": authors,
            "segments": segments,
            "style_hint": style_hint,
            "venue_backfill_source": venue_backfill_source,
        },
    )


# Heuristic parse must complete in seconds for typical entries (it's pure
# regex + string ops). Pathological reference strings — long, with many
# punctuation marks — can trigger catastrophic backtracking in
# nested-lazy patterns (APA_RE / HARVARD_RE / COLON_STYLE_RE), wedging a
# single entry for hours and freezing the whole paper-worker. We wrap
# every call in a thread with a hard timeout. On timeout we return a
# minimal record carrying only the raw_text; downstream LLM reparse
# treats it as "missing core fields" and rebuilds the structured form
# from scratch.
_HEURISTIC_PARSE_TIMEOUT_S = float(
    __import__("os").getenv("CITATION_CHECKER_HEURISTIC_PARSE_TIMEOUT_S", "10.0")
)


def _entry_is_pathological(entry: str) -> tuple[bool, str]:
    """Pre-flight check: return (True, reason) for entries that are very
    likely to wedge the parse_reference_entry regex chain in catastrophic
    backtracking. Triggered ONLY on extreme OCR garbage — clean refs
    (200-600 chars with normal spacing) sail through.

    The thread-timeout wrapper around parse_reference_entry cannot rescue
    such entries because Python's `re` module holds the GIL for the
    entire match, so the watchdog thread starves. The only safe action
    is to bypass the regex entirely and let the downstream LLM reparse
    rebuild structured fields from the raw OCR text.
    """
    n = len(entry)
    if n > 1500:
        return True, f"length={n}>1500"
    # Look for any whitespace-free run > 50 chars — typical OCR artifact
    # where words got merged ("SCIENCE&TECHNOLOGYORGANIZATIONCENTREFOR").
    for token in entry.split():
        if len(token) > 50:
            return True, f"no-space run len={len(token)}"
    # Compact-dot density: > 8 dots in any 100-char window suggests a
    # name-initial / abbreviation cluster that lazy regex chokes on.
    for start in range(0, max(1, n - 100), 50):
        if entry.count(".", start, start + 100) > 8:
            return True, f"dot-density>8 in window {start}..{start+100}"
    return False, ""


def _parse_one_with_timeout(
    entry: str, citation_id: str, paper_style: str = "",
) -> CitationRecord:
    """Parse one reference entry, bypassing the regex on pathological OCR.

    Two-layer protection:
      1. Pre-flight ``_entry_is_pathological`` — synchronous, sub-millisecond.
         Catches the OCR garbage that would otherwise wedge the regex.
         (Thread-based timeout cannot rescue catastrophic re.match because
         Python's re module holds the GIL for the full match, so the
         watchdog thread can never wake to print and return.)
      2. Thread timeout wrapper — last-resort safety net for entries that
         pass pre-flight but still take too long. Falls back to the same
         stub record. Useless for true catastrophic backtracking (GIL
         starvation), but catches mild-cost runaway cases where the regex
         occasionally yields the GIL via Python-level callbacks.
    """
    bad, reason = _entry_is_pathological(entry)
    if bad:
        print(
            f"      [WARN] heuristic parse SKIPPED for {citation_id} "
            f"(pathological: {reason}) — falling back to LLM reparse",
            flush=True,
        )
        return CitationRecord(citation_id=citation_id, raw_text=entry[:4000])

    import threading as _t
    result_holder: list = [None]
    exc_holder: list = [None]

    def _worker() -> None:
        try:
            result_holder[0] = parse_reference_entry(
                entry, citation_id, paper_style=paper_style,
            )
        except Exception as exc:  # noqa: BLE001
            exc_holder[0] = exc

    th = _t.Thread(target=_worker, daemon=True, name=f"parse-{citation_id}")
    th.start()
    th.join(timeout=_HEURISTIC_PARSE_TIMEOUT_S)
    if th.is_alive():
        print(
            f"      [WARN] heuristic parse timeout on {citation_id} "
            f"(entry len={len(entry)}) — falling back to LLM reparse",
            flush=True,
        )
        return CitationRecord(citation_id=citation_id, raw_text=entry[:4000])
    if exc_holder[0] is not None:
        raise exc_holder[0]
    return result_holder[0]


def parse_reference_entries(
    entries: list[str],
    llm_reparse_config: Mapping[str, Any] | None = None,
    reference_page_images: list[list[bytes]] | None = None,
) -> list[CitationRecord]:
    """Parse reference entries into CitationRecords.

    Args:
        entries: Raw reference strings.
        llm_reparse_config: LLM reparse configuration.
        reference_page_images: Per-entry list of page image(s) (PNG bytes).
            reference_page_images[i] = list of images for entries[i].
            For cross-page entries, the list has 2 images.
    """
    cfg = llm_reparse_config or {}
    if isinstance(cfg, Mapping):
        get_value = cfg.get
    else:
        get_value = lambda key, default=None: getattr(cfg, key, default)
    _paper_style_for_heuristic = str(get_value("paper_style", "") or "").strip().lower()

    records = []
    total = len(entries)
    for idx, entry in enumerate(entries, start=1):
        print(f"      [parse {idx}/{total}] heuristic parsing...", end="\r", flush=True)
        records.append(
            _parse_one_with_timeout(
                entry, f"pdf-ref:{idx}", paper_style=_paper_style_for_heuristic,
            )
        )
    print(f"      [parse] heuristic parsing done ({total} entries)           ", flush=True)

    enabled = bool(get_value("enabled", False))
    provider = str(get_value("provider", "local") or "local").strip().lower()
    model_path = str(get_value("model_path", "") or "").strip()
    bedrock_cfg = get_value("bedrock", {}) or {}
    if isinstance(bedrock_cfg, Mapping):
        bedrock_get = bedrock_cfg.get
    else:
        bedrock_get = lambda key, default=None: getattr(bedrock_cfg, key, default)
    bedrock_model_id = str(bedrock_get("model_id", "") or "").strip()
    bedrock_region = str(bedrock_get("region", "") or "").strip() or "us-east-1"
    bedrock_bearer_token = str(bedrock_get("bearer_token", "") or "").strip() or None
    max_new_tokens = int(get_value("max_new_tokens", 32768) or 32768)
    temperature = float(get_value("temperature", 0.0) or 0.0)
    force_all_entries = bool(get_value("force_all_entries", False))
    parallel_workers = int(get_value("parallel_workers", 1) or 1)
    parallel_workers = max(1, parallel_workers)

    # Build per-record image mapping for VLM-assisted reparse (Bedrock only).
    # reference_page_images[i] = list of PNG bytes for the i-th entry.
    _per_record_images: dict[str, list[bytes]] = {}
    if provider == "bedrock" and reference_page_images:
        for i, record in enumerate(records):
            if i < len(reference_page_images) and reference_page_images[i]:
                _per_record_images[record.citation_id] = reference_page_images[i]

    can_run = bool(model_path) if provider != "bedrock" else bool(bedrock_model_id)
    if enabled and can_run:
        targets = [
            record
            for record in records
            if force_all_entries or _citation_has_missing_core_fields(record)
        ]
        n_targets = len(targets)
        n_with_images = sum(1 for r in targets if r.citation_id in _per_record_images)
        img_tag = f" +image({n_with_images}/{n_targets})" if n_with_images else ""
        print(f"      [reparse] LLM reparsing {n_targets} citations (provider={provider}{img_tag})...", flush=True)
        if parallel_workers <= 1 or n_targets <= 1:
            for i, record in enumerate(targets, start=1):
                print(f"      [reparse {i}/{n_targets}] {record.citation_id} title={record.title[:50]!r}", end="\r", flush=True)
                _apply_llm_reparse_if_needed(
                    record=record,
                    provider=provider,
                    model_path=model_path,
                    bedrock_model_id=bedrock_model_id,
                    bedrock_region=bedrock_region,
                    bedrock_bearer_token=bedrock_bearer_token,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    reference_page_images=_per_record_images.get(record.citation_id),
                )
            print(f"      [reparse] done ({n_targets} citations)                                        ", flush=True)
        else:
            import threading
            _reparse_counter = {"done": 0}
            _reparse_lock = threading.Lock()
            max_workers = min(parallel_workers, n_targets)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(
                        _apply_llm_reparse_if_needed,
                        record=record,
                        provider=provider,
                        model_path=model_path,
                        bedrock_model_id=bedrock_model_id,
                        bedrock_region=bedrock_region,
                        bedrock_bearer_token=bedrock_bearer_token,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        reference_page_images=_per_record_images.get(record.citation_id),
                    ): record.citation_id
                    for record in targets
                }
                for fut in as_completed(future_map):
                    fut.result()
                    with _reparse_lock:
                        _reparse_counter["done"] += 1
                        print(f"      [reparse {_reparse_counter['done']}/{n_targets}] {future_map[fut]}", end="\r", flush=True)
            print(f"      [reparse] done ({n_targets} citations, workers={max_workers})                ", flush=True)
    return records


def parse_reference_entries_streaming(
    entries: list[str],
    llm_reparse_config: Mapping[str, Any] | None = None,
    reference_page_images: list[list[bytes]] | None = None,
) -> tuple[list[CitationRecord], ThreadPoolExecutor | None, dict[str, "Future[None]"]]:
    """Streaming counterpart to parse_reference_entries.

    Returns:
        records: heuristic-parsed list (in original order). Each record may be
            mutated in-place when its reparse future completes.
        reparse_pool: ThreadPoolExecutor running the reparse jobs (or None if
            reparse is disabled / unnecessary). Caller is responsible for
            shutdown(wait=True) once it has consumed the futures.
        reparse_futures: {citation_id: Future}. Records whose citation_id is
            NOT a key in this dict are already verify-ready; records that ARE
            keyed must wait for their future to complete before verification.
    """
    cfg = llm_reparse_config or {}
    if isinstance(cfg, Mapping):
        get_value = cfg.get
    else:
        get_value = lambda key, default=None: getattr(cfg, key, default)
    # Read paper_style up-front so the heuristic year extractor (used during
    # the parse_reference_entry loop below) can apply style-aware rules.
    _paper_style_for_heuristic = str(get_value("paper_style", "") or "").strip().lower()

    records: list[CitationRecord] = []
    total = len(entries)
    for idx, entry in enumerate(entries, start=1):
        print(f"      [parse {idx}/{total}] heuristic parsing...", end="\r", flush=True)
        records.append(
            _parse_one_with_timeout(
                entry, f"pdf-ref:{idx}", paper_style=_paper_style_for_heuristic,
            )
        )
    print(f"      [parse] heuristic parsing done ({total} entries)           ", flush=True)

    enabled = bool(get_value("enabled", False))
    provider = str(get_value("provider", "local") or "local").strip().lower()
    model_path = str(get_value("model_path", "") or "").strip()
    bedrock_cfg = get_value("bedrock", {}) or {}
    if isinstance(bedrock_cfg, Mapping):
        bedrock_get = bedrock_cfg.get
    else:
        bedrock_get = lambda key, default=None: getattr(bedrock_cfg, key, default)
    bedrock_model_id = str(bedrock_get("model_id", "") or "").strip()
    bedrock_region = str(bedrock_get("region", "") or "").strip() or "us-east-1"
    bedrock_bearer_token = str(bedrock_get("bearer_token", "") or "").strip() or None
    max_new_tokens = int(get_value("max_new_tokens", 32768) or 32768)
    temperature = float(get_value("temperature", 0.0) or 0.0)
    force_all_entries = bool(get_value("force_all_entries", False))
    parallel_workers = int(get_value("parallel_workers", 1) or 1)
    parallel_workers = max(1, parallel_workers)
    paper_style = str(get_value("paper_style", "") or "").strip().lower()

    _per_record_images: dict[str, list[bytes]] = {}
    if provider == "bedrock" and reference_page_images:
        for i, record in enumerate(records):
            if i < len(reference_page_images) and reference_page_images[i]:
                _per_record_images[record.citation_id] = reference_page_images[i]

    can_run = bool(model_path) if provider != "bedrock" else bool(bedrock_model_id)
    if not (enabled and can_run):
        return records, None, {}

    targets = [
        record
        for record in records
        if force_all_entries or _citation_has_missing_core_fields(record)
    ]
    n_targets = len(targets)
    if n_targets == 0:
        return records, None, {}

    n_with_images = sum(1 for r in targets if r.citation_id in _per_record_images)
    img_tag = f" +image({n_with_images}/{n_targets})" if n_with_images else ""
    style_tag = f" style={paper_style}" if paper_style else ""
    max_workers = min(parallel_workers, n_targets)
    print(
        f"      [reparse] LLM reparsing {n_targets} citations "
        f"(provider={provider}{img_tag}{style_tag}, workers={max_workers}, streaming→verify)...",
        flush=True,
    )

    pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="reparse")
    futures: dict[str, "Future[None]"] = {}
    for record in targets:
        futures[record.citation_id] = pool.submit(
            _apply_llm_reparse_if_needed,
            record=record,
            provider=provider,
            model_path=model_path,
            bedrock_model_id=bedrock_model_id,
            bedrock_region=bedrock_region,
            bedrock_bearer_token=bedrock_bearer_token,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            reference_page_images=_per_record_images.get(record.citation_id),
            paper_style=paper_style,
        )
    return records, pool, futures
