from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher


VENUE_ALIASES = {
    # NeurIPS
    "nips": "neurips",
    "neurips": "neurips",
    "advances in neural information processing systems": "neurips",
    # ICML
    "icml": "icml",
    "international conference on machine learning": "icml",
    "proceedings of the international conference on machine learning": "icml",
    # ICLR
    "iclr": "iclr",
    "international conference on learning representations": "iclr",
    # CVPR
    "cvpr": "cvpr",
    "ieee conference on computer vision and pattern recognition": "cvpr",
    "ieee/cvf conference on computer vision and pattern recognition": "cvpr",
    # ECCV / ICCV
    "eccv": "eccv",
    "european conference on computer vision": "eccv",
    "iccv": "iccv",
    "ieee international conference on computer vision": "iccv",
    # ACL
    "acl": "acl",
    "association for computational linguistics": "acl",
    "annual meeting of the association for computational linguistics": "acl",
    # EMNLP
    "emnlp": "emnlp",
    "empirical methods in natural language processing": "emnlp",
    # NAACL
    "naacl": "naacl",
    "north american chapter of the association for computational linguistics": "naacl",
    # AAAI
    "aaai": "aaai",
    "aaai conference on artificial intelligence": "aaai",
    # AI journal
    "artificial intelligence": "ai_journal",
    "artif intell": "ai_journal",
    # Nature family
    "nat": "nature",
    "nature": "nature",
    # Science
    "science": "science",
    # JMLR
    "jmlr": "jmlr",
    "journal of machine learning research": "jmlr",
    # PMLR
    "pmlr": "pmlr",
    "proceedings of machine learning research": "pmlr",
    # arXiv
    "arxiv": "arxiv",
    "corr": "arxiv",
    "arxiv preprint": "arxiv",
}

NAME_ALIASES = {
    "mike": "michael",
    "matt": "matthew",
    "alex": "alexander",
    "ben": "benjamin",
    "chris": "christopher",
    "dan": "daniel",
    "jon": "jonathan",
    "nick": "nicholas",
}


DOI_RE = re.compile(r"10\.[0-9]{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
ARXIV_RE = re.compile(r"(?:arxiv:)?([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)", re.IGNORECASE)


def normalize_text(value: str) -> str:
    if not value:
        return ""
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = value.lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _normalize_math_markup(value: str) -> str:
    if not value:
        return ""
    value = value.replace("\\times", " x ")
    value = value.replace("\\cdot", " x ")
    value = value.replace("×", " x ")
    value = value.replace("\\(", " ")
    value = value.replace("\\)", " ")
    value = value.replace("\\[", " ")
    value = value.replace("\\]", " ")
    value = value.replace("$", " ")
    value = value.replace("`", " ")
    return value


def normalize_title(title: str) -> str:
    return normalize_text(_normalize_math_markup(title)).replace(" ", "")


def normalize_arxiv_id(value: str) -> str:
    if not value:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"^https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/", "", text)
    text = re.sub(r"\.pdf$", "", text)
    text = re.sub(r"^arxiv:", "", text)
    text = re.sub(r"v\d+$", "", text)
    return text.strip()


# Applied AFTER normalize_text, so unicode dashes are already stripped.
# Matches volume:pages patterns like "18 279284", "591 7850 379384", "pp 1 5".
_VENUE_STRIP_RE = re.compile(
    r",?\s*(?:"
    r"pages?\s*[\d\s]+|"                    # pages 123 or pages 1 5
    r"vol(?:ume)?\.?\s*[\d]+|"              # volume 5 / vol. 5
    r"[\d]+\s*\([^)]*\)\s*[\d\s]+|"        # 591(7850) 379384
    r"[\d]+\s+[\d]{2,}[\d\s]*|"            # 18 279284 (vol:pages after normalize)
    r"pp\.?\s*[\d\s]+"                      # pp 1 5
    r")"
)


def normalize_venue(venue: str) -> str:
    cleaned = normalize_text(venue)
    # Strip leading "in " (common in citation venue fields)
    cleaned = re.sub(r"^in\s+", "", cleaned)
    # Strip trailing year
    cleaned = re.sub(r",?\s*(19|20)\d{2}\w?$", "", cleaned)
    # Strip volume/page info
    cleaned = _VENUE_STRIP_RE.sub("", cleaned).strip()
    # Strip standalone numbers (residual volume/issue numbers)
    cleaned = re.sub(r"\b\d+\b", "", cleaned)
    # Collapse whitespace and strip punctuation
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;.")
    # Try exact alias match
    if cleaned in VENUE_ALIASES:
        return VENUE_ALIASES[cleaned]
    # Try matching any alias as a substring of the venue
    for alias, canonical in VENUE_ALIASES.items():
        if len(alias) > 3 and alias in cleaned:
            return canonical
    return cleaned


def normalize_author(author: str) -> str:
    cleaned = normalize_text(author)
    cleaned = cleaned.replace(" ", " ").strip()
    parts = [p for p in cleaned.split(" ") if p]
    if not parts:
        return ""
    if parts[0] in NAME_ALIASES:
        parts[0] = NAME_ALIASES[parts[0]]
    return " ".join(parts)


def author_tokens(author: str) -> set[str]:
    normalized = normalize_author(author)
    return {token for token in normalized.split(" ") if token}


def extract_identifier(raw_text: str) -> tuple[str, str]:
    doi_match = DOI_RE.search(raw_text)
    arxiv_match = ARXIV_RE.search(raw_text)
    doi = doi_match.group(0) if doi_match else ""
    arxiv_id = arxiv_match.group(1) if arxiv_match else ""
    return doi.lower(), arxiv_id.lower()


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def author_overlap_score(left: list[str], right: list[str]) -> float:
    if not left or not right:
        return 0.0
    left_tokens = [author_tokens(author) for author in left]
    right_tokens = [author_tokens(author) for author in right]
    matches = 0
    for l_tokens in left_tokens:
        if not l_tokens:
            continue
        found = False
        for r_tokens in right_tokens:
            if not r_tokens:
                continue
            inter = l_tokens.intersection(r_tokens)
            union = l_tokens.union(r_tokens)
            if union and (len(inter) / len(union)) >= 0.5:
                found = True
                break
        if found:
            matches += 1
    return matches / max(len(left_tokens), 1)


def venue_match_score(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return similarity(normalize_venue(left), normalize_venue(right))


def year_consistency(left: int | None, right: int | None) -> float:
    if left is None or right is None:
        return 0.5
    if left == right:
        return 1.0
    if abs(left - right) == 1:
        return 0.5
    return 0.0
