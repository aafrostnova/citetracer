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
    "j mach learn res": "jmlr",
    "j of mach learn res": "jmlr",
    # PMLR
    "pmlr": "pmlr",
    "proceedings of machine learning research": "pmlr",
    # SIAM journals
    "sicomp": "siam_j_comput",
    "siam j comput": "siam_j_comput",
    "siam journal on computing": "siam_j_comput",
    "siam j optim": "siam_j_optim",
    "siam journal on optimization": "siam_j_optim",
    "siam j numer anal": "siam_j_numer_anal",
    "siam journal on numerical analysis": "siam_j_numer_anal",
    # COLT
    "colt": "colt",
    "conference on learning theory": "colt",
    "annual conference computational learning theory": "colt",
    "annual conference on computational learning theory": "colt",
    "annual conference on learning theory": "colt",
    # AISTATS
    "aistats": "aistats",
    "international conference on artificial intelligence and statistics": "aistats",
    "artificial intelligence and statistics": "aistats",
    # UAI
    "uai": "uai",
    "conference on uncertainty in artificial intelligence": "uai",
    # KDD
    "kdd": "kdd",
    "sigkdd": "kdd",
    "acm sigkdd": "kdd",
    "knowledge discovery and data mining": "kdd",
    # IJCAI
    "ijcai": "ijcai",
    "international joint conference on artificial intelligence": "ijcai",
    # AAMAS
    "aamas": "aamas",
    "autonomous agents and multiagent systems": "aamas",
    # STOC / FOCS
    "stoc": "stoc",
    "acm symposium on theory of computing": "stoc",
    "symposium on theory of computing": "stoc",
    "focs": "focs",
    "ieee symposium on foundations of computer science": "focs",
    "foundations of computer science": "focs",
    # SODA
    "soda": "soda",
    "acm siam symposium on discrete algorithms": "soda",
    "symposium on discrete algorithms": "soda",
    # TPAMI / PAMI
    "tpami": "tpami",
    "pami": "tpami",
    "ieee transactions on pattern analysis and machine intelligence": "tpami",
    "ieee trans pattern anal mach intell": "tpami",
    # Mathematics of Operations Research
    "math oper res": "math_oper_res",
    "mathematics of operations research": "math_oper_res",
    # Econometrica
    "econometrica": "econometrica",
    # Mathematical Programming
    "math program": "math_program",
    "mathematical programming": "math_program",
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


_VENUE_STOPWORDS = {
    "of", "on", "the", "and", "for", "in", "to", "a", "an", "at", "by",
    "proceedings", "proc", "international", "intl", "annual",
    "conference", "conf", "symposium", "symp", "workshop", "meeting",
    "journal", "trans", "transactions",  # kept optional; see heuristics
}


def _letters_only(text: str) -> str:
    return re.sub(r"[^a-z]", "", (text or "").lower())


def _is_acronym_of(short: str, long: str) -> bool:
    """True if `short` is an acronym built from initials of `long`'s content words.

    Examples that return True:
      - "JMLR"   vs "Journal of Machine Learning Research"  (skip 'of')
      - "SICOMP" vs "SIAM Journal on Computing"             (takes SIAMJC... hmm no)
      - "NIPS"   vs "Neural Information Processing Systems"
      - "COLT"   vs "Conference on Learning Theory"         (skip 'on')
    """
    s = _letters_only(short)
    if not s or len(s) < 2 or len(s) > 10:
        return False
    # If short form has spaces or >10 chars, it's not an acronym
    if " " in short.strip():
        return False

    # Try multiple initial-building strategies:
    words = [w for w in _letters_only(long).replace(" ", "|").split("|") if w]
    # Re-get with actual word boundaries
    words = [w.lower() for w in re.split(r"\W+", long or "") if w]
    if not words:
        return False

    # Strategy 1: initial of EVERY word
    init_all = "".join(w[0] for w in words)
    # Strategy 2: initial of non-stopword words only
    init_content = "".join(w[0] for w in words if w not in _VENUE_STOPWORDS)
    # Strategy 3: initial of first letter of every capitalized word
    # (can't do without original casing)

    candidates = {init_all, init_content}
    for init in candidates:
        if not init:
            continue
        if s == init:
            return True
        # Allow short to be prefix of initials (e.g. "JMLR" vs initials "JMLRR")
        if len(s) >= 3 and (init.startswith(s) or s.startswith(init)):
            return True
    return False


def _is_word_truncation_of(short: str, long: str) -> bool:
    """True if `short` is a dot-separated truncation of `long`'s content words.

    Examples that return True:
      - "J. Mach. Learn. Res." vs "Journal of Machine Learning Research"
      - "Artif. Intell."       vs "Artificial Intelligence"
      - "SIAM J. Comput."      vs "SIAM Journal on Computing"
    """
    if not short or not long:
        return False
    # Split short by dots/spaces into tokens, keep only alphabetic
    short_tokens = [t for t in re.split(r"[.\s,]+", short.lower()) if t and re.match(r"^[a-z]+$", t)]
    if len(short_tokens) < 2:
        return False

    # Split long into words, keep only alphabetic content words (not stopwords)
    long_words = [w.lower() for w in re.split(r"\W+", long or "") if w and re.match(r"^[a-z]+$", w, re.IGNORECASE)]
    long_content = [w for w in long_words if w not in _VENUE_STOPWORDS]

    if len(short_tokens) != len(long_content):
        return False

    # Every short token must be a prefix of the corresponding long word
    for st, lw in zip(short_tokens, long_content):
        if len(st) < 2:
            return False  # single-letter tokens too ambiguous
        if not lw.startswith(st):
            return False
    return True


def venues_equivalent_heuristic(a: str, b: str) -> bool:
    """Return True if a and b are equivalent venues via general heuristics.

    Tries (in order):
      1. Exact alias normalization (via normalize_venue).
      2. Acronym ↔ full form (e.g. JMLR vs Journal of Machine Learning Research).
      3. Word truncation ↔ full form (e.g. J. Mach. Learn. Res. vs Journal ...).
    """
    if not a or not b:
        return False
    na = normalize_venue(a)
    nb = normalize_venue(b)
    if na and nb and na == nb:
        return True
    # Try both directions of acronym / truncation
    if _is_acronym_of(a, b) or _is_acronym_of(b, a):
        return True
    if _is_word_truncation_of(a, b) or _is_word_truncation_of(b, a):
        return True
    return False


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
    # Try matching any alias as a substring of the venue — longest aliases first
    # so specific phrases win over generic ones (e.g. "international conference on
    # artificial intelligence and statistics" > "artificial intelligence").
    sorted_aliases = sorted(VENUE_ALIASES.items(), key=lambda x: -len(x[0]))
    for alias, canonical in sorted_aliases:
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
