from __future__ import annotations

import re
from pathlib import Path
import threading
from typing import Any

from packages.core.models import CitationRecord
from packages.core.normalize import normalize_title

from .base import BaseConnector, RequestPolicy


class ACLAnthologyConnector(BaseConnector):
    name = "acl_anthology"
    ttl_s = 60 * 60 * 24

    _official_index_cache: dict[str, list[dict[str, Any]]] = {}
    _official_index_lock = threading.Lock()

    def __init__(
        self,
        data_dir: str | Path | None = None,
        repo_path: str | Path | None = None,
    ) -> None:
        self.data_dir = str(data_dir or "").strip()
        self.repo_path = str(repo_path or "").strip()

    # Map ACL-family venue keyword → per-year volume URL suffixes.
    # Each suffix produces a candidate URL like:
    #     https://aclanthology.org/volumes/{year}.{suffix}.bib
    # List is ordered by expected hit rate (main tracks first).
    _ACL_VOLUME_SUFFIXES = {
        "acl": [
            "acl-long", "acl-short", "acl-main", "findings-acl",
            "acl-industry", "acl-demos", "acl-srw", "acl-tutorials",
            "acl-ijcnlp-long", "acl-ijcnlp-short", "acl-ijcnlp-main",
        ],
        "emnlp": [
            "emnlp-main", "emnlp-long", "emnlp-short", "findings-emnlp",
            "emnlp-industry", "emnlp-demos", "emnlp-srw", "emnlp-tutorials",
        ],
        "naacl": [
            "naacl-long", "naacl-short", "naacl-main", "findings-naacl",
            "naacl-industry", "naacl-demo", "naacl-srw", "naacl-tutorials",
        ],
        "eacl": [
            "eacl-long", "eacl-short", "eacl-main", "findings-eacl",
            "eacl-demo", "eacl-srw", "eacl-tutorials",
        ],
        "coling": [
            "coling-main", "coling", "coling-industry", "coling-demos",
            "coling-srw", "coling-tutorials",
        ],
        "aacl": [
            "aacl-main", "aacl-short", "findings-aacl",
            "aacl-demo", "aacl-srw", "aacl-tutorials",
        ],
        # Journals (volumes typically 1-4 per year)
        "tacl":    ["tacl-1", "tacl.1", "tacl"],
        "cl":      ["cl-1", "cl-2", "cl-3", "cl-4", "cl"],
        # Other conferences / evaluation venues
        "lrec":    ["lrec-main", "lrec-1"],
        "wmt":     ["wmt-1", "wmt"],
        "conll":   ["conll-1", "conll"],
        "semeval": ["semeval-1", "semeval"],
        "sigdial": ["sigdial-1", "sigdial"],
        "bionlp":  ["bionlp-1", "bionlp"],
        "iwslt":   ["iwslt-1", "iwslt"],
        "sigmorphon": ["sigmorphon-1", "sigmorphon"],
        "inlg":    ["inlg-main", "inlg-1"],
        "ranlp":   ["ranlp-1", "ranlp"],
    }
    _volume_bib_cache: dict[str, list[dict[str, Any]]] = {}
    _volume_bib_lock = threading.Lock()

    def search(self, citation: CitationRecord, policy: RequestPolicy) -> list[dict[str, object]]:
        query = citation.title or citation.raw_text
        if not query:
            return []

        official_records = self._search_official_data(citation)
        if official_records:
            return official_records

        # 1. Direct anthology-id lookup via DOI or URL (fastest, most accurate)
        direct_records = self._search_by_anthology_id(citation, policy)
        if direct_records:
            return direct_records

        # 2. Per-year volume BibTeX fetch (works for ACL-family venues with a year)
        volume_records = self._search_volume_bibtex(citation, policy)
        if volume_records:
            return volume_records

        html = self._request_text(
            "https://aclanthology.org/search/",
            {"q": query},
            policy,
        )
        return self._parse_results(html)

    # ---------------------------------------------------------------
    # Direct anthology-id lookup (via DOI or URL)
    # ---------------------------------------------------------------

    # ACL Anthology ID patterns:
    #   Modern (2020+):  "2023.acl-long.5" / "2023.emnlp-main.12" / "2024.findings-acl.3"
    #   Legacy (pre-2020): "P19-1001" / "N18-1042" / "D17-1069"
    _ANTHOLOGY_ID_RE = re.compile(
        r"(?:aclanthology\.org/|10\.18653/v1/)"
        r"([A-Z]\d{2}-\d+|\d{4}\.[\w\-]+\.\d+)",
        re.IGNORECASE,
    )

    def _search_by_anthology_id(self, citation: CitationRecord, policy: RequestPolicy) -> list[dict[str, object]]:
        """If citation has a DOI or URL that contains an ACL Anthology ID,
        directly fetch the per-paper BibTeX."""
        seen_ids: set[str] = set()
        records: list[dict[str, object]] = []

        for source in (citation.doi, citation.url):
            if not source:
                continue
            m = self._ANTHOLOGY_ID_RE.search(source)
            if not m:
                continue
            anthology_id = m.group(1)
            if anthology_id in seen_ids:
                continue
            seen_ids.add(anthology_id)

            url = f"https://aclanthology.org/{anthology_id}.bib"
            try:
                text = self._request_text(url, {}, policy)
            except Exception:
                continue
            if not text or len(text) < 20 or "<html" in text[:200].lower():
                continue

            entries = self._parse_bibtex_entries(text)
            year = citation.year or 0
            for entry in entries:
                # Extract year from the anthology_id if not provided
                if entry.get("year", "").isdigit():
                    year_val = int(entry["year"])
                elif anthology_id[:4].isdigit():
                    year_val = int(anthology_id[:4])
                else:
                    # legacy format P19-1001 → year 2019
                    yy = anthology_id[1:3]
                    year_val = 1900 + int(yy) if int(yy) >= 60 else 2000 + int(yy)

                suffix_hint = anthology_id.split(".", 1)[1] if "." in anthology_id else anthology_id
                records.append(self._normalize_bibtex_entry(entry, year_val, suffix_hint))
                if len(records) >= 3:
                    return records

        return records

    # ---------------------------------------------------------------
    # Per-year volume BibTeX fetch
    # ---------------------------------------------------------------

    def _detect_acl_venue_key(self, venue: str) -> str:
        """Return canonical ACL family key if venue matches, else ''."""
        v = (venue or "").lower()
        for key in self._ACL_VOLUME_SUFFIXES:
            if key in v:
                return key
        # Spelled-out names
        spelled = {
            "association for computational linguistics": "acl",
            "empirical methods in natural language processing": "emnlp",
            "north american chapter": "naacl",
            "european chapter of the association for computational linguistics": "eacl",
            "international conference on computational linguistics": "coling",
            "asia-pacific chapter of the association for computational linguistics": "aacl",
            "transactions of the association for computational linguistics": "tacl",
            "computational linguistics": "cl",
            "language resources and evaluation": "lrec",
            "conference on machine translation": "wmt",
            "computational natural language learning": "conll",
            "semantic evaluation": "semeval",
            "special interest group on discourse and dialogue": "sigdial",
            "biomedical natural language processing": "bionlp",
            "spoken language translation": "iwslt",
            "natural language generation": "inlg",
        }
        for needle, key in spelled.items():
            if needle in v:
                return key
        return ""

    def _search_volume_bibtex(self, citation: CitationRecord, policy: RequestPolicy) -> list[dict[str, object]]:
        """Fetch per-year volume BibTeX files and match by title."""
        venue_key = self._detect_acl_venue_key(citation.venue or "")
        if not venue_key:
            return []
        year = citation.year
        if not year:
            return []
        title = (citation.title or "").strip()
        if not title:
            return []

        title_norm = normalize_title(title)
        suffixes = self._ACL_VOLUME_SUFFIXES.get(venue_key, [])
        matches: list[dict[str, object]] = []

        for suffix in suffixes:
            try:
                entries = self._fetch_volume_bibtex(year, suffix, policy)
            except Exception:
                continue
            for entry in entries:
                entry_title = (entry.get("title") or "").strip()
                if not entry_title:
                    continue
                entry_norm = normalize_title(entry_title)
                if entry_norm == title_norm or title_norm in entry_norm or entry_norm in title_norm:
                    matches.append(self._normalize_bibtex_entry(entry, year, suffix))
                    if len(matches) >= 5:
                        return matches
        return matches

    def _fetch_volume_bibtex(self, year: int, suffix: str, policy: RequestPolicy) -> list[dict[str, str]]:
        cache_key = f"{year}.{suffix}"
        with self._volume_bib_lock:
            cached = self._volume_bib_cache.get(cache_key)
            if cached is not None:
                return cached

        url = f"https://aclanthology.org/volumes/{year}.{suffix}.bib"
        try:
            text = self._request_text(url, {}, policy)
        except Exception:
            text = ""
        if not text or len(text) < 20 or "<html" in text[:200].lower():
            # Non-200 or HTML error page (volume doesn't exist)
            with self._volume_bib_lock:
                self._volume_bib_cache[cache_key] = []
            return []

        entries = self._parse_bibtex_entries(text)
        with self._volume_bib_lock:
            self._volume_bib_cache[cache_key] = entries
        return entries

    @staticmethod
    def _parse_bibtex_entries(bibtex_text: str) -> list[dict[str, str]]:
        """Minimal BibTeX parser — handles `@type{key, field = {value}, ...}` entries."""
        entries = []
        for m in re.finditer(r"@(\w+)\s*\{\s*([^,]+),\s*(.*?)\n\}\n", bibtex_text, re.DOTALL):
            entry_type = m.group(1).lower()
            if entry_type in ("comment", "string", "preamble"):
                continue
            body = m.group(3)
            fields: dict[str, str] = {"entry_type": entry_type, "key": m.group(2).strip()}
            for fm in re.finditer(r'(\w+)\s*=\s*[\{"]((?:[^{}"]|\{[^{}]*\})*)[\}"]\s*,?', body):
                fname = fm.group(1).lower()
                fval = fm.group(2).strip()
                fval = re.sub(r"[{}]", "", fval).strip()
                fields[fname] = fval
            entries.append(fields)
        return entries

    @staticmethod
    def _normalize_bibtex_entry(entry: dict, year: int, suffix: str) -> dict[str, object]:
        authors_raw = entry.get("author", "")
        authors_raw = re.sub(r"\s+", " ", authors_raw).strip()
        authors = [a.strip() for a in re.split(r"\s+and\s+", authors_raw) if a.strip()]
        # Convert "Last, First" → "First Last"
        authors = [
            (a.split(",", 1)[1].strip() + " " + a.split(",", 1)[0].strip())
            if "," in a else a
            for a in authors
        ]
        pages = entry.get("pages", "").replace("--", "-")
        return {
            "title": entry.get("title", ""),
            "authors": authors,
            "venue": entry.get("booktitle", "") or entry.get("journal", ""),
            "year": int(entry["year"]) if entry.get("year", "").isdigit() else year,
            "doi": (entry.get("doi", "") or "").lower(),
            "arxiv_id": "",
            "url": entry.get("url", ""),
            "volume": entry.get("volume", ""),
            "pages": pages,
            "publisher": entry.get("publisher", "") or "Association for Computational Linguistics",
            "location": entry.get("address", ""),
        }

    def _search_official_data(self, citation: CitationRecord) -> list[dict[str, object]]:
        if not (self.data_dir or self.repo_path):
            return []

        query = str(citation.title or citation.raw_text or "").strip()
        if not query:
            return []

        try:
            records = self._load_official_index()
        except Exception:
            return []

        query_norm = normalize_title(query)
        query_lower = query.lower()
        exact: list[dict[str, object]] = []
        partial: list[dict[str, object]] = []
        for record in records:
            title = str(record.get("title", "") or "")
            if not title:
                continue
            title_norm = normalize_title(title)
            if title_norm == query_norm:
                exact.append(record)
                continue
            if query_lower in title.lower():
                partial.append(record)
        return (exact or partial)[:5]

    def _load_official_index(self) -> list[dict[str, Any]]:
        cache_key = self.data_dir or self.repo_path
        with self._official_index_lock:
            cached = self._official_index_cache.get(cache_key)
            if cached is not None:
                return cached
            records = self._build_official_index()
            self._official_index_cache[cache_key] = records
            return records

    def _build_official_index(self) -> list[dict[str, Any]]:
        try:
            from acl_anthology import Anthology
        except ImportError as exc:
            raise RuntimeError("acl-anthology package is not installed") from exc

        if self.data_dir:
            anthology = Anthology(datadir=self.data_dir, verbose=False)
        else:
            anthology = Anthology.from_repo(path=self.repo_path or None, verbose=False)

        records: list[dict[str, Any]] = []
        for paper in anthology.papers():
            title = str(getattr(paper, "title", "") or "").strip()
            if not title:
                continue
            records.append(
                {
                    "title": title,
                    "authors": _paper_authors(paper),
                    "venue": _paper_venue(paper) or "ACL Anthology",
                    "year": _coerce_year(getattr(paper, "year", None)),
                    "doi": str(getattr(paper, "doi", "") or "").strip().lower(),
                    "arxiv_id": "",
                    "url": str(getattr(paper, "url", "") or "").strip(),
                    "volume": "",
                    "pages": str(getattr(paper, "pages", "") or "").strip(),
                    "publisher": str(getattr(paper, "publisher", "") or "").strip() or "Association for Computational Linguistics",
                }
            )
        return records

    def _parse_results(self, body: str) -> list[dict[str, object]]:
        records = []
        seen_urls: set[str] = set()
        pattern = re.compile(
            r'<a[^>]+href="(?P<href>/[^"#?]+/?)"[^>]*>(?P<title>.*?)</a>',
            flags=re.IGNORECASE | re.DOTALL,
        )
        for match in pattern.finditer(body):
            href = match.group("href").strip()
            if not href.startswith("/") or href in seen_urls:
                continue
            title = self._clean_text(match.group("title"))
            if not title or len(title) < 10:
                continue
            if not re.search(r"[A-Za-z]{3,}", title):
                continue
            seen_urls.add(href)
            year = None
            year_match = re.search(r"/(?P<year>(?:19|20)\d{2})\.", href)
            if year_match:
                year = int(year_match.group("year"))
            records.append(
                {
                    "title": title,
                    "authors": [],
                    "venue": "ACL Anthology",
                    "year": year,
                    "doi": "",
                    "arxiv_id": "",
                    "url": f"https://aclanthology.org{href}",
                }
            )
            if len(records) >= 5:
                break
        return records


def _coerce_year(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _paper_authors(paper: Any) -> list[str]:
    authors = []
    for author in getattr(paper, "authors", []) or []:
        value = str(author).strip()
        if value:
            authors.append(value)
    return authors


def _paper_venue(paper: Any) -> str:
    parent = getattr(paper, "parent", None)
    title = str(getattr(parent, "title", "") or "").strip()
    if title:
        return title
    get_volume = getattr(paper, "get_volume", None)
    if callable(get_volume):
        try:
            volume = get_volume()
        except Exception:
            volume = None
        title = str(getattr(volume, "title", "") or "").strip()
        if title:
            return title
    return ""
