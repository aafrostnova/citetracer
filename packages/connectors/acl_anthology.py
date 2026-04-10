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

    def search(self, citation: CitationRecord, policy: RequestPolicy) -> list[dict[str, object]]:
        query = citation.title or citation.raw_text
        if not query:
            return []

        official_records = self._search_official_data(citation)
        if official_records:
            return official_records

        html = self._request_text(
            "https://aclanthology.org/search/",
            {"q": query},
            policy,
        )
        return self._parse_results(html)

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
