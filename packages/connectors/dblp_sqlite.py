from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

from packages.core.models import CitationRecord

from .base import BaseConnector, RequestPolicy


DOI_RE = re.compile(r"(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)")
ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([^/?#]+)", re.IGNORECASE)


def _extract_doi(url: str) -> str:
    if not url:
        return ""
    match = DOI_RE.search(url)
    return match.group(1).lower() if match else ""


def _extract_arxiv_id(url: str) -> str:
    if not url:
        return ""
    match = ARXIV_RE.search(url)
    return match.group(1).lower() if match else ""


class DblpSQLiteConnector(BaseConnector):
    name = "dblp_sqlite"
    ttl_s = 60 * 60 * 6

    def __init__(self, sqlite_path: str | Path) -> None:
        self.sqlite_path = Path(sqlite_path)

    def _to_records(self, rows: list[tuple[Any, ...]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for _, title, venue, year, url, authors_blob in rows:
            url = str(url or "")
            authors = [name.strip() for name in str(authors_blob or "").split("|||") if name.strip()]
            results.append(
                {
                    "title": str(title or ""),
                    "authors": authors,
                    "venue": str(venue or ""),
                    "year": int(year) if year is not None else None,
                    "doi": _extract_doi(url),
                    "arxiv_id": _extract_arxiv_id(url),
                    "url": url,
                }
            )
        return results

    def _run_query(self, sql: str, param: str) -> list[dict[str, Any]]:
        if not self.sqlite_path.exists():
            return []
        conn = sqlite3.connect(str(self.sqlite_path))
        try:
            rows = conn.execute(sql, (param,)).fetchall()
        finally:
            conn.close()
        return self._to_records(rows)

    def _query_exact(self, title: str) -> list[dict[str, Any]]:
        sql = """
            SELECT
                p.id,
                p.title,
                p.venue,
                p.year,
                p.url,
                (
                    SELECT GROUP_CONCAT(a.name, '|||')
                    FROM paper_authors pa
                    JOIN authors a ON a.id = pa.author_id
                    WHERE pa.paper_id = p.id
                    ORDER BY pa.author_order
                ) AS authors_blob
            FROM papers p
            WHERE p.title = ?
            ORDER BY p.year DESC
            LIMIT 30
        """
        return self._run_query(sql, title)

    @staticmethod
    def _glob_escape(text: str) -> str:
        return text.replace("[", "[[]").replace("*", "[*]").replace("?", "[?]")

    def _query_prefix(self, title_prefix: str) -> list[dict[str, Any]]:
        sql = """
            SELECT
                p.id,
                p.title,
                p.venue,
                p.year,
                p.url,
                (
                    SELECT GROUP_CONCAT(a.name, '|||')
                    FROM paper_authors pa
                    JOIN authors a ON a.id = pa.author_id
                    WHERE pa.paper_id = p.id
                    ORDER BY pa.author_order
                ) AS authors_blob
            FROM papers p
            WHERE p.title GLOB ?
            ORDER BY p.year DESC
            LIMIT 30
        """
        return self._run_query(sql, f"{self._glob_escape(title_prefix)}*")

    def search(self, citation: CitationRecord, policy: RequestPolicy) -> list[dict[str, Any]]:
        del policy
        title = str(citation.title or "").strip()
        if not title:
            return []

        records: list[dict[str, Any]] = []

        exact_variants = [title]
        if title.endswith("."):
            exact_variants.append(title[:-1].strip())
        else:
            exact_variants.append(f"{title}.")

        for variant in exact_variants:
            if not variant:
                continue
            records = self._query_exact(variant)
            if records:
                break

        if not records:
            prefix = re.sub(r"\s+", " ", title).strip()
            if len(prefix) > 96:
                prefix = prefix[:96].rstrip()
            records = self._query_prefix(prefix)

        if records:
            return records
        return []
