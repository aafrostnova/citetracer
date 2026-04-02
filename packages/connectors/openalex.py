from __future__ import annotations

from typing import Any

from packages.core.models import CitationRecord

from .base import BaseConnector, RequestPolicy


class OpenAlexConnector(BaseConnector):
    name = "openalex"
    ttl_s = 60 * 60 * 24

    def search(self, citation: CitationRecord, policy: RequestPolicy) -> list[dict[str, Any]]:
        query = citation.doi or citation.title
        if not query:
            return []
        payload = self._request_json(
            "https://api.openalex.org/works",
            {"search": query, "per-page": 5},
            policy,
        )
        records = []
        for work in payload.get("results", []):
            authors = []
            for authorship in work.get("authorships", []):
                author = authorship.get("author") or {}
                name = author.get("display_name", "")
                if name:
                    authors.append(str(name))
            doi = str(work.get("doi", "") or "")
            if doi.startswith("https://doi.org/"):
                doi = doi.rsplit("/", maxsplit=1)[-1]
            primary_location = work.get("primary_location") or {}
            source = primary_location.get("source") or {}
            records.append(
                {
                    "title": str(work.get("display_name", "") or ""),
                    "authors": authors,
                    "venue": str(source.get("display_name", "") or ""),
                    "year": work.get("publication_year"),
                    "doi": doi.lower(),
                    "arxiv_id": "",
                    "url": str(work.get("id", "") or ""),
                }
            )
        return records

    def fetch_author_info(self, author_name: str, policy: RequestPolicy) -> dict[str, Any]:
        """Search for an author by name and return identity info for name variant verification."""
        if not author_name or not author_name.strip():
            return {}
        try:
            payload = self._request_json(
                "https://api.openalex.org/authors",
                {"search": author_name.strip(), "per-page": 3},
                policy,
            )
        except Exception:
            return {}
        results = payload.get("results", [])
        if not results:
            return {}
        top = results[0]
        return {
            "openalex_id": str(top.get("id", "") or ""),
            "display_name": str(top.get("display_name", "") or ""),
            "alternate_names": top.get("display_name_alternatives") or [],
            "orcid": str(top.get("orcid", "") or ""),
            "works_count": top.get("works_count", 0),
        }
