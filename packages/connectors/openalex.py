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
                name = authorship.get("author", {}).get("display_name", "")
                if name:
                    authors.append(str(name))
            doi = str(work.get("doi", "") or "")
            if doi.startswith("https://doi.org/"):
                doi = doi.rsplit("/", maxsplit=1)[-1]
            records.append(
                {
                    "title": str(work.get("display_name", "") or ""),
                    "authors": authors,
                    "venue": str(work.get("primary_location", {}).get("source", {}).get("display_name", "") or ""),
                    "year": work.get("publication_year"),
                    "doi": doi.lower(),
                    "arxiv_id": "",
                    "url": str(work.get("id", "") or ""),
                }
            )
        return records
