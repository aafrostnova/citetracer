from __future__ import annotations

from typing import Any

from packages.core.models import CitationRecord

from .base import BaseConnector, RequestPolicy


class OpenAlexConnector(BaseConnector):
    name = "openalex"
    ttl_s = 60 * 60 * 24

    def __init__(
        self, mailto: str | None = None, api_key: str | None = None,
    ) -> None:
        # Polite-pool opt-in: add mailto=<email> to every request. OpenAlex
        # promises a separate, faster queue and a higher daily quota to polite
        # clients. See https://docs.openalex.org/how-to-use-the-api/rate-limits.
        self.mailto = (mailto or "").strip() or None
        # Premium/authenticated api_key: passed as api_key=<key> URL param.
        # Upgrades from polite pool to authenticated rate limits (much higher
        # daily quota).
        self.api_key = (api_key or "").strip() or None

    def _auth_params(self) -> dict[str, Any]:
        """Return the mailto / api_key URL params to include on every call."""
        extra: dict[str, Any] = {}
        if self.mailto:
            extra["mailto"] = self.mailto
        if self.api_key:
            extra["api_key"] = self.api_key
        return extra

    def search(self, citation: CitationRecord, policy: RequestPolicy) -> list[dict[str, Any]]:
        query = citation.doi or citation.title
        if not query:
            return []
        params: dict[str, Any] = {"search": query, "per-page": 5}
        params.update(self._auth_params())
        payload = self._request_json(
            "https://api.openalex.org/works",
            params,
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
            biblio = work.get("biblio") or {}
            volume = str(biblio.get("volume", "") or "")
            first_page = str(biblio.get("first_page", "") or "")
            last_page = str(biblio.get("last_page", "") or "")
            if first_page and last_page:
                pages = f"{first_page}-{last_page}"
            else:
                pages = first_page or last_page
            publisher = str(source.get("host_organization_name", "") or source.get("publisher", "") or "")
            records.append(
                {
                    "title": str(work.get("display_name", "") or ""),
                    "authors": authors,
                    "venue": str(source.get("display_name", "") or ""),
                    "year": work.get("publication_year"),
                    "doi": doi.lower(),
                    "arxiv_id": "",
                    "url": str(work.get("id", "") or ""),
                    "volume": volume,
                    "pages": pages,
                    "publisher": publisher,
                }
            )
        return records

    def fetch_author_info(self, author_name: str, policy: RequestPolicy) -> dict[str, Any]:
        """Search for an author by name and return identity info for name variant verification."""
        if not author_name or not author_name.strip():
            return {}
        try:
            params: dict[str, Any] = {"search": author_name.strip(), "per-page": 3}
            params.update(self._auth_params())
            payload = self._request_json(
                "https://api.openalex.org/authors",
                params,
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
