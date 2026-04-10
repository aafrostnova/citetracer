from __future__ import annotations

from typing import Any

from packages.core.models import CitationRecord

from .base import BaseConnector, RequestPolicy


class EuropePMCConnector(BaseConnector):
    name = "europepmc"
    ttl_s = 60 * 60 * 24

    def search(self, citation: CitationRecord, policy: RequestPolicy) -> list[dict[str, Any]]:
        query = citation.doi or citation.title or citation.raw_text
        if not query:
            return []
        payload = self._request_json(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            {
                "query": query,
                "resultType": "core",
                "pageSize": 5,
                "format": "json",
            },
            policy,
        )
        results = payload.get("resultList", {}).get("result", []) or []
        return [self._normalize_item(item) for item in results]

    @staticmethod
    def _normalize_item(item: dict[str, Any]) -> dict[str, Any]:
        authors = []
        author_string = str(item.get("authorString", "") or "").strip()
        if author_string:
            authors = [part.strip().rstrip(".") for part in author_string.split(",") if part.strip()]
        doi = str(item.get("doi", "") or "").strip().lower()
        pmcid = str(item.get("pmcid", "") or "").strip()
        pmid = str(item.get("pmid", "") or "").strip()
        url = ""
        if pmcid:
            url = f"https://europepmc.org/article/PMC/{pmcid}"
        elif pmid:
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        journal_info = item.get("journalInfo") or {}
        journal_volume = str(item.get("journalVolume", "") or journal_info.get("volume", "") or "")
        return {
            "title": str(item.get("title", "") or ""),
            "authors": authors,
            "venue": str(item.get("journalTitle", "") or ""),
            "year": _safe_int(item.get("pubYear")),
            "doi": doi,
            "arxiv_id": "",
            "url": url,
            "volume": journal_volume,
            "pages": str(item.get("pageInfo", "") or ""),
            "publisher": "",
        }


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
