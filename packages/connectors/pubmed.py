from __future__ import annotations

from typing import Any

from packages.core.models import CitationRecord

from .base import BaseConnector, RequestPolicy


class PubMedConnector(BaseConnector):
    name = "pubmed"
    ttl_s = 60 * 60 * 24

    def __init__(self, api_key: str | None = None, email: str | None = None) -> None:
        self.api_key = api_key
        self.email = email

    def search(self, citation: CitationRecord, policy: RequestPolicy) -> list[dict[str, Any]]:
        query = citation.doi or citation.title or citation.raw_text
        if not query:
            return []

        search_payload = self._request_json(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            {
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": 5,
                "sort": "relevance",
                "api_key": self.api_key,
                "email": self.email,
            },
            policy,
        )
        id_list = search_payload.get("esearchresult", {}).get("idlist", []) or []
        if not id_list:
            return []

        summary_payload = self._request_json(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            {
                "db": "pubmed",
                "id": ",".join(str(pmid) for pmid in id_list),
                "retmode": "json",
                "api_key": self.api_key,
                "email": self.email,
            },
            policy,
        )
        result = summary_payload.get("result", {}) or {}
        records = []
        for pmid in id_list:
            item = result.get(str(pmid), {}) or {}
            if not item:
                continue
            records.append(self._normalize_item(str(pmid), item))
        return records

    @staticmethod
    def _normalize_item(pmid: str, item: dict[str, Any]) -> dict[str, Any]:
        article_ids = item.get("articleids", []) or []
        doi = ""
        for article_id in article_ids:
            if str(article_id.get("idtype", "")).lower() == "doi":
                doi = str(article_id.get("value", "") or "").lower()
                break
        authors = []
        for author in item.get("authors", []) or []:
            name = str(author.get("name", "") or "").strip()
            if name:
                authors.append(name)
        pubdate = str(item.get("pubdate", "") or "")
        year = None
        for token in pubdate.replace("/", " ").split():
            if token.isdigit() and len(token) == 4:
                year = int(token)
                break
        return {
            "title": str(item.get("title", "") or ""),
            "authors": authors,
            "venue": str(item.get("fulljournalname", "") or item.get("source", "") or ""),
            "year": year,
            "doi": doi,
            "arxiv_id": "",
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        }
