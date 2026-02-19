from __future__ import annotations

from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

from packages.core.models import CitationRecord

from .base import BaseConnector, RequestPolicy


class ArxivConnector(BaseConnector):
    name = "arxiv"
    ttl_s = 60 * 60 * 24

    def search(self, citation: CitationRecord, policy: RequestPolicy) -> list[dict[str, Any]]:
        query = citation.arxiv_id or citation.title
        if not query:
            return []

        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": 5,
        }
        target = f"http://export.arxiv.org/api/query?{urlencode(params)}"
        request = Request(target, headers={"User-Agent": "citation-checker/1.0"})
        with urlopen(request, timeout=policy.timeout_s) as response:
            body = response.read().decode("utf-8")
        return self._parse_feed(body)

    @staticmethod
    def _parse_feed(feed_xml: str) -> list[dict[str, Any]]:
        namespace = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(feed_xml)
        records = []
        for entry in root.findall("atom:entry", namespace):
            title = (entry.findtext("atom:title", default="", namespaces=namespace) or "").strip()
            id_text = (entry.findtext("atom:id", default="", namespaces=namespace) or "").strip()
            published = (entry.findtext("atom:published", default="", namespaces=namespace) or "").strip()
            year = int(published[:4]) if len(published) >= 4 and published[:4].isdigit() else None
            authors = []
            for author_node in entry.findall("atom:author", namespace):
                name_text = (author_node.findtext("atom:name", default="", namespaces=namespace) or "").strip()
                if name_text:
                    authors.append(name_text)
            arxiv_id = id_text.rsplit("/", maxsplit=1)[-1]
            records.append(
                {
                    "title": title,
                    "authors": authors,
                    "venue": "arXiv",
                    "year": year,
                    "doi": "",
                    "arxiv_id": arxiv_id,
                    "url": id_text,
                }
            )
        return records
