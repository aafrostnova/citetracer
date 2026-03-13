from __future__ import annotations

import html
import re
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
        return self._parse_feed(body, policy)

    @staticmethod
    def _extract_version_history(abs_url: str, policy: RequestPolicy) -> tuple[list[dict[str, Any]], list[int]]:
        request = Request(abs_url, headers={"User-Agent": "citation-checker/1.0"})
        try:
            with urlopen(request, timeout=policy.timeout_s) as response:
                page_html = response.read().decode("utf-8", errors="ignore")
        except Exception:
            return [], []

        cleaned = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", page_html)
        text = html.unescape(re.sub(r"(?s)<[^>]+>", " ", cleaned))
        text = re.sub(r"\s+", " ", text).strip()

        if "Submission history" not in text:
            return [], []

        history_text = text.split("Submission history", maxsplit=1)[-1]
        matches = re.findall(
            r"\[(v\d+)\]\s*(.*?)(?=\s*\[(?:v\d+)\]\s*|$)",
            history_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        version_timestamps: list[dict[str, Any]] = []
        version_years: list[int] = []
        for version_label, stamp_text in matches:
            stamp = " ".join(str(stamp_text or "").split()).strip()
            year_match = re.search(r"\b(19\d{2}|20\d{2})\b", stamp)
            year = int(year_match.group(1)) if year_match else None
            version_timestamps.append(
                {
                    "version": version_label.lower(),
                    "timestamp": stamp,
                    "year": year,
                }
            )
            if year is not None and year not in version_years:
                version_years.append(year)
        return version_timestamps, version_years

    @classmethod
    def _parse_feed(cls, feed_xml: str, policy: RequestPolicy) -> list[dict[str, Any]]:
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
            version_timestamps, version_years = cls._extract_version_history(id_text, policy)
            if year is not None and year not in version_years:
                version_years.insert(0, year)
            records.append(
                {
                    "title": title,
                    "authors": authors,
                    "venue": "arXiv",
                    "year": year,
                    "doi": "",
                    "arxiv_id": arxiv_id,
                    "url": id_text,
                    "version_timestamps": version_timestamps,
                    "version_years": version_years,
                }
            )
        return records
