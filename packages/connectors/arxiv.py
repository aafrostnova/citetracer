from __future__ import annotations

import re
import threading
import time
from typing import Any
import xml.etree.ElementTree as ET

from packages.core.models import CitationRecord

from .base import BaseConnector, RequestPolicy


class ArxivConnector(BaseConnector):
    name = "arxiv"
    ttl_s = 60 * 60 * 24
    api_url = "https://export.arxiv.org/api/query"
    min_interval_s = 3.0

    _request_lock = threading.Lock()
    _last_request_at = 0.0

    # arXiv search queries can take 30s+; override global timeout for this connector.
    _search_timeout_s: float = 45.0

    def search(self, citation: CitationRecord, policy: RequestPolicy) -> list[dict[str, Any]]:
        params = self._build_params(citation)
        if not params:
            return []
        arxiv_policy = RequestPolicy(
            timeout_s=max(policy.timeout_s, self._search_timeout_s),
            max_retries=policy.max_retries,
            backoff_base_s=policy.backoff_base_s,
            health_recover_step=policy.health_recover_step,
            health_decay_step=policy.health_decay_step,
        )
        body = self._request_feed(params=params, policy=arxiv_policy)
        return self._parse_feed(body)

    @classmethod
    def _build_params(cls, citation: CitationRecord) -> dict[str, Any]:
        arxiv_id = cls._clean_arxiv_id(citation.arxiv_id)
        if arxiv_id:
            return {
                "id_list": arxiv_id,
                "start": 0,
                "max_results": 1,
            }

        title = str(citation.title or "").strip()
        if not title:
            return {}

        return {
            "search_query": f"all:{title}",
            "start": 0,
            "max_results": 5,
        }

    def _request_feed(self, params: dict[str, Any], policy: RequestPolicy) -> str:
        with self._request_lock:
            now = time.monotonic()
            wait_s = self.min_interval_s - (now - self._last_request_at)
            if wait_s > 0:
                time.sleep(wait_s)

            body = self._request_text(
                self.api_url,
                params,
                policy,
                headers={
                    "Accept": "application/atom+xml",
                },
            )
            self.__class__._last_request_at = time.monotonic()
            return body

    @staticmethod
    def _clean_arxiv_id(value: str) -> str:
        text = str(value or "").strip().lower()
        if not text:
            return ""
        text = re.sub(r"^https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/", "", text)
        text = re.sub(r"\.pdf$", "", text)
        text = re.sub(r"^arxiv:", "", text)
        return text.strip()

    @classmethod
    def _parse_feed(cls, feed_xml: str) -> list[dict[str, Any]]:
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
            version_years = [year] if year is not None else []
            records.append(
                {
                    "title": title,
                    "authors": authors,
                    "venue": "arXiv",
                    "year": year,
                    "doi": "",
                    "arxiv_id": arxiv_id,
                    "url": id_text,
                    "version_timestamps": [],
                    "version_years": version_years,
                }
            )
        return records

    def fetch_version_history(self, arxiv_id: str, policy: RequestPolicy) -> list[dict[str, Any]]:
        """Fetch submission version history for an arXiv paper.

        Returns a list of dicts: [{"version": "v1", "date": "2017-06-12", "year": 2017}, ...]
        """
        base_id = self._clean_arxiv_id(arxiv_id)
        if not base_id:
            return []
        base_id = re.sub(r"v\d+$", "", base_id)

        url = f"https://arxiv.org/abs/{base_id}"
        try:
            html = self._request_feed(params={}, policy=policy) if False else ""
            with self._request_lock:
                now = time.monotonic()
                wait_s = self.min_interval_s - (now - self._last_request_at)
                if wait_s > 0:
                    time.sleep(wait_s)
                html = self._request_text(url, {}, policy)
                self.__class__._last_request_at = time.monotonic()
        except Exception:
            return []

        return self._parse_version_history(html)

    @staticmethod
    def _parse_version_history(html: str) -> list[dict[str, Any]]:
        versions: list[dict[str, Any]] = []
        pattern = re.compile(
            r"\[v(\d+)\]\s+\w+,\s+(\d{1,2})\s+(\w+)\s+(\d{4})"
        )
        month_map = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        }
        for match in pattern.finditer(html):
            version_num = int(match.group(1))
            day = int(match.group(2))
            month_str = match.group(3).lower()[:3]
            year = int(match.group(4))
            month = month_map.get(month_str, 1)
            date_str = f"{year:04d}-{month:02d}-{day:02d}"
            versions.append({
                "version": f"v{version_num}",
                "date": date_str,
                "year": year,
            })
        return versions
