from __future__ import annotations

import re

from packages.core.models import CitationRecord

from .base import BaseConnector, RequestPolicy


class ACLAnthologyConnector(BaseConnector):
    name = "acl_anthology"
    ttl_s = 60 * 60 * 24

    def search(self, citation: CitationRecord, policy: RequestPolicy) -> list[dict[str, object]]:
        query = citation.title or citation.raw_text
        if not query:
            return []
        html = self._request_text(
            "https://aclanthology.org/search/",
            {"q": query},
            policy,
        )
        return self._parse_results(html)

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
