from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from packages.core.models import CitationRecord

from .base import BaseConnector, RequestPolicy


class GovInfoConnector(BaseConnector):
    name = "govinfo"
    ttl_s = 60 * 60 * 12

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = (api_key or "").strip()

    def search(self, citation: CitationRecord, policy: RequestPolicy) -> list[dict[str, Any]]:
        if not self.api_key:
            return []
        query = citation.title or citation.raw_text or citation.venue
        if not query:
            return []
        payload = self._post_json(
            "https://api.govinfo.gov/search",
            {
                "query": query,
                "pageSize": 5,
                "offsetMark": "*",
                "sorts": [{"field": "score", "sortOrder": "DESC"}],
            },
            policy,
        )
        records: list[dict[str, Any]] = []
        for item in payload.get("results", []) or []:
            title = str(
                item.get("title")
                or item.get("packageTitle")
                or item.get("documentTitle")
                or ""
            )
            url = str(item.get("url") or item.get("packageLink") or item.get("detailsLink") or "")
            venue = str(item.get("collectionName") or item.get("collectionCode") or "GovInfo")
            year = _extract_year(item)
            if not title and not url:
                continue
            records.append(
                {
                    "title": title,
                    "authors": [],
                    "venue": venue,
                    "year": year,
                    "doi": "",
                    "arxiv_id": "",
                    "url": url,
                }
            )
        return records

    def _post_json(self, url: str, payload: dict[str, Any], policy: RequestPolicy) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        target = f"{url}?api_key={self.api_key}"
        request = Request(
            target,
            data=body,
            headers={
                "User-Agent": "citation-checker/1.0",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(policy.max_retries + 1):
            try:
                with urlopen(request, timeout=policy.timeout_s) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt >= policy.max_retries:
                    break
        raise RuntimeError(f"{self.name} request failed: {last_error}")


def _extract_year(item: dict[str, Any]) -> int | None:
    for key in ("publishDate", "dateIssued", "lastModified", "date"):
        value = str(item.get(key, "") or "")
        for token in value.replace("/", "-").split("-"):
            if token.isdigit() and len(token) == 4:
                return int(token)
    return None
