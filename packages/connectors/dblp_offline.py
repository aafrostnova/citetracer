from __future__ import annotations

import json
from pathlib import Path

from packages.core.models import CitationRecord
from packages.core.normalize import normalize_text

from .base import BaseConnector, RequestPolicy


class DblpOfflineConnector(BaseConnector):
    name = "dblp_offline"
    ttl_s = 60 * 60 * 6

    def __init__(self, mirror_path: str | Path) -> None:
        self.mirror_path = Path(mirror_path)
        self._records = self._load_records()

    def _load_records(self) -> list[dict]:
        if not self.mirror_path.exists():
            return []
        records = []
        for line in self.mirror_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records

    def search(self, citation: CitationRecord, policy: RequestPolicy) -> list[dict]:
        del policy
        query = normalize_text(citation.title or citation.raw_text)
        if not query:
            return []
        candidates = []
        for record in self._records:
            title = normalize_text(str(record.get("title", "")))
            if not title:
                continue
            if query in title or title in query:
                # Ensure volume/pages/publisher are surfaced if present in mirror
                record.setdefault("volume", str(record.get("volume", "") or ""))
                record.setdefault("pages", str(record.get("pages", "") or ""))
                record.setdefault("publisher", str(record.get("publisher", "") or ""))
                candidates.append(record)
            if len(candidates) >= 5:
                break
        return candidates
