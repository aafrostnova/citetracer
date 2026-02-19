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

    def search(self, citation: CitationRecord, policy: RequestPolicy) -> list[dict]:
        del policy
        query = normalize_text(citation.title or citation.raw_text)
        if not query or not self.mirror_path.exists():
            return []
        # Stream the file line-by-line to avoid loading the entire mirror into
        # memory.  This trades per-query I/O for a bounded memory footprint; for
        # very high query rates consider building a lightweight SQLite index.
        candidates = []
        with self.mirror_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                title = normalize_text(str(record.get("title", "")))
                if not title:
                    continue
                if query in title or title in query:
                    candidates.append(record)
                if len(candidates) >= 5:
                    break
        return candidates
