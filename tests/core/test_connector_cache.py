from __future__ import annotations

import tempfile
import unittest

from packages.connectors.base import BaseConnector, ConnectorOrchestrator, RequestPolicy, SQLiteCache
from packages.core.models import CitationRecord


class _CountingConnector(BaseConnector):
    name = "counting"

    def __init__(self) -> None:
        self.calls = 0

    def search(self, citation: CitationRecord, policy: RequestPolicy):
        del policy
        self.calls += 1
        return [{"title": citation.title, "authors": [], "venue": "", "year": citation.year, "doi": "", "arxiv_id": "", "url": ""}]


class ConnectorCacheTests(unittest.TestCase):
    def test_cache_prevents_duplicate_call(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            connector = _CountingConnector()
            cache = SQLiteCache(f"{tempdir}/cache.sqlite")
            orchestrator = ConnectorOrchestrator([connector], cache)
            citation = CitationRecord(citation_id="c1", title="A Paper", year=2024)

            orchestrator.query(citation)
            orchestrator.query(citation)

            self.assertEqual(connector.calls, 1)


if __name__ == "__main__":
    unittest.main()
