from __future__ import annotations

import unittest

from packages.connectors.arxiv import ArxivConnector
from packages.connectors.base import RequestPolicy
from packages.core.models import CitationRecord


SAMPLE_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>https://arxiv.org/abs/1706.03762v5</id>
    <published>2017-06-12T17:57:00Z</published>
    <title>Attention Is All You Need</title>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
  </entry>
</feed>
"""


class _RecordingArxivConnector(ArxivConnector):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, object]] = []

    def _request_feed(self, params: dict[str, object], policy: RequestPolicy) -> str:
        del policy
        self.calls.append(dict(params))
        return SAMPLE_FEED


class ArxivConnectorTests(unittest.TestCase):
    def test_arxiv_id_queries_use_id_list(self) -> None:
        connector = _RecordingArxivConnector()
        citation = CitationRecord(citation_id="c1", arxiv_id="https://arxiv.org/abs/1706.03762v5")

        records = connector.search(citation, RequestPolicy())

        self.assertEqual(len(records), 1)
        self.assertEqual(connector.calls, [{"id_list": "1706.03762v5", "start": 0, "max_results": 1}])
        self.assertEqual(records[0]["arxiv_id"], "1706.03762v5")
        self.assertEqual(records[0]["version_years"], [2017])

    def test_title_queries_use_search_query(self) -> None:
        connector = _RecordingArxivConnector()
        citation = CitationRecord(citation_id="c2", title="Attention Is All You Need")

        connector.search(citation, RequestPolicy())

        self.assertEqual(
            connector.calls,
            [{"search_query": "all:Attention Is All You Need", "start": 0, "max_results": 5}],
        )


if __name__ == "__main__":
    unittest.main()
