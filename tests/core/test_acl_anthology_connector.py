from __future__ import annotations

import sys
import types
import unittest

from packages.connectors.acl_anthology import ACLAnthologyConnector
from packages.connectors.base import RequestPolicy
from packages.core.models import CitationRecord


class _FakePaper:
    def __init__(self, title: str, authors: list[str], year: int, doi: str, url: str, venue: str) -> None:
        self.title = title
        self.authors = authors
        self.year = year
        self.doi = doi
        self.url = url
        self.parent = types.SimpleNamespace(title=venue)


class _FakeAnthology:
    def __init__(self, datadir=None, verbose=None) -> None:
        del datadir, verbose

    @classmethod
    def from_repo(cls, path=None, verbose=None):
        del path, verbose
        return cls()

    def papers(self):
        return [
            _FakePaper(
                title="Attention Is All You Need",
                authors=["Ashish Vaswani", "Noam Shazeer"],
                year=2017,
                doi="10.18653/v1/n17-3002",
                url="https://aclanthology.org/N17-3002/",
                venue="Proceedings of the Workshop on Machine Translation",
            ),
            _FakePaper(
                title="Completely Different Paper",
                authors=["Jane Doe"],
                year=2018,
                doi="",
                url="https://aclanthology.org/P18-1001/",
                venue="ACL Anthology",
            ),
        ]


class ACLAnthologyConnectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_module = sys.modules.get("acl_anthology")
        sys.modules["acl_anthology"] = types.SimpleNamespace(Anthology=_FakeAnthology)
        ACLAnthologyConnector._official_index_cache.clear()

    def tearDown(self) -> None:
        ACLAnthologyConnector._official_index_cache.clear()
        if self._old_module is None:
            sys.modules.pop("acl_anthology", None)
        else:
            sys.modules["acl_anthology"] = self._old_module

    def test_official_data_search_uses_exact_title_match(self) -> None:
        connector = ACLAnthologyConnector(data_dir="/tmp/acl-data")
        citation = CitationRecord(citation_id="c1", title="Attention Is All You Need")

        records = connector.search(citation, RequestPolicy())

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["title"], "Attention Is All You Need")
        self.assertEqual(records[0]["authors"], ["Ashish Vaswani", "Noam Shazeer"])
        self.assertEqual(records[0]["year"], 2017)
        self.assertEqual(records[0]["doi"], "10.18653/v1/n17-3002")
        self.assertEqual(records[0]["url"], "https://aclanthology.org/N17-3002/")

    def test_fallback_html_scrape_still_works_without_official_config(self) -> None:
        connector = ACLAnthologyConnector()
        connector._request_text = lambda url, params, policy: (
            '<a href="/2022.acl-long.220/">Learned Incremental Representations for Parsing</a>'
        )

        records = connector.search(CitationRecord(citation_id="c2", title="parsing"), RequestPolicy())

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["url"], "https://aclanthology.org/2022.acl-long.220/")


if __name__ == "__main__":
    unittest.main()
