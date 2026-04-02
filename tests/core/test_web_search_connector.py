from __future__ import annotations

import tempfile
import unittest

from packages.connectors import default_orchestrator
from packages.connectors.base import RequestPolicy
from packages.connectors.google_search import WebSearchConnector
from packages.core.models import CitationRecord


class WebSearchConnectorTests(unittest.TestCase):
    def test_serpapi_and_tavily_share_normalized_schema(self) -> None:
        citation = CitationRecord(
            citation_id="c1",
            title="A Paper",
            authors=["Smith, John", "Doe, Jane"],
            venue="ICML",
            year=2024,
        )
        policy = RequestPolicy()

        serpapi = WebSearchConnector(provider="serpapi", serpapi_key="serp-key")
        serpapi._request_json = lambda url, params, policy, headers=None: {
            "organic_results": [
                {
                    "title": "A Paper",
                    "link": "https://example.org/paper",
                    "snippet": "John Smith and Jane Doe, ICML 2024, doi:10.1234/example",
                    "source": "Example",
                    "date": "2024",
                }
            ]
        }

        tavily = WebSearchConnector(provider="tavily", tavily_api_key="tvly-key")
        tavily._request_json_post = lambda url, payload, policy, headers=None: {
            "results": [
                {
                    "title": "A Paper",
                    "url": "https://openreview.net/forum?id=abc123",
                    "content": "John Smith and Jane Doe. ICML 2024. arXiv:2401.12345",
                    "score": 0.91,
                    "favicon": "https://openreview.net/favicon.ico",
                }
            ]
        }

        serpapi_record = serpapi.search(citation, policy)[0]
        tavily_record = tavily.search(citation, policy)[0]

        expected_query = "A Paper Smith John Doe Jane ICML 2024"
        for record in (serpapi_record, tavily_record):
            self.assertEqual(record["title"], "")
            self.assertEqual(record["authors"], [])
            self.assertIsNone(record["venue"])
            self.assertIsNone(record["year"])
            self.assertEqual(record["search_query"], expected_query)
            self.assertIn("search_result_text", record)
            self.assertIn("search_result_json", record)
            self.assertIn("heuristic_doi", record)
            self.assertIn("heuristic_arxiv_id", record)

        self.assertEqual(serpapi_record["url"], "https://example.org/paper")
        self.assertEqual(serpapi_record["search_source"], "Example")
        self.assertEqual(serpapi_record["heuristic_doi"], "10.1234/example")

        self.assertEqual(tavily_record["url"], "https://openreview.net/forum?id=abc123")
        self.assertEqual(tavily_record["search_source"], "openreview.net")
        self.assertEqual(tavily_record["heuristic_arxiv_id"], "2401.12345")
        self.assertEqual(tavily_record["search_favicon"], "https://openreview.net/favicon.ico")
        self.assertEqual(tavily_record["search_score"], 0.91)

    def test_old_google_search_enabled_source_alias_keeps_web_search_connector(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            orchestrator = default_orchestrator(
                cache_path=f"{tempdir}/cache.sqlite",
                dblp_mirror_path=f"{tempdir}/dblp.jsonl",
                enabled_sources=("google_search",),
                web_search_provider="serpapi",
                serpapi_key="serp-key",
            )

        self.assertEqual([connector.name for connector in orchestrator.connectors], ["web_search"])


if __name__ == "__main__":
    unittest.main()
