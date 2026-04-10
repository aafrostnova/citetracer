#!/usr/bin/env python3
"""Test web_search connector on specific citations from a report.

Usage:
  python scripts/test_websearch_only.py \
    --report artifacts/icml2026_genresult \
    --parsed-citations artifacts/parsed_citations/icml2026_genresult.json/A_Generalization_Result_for_Convergence_in_Learning-to-Optimize.parsed_citations.json \
    --ids pdf-ref:8 pdf-ref:16 pdf-ref:18 pdf-ref:24 pdf-ref:27 pdf-ref:31 pdf-ref:40 pdf-ref:43 pdf-ref:55 pdf-ref:67
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from packages.core.models import CitationRecord
from packages.connectors.google_search import WebSearchConnector, _build_query
from packages.connectors.base import RequestPolicy


def _load_citations(parsed_path: str) -> dict[str, CitationRecord]:
    data = json.loads(Path(parsed_path).read_text())
    records: dict[str, CitationRecord] = {}
    for item in data.get("citations", []):
        cid = item.get("citation_id", "")
        records[cid] = CitationRecord(
            citation_id=cid,
            title=item.get("title", ""),
            authors=item.get("authors", []),
            venue=item.get("venue", ""),
            year=item.get("year"),
            volume=item.get("volume", ""),
            pages=item.get("pages", ""),
            publisher=item.get("publisher", ""),
            doi=item.get("doi", ""),
            arxiv_id=item.get("arxiv_id", ""),
            url=item.get("url", ""),
            raw_text=item.get("raw_text", ""),
        )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--parsed-citations", required=True)
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--tavily-api-key", default="")
    parser.add_argument("--config-path", default="config.json")
    args = parser.parse_args()

    # Load config for API keys
    cfg = {}
    if Path(args.config_path).exists():
        cfg = json.loads(Path(args.config_path).read_text())
    conn_cfg = cfg.get("connectors", {})
    tavily_key = args.tavily_api_key or conn_cfg.get("tavily_api_key", "")

    if not tavily_key:
        print("ERROR: No tavily API key found. Provide --tavily-api-key or put it in config.json")
        return

    connector = WebSearchConnector(
        provider="tavily",
        tavily_api_key=tavily_key,
    )
    policy = RequestPolicy()

    citations = _load_citations(args.parsed_citations)

    for cid in args.ids:
        citation = citations.get(cid)
        if not citation:
            print(f"\n=== {cid}: NOT FOUND in parsed citations ===")
            continue

        query = _build_query(citation)
        print(f"\n{'='*80}")
        print(f"=== {cid}: {citation.title[:80]}")
        print(f"  Query: {query[:200]}")
        print(f"  Authors: {citation.authors}")
        print(f"  Venue: {citation.venue}, Year: {citation.year}")

        try:
            results = connector.search(citation, policy)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        if not results:
            print("  => NO RESULTS")
            continue

        print(f"  => {len(results)} result(s):")
        for i, rec in enumerate(results):
            title = rec.get("title", "") or rec.get("search_result_text", "")[:100]
            url = rec.get("url", "")
            snippet = rec.get("snippet", "")[:150]
            ext = rec.get("extractor_agent_result", {})
            print(f"  [{i}] title: {title[:120]}")
            print(f"      url: {url}")
            if ext:
                print(f"      extracted: title={ext.get('title','')[:60]} venue={ext.get('venue','')} year={ext.get('year','')}")
            if snippet:
                print(f"      snippet: {snippet[:120]}")


if __name__ == "__main__":
    main()
