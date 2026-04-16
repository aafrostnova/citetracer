"""Quick test script for WebSearchConnector.

Usage:
    python scripts/test_web_search.py "title" --authors "A,B" --venue "..." --year 2024
    python scripts/test_web_search.py --citation-id pdf-ref:6 \
        --input-file artifacts/iclr_2026_reference_extracts_v4/A_superpersuasive_autonomous_policy_debating_system.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from packages.connectors.google_search import WebSearchConnector
from packages.connectors.base import RequestPolicy
from packages.core.models import CitationRecord
from apps.pdf_checker.config import load_pdf_checker_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("title", nargs="?", help="Citation title (or use --input-file + --citation-id)")
    parser.add_argument("--authors", default="", help="Comma-separated authors")
    parser.add_argument("--venue", default="", help="Venue")
    parser.add_argument("--year", type=int, default=None, help="Year")
    parser.add_argument("--doi", default="", help="DOI")
    parser.add_argument("--url", default="", help="URL")
    parser.add_argument("--input-file", help="Load citation from a verification input JSON file")
    parser.add_argument("--citation-id", help="Citation ID inside --input-file")
    parser.add_argument("--max-results", type=int, default=5, help="Show first N results")
    parser.add_argument("--full-content", action="store_true", help="Show full raw_content for each result")
    args = parser.parse_args()

    # Build citation
    if args.input_file:
        with open(args.input_file) as f:
            data = json.load(f)
        target = None
        for c in data.get("citations", []):
            if c.get("citation_id") == args.citation_id:
                target = c
                break
        if not target:
            print(f"Citation {args.citation_id} not found in {args.input_file}", file=sys.stderr)
            sys.exit(1)
        citation = CitationRecord(
            citation_id=target.get("citation_id", "test"),
            title=target.get("title", ""),
            authors=target.get("authors", []),
            venue=target.get("venue", ""),
            year=target.get("year"),
            doi=target.get("doi", ""),
            arxiv_id=target.get("arxiv_id", ""),
            url=target.get("url", ""),
        )
    else:
        if not args.title:
            parser.error("Provide a title or use --input-file + --citation-id")
        citation = CitationRecord(
            citation_id="test",
            title=args.title,
            authors=[a.strip() for a in args.authors.split(",") if a.strip()],
            venue=args.venue,
            year=args.year,
            doi=args.doi,
            url=args.url,
        )

    print("=" * 70)
    print("CITATION INPUT:")
    print(f"  title: {citation.title}")
    print(f"  authors: {citation.authors}")
    print(f"  venue: {citation.venue}")
    print(f"  year: {citation.year}")
    print(f"  doi: {citation.doi}")
    print(f"  url: {citation.url}")
    print()

    # Build connector from config
    cfg = load_pdf_checker_config()
    conn = WebSearchConnector(
        provider=cfg.connectors.web_search_provider,
        api_key=cfg.connectors.google_api_key,
        cse_id=cfg.connectors.google_cse_id,
        serpapi_key=cfg.connectors.serpapi_key,
        tavily_api_key=cfg.connectors.tavily_api_key,
    )
    print(f"Provider: {conn._resolve_provider()}")
    print()

    policy = RequestPolicy(timeout_s=30)
    results = conn.search(citation, policy)

    print(f"=" * 70)
    print(f"RESULTS: {len(results)} hits")
    print(f"=" * 70)

    for i, r in enumerate(results[: args.max_results], 1):
        print(f"\n--- Result {i} ---")
        print(f"  title:   {r.get('title', '')[:100]}")
        print(f"  authors: {r.get('authors', '')}")
        print(f"  venue:   {r.get('venue', '')}")
        print(f"  year:    {r.get('year', '')}")
        print(f"  url:     {r.get('url', '')}")
        print(f"  doi:     {r.get('doi', '')}")
        print(f"  arxiv:   {r.get('arxiv_id', '')}")
        # snippet / raw_content from raw_record
        raw = r.get("raw_record", {}) or {}
        snippet = raw.get("snippet", "") or raw.get("search_result_text", "")
        if snippet:
            print(f"  snippet: {snippet[:200]}...")
        if args.full_content:
            raw_content = raw.get("raw_content", "")
            if raw_content:
                print(f"  raw_content ({len(raw_content)} chars):")
                print(raw_content[:1500])
                print("  ...")

    print()
    print("=" * 70)
    # Dump full JSON for inspection
    print("FULL FIRST RESULT (JSON):")
    if results:
        print(json.dumps(results[0], indent=2, default=str)[:2000])


if __name__ == "__main__":
    main()
