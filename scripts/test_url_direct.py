"""Quick test script for URLDirectConnector.

Usage:
    python scripts/test_url_direct.py "https://arxiv.org/abs/2106.09685"
    python scripts/test_url_direct.py --doi 10.1145/1234.5678
    python scripts/test_url_direct.py --input-file artifacts/.../paper.json --citation-id pdf-ref:6
    python scripts/test_url_direct.py "https://example.com/blog" --full-content
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from packages.connectors.url_direct import URLDirectConnector
from packages.connectors.base import RequestPolicy
from packages.core.models import CitationRecord
from apps.pdf_checker.config import load_pdf_checker_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url", nargs="?", help="URL to fetch (or use --doi or --input-file)")
    parser.add_argument("--doi", default="", help="DOI (will resolve via doi.org)")
    parser.add_argument("--input-file", help="Load citation from a verification input JSON file")
    parser.add_argument("--citation-id", help="Citation ID inside --input-file")
    parser.add_argument("--no-tavily", action="store_true", help="Disable Tavily fallback to test direct fetch only")
    parser.add_argument("--full-content", action="store_true", help="Show full raw markdown/HTML response")
    parser.add_argument("--timeout", type=int, default=20, help="Request timeout seconds")
    args = parser.parse_args()

    # Resolve URL to fetch
    url = ""
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
        url = (target.get("url") or "").strip()
        doi = (target.get("doi") or "").strip()
        if not url and doi:
            url = f"https://doi.org/{doi}"
        print("=" * 70)
        print(f"Citation {args.citation_id}:")
        print(f"  title: {target.get('title','')}")
        print(f"  url:   {target.get('url','')}")
        print(f"  doi:   {target.get('doi','')}")
        print(f"  arxiv: {target.get('arxiv_id','')}")
        print()
    elif args.doi:
        url = f"https://doi.org/{args.doi}"
    elif args.url:
        url = args.url
    else:
        parser.error("Provide a URL, --doi, or --input-file + --citation-id")

    print(f"Fetching: {url}")
    print()

    # Build connector with optional Tavily
    cfg = load_pdf_checker_config()
    tavily_key = "" if args.no_tavily else (cfg.connectors.tavily_api_key or "").strip()
    print(f"Tavily fallback: {'enabled' if tavily_key else 'DISABLED'}")
    print()

    conn = URLDirectConnector(tavily_api_key=tavily_key)
    policy = RequestPolicy(timeout_s=args.timeout)
    citation = CitationRecord(citation_id="test", url=url)

    # Try direct fetch first to see if it succeeds
    print("=" * 70)
    print("STEP 1: Direct HTTP fetch")
    print("=" * 70)
    direct_body = ""
    direct_error = None
    try:
        direct_body = conn._request_text(
            url,
            {},
            policy,
            headers={"Accept": "text/html,application/xhtml+xml"},
        )
        print(f"  ✓ Direct fetch OK ({len(direct_body)} chars)")
    except Exception as e:
        direct_error = e
        print(f"  ✗ Direct fetch FAILED: {type(e).__name__}: {e}")

    # If direct failed and tavily enabled, try tavily
    if not direct_body and tavily_key:
        print()
        print("=" * 70)
        print("STEP 2: Tavily Extract API fallback")
        print("=" * 70)
        try:
            tavily_record = conn._fetch_via_tavily(url, policy)
            if tavily_record:
                print(f"  ✓ Tavily OK")
                print(f"    title: {tavily_record.get('title', '')}")
                print(f"    year:  {tavily_record.get('year', '')}")
                print(f"    doi:   {tavily_record.get('doi', '')}")
            else:
                print(f"  ✗ Tavily returned empty")
        except Exception as e:
            print(f"  ✗ Tavily FAILED: {type(e).__name__}: {e}")

    # Now run the full search() pipeline
    print()
    print("=" * 70)
    print("STEP 3: URLDirectConnector.search() — full pipeline")
    print("=" * 70)
    try:
        results = conn.search(citation, policy)
        print(f"Records: {len(results)}")
        if results:
            r = results[0]
            print()
            print("EXTRACTED RECORD:")
            print(f"  title:    {r.get('title', '')}")
            print(f"  authors:  {r.get('authors', [])}")
            print(f"  venue:    {r.get('venue', '')}")
            print(f"  year:     {r.get('year', '')}")
            print(f"  doi:      {r.get('doi', '')}")
            print(f"  arxiv_id: {r.get('arxiv_id', '')}")
            print(f"  url:      {r.get('url', '')}")
            print()
            print("FULL RECORD (JSON):")
            print(json.dumps(r, indent=2, default=str)[:2000])
    except Exception as e:
        print(f"  ✗ search() FAILED: {type(e).__name__}: {e}")

    # Show raw body content if requested
    if args.full_content:
        print()
        print("=" * 70)
        print("RAW CONTENT (first 3000 chars)")
        print("=" * 70)
        if direct_body:
            print("--- Direct HTTP body ---")
            print(direct_body[:3000])
        elif tavily_key:
            try:
                payload = conn._request_json_post(
                    "https://api.tavily.com/extract",
                    {"urls": [url], "extract_depth": "basic", "format": "markdown"},
                    policy,
                    headers={"Authorization": f"Bearer {tavily_key}"},
                )
                if payload.get("results"):
                    raw = payload["results"][0].get("raw_content", "")
                    print("--- Tavily markdown ---")
                    print(raw[:3000])
                elif payload.get("failed_results"):
                    print(f"Tavily failed: {payload['failed_results']}")
            except Exception as e:
                print(f"Could not fetch raw content: {e}")


if __name__ == "__main__":
    main()
