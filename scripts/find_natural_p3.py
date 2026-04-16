"""Find real P3 (cross-candidate coverage) citations from connector behavior.

For each seed citation, queries a fixed set of connectors and checks:
  1. NO single connector returns a candidate where ALL citation fields match.
  2. Across ALL connectors, EVERY field the citation provides IS matched by
     at least one connector's candidate.

If both hold → this is a natural P3 sample.

Usage:
    python scripts/find_natural_p3.py \
        --seeds data/bib/harvested_seeds_clean.json \
        --out data/synthetic_data/v2/P3_candidates.json \
        --max 200 \
        --workers 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from packages.connectors.base import BaseConnector, ConnectorOrchestrator, RequestPolicy, SQLiteCache, SourceHealth
from packages.connectors.crossref import CrossrefConnector
from packages.connectors.dblp_online import DBLPOnlineConnector
from packages.connectors.openalex import OpenAlexConnector
from packages.connectors.semanticscholar import SemanticScholarConnector
from packages.connectors.arxiv import ArxivConnector
from packages.core.adjudicate import _build_comparison, _is_direct_valid, _has_soft_discrepancies
from packages.core.agents import compare_fields
from packages.core.matching import build_candidate_match, collect_candidates
from packages.core.models import CitationRecord

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Fields we check for coverage. These match what the P3 fallback in
# cascading_agents.py checks: venue, year, doi, arxiv_id.
# We ALSO check pages/volume/publisher/location for richer analysis.
_COVERAGE_FIELDS = ("venue", "year", "doi", "arxiv_id", "pages", "volume", "publisher", "location")

# Match statuses that count as "covered"
_MATCH_STATUSES = {"match", "both_missing", "reference_missing"}


def _make_citation(seed: dict) -> CitationRecord:
    return CitationRecord(
        citation_id=seed.get("citation_id", ""),
        title=seed.get("title", ""),
        authors=seed.get("authors", []),
        venue=seed.get("venue", ""),
        year=seed.get("year"),
        doi=seed.get("doi", ""),
        arxiv_id=seed.get("arxiv_id", ""),
        url=seed.get("url", ""),
        volume=seed.get("volume", ""),
        pages=seed.get("pages", ""),
        publisher=seed.get("publisher", ""),
        location=seed.get("location", ""),
    )


def _ref_fields_with_value(citation: CitationRecord) -> set[str]:
    """Fields the citation actually provides (non-empty)."""
    fields = set()
    for f in _COVERAGE_FIELDS:
        val = getattr(citation, f, None)
        if val and str(val).strip():
            fields.add(f)
    return fields


def _analyze_one(
    seed: dict,
    connectors: list[BaseConnector],
    policy: RequestPolicy,
) -> dict | None:
    """Query all connectors for one citation. Return P3 analysis or None if not P3."""
    citation = _make_citation(seed)
    ref_fields = _ref_fields_with_value(citation)
    if len(ref_fields) < 2:
        return None  # Need at least 2 optional fields for P3 to be meaningful

    per_connector: dict[str, dict] = {}
    any_direct_valid = False
    union_matched: set[str] = set()

    for conn in connectors:
        try:
            records = conn.search(citation, policy)
        except Exception:
            records = []

        if not records:
            per_connector[conn.name] = {"found": False, "matched_fields": [], "status": {}}
            continue

        # Build candidates and find best match
        best_result = None
        best_score = -1
        for rec in records:
            cand = build_candidate_match(citation, conn.name, rec)
            result = compare_fields(
                citation, cand,
                _build_comparison, _is_direct_valid, _has_soft_discrepancies,
            )
            # Score: count matched fields
            matched = set()
            for f in _COVERAGE_FIELDS:
                status = result.field_status.get(f, {}).get("status", "")
                if status in _MATCH_STATUSES:
                    matched.add(f)
            score = len(matched & ref_fields)
            if score > best_score:
                best_score = score
                best_result = result
                best_matched = matched

        if best_result is None:
            per_connector[conn.name] = {"found": False, "matched_fields": [], "status": {}}
            continue

        if best_result.signal == "direct_valid":
            any_direct_valid = True

        union_matched |= (best_matched & ref_fields)

        field_statuses = {}
        for f in _COVERAGE_FIELDS:
            field_statuses[f] = best_result.field_status.get(f, {}).get("status", "n/a")

        per_connector[conn.name] = {
            "found": True,
            "signal": best_result.signal,
            "matched_fields": sorted(best_matched & ref_fields),
            "missing_fields": sorted(ref_fields - best_matched),
            "status": field_statuses,
        }

    # P3 check:
    # - NO single connector has direct_valid
    # - EVERY ref field is matched by at least one connector
    is_p3 = (not any_direct_valid) and (ref_fields <= union_matched)

    if not is_p3:
        return None

    return {
        "citation": seed,
        "ref_fields_with_value": sorted(ref_fields),
        "union_matched": sorted(union_matched),
        "any_direct_valid": any_direct_valid,
        "per_connector": per_connector,
        "is_p3": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", required=True, help="Input seeds JSON")
    parser.add_argument("--out", required=True, help="Output P3 candidates JSON")
    parser.add_argument("--max", type=int, default=200, help="Stop after finding this many P3 samples")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--cache-path", default="data/cache/p3_discovery_cache.sqlite")
    parser.add_argument("--sleep", type=float, default=0.3, help="Seconds between per-citation queries")
    args = parser.parse_args()

    seeds = json.loads(Path(args.seeds).read_text())
    # Prioritize seeds with most optional fields filled (higher chance of P3)
    seeds.sort(key=lambda s: sum(1 for f in _COVERAGE_FIELDS if s.get(f)), reverse=True)
    print(f"Loaded {len(seeds)} seeds (sorted by field richness)", flush=True)

    # Setup connectors
    policy = RequestPolicy()
    connectors: list[BaseConnector] = [
        DBLPOnlineConnector(),
        ArxivConnector(),
    ]
    conn_names = [c.name for c in connectors]
    print(f"Connectors: {conn_names}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    found: list[dict] = []
    scanned = 0

    pbar = tqdm(total=args.max, desc="Finding P3s", unit="found", dynamic_ncols=True) if tqdm else None

    for seed in seeds:
        if len(found) >= args.max:
            break
        scanned += 1
        result = _analyze_one(seed, connectors, policy)
        if result is not None:
            found.append(result)
            if pbar:
                pbar.update(1)
                pbar.set_postfix({"scanned": scanned, "hit_rate": f"{len(found)/scanned:.1%}"})
            # Incremental write
            out_path.write_text(json.dumps(found, indent=2, ensure_ascii=False))
        else:
            if pbar:
                pbar.set_postfix({"scanned": scanned, "hit_rate": f"{len(found)/scanned:.1%}" if scanned else "0%"})

        time.sleep(args.sleep)

    if pbar:
        pbar.close()

    # Final write
    out_path.write_text(json.dumps(found, indent=2, ensure_ascii=False))

    print(f"\n=== Results ===")
    print(f"Scanned: {scanned}")
    print(f"P3 found: {len(found)}")
    print(f"Hit rate: {len(found)/scanned:.1%}" if scanned else "n/a")

    # Field coverage analysis
    if found:
        from collections import Counter
        field_gaps = Counter()
        for r in found:
            for conn_name, info in r["per_connector"].items():
                for f in info.get("missing_fields", []):
                    field_gaps[f"{conn_name}:{f}"] += 1
        print(f"\nTop field gaps (connector:field → how often that connector misses it):")
        for gap, count in field_gaps.most_common(15):
            print(f"  {gap:35s} {count}")


if __name__ == "__main__":
    main()
