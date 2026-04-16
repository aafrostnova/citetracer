"""Query all connectors for each seed citation and store field-level results.

Produces a "connector response database" — one JSON file where each entry has:
  - The original seed citation
  - Per-connector: best candidate + field-level match status

This database is queried offline (no network) to find P3 candidates under
any connector combination.

Usage:
    python scripts/build_connector_response_db.py \
        --seeds data/bib/harvested_seeds_clean.json \
        --out data/cache/connector_response_db.json \
        --workers 4 --sleep 0.3

Then find P3s (offline, instant):
    python scripts/build_connector_response_db.py \
        --mode find-p3 \
        --db data/cache/connector_response_db.json \
        --connectors dblp_online,arxiv \
        --out data/synthetic_data/v2/P3.json \
        --max 200
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from packages.connectors.base import BaseConnector, RequestPolicy
from packages.connectors.crossref import CrossrefConnector
from packages.connectors.dblp_online import DBLPOnlineConnector
from packages.connectors.openalex import OpenAlexConnector
from packages.connectors.acl_anthology import ACLAnthologyConnector
from packages.connectors.arxiv import ArxivConnector
from packages.core.adjudicate import _build_comparison, _is_direct_valid, _has_soft_discrepancies
from packages.core.agents import compare_fields
from packages.core.matching import build_candidate_match
from packages.core.models import CitationRecord

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

_COVERAGE_FIELDS = ("venue", "year", "doi", "arxiv_id", "pages", "volume", "publisher", "location")
_MATCH_STATUSES = {"match", "both_missing", "reference_missing"}

ALL_CONNECTORS = {
    "dblp_online": DBLPOnlineConnector,
    "crossref": CrossrefConnector,
    "openalex": OpenAlexConnector,
    "acl_anthology": ACLAnthologyConnector,
    "arxiv": ArxivConnector,
}


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


def _query_one_connector(
    citation: CitationRecord,
    connector: BaseConnector,
    policy: RequestPolicy,
) -> dict:
    """Query one connector. Return best candidate's field-level analysis."""
    try:
        records = connector.search(citation, policy)
    except Exception as exc:
        return {"found": False, "error": str(exc)[:200]}

    if not records:
        return {"found": False, "error": None}

    # Find best candidate by matched field count
    best_info = None
    best_score = -1
    for rec in records[:5]:  # cap to top 5
        cand = build_candidate_match(citation, connector.name, rec)
        result = compare_fields(
            citation, cand,
            _build_comparison, _is_direct_valid, _has_soft_discrepancies,
        )
        matched = set()
        field_status = {}
        for f in _COVERAGE_FIELDS:
            status = result.field_status.get(f, {}).get("status", "n/a")
            field_status[f] = status
            if status in _MATCH_STATUSES:
                matched.add(f)
        # Also check title/authors
        for f in ("title", "authors", "year"):
            status = result.field_status.get(f, {}).get("status", "n/a")
            field_status[f] = status

        score = len(matched)
        if score > best_score:
            best_score = score
            best_info = {
                "found": True,
                "signal": result.signal,
                "direct_valid": result.signal == "direct_valid",
                "field_status": field_status,
                "matched_fields": sorted(matched),
                "candidate_title": cand.title[:100],
                "candidate_venue": cand.venue[:60],
                "candidate_year": cand.year,
            }

    return best_info or {"found": False, "error": None}


# ---------------------------------------------------------------------------
# Mode 1: BUILD the database
# ---------------------------------------------------------------------------

def build_db(args) -> None:
    seeds = json.loads(Path(args.seeds).read_text())
    print(f"Loaded {len(seeds)} seeds", flush=True)

    policy = RequestPolicy()
    connector_names = [c.strip() for c in args.connectors.split(",")]
    connectors = {}
    for name in connector_names:
        cls = ALL_CONNECTORS.get(name)
        if cls:
            connectors[name] = cls()
        else:
            print(f"WARNING: unknown connector '{name}', skipping")
    print(f"Connectors: {list(connectors.keys())}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: load existing db if present
    db: list[dict] = []
    done_titles: set[str] = set()
    if out_path.exists() and out_path.stat().st_size > 0:
        try:
            db = json.loads(out_path.read_text())
            for entry in db:
                t = (entry.get("citation", {}).get("title") or "").lower()[:80]
                done_titles.add(t)
            print(f"Resuming: {len(db)} entries already in db", flush=True)
        except json.JSONDecodeError:
            pass

    # Filter to seeds not yet done
    pending = [s for s in seeds
               if (s.get("title") or "").lower()[:80] not in done_titles]
    print(f"Pending: {len(pending)} / {len(seeds)}", flush=True)

    pbar = tqdm(total=len(pending), desc="Querying connectors",
                unit="cite", dynamic_ncols=True) if tqdm else None

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    db_lock = threading.Lock()

    def _process_seed(seed: dict) -> dict:
        citation = _make_citation(seed)
        conn_results = {}
        for name, conn in connectors.items():
            conn_results[name] = _query_one_connector(citation, conn, policy)
            time.sleep(args.sleep)  # rate-limit per connector call INSIDE worker
        return {"citation": seed, "connectors": conn_results}

    workers = max(1, args.workers)
    flush_every = max(10, workers * 2)
    new_count = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_seed, s): s for s in pending}
        for fut in as_completed(futures):
            entry = fut.result()
            title_key = (entry["citation"].get("title") or "").lower()[:80]
            with db_lock:
                if title_key not in done_titles:
                    db.append(entry)
                    done_titles.add(title_key)
                    new_count += 1
                    if new_count % flush_every == 0:
                        out_path.write_text(json.dumps(db, indent=2, ensure_ascii=False))
            if pbar:
                found = sum(1 for v in entry["connectors"].values() if v.get("found"))
                pbar.update(1)
                pbar.set_postfix({"db": len(db), "found": f"{found}/{len(connectors)}"})

        if pbar:
            pbar.update(1)
            found_count = sum(1 for e in db if any(
                v.get("found") for v in e.get("connectors", {}).values()))
            pbar.set_postfix({"db": len(db), "found_by_any": found_count})

    if pbar:
        pbar.close()

    out_path.write_text(json.dumps(db, indent=2, ensure_ascii=False))
    print(f"\nDB saved: {len(db)} entries → {out_path}")

    # Quick stats
    from collections import Counter
    for cname in connectors:
        found = sum(1 for e in db if e["connectors"].get(cname, {}).get("found"))
        dv = sum(1 for e in db if e["connectors"].get(cname, {}).get("direct_valid"))
        print(f"  {cname:20s}  found={found:5d}  direct_valid={dv:5d}")


# ---------------------------------------------------------------------------
# Mode 2: FIND P3 from the database (offline, instant)
# ---------------------------------------------------------------------------

def find_p3(args) -> None:
    db = json.loads(Path(args.db).read_text())
    connector_names = [c.strip() for c in args.connectors.split(",")]
    print(f"DB: {len(db)} entries, checking P3 with connectors: {connector_names}", flush=True)

    p3_results = []

    for entry in db:
        citation = entry["citation"]
        cr = _make_citation(citation)

        # Ref fields with value
        ref_fields = set()
        for f in _COVERAGE_FIELDS:
            val = getattr(cr, f, None)
            if val and str(val).strip():
                ref_fields.add(f)
        if len(ref_fields) < 2:
            continue

        any_direct_valid = False
        union_matched: set[str] = set()
        per_conn_info = {}

        for cname in connector_names:
            info = entry.get("connectors", {}).get(cname, {})
            if not info.get("found"):
                per_conn_info[cname] = {"found": False, "matched": [], "missing": sorted(ref_fields)}
                continue
            if info.get("direct_valid"):
                any_direct_valid = True

            matched = set(info.get("matched_fields", [])) & ref_fields
            union_matched |= matched
            per_conn_info[cname] = {
                "found": True,
                "direct_valid": info.get("direct_valid", False),
                "signal": info.get("signal"),
                "matched": sorted(matched),
                "missing": sorted(ref_fields - matched),
                "field_status": info.get("field_status", {}),
            }

        is_p3 = (not any_direct_valid) and (ref_fields <= union_matched) and len(ref_fields) >= 2

        if is_p3:
            p3_results.append({
                "citation": citation,
                "ref_fields": sorted(ref_fields),
                "union_matched": sorted(union_matched),
                "connectors_used": connector_names,
                "per_connector": per_conn_info,
            })
            if len(p3_results) >= args.max:
                break

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(p3_results, indent=2, ensure_ascii=False))

    print(f"\nP3 found: {len(p3_results)} (scanned {len(db)})")
    print(f"Saved to: {out_path}")

    if p3_results:
        # Show field gap analysis
        from collections import Counter
        gaps = Counter()
        for r in p3_results:
            for cname, info in r["per_connector"].items():
                for f in info.get("missing", []):
                    gaps[f"{cname}:{f}"] += 1
        print(f"\nTop field gaps (connector:field):")
        for gap, count in gaps.most_common(15):
            print(f"  {gap:35s} {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["build", "find-p3"], default="build")
    # Build mode
    parser.add_argument("--seeds", help="Input seeds JSON (build mode)")
    parser.add_argument("--connectors", default="dblp_online,crossref,arxiv",
                        help="Comma-separated connector names")
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=1, help="(reserved)")
    # Find-p3 mode
    parser.add_argument("--db", help="Connector response DB JSON (find-p3 mode)")
    parser.add_argument("--max", type=int, default=200)
    # Shared
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.mode == "build":
        if not args.seeds:
            parser.error("--seeds required for build mode")
        build_db(args)
    else:
        if not args.db:
            parser.error("--db required for find-p3 mode")
        find_p3(args)


if __name__ == "__main__":
    main()
