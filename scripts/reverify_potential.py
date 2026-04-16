"""Re-verify only POTENTIAL_REFERENCE citations from an existing report.

Useful when you've improved the verifier (new guards, prompt fixes, etc.) and
want to re-check the borderline cases without re-running the entire pipeline.

Usage:
    python scripts/reverify_potential.py \
        --report artifacts/2026COLM_MetaState_report.json \
        --parsed-citations artifacts/parsed_citations/2026COLM_MetaState.parsed_citations.json \
        --out artifacts/2026COLM_MetaState_report.json   # overwrite same file

The script:
  1. Loads `report.json` and finds citation_ids whose verdict is POTENTIAL_REFERENCE.
  2. Loads `parsed_citations.json` to recover the full input CitationRecord
     for each id (title/authors/year/doi/...).
  3. Builds a verifier (cascading 3-agent + secondary verifier + guards).
  4. Re-runs verify_citation on each POTENTIAL.
  5. Patches those entries in the report and writes it back.

Citations with other verdicts (VALID / FAKE / INSUFFICIENT) are kept as-is.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apps.pdf_checker.config import load_pdf_checker_config
from apps.pdf_checker.run import _build_verifier
from packages.core.models import (
    CitationRecord,
    ExtractionQuality,
    citation_verdict_to_dict,
)


def _payload_to_record(payload: dict[str, Any]) -> CitationRecord:
    """Reconstruct a CitationRecord from a parsed_citations.json entry."""
    year_value = payload.get("year")
    try:
        year = int(year_value) if year_value is not None else None
    except (TypeError, ValueError):
        year = None
    return CitationRecord(
        citation_id=str(payload.get("citation_id", "")),
        raw_text=str(payload.get("raw_text", "")),
        title=str(payload.get("title", "")),
        authors=[str(item) for item in (payload.get("authors") or [])],
        venue=str(payload.get("venue", "")),
        year=year,
        doi=str(payload.get("doi", "")),
        arxiv_id=str(payload.get("arxiv_id", "")),
        url=str(payload.get("url", "")),
        volume=str(payload.get("volume", "")),
        pages=str(payload.get("pages", "")),
        publisher=str(payload.get("publisher", "")),
        location=str(payload.get("location", "")),
        source_span=str(payload.get("source_span", "")),
        provenance=payload.get("provenance") or {},
        parsed_fields=payload.get("parsed_fields") or {},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="Path to existing report JSON.")
    parser.add_argument(
        "--parsed-citations",
        required=True,
        help="Path to parsed_citations JSON (the artifact saved by --save-parsed-citations).",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output report path. Empty = overwrite --report.",
    )
    parser.add_argument(
        "--target-verdict",
        default="POTENTIAL_REFERENCE",
        help="Verdict label to re-verify (default: POTENTIAL_REFERENCE). Use ALL to re-verify everything.",
    )
    parser.add_argument(
        "--citation-id",
        action="append",
        default=[],
        help="Re-verify only this citation_id (can be passed multiple times). Overrides --target-verdict.",
    )
    args = parser.parse_args()

    report_path = Path(args.report)
    parsed_path = Path(args.parsed_citations)
    out_path = Path(args.out) if args.out else report_path

    report = json.loads(report_path.read_text(encoding="utf-8"))
    parsed = json.loads(parsed_path.read_text(encoding="utf-8"))

    # Build a citation_id → CitationRecord lookup from parsed_citations
    parsed_records: dict[str, CitationRecord] = {}
    for entry in parsed.get("citations", []):
        if not isinstance(entry, dict):
            continue
        rec = _payload_to_record(entry)
        if rec.citation_id:
            parsed_records[rec.citation_id] = rec

    # Find target citation_ids in the report
    if args.citation_id:
        # Explicit list of citation_ids overrides verdict filter
        explicit_ids = set(args.citation_id)
        report_ids = {c.get("citation_id", "") for c in report.get("citations", []) if isinstance(c, dict)}
        target_ids = [cid for cid in args.citation_id if cid in report_ids]
        missing = explicit_ids - set(target_ids)
        if missing:
            print(f"WARNING: citation_ids not in report: {sorted(missing)}")
    else:
        target = args.target_verdict.upper()
        target_ids = []
        for c in report.get("citations", []):
            if not isinstance(c, dict):
                continue
            verdict = str(c.get("verdict", "")).upper()
            if target == "ALL" or verdict == target:
                target_ids.append(c.get("citation_id", ""))

    if not target_ids:
        print(f"No citations to re-verify in {report_path}")
        return

    label = "explicit" if args.citation_id else f"verdict={args.target_verdict.upper()}"
    print(f"Found {len(target_ids)} citations to re-verify ({label})")
    missing_in_parsed = [cid for cid in target_ids if cid not in parsed_records]
    if missing_in_parsed:
        print(f"WARNING: {len(missing_in_parsed)} citations missing from parsed_citations: {missing_in_parsed[:5]}")

    cfg = load_pdf_checker_config()
    # build_verifier expects an args namespace; create a minimal one
    fake_args = argparse.Namespace(
        cache_path="",
        dblp_mirror_path="",
        dblp_sqlite_path="",
        semantic_scholar_api_key="",
        offline_only=False,
        enable_llm_source_router=False,
        source_router_max_sources=0,
        connector_workers=8,
        citation_workers=4,
    )
    verifier = _build_verifier(cfg, fake_args, verbose=True)

    paper_title = report.get("paper_id", "").replace("_", " ")

    # Helper: recompute summary + write report to disk
    def _flush_report():
        from collections import Counter
        summary_counts: Counter = Counter()
        for c in report.get("citations", []):
            v = str(c.get("verdict", "")).lower()
            summary_counts["total_citations"] += 1
            if v == "valid":
                summary_counts["valid"] += 1
            elif v == "potential_reference":
                summary_counts["potential_reference"] += 1
                summary_counts["flawed_citation"] += 1  # legacy alias
            elif v == "fake_reference":
                summary_counts["fake_reference"] += 1
                summary_counts["suspected_hallucination"] += 1  # legacy alias
            elif v == "insufficient_evidence":
                summary_counts["insufficient_evidence"] += 1
            if c.get("needs_human_review"):
                summary_counts["needs_human_review"] += 1
        report["summary"] = dict(summary_counts)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # Build citation_id → index lookup for fast patching
    _cid_to_idx: dict[str, int] = {}
    for i, c in enumerate(report.get("citations", [])):
        if isinstance(c, dict):
            _cid_to_idx[c.get("citation_id", "")] = i

    # Re-verify target citations in parallel — write after each one
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    patched = 0
    write_lock = threading.Lock()
    citation_workers = getattr(args, "citation_workers", 4) if hasattr(args, "citation_workers") else 4
    # reverify uses fake_args, so read from it
    citation_workers = max(1, int(getattr(fake_args, "citation_workers", 4)))

    # Filter to only those in parsed_records
    valid_targets = [(cid, parsed_records[cid]) for cid in target_ids if cid in parsed_records]

    def _verify_one(item: tuple[int, str, CitationRecord]) -> None:
        nonlocal patched
        idx, cid, record = item
        print(
            f"\n[{idx}/{len(valid_targets)}] {cid} title={record.title[:60]!r}",
            flush=True,
        )
        try:
            verdict = verifier.verify_citation(
                record,
                extraction_quality=ExtractionQuality.UNKNOWN,
                source_paper_title=paper_title,
            )
        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}", flush=True)
            return
        verdict_dict = citation_verdict_to_dict(verdict)

        with write_lock:
            print(
                f"  → verdict={verdict.verdict.value} taxonomy={verdict.taxonomy_subtype}",
                flush=True,
            )
            reason = (verdict.adjudication_reason or "").strip()
            if reason:
                print(f"  reason: {reason[:300]}", flush=True)

            if cid in _cid_to_idx:
                report["citations"][_cid_to_idx[cid]].update(verdict_dict)
                patched += 1
                _flush_report()
                print(f"  (saved to {out_path})", flush=True)

    items = [(idx, cid, rec) for idx, (cid, rec) in enumerate(valid_targets, start=1)]

    if citation_workers <= 1 or len(items) <= 1:
        for item in items:
            _verify_one(item)
    else:
        print(f"Running with {citation_workers} parallel workers", flush=True)
        with ThreadPoolExecutor(max_workers=citation_workers) as pool:
            futures = [pool.submit(_verify_one, item) for item in items]
            for fut in as_completed(futures):
                fut.result()  # propagate exceptions

    print(f"\nDone. Patched {patched} citations total. Output: {out_path}")
    print(f"New summary: {report['summary']}")


if __name__ == "__main__":
    main()
