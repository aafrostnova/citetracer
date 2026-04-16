"""Test synthetic samples against the verification pipeline.

Feeds citation records directly into CitationVerifier and compares
pipeline verdict vs expected ground truth label.

Usage:
    # Test one code
    python scripts/test_synthetic_samples.py \
        --samples data/synthetic_data/v2/R1.json \
        --expected-verdict VALID \
        --max 20 --workers 4

    # Test with meta.json (auto-detects expected label per sample)
    python scripts/test_synthetic_samples.py \
        --samples data/synthetic_data/v2/R1.json \
        --meta data/synthetic_data/v2/meta.json \
        --max 20 --workers 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apps.pdf_checker.config import load_pdf_checker_config
from packages.connectors import default_orchestrator
from packages.core import CitationVerifier
from packages.core.models import CitationRecord, ExtractionQuality, citation_verdict_to_dict

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

_LABEL_TO_VERDICT = {
    "REAL": "VALID",
    "POTENTIAL_HALLUCINATED": "POTENTIAL_REFERENCE",
    "HALLUCINATED": "FAKE_REFERENCE",
}


def _build_verifier(cfg) -> CitationVerifier:
    """Build a verifier identical to apps/pdf_checker/run.py::_build_verifier.

    Mirrors the production pipeline: orchestrator (full connector list with all
    API keys and web search provider), LLM resolver, SecondaryVerifier,
    cascading 3-agent + ExtractorAgent, and VerifiedReferenceCache.
    """
    from pathlib import Path as _Path
    conn_cfg = cfg.connectors

    cache_path = _Path(conn_cfg.cache_path) if conn_cfg.cache_path else _Path(
        "data/cache/test_eval_cache.sqlite"
    )
    dblp_mirror_path = conn_cfg.dblp_mirror_path or ""
    dblp_sqlite_path = str(conn_cfg.dblp_sqlite_path or "").strip()
    semscholar_key = (conn_cfg.semantic_scholar_api_key or "").strip() or None

    orch = default_orchestrator(
        cache_path=cache_path,
        dblp_mirror_path=dblp_mirror_path,
        dblp_sqlite_path=_Path(dblp_sqlite_path) if dblp_sqlite_path else None,
        enabled_sources=conn_cfg.enabled_sources,
        acl_anthology_data_dir=getattr(conn_cfg, "acl_anthology_data_dir", "") or "",
        acl_anthology_repo_path=getattr(conn_cfg, "acl_anthology_repo_path", "") or "",
        semantic_scholar_api_key=semscholar_key,
        govinfo_api_key=getattr(conn_cfg, "govinfo_api_key", "") or None,
        searxng_base_url=getattr(conn_cfg, "searxng_base_url", "") or None,
        ncbi_api_key=getattr(conn_cfg, "ncbi_api_key", "") or None,
        ncbi_email=getattr(conn_cfg, "ncbi_email", "") or None,
        web_search_provider=getattr(conn_cfg, "web_search_provider", "") or None,
        google_api_key=getattr(conn_cfg, "google_api_key", "") or None,
        google_cse_id=getattr(conn_cfg, "google_cse_id", "") or None,
        serpapi_key=getattr(conn_cfg, "serpapi_key", "") or None,
        tavily_api_key=getattr(conn_cfg, "tavily_api_key", "") or None,
        max_workers=8,
    )
    from packages.core.adjudicate import AdjudicationPolicy

    # LLM resolver (Bedrock strict-match) — same as pipeline
    llm_resolver = None
    if cfg.verification_llm.enabled and cfg.verification_llm.provider == "bedrock":
        try:
            from citation_verification_demo import BedrockStrictMatchResolver
            llm_resolver = BedrockStrictMatchResolver(
                model_id=str(cfg.verification_llm.bedrock.model_id),
                region=cfg.verification_llm.bedrock.region,
                bearer_token=cfg.verification_llm.bedrock.bearer_token,
                max_candidates=cfg.verification_llm.max_candidates,
            )
        except Exception as exc:
            print(f"  WARNING: llm_resolver init failed: {exc}", flush=True)

    # Secondary verifier (arxiv / semantic_scholar / openalex / url_direct)
    from packages.connectors.arxiv import ArxivConnector
    from packages.connectors.openalex import OpenAlexConnector
    from packages.connectors.semanticscholar import SemanticScholarConnector
    from packages.connectors.url_direct import URLDirectConnector
    from packages.core.secondary_verification import SecondaryVerifier

    arxiv_conn = semscholar_conn = openalex_conn = url_direct_conn = None
    for conn in orch.connectors:
        if isinstance(conn, ArxivConnector):
            arxiv_conn = conn
        elif isinstance(conn, SemanticScholarConnector):
            semscholar_conn = conn
        elif isinstance(conn, OpenAlexConnector):
            openalex_conn = conn
        elif isinstance(conn, URLDirectConnector):
            url_direct_conn = conn

    secondary_verifier = SecondaryVerifier(
        arxiv_connector=arxiv_conn,
        semantic_scholar_connector=semscholar_conn,
        openalex_connector=openalex_conn,
        url_direct_connector=url_direct_conn,
    )

    # Cascading 3-agent + ExtractorAgent (Bedrock) — same as pipeline
    valid_agent = potential_agent = hallucinated_agent = extractor_agent = None
    if cfg.verification_llm.enabled and cfg.verification_llm.provider == "bedrock":
        try:
            from citation_verification_demo import _build_bedrock_client
            from packages.core.bedrock_agents import (
                BedrockHallucinatedAgent,
                BedrockPotentialAgent,
                BedrockValidAgent,
            )
            from packages.core.extractor_agent import BedrockExtractorAgent
            bcfg = cfg.verification_llm.bedrock
            bedrock_client = _build_bedrock_client(
                region=bcfg.region,
                bearer_token=bcfg.bearer_token,
            )
            model_id = str(bcfg.model_id)
            valid_agent = BedrockValidAgent(bedrock_client, model_id)
            potential_agent = BedrockPotentialAgent(bedrock_client, model_id)
            hallucinated_agent = BedrockHallucinatedAgent(bedrock_client, model_id)
            extractor_agent = BedrockExtractorAgent(bedrock_client, model_id)
            print(f"  Cascading 3-agent + extractor enabled (model={model_id})", flush=True)
        except Exception as exc:
            print(f"  WARNING: cascading agents failed to init: {exc}", flush=True)
    else:
        print("  Cascading agents disabled (verification_llm not enabled/bedrock)", flush=True)

    # Verified reference cache (reuses connector_cache.sqlite storage)
    from packages.connectors.base import VerifiedReferenceCache
    verified_ref_cache = VerifiedReferenceCache(cache_path)
    print(f"  Verified reference cache: {cache_path}", flush=True)

    return CitationVerifier(
        orchestrator=orch,
        policy=AdjudicationPolicy(),
        llm_resolver=llm_resolver,
        secondary_verifier=secondary_verifier,
        valid_agent=valid_agent,
        potential_agent=potential_agent,
        hallucinated_agent=hallucinated_agent,
        extractor_agent=extractor_agent,
        verified_ref_cache=verified_ref_cache,
    )


def _make_citation(sample: dict) -> CitationRecord:
    return CitationRecord(
        citation_id=sample.get("citation_id", ""),
        title=sample.get("title", ""),
        authors=sample.get("authors", []),
        venue=sample.get("venue", ""),
        year=sample.get("year"),
        doi=sample.get("doi", ""),
        arxiv_id=sample.get("arxiv_id", ""),
        url=sample.get("url", ""),
        volume=sample.get("volume", ""),
        pages=sample.get("pages", ""),
        publisher=sample.get("publisher", ""),
        location=sample.get("location", ""),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", required=True, help="Code JSON file (e.g. R1.json)")
    parser.add_argument("--meta", default=None, help="meta.json for expected labels")
    parser.add_argument("--expected-verdict", default=None,
                        choices=["VALID", "POTENTIAL_REFERENCE", "FAKE_REFERENCE"],
                        help="Expected verdict for all samples (alternative to --meta)")
    parser.add_argument("--max", type=int, default=None, help="Test only first N samples")
    parser.add_argument("--workers", type=int, default=1, help="Parallel verification workers")
    parser.add_argument("--out", default=None, help="Save detailed results JSON")
    args = parser.parse_args()

    samples = json.loads(Path(args.samples).read_text())
    if args.max:
        samples = samples[:args.max]

    meta = {}
    if args.meta:
        meta = json.loads(Path(args.meta).read_text())

    print(f"Testing {len(samples)} samples from {args.samples}", flush=True)

    cfg = load_pdf_checker_config()
    verifier = _build_verifier(cfg)
    print("Verifier ready.\n", flush=True)

    results = []
    correct = 0
    total = 0

    pbar = tqdm(total=len(samples), desc="Verifying", dynamic_ncols=True) if tqdm else None

    def _verify_one(sample):
        citation = _make_citation(sample)
        verdict = verifier.verify_citation(citation)
        return sample, citation.citation_id, verdict

    def _build_result(sample, cid, verdict):
        vdict = citation_verdict_to_dict(verdict)
        if args.expected_verdict:
            expected = args.expected_verdict
        elif cid in meta:
            expected = _LABEL_TO_VERDICT.get(meta[cid]["label"], meta[cid]["label"])
        else:
            expected = "?"

        match = verdict.verdict.value == expected

        # Full field-level comparison from the verdict
        comparison = vdict.get("comparison", {})
        field_status = comparison.get("field_status", {})
        ref_snapshot = vdict.get("reference_snapshot", {})
        matched_candidate = vdict.get("matched_candidate", {})

        _ALL_FIELDS = ("title", "authors", "venue", "year", "doi", "arxiv_id",
                       "pages", "volume", "publisher", "location")

        # Build per-candidate evaluations with full detail
        candidate_evaluations = []
        for ev in vdict.get("candidate_evaluations", []):
            ev_comparison = ev.get("comparison", {})
            ev_field_status = ev_comparison.get("field_status", {})
            candidate_evaluations.append({
                "connector": ev.get("connector", ""),
                "agent": ev.get("agent", ""),
                "verdict": ev.get("verdict", ""),
                "reason": ev.get("reason", ""),
                "taxonomy": ev.get("taxonomy", []),
                "candidate": {
                    "title": (ev.get("candidate", {}) or {}).get("title", "")[:100],
                    "authors": (ev.get("candidate", {}) or {}).get("authors", []),
                    "venue": (ev.get("candidate", {}) or {}).get("venue", ""),
                    "year": (ev.get("candidate", {}) or {}).get("year"),
                    "doi": (ev.get("candidate", {}) or {}).get("doi", ""),
                    "arxiv_id": (ev.get("candidate", {}) or {}).get("arxiv_id", ""),
                    "volume": (ev.get("candidate", {}) or {}).get("volume", ""),
                    "pages": (ev.get("candidate", {}) or {}).get("pages", ""),
                    "publisher": (ev.get("candidate", {}) or {}).get("publisher", ""),
                    "location": (ev.get("candidate", {}) or {}).get("location", ""),
                } if ev.get("candidate") else None,
                "field_status": {
                    f: ev_field_status.get(f, {}).get("status", "n/a")
                    if isinstance(ev_field_status.get(f), dict) else ev_field_status.get(f, "n/a")
                    for f in _ALL_FIELDS
                } if ev_field_status else {},
            })

        return {
            "citation_id": cid,
            "expected": expected,
            "predicted": verdict.verdict.value,
            "taxonomy": vdict.get("taxonomy_subtype", []),
            "match": match,
            "adjudication_reason": vdict.get("adjudication_reason", ""),
            "evidence_sources": vdict.get("evidence_sources", []),
            "conflicts": vdict.get("conflicts", []),
            "needs_human_review": vdict.get("needs_human_review", False),
            "field_status": {
                f: field_status.get(f, {}).get("status", "n/a")
                if isinstance(field_status.get(f), dict) else field_status.get(f, "n/a")
                for f in _ALL_FIELDS
            },
            "reference_snapshot": ref_snapshot,
            "matched_candidate": {
                "connector": matched_candidate.get("connector", ""),
                "title": matched_candidate.get("title", "")[:100],
                "authors": matched_candidate.get("authors", []),
                "venue": matched_candidate.get("venue", ""),
                "year": matched_candidate.get("year"),
                "doi": matched_candidate.get("doi", ""),
                "arxiv_id": matched_candidate.get("arxiv_id", ""),
                "volume": matched_candidate.get("volume", ""),
                "pages": matched_candidate.get("pages", ""),
                "publisher": matched_candidate.get("publisher", ""),
                "location": matched_candidate.get("location", ""),
                "matched_fields": matched_candidate.get("matched_fields", []),
                "conflicts": matched_candidate.get("conflicts", []),
            } if matched_candidate else None,
            "candidate_evaluations": candidate_evaluations,
            "input_citation": {
                "title": sample.get("title", ""),
                "authors": sample.get("authors", []),
                "venue": sample.get("venue", ""),
                "year": sample.get("year"),
                "doi": sample.get("doi", ""),
                "arxiv_id": sample.get("arxiv_id", ""),
                "pages": sample.get("pages", ""),
                "volume": sample.get("volume", ""),
                "publisher": sample.get("publisher", ""),
                "location": sample.get("location", ""),
            },
        }, match

    if args.workers <= 1:
        for sample in samples:
            _, cid, verdict = _verify_one(sample)
            result, match = _build_result(sample, cid, verdict)
            results.append(result)
            if match:
                correct += 1
            total += 1
            if pbar:
                pbar.update(1)
                pbar.set_postfix({"acc": f"{correct}/{total}"})
            else:
                status = "✅" if match else "❌"
                print(f"  {status} {cid:12s}  expected={result['expected']:20s}  got={result['predicted']:20s}  tax={result['taxonomy']}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_verify_one, s): s for s in samples}
            for fut in as_completed(futures):
                sample, cid, verdict = fut.result()
                result, match = _build_result(sample, cid, verdict)
                results.append(result)
                if match:
                    correct += 1
                total += 1
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix({"acc": f"{correct}/{total}"})

    if pbar:
        pbar.close()

    # Summary
    from collections import Counter

    print(f"\n{'='*60}")
    print(f"Accuracy: {correct}/{total} ({100*correct/total:.1f}%)" if total else "No samples tested")

    # Verdict distribution
    pred_dist = Counter(r["predicted"] for r in results)
    print(f"\nPredicted verdict distribution:")
    for v, c in pred_dist.most_common():
        print(f"  {v}: {c}")

    # Taxonomy distribution
    tax_dist = Counter()
    for r in results:
        for t in r.get("taxonomy", []):
            tax_dist[t] += 1
    if tax_dist:
        print(f"\nTaxonomy distribution:")
        for t, c in tax_dist.most_common():
            print(f"  {t}: {c}")

    # Field-level mismatch summary (which fields cause failures?)
    field_issues = Counter()
    for r in results:
        if not r["match"]:
            for f, status in r.get("field_status", {}).items():
                if status not in ("match", "both_missing", "reference_missing", "n/a"):
                    field_issues[f"{f}:{status}"] += 1
    if field_issues:
        print(f"\nField issues in mismatched samples:")
        for issue, c in field_issues.most_common(15):
            print(f"  {issue:35s} {c}")

    # Evidence source distribution
    source_dist = Counter()
    for r in results:
        for s in r.get("evidence_sources", []):
            source_dist[s] += 1
    if source_dist:
        print(f"\nEvidence sources used:")
        for s, c in source_dist.most_common():
            print(f"  {s}: {c}")

    # Show mismatches with detail
    mismatches = [r for r in results if not r["match"]]
    if mismatches:
        print(f"\n{'='*60}")
        print(f"Mismatches ({len(mismatches)}):")
        for r in mismatches[:10]:
            print(f"\n  {r['citation_id']}  expected={r['expected']}  got={r['predicted']}  tax={r['taxonomy']}")
            print(f"    reason: {r['adjudication_reason'][:150]}")
            print(f"    conflicts: {r['conflicts']}")
            bad_fields = {f: s for f, s in r.get('field_status', {}).items()
                          if s not in ('match', 'both_missing', 'reference_missing', 'n/a')}
            if bad_fields:
                print(f"    problematic fields: {bad_fields}")
            if r.get("matched_candidate"):
                mc = r["matched_candidate"]
                print(f"    best candidate: [{mc['connector']}] matched={mc['matched_fields']} conflicts={mc['conflicts']}")

    # Save
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "accuracy": correct / total if total else 0,
            "total": total,
            "correct": correct,
            "predicted_distribution": dict(pred_dist),
            "taxonomy_distribution": dict(tax_dist),
            "field_issues_in_mismatches": dict(field_issues),
            "results": results,
        }, indent=2, ensure_ascii=False))
        print(f"\nDetailed results saved to: {out_path}")


if __name__ == "__main__":
    main()
