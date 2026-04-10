"""End-to-end PDF citation hallucination check pipeline.

Combines the extraction modes from `pdf_extractor_demo.py` (heuristic config /
ocr_llm_extract) with the verifier wiring from `citation_verification_demo.py`
(cascading 3-agent + secondary verifier + optional source router).

Replaces the old minimal `run_pdf_check` which used only rule-based adjudication.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from packages.connectors import default_orchestrator
from packages.connectors.base import ConnectorOrchestrator
from packages.core import CitationVerifier
from packages.core.models import CitationRecord, ExtractionQuality
from packages.core.report import build_summary, compute_risk_score, render_markdown
from packages.core.models import citation_verdict_to_dict

from .config import load_pdf_checker_config
from .ingest.citation_marker_linker import estimate_link_quality, extract_inline_markers
from .ingest.citation_parser import parse_reference_entries
from .ingest.pdf_text_extract import extract_pdf_pages
from .ingest.reference_segmenter import segment_references_from_pages


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _env_flag(name: str, default: str = "1") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _log(enabled: bool, message: str) -> None:
    if enabled:
        print(message, flush=True)


def _wrap_for_terminal(text: str, prefix: str = "", width: int = 200) -> str:
    """Wrap a long string for terminal output, adding `prefix` to the first line
    and an indent equal to the prefix's length on continuation lines.
    """
    import textwrap
    indent = " " * len(prefix)
    wrapped = textwrap.wrap(text, width=width - len(prefix))
    if not wrapped:
        return prefix
    return prefix + wrapped[0] + ("\n" + "\n".join(indent + line for line in wrapped[1:]) if len(wrapped) > 1 else "")


# ---------------------------------------------------------------------------
# Step 1+2+3: PDF → reference entries → parsed CitationRecords
# (mirrors pdf_extractor_demo.py's extraction logic)
# ---------------------------------------------------------------------------

def _build_llm_reparse_config(cfg) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the llm_reparse_config dict and metadata describing it.

    Path is selected by `cfg.citation_parse_method`:
      - "citation_reparse" (default) → use cfg.citation_reparse as-is
      - "ocr_llm_extract"            → use cfg.ocr_llm_extract (force-all + bedrock)
    """
    method = cfg.citation_parse_method

    if method == "citation_reparse":
        llm_cfg = asdict(cfg.citation_reparse)
        meta = {
            "pipeline_method": "citation_reparse",
            "llm_reparse_enabled": bool(llm_cfg.get("enabled")),
            "llm_reparse_provider": str(llm_cfg.get("provider", "")),
        }
        return llm_cfg, meta

    # ocr_llm_extract — force-all Bedrock reparse on every entry
    if not cfg.ocr_llm_extract.enabled:
        raise RuntimeError(
            "citation_parse_method=ocr_llm_extract requires ocr_llm_extract.enabled=true "
            "and a valid bedrock model_id in config.json."
        )

    model_id = str(cfg.ocr_llm_extract.bedrock.model_id or "").strip()
    region = str(cfg.ocr_llm_extract.bedrock.region or "").strip() or "us-east-1"
    bearer_token = str(cfg.ocr_llm_extract.bedrock.bearer_token or "").strip() or None
    if not model_id:
        raise RuntimeError(
            "citation_parse_method=ocr_llm_extract requires "
            "ocr_llm_extract.bedrock.model_id in config.json."
        )

    llm_cfg = {
        "enabled": True,
        "provider": "bedrock",
        "model_path": "",
        "max_new_tokens": int(cfg.ocr_llm_extract.max_new_tokens),
        "temperature": float(cfg.ocr_llm_extract.temperature),
        "force_all_entries": bool(cfg.ocr_llm_extract.force_all_entries),
        "parallel_workers": 8,
        "bedrock": {
            "model_id": model_id,
            "region": region,
            "bearer_token": bearer_token,
        },
    }

    meta = {
        "pipeline_method": "ocr_llm_extract",
        "llm_reparse_provider_forced": "bedrock",
        "llm_reparse_model_id": model_id,
        "llm_reparse_region": region,
        "llm_reparse_force_all_entries": bool(llm_cfg["force_all_entries"]),
        "llm_reparse_max_new_tokens": int(llm_cfg["max_new_tokens"]),
        "llm_reparse_temperature": float(llm_cfg["temperature"]),
        "llm_reparse_parallel_workers": int(llm_cfg["parallel_workers"]),
    }
    return llm_cfg, meta


def _extract_and_parse(
    pdf: Path, cfg, args: argparse.Namespace, verbose: bool,
) -> tuple[list[CitationRecord], ExtractionQuality, dict[str, Any], list[str]]:
    """Run steps 1-3: PDF → pages → reference entries → CitationRecord[]."""
    _log(verbose, f"[1/7] Reading PDF pages: {pdf}")
    pages = extract_pdf_pages(pdf)
    _log(verbose, f"      Loaded {len(pages)} pages")

    _log(verbose, "[2/7] Extracting reference entries")
    reference_entries, extraction_quality, extraction_metadata = segment_references_from_pages(
        pages,
        entry_extraction=cfg.entry_extraction.mode,
        model_provider=cfg.entry_extraction.provider,
        model_id=cfg.entry_extraction.bedrock.model_id,
        region=cfg.entry_extraction.bedrock.region,
        entry_chunk_chars=cfg.entry_extraction.chunk_chars,
        bearer_token=cfg.entry_extraction.bedrock.bearer_token,
        local_model_path=cfg.entry_extraction.local.model_path,
        local_inference_backend=cfg.entry_extraction.local.inference_backend,
        source_pdf_path=pdf,
    )
    _log(
        verbose,
        f"      Extracted {len(reference_entries)} reference entries "
        f"(mode={extraction_metadata.get('entry_extraction_mode')}, "
        f"provider={extraction_metadata.get('entry_extraction_provider')})",
    )

    llm_cfg, llm_meta = _build_llm_reparse_config(cfg)
    _log(verbose, f"[3/7] Parsing references (citation_parse_method={cfg.citation_parse_method})")
    citations = parse_reference_entries(
        reference_entries,
        llm_reparse_config=llm_cfg,
    )
    _log(verbose, f"      Parsed {len(citations)} citation records")

    extraction_metadata.update(llm_meta)
    return citations, extraction_quality, extraction_metadata, pages


# ---------------------------------------------------------------------------
# Step 4-7: Verifier setup + run + reports
# (mirrors citation_verification_demo.py's verifier wiring)
# ---------------------------------------------------------------------------

def _build_verifier(cfg, args: argparse.Namespace, verbose: bool) -> CitationVerifier:
    """Build a fully-wired CitationVerifier with cascading agents + secondary verifier."""

    _log(verbose, "[5/7] Initializing bibliographic connectors")
    cache_path = Path(args.cache_path) if args.cache_path else cfg.connectors.cache_path
    dblp_mirror_path = (
        Path(args.dblp_mirror_path) if args.dblp_mirror_path else cfg.connectors.dblp_mirror_path
    )
    cli_dblp_sqlite = (args.dblp_sqlite_path or "").strip()
    cfg_dblp_sqlite = str(cfg.connectors.dblp_sqlite_path or "").strip()
    final_dblp_sqlite = cli_dblp_sqlite or cfg_dblp_sqlite
    final_semscholar_key = (
        (args.semantic_scholar_api_key or "").strip()
        or (cfg.connectors.semantic_scholar_api_key or "").strip()
        or None
    )

    orchestrator = default_orchestrator(
        cache_path=cache_path,
        dblp_mirror_path=dblp_mirror_path,
        dblp_sqlite_path=Path(final_dblp_sqlite) if final_dblp_sqlite else None,
        enabled_sources=cfg.connectors.enabled_sources,
        acl_anthology_data_dir=cfg.connectors.acl_anthology_data_dir,
        acl_anthology_repo_path=cfg.connectors.acl_anthology_repo_path,
        semantic_scholar_api_key=final_semscholar_key,
        govinfo_api_key=cfg.connectors.govinfo_api_key,
        searxng_base_url=cfg.connectors.searxng_base_url,
        ncbi_api_key=cfg.connectors.ncbi_api_key,
        ncbi_email=cfg.connectors.ncbi_email,
        web_search_provider=cfg.connectors.web_search_provider,
        google_api_key=cfg.connectors.google_api_key,
        google_cse_id=cfg.connectors.google_cse_id,
        serpapi_key=cfg.connectors.serpapi_key,
        tavily_api_key=cfg.connectors.tavily_api_key,
    )

    # Optional LLM-based source router
    llm_resolver = None
    if cfg.verification_llm.enabled and cfg.verification_llm.provider == "bedrock":
        from citation_verification_demo import (
            BedrockStrictMatchResolver,
            BedrockSourceSuitabilityRouter,
            RoutedConnectorOrchestrator,
        )
        llm_resolver = BedrockStrictMatchResolver(
            model_id=str(cfg.verification_llm.bedrock.model_id),
            region=cfg.verification_llm.bedrock.region,
            bearer_token=cfg.verification_llm.bedrock.bearer_token,
            max_candidates=cfg.verification_llm.max_candidates,
        )
        if args.enable_llm_source_router:
            source_router = BedrockSourceSuitabilityRouter(
                model_id=str(cfg.verification_llm.bedrock.model_id),
                region=cfg.verification_llm.bedrock.region,
                bearer_token=cfg.verification_llm.bedrock.bearer_token,
                max_sources_cap=args.source_router_max_sources,
            )
            orchestrator = RoutedConnectorOrchestrator(orchestrator, source_router)
            _log(verbose, "      llm_source_router enabled")

    # Secondary verifier (uses arxiv / semantic_scholar / openalex / url_direct connectors)
    from packages.connectors.arxiv import ArxivConnector
    from packages.connectors.openalex import OpenAlexConnector
    from packages.connectors.semanticscholar import SemanticScholarConnector
    from packages.connectors.url_direct import URLDirectConnector
    from packages.core.secondary_verification import SecondaryVerifier

    arxiv_conn = semscholar_conn = openalex_conn = url_direct_conn = None
    for conn in orchestrator.connectors:
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

    # Cascading 3-agent + extractor agent (Bedrock)
    valid_agent = potential_agent = hallucinated_agent = extractor_agent = None
    if cfg.verification_llm.enabled and cfg.verification_llm.provider == "bedrock":
        from citation_verification_demo import _build_bedrock_client
        from packages.core.bedrock_agents import (
            BedrockHallucinatedAgent,
            BedrockPotentialAgent,
            BedrockValidAgent,
        )
        from packages.core.extractor_agent import BedrockExtractorAgent

        bedrock_client = _build_bedrock_client(
            region=cfg.verification_llm.bedrock.region,
            bearer_token=cfg.verification_llm.bedrock.bearer_token,
        )
        model_id = str(cfg.verification_llm.bedrock.model_id)
        valid_agent = BedrockValidAgent(bedrock_client, model_id)
        potential_agent = BedrockPotentialAgent(bedrock_client, model_id)
        hallucinated_agent = BedrockHallucinatedAgent(bedrock_client, model_id)
        extractor_agent = BedrockExtractorAgent(bedrock_client, model_id)
        _log(verbose, f"      cascading 3-agent + extractor enabled (model={model_id})")
    else:
        _log(verbose, "      cascading agents disabled (no verification LLM)")

    # Verified reference cache — reuse connector_cache.sqlite for storage
    from packages.connectors.base import VerifiedReferenceCache
    verified_ref_cache = VerifiedReferenceCache(cache_path)
    _log(verbose, f"      verified reference cache: {cache_path}")

    return CitationVerifier(
        orchestrator=orchestrator,
        llm_resolver=llm_resolver,
        secondary_verifier=secondary_verifier,
        valid_agent=valid_agent,
        potential_agent=potential_agent,
        hallucinated_agent=hallucinated_agent,
        extractor_agent=extractor_agent,
        verified_ref_cache=verified_ref_cache,
    )


def _save_citations_artifact(
    pdf: Path,
    citations: list[CitationRecord],
    extraction_quality: ExtractionQuality,
    extraction_metadata: dict[str, Any],
    out_dir: Path,
) -> Path:
    """Persist the parsed citations (post-step-3) to a JSON file for inspection.

    This is the artifact that pdf_extractor_demo.py would produce — useful for
    debugging the extraction step independently of verification.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / f"{pdf.stem}.parsed_citations.json"
    payload = {
        "input_pdf": str(pdf),
        "reference_entries_count": len(citations),
        "extraction_quality": extraction_quality.value,
        "metadata": extraction_metadata,
        "citations": [asdict(c) for c in citations],
    }
    artifact_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return artifact_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pdf_check(
    input_pdf: str | Path,
    out_path: str | Path,
    args: argparse.Namespace | None = None,
) -> dict:
    """Run the full PDF → verification pipeline."""
    input_pdf = Path(input_pdf)
    out_path = Path(out_path)
    cfg = load_pdf_checker_config()
    verbose = _env_flag("CITATION_CHECKER_VERBOSE", "1")

    if args is None:
        args = build_arg_parser().parse_args(
            ["--input", str(input_pdf), "--out", str(out_path)]
        )

    citations, extraction_quality, extraction_metadata, pages = _extract_and_parse(
        input_pdf, cfg, args, verbose,
    )

    _log(verbose, "[4/7] Linking inline citation markers")
    combined_text = "\n".join(pages)
    markers = extract_inline_markers(combined_text)
    link_quality = estimate_link_quality(markers, len(citations))
    _log(verbose, f"      marker_link_quality={round(link_quality, 4)} ({len(markers)} markers)")

    # Optionally save the parsed citations as an inspectable artifact
    if args.save_parsed_citations:
        artifact_dir = Path(args.save_parsed_citations)
        artifact_path = _save_citations_artifact(
            input_pdf, citations, extraction_quality, extraction_metadata, artifact_dir,
        )
        _log(verbose, f"      Saved parsed citations to {artifact_path}")

    if args.extract_only:
        _log(verbose, "[skip] --extract-only set; skipping verification")
        return {
            "input_pdf": str(input_pdf),
            "reference_entries_count": len(citations),
            "extraction_quality": extraction_quality.value,
            "metadata": extraction_metadata,
            "citations": [asdict(c) for c in citations],
        }

    verifier = _build_verifier(cfg, args, verbose)

    # Build per-citation extraction quality map
    quality_map: dict[str, ExtractionQuality] = {
        c.citation_id: extraction_quality for c in citations
    }
    if link_quality < 0.5:
        for c in citations:
            quality_map[c.citation_id] = ExtractionQuality.LOW

    _log(verbose, f"[6/7] Verifying citations (n={len(citations)})")
    paper_title = input_pdf.stem.replace("_", " ")

    # Verify citation-by-citation with incremental writes (mirrors verification demo)
    partial_verdicts: list[Any] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for idx, citation in enumerate(citations, start=1):
        _log(
            verbose,
            f"      [{idx}/{len(citations)}] {citation.citation_id} "
            f"title={citation.title[:60]!r}",
        )
        verdict = verifier.verify_citation(
            citation,
            extraction_quality=quality_map.get(citation.citation_id, ExtractionQuality.UNKNOWN),
            source_paper_title=paper_title,
        )
        partial_verdicts.append(verdict)
        # Incremental save so a long run can be inspected mid-flight
        report_dict = _build_report_dict(
            input_pdf, partial_verdicts, pages, markers, link_quality,
            extraction_quality, extraction_metadata,
        )
        _write_json_atomic(out_path, report_dict)

        # Print verdict + full reason + conflicts. Reasons are wrapped at 200 chars
        # so long Bedrock LLM explanations don't blow up the terminal.
        verdict_label = verdict.verdict.value
        taxonomy = verdict.taxonomy_subtype or []
        reason = (verdict.adjudication_reason or "").strip()
        conflicts = verdict.conflicts or []
        _log(
            verbose,
            f"      → verdict={verdict_label} taxonomy={taxonomy}",
        )
        if reason:
            # Wrap reason for terminal readability
            wrapped = _wrap_for_terminal(reason, prefix="        reason: ", width=200)
            _log(verbose, wrapped)
        if conflicts:
            _log(verbose, f"        conflicts: {conflicts}")
        _log(verbose, "")  # blank line between citations

    final_report = _build_report_dict(
        input_pdf, partial_verdicts, pages, markers, link_quality,
        extraction_quality, extraction_metadata,
    )

    _log(verbose, f"[7/7] Writing markdown report to {out_path.with_suffix('.md')}")
    out_path.with_suffix(".md").write_text(
        render_markdown_dict(final_report), encoding="utf-8",
    )
    _log(verbose, "      Done")
    return final_report


def _build_report_dict(
    input_pdf: Path,
    verdicts: list[Any],
    pages: list[str],
    markers: list[Any],
    link_quality: float,
    extraction_quality: ExtractionQuality,
    extraction_metadata: dict[str, Any],
) -> dict:
    return {
        "report_version": "1.0",
        "paper_id": input_pdf.stem,
        "pipeline_type": "PDF",
        "citations": [citation_verdict_to_dict(v) for v in verdicts],
        "summary": build_summary(verdicts),
        "risk_score": compute_risk_score(verdicts),
        "requires_human_review": any(v.needs_human_review for v in verdicts),
        "metadata": {
            "input_pdf": str(input_pdf),
            "pages": len(pages),
            "reference_entries": len(verdicts),
            "inline_marker_count": len(markers),
            "marker_link_quality": round(link_quality, 4),
            "extraction_quality": extraction_quality.value,
            **extraction_metadata,
        },
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def render_markdown_dict(report: dict) -> str:
    """Lightweight markdown rendering when we don't have a CheckReport object."""
    lines = [
        f"# Citation Hallucination Report — {report.get('paper_id', '')}",
        "",
        f"- Pipeline: {report.get('pipeline_type', '')}",
        f"- Risk score: {report.get('risk_score', 0)}",
        f"- Needs human review: {report.get('requires_human_review', False)}",
        "",
        "## Summary",
    ]
    summary = report.get("summary", {}) or {}
    for k, v in summary.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Citations")
    for c in report.get("citations", []):
        lines.append(
            f"- {c.get('citation_id')} — **{c.get('verdict', '?')}** "
            f"{c.get('taxonomy_subtype', [])}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run end-to-end citation hallucination check on a PDF manuscript."
    )
    parser.add_argument("--input", required=True, help="Path to input PDF.")
    parser.add_argument("--out", required=True, help="Path to output JSON report.")

    # NOTE: extraction method is now controlled entirely by config.json's
    # `citation_parse_method` field ("citation_reparse" or "ocr_llm_extract"),
    # not by CLI flags. To switch methods, edit config.json.

    # Connector overrides (mirrors citation_verification_demo.py)
    parser.add_argument(
        "--cache-path", default="",
        help="Connector cache sqlite path. Empty = use config.",
    )
    parser.add_argument(
        "--dblp-mirror-path", default="",
        help="Offline DBLP mirror JSONL path. Empty = use config.",
    )
    parser.add_argument(
        "--dblp-sqlite-path", default="",
        help="DBLP sqlite path override. Empty = use config.",
    )
    parser.add_argument(
        "--semantic-scholar-api-key", default="",
        help="Semantic Scholar API key override. Empty = use config.",
    )
    parser.add_argument(
        "--offline-only", action="store_true",
        help="Set CITATION_CHECKER_OFFLINE_ONLY=1 (skip online connectors).",
    )

    # Verifier wiring (mirrors citation_verification_demo.py)
    parser.add_argument(
        "--enable-llm-source-router", action="store_true",
        help="Use a Bedrock LLM-based source router to rank/cap connectors.",
    )
    parser.add_argument(
        "--source-router-max-sources", type=int, default=0,
        help="Hard cap for the LLM source router. 0 = trust router's budget.",
    )

    # Pipeline shortcuts
    parser.add_argument(
        "--extract-only", action="store_true",
        help="Skip verification; only run extraction (steps 1-4) and dump citations.",
    )
    parser.add_argument(
        "--save-parsed-citations", default="",
        help="Optional directory to also save the parsed citations JSON before verification.",
    )

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.offline_only:
        os.environ["CITATION_CHECKER_OFFLINE_ONLY"] = "1"
    run_pdf_check(args.input, args.out, args=args)


if __name__ == "__main__":
    main()
