from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

from packages.connectors import default_orchestrator
from packages.connectors.base import ConnectorOrchestrator
from packages.core import CitationVerifier
from packages.core.models import ExtractionQuality
from packages.core.report import render_markdown

from .config import load_pdf_checker_config
from .ingest.citation_marker_linker import estimate_link_quality, extract_inline_markers
from .ingest.citation_parser import parse_reference_entries
from .ingest.pdf_text_extract import extract_pdf_pages
from .ingest.reference_segmenter import segment_references_from_pages


def _env_flag(name: str, default: str = "1") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _log(enabled: bool, message: str) -> None:
    if enabled:
        print(message, flush=True)


def run_pdf_check(
    input_pdf: str | Path,
    out_path: str | Path,
    orchestrator: ConnectorOrchestrator | None = None,
) -> dict:
    input_pdf = Path(input_pdf)
    out_path = Path(out_path)
    config = load_pdf_checker_config()
    verbose = _env_flag("CITATION_CHECKER_VERBOSE", "1")

    _log(verbose, f"[1/7] Reading PDF pages: {input_pdf}")
    pages = extract_pdf_pages(input_pdf)
    _log(verbose, f"      Loaded {len(pages)} pages")
    combined_text = "\n".join(pages)

    _log(verbose, "[2/7] Extracting reference entries")
    reference_entries, extraction_quality, extraction_metadata = segment_references_from_pages(
        pages,
        entry_extraction=config.entry_extraction.mode,
        model_provider=config.entry_extraction.provider,
        model_id=config.entry_extraction.bedrock.model_id,
        region=config.entry_extraction.bedrock.region,
        entry_chunk_chars=config.entry_extraction.chunk_chars,
        bearer_token=config.entry_extraction.bedrock.bearer_token,
        local_model_path=config.entry_extraction.local.model_path,
        local_inference_backend=config.entry_extraction.local.inference_backend,
        source_pdf_path=input_pdf,
    )
    _log(
        verbose,
        (
            f"      Extracted {len(reference_entries)} reference entries "
            f"(mode={extraction_metadata.get('entry_extraction_mode')}, "
            f"input={extraction_metadata.get('entry_extraction_input')}, "
            f"provider={extraction_metadata.get('entry_extraction_provider')})"
        ),
    )

    _log(verbose, "[3/7] Parsing references into citation records")
    citations = parse_reference_entries(
        reference_entries,
        llm_reparse_config=asdict(config.citation_reparse),
    )
    _log(verbose, f"      Parsed {len(citations)} citation records")

    _log(verbose, "[4/7] Linking inline citation markers")
    markers = extract_inline_markers(combined_text)
    link_quality = estimate_link_quality(markers, len(citations))
    _log(verbose, f"      marker_link_quality={round(link_quality, 4)} ({len(markers)} markers)")

    _log(verbose, "[5/7] Initializing bibliographic connectors")
    orchestrator = orchestrator or default_orchestrator(
        cache_path=config.connectors.cache_path,
        dblp_mirror_path=config.connectors.dblp_mirror_path,
        semantic_scholar_api_key=config.connectors.semantic_scholar_api_key,
        dblp_sqlite_path=config.connectors.dblp_sqlite_path,
        enabled_sources=config.connectors.enabled_sources,
        govinfo_api_key=config.connectors.govinfo_api_key,
        searxng_base_url=config.connectors.searxng_base_url,
        ncbi_api_key=config.connectors.ncbi_api_key,
        ncbi_email=config.connectors.ncbi_email,
        google_api_key=config.connectors.google_api_key,
        google_cse_id=config.connectors.google_cse_id,
        serpapi_key=config.connectors.serpapi_key,
    )
    verifier = CitationVerifier(orchestrator=orchestrator)

    quality_map: dict[str, ExtractionQuality] = {
        citation.citation_id: extraction_quality for citation in citations
    }
    if link_quality < 0.5:
        for citation in citations:
            quality_map[citation.citation_id] = ExtractionQuality.LOW

    _log(verbose, f"[6/7] Verifying citations (progress: n/{len(citations)})")
    report = verifier.verify_paper(
        paper_id=input_pdf.stem,
        pipeline_type="PDF",
        citations=citations,
        extraction_quality_map=quality_map,
        show_progress=verbose,
        metadata={
            "input_pdf": str(input_pdf),
            "pages": len(pages),
            "reference_entries": len(reference_entries),
            "inline_marker_count": len(markers),
            "marker_link_quality": round(link_quality, 4),
            "extraction_quality": extraction_quality.value,
            **extraction_metadata,
        },
    )

    _log(verbose, f"[7/7] Writing reports to {out_path} and {out_path.with_suffix('.md')}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    out_path.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")

    _log(verbose, "      Done")
    return report.to_dict()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run citation hallucination checks on a PDF manuscript.")
    parser.add_argument("--input", required=True, help="Path to input PDF.")
    parser.add_argument("--out", required=True, help="Path to output JSON report.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_pdf_check(args.input, args.out)


if __name__ == "__main__":
    main()
