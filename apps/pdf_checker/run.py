from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from packages.connectors import default_orchestrator
from packages.connectors.base import ConnectorOrchestrator
from packages.core import CitationVerifier
from packages.core.models import ExtractionQuality
from packages.core.report import render_markdown

from .ingest.citation_marker_linker import estimate_link_quality, extract_inline_markers
from .ingest.citation_parser import parse_reference_entries
from .ingest.pdf_text_extract import extract_pdf_pages
from .ingest.reference_segmenter import segment_references



def run_pdf_check(
    input_pdf: str | Path,
    out_path: str | Path,
    orchestrator: ConnectorOrchestrator | None = None,
) -> dict:
    input_pdf = Path(input_pdf)
    out_path = Path(out_path)

    pages = extract_pdf_pages(input_pdf)
    combined_text = "\n".join(pages)
    reference_entries, extraction_quality = segment_references(combined_text)
    citations = parse_reference_entries(reference_entries)

    markers = extract_inline_markers(combined_text)
    link_quality = estimate_link_quality(markers, len(citations))

    orchestrator = orchestrator or default_orchestrator(
        cache_path=Path("data/cache/connector_cache.sqlite"),
        dblp_mirror_path=Path("data/cache/dblp_mirror.jsonl"),
        semantic_scholar_api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
    )
    verifier = CitationVerifier(orchestrator=orchestrator)

    quality_map: dict[str, ExtractionQuality] = {
        citation.citation_id: extraction_quality for citation in citations
    }
    if link_quality < 0.5:
        for citation in citations:
            quality_map[citation.citation_id] = ExtractionQuality.LOW

    report = verifier.verify_paper(
        paper_id=input_pdf.stem,
        pipeline_type="PDF",
        citations=citations,
        extraction_quality_map=quality_map,
        metadata={
            "input_pdf": str(input_pdf),
            "pages": len(pages),
            "reference_entries": len(reference_entries),
            "inline_marker_count": len(markers),
            "marker_link_quality": round(link_quality, 4),
            "extraction_quality": extraction_quality.value,
        },
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    out_path.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")

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
