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

from .ingest.bib_parser import parse_bib_directory
from .ingest.citation_linker import build_citation_records
from .ingest.tex_parser import parse_tex_directory



def run_source_check(
    input_dir: str | Path,
    out_path: str | Path,
    orchestrator: ConnectorOrchestrator | None = None,
) -> dict:
    input_dir = Path(input_dir)
    out_path = Path(out_path)

    key_to_locations = parse_tex_directory(input_dir)
    bib_entries = parse_bib_directory(input_dir)
    citations = build_citation_records(key_to_locations, bib_entries)

    orchestrator = orchestrator or default_orchestrator(
        cache_path=Path("data/cache/connector_cache.sqlite"),
        dblp_mirror_path=Path("data/cache/dblp_mirror.jsonl"),
        semantic_scholar_api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
    )
    verifier = CitationVerifier(orchestrator=orchestrator)

    quality_map = {citation.citation_id: ExtractionQuality.HIGH for citation in citations}
    report = verifier.verify_paper(
        paper_id=input_dir.name,
        pipeline_type="SOURCE",
        citations=citations,
        extraction_quality_map=quality_map,
        metadata={"input_dir": str(input_dir), "citation_count": len(citations)},
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    markdown_path = out_path.with_suffix(".md")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")

    return report.to_dict()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run citation hallucination checks on LaTeX+BibTeX source.")
    parser.add_argument("--input", required=True, help="Path to the manuscript source directory.")
    parser.add_argument("--out", required=True, help="Path to output JSON report.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_source_check(args.input, args.out)


if __name__ == "__main__":
    main()
