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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_input_dir(input_dir: str | Path) -> Path:
    raw_path = Path(input_dir)
    if raw_path.exists():
        return raw_path

    repo_candidate = (_repo_root() / raw_path).resolve()
    if repo_candidate.exists():
        return repo_candidate

    raw_text = raw_path.as_posix()
    legacy_prefix = "data/fixtures/"
    if raw_text.startswith(legacy_prefix):
        remapped = Path("data") / raw_text[len(legacy_prefix):]
        if remapped.exists():
            return remapped
        repo_remapped = (_repo_root() / remapped).resolve()
        if repo_remapped.exists():
            return repo_remapped

    return raw_path


def _cache_path() -> Path:
    return Path(os.getenv("CITATION_CHECKER_CACHE_PATH", "data/cache/connector_cache.sqlite"))


def _mirror_path() -> Path:
    return Path(os.getenv("CITATION_CHECKER_DBLP_MIRROR_PATH", "data/cache/dblp_mirror.jsonl"))


def run_source_check(
    input_dir: str | Path,
    out_path: str | Path,
    orchestrator: ConnectorOrchestrator | None = None,
) -> dict:
    input_dir = _resolve_input_dir(input_dir)
    out_path = Path(out_path)

    key_to_locations = parse_tex_directory(input_dir)
    bib_entries = parse_bib_directory(input_dir)
    citations = build_citation_records(key_to_locations, bib_entries)

    orchestrator = orchestrator or default_orchestrator(
        cache_path=_cache_path(),
        dblp_mirror_path=_mirror_path(),
        acl_anthology_data_dir=os.getenv("CITATION_CHECKER_ACL_ANTHOLOGY_DATA_DIR"),
        acl_anthology_repo_path=os.getenv("CITATION_CHECKER_ACL_ANTHOLOGY_REPO_PATH"),
        semantic_scholar_api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
        govinfo_api_key=os.getenv("GOVINFO_API_KEY") or os.getenv("CITATION_CHECKER_GOVINFO_API_KEY"),
        searxng_base_url=os.getenv("CITATION_CHECKER_SEARXNG_BASE_URL"),
        ncbi_api_key=os.getenv("NCBI_API_KEY") or os.getenv("CITATION_CHECKER_NCBI_API_KEY"),
        ncbi_email=os.getenv("NCBI_EMAIL") or os.getenv("CITATION_CHECKER_NCBI_EMAIL"),
        web_search_provider=os.getenv("CITATION_CHECKER_WEB_SEARCH_PROVIDER"),
        google_api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("CITATION_CHECKER_GOOGLE_API_KEY"),
        google_cse_id=os.getenv("GOOGLE_CSE_ID") or os.getenv("CITATION_CHECKER_GOOGLE_CSE_ID"),
        serpapi_key=os.getenv("SERPAPI_KEY") or os.getenv("CITATION_CHECKER_SERPAPI_KEY"),
        tavily_api_key=os.getenv("TAVILY_API_KEY") or os.getenv("CITATION_CHECKER_TAVILY_API_KEY"),
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
