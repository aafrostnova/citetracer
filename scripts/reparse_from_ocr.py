"""Re-extract citations from OCR intermediate results using Bedrock LLM only (no heuristic parsing)."""
from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apps.pdf_checker.config import load_pdf_checker_config
from apps.pdf_checker.ingest.citation_parser import (
    _run_llm_reparse_on_raw_bedrock,
    _extract_url,
)
from packages.core.models import CitationRecord
from packages.core.normalize import extract_identifier


def _raw_to_citation_via_bedrock(
    raw_text: str,
    citation_id: str,
    model_id: str,
    region: str,
    bearer_token: str | None,
    max_new_tokens: int,
    temperature: float,
) -> dict:
    """Parse a raw reference string entirely via Bedrock LLM."""
    parsed = _run_llm_reparse_on_raw_bedrock(
        raw_text=raw_text,
        model_id=model_id,
        region=region,
        bearer_token=bearer_token,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    if not parsed:
        # LLM failed — return minimal record with raw_text only
        return asdict(CitationRecord(
            citation_id=citation_id,
            raw_text=raw_text,
            parsed_fields={"parsed_from": "bedrock_llm_failed"},
        ))

    # Extract identifiers from raw text as fallback
    doi_fallback, arxiv_fallback = extract_identifier(raw_text)
    url_fallback = _extract_url(raw_text)

    title = str(parsed.get("title", "") or "").strip()
    authors = parsed.get("authors", []) or []
    venue = str(parsed.get("venue", "") or "").strip()
    year = parsed.get("year")
    volume = str(parsed.get("volume", "") or "").strip()
    pages = str(parsed.get("pages", "") or "").strip()
    publisher = str(parsed.get("publisher", "") or "").strip()
    location = str(parsed.get("location", "") or "").strip()
    doi = str(parsed.get("doi", "") or "").strip() or doi_fallback
    arxiv_id = str(parsed.get("arxiv_id", "") or "").strip() or arxiv_fallback
    url = str(parsed.get("url", "") or "").strip() or url_fallback

    record = CitationRecord(
        citation_id=citation_id,
        raw_text=raw_text,
        title=title,
        authors=authors,
        venue=venue,
        year=int(year) if year is not None else None,
        doi=doi.lower() if doi else "",
        arxiv_id=arxiv_id.lower() if arxiv_id else "",
        url=url,
        volume=volume,
        pages=pages,
        publisher=publisher,
        location=location,
        parsed_fields={"parsed_from": "bedrock_llm_only"},
    )
    return asdict(record)


def main() -> None:
    cfg = load_pdf_checker_config()

    input_dir = Path(".tmp/pdf_checker_local_ocr_pages")
    output_dir = Path("artifacts/iclr_2026_reference_extracts_v4")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Bedrock config from citation_reparse section
    bedrock_cfg = cfg.citation_reparse.bedrock
    model_id = str(bedrock_cfg.model_id or "")
    region = str(bedrock_cfg.region or "us-east-1")
    bearer_token = bedrock_cfg.bearer_token or cfg.entry_extraction.bedrock.bearer_token
    max_new_tokens = cfg.citation_reparse.max_new_tokens
    temperature = cfg.citation_reparse.temperature

    print(f"Bedrock LLM reparse: model_id={model_id} region={region}")
    print(f"  max_new_tokens={max_new_tokens} temperature={temperature}")
    print()

    folders = sorted(d for d in input_dir.iterdir() if d.is_dir())
    print(f"Found {len(folders)} papers\n")

    for idx, folder in enumerate(folders, 1):
        ref_json = folder / "model_outputs" / "reference_entries_from_markdown.json"
        if not ref_json.exists():
            print(f"[{idx}/{len(folders)}] SKIP {folder.name} (no reference_entries)")
            continue

        with open(ref_json) as f:
            data = json.load(f)

        raw_refs = data.get("references", [])
        entries = [r.get("raw_reference", "") for r in raw_refs if r.get("raw_reference", "").strip()]

        # Filter out non-reference entries (e.g. "Under review as a conference paper at...")
        entries = [e for e in entries if len(e) > 20 and not e.lower().startswith(("under review", "published as"))]

        if not entries:
            print(f"[{idx}/{len(folders)}] SKIP {folder.name} (no valid entries)")
            continue

        paper_name = folder.name.rsplit("_p", 1)[0]
        print(f"[{idx}/{len(folders)}] {paper_name}: {len(entries)} entries", flush=True)

        citations = []
        for i, raw in enumerate(entries, 1):
            cid = f"pdf-ref:{i}"
            try:
                citation = _raw_to_citation_via_bedrock(
                    raw_text=raw,
                    citation_id=cid,
                    model_id=model_id,
                    region=region,
                    bearer_token=bearer_token,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                citations.append(citation)
                title = citation.get("title", "")[:50]
                print(f"  [{cid}] {title}", flush=True)
            except Exception as e:
                print(f"  [{cid}] ERROR: {e}", flush=True)
                citations.append(asdict(CitationRecord(
                    citation_id=cid,
                    raw_text=raw,
                    parsed_fields={"parsed_from": "bedrock_llm_error", "error": str(e)},
                )))

        payload = {
            "input_pdf": str(folder),
            "reference_entries_count": len(entries),
            "extraction_quality": "UNKNOWN",
            "metadata": {"pipeline_method": "bedrock_llm_only"},
            "citations": citations,
        }

        out_path = output_dir / f"{paper_name}.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        has_title = sum(1 for c in citations if c.get("title"))
        has_pages = sum(1 for c in citations if c.get("pages"))
        has_pub = sum(1 for c in citations if c.get("publisher"))
        has_loc = sum(1 for c in citations if c.get("location"))
        print(f"  → {len(citations)} citations (title={has_title} pages={has_pages} pub={has_pub} loc={has_loc})\n", flush=True)

    print("Done!")


if __name__ == "__main__":
    main()
