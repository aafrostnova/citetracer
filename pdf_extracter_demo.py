import argparse
from dataclasses import asdict
from pathlib import Path
import json
from apps.pdf_checker.config import load_pdf_checker_config
from apps.pdf_checker.ingest.citation_parser import parse_reference_entries
from apps.pdf_checker.ingest.pdf_text_extract import extract_pdf_pages
from apps.pdf_checker.ingest.reference_segmenter import segment_references_from_pages


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract references from PDF and parse to structured citation fields.")
    parser.add_argument("--input", required=True, help="Path to input PDF.")
    parser.add_argument("--out", default="artifacts/reference_extract_only.json", help="Path to output JSON.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    pdf = Path(args.input)
    out = Path(args.out)
    cfg = load_pdf_checker_config()

    pages = extract_pdf_pages(pdf)
    entries, quality, meta = segment_references_from_pages(
        pages,
        entry_extraction=cfg.entry_extraction.mode,
        model_provider=cfg.entry_extraction.provider,
        model_id=cfg.entry_extraction.bedrock.model_id,
        region=cfg.entry_extraction.bedrock.region,
        entry_chunk_chars=cfg.entry_extraction.chunk_chars,
        bearer_token=cfg.entry_extraction.bedrock.bearer_token,
        local_model_path=cfg.entry_extraction.local.model_path,
        source_pdf_path=pdf,
    )
    parsed_citations = [asdict(record) for record in parse_reference_entries(entries)]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "input_pdf": str(pdf),
                "reference_entries_count": len(entries),
                "extraction_quality": quality.value,
                "metadata": meta,
                "citations": parsed_citations,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"saved: {out} (citations={len(parsed_citations)})")


if __name__ == "__main__":
    main()
