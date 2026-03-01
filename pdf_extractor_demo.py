import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import json
from typing import Any, Dict, List

from apps.pdf_checker.config import load_pdf_checker_config
from apps.pdf_checker.ingest.citation_parser import parse_reference_entries
from apps.pdf_checker.ingest.pdf_text_extract import extract_pdf_pages
from apps.pdf_checker.ingest.reference_segmenter import segment_references_from_pages


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract references from PDF and parse to structured citation fields.")
    parser.add_argument("--input", required=True, help="Path to input PDF or a directory containing PDFs.")
    parser.add_argument(
        "--out",
        default="artifacts/reference_extract_only.json",
        help="Output JSON path for single-PDF mode, or output directory for batch mode.",
    )
    parser.add_argument(
        "--glob",
        default="*.pdf",
        help="Glob pattern for batch mode (default: *.pdf). Use **/*.pdf with --recursive for nested directories.",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively search for PDFs in batch mode.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip PDFs whose output JSON already exists.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue batch processing even if a PDF fails.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Batch concurrency. Recommended >1 for bedrock/heuristic, keep 1 for local GPU model.",
    )
    return parser


def _extract_pdf_to_payload(pdf: Path) -> Dict[str, Any]:
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

    return {
        "input_pdf": str(pdf),
        "reference_entries_count": len(entries),
        "extraction_quality": quality.value,
        "metadata": meta,
        "citations": parsed_citations,
    }


def _save_payload(payload: Dict[str, Any], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _run_single(pdf: Path, out: Path) -> None:
    payload = _extract_pdf_to_payload(pdf)
    _save_payload(payload, out)
    print(f"saved: {out} (citations={len(payload['citations'])})")


def _iter_pdfs(input_dir: Path, pattern: str, recursive: bool) -> List[Path]:
    if recursive:
        pdfs = sorted(p for p in input_dir.rglob(pattern) if p.is_file() and p.suffix.lower() == ".pdf")
    else:
        pdfs = sorted(p for p in input_dir.glob(pattern) if p.is_file() and p.suffix.lower() == ".pdf")
    return pdfs


def _process_one_batch_pdf(pdf: Path, out_path: Path) -> Dict[str, Any]:
    payload = _extract_pdf_to_payload(pdf)
    _save_payload(payload, out_path)
    return {
        "input_pdf": str(pdf),
        "output_json": str(out_path),
        "status": "ok",
        "reference_entries_count": payload["reference_entries_count"],
        "citations_count": len(payload["citations"]),
        "extraction_quality": payload["extraction_quality"],
    }


def _run_batch(
    input_dir: Path,
    out_dir: Path,
    pattern: str,
    recursive: bool,
    skip_existing: bool,
    continue_on_error: bool,
    workers: int,
) -> None:
    pdfs = _iter_pdfs(input_dir, pattern, recursive)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pdfs:
        raise RuntimeError(f"No PDF files found in {input_dir} with pattern {pattern!r}")

    cfg = load_pdf_checker_config()
    mode = (cfg.entry_extraction.mode or "heuristic").lower()
    provider = (cfg.entry_extraction.provider or "bedrock").lower()
    requested_workers = max(1, int(workers))
    effective_workers = requested_workers
    if requested_workers > 1 and mode == "model" and provider == "local":
        # Local DeepSeek-OCR on a single GPU is generally not safe/beneficial to run across papers concurrently.
        print("warning: local model extraction detected (local); forcing --workers=1 to avoid GPU OOM/contention")
        effective_workers = 1

    total = len(pdfs)
    successes = 0
    failures = 0
    skipped = 0
    manifest: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = []

    print(f"Batch extraction start: {total} PDFs (workers={effective_workers}, requested={requested_workers})")
    for idx, pdf in enumerate(pdfs, start=1):
        out_path = out_dir / f"{pdf.stem}.json"

        if skip_existing and out_path.exists():
            skipped += 1
            print(f"[{idx}/{total}] {pdf.name}\n  skip existing: {out_path.name}")
            manifest.append(
                {
                    "input_pdf": str(pdf),
                    "output_json": str(out_path),
                    "status": "skipped_existing",
                }
            )
            continue
        pending.append(
            {
                "seq": idx,
                "pdf": pdf,
                "out_path": out_path,
            }
        )

    if effective_workers <= 1:
        for item in pending:
            idx = item["seq"]
            pdf = item["pdf"]
            out_path = item["out_path"]
            print(f"[{idx}/{total}] {pdf.name}")
            try:
                result = _process_one_batch_pdf(pdf, out_path)
                successes += 1
                print(f"  saved: {out_path.name} (citations={result['citations_count']})")
                manifest.append(result)
            except Exception as exc:
                failures += 1
                print(f"  error: {type(exc).__name__}: {exc}")
                manifest.append(
                    {
                        "input_pdf": str(pdf),
                        "output_json": str(out_path),
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                if not continue_on_error:
                    manifest_path = out_dir / "_batch_manifest.json"
                    _save_payload(
                        {
                            "input_dir": str(input_dir),
                            "total": total,
                            "processed_before_stop": idx,
                            "successes": successes,
                            "failures": failures,
                            "skipped": skipped,
                            "results": manifest,
                        },
                        manifest_path,
                    )
                    raise
    else:
        for item in pending:
            print(f"[queue] [{item['seq']}/{total}] {item['pdf'].name}")

        with ThreadPoolExecutor(max_workers=effective_workers) as ex:
            future_map = {
                ex.submit(_process_one_batch_pdf, item["pdf"], item["out_path"]): item for item in pending
            }
            completed = 0
            for fut in as_completed(future_map):
                item = future_map[fut]
                completed += 1
                idx = item["seq"]
                pdf = item["pdf"]
                out_path = item["out_path"]
                try:
                    result = fut.result()
                    successes += 1
                    manifest.append(result)
                    print(
                        f"[done {completed}/{len(pending)}] [{idx}/{total}] {pdf.name}\n"
                        f"  saved: {out_path.name} (citations={result['citations_count']})"
                    )
                except Exception as exc:
                    failures += 1
                    manifest.append(
                        {
                            "input_pdf": str(pdf),
                            "output_json": str(out_path),
                            "status": "error",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                    print(f"[done {completed}/{len(pending)}] [{idx}/{total}] {pdf.name}\n  error: {type(exc).__name__}: {exc}")
                    if not continue_on_error:
                        for other in future_map:
                            other.cancel()
                        manifest_path = out_dir / "_batch_manifest.json"
                        _save_payload(
                            {
                                "input_dir": str(input_dir),
                                "total": total,
                                "processed_before_stop": completed + skipped,
                                "successes": successes,
                                "failures": failures,
                                "skipped": skipped,
                                "results": manifest,
                            },
                            manifest_path,
                        )
                        raise

    manifest.sort(key=lambda x: (x.get("input_pdf", ""), x.get("status", "")))

    manifest_path = out_dir / "_batch_manifest.json"
    _save_payload(
        {
            "input_dir": str(input_dir),
            "total": total,
            "workers_requested": requested_workers,
            "workers_used": effective_workers,
            "successes": successes,
            "failures": failures,
            "skipped": skipped,
            "results": manifest,
        },
        manifest_path,
    )
    print(
        f"Batch extraction done: total={total}, ok={successes}, skipped={skipped}, failed={failures}\n"
        f"manifest: {manifest_path}"
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    input_path = Path(args.input)
    out = Path(args.out)

    if input_path.is_file():
        _run_single(input_path, out)
        return

    if input_path.is_dir():
        _run_batch(
            input_dir=input_path,
            out_dir=out,
            pattern=args.glob,
            recursive=args.recursive,
            skip_existing=args.skip_existing,
            continue_on_error=args.continue_on_error,
            workers=args.workers,
        )
        return

    raise RuntimeError(f"Input path does not exist: {input_path}")


if __name__ == "__main__":
    main()
