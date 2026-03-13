from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from dataclasses import asdict
import json
from pathlib import Path
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
        "--method",
        default="config",
        choices=["config", "ocr_llm_extract", "ocr_parse_llm_fix"],
        help=(
            "Extraction method: "
            "config=use config.json as-is; "
            "ocr_llm_extract=local OCR extraction + Bedrock LLM structured extraction for all entries; "
            "ocr_parse_llm_fix=backward-compatible alias of ocr_llm_extract."
        ),
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
        help="Batch concurrency. Keep 1 when local OCR model is used.",
    )
    parser.add_argument(
        "--bedrock-model-id",
        default="",
        help="Optional Bedrock model id override for ocr_llm_extract.",
    )
    parser.add_argument(
        "--bedrock-region",
        default="",
        help="Optional Bedrock region override for ocr_llm_extract.",
    )
    parser.add_argument(
        "--bedrock-bearer-token",
        default="",
        help="Optional Bedrock bearer token override for ocr_llm_extract.",
    )
    parser.add_argument(
        "--llm-max-new-tokens",
        type=int,
        default=0,
        help="Optional override for ocr_llm_extract.max_new_tokens when > 0.",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=-1.0,
        help="Optional override for ocr_llm_extract.temperature when >= 0.",
    )
    parser.add_argument(
        "--llm-parallel-workers",
        type=int,
        default=8,
        help="Parallel workers for Bedrock LLM reparse calls per PDF (ocr_llm_extract).",
    )
    parser.add_argument(
        "--llm-stage-processes",
        type=int,
        default=1,
        help="Process count for LLM stage in ocr_llm_extract batch pipeline.",
    )
    parser.add_argument(
        "--pipeline-max-inflight",
        type=int,
        default=2,
        help="Max in-flight PDFs buffered between OCR stage and LLM stage in ocr_llm_extract batch mode.",
    )
    parser.add_argument(
        "--disable-ocr-llm-pipeline",
        action="store_true",
        help="Disable two-stage OCR->LLM pipeline optimization for ocr_llm_extract batch mode.",
    )
    return parser


def _extract_pdf_to_payload_config(pdf: Path, args: argparse.Namespace) -> Dict[str, Any]:
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
        local_inference_backend=cfg.entry_extraction.local.inference_backend,
        source_pdf_path=pdf,
    )
    parsed_citations = [
        asdict(record)
        for record in parse_reference_entries(
            entries,
            llm_reparse_config=asdict(cfg.citation_reparse),
        )
    ]

    metadata = dict(meta or {})
    metadata["pipeline_method"] = "config"
    return {
        "input_pdf": str(pdf),
        "reference_entries_count": len(entries),
        "extraction_quality": quality.value,
        "metadata": metadata,
        "citations": parsed_citations,
    }


def _prepare_ocr_llm_extract_intermediate(pdf: Path, args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_pdf_checker_config()
    extract_cfg = cfg.ocr_llm_extract.entry_extraction
    extract_mode = str(extract_cfg.mode or "model").strip().lower()
    extract_provider = str(extract_cfg.provider or "local").strip().lower()
    local_model_path = str(extract_cfg.local.model_path or "").strip()
    if extract_provider == "local" and extract_mode == "model" and not local_model_path:
        raise RuntimeError(
            "method=ocr_llm_extract requires ocr_llm_extract.entry_extraction.local.model_path in config "
            "when provider=local and mode=model."
        )

    if not cfg.ocr_llm_extract.enabled and not str(args.bedrock_model_id or "").strip():
        raise RuntimeError(
            "method=ocr_llm_extract is disabled in config. "
            "Set ocr_llm_extract.enabled=true or pass --bedrock-model-id explicitly."
        )

    model_id = (
        str(args.bedrock_model_id or "").strip()
        or str(cfg.ocr_llm_extract.bedrock.model_id or "").strip()
    )
    region = (
        str(args.bedrock_region or "").strip()
        or str(cfg.ocr_llm_extract.bedrock.region or "").strip()
        or "us-east-1"
    )
    bearer_token = (
        str(args.bedrock_bearer_token or "").strip()
        or str(cfg.ocr_llm_extract.bedrock.bearer_token or "").strip()
        or None
    )
    if not model_id:
        raise RuntimeError("method=ocr_llm_extract requires Bedrock model_id (config or --bedrock-model-id).")

    pages = extract_pdf_pages(pdf)
    entries, quality, meta = segment_references_from_pages(
        pages,
        entry_extraction=extract_mode,
        model_provider=extract_provider,
        model_id=extract_cfg.bedrock.model_id,
        region=extract_cfg.bedrock.region,
        entry_chunk_chars=extract_cfg.chunk_chars,
        bearer_token=extract_cfg.bedrock.bearer_token,
        local_model_path=local_model_path,
        local_inference_backend=extract_cfg.local.inference_backend,
        source_pdf_path=pdf,
    )

    llm_cfg = {
        "enabled": True,
        "provider": "bedrock",
        "model_path": "",
        "max_new_tokens": int(cfg.ocr_llm_extract.max_new_tokens),
        "temperature": float(cfg.ocr_llm_extract.temperature),
        "force_all_entries": bool(cfg.ocr_llm_extract.force_all_entries),
        "parallel_workers": max(1, int(args.llm_parallel_workers)),
        "bedrock": {
            "model_id": model_id,
            "region": region,
            "bearer_token": bearer_token,
        },
    }
    if int(args.llm_max_new_tokens) > 0:
        llm_cfg["max_new_tokens"] = int(args.llm_max_new_tokens)
    if float(args.llm_temperature) >= 0.0:
        llm_cfg["temperature"] = float(args.llm_temperature)

    metadata = dict(meta or {})
    metadata.update(
        {
            "pipeline_method": "ocr_llm_extract",
            "entry_extraction_mode_forced": extract_mode,
            "entry_extraction_provider_forced": extract_provider,
            "llm_reparse_provider_forced": "bedrock",
            "llm_reparse_model_id": model_id,
            "llm_reparse_region": region,
            "llm_reparse_force_all_entries": bool(llm_cfg["force_all_entries"]),
            "llm_reparse_max_new_tokens": int(llm_cfg["max_new_tokens"]),
            "llm_reparse_temperature": float(llm_cfg["temperature"]),
            "llm_reparse_parallel_workers": int(llm_cfg["parallel_workers"]),
        }
    )
    return {
        "input_pdf": str(pdf),
        "reference_entries_count": len(entries),
        "extraction_quality": quality.value,
        "metadata": metadata,
        "entries": entries,
        "llm_cfg": llm_cfg,
    }


def _finalize_ocr_llm_extract_payload(intermediate: Dict[str, Any]) -> Dict[str, Any]:
    entries = list(intermediate.get("entries", []) or [])
    llm_cfg = dict(intermediate.get("llm_cfg", {}) or {})
    parsed_citations = [
        asdict(record)
        for record in parse_reference_entries(
            entries,
            llm_reparse_config=llm_cfg,
        )
    ]

    metadata = dict(intermediate.get("metadata", {}) or {})
    return {
        "input_pdf": str(intermediate.get("input_pdf", "")),
        "reference_entries_count": int(intermediate.get("reference_entries_count", len(entries))),
        "extraction_quality": str(intermediate.get("extraction_quality", "")),
        "metadata": metadata,
        "citations": parsed_citations,
    }


def _extract_pdf_to_payload_ocr_llm_extract(pdf: Path, args: argparse.Namespace) -> Dict[str, Any]:
    intermediate = _prepare_ocr_llm_extract_intermediate(pdf, args)
    return _finalize_ocr_llm_extract_payload(intermediate)


def _extract_pdf_to_payload(pdf: Path, args: argparse.Namespace) -> Dict[str, Any]:
    method = "ocr_llm_extract" if args.method == "ocr_parse_llm_fix" else args.method
    if method == "ocr_llm_extract":
        return _extract_pdf_to_payload_ocr_llm_extract(pdf, args)
    return _extract_pdf_to_payload_config(pdf, args)


def _save_payload(payload: Dict[str, Any], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _run_single(pdf: Path, out: Path, args: argparse.Namespace) -> None:
    payload = _extract_pdf_to_payload(pdf, args)
    _save_payload(payload, out)
    print(f"saved: {out} (citations={len(payload['citations'])}, method={args.method})")


def _iter_pdfs(input_dir: Path, pattern: str, recursive: bool) -> List[Path]:
    if recursive:
        pdfs = sorted(p for p in input_dir.rglob(pattern) if p.is_file() and p.suffix.lower() == ".pdf")
    else:
        pdfs = sorted(p for p in input_dir.glob(pattern) if p.is_file() and p.suffix.lower() == ".pdf")
    return pdfs


def _process_one_batch_pdf(pdf: Path, out_path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    payload = _extract_pdf_to_payload(pdf, args)
    _save_payload(payload, out_path)
    return {
        "input_pdf": str(pdf),
        "output_json": str(out_path),
        "status": "ok",
        "method": args.method,
        "reference_entries_count": payload["reference_entries_count"],
        "citations_count": len(payload["citations"]),
        "extraction_quality": payload["extraction_quality"],
    }


def _run_batch_ocr_llm_pipeline(
    pending: List[Dict[str, Any]],
    total: int,
    continue_on_error: bool,
    args: argparse.Namespace,
    manifest: List[Dict[str, Any]],
) -> tuple[int, int]:
    llm_stage_processes = max(1, int(args.llm_stage_processes))
    max_inflight = max(1, int(args.pipeline_max_inflight))
    successes = 0
    failures = 0
    inflight: Dict[Any, Dict[str, Any]] = {}

    print(
        "ocr_llm_extract pipeline enabled: "
        f"ocr_stage=main_process, llm_stage_processes={llm_stage_processes}, "
        f"llm_parallel_workers={max(1, int(args.llm_parallel_workers))}, "
        f"max_inflight={max_inflight}"
    )

    def _drain_completed(block: bool) -> None:
        nonlocal successes, failures
        if not inflight:
            return
        timeout = None if block else 0
        done, _ = wait(inflight.keys(), timeout=timeout, return_when=FIRST_COMPLETED)
        if not done:
            return
        for fut in done:
            item = inflight.pop(fut)
            idx = item["seq"]
            pdf = item["pdf"]
            out_path = item["out_path"]
            try:
                payload = fut.result()
                _save_payload(payload, out_path)
                successes += 1
                manifest.append(
                    {
                        "input_pdf": str(pdf),
                        "output_json": str(out_path),
                        "status": "ok",
                        "method": "ocr_llm_extract",
                        "reference_entries_count": payload["reference_entries_count"],
                        "citations_count": len(payload["citations"]),
                        "extraction_quality": payload["extraction_quality"],
                    }
                )
                print(f"[llm done] [{idx}/{total}] {pdf.name}\n  saved: {out_path.name} (citations={len(payload['citations'])})")
            except Exception as exc:
                failures += 1
                manifest.append(
                    {
                        "input_pdf": str(pdf),
                        "output_json": str(out_path),
                        "status": "error",
                        "method": "ocr_llm_extract",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                print(f"[llm done] [{idx}/{total}] {pdf.name}\n  error: {type(exc).__name__}: {exc}")
                if not continue_on_error:
                    for rest in list(inflight.keys()):
                        rest.cancel()
                    raise

    with ProcessPoolExecutor(max_workers=llm_stage_processes) as llm_pool:
        for item in pending:
            idx = item["seq"]
            pdf = item["pdf"]
            out_path = item["out_path"]
            print(f"[ocr stage] [{idx}/{total}] {pdf.name}")
            try:
                intermediate = _prepare_ocr_llm_extract_intermediate(pdf, args)
                fut = llm_pool.submit(_finalize_ocr_llm_extract_payload, intermediate)
                inflight[fut] = item
                print(
                    f"  queued to llm stage: {out_path.name} "
                    f"(entries={intermediate.get('reference_entries_count', 0)})"
                )
            except Exception as exc:
                failures += 1
                manifest.append(
                    {
                        "input_pdf": str(pdf),
                        "output_json": str(out_path),
                        "status": "error",
                        "method": "ocr_llm_extract",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                print(f"  ocr stage error: {type(exc).__name__}: {exc}")
                if not continue_on_error:
                    for rest in list(inflight.keys()):
                        rest.cancel()
                    raise
            while len(inflight) >= max_inflight:
                _drain_completed(block=True)

        while inflight:
            _drain_completed(block=True)

    return successes, failures


def _run_batch(
    input_dir: Path,
    out_dir: Path,
    pattern: str,
    recursive: bool,
    skip_existing: bool,
    continue_on_error: bool,
    workers: int,
    args: argparse.Namespace,
) -> None:
    pdfs = _iter_pdfs(input_dir, pattern, recursive)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pdfs:
        raise RuntimeError(f"No PDF files found in {input_dir} with pattern {pattern!r}")

    requested_workers = max(1, int(workers))
    effective_workers = requested_workers
    method = "ocr_llm_extract" if args.method == "ocr_parse_llm_fix" else args.method
    if requested_workers > 1 and method == "ocr_llm_extract":
        print("warning: method=ocr_llm_extract uses local OCR model; forcing --workers=1 to avoid GPU contention")
        effective_workers = 1
    elif requested_workers > 1 and method == "config":
        cfg = load_pdf_checker_config()
        mode = (cfg.entry_extraction.mode or "heuristic").lower()
        provider = (cfg.entry_extraction.provider or "bedrock").lower()
        if mode == "model" and provider == "local":
            print("warning: local model extraction detected (config); forcing --workers=1 to avoid GPU contention")
            effective_workers = 1

    total = len(pdfs)
    successes = 0
    failures = 0
    skipped = 0
    manifest: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = []

    print(
        f"Batch extraction start: {total} PDFs "
        f"(workers={effective_workers}, requested={requested_workers}, method={method})"
    )
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
                    "method": method,
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
        if method == "ocr_llm_extract" and not args.disable_ocr_llm_pipeline:
            try:
                ok, err = _run_batch_ocr_llm_pipeline(
                    pending=pending,
                    total=total,
                    continue_on_error=continue_on_error,
                    args=args,
                    manifest=manifest,
                )
                successes += ok
                failures += err
            except Exception:
                manifest_path = out_dir / "_batch_manifest.json"
                _save_payload(
                    {
                        "input_dir": str(input_dir),
                        "total": total,
                        "processed_before_stop": successes + failures + skipped,
                        "method": method,
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
                idx = item["seq"]
                pdf = item["pdf"]
                out_path = item["out_path"]
                print(f"[{idx}/{total}] {pdf.name}")
                try:
                    result = _process_one_batch_pdf(pdf, out_path, args)
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
                            "method": method,
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
                                "method": method,
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
                ex.submit(_process_one_batch_pdf, item["pdf"], item["out_path"], args): item for item in pending
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
                            "method": method,
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
                                "method": method,
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
            "method": method,
            "workers_requested": requested_workers,
            "workers_used": effective_workers,
            "llm_parallel_workers": max(1, int(args.llm_parallel_workers)),
            "llm_stage_processes": max(1, int(args.llm_stage_processes)),
            "pipeline_max_inflight": max(1, int(args.pipeline_max_inflight)),
            "ocr_llm_pipeline_enabled": bool(method == "ocr_llm_extract" and not args.disable_ocr_llm_pipeline),
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
        _run_single(input_path, out, args)
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
            args=args,
        )
        return

    raise RuntimeError(f"Input path does not exist: {input_path}")


if __name__ == "__main__":
    main()
