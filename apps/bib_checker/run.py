"""End-to-end citation hallucination check on a BibTeX file (or directory of BibTeX files).

Mirrors apps.pdf_checker.run but skips Stage 1 (PDF -> reference entries),
because BibTeX records are already structured. Each ``@<type>{key, ...}``
entry is parsed into a CitationRecord and handed to the same cascading
multi-agent verifier the PDF entry point uses.

Usage:
    python -m apps.bib_checker.run --input refs.bib --out artifacts/bib_demo
    python -m apps.bib_checker.run --input dir/ --out artifacts/bib_batch
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from apps.pdf_checker.config import load_pdf_checker_config
from apps.pdf_checker.run import _build_verifier, build_arg_parser as _pdf_build_arg_parser, render_markdown_dict
from apps.source_checker.ingest.bib_parser import parse_bib_file
from packages.core.models import (
    CitationRecord,
    ExtractionQuality,
    citation_verdict_to_dict,
)
from packages.core.report import build_summary, compute_risk_score


_INT_RE = ("year",)


def _coerce_int(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _split_authors(value: str) -> list[str]:
    """BibTeX joins authors with ' and '. Be tolerant of newlines + extra space."""
    if not value:
        return []
    # Normalize whitespace and split on ' and '
    flat = " ".join(value.split())
    parts = [a.strip() for a in flat.split(" and ")]
    return [a for a in parts if a]


def _extract_arxiv_id(entry: dict[str, str]) -> str:
    """Recognize arXiv IDs across the common BibTeX conventions:

      - eprint=2310.12345, archivePrefix=arXiv (BibLaTeX-Biber)
      - eprinttype=arxiv, eprint=2310.12345 (alternative)
      - journal={arXiv preprint arXiv:2310.12345} (informal styles)
    """
    eprint = (entry.get("eprint") or "").strip()
    arch = (entry.get("archiveprefix") or entry.get("eprinttype") or "").strip().lower()
    if eprint and (not arch or arch in {"arxiv", "arxiv.org"}):
        return eprint
    # Fallback: scrape an arXiv ID out of journal / note / howpublished
    import re as _re
    for fld in ("journal", "note", "howpublished", "url"):
        m = _re.search(r"(?:arxiv[:\s]*|abs/)(\d{4}\.\d{4,5})", str(entry.get(fld) or ""), _re.I)
        if m:
            return m.group(1)
    return ""


def _bib_entry_to_citation(citation_key: str, entry: dict[str, str]) -> CitationRecord:
    """Map a parsed BibTeX entry (any @<type>{...}) into a CitationRecord.

    The parser accepts every entry type bib_parser.py recognises (article,
    inproceedings, incollection, book, inbook, misc, techreport, phdthesis,
    mastersthesis, manual, unpublished, ...). The field mapping below is
    deliberately tolerant of which fields are missing.
    """
    venue = (
        entry.get("booktitle")          # @inproceedings / @incollection
        or entry.get("journal")         # @article
        or entry.get("series")          # some @book / @incollection styles
        or entry.get("howpublished")    # @misc
        or ""
    )
    publisher = (
        entry.get("publisher")
        or entry.get("school")          # @phdthesis / @mastersthesis
        or entry.get("institution")     # @techreport
        or entry.get("organization")    # @manual
        or ""
    )
    return CitationRecord(
        citation_id=citation_key,
        title=entry.get("title", "") or "",
        authors=_split_authors(entry.get("author", "") or entry.get("editor", "")),
        venue=venue,
        year=_coerce_int(entry.get("year")),
        doi=entry.get("doi", "") or "",
        arxiv_id=_extract_arxiv_id(entry),
        url=entry.get("url", "") or "",
        volume=str(entry.get("volume", "") or ""),
        pages=str(entry.get("pages", "") or ""),
        publisher=publisher,
        location=entry.get("address", "") or "",
    )


def _collect_bib_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.rglob("*.bib"))


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run end-to-end citation hallucination check on BibTeX input."
    )
    p.add_argument("--input", required=True,
                   help="Path to a .bib file or directory of .bib files.")
    p.add_argument("--out", required=True,
                   help="Output directory; one <stem>_report.{json,md} per .bib.")
    p.add_argument("--cache-path", default="",
                   help="Connector cache sqlite path. Empty = use config.")
    p.add_argument("--dblp-mirror-path", default="",
                   help="Offline DBLP mirror JSONL path. Empty = use config.")
    p.add_argument("--dblp-sqlite-path", default="",
                   help="DBLP sqlite path override. Empty = use config.")
    p.add_argument("--semantic-scholar-api-key", default="",
                   help="Semantic Scholar API key override. Empty = use config.")
    p.add_argument("--offline-only", action="store_true",
                   help="Skip every online connector.")
    p.add_argument("--citation-workers", type=int, default=8,
                   help="Parallel citation verifications per file (default: 8).")
    p.add_argument("--connector-workers", type=int, default=8,
                   help="Parallel connector calls per citation (default: 8).")
    p.add_argument("--paper-workers", type=int, default=2,
                   help="Parallel .bib files when --input is a directory (default: 2).")
    p.add_argument("--resume", action="store_true",
                   help="Skip .bib files whose <stem>_report.md already exists.")
    return p


def _verify_one_bib(
    bib_path: Path,
    out_dir: Path,
    verifier,
    citation_workers: int,
) -> dict[str, Any]:
    stem = bib_path.stem
    out_json = out_dir / f"{stem}_report.json"
    out_md = out_dir / f"{stem}_report.md"

    entries = parse_bib_file(bib_path)
    if not entries:
        print(f"[skip] {bib_path.name}: no @<type>{{key, ...}} entries parsed", flush=True)
        return {"path": str(bib_path), "status": "empty"}

    citations = [_bib_entry_to_citation(k, e) for k, e in entries.items()]
    print(f"[verify] {bib_path.name}: {len(citations)} citations", flush=True)

    t0 = time.perf_counter()
    verdict_objs: list[Any] = [None] * len(citations)

    def _do(i_cit):
        i, cit = i_cit
        return i, verifier.verify_citation(
            cit, extraction_quality=ExtractionQuality.HIGH, source_paper_title="",
        )

    with ThreadPoolExecutor(
        max_workers=citation_workers, thread_name_prefix=f"bib-{stem[:30]}"
    ) as pool:
        for fut in as_completed({pool.submit(_do, item): item for item in enumerate(citations)}):
            i, v = fut.result()
            verdict_objs[i] = v

    elapsed = time.perf_counter() - t0
    summary = build_summary(verdict_objs)
    risk = compute_risk_score(verdict_objs)

    report = {
        "report_version": "1.0",
        "paper_id": stem,
        "pipeline_type": "BIB",
        "runtime_s": round(elapsed, 2),
        "citations": [citation_verdict_to_dict(v) for v in verdict_objs],
        "summary": summary,
        "risk_score": risk,
        "requires_human_review": any(v.needs_human_review for v in verdict_objs),
        "metadata": {
            "input_bib": str(bib_path),
            "n_entries": len(citations),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    if render_markdown_dict is not None:
        try:
            out_md.write_text(render_markdown_dict(report), encoding="utf-8")
        except Exception as exc:
            print(f"  [md-skip] {bib_path.name}: {exc}", flush=True)

    print(f"[done] {bib_path.name}: {len(citations)} citations in {elapsed:.1f}s "
          f"-> {out_json.name}", flush=True)
    return {"path": str(bib_path), "status": "ok", "n": len(citations)}


def main() -> None:
    args = _build_arg_parser().parse_args()
    input_path = Path(args.input).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bib_files = _collect_bib_files(input_path)
    if not bib_files:
        raise SystemExit(f"no .bib files found at {input_path}")

    if args.resume:
        before = len(bib_files)
        bib_files = [b for b in bib_files
                     if not (out_dir / f"{b.stem}_report.md").exists()]
        print(f"[resume] {before - len(bib_files)} already done, "
              f"{len(bib_files)} remaining", flush=True)

    if args.offline_only:
        os.environ["CITATION_CHECKER_OFFLINE_ONLY"] = "1"

    cfg = load_pdf_checker_config()
    cli_args = _pdf_build_arg_parser().parse_args([
        "--input", "x", "--out", str(out_dir),
        "--citation-workers", str(args.citation_workers),
        "--connector-workers", str(args.connector_workers),
        *(["--cache-path", args.cache_path] if args.cache_path else []),
        *(["--dblp-mirror-path", args.dblp_mirror_path] if args.dblp_mirror_path else []),
        *(["--dblp-sqlite-path", args.dblp_sqlite_path] if args.dblp_sqlite_path else []),
        *(["--semantic-scholar-api-key", args.semantic_scholar_api_key]
          if args.semantic_scholar_api_key else []),
    ])
    print("[init] building verifier (full connector list + cascading agents)...",
          flush=True)
    verifier = _build_verifier(cfg, cli_args, verbose=False)

    if args.paper_workers <= 1 or len(bib_files) <= 1:
        for bib in bib_files:
            _verify_one_bib(bib, out_dir, verifier, args.citation_workers)
    else:
        with ThreadPoolExecutor(
            max_workers=args.paper_workers, thread_name_prefix="bib-paper"
        ) as pool:
            futs = {
                pool.submit(_verify_one_bib, bib, out_dir, verifier, args.citation_workers): bib
                for bib in bib_files
            }
            for fut in as_completed(futs):
                fut.result()

    print(f"\n[summary] processed {len(bib_files)} .bib file(s) -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
