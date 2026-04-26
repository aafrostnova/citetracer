"""Re-run only the boundary-merge and citation-reparse stages using the
already-OCR'd markdown cached under ``.tmp/pdf_checker_local_ocr_pages/``.

Reads each paper's cached ``reference_pages_markdown.mmd`` and produces an
``ocr_llm_extract``-shaped JSON with ``boundary_merge_mode='llm'``. Skips
the OCR stage entirely so the second comparison run does not need the GPU.

When ``--with-images`` is set, also renders the reference pages of the
source PDF and passes them to the LLM reparse stage as visual ground
truth, mirroring what ``apps/pdf_checker/run.py`` does in production. This
helps the LLM correct OCR character-level errors (dropped digits in DOIs,
wrong capitalization, etc.).

Usage:
    # Text-only reparse (no image)
    python scripts/reprocess_with_llm_merge.py --out results/extraction_ocr_llm

    # VLM-assisted reparse (with rendered page images)
    python scripts/reprocess_with_llm_merge.py --out results/extraction_ocr_llm_image \
        --with-images
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.pdf_checker.config import load_pdf_checker_config
from apps.pdf_checker.ingest import reference_segmenter as seg
from apps.pdf_checker.ingest.citation_parser import parse_reference_entries

OCR_CACHE_ROOT = Path(".tmp/pdf_checker_local_ocr_pages")
PDF_ROOT = Path("pdf_samples_full")
STYLES = ("acm", "splncs04", "iclr", "ieee", "plain")
CACHE_DIR_RE = re.compile(r"_p(\d+)-(\d+)_")


def find_latest_cache(style: str, paper: str) -> tuple[Path | None, int, int]:
    """Return (cache_dir, ref_start_1idx, ref_end_1idx) for the paper's most
    recent OCR cache. Page indices are 1-based as encoded in the cache dir
    name (e.g. ``acm_paper_001_p1-4_<ts>``)."""
    candidates = sorted(
        OCR_CACHE_ROOT.glob(f"{style}_{paper}_p*_*"),
        reverse=True,
    )
    for cache_dir in candidates:
        md = cache_dir / "model_outputs" / "reference_pages_markdown.mmd"
        if not md.exists():
            continue
        m = CACHE_DIR_RE.search(cache_dir.name)
        if not m:
            continue
        return cache_dir, int(m.group(1)), int(m.group(2))
    return None, 0, 0


def render_reference_pages(pdf_path: Path, ref_start_0idx: int, ref_end_0idx: int) -> dict[int, bytes]:
    """Render the reference pages of ``pdf_path`` as PNG bytes keyed by
    0-indexed page number. Mirrors apps/pdf_checker/run.py's logic."""
    page_images: dict[int, bytes] = {}
    try:
        from apps.pdf_checker.ingest.reference_segmenter import _render_pdf_reference_pages
        with tempfile.TemporaryDirectory() as tmpdir:
            image_paths = _render_pdf_reference_pages(
                pdf_path, ref_start_0idx, ref_end_0idx, Path(tmpdir), dpi=150,
            )
            for img_path in image_paths:
                m = re.search(r"reference_page_(\d+)\.png$", img_path.name)
                if m:
                    page_idx = int(m.group(1)) - 1  # 0-indexed
                    page_images[page_idx] = img_path.read_bytes()
    except Exception:
        pass
    return page_images


def map_entries_to_pages(
    entries: list[str],
    cache_dir: Path,
    ref_start_1idx: int,
    ref_end_1idx: int,
) -> list[list[int]]:
    """For each entry text, locate the source page(s) by searching the
    per-page OCR markdown files under the cache dir.
    Returns a list of 0-indexed page lists, aligned with ``entries``."""
    page_texts: list[tuple[int, str]] = []  # (0-indexed page, lowercased text)
    for page_1 in range(ref_start_1idx, ref_end_1idx + 1):
        ppath = cache_dir / "model_outputs" / f"page_{page_1}" / "page_reference_markdown.mmd"
        if ppath.exists():
            page_texts.append((page_1 - 1, ppath.read_text(encoding="utf-8").lower()))

    out: list[list[int]] = []
    for entry in entries:
        cleaned = re.sub(r"\s+", " ", entry).strip().lower()
        # First try a 60-char prefix probe (handles single-page entries)
        probe_head = cleaned[:60]
        hits: list[int] = []
        for pi, ptext in page_texts:
            if probe_head and probe_head in re.sub(r"\s+", " ", ptext):
                hits.append(pi)
        # Fallback: split into head + tail probes for cross-page entries
        if not hits and len(cleaned) > 80:
            head = cleaned[:40]
            tail = cleaned[-40:]
            for pi, ptext in page_texts:
                squashed = re.sub(r"\s+", " ", ptext)
                if head in squashed:
                    hits.append(pi)
                elif tail in squashed:
                    hits.append(pi)
        out.append(sorted(set(hits)))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rule-results", default="results/extraction_ocr_rule",
                   help="Directory holding the rule-mode results, used only to mirror the metadata defaults.")
    p.add_argument("--out", default="results/extraction_ocr_llm")
    p.add_argument("--llm-parallel-workers", type=int, default=8)
    p.add_argument("--styles", default=",".join(STYLES))
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip papers whose output JSON already exists.")
    p.add_argument("--watch", action="store_true",
                   help="Re-scan caches in a loop until all 50 papers are produced.")
    p.add_argument("--watch-target", type=int, default=50,
                   help="In --watch mode, exit when this many output JSONs exist.")
    p.add_argument("--watch-interval", type=int, default=30,
                   help="Seconds between scans in --watch mode.")
    p.add_argument("--with-images", action="store_true",
                   help="Render reference pages and pass them to the LLM reparse stage so it can visually correct OCR character errors (mirrors run.py).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_pdf_checker_config()

    bedrock_region = cfg.ocr_llm_extract.bedrock.region
    bedrock_token = cfg.ocr_llm_extract.bedrock.bearer_token
    bedrock_model = cfg.ocr_llm_extract.bedrock.model_id

    if not bedrock_model:
        raise RuntimeError("ocr_llm_extract.bedrock.model_id missing; cannot run LLM merge.")

    bedrock_client = seg._build_bedrock_client(region=bedrock_region, bearer_token=bedrock_token)

    llm_merge_kwargs = {
        "bedrock_client": bedrock_client,
        "model_id": bedrock_model,
        "max_new_tokens": 16,
        "temperature": 0.0,
    }

    llm_reparse_cfg = {
        "enabled": True,
        "provider": "bedrock",
        "model_path": "",
        "max_new_tokens": int(cfg.ocr_llm_extract.max_new_tokens),
        "temperature": float(cfg.ocr_llm_extract.temperature),
        "force_all_entries": bool(cfg.ocr_llm_extract.force_all_entries),
        "parallel_workers": max(1, int(args.llm_parallel_workers)),
        "bedrock": {
            "model_id": bedrock_model,
            "region": bedrock_region,
            "bearer_token": bedrock_token,
        },
    }

    out_root = Path(args.out)
    styles = [s.strip() for s in args.styles.split(",") if s.strip()]

    def scan_once() -> tuple[int, int, int]:
        n_done = n_skip = n_err = 0
        for style in styles:
            for paper_dir in sorted((PDF_ROOT / style).glob("paper_*")):
                paper = paper_dir.name
                out_path = out_root / style / f"{paper}.json"
                if args.skip_existing and out_path.exists():
                    continue
                cache_dir, ref_start_1, ref_end_1 = find_latest_cache(style, paper)
                if cache_dir is None:
                    n_skip += 1
                    continue
                md_path = cache_dir / "model_outputs" / "reference_pages_markdown.mmd"
                if not md_path.exists():
                    n_skip += 1
                    continue
                md = md_path.read_text(encoding="utf-8")
                seg_entries = seg._extract_reference_entries_from_tagged_markdown(md)
                meta_sink: dict = {}
                entries = seg._finalize_raw_entries(
                    seg_entries,
                    merge_mode="llm",
                    llm_merge_kwargs=llm_merge_kwargs,
                    metadata_sink=meta_sink,
                )
                stats = meta_sink.get("boundary_merge_stats", {})
                print(f"  [{style}/{paper}] segmenter={len(seg_entries)} -> finalize={len(entries)} "
                      f"(llm_called={stats.get('pairs_llm_called', 0)}, "
                      f"merged={stats.get('pairs_merged', 0)})", flush=True)

                # Build per-entry image bundles for VLM-assisted reparse.
                per_entry_images: list[list[bytes]] | None = None
                n_imgs_total = 0
                if args.with_images and ref_start_1 > 0:
                    pdf_path = paper_dir / "main.pdf"
                    page_images = render_reference_pages(pdf_path, ref_start_1 - 1, ref_end_1 - 1)
                    n_imgs_total = len(page_images)
                    if page_images:
                        entry_pages = map_entries_to_pages(entries, cache_dir, ref_start_1, ref_end_1)
                        per_entry_images = [[page_images[pi] for pi in pages if pi in page_images]
                                            for pages in entry_pages]
                    print(f"    images: rendered {n_imgs_total} pages, "
                          f"entries with images={sum(1 for x in (per_entry_images or []) if x)}/{len(entries)}",
                          flush=True)

                try:
                    citations = [
                        asdict(record)
                        for record in parse_reference_entries(
                            entries,
                            llm_reparse_config=llm_reparse_cfg,
                            reference_page_images=per_entry_images,
                        )
                    ]
                except Exception as exc:
                    print(f"    ERROR reparse {style}/{paper}: {type(exc).__name__}: {exc}", flush=True)
                    n_err += 1
                    continue
                payload = {
                    "input_pdf": str(paper_dir / "main.pdf"),
                    "reference_entries_count": len(entries),
                    "extraction_quality": "MEDIUM",
                    "metadata": {
                        "ocr_intermediate_dir": str(cache_dir),
                        "entry_extraction_mode": "model",
                        "entry_extraction_provider": "local",
                        "entry_extraction_input": "reference_page_images_markdown",
                        "entry_extraction_fallback": False,
                        "pipeline_method": "ocr_llm_extract",
                        "llm_reparse_provider_forced": "bedrock",
                        "llm_reparse_model_id": bedrock_model,
                        "llm_reparse_region": bedrock_region,
                        "llm_reparse_with_images": bool(args.with_images),
                        "llm_reparse_pages_rendered": n_imgs_total,
                        **meta_sink,
                    },
                    "citations": citations,
                }
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"    saved: {out_path} (citations={len(citations)})", flush=True)
                n_done += 1
        return n_done, n_skip, n_err

    if not args.watch:
        n_done, n_skip, n_err = scan_once()
        print(f"\nDone. ok={n_done} skip={n_skip} err={n_err} -> {out_root}", flush=True)
        return

    import time
    total_done = 0
    total_existing = sum(1 for s in styles for p in (out_root / s).glob("paper_*.json"))
    while True:
        n_done, n_skip, n_err = scan_once()
        total_done += n_done
        existing = sum(1 for s in styles for p in (out_root / s).glob("paper_*.json"))
        print(f"  [watch] this pass: ok={n_done} pending={n_skip} err={n_err} "
              f"| total existing={existing}/{args.watch_target}", flush=True)
        if existing >= args.watch_target:
            print(f"\nWatch done. total_processed={total_done} -> {out_root}", flush=True)
            return
        time.sleep(args.watch_interval)


if __name__ == "__main__":
    main()
