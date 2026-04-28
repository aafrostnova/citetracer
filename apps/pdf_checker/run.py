"""End-to-end PDF citation hallucination check pipeline.

Combines the extraction modes from `pdf_extractor_demo.py` (heuristic config /
ocr_llm_extract) with the verifier wiring from `citation_verification_demo.py`
(cascading 3-agent + secondary verifier + optional source router).

Replaces the old minimal `run_pdf_check` which used only rule-based adjudication.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from packages.connectors import default_orchestrator
from packages.connectors.base import ConnectorOrchestrator
from packages.core import CitationVerifier
from packages.core.models import CitationRecord, ExtractionQuality
from packages.core.report import build_summary, compute_risk_score, render_markdown
from packages.core.models import citation_verdict_to_dict

from .config import load_pdf_checker_config
from .ingest.citation_marker_linker import estimate_link_quality, extract_inline_markers
from .ingest.citation_parser import parse_reference_entries, parse_reference_entries_streaming
from .ingest.pdf_text_extract import extract_pdf_pages
from .ingest.reference_segmenter import segment_references_from_pages


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _env_flag(name: str, default: str = "1") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _log(enabled: bool, message: str) -> None:
    if enabled:
        print(message, flush=True)


def _wrap_for_terminal(text: str, prefix: str = "", width: int = 200) -> str:
    """Wrap a long string for terminal output, adding `prefix` to the first line
    and an indent equal to the prefix's length on continuation lines.
    """
    import textwrap
    indent = " " * len(prefix)
    wrapped = textwrap.wrap(text, width=width - len(prefix))
    if not wrapped:
        return prefix
    return prefix + wrapped[0] + ("\n" + "\n".join(indent + line for line in wrapped[1:]) if len(wrapped) > 1 else "")


# ---------------------------------------------------------------------------
# Step 1+2+3: PDF → reference entries → parsed CitationRecords
# (mirrors pdf_extractor_demo.py's extraction logic)
# ---------------------------------------------------------------------------

def _build_llm_reparse_config(cfg) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the llm_reparse_config dict and metadata describing it.

    Path is selected by `cfg.citation_parse_method`:
      - "citation_reparse" (default) → use cfg.citation_reparse as-is
      - "ocr_llm_extract"            → use cfg.ocr_llm_extract (force-all + bedrock)
    """
    method = cfg.citation_parse_method

    if method == "citation_reparse":
        llm_cfg = asdict(cfg.citation_reparse)
        meta = {
            "pipeline_method": "citation_reparse",
            "llm_reparse_enabled": bool(llm_cfg.get("enabled")),
            "llm_reparse_provider": str(llm_cfg.get("provider", "")),
        }
        return llm_cfg, meta

    # ocr_llm_extract — force-all Bedrock reparse on every entry
    if not cfg.ocr_llm_extract.enabled:
        raise RuntimeError(
            "citation_parse_method=ocr_llm_extract requires ocr_llm_extract.enabled=true "
            "and a valid bedrock model_id in config.json."
        )

    model_id = str(cfg.ocr_llm_extract.bedrock.model_id or "").strip()
    region = str(cfg.ocr_llm_extract.bedrock.region or "").strip() or "us-east-1"
    bearer_token = str(cfg.ocr_llm_extract.bedrock.bearer_token or "").strip() or None
    if not model_id:
        raise RuntimeError(
            "citation_parse_method=ocr_llm_extract requires "
            "ocr_llm_extract.bedrock.model_id in config.json."
        )

    llm_cfg = {
        "enabled": True,
        "provider": "bedrock",
        "model_path": "",
        "max_new_tokens": int(cfg.ocr_llm_extract.max_new_tokens),
        "temperature": float(cfg.ocr_llm_extract.temperature),
        "force_all_entries": bool(cfg.ocr_llm_extract.force_all_entries),
        "parallel_workers": 8,
        "bedrock": {
            "model_id": model_id,
            "region": region,
            "bearer_token": bearer_token,
        },
    }

    meta = {
        "pipeline_method": "ocr_llm_extract",
        "llm_reparse_provider_forced": "bedrock",
        "llm_reparse_model_id": model_id,
        "llm_reparse_region": region,
        "llm_reparse_force_all_entries": bool(llm_cfg["force_all_entries"]),
        "llm_reparse_max_new_tokens": int(llm_cfg["max_new_tokens"]),
        "llm_reparse_temperature": float(llm_cfg["temperature"]),
        "llm_reparse_parallel_workers": int(llm_cfg["parallel_workers"]),
    }
    return llm_cfg, meta


def _extract_and_parse(
    pdf: Path, cfg, args: argparse.Namespace, verbose: bool, streaming: bool = False,
):
    """Run steps 1-3: PDF → pages → reference entries → CitationRecord[].

    When streaming=True returns (records, reparse_pool, reparse_futures,
    extraction_quality, extraction_metadata, pages); reparse_pool/futures may
    be None/{} if no LLM reparse is needed.
    When streaming=False returns the legacy tuple
    (citations, extraction_quality, extraction_metadata, pages).
    """
    _log(verbose, f"[1/7] Reading PDF pages: {pdf}")
    pages = extract_pdf_pages(pdf)
    _log(verbose, f"      Loaded {len(pages)} pages")

    _log(verbose, "[2/7] Extracting reference entries")
    reference_entries, extraction_quality, extraction_metadata = segment_references_from_pages(
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
        boundary_merge_mode=cfg.ocr_llm_extract.boundary_merge_mode,
        boundary_merge_model_id=cfg.ocr_llm_extract.bedrock.model_id,
        boundary_merge_region=cfg.ocr_llm_extract.bedrock.region,
        boundary_merge_bearer_token=cfg.ocr_llm_extract.bedrock.bearer_token,
    )
    _log(
        verbose,
        f"      Extracted {len(reference_entries)} reference entries "
        f"(mode={extraction_metadata.get('entry_extraction_mode')}, "
        f"provider={extraction_metadata.get('entry_extraction_provider')})",
    )

    llm_cfg, llm_meta = _build_llm_reparse_config(cfg)
    _log(verbose, f"[3/7] Parsing references (citation_parse_method={cfg.citation_parse_method})")

    # Render reference pages as images for VLM-assisted reparse (fixes OCR char errors).
    # Build a per-citation image mapping: each citation gets the specific page(s) it appears on.
    ref_start = extraction_metadata.get("reference_page_start", 0) - 1  # 0-indexed
    ref_end = extraction_metadata.get("reference_page_end", 0) - 1
    page_images: dict[int, bytes] = {}  # 0-indexed page → PNG bytes
    if ref_start >= 0 and ref_end >= ref_start:
        try:
            from apps.pdf_checker.ingest.reference_segmenter import _render_pdf_reference_pages
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                # Render at 250 DPI so the VLM has ~30 px per 9pt character —
                # enough to distinguish 'i'/'a', 'g'/'i' (Haignet vs HiAgent,
                # Thainyi vs Tihanyi) which 150 DPI loses. Bedrock Converse
                # accepts up to 3.75 MB / image; at 250 DPI a Letter page is
                # ~2125x2750 px and the PNG stays under 2 MB.
                image_paths = _render_pdf_reference_pages(pdf, ref_start, ref_end, Path(tmpdir), dpi=250)
                for img_path in image_paths:
                    # Parse page number from filename: "reference_page_12.png" → page_idx=11
                    import re as _re
                    m = _re.search(r"reference_page_(\d+)\.png$", img_path.name)
                    if m:
                        page_idx = int(m.group(1)) - 1  # 0-indexed
                        page_images[page_idx] = img_path.read_bytes()
            if page_images:
                _log(verbose, f"      Rendered {len(page_images)} reference page image(s) for VLM reparse")
        except ImportError as exc:
            # pymupdf missing means VLM reparse silently degrades to text-only,
            # which costs us OCR-error correction (Haignet→HiAgent etc).
            # Refuse instead of silently producing worse output.
            raise RuntimeError(
                f"pymupdf is required for VLM-assisted reparse but is not installed: {exc}. "
                f"Install with `pip install pymupdf`."
            ) from exc
        except Exception as exc:
            _log(verbose, f"      Could not render reference pages: {exc}")

    # Map each reference entry to the page(s) it appears on.
    # Match by checking which reference page text contains the entry's raw_text.
    ref_page_texts: list[tuple[int, str]] = []  # (0-indexed page, text)
    for pi in range(max(0, ref_start), min(ref_end + 1, len(pages))):
        ref_page_texts.append((pi, pages[pi]))

    import re as _re_token
    def _entry_tokens(text: str) -> set[str]:
        return set(_re_token.findall(r"[A-Za-z]{4,}", text.lower()))

    per_entry_images: list[list[bytes]] = []
    page_token_cache: list[tuple[int, str, set[str]]] = [
        (pi, ptext, _entry_tokens(ptext)) for pi, ptext in ref_page_texts
    ]

    for entry_text in reference_entries:
        entry_pages: list[int] = []
        # Use first 60 chars of entry as search key (enough to locate, handles line breaks)
        search_key = entry_text[:60].strip().lower()
        for pi, ptext, _ in page_token_cache:
            if search_key and search_key in ptext.lower():
                entry_pages.append(pi)
        # If not found by prefix, check if entry spans consecutive pages
        if not entry_pages and len(entry_text) > 80:
            first_half = entry_text[:40].strip().lower()
            second_half = entry_text[-40:].strip().lower()
            for pi, ptext, _ in page_token_cache:
                pl = ptext.lower()
                if first_half in pl:
                    entry_pages.append(pi)
                elif second_half in pl:
                    entry_pages.append(pi)
        # Final fallback: token-overlap. Substring matching breaks when OCR
        # has typos ("Haignet" vs "HiAgent") so the entry never finds its
        # page; use word-set intersection to recover those cases.
        if not entry_pages:
            etokens = _entry_tokens(entry_text)
            if len(etokens) >= 5:
                for pi, _, ptokens in page_token_cache:
                    overlap = len(etokens & ptokens) / max(len(etokens), 1)
                    if overlap >= 0.4:
                        entry_pages.append(pi)
        entry_pages = sorted(set(entry_pages))
        # Collect images for these pages
        imgs = [page_images[pi] for pi in entry_pages if pi in page_images]
        per_entry_images.append(imgs)

    no_image_idxs = [i + 1 for i, imgs in enumerate(per_entry_images) if not imgs]
    if no_image_idxs:
        _log(verbose, f"      [WARN] {len(no_image_idxs)}/{len(reference_entries)} entries got no page image: pdf-ref indices {no_image_idxs[:15]}")

    if streaming:
        citations, reparse_pool, reparse_futures = parse_reference_entries_streaming(
            reference_entries,
            llm_reparse_config=llm_cfg,
            reference_page_images=per_entry_images,
        )
        _log(verbose, f"      Parsed {len(citations)} citation records "
                     f"({len(reparse_futures)} awaiting LLM reparse)")
        extraction_metadata.update(llm_meta)
        return citations, reparse_pool, reparse_futures, extraction_quality, extraction_metadata, pages

    citations = parse_reference_entries(
        reference_entries,
        llm_reparse_config=llm_cfg,
        reference_page_images=per_entry_images,
    )
    _log(verbose, f"      Parsed {len(citations)} citation records")

    extraction_metadata.update(llm_meta)
    return citations, extraction_quality, extraction_metadata, pages


# ---------------------------------------------------------------------------
# Step 4-7: Verifier setup + run + reports
# (mirrors citation_verification_demo.py's verifier wiring)
# ---------------------------------------------------------------------------

def _build_verifier(cfg, args: argparse.Namespace, verbose: bool) -> CitationVerifier:
    """Build a fully-wired CitationVerifier with cascading agents + secondary verifier."""

    _log(verbose, "[5/7] Initializing bibliographic connectors")
    cache_path = Path(args.cache_path) if args.cache_path else cfg.connectors.cache_path
    dblp_mirror_path = (
        Path(args.dblp_mirror_path) if args.dblp_mirror_path else cfg.connectors.dblp_mirror_path
    )
    cli_dblp_sqlite = (args.dblp_sqlite_path or "").strip()
    cfg_dblp_sqlite = str(cfg.connectors.dblp_sqlite_path or "").strip()
    final_dblp_sqlite = cli_dblp_sqlite or cfg_dblp_sqlite
    final_semscholar_key = (
        (args.semantic_scholar_api_key or "").strip()
        or (cfg.connectors.semantic_scholar_api_key or "").strip()
        or None
    )

    orchestrator = default_orchestrator(
        cache_path=cache_path,
        dblp_mirror_path=dblp_mirror_path,
        dblp_sqlite_path=Path(final_dblp_sqlite) if final_dblp_sqlite else None,
        enabled_sources=cfg.connectors.enabled_sources,
        acl_anthology_data_dir=cfg.connectors.acl_anthology_data_dir,
        acl_anthology_repo_path=cfg.connectors.acl_anthology_repo_path,
        semantic_scholar_api_key=final_semscholar_key,
        govinfo_api_key=cfg.connectors.govinfo_api_key,
        searxng_base_url=cfg.connectors.searxng_base_url,
        ncbi_api_key=cfg.connectors.ncbi_api_key,
        ncbi_email=cfg.connectors.ncbi_email,
        web_search_provider=cfg.connectors.web_search_provider,
        google_api_key=cfg.connectors.google_api_key,
        google_cse_id=cfg.connectors.google_cse_id,
        serpapi_key=cfg.connectors.serpapi_key,
        tavily_api_key=cfg.connectors.tavily_api_key,
        max_workers=getattr(args, "connector_workers", 8),
    )

    # Optional LLM-based source router
    llm_resolver = None
    if cfg.verification_llm.enabled and cfg.verification_llm.provider == "bedrock":
        from citation_verification_demo import (
            BedrockStrictMatchResolver,
            BedrockSourceSuitabilityRouter,
            RoutedConnectorOrchestrator,
        )
        llm_resolver = BedrockStrictMatchResolver(
            model_id=str(cfg.verification_llm.bedrock.model_id),
            region=cfg.verification_llm.bedrock.region,
            bearer_token=cfg.verification_llm.bedrock.bearer_token,
            max_candidates=cfg.verification_llm.max_candidates,
        )
        if args.enable_llm_source_router:
            source_router = BedrockSourceSuitabilityRouter(
                model_id=str(cfg.verification_llm.bedrock.model_id),
                region=cfg.verification_llm.bedrock.region,
                bearer_token=cfg.verification_llm.bedrock.bearer_token,
                max_sources_cap=args.source_router_max_sources,
            )
            orchestrator = RoutedConnectorOrchestrator(orchestrator, source_router)
            _log(verbose, "      llm_source_router enabled")

    # Secondary verifier (uses arxiv / semantic_scholar / openalex / url_direct connectors)
    from packages.connectors.arxiv import ArxivConnector
    from packages.connectors.openalex import OpenAlexConnector
    from packages.connectors.semanticscholar import SemanticScholarConnector
    from packages.connectors.url_direct import URLDirectConnector
    from packages.core.secondary_verification import SecondaryVerifier

    arxiv_conn = semscholar_conn = openalex_conn = url_direct_conn = None
    for conn in orchestrator.connectors:
        if isinstance(conn, ArxivConnector):
            arxiv_conn = conn
        elif isinstance(conn, SemanticScholarConnector):
            semscholar_conn = conn
        elif isinstance(conn, OpenAlexConnector):
            openalex_conn = conn
        elif isinstance(conn, URLDirectConnector):
            url_direct_conn = conn

    secondary_verifier = SecondaryVerifier(
        arxiv_connector=arxiv_conn,
        semantic_scholar_connector=semscholar_conn,
        openalex_connector=openalex_conn,
        url_direct_connector=url_direct_conn,
    )

    # Cascading 3-agent + extractor agent + author classifier (Bedrock)
    valid_agent = potential_agent = hallucinated_agent = extractor_agent = None
    author_classifier = None
    if cfg.verification_llm.enabled and cfg.verification_llm.provider == "bedrock":
        from citation_verification_demo import _build_bedrock_client
        from packages.core.bedrock_agents import (
            BedrockFieldClassifier,
            BedrockHallucinatedAgent,
            BedrockPotentialAgent,
            BedrockValidAgent,
        )
        from packages.core.extractor_agent import BedrockExtractorAgent

        bedrock_client = _build_bedrock_client(
            region=cfg.verification_llm.bedrock.region,
            bearer_token=cfg.verification_llm.bedrock.bearer_token,
        )
        model_id = str(cfg.verification_llm.bedrock.model_id)
        valid_agent = BedrockValidAgent(bedrock_client, model_id)
        potential_agent = BedrockPotentialAgent(bedrock_client, model_id)
        hallucinated_agent = BedrockHallucinatedAgent(bedrock_client, model_id)
        extractor_agent = BedrockExtractorAgent(bedrock_client, model_id)
        author_classifier = BedrockFieldClassifier(bedrock_client, model_id)
        _log(verbose, f"      cascading 3-agent + field classifier (authors/venue/publisher) + extractor enabled (model={model_id})")
    else:
        _log(verbose, "      cascading agents disabled (no verification LLM)")

    # Verified reference cache — reuse connector_cache.sqlite for storage
    from packages.connectors.base import VerifiedReferenceCache
    verified_ref_cache = VerifiedReferenceCache(cache_path)
    _log(verbose, f"      verified reference cache: {cache_path}")

    return CitationVerifier(
        orchestrator=orchestrator,
        llm_resolver=llm_resolver,
        secondary_verifier=secondary_verifier,
        valid_agent=valid_agent,
        potential_agent=potential_agent,
        hallucinated_agent=hallucinated_agent,
        extractor_agent=extractor_agent,
        author_classifier=author_classifier,
        verified_ref_cache=verified_ref_cache,
    )


def _save_citations_artifact(
    pdf: Path,
    citations: list[CitationRecord],
    extraction_quality: ExtractionQuality,
    extraction_metadata: dict[str, Any],
    out_dir: Path,
) -> Path:
    """Persist the parsed citations (post-step-3) to a JSON file for inspection.

    This is the artifact that pdf_extractor_demo.py would produce — useful for
    debugging the extraction step independently of verification.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / f"{pdf.stem}.parsed_citations.json"
    payload = {
        "input_pdf": str(pdf),
        "reference_entries_count": len(citations),
        "extraction_quality": extraction_quality.value,
        "metadata": extraction_metadata,
        "citations": [asdict(c) for c in citations],
    }
    artifact_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return artifact_path


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _acquire_out_lock(out_path: Path) -> Path:
    """Reserve out_path for this process; refuse if another live writer holds it.

    Two concurrent runs writing the same --out raced and one overwrote the
    other's complete report (Citeaudit run, 2026-04-28). The lockfile
    <out>.lock holds the writer's PID; if another live PID is in there we
    refuse so the user picks a different --out instead of silently losing work.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lock = out_path.with_suffix(out_path.suffix + ".lock")
    if lock.exists():
        try:
            existing_pid = int(lock.read_text().strip())
        except (ValueError, OSError):
            existing_pid = -1
        if existing_pid > 0 and _pid_alive(existing_pid):
            raise RuntimeError(
                f"Another pdf_checker process (pid={existing_pid}) is already "
                f"writing to {out_path}. Pick a different --out path, or stop "
                f"that process and remove {lock} if it is stale."
            )
    lock.write_text(str(os.getpid()))
    return lock


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pdf_check(
    input_pdf: str | Path,
    out_path: str | Path,
    args: argparse.Namespace | None = None,
) -> dict:
    """Run the full PDF → verification pipeline."""
    input_pdf = Path(input_pdf)
    out_path = Path(out_path)
    out_lock = _acquire_out_lock(out_path)
    try:
        return _run_pdf_check_inner(input_pdf, out_path, args)
    finally:
        try:
            out_lock.unlink(missing_ok=True)
        except OSError:
            pass


def _run_pdf_check_inner(
    input_pdf: Path,
    out_path: Path,
    args: argparse.Namespace | None,
) -> dict:
    cfg = load_pdf_checker_config()
    verbose = _env_flag("CITATION_CHECKER_VERBOSE", "1")

    if args is None:
        args = build_arg_parser().parse_args(
            ["--input", str(input_pdf), "--out", str(out_path)]
        )

    # Streaming mode: heuristic-parse returns immediately; LLM reparse runs in
    # the background so we can pipeline reparse-completion → verify-submission.
    citations, reparse_pool, reparse_futures, extraction_quality, extraction_metadata, pages = (
        _extract_and_parse(input_pdf, cfg, args, verbose, streaming=True)
    )

    _log(verbose, "[4/7] Linking inline citation markers")
    combined_text = "\n".join(pages)
    markers = extract_inline_markers(combined_text)
    link_quality = estimate_link_quality(markers, len(citations))
    _log(verbose, f"      marker_link_quality={round(link_quality, 4)} ({len(markers)} markers)")

    # Optionally save the parsed citations as an inspectable artifact
    if args.save_parsed_citations:
        artifact_dir = Path(args.save_parsed_citations)
        artifact_path = _save_citations_artifact(
            input_pdf, citations, extraction_quality, extraction_metadata, artifact_dir,
        )
        _log(verbose, f"      Saved parsed citations to {artifact_path}")

    if args.extract_only:
        # Wait for any in-flight reparse to mutate records before dumping.
        if reparse_futures:
            for fut in as_completed(list(reparse_futures.values())):
                try:
                    fut.result()
                except Exception as exc:
                    _log(verbose, f"      [reparse failed] {exc}")
        if reparse_pool is not None:
            reparse_pool.shutdown(wait=True)
        _log(verbose, "[skip] --extract-only set; skipping verification")
        return {
            "input_pdf": str(input_pdf),
            "reference_entries_count": len(citations),
            "extraction_quality": extraction_quality.value,
            "metadata": extraction_metadata,
            "citations": [asdict(c) for c in citations],
        }

    verifier = _build_verifier(cfg, args, verbose)

    # Build per-citation extraction quality map
    quality_map: dict[str, ExtractionQuality] = {
        c.citation_id: extraction_quality for c in citations
    }
    if link_quality < 0.5:
        for c in citations:
            quality_map[c.citation_id] = ExtractionQuality.LOW

    _log(verbose, f"[6/7] Verifying citations (n={len(citations)})")
    paper_title = input_pdf.stem.replace("_", " ")

    # Parallel citation verification, results stored in original order via index map
    citation_workers = max(1, getattr(args, "citation_workers", 4))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    verdicts_by_index: dict[int, Any] = {}
    write_lock = threading.Lock()

    def _verify_one(idx_citation: tuple[int, Any]) -> tuple[int, Any]:
        idx, citation = idx_citation
        verdict = verifier.verify_citation(
            citation,
            extraction_quality=quality_map.get(citation.citation_id, ExtractionQuality.UNKNOWN),
            source_paper_title=paper_title,
        )
        with write_lock:
            # Log start + verdict block atomically so concurrent threads do not
            # interleave their output (prior layout printed start lines outside
            # the lock, which made each verdict appear under a different
            # citation's start line).
            _log(
                verbose,
                f"      [{idx}/{len(citations)}] {citation.citation_id} "
                f"title={citation.title[:60]!r}",
            )
            verdicts_by_index[idx - 1] = verdict
            ordered = [verdicts_by_index[i] for i in sorted(verdicts_by_index)]
            report_dict = _build_report_dict(
                input_pdf, ordered, pages, markers, link_quality,
                extraction_quality, extraction_metadata,
            )
            _write_json_atomic(out_path, report_dict)

            verdict_label = verdict.verdict.value
            taxonomy = verdict.taxonomy_subtype or []
            reason = (verdict.adjudication_reason or "").strip()
            conflicts = verdict.conflicts or []
            _log(verbose, f"      → verdict={verdict_label} taxonomy={taxonomy}")
            if reason:
                wrapped = _wrap_for_terminal(reason, prefix="        reason: ", width=200)
                _log(verbose, wrapped)
            if conflicts:
                _log(verbose, f"        conflicts: {conflicts}")
            _log(verbose, "")  # blank line between citations
        return idx, verdict

    if citation_workers == 1 or len(citations) <= 1:
        # Serial path: wait for any reparse, then verify in order.
        if reparse_futures:
            for fut in as_completed(list(reparse_futures.values())):
                try:
                    fut.result()
                except Exception as exc:
                    _log(verbose, f"      [reparse failed] {exc}")
        for pair in enumerate(citations, start=1):
            _verify_one(pair)
    else:
        # Pipelined: kick off verify for any citation that doesn't need
        # reparse immediately; submit the rest as their reparse futures fire.
        verify_pool = ThreadPoolExecutor(max_workers=citation_workers, thread_name_prefix="verify")
        verify_futures: list = []
        submitted_idxs: set[int] = set()
        try:
            for idx, citation in enumerate(citations, start=1):
                if citation.citation_id not in reparse_futures:
                    verify_futures.append(verify_pool.submit(_verify_one, (idx, citation)))
                    submitted_idxs.add(idx)

            if reparse_futures:
                fut_to_idx = {
                    reparse_futures[c.citation_id]: i
                    for i, c in enumerate(citations, start=1)
                    if c.citation_id in reparse_futures
                }
                for rep_fut in as_completed(list(reparse_futures.values())):
                    try:
                        rep_fut.result()
                    except Exception as exc:
                        _log(verbose, f"      [reparse failed] {exc}")
                    idx = fut_to_idx[rep_fut]
                    verify_futures.append(
                        verify_pool.submit(_verify_one, (idx, citations[idx - 1]))
                    )
                    submitted_idxs.add(idx)

            # Defensive guard: ensure every citation got exactly one verify
            # submission, even if a citation_id collision or any other race
            # in the post-loop dispatch would have dropped one.
            missing_idxs = set(range(1, len(citations) + 1)) - submitted_idxs
            if missing_idxs:
                _log(
                    verbose,
                    f"      [pipelining guard] never-submitted indices: "
                    f"{sorted(missing_idxs)}; submitting now",
                )
                for idx in sorted(missing_idxs):
                    verify_futures.append(
                        verify_pool.submit(_verify_one, (idx, citations[idx - 1]))
                    )
                    submitted_idxs.add(idx)

            for fut in as_completed(verify_futures):
                try:
                    fut.result()
                except Exception as exc:
                    _log(verbose, f"      [verify failed] {exc}")
        finally:
            verify_pool.shutdown(wait=True)
            if reparse_pool is not None:
                reparse_pool.shutdown(wait=True)

    partial_verdicts = [verdicts_by_index[i] for i in sorted(verdicts_by_index)]

    final_report = _build_report_dict(
        input_pdf, partial_verdicts, pages, markers, link_quality,
        extraction_quality, extraction_metadata,
    )

    _log(verbose, f"[7/7] Writing markdown report to {out_path.with_suffix('.md')}")
    out_path.with_suffix(".md").write_text(
        render_markdown_dict(final_report), encoding="utf-8",
    )
    _log(verbose, "      Done")
    return final_report


def _build_report_dict(
    input_pdf: Path,
    verdicts: list[Any],
    pages: list[str],
    markers: list[Any],
    link_quality: float,
    extraction_quality: ExtractionQuality,
    extraction_metadata: dict[str, Any],
) -> dict:
    return {
        "report_version": "1.0",
        "paper_id": input_pdf.stem,
        "pipeline_type": "PDF",
        "citations": [citation_verdict_to_dict(v) for v in verdicts],
        "summary": build_summary(verdicts),
        "risk_score": compute_risk_score(verdicts),
        "requires_human_review": any(v.needs_human_review for v in verdicts),
        "metadata": {
            "input_pdf": str(input_pdf),
            "pages": len(pages),
            "reference_entries": len(verdicts),
            "inline_marker_count": len(markers),
            "marker_link_quality": round(link_quality, 4),
            "extraction_quality": extraction_quality.value,
            **extraction_metadata,
        },
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def render_markdown_dict(report: dict) -> str:
    """Lightweight markdown rendering when we don't have a CheckReport object."""
    lines = [
        f"# Citation Hallucination Report — {report.get('paper_id', '')}",
        "",
        f"- Pipeline: {report.get('pipeline_type', '')}",
        f"- Risk score: {report.get('risk_score', 0)}",
        f"- Needs human review: {report.get('requires_human_review', False)}",
        "",
        "## Summary",
    ]
    summary = report.get("summary", {}) or {}
    for k, v in summary.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Citations")
    for c in report.get("citations", []):
        lines.append(
            f"- {c.get('citation_id')} — **{c.get('verdict', '?')}** "
            f"{c.get('taxonomy_subtype', [])}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run end-to-end citation hallucination check on a PDF manuscript."
    )
    parser.add_argument("--input", required=True, help="Path to input PDF.")
    parser.add_argument("--out", required=True, help="Path to output JSON report.")

    # NOTE: extraction method is now controlled entirely by config.json's
    # `citation_parse_method` field ("citation_reparse" or "ocr_llm_extract"),
    # not by CLI flags. To switch methods, edit config.json.

    # Connector overrides (mirrors citation_verification_demo.py)
    parser.add_argument(
        "--cache-path", default="",
        help="Connector cache sqlite path. Empty = use config.",
    )
    parser.add_argument(
        "--dblp-mirror-path", default="",
        help="Offline DBLP mirror JSONL path. Empty = use config.",
    )
    parser.add_argument(
        "--dblp-sqlite-path", default="",
        help="DBLP sqlite path override. Empty = use config.",
    )
    parser.add_argument(
        "--semantic-scholar-api-key", default="",
        help="Semantic Scholar API key override. Empty = use config.",
    )
    parser.add_argument(
        "--offline-only", action="store_true",
        help="Set CITATION_CHECKER_OFFLINE_ONLY=1 (skip online connectors).",
    )

    # Verifier wiring (mirrors citation_verification_demo.py)
    parser.add_argument(
        "--enable-llm-source-router", action="store_true",
        help="Use a Bedrock LLM-based source router to rank/cap connectors.",
    )
    parser.add_argument(
        "--source-router-max-sources", type=int, default=0,
        help="Hard cap for the LLM source router. 0 = trust router's budget.",
    )

    # Pipeline shortcuts
    parser.add_argument(
        "--extract-only", action="store_true",
        help="Skip verification; only run extraction (steps 1-4) and dump citations.",
    )
    parser.add_argument(
        "--save-parsed-citations", default="",
        help="Optional directory to also save the parsed citations JSON before verification.",
    )

    # Parallelism (3-level: paper × citation × connector)
    parser.add_argument(
        "--connector-workers", type=int, default=8,
        help="Max parallel connector queries per citation (default: 8).",
    )
    parser.add_argument(
        "--citation-workers", type=int, default=4,
        help="Max parallel citation verifications per paper (default: 4).",
    )
    parser.add_argument(
        "--paper-workers", type=int, default=2,
        help="Max parallel papers when --input is a directory (default: 2).",
    )

    return parser


def _collect_pdfs(input_path: Path) -> list[Path]:
    """Return list of PDF files from a path. Single file → [path]; directory → recursive *.pdf."""
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.rglob("*.pdf"))
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.offline_only:
        os.environ["CITATION_CHECKER_OFFLINE_ONLY"] = "1"

    input_path = Path(args.input)
    out_arg = Path(args.out)

    pdfs = _collect_pdfs(input_path)
    if not pdfs:
        print(f"[run] No PDFs found at {input_path}", flush=True)
        return

    # Single PDF → keep original behavior (out_arg is the report path)
    if len(pdfs) == 1 and input_path.is_file():
        run_pdf_check(pdfs[0], out_arg, args=args)
        return

    # Multiple PDFs → out_arg is treated as a directory; one report per paper
    out_arg.mkdir(parents=True, exist_ok=True)
    paper_workers = max(1, getattr(args, "paper_workers", 2))
    print(f"[run] Found {len(pdfs)} PDFs; running with paper_workers={paper_workers}", flush=True)

    import time as _time

    def _run_one(pdf: Path) -> dict[str, Any]:
        report_path = out_arg / f"{pdf.stem}_report.json"
        t0 = _time.perf_counter()
        try:
            report = run_pdf_check(pdf, report_path, args=args)
            return {
                "pdf": pdf,
                "report_path": report_path,
                "report": report,
                "runtime_s": _time.perf_counter() - t0,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "pdf": pdf,
                "report_path": report_path,
                "report": None,
                "runtime_s": _time.perf_counter() - t0,
                "error": str(exc),
            }

    results: list[dict[str, Any]] = []
    if paper_workers == 1:
        for pdf in pdfs:
            r = _run_one(pdf)
            results.append(r)
            tag = "FAILED" if r["error"] else "DONE  "
            print(f"[run] {tag} {pdf} ({r['runtime_s']:.1f}s)" + (f": {r['error']}" if r["error"] else ""), flush=True)
    else:
        with ThreadPoolExecutor(max_workers=paper_workers) as pool:
            for r in pool.map(_run_one, pdfs):
                results.append(r)
                tag = "FAILED" if r["error"] else "DONE  "
                pdf = r["pdf"]
                print(f"[run] {tag} {pdf} ({r['runtime_s']:.1f}s)" + (f": {r['error']}" if r["error"] else ""), flush=True)

    _write_batch_summary(results, out_arg)


def _write_batch_summary(results: list[dict[str, Any]], out_dir: Path) -> None:
    """Emit cross-paper aggregate stats: summary.json + summary.csv + summary.md."""
    import csv as _csv
    from collections import Counter as _Counter

    per_paper_rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    overall_verdict = _Counter()
    overall_taxonomy = _Counter()
    overall_runtime = 0.0
    total_citations = 0
    total_papers_ok = 0
    total_human_review = 0

    for r in results:
        pdf = r["pdf"]
        runtime = r["runtime_s"]
        overall_runtime += runtime

        if r["error"] is not None or r["report"] is None:
            failed.append({"pdf": str(pdf), "error": r["error"], "runtime_s": round(runtime, 2)})
            continue

        report = r["report"]
        cits = report.get("citations", []) or []
        summary = report.get("summary", {}) or {}
        risk = report.get("risk_score", 0.0)
        needs_review = report.get("requires_human_review", False)

        verdict_counts = _Counter(c.get("verdict", "") for c in cits)
        taxonomy_counts = _Counter(t for c in cits for t in (c.get("taxonomy_subtype") or []))
        overall_verdict.update(verdict_counts)
        overall_taxonomy.update(taxonomy_counts)

        total_citations += len(cits)
        total_papers_ok += 1
        if needs_review:
            total_human_review += 1

        per_paper_rows.append({
            "paper_id": report.get("paper_id", pdf.stem),
            "pdf": str(pdf),
            "report_path": str(r["report_path"]),
            "runtime_s": round(runtime, 2),
            "n_citations": len(cits),
            "n_valid": verdict_counts.get("VALID", 0),
            "n_potential": verdict_counts.get("POTENTIAL_REFERENCE", 0),
            "n_fake": verdict_counts.get("FAKE_REFERENCE", 0),
            "risk_score": round(float(risk), 3),
            "needs_human_review": bool(needs_review),
            "taxonomy_histogram": dict(taxonomy_counts),
        })

    per_paper_rows.sort(key=lambda r: r["risk_score"], reverse=True)

    summary_payload = {
        "n_papers_processed": len(results),
        "n_papers_ok": total_papers_ok,
        "n_papers_failed": len(failed),
        "total_citations": total_citations,
        "total_runtime_s": round(overall_runtime, 1),
        "papers_needing_human_review": total_human_review,
        "verdict_totals": dict(overall_verdict),
        "taxonomy_histogram": dict(overall_taxonomy.most_common()),
        "papers": per_paper_rows,
        "failed_papers": failed,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    csv_path = out_dir / "summary.csv"
    csv_fields = [
        "paper_id", "n_citations", "n_valid", "n_potential", "n_fake",
        "risk_score", "needs_human_review", "runtime_s", "report_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        w.writeheader()
        for row in per_paper_rows:
            w.writerow(row)

    md_lines = [
        "# Batch Citation Check — Summary",
        "",
        f"- Papers processed: **{summary_payload['n_papers_processed']}** "
        f"(ok: {total_papers_ok}, failed: {len(failed)})",
        f"- Total citations: **{total_citations}**",
        f"- Wall-clock: **{summary_payload['total_runtime_s']:.1f}s**",
        f"- Papers flagged for human review: **{total_human_review}**",
        "",
        "## Verdict totals (across all papers)",
        "",
    ]
    for k in ("VALID", "POTENTIAL_REFERENCE", "FAKE_REFERENCE"):
        n = overall_verdict.get(k, 0)
        pct = (100.0 * n / total_citations) if total_citations else 0.0
        md_lines.append(f"- **{k}**: {n} ({pct:.1f}%)")
    md_lines.append("")
    md_lines.append("## Taxonomy histogram")
    md_lines.append("")
    md_lines.append("| code | count |")
    md_lines.append("|------|------:|")
    for code, n in overall_taxonomy.most_common():
        md_lines.append(f"| {code} | {n} |")
    md_lines.extend(["", "## Per-paper (sorted by risk_score desc)", "",
                     "| paper_id | N | VALID | POT | FAKE | risk | review? | runtime (s) |",
                     "|---|---:|---:|---:|---:|---:|:---:|---:|"])
    for r in per_paper_rows:
        md_lines.append(
            f"| {r['paper_id']} | {r['n_citations']} | "
            f"{r['n_valid']} | {r['n_potential']} | {r['n_fake']} | "
            f"{r['risk_score']:.3f} | {'✓' if r['needs_human_review'] else ''} | "
            f"{r['runtime_s']:.1f} |"
        )
    if failed:
        md_lines.extend(["", "## Failed papers", "", "| pdf | error | runtime (s) |", "|---|---|---:|"])
        for f in failed:
            md_lines.append(f"| {f['pdf']} | `{(f['error'] or '')[:80]}` | {f['runtime_s']:.1f} |")
    (out_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(
        f"[run] Batch summary: {total_papers_ok}/{len(results)} papers OK, "
        f"{total_citations} citations, "
        f"{overall_verdict.get('VALID',0)}/{overall_verdict.get('POTENTIAL_REFERENCE',0)}/{overall_verdict.get('FAKE_REFERENCE',0)} V/P/F. "
        f"Wrote summary.json/csv/md to {out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
