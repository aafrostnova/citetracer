"""End-to-end PDF citation hallucination check pipeline.

Combines the OCR + VLM force-reparse Parser Agent (ocr_vlm_extract) with the
verifier wiring from citation_verification_demo.py (cascading multi-agent
collector + secondary verifier + optional source router).
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
from packages.core.models import (
    CitationRecord, CitationVerdict, ExtractionQuality, VerdictLabel,
    citation_verdict_from_dict, citation_verdict_to_dict,
)
from packages.core.report import build_summary, compute_risk_score, render_markdown

from .config import load_pdf_checker_config


# ---------------------------------------------------------------------------
# Live worker-pool status (toggled by --live-status). Renders a per-worker
# snapshot to stderr every N seconds so concurrency is visible in real time.
# ---------------------------------------------------------------------------
_LIVE_STATE: dict[int, dict] = {}        # thread_ident → {slot, processed, current, started}
_LIVE_LOCK = threading.Lock()
_LIVE_NEXT_SLOT = [0]                    # mutable container so workers can bump it
_LIVE_STOP = threading.Event()


def _live_start(citation_id: str) -> None:
    ident = threading.get_ident()
    with _LIVE_LOCK:
        st = _LIVE_STATE.get(ident)
        if st is None:
            slot = _LIVE_NEXT_SLOT[0]
            _LIVE_NEXT_SLOT[0] += 1
            st = {"slot": slot, "processed": 0, "current": None, "started": None}
            _LIVE_STATE[ident] = st
        st["current"] = citation_id
        st["started"] = time.monotonic()


def _live_done() -> None:
    ident = threading.get_ident()
    with _LIVE_LOCK:
        st = _LIVE_STATE.get(ident)
        if st is not None:
            st["processed"] += 1
            st["current"] = None
            st["started"] = None


def _live_monitor_loop(interval: float) -> None:
    import sys as _sys
    while not _LIVE_STOP.wait(interval):
        with _LIVE_LOCK:
            snapshot = sorted(_LIVE_STATE.values(), key=lambda s: s["slot"])
        if not snapshot:
            continue
        now = time.monotonic()
        busy = sum(1 for s in snapshot if s["current"])
        total_done = sum(s["processed"] for s in snapshot)
        lines = [
            f"=== live pool: {busy}/{len(snapshot)} busy, "
            f"{total_done} total done ==="
        ]
        for st in snapshot:
            cur = st["current"]
            if cur:
                elapsed = now - (st["started"] or now)
                cur_str = (cur[:34] + "…") if len(cur) > 35 else cur
                lines.append(
                    f"  [slot {st['slot']:02d}] busy {elapsed:5.1f}s  "
                    f"cur={cur_str:36s} done={st['processed']}"
                )
            else:
                lines.append(
                    f"  [slot {st['slot']:02d}] idle              "
                    f"{'':40s}done={st['processed']}"
                )
        print("\n".join(lines), file=_sys.stderr, flush=True)
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

    The pipeline always runs the OCR+VLM force-reparse path. Provider is
    one of ``bedrock`` / ``openai`` / ``azure_openai`` (cloud APIs only;
    the legacy local-HF reparse path was removed for GPU-memory reasons).
    """
    if not cfg.ocr_vlm_extract.enabled:
        raise RuntimeError(
            "ocr_vlm_extract.enabled=true and a remote model_id are required."
        )

    provider = (cfg.ocr_vlm_extract.provider or "bedrock").strip().lower()
    paper_style = str(
        getattr(cfg.ocr_vlm_extract, "paper_style", "") or ""
    ).strip().lower()

    # provider in {"bedrock", "openai", "azure_openai"} — they all read the
    # remote model_id from cfg.ocr_vlm_extract.bedrock.model_id (interpreted
    # as deployment name for azure_openai, model name for openai). Auth for
    # openai/azure_openai is taken from environment variables; bedrock auth
    # uses cfg.ocr_vlm_extract.bedrock.bearer_token.
    model_id = str(cfg.ocr_vlm_extract.bedrock.model_id or "").strip()
    region = str(cfg.ocr_vlm_extract.bedrock.region or "").strip() or "us-east-1"
    bearer_token = str(cfg.ocr_vlm_extract.bedrock.bearer_token or "").strip() or None
    if not model_id:
        raise RuntimeError(
            f"ocr_vlm_extract.provider={provider} requires "
            "ocr_vlm_extract.bedrock.model_id in config.json "
            "(used as deployment name for azure_openai, model name for openai)."
        )

    llm_cfg = {
        "enabled": True,
        "provider": provider,
        "model_path": "",
        "max_new_tokens": int(cfg.ocr_vlm_extract.max_new_tokens),
        "temperature": float(cfg.ocr_vlm_extract.temperature),
        "force_all_entries": bool(cfg.ocr_vlm_extract.force_all_entries),
        "parallel_workers": 8,
        "paper_style": paper_style,
        "bedrock": {
            "model_id": model_id,
            "region": region,
            "bearer_token": bearer_token,
        },
    }
    meta = {
        "pipeline_method": "ocr_vlm_extract",
        "llm_reparse_provider": provider,
        "llm_reparse_model_id": model_id,
        "llm_reparse_region": region,
        "llm_reparse_force_all_entries": bool(llm_cfg["force_all_entries"]),
        "llm_reparse_max_new_tokens": int(llm_cfg["max_new_tokens"]),
        "llm_reparse_temperature": float(llm_cfg["temperature"]),
        "llm_reparse_parallel_workers": int(llm_cfg["parallel_workers"]),
        "llm_reparse_paper_style": paper_style or "(generic)",
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
        boundary_merge_mode=cfg.ocr_vlm_extract.boundary_merge_mode,
        boundary_merge_model_id=cfg.ocr_vlm_extract.bedrock.model_id,
        boundary_merge_region=cfg.ocr_vlm_extract.bedrock.region,
        boundary_merge_bearer_token=cfg.ocr_vlm_extract.bedrock.bearer_token,
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

    # Path A — bbox-aware per-entry crops. The segmenter exposes a
    # parallel list of (page_marker, bbox) tuples per entry in
    # extraction_metadata["entry_layouts"]; bboxes come from DeepSeek-OCR-2
    # which normalizes coordinates to a 1000x1000 space. Cropping to the
    # entry's exact region (instead of the whole page) makes the
    # citation occupy ~80% of the image and gives the VLM enough character
    # detail to fix OCR typos like "Akocora" → "Akcora".
    entry_layouts = extraction_metadata.get("entry_layouts") or []

    def _crop_entry_image(page_png: bytes, bbox: tuple[int, int, int, int],
                          norm: int = 1000, padding: int = 15) -> bytes | None:
        try:
            from PIL import Image as _Image
            import io as _io
            img = _Image.open(_io.BytesIO(page_png))
            W, H = img.size
            x1, y1, x2, y2 = bbox
            sx, sy = W / norm, H / norm
            cx1 = max(0, int(x1 * sx) - padding)
            cy1 = max(0, int(y1 * sy) - padding)
            cx2 = min(W, int(x2 * sx) + padding)
            cy2 = min(H, int(y2 * sy) + padding)
            if cx2 <= cx1 or cy2 <= cy1:
                return None
            crop = img.crop((cx1, cy1, cx2, cy2))
            buf = _io.BytesIO()
            crop.save(buf, format="PNG")
            return buf.getvalue()
        except Exception as exc:
            _log(verbose, f"      [crop-bbox failed] {exc}")
            return None

    # Path A — bbox-aware per-entry crops. Aligned by title prefix instead
    # of strict positional zip, so a count mismatch between entry_layouts and
    # reference_entries (e.g. segmenter merged 1 less entry than listed) only
    # degrades the un-matched citation, not the entire paper.
    import re as _re_align
    def _align_key(text: str) -> str:
        """First ~30 chars after stripping enumeration / whitespace."""
        cleaned = _re_align.sub(r"^\s*(?:\[\d+\]|\(\d+\)|\d+\.)\s*", "", text or "")
        cleaned = _re_align.sub(r"\s+", " ", cleaned).strip().lower()
        return cleaned[:30]

    # Pair each layout entry with its key. layouts comes positionally aligned
    # to the segmenter's text entries; we pair via the metadata's
    # `entry_layouts_keys` if provided, otherwise fall back to indexing.
    layout_keys = extraction_metadata.get("entry_layouts_keys") or []
    layouts_by_key: dict[str, list] = {}
    if entry_layouts and layout_keys and len(layout_keys) == len(entry_layouts):
        for k, lay in zip(layout_keys, entry_layouts):
            if k and k not in layouts_by_key:
                layouts_by_key[k] = lay
    elif entry_layouts:
        # No key list provided — assume entry_layouts is positional with
        # reference_entries (the common case). Use the entry text itself as
        # the key so prefix lookup still works when counts match.
        for entry, lay in zip(reference_entries, entry_layouts):
            k = _align_key(entry)
            if k and k not in layouts_by_key:
                layouts_by_key[k] = lay

    # Pre-build path-B fallback structures (used per-entry when bbox lookup misses)
    ref_page_texts: list[tuple[int, str]] = [
        (pi, pages[pi]) for pi in range(max(0, ref_start), min(ref_end + 1, len(pages)))
    ]
    page_token_cache: list[tuple[int, str, set[str]]] = [
        (pi, ptext, _entry_tokens(ptext)) for pi, ptext in ref_page_texts
    ]

    def _whole_page_fallback(entry_text: str) -> list[bytes]:
        entry_pages: list[int] = []
        search_key = entry_text[:60].strip().lower()
        for pi, ptext, _ in page_token_cache:
            if search_key and search_key in ptext.lower():
                entry_pages.append(pi)
        if not entry_pages and len(entry_text) > 80:
            first_half = entry_text[:40].strip().lower()
            second_half = entry_text[-40:].strip().lower()
            for pi, ptext, _ in page_token_cache:
                pl = ptext.lower()
                if first_half in pl or second_half in pl:
                    entry_pages.append(pi)
        if not entry_pages:
            etokens = _entry_tokens(entry_text)
            if len(etokens) >= 5:
                for pi, _, ptokens in page_token_cache:
                    overlap = len(etokens & ptokens) / max(len(etokens), 1)
                    if overlap >= 0.4:
                        entry_pages.append(pi)
        entry_pages = sorted(set(entry_pages))
        return [page_images[pi] for pi in entry_pages if pi in page_images]

    n_bbox_aligned = 0
    n_fallback = 0
    can_bbox = bool(layouts_by_key) and bool(page_images)

    for entry_text in reference_entries:
        crops: list[bytes] = []
        layouts = layouts_by_key.get(_align_key(entry_text)) if can_bbox else None
        if layouts:
            for page_marker, bbox in layouts:
                # OCR marker is 1-based and counts only reference pages, so
                # marker N maps to PDF 0-indexed page (ref_start + N - 1).
                pdf_page_idx = ref_start + int(page_marker) - 1
                src = page_images.get(pdf_page_idx)
                if src is None:
                    continue
                crop = _crop_entry_image(src, tuple(bbox))
                if crop:
                    crops.append(crop)
        if crops:
            n_bbox_aligned += 1
            per_entry_images.append(crops)
        else:
            # No bbox match (or bbox crop failed) → whole-page fallback for
            # this single entry. Other entries still keep their bbox crops.
            imgs = _whole_page_fallback(entry_text)
            if imgs:
                n_fallback += 1
            per_entry_images.append(imgs)

    if can_bbox:
        _log(verbose, f"      Per-entry bbox crops: {n_bbox_aligned}/{len(reference_entries)}  "
                     f"(whole-page fallback: {n_fallback})")
    no_image_idxs = [i + 1 for i, imgs in enumerate(per_entry_images) if not imgs]
    if no_image_idxs:
        _log(verbose, f"      [WARN] {len(no_image_idxs)}/{len(reference_entries)} entries got no image: pdf-ref indices {no_image_idxs[:15]}")

    if streaming:
        citations, reparse_pool, reparse_futures = parse_reference_entries_streaming(
            reference_entries,
            llm_reparse_config=llm_cfg,
            reference_page_images=per_entry_images,
        )
        _log(verbose, f"      Parsed {len(citations)} citation records "
                     f"({len(reparse_futures)} awaiting LLM reparse)")
        extraction_metadata.update(llm_meta)

        # Save OCR debug artifacts BEFORE reparse futures finish mutating the
        # records. The deepcopy snapshots the heuristic-only state; in-flight
        # reparse may have started but is unlikely to have completed within
        # the few ms between streaming-return and this snapshot.
        if getattr(args, "save_ocr_artifacts", False):
            import copy as _copy
            heuristic_snapshot = [_copy.deepcopy(c) for c in citations]
            out_path = Path(args.out)
            art_dir = _dump_ocr_artifacts_pre_reparse(
                pdf, out_path, heuristic_snapshot, per_entry_images,
                extraction_metadata,
            )
            _log(verbose, f"      [ocr-artifacts] pre-reparse dump → {art_dir}")
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
        openalex_mailto=cfg.connectors.openalex_mailto,
        openalex_api_key=cfg.connectors.openalex_api_key,
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

    # Cascading 3-agent + extractor agent + author classifier. Provider can be
    # bedrock / openai / azure_openai / local — they all expose the same
    # `.converse(...)` API via packages.llm.client.build_chat_client, so the
    # agent classes accept any of them.
    valid_agent = potential_agent = hallucinated_agent = extractor_agent = None
    author_classifier = None
    if cfg.verification_llm.enabled:
        from packages.llm.client import build_chat_client
        from packages.core.bedrock_agents import (
            BedrockFieldClassifier,
            BedrockHallucinatedAgent,
            BedrockPotentialAgent,
            BedrockValidAgent,
        )
        from packages.core.extractor_agent import BedrockExtractorAgent

        provider = cfg.verification_llm.provider or "bedrock"
        chat_client = build_chat_client(
            provider=provider,
            region=cfg.verification_llm.bedrock.region,
            bearer_token=cfg.verification_llm.bedrock.bearer_token,
        )
        model_id = str(cfg.verification_llm.bedrock.model_id)
        valid_agent = BedrockValidAgent(chat_client, model_id)
        potential_agent = BedrockPotentialAgent(chat_client, model_id)
        hallucinated_agent = BedrockHallucinatedAgent(chat_client, model_id)
        extractor_agent = BedrockExtractorAgent(chat_client, model_id)
        author_classifier = BedrockFieldClassifier(chat_client, model_id)
        _log(verbose, f"      cascading 3-agent + field classifier + extractor enabled (provider={provider}, model={model_id})")
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


def _ocr_artifact_dir(out_path: Path, pdf: Path) -> Path:
    """Derive the per-paper OCR-artifact dir from the report out path."""
    base = out_path.parent if out_path.suffix else out_path
    return base / "ocr_artifacts" / pdf.stem


def _dump_ocr_artifacts_pre_reparse(
    pdf: Path,
    out_path: Path,
    citations_heuristic: list[CitationRecord],
    per_entry_images: list[list[bytes]],
    extraction_metadata: dict[str, Any],
) -> Path:
    """Dump OCR intermediate, per-citation crops, and the heuristic-only
    citation snapshot. Called after parse but before LLM reparse mutates the
    citation records.
    """
    art_dir = _ocr_artifact_dir(out_path, pdf)
    art_dir.mkdir(parents=True, exist_ok=True)

    # 1. OCR intermediate dir (symlink to the .tmp run dir if available)
    ocr_dir = extraction_metadata.get("ocr_intermediate_dir")
    if ocr_dir:
        link = art_dir / "ocr_intermediate"
        try:
            if link.is_symlink() or link.exists():
                link.unlink()
        except OSError:
            pass
        try:
            link.symlink_to(Path(ocr_dir).resolve())
        except OSError:
            # Fall back to recording the path in a text file if symlink fails
            (art_dir / "ocr_intermediate_path.txt").write_text(str(ocr_dir))

    # 2. Per-entry bbox crops
    crops_dir = art_dir / "crops"
    crops_dir.mkdir(exist_ok=True)
    for i, imgs in enumerate(per_entry_images, start=1):
        for j, png_bytes in enumerate(imgs):
            (crops_dir / f"entry_{i:03d}_{j}.png").write_bytes(png_bytes)

    # 3. Heuristic-only citation snapshot (deepcopy was already taken by caller)
    heuristic_path = art_dir / "parsed_heuristic.json"
    heuristic_path.write_text(
        json.dumps(
            {"input_pdf": str(pdf), "citations": [asdict(c) for c in citations_heuristic]},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return art_dir


def _dump_ocr_artifacts_post_reparse(
    pdf: Path,
    out_path: Path,
    citations_final: list[CitationRecord],
) -> Path:
    """Dump the post-LLM-reparse citation snapshot. Call after every reparse
    future has completed.
    """
    art_dir = _ocr_artifact_dir(out_path, pdf)
    art_dir.mkdir(parents=True, exist_ok=True)
    final_path = art_dir / "parsed_after_reparse.json"
    final_path.write_text(
        json.dumps(
            {"input_pdf": str(pdf), "citations": [asdict(c) for c in citations_final]},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return final_path


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
    import time as _time
    _t_start = _time.perf_counter()
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
        if getattr(args, "save_ocr_artifacts", False):
            p = _dump_ocr_artifacts_post_reparse(input_pdf, out_path, citations)
            _log(verbose, f"      [ocr-artifacts] post-reparse dump → {p}")
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

    from packages.core.adjudicate import _RELAXED_FIELDS as _relaxed
    if _relaxed:
        _log(verbose, "      [relaxed-fields mode] only title/authors/year/venue gate VALID")
    _log(verbose, f"[6/7] Verifying citations (n={len(citations)})")
    paper_title = input_pdf.stem.replace("_", " ")

    # Parallel citation verification, results stored in original order via index map
    citation_workers = max(1, getattr(args, "citation_workers", 4))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    verdicts_by_index: dict[int, Any] = {}
    write_lock = threading.Lock()

    # Resume support: if --resume is set and this paper's report.json already
    # exists, reuse the verdicts already in it (per citation_id) so we only
    # re-verify citations that haven't been done yet. The OCR + extraction +
    # reparse stages still ran (they're deterministic and already complete);
    # we just skip the slow connector cascade + LLM agents for done citations.
    resume_done: dict[str, Any] = {}
    if getattr(args, "resume", False) and out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            for c in (existing.get("citations") or []):
                cid = c.get("citation_id", "")
                if cid:
                    resume_done[cid] = citation_verdict_from_dict(c, citation_id=cid)
            _log(verbose, f"      [resume] loaded {len(resume_done)} existing verdicts from {out_path.name}")
        except Exception as exc:
            _log(verbose, f"      [resume] failed to load existing report ({exc}); re-verifying all citations")
            resume_done = {}

    def _verify_one(idx_citation: tuple[int, Any]) -> tuple[int, Any]:
        idx, citation = idx_citation
        # Resume short-circuit: if this citation_id is already in the existing
        # report, reuse its verdict and skip the connector cascade + agents.
        if citation.citation_id in resume_done:
            verdict = resume_done[citation.citation_id]
            with write_lock:
                _log(
                    verbose,
                    f"      [{idx}/{len(citations)}] {citation.citation_id} "
                    f"(reused from previous run) verdict={verdict.verdict.value}",
                )
                verdicts_by_index[idx - 1] = verdict
                ordered = [verdicts_by_index[i] for i in sorted(verdicts_by_index)]
                report_dict = _build_report_dict(
                    input_pdf, ordered, pages, markers, link_quality,
                    extraction_quality, extraction_metadata,
                )
                _write_json_atomic(out_path, report_dict)
            return idx, verdict
        _live_start(citation.citation_id)
        try:
            verdict = verifier.verify_citation(
                citation,
                extraction_quality=quality_map.get(citation.citation_id, ExtractionQuality.UNKNOWN),
                source_paper_title=paper_title,
            )
        finally:
            _live_done()
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
        # Map future -> (idx, citation) so we can write a placeholder verdict
        # if the future raises (otherwise the citation silently disappears
        # from the report and the index sequence has gaps).
        future_meta: dict = {}
        def _submit_verify(idx_: int, citation_: Any):
            fut = verify_pool.submit(_verify_one, (idx_, citation_))
            future_meta[fut] = (idx_, citation_)
            return fut
        try:
            for idx, citation in enumerate(citations, start=1):
                if citation.citation_id not in reparse_futures:
                    verify_futures.append(_submit_verify(idx, citation))
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
                    verify_futures.append(_submit_verify(idx, citations[idx - 1]))
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
                    verify_futures.append(_submit_verify(idx, citations[idx - 1]))
                    submitted_idxs.add(idx)

            for fut in as_completed(verify_futures):
                try:
                    fut.result()
                except Exception as exc:
                    # Write a placeholder verdict so the citation appears in
                    # the final report instead of silently vanishing.
                    meta = future_meta.get(fut)
                    if meta is not None:
                        idx_, citation_ = meta
                        with write_lock:
                            verdicts_by_index[idx_ - 1] = CitationVerdict(
                                citation_id=citation_.citation_id,
                                verdict=VerdictLabel.INSUFFICIENT_EVIDENCE,
                                evidence_sources=[],
                                conflicts=["verify_failed"],
                                adjudication_reason=f"verify failed: {exc}",
                                taxonomy_subtype=[],
                                needs_human_review=True,
                                extraction_quality=quality_map.get(
                                    citation_.citation_id, ExtractionQuality.UNKNOWN,
                                ),
                            )
                    _log(verbose, f"      [verify failed] {exc}")
        finally:
            verify_pool.shutdown(wait=True)
            if reparse_pool is not None:
                reparse_pool.shutdown(wait=True)

    if getattr(args, "save_ocr_artifacts", False):
        p = _dump_ocr_artifacts_post_reparse(input_pdf, out_path, citations)
        _log(verbose, f"      [ocr-artifacts] post-reparse dump → {p}")

    partial_verdicts = [verdicts_by_index[i] for i in sorted(verdicts_by_index)]

    runtime_s = round(_time.perf_counter() - _t_start, 2)
    final_report = _build_report_dict(
        input_pdf, partial_verdicts, pages, markers, link_quality,
        extraction_quality, extraction_metadata,
        runtime_s=runtime_s,
    )

    _log(verbose, f"[7/7] Writing markdown report to {out_path.with_suffix('.md')} (took {runtime_s}s)")
    out_path.with_suffix(".md").write_text(
        render_markdown_dict(final_report), encoding="utf-8",
    )

    # Dump per-citation phase timing for bottleneck analysis
    timing_log = getattr(verifier, "_timing_log", None)
    if timing_log:
        timing_path = out_path.with_suffix(".timing.json")
        timing_path.write_text(json.dumps(timing_log, indent=2), encoding="utf-8")
        _log(verbose, f"      timing log → {timing_path} (n={len(timing_log)})")

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
    runtime_s: float | None = None,
) -> dict:
    # Strip extractor / reparse internal trace fields from metadata.
    # We keep only top-level facts that downstream readers care about; the
    # OCR run dir, entry-layout bboxes, llm_reparse_* config, etc. are
    # debugging breadcrumbs that bloat the report.
    _META_KEEP = {
        "entry_extraction_mode",
        "entry_extraction_provider",
        "reference_page_start",
        "reference_page_end",
        "reference_heading_detected",
        "entry_extraction_fallback",  # keep so failed runs are still flagged
    }
    extraction_meta_filtered = {
        k: v for k, v in extraction_metadata.items() if k in _META_KEEP
    }
    return {
        "report_version": "1.0",
        "paper_id": input_pdf.stem,
        "pipeline_type": "PDF",
        "runtime_s": runtime_s,
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
            **extraction_meta_filtered,
        },
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def render_markdown_dict(report: dict) -> str:
    """Lightweight markdown rendering when we don't have a CheckReport object."""
    runtime = report.get("runtime_s")
    runtime_line = f"- Runtime: {runtime:.2f}s" if isinstance(runtime, (int, float)) else None
    lines = [
        f"# Citation Hallucination Report — {report.get('paper_id', '')}",
        "",
        f"- Pipeline: {report.get('pipeline_type', '')}",
        f"- Risk score: {report.get('risk_score', 0)}",
        f"- Needs human review: {report.get('requires_human_review', False)}",
    ]
    if runtime_line is not None:
        lines.append(runtime_line)
    lines.extend([
        "",
        "## Summary",
    ])
    summary = report.get("summary", {}) or {}
    for k, v in summary.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Citations")
    for c in report.get("citations", []):
        verdict = c.get("verdict", "?")
        lines.append(
            f"- {c.get('citation_id')} — **{verdict}** "
            f"{c.get('taxonomy_subtype', [])}"
        )
        if verdict != "VALID":
            reason = (c.get("adjudication_reason") or "").strip()
            if reason:
                # Indent under the bullet so it renders as a sub-line.
                wrapped = _wrap_for_terminal(reason, prefix="    reason: ", width=140)
                lines.append(wrapped)
            conflicts = c.get("conflicts") or []
            if conflicts:
                lines.append(f"    conflicts: {conflicts}")
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

    # NOTE: extraction method is fixed to ocr_vlm_extract; provider
    # (bedrock vs local) is controlled by config.json's
    # ocr_vlm_extract.provider field.

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
    parser.add_argument(
        "--save-ocr-artifacts", action="store_true",
        help=(
            "Persist OCR debug artifacts under <out>/ocr_artifacts/<paper>/: "
            "ocr_intermediate (symlink to .tmp run dir with raw markdown + entries), "
            "crops/entry_NNN_M.png (per-citation bbox crops), "
            "parsed_heuristic.json (citations after rule-based parse), and "
            "parsed_after_reparse.json (citations after LLM reparse)."
        ),
    )
    parser.add_argument(
        "--live-status", action="store_true",
        help="Print a per-worker live snapshot to stderr every --live-interval seconds.",
    )
    parser.add_argument(
        "--live-interval", type=float, default=2.0,
        help="Seconds between live-status snapshots (default: 2.0).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help=(
            "Resume an interrupted batch. Papers whose <stem>_report.md exists "
            "in --out are skipped entirely. For papers whose <stem>_report.json "
            "exists but the .md does not (interrupted mid-paper), reuse the "
            "verdicts already in the .json and only run the missing citations."
        ),
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

    monitor_thread: threading.Thread | None = None
    if args.live_status:
        _LIVE_STOP.clear()
        monitor_thread = threading.Thread(
            target=_live_monitor_loop,
            args=(args.live_interval,),
            daemon=True,
            name="live-status-monitor",
        )
        monitor_thread.start()
        print(
            f"[run] live-status enabled (interval={args.live_interval}s). "
            f"Snapshots go to stderr.",
            flush=True,
        )

    # Single PDF → keep original behavior (out_arg is the report path)
    if len(pdfs) == 1 and input_path.is_file():
        run_pdf_check(pdfs[0], out_arg, args=args)
        if monitor_thread is not None:
            _LIVE_STOP.set()
            monitor_thread.join(timeout=5)
        return

    # Multiple PDFs → out_arg is treated as a directory; one report per paper
    out_arg.mkdir(parents=True, exist_ok=True)
    paper_workers = max(1, getattr(args, "paper_workers", 2))
    print(f"[run] Found {len(pdfs)} PDFs; running with paper_workers={paper_workers}", flush=True)

    import time as _time

    def _run_one(pdf: Path) -> dict[str, Any]:
        report_path = out_arg / f"{pdf.stem}_report.json"
        report_md = out_arg / f"{pdf.stem}_report.md"
        t0 = _time.perf_counter()
        # Resume: skip the whole paper if the .md (written LAST after every
        # citation succeeded) is already present.
        if getattr(args, "resume", False) and report_md.exists():
            return {
                "pdf": pdf,
                "report_path": report_path,
                "report": None,
                "runtime_s": _time.perf_counter() - t0,
                "error": None,
                "skipped": "already done (resume)",
            }
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
    def _tag_for(r: dict[str, Any]) -> str:
        if r.get("error"):
            return "FAILED"
        if r.get("skipped"):
            return "SKIP  "
        return "DONE  "

    if paper_workers == 1:
        for pdf in pdfs:
            r = _run_one(pdf)
            results.append(r)
            note = r.get("error") or r.get("skipped") or ""
            print(f"[run] {_tag_for(r)} {pdf} ({r['runtime_s']:.1f}s)" + (f": {note}" if note else ""), flush=True)
    else:
        # Concurrent paper workers; the local vLLM bundle is now loaded under
        # a per-key lock (reference_segmenter._load_local_vllm_bundle), so the
        # first thread to call it will hold the lock through the cold init and
        # all subsequent threads block until the cache is populated, then hit
        # the fast cached path. No warm-up pass is needed.
        with ThreadPoolExecutor(max_workers=paper_workers) as pool:
            for r in pool.map(_run_one, pdfs):
                results.append(r)
                pdf = r["pdf"]
                note = r.get("error") or r.get("skipped") or ""
                print(f"[run] {_tag_for(r)} {pdf} ({r['runtime_s']:.1f}s)" + (f": {note}" if note else ""), flush=True)

    _write_batch_summary(results, out_arg)

    if monitor_thread is not None:
        _LIVE_STOP.set()
        monitor_thread.join(timeout=5)


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

    paper_runtimes = sorted(r["runtime_s"] for r in per_paper_rows)
    if paper_runtimes:
        avg_per_paper = sum(paper_runtimes) / len(paper_runtimes)
        median_per_paper = paper_runtimes[len(paper_runtimes) // 2]
        max_per_paper = paper_runtimes[-1]
        min_per_paper = paper_runtimes[0]
    else:
        avg_per_paper = median_per_paper = max_per_paper = min_per_paper = 0.0
    avg_per_citation = (
        overall_runtime / total_citations if total_citations else 0.0
    )

    summary_payload = {
        "n_papers_processed": len(results),
        "n_papers_ok": total_papers_ok,
        "n_papers_failed": len(failed),
        "total_citations": total_citations,
        "total_runtime_s": round(overall_runtime, 1),
        "avg_runtime_per_paper_s": round(avg_per_paper, 2),
        "median_runtime_per_paper_s": round(median_per_paper, 2),
        "min_runtime_per_paper_s": round(min_per_paper, 2),
        "max_runtime_per_paper_s": round(max_per_paper, 2),
        "avg_runtime_per_citation_s": round(avg_per_citation, 3),
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
        f"- Per-paper runtime: avg **{summary_payload['avg_runtime_per_paper_s']:.1f}s**, "
        f"median **{summary_payload['median_runtime_per_paper_s']:.1f}s**, "
        f"min **{summary_payload['min_runtime_per_paper_s']:.1f}s**, "
        f"max **{summary_payload['max_runtime_per_paper_s']:.1f}s**",
        f"- Avg runtime per citation: **{summary_payload['avg_runtime_per_citation_s']:.3f}s**",
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
        f"Wall-clock {summary_payload['total_runtime_s']:.1f}s "
        f"(avg {summary_payload['avg_runtime_per_paper_s']:.1f}s/paper, "
        f"median {summary_payload['median_runtime_per_paper_s']:.1f}s/paper, "
        f"{summary_payload['avg_runtime_per_citation_s']:.3f}s/citation). "
        f"Wrote summary.json/csv/md to {out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
