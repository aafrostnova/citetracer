"""Parse desk-reject comments into structured citations via Bedrock LLM.

Reuses the existing pipeline from
  /project/pi_shiqingma_umass_edu/mingzheli/Citation_Hallucination_Detection
directly:

  1. Segmentation via `_extract_entries_with_bedrock` (reference_segmenter.py)
     — splits each desk-reject comment into individual raw reference strings.
  2. Per-entry parse via `_run_llm_reparse_on_raw_bedrock` (citation_parser.py)
     — returns the structured fields.
  3. Normalization via `_normalize_llm_reparse_result`.

No new fields added: each citation object matches the existing schema
{"title", "authors", "venue", "year", "volume", "pages", "publisher",
 "location", "doi", "arxiv_id", "url"}.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = "/project/pi_shiqingma_umass_edu/mingzheli/Citation_Hallucination_Detection"
sys.path.insert(0, REPO)

# Load bearer token BEFORE importing modules that build boto3 clients.
with open(os.path.join(REPO, "config.json")) as _cfg_f:
    _CFG = json.load(_cfg_f)
_BEARER = _CFG["entry_extraction"]["bedrock"]["bearer_token"]
_REGION = _CFG["ocr_llm_extract"]["bedrock"]["region"]
_MODEL_ID = _CFG["ocr_llm_extract"]["bedrock"]["model_id"]
_MAX_NEW_TOKENS = int(_CFG["ocr_llm_extract"].get("max_new_tokens", 4048))
_TEMPERATURE = float(_CFG["ocr_llm_extract"].get("temperature", 0.0))
os.environ["AWS_BEARER_TOKEN_BEDROCK"] = _BEARER

# Import after env is set — these modules call boto3 internally.
from apps.pdf_checker.ingest.reference_segmenter import (  # noqa: E402
    _extract_entries_with_bedrock,
)
from apps.pdf_checker.ingest.citation_parser import (  # noqa: E402
    _run_llm_reparse_on_raw_bedrock,
    _normalize_llm_reparse_result,
    _ensure_et_al_preserved,
)

INPUT = "/project/pi_shiqingma_umass_edu/iclr2026_hallucinated/hallucinated_refs.json"
OUTPUT = "/project/pi_shiqingma_umass_edu/iclr2026_hallucinated/hallucinated_refs_structured_bedrock.json"


def parse_one_comment(comment: str, paper_id: str) -> dict:
    """Run the two-stage Bedrock pipeline on a single desk-reject comment.

    Returns a dict with a `hallucinated_citations` list whose items conform to
    the existing `_normalize_llm_reparse_result` schema.
    """
    # Stage 1: Bedrock segmentation only (no heuristic fallback).
    try:
        raw_entries = _extract_entries_with_bedrock(
            ref_block=comment,
            model_id=_MODEL_ID,
            region=_REGION,
            chunk_chars=int(_CFG["entry_extraction"].get("chunk_chars", 12000)),
            bearer_token=_BEARER,
        )
    except Exception as exc:
        return {"segmentation_error": str(exc), "citations": []}

    citations: list[dict] = []
    for raw in raw_entries:
        # Stage 2: per-entry field extraction
        try:
            parsed = _run_llm_reparse_on_raw_bedrock(
                raw_text=raw,
                model_id=_MODEL_ID,
                region=_REGION,
                bearer_token=_BEARER,
                max_new_tokens=_MAX_NEW_TOKENS,
                temperature=_TEMPERATURE,
                reference_page_images=None,
            )
        except Exception as exc:
            citations.append({"raw_text": raw, "parse_error": str(exc)})
            continue

        if not parsed:
            citations.append({"raw_text": raw, "parse_error": "no_payload"})
            continue

        # Re-attach et al. marker if dropped by LLM.
        parsed["authors"] = _ensure_et_al_preserved(parsed.get("authors", []), raw, [])
        record = {"raw_text": raw, **parsed}
        citations.append(record)

    return {"citations": citations}


def _process_entry(entry: dict) -> dict:
    comment = entry["comment"]
    pid = entry["paper_id"]
    result = parse_one_comment(comment, pid)
    out = {
        "paper_id": pid,
        "paper_title": entry["title"],
        "url": entry["site"],
        "desk_reject_comment_raw": comment,
        "hallucinated_citations": result.get("citations", []),
        "num_citations": len(result.get("citations", [])),
    }
    if result.get("segmentation_error"):
        out["segmentation_error"] = result["segmentation_error"]
    return out


def main() -> None:
    workers = int(os.environ.get("PARSE_WORKERS", "8"))
    print(f"Model: {_MODEL_ID}, region: {_REGION}, workers: {workers}", flush=True)

    with open(INPUT) as f:
        entries = json.load(f)
    print(f"Processing {len(entries)} entries", flush=True)

    results: dict[str, dict] = {}
    if os.path.exists(OUTPUT):
        try:
            with open(OUTPUT) as f:
                existing = json.load(f)
            for r in existing:
                pid = r.get("paper_id")
                if pid and "segmentation_error" not in r:
                    results[pid] = r
            print(f"Resuming: {len(results)} already done", flush=True)
        except Exception:
            results = {}

    pending = [e for e in entries if e["paper_id"] not in results]
    print(f"Pending: {len(pending)}", flush=True)

    lock = threading.Lock()
    done = [0]

    def snapshot():
        ordered = []
        for e in entries:
            r = results.get(e["paper_id"])
            if r:
                ordered.append(r)
        with open(OUTPUT, "w") as f:
            json.dump(ordered, f, indent=2, ensure_ascii=False, default=str)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_process_entry, e): e["paper_id"] for e in pending}
        for fut in as_completed(futs):
            pid = futs[fut]
            try:
                r = fut.result()
            except Exception as exc:
                r = {
                    "paper_id": pid,
                    "segmentation_error": str(exc),
                    "hallucinated_citations": [],
                    "num_citations": 0,
                }
            with lock:
                results[pid] = r
                done[0] += 1
                if done[0] % 25 == 0:
                    print(f"  {done[0]}/{len(pending)}", flush=True)
                    snapshot()

    snapshot()

    total = sum(r.get("num_citations", 0) for r in results.values())
    errs = sum(1 for r in results.values() if r.get("segmentation_error"))
    zero = sum(
        1
        for r in results.values()
        if r.get("num_citations", 0) == 0 and not r.get("segmentation_error")
    )
    print(f"\nDone. Papers: {len(results)}, citations: {total}, seg_errors: {errs}, zero-citation: {zero}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
