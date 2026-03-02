from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

# Allow running this script from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.pdf_checker.config import load_pdf_checker_config
from apps.pdf_checker.ingest.citation_parser import parse_reference_entries


def _core_missing(citation: dict[str, Any]) -> bool:
    return (
        not str(citation.get("title", "") or "").strip()
        or not citation.get("authors")
        or not str(citation.get("venue", "") or "").strip()
        or citation.get("year") is None
    )


def _missing_any_of_7(citation: dict[str, Any]) -> bool:
    return (
        not str(citation.get("title", "") or "").strip()
        or not citation.get("authors")
        or not str(citation.get("venue", "") or "").strip()
        or citation.get("year") is None
        or not str(citation.get("doi", "") or "").strip()
        or not str(citation.get("arxiv_id", "") or "").strip()
        or not str(citation.get("url", "") or "").strip()
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply parser + optional LLM reparsing to existing artifact JSON files."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing existing artifact JSON files (with citations[].raw_text).",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Default: artifacts/<input_dir_name>_llm_reparse_test_<UTC timestamp>",
    )
    parser.add_argument(
        "--config-path",
        default="",
        help="Path to config JSON. If provided, sets CITATION_CHECKER_CONFIG_PATH for this run.",
    )
    parser.add_argument(
        "--model-path",
        default="",
        help="Override citation_reparse.model_path from config.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=0,
        help="Override citation_reparse.max_new_tokens when > 0.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=-1.0,
        help="Override citation_reparse.temperature when >= 0.",
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Also process files starting with '_' (default: skip).",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.config_path:
        os.environ["CITATION_CHECKER_CONFIG_PATH"] = args.config_path

    cfg = load_pdf_checker_config()
    llm_cfg = asdict(cfg.citation_reparse)
    llm_cfg["enabled"] = True

    if args.model_path:
        llm_cfg["model_path"] = args.model_path
    if args.max_new_tokens > 0:
        llm_cfg["max_new_tokens"] = int(args.max_new_tokens)
    if args.temperature >= 0:
        llm_cfg["temperature"] = float(args.temperature)

    model_path = str(llm_cfg.get("model_path") or "").strip()
    if not model_path:
        raise RuntimeError(
            "LLM reparse model path is empty. Set citation_reparse.model_path in config or pass --model-path."
        )

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise RuntimeError(f"Input directory does not exist: {input_dir}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("artifacts") / f"{input_dir.name}_llm_reparse_test_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))
    if not args.include_hidden:
        json_files = [p for p in json_files if not p.name.startswith("_")]
    json_files = [p for p in json_files if p.name != "incomplete_entries.json"]

    stats: dict[str, Any] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "files_total": len(json_files),
        "files_processed": 0,
        "files_changed": 0,
        "citations_total": 0,
        "citations_with_core_missing_before": 0,
        "citations_with_core_missing_after": 0,
        "citations_with_any7_missing_before": 0,
        "citations_with_any7_missing_after": 0,
        "llm_reparse_attempted": 0,
        "llm_reparse_with_any_applied_fields": 0,
        "llm_reparse_errors": 0,
        "llm_reparse_config": llm_cfg,
    }

    for src in json_files:
        data = json.loads(src.read_text(encoding="utf-8"))
        citations = data.get("citations", [])
        if not isinstance(citations, list):
            dst = output_dir / src.name
            dst.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            continue

        stats["files_processed"] += 1
        changed = False

        valid_indices: list[int] = []
        entries: list[str] = []

        for idx, citation in enumerate(citations):
            if not isinstance(citation, dict):
                continue
            raw_text = str(citation.get("raw_text", "") or "").strip()
            if not raw_text:
                continue
            valid_indices.append(idx)
            entries.append(raw_text)
            stats["citations_total"] += 1
            if _core_missing(citation):
                stats["citations_with_core_missing_before"] += 1
            if _missing_any_of_7(citation):
                stats["citations_with_any7_missing_before"] += 1

        if entries:
            parsed = parse_reference_entries(entries, llm_reparse_config=llm_cfg)
            for citation_idx, record in zip(valid_indices, parsed):
                target = citations[citation_idx]
                new_fields = {
                    "title": record.title,
                    "authors": record.authors,
                    "venue": record.venue,
                    "year": record.year,
                    "doi": record.doi,
                    "arxiv_id": record.arxiv_id,
                    "url": record.url,
                    "parsed_fields": record.parsed_fields,
                }
                for key, value in new_fields.items():
                    if target.get(key) != value:
                        target[key] = value
                        changed = True

                parsed_fields = record.parsed_fields or {}
                if parsed_fields.get("llm_reparse_attempted"):
                    stats["llm_reparse_attempted"] += 1
                if parsed_fields.get("llm_reparse_error"):
                    stats["llm_reparse_errors"] += 1
                if parsed_fields.get("llm_reparse_applied_fields"):
                    stats["llm_reparse_with_any_applied_fields"] += 1

                if _core_missing(target):
                    stats["citations_with_core_missing_after"] += 1
                if _missing_any_of_7(target):
                    stats["citations_with_any7_missing_after"] += 1

        if changed:
            stats["files_changed"] += 1

        dst = output_dir / src.name
        dst.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary_path = output_dir / "_llm_reparse_test_summary.json"
    summary_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"summary_file={summary_path}")


if __name__ == "__main__":
    main()
