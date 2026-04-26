"""Post-process the image-assisted LLM run by replacing the LLM-modified
fields with the corresponding values from the text-only LLM run.

Why: image-assisted LLM reparse can silently "correct" fields that the
benchmark relies on for hallucination detection. The clearest case is
``year``: H4 mutations introduce wrong years on purpose, but the image
LLM tends to override the printed year with the conference year it sees
in the venue name (or with its own training knowledge). Reverting these
fields back to what the text-only LLM extracted preserves the H4-style
detection signal while keeping the OCR cache and segmentation that
``--with-images`` produced.

Usage:
    # Revert only year (default)
    python scripts/revert_image_to_text_fields.py

    # Revert every field listed in metadata.llm_reparse_applied_fields
    python scripts/revert_image_to_text_fields.py --revert-applied

    # Revert a custom subset
    python scripts/revert_image_to_text_fields.py --fields year,title,venue
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_REVERT_FIELDS = ("year",)
STYLES = ("acm", "splncs04", "iclr", "ieee", "plain")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--text-root", default="results/extraction_ocr_llm",
                   help="Source of fields to copy back (text-only LLM run).")
    p.add_argument("--image-root", default="results/extraction_ocr_llm_image",
                   help="Run to mutate.")
    p.add_argument("--out-root", default="results/extraction_ocr_llm_image_reverted",
                   help="Where to write the reverted run.")
    p.add_argument("--fields", default=",".join(DEFAULT_REVERT_FIELDS),
                   help="Comma-separated fields to revert (default: %(default)s).")
    p.add_argument("--revert-applied", action="store_true",
                   help="Revert every field listed in each citation's "
                        "llm_reparse_applied_fields metadata, in addition to --fields.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fields_to_revert = tuple(s.strip() for s in args.fields.split(",") if s.strip())
    text_root = Path(args.text_root)
    image_root = Path(args.image_root)
    out_root = Path(args.out_root)

    n_papers = 0
    n_cites = 0
    n_reverted_per_field: dict[str, int] = {}

    for style in STYLES:
        for image_path in sorted((image_root / style).glob("paper_*.json")):
            text_path = text_root / style / image_path.name
            if not text_path.exists():
                continue
            text_doc = json.loads(text_path.read_text())
            image_doc = json.loads(image_path.read_text())
            text_by_id = {c["citation_id"]: c for c in text_doc.get("citations", [])}

            for ic in image_doc.get("citations", []):
                tc = text_by_id.get(ic["citation_id"])
                if tc is None:
                    continue
                applied = (ic.get("parsed_fields", {}) or {}).get("llm_reparse_applied_fields", []) or []
                target_fields: set[str] = set(fields_to_revert)
                if args.revert_applied:
                    target_fields.update(applied)
                for f in target_fields:
                    if f not in ic:
                        continue
                    if ic.get(f) != tc.get(f):
                        ic[f] = tc.get(f)
                        n_reverted_per_field[f] = n_reverted_per_field.get(f, 0) + 1
                # Also flag in metadata that this field was reverted
                pf = ic.setdefault("parsed_fields", {})
                pf.setdefault("reverted_to_text_only", []).extend(sorted(target_fields))
                pf["reverted_to_text_only"] = sorted(set(pf["reverted_to_text_only"]))
                n_cites += 1

            out_path = out_root / style / image_path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(image_doc, indent=2, ensure_ascii=False))
            n_papers += 1

    print(f"wrote {n_papers} papers (over {n_cites} citations) -> {out_root}")
    print("revert counts per field:")
    for f, n in sorted(n_reverted_per_field.items()):
        print(f"  {f:20s} {n}")


if __name__ == "__main__":
    main()
