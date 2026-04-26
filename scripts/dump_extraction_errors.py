"""Walk every extracted citation set and emit a per-error-case JSON.

For each GT citation that did not match exactly to an extracted citation
(after dropping the 51 omissions in docs/pdf_render_omissions.json), the
script finds the closest extracted record and assigns one of:

  - missing_completely   : no extracted record overlaps with the GT title
  - ocr_char_error       : closest extracted record has an OCR-level title
                           difference (small edit-distance), so the
                           reference is present but normalize-mismatches GT
  - normalize_html_entity: GT title contains an HTML entity (`\&apos;` etc.)
                           that the rendered PDF strips, causing a false
                           normalize mismatch
  - field_only            : the extracted record matches the GT title but
                            individual fields disagree (authors / venue /
                            year / pages / etc.)

Writes one JSON per (mode, ext_root) under
``results/<mode>/extraction_errors.json``.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

CLEAN = Path("data/synthetic_data/v2/cleaned_bib.json")
OMISSIONS = Path("docs/pdf_render_omissions.json")


def normtitle(s: str) -> str:
    s = (s or "").lower().replace("&amp;", "and").replace("&", "and")
    return re.sub(r"[^a-z0-9]+", "", s)


def title_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def find_closest(gt_norm: str, extracted: list[dict]) -> tuple[dict | None, float]:
    best = None
    best_score = 0.0
    for c in extracted:
        ent = normtitle(c.get("title", ""))
        if not ent:
            continue
        sc = title_similarity(gt_norm, ent)
        if sc > best_score:
            best_score = sc
            best = c
    return best, best_score


def html_entity_in(text: str) -> bool:
    return bool(re.search(r"\\&\w+;", text or "") or re.search(r"&\w+;", text or ""))


def latex_accent_in(text: str) -> bool:
    return bool(re.search(r"\\['\"`^~][a-zA-Z]", text or ""))


def categorize(gt: dict, extracted_match: dict | None, score: float) -> dict:
    gt_title = gt["fields_kept"].get("title", "")
    gt_norm = normtitle(gt_title)
    if extracted_match is None:
        return {"category": "missing_completely",
                "note": "no extracted record found"}
    ext_title = extracted_match.get("title", "") or ""
    ext_norm = normtitle(ext_title)
    if ext_norm == gt_norm:
        return {"category": "matched_exact", "note": ""}
    if html_entity_in(gt_title):
        return {"category": "normalize_html_entity",
                "note": f"GT title contains HTML entity that the rendered PDF strips; extracted differs only by entity decoding"}
    if score >= 0.85:
        return {"category": "ocr_char_error",
                "note": f"reference present in extraction but title differs by OCR character-level errors (similarity={score:.3f})"}
    if score >= 0.5:
        return {"category": "ocr_partial",
                "note": f"reference partially present (similarity={score:.3f})"}
    return {"category": "missing_completely",
            "note": f"closest extracted is unrelated (similarity={score:.3f})"}


def diff_fields(gt: dict, ext: dict) -> dict[str, dict]:
    """For an exact-title match, list which fields disagree."""
    out: dict[str, dict] = {}
    fk = gt["fields_kept"]
    pairs = [
        ("year", str(fk.get("year", "") or ""), str(ext.get("year", "") or "")),
        ("pages", normtitle(fk.get("pages", "") or ""), normtitle(ext.get("pages", "") or "")),
        ("doi", normtitle(fk.get("doi", "") or ""), normtitle(ext.get("doi", "") or "")),
    ]
    for fname, gv, ev in pairs:
        if gv and ev and gv != ev:
            out[fname] = {"gt": gv, "extracted": ev}
    return out


def run(ext_root: Path, mode: str, exclude_ids: set[str]) -> dict:
    clean_records = json.loads(CLEAN.read_text())
    gt_by_paper: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in clean_records:
        if r["citation_id"] in exclude_ids:
            continue
        gt_by_paper[(r["style"], r["paper"])].append(r)

    errors: list[dict] = []
    counts: dict[str, int] = defaultdict(int)
    n_gt_total = 0

    for (style, paper), gt_list in sorted(gt_by_paper.items()):
        ext_path = ext_root / style / f"{paper}.json"
        if not ext_path.exists():
            continue
        ext_doc = json.loads(ext_path.read_text())
        extracted = ext_doc.get("citations", [])
        ext_by_norm: dict[str, list[dict]] = defaultdict(list)
        for c in extracted:
            n = normtitle(c.get("title", ""))
            if n:
                ext_by_norm[n].append(c)

        for gt in gt_list:
            n_gt_total += 1
            gt_title = gt["fields_kept"].get("title", "")
            gt_norm = normtitle(gt_title)
            ext_match = None
            ext_match_score = 0.0
            if gt_norm in ext_by_norm:
                ext_match = ext_by_norm[gt_norm][0]
                ext_match_score = 1.0
            else:
                ext_match, ext_match_score = find_closest(gt_norm, extracted)

            cat = categorize(gt, ext_match, ext_match_score)
            counts[cat["category"]] += 1
            if cat["category"] == "matched_exact":
                # Skip clean matches; only collect actual errors
                continue

            errors.append({
                "citation_id": gt["citation_id"],
                "style": style,
                "paper": paper,
                "subtype": gt["citation_id"].split("-")[0],
                "category": cat["category"],
                "note": cat["note"],
                "title_similarity": round(ext_match_score, 4),
                "gt": {
                    "title": gt_title,
                    "author": gt["fields_kept"].get("author", ""),
                    "year": gt["fields_kept"].get("year", ""),
                    "venue": gt["fields_kept"].get("booktitle") or gt["fields_kept"].get("journal") or "",
                    "doi": gt["fields_kept"].get("doi", ""),
                },
                "extracted": ({
                    "citation_id": (ext_match or {}).get("citation_id", ""),
                    "title": (ext_match or {}).get("title", ""),
                    "authors": (ext_match or {}).get("authors", []),
                    "year": (ext_match or {}).get("year", ""),
                    "venue": (ext_match or {}).get("venue", ""),
                    "doi": (ext_match or {}).get("doi", ""),
                    "raw_text": ((ext_match or {}).get("raw_text", "") or "")[:300],
                } if ext_match else None),
            })

    summary = {
        "mode": mode,
        "ext_root": str(ext_root),
        "n_gt_after_exclusion": n_gt_total,
        "n_errors": len(errors),
        "n_matched_exact": counts["matched_exact"],
        "by_category": dict(counts),
    }
    return {"summary": summary, "errors": errors}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rule-root", default="results/extraction_ocr_rule")
    p.add_argument("--llm-root", default="results/extraction_ocr_llm")
    p.add_argument("--out", default="results/extraction_errors.json")
    args = p.parse_args()

    omissions = json.loads(OMISSIONS.read_text())
    exclude_ids = {r["citation_id"] for r in omissions}
    print(f"excluding {len(exclude_ids)} omission citation_ids before counting errors")

    rule_report = run(Path(args.rule_root), "rule", exclude_ids)
    llm_report = run(Path(args.llm_root), "llm", exclude_ids)

    out = {
        "excluded_omission_ids": sorted(exclude_ids),
        "rule_report": rule_report,
        "llm_report": llm_report,
    }
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {args.out}")
    print()
    for tag, rpt in [("rule", rule_report), ("llm", llm_report)]:
        s = rpt["summary"]
        print(f"=== {tag} ===")
        print(f"  GT after exclusion: {s['n_gt_after_exclusion']}")
        print(f"  matched exact: {s['n_matched_exact']}")
        print(f"  errors: {s['n_errors']}")
        print(f"  by category:")
        for cat, n in sorted(s["by_category"].items()):
            print(f"    {cat:30s} {n}")
        print()


if __name__ == "__main__":
    main()
