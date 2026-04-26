"""Per-field accuracy over **all** ground-truth citations (not just matched).

Difference from ``score_extraction_clean.py``:
- That script pairs extracted ↔ GT by normalized title and computes per-field
  accuracy *within the matched pool*, so an OCR-character-error title that
  cannot be paired drops out of every per-field column.
- This script pairs by **fuzzy title similarity** (SequenceMatcher ratio
  >= ``--title-threshold``, default 0.85) so OCR-character-error references
  are still paired, then computes per-field accuracy with the denominator
  set to **every GT citation that has the field present** (after dropping
  the 58 omissions in ``docs/pdf_render_omissions.json``).

Output columns: per-field-accuracy = (# GT where field matches the paired
extraction) / (# GT where field is present). For unpaired GT, every field
counts as a miss.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style_aware_match import (  # noqa: E402
    authors_match, normalize_id, normalize_simple, normalize_title,
    normalize_venue, simple_match, pages_match, venue_match, year_match,
)

CLEAN = Path("data/synthetic_data/v2/cleaned_bib.json")
OMISSIONS = Path("docs/pdf_render_omissions.json")

BIB_TO_EXT = {
    "title":        "title",
    "author":       "authors",
    "year":         "year",
    "booktitle":    "venue",
    "journal":      "venue",
    "howpublished": "venue",
    "school":       "venue",
    "institution":  "venue",
    "pages":        "pages",
    "volume":       "volume",
    "publisher":    "publisher",
    "doi":          "doi",
    "url":          "url",
    "eprint":       "arxiv_id",
    "address":      "location",
}


def title_sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def field_match(ext_field: str, bib_value, extracted: dict) -> bool:
    if ext_field == "title":
        return normalize_title(extracted.get("title", "")) == normalize_title(bib_value)
    if ext_field == "authors":
        return authors_match(extracted.get("authors", []), bib_value)
    if ext_field == "venue":
        return venue_match(extracted.get("venue", ""), bib_value)
    if ext_field == "year":
        return year_match(extracted.get("year", ""), bib_value)
    if ext_field == "pages":
        return pages_match(extracted.get("pages", ""), bib_value)
    if ext_field == "arxiv_id":
        ne = normalize_simple(extracted.get("arxiv_id", ""))
        ng = normalize_simple(bib_value)
        return bool(ng) and ne == ng
    if ext_field in ("volume", "publisher", "location"):
        return simple_match(extracted.get(ext_field, ""), bib_value)
    return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ext-root", required=True)
    p.add_argument("--out-root", default=None)
    p.add_argument("--tag", default="")
    p.add_argument("--title-threshold", type=float, default=0.85)
    p.add_argument("--exclude-omissions", action="store_true", default=True)
    p.add_argument("--no-exclude-omissions", dest="exclude_omissions", action="store_false")
    args = p.parse_args()

    EXT_ROOT = Path(args.ext_root)
    OUT_ROOT = Path(args.out_root) if args.out_root else EXT_ROOT
    TAG = f"_{args.tag}" if args.tag else ""

    exclude_ids = set()
    if args.exclude_omissions and OMISSIONS.exists():
        exclude_ids = {r["citation_id"] for r in json.loads(OMISSIONS.read_text())}
        print(f"excluding {len(exclude_ids)} omission citations")

    clean_records = json.loads(CLEAN.read_text())
    gt_by_paper: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in clean_records:
        if r["citation_id"] in exclude_ids:
            continue
        gt_by_paper[(r["style"], r["paper"])].append(r)

    fields = ["title", "authors", "venue", "year", "identifier",
              "pages", "volume", "publisher", "location", "arxiv_id"]

    csv_rows = []
    agg = defaultdict(lambda: {"papers": 0, "n_gt": 0, "n_ext": 0,
                                "title_paired": 0,
                                **{f: {"match": 0, "total": 0} for f in fields}})

    for (style, paper), gt_list in sorted(gt_by_paper.items()):
        ext_path = EXT_ROOT / style / f"{paper}.json"
        if not ext_path.exists():
            continue
        ext_doc = json.loads(ext_path.read_text())
        extracted = ext_doc.get("citations") or ext_doc.get("references") or []
        ext_norms = [(c, normalize_title(c.get("title", ""))) for c in extracted]

        per_field_match: Counter = Counter()
        per_field_total: Counter = Counter()
        title_paired = 0

        for gt in gt_list:
            kept = gt["fields_kept"]
            gt_title_norm = normalize_title(kept.get("title", ""))

            # Fuzzy pair: best-similarity extracted whose similarity >= threshold
            best = None
            best_sim = 0.0
            for c, en in ext_norms:
                if not en:
                    continue
                if en == gt_title_norm:
                    best = c; best_sim = 1.0
                    break
                sim = title_sim(gt_title_norm, en)
                if sim > best_sim:
                    best_sim = sim; best = c
            paired = best is not None and best_sim >= args.title_threshold

            # ---- title field ----
            if kept.get("title"):
                per_field_total["title"] += 1
                if paired and best_sim == 1.0:
                    per_field_match["title"] += 1
                # Note: title is "match" only on exact normalize equality.
                # Fuzzy-paired but inexact still counts as title miss.

            if not paired:
                # Unpaired: every present field counts as miss in totals
                for bib_f in BIB_TO_EXT:
                    if bib_f in ("doi", "url"): continue
                    if bib_f == "title": continue  # already counted above
                    ex_f = BIB_TO_EXT[bib_f]
                    if bib_f in kept:
                        if ex_f == "venue" and per_field_total.get("__venue_seen") == gt["citation_id"]:
                            continue
                        per_field_total[ex_f] += 1
                        if ex_f == "venue":
                            per_field_total["__venue_seen"] = gt["citation_id"]
                # identifier
                if kept.get("doi") or kept.get("url"):
                    per_field_total["identifier"] += 1
                continue

            title_paired += 1

            # identifier (union doi+url)
            kept_doi = kept.get("doi", "")
            kept_url = kept.get("url", "")
            if kept_doi or kept_url:
                ex_doi, ex_url = best.get("doi", ""), best.get("url", "")
                ex_ids = {normalize_id(x) for x in [ex_doi, ex_url] if normalize_id(x)}
                gt_ids = {normalize_id(x) for x in [kept_doi, kept_url] if normalize_id(x)}
                hit = bool(ex_ids & gt_ids) or any(
                    a and b and (a in b or b in a) for a in ex_ids for b in gt_ids
                )
                per_field_total["identifier"] += 1
                if hit:
                    per_field_match["identifier"] += 1

            for bib_f, ex_f in BIB_TO_EXT.items():
                if bib_f in ("doi", "url", "title"): continue
                if bib_f not in kept: continue
                if ex_f == "venue" and per_field_total.get("__venue_seen") == gt["citation_id"]:
                    continue
                per_field_total[ex_f] += 1
                if field_match(ex_f, kept[bib_f], best):
                    per_field_match[ex_f] += 1
                if ex_f == "venue":
                    per_field_total["__venue_seen"] = gt["citation_id"]

        row = {"style": style, "paper": paper,
               "n_gt": len(gt_list), "n_ext": len(extracted),
               "title_paired": title_paired}
        for f in fields:
            m, t = per_field_match[f], per_field_total[f]
            row[f"{f}_acc"] = f"{m / t:.3f}" if t else "n/a"
        csv_rows.append(row)

        b = agg[style]
        b["papers"] += 1
        b["n_gt"] += len(gt_list)
        b["n_ext"] += len(extracted)
        b["title_paired"] += title_paired
        for f in fields:
            b[f]["match"] += per_field_match[f]
            b[f]["total"] += per_field_total[f]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_ROOT / f"scoring_all{TAG}.csv"
    cols = ["style", "paper", "n_gt", "n_ext", "title_paired"] + [f"{f}_acc" for f in fields]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in csv_rows: w.writerow(r)
    print(f"wrote {csv_path}")

    summary = {}
    for style, b in agg.items():
        s = {"papers": b["papers"], "n_gt": b["n_gt"], "n_ext": b["n_ext"],
             "title_paired": b["title_paired"],
             "title_pair_rate": b["title_paired"] / b["n_gt"] if b["n_gt"] else 0,
             "fields": {}}
        for f in fields:
            m, t = b[f]["match"], b[f]["total"]
            s["fields"][f] = {"match": m, "total": t,
                              "accuracy": (m / t) if t else None}
        summary[style] = s
    sum_path = OUT_ROOT / f"scoring_all_summary{TAG}.json"
    sum_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {sum_path}\n")

    # Pretty
    print(f"{'style':<10} {'papers':>6} {'n_gt':>5} {'paired':>6} " +
          "  ".join(f"{f[:8]:>8}" for f in fields))
    for style in ("acm", "splncs04", "iclr", "ieee", "plain"):
        s = summary.get(style)
        if not s: continue
        fa = "  ".join(
            f"{(s['fields'][f]['accuracy'] or 0)*100:>8.1f}" for f in fields)
        print(f"{style:<10} {s['papers']:>6} {s['n_gt']:>5} {s['title_paired']:>6} {fa}")


if __name__ == "__main__":
    main()
