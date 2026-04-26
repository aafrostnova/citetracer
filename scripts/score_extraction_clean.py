"""Re-score the PDF extractor against `cleaned_bib.json` ground truth — i.e.
only over fields that the rendering bst actually kept on the page. Fields
the bst dropped are skipped, so the extractor is not penalised for
information that never reached the PDF.

Outputs:
  results/extraction/scoring_clean.csv          (per-paper)
  results/extraction/scoring_clean_summary.json (per-style aggregate)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style_aware_match import (  # noqa: E402
    authors_match, identifier_match, normalize_id, normalize_pages,
    normalize_simple, normalize_title, normalize_venue, simple_match,
    pages_match, venue_match, year_match,
)

CLEAN = Path("data/synthetic_data/v2/cleaned_bib.json")
EXT_ROOT = Path("results/extraction")
OUT_ROOT = Path("results/extraction")
TAG = ""

# bib field name -> extracted field name
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

# extracted-field-name -> comparison style
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
        ng = normalize_simple(bib_value)
        if not ng:
            return False
        ne = normalize_simple(extracted.get("arxiv_id", ""))
        if ne == ng:
            return True
        # Fallback: arxiv id may live in the url field as "arxiv.org/abs/<id>".
        url = (extracted.get("url", "") or "").lower()
        m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9a-zA-Z./_\-]+?)(?:\.pdf)?(?:[?#].*)?$", url)
        if m and normalize_simple(m.group(1)) == ng:
            return True
        return False
    if ext_field in ("volume", "publisher", "location"):
        return simple_match(extracted.get(ext_field, ""), bib_value)
    return False


OMISSIONS = Path("docs/pdf_render_omissions.json")


def load_clean(exclude_ids: set[str] | None = None) -> dict[tuple[str, str], list[dict]]:
    exclude_ids = exclude_ids or set()
    out: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in json.loads(CLEAN.read_text()):
        if r["citation_id"] in exclude_ids:
            continue
        out[(r["style"], r["paper"])].append(r)
    return out


def load_omission_ids(reasons: tuple[str, ...] | None = None) -> set[str]:
    """Read docs/pdf_render_omissions.json and return the set of citation_ids
    to drop. ``reasons`` filters by ``omission_reason`` (default: all)."""
    if not OMISSIONS.exists():
        return set()
    out: set[str] = set()
    for r in json.loads(OMISSIONS.read_text()):
        if reasons is None or r.get("omission_reason") in reasons:
            out.add(r["citation_id"])
    return out


def main() -> None:
    global EXT_ROOT, OUT_ROOT, TAG
    p = argparse.ArgumentParser()
    p.add_argument("--ext-root", default=str(EXT_ROOT),
                   help="Directory containing per-style extracted JSONs (default: %(default)s)")
    p.add_argument("--out-root", default=None,
                   help="Where to write scoring CSV/JSON (default: same as --ext-root)")
    p.add_argument("--tag", default="",
                   help="Suffix appended to output filenames (e.g. 'rule' -> scoring_clean_rule.csv)")
    p.add_argument("--exclude-omissions", action="store_true",
                   help="Drop the citations listed in docs/pdf_render_omissions.json (latex accent escapes + bst-suppressed fields).")
    p.add_argument("--exclude-reasons", default="",
                   help="Comma-separated subset of omission_reason values to exclude (default: all when --exclude-omissions is set).")
    args = p.parse_args()
    EXT_ROOT = Path(args.ext_root)
    OUT_ROOT = Path(args.out_root) if args.out_root else EXT_ROOT
    TAG = f"_{args.tag}" if args.tag else ""

    exclude_ids: set[str] = set()
    if args.exclude_omissions:
        reasons = tuple(s.strip() for s in args.exclude_reasons.split(",") if s.strip()) or None
        exclude_ids = load_omission_ids(reasons)
        print(f"excluding {len(exclude_ids)} omission citations (reasons={reasons or 'all'})")
    clean = load_clean(exclude_ids=exclude_ids)
    fields = ["title", "authors", "venue", "year", "identifier",
              "pages", "volume", "publisher", "location", "arxiv_id"]

    csv_rows = []
    agg = defaultdict(lambda: {"papers": 0, "n_extracted": 0, "n_ground_truth": 0,
                                "matched_pairs": 0,
                                **{f: {"match": 0, "total": 0} for f in fields}})

    for (style, paper), records in clean.items():
        ext_path = EXT_ROOT / style / f"{paper}.json"
        if not ext_path.exists():
            continue
        ext_doc = json.loads(ext_path.read_text())
        extracted = ext_doc.get("citations") or ext_doc.get("references") or []

        # Pair extracted against cleaned ground truth by normalized title
        gt_by_title = {normalize_title(r["fields_kept"].get("title", "")): r for r in records}
        used = set()
        per_field_match: dict[str, int] = Counter()
        per_field_total: dict[str, int] = Counter()
        matched = 0

        for ex in extracted:
            t = normalize_title(ex.get("title", ""))
            gt = gt_by_title.get(t)
            if gt is None or gt["citation_id"] in used:
                continue
            used.add(gt["citation_id"])
            matched += 1
            kept = gt["fields_kept"]
            # Identifier handled separately (union doi+url)
            kept_doi = kept.get("doi", "")
            kept_url = kept.get("url", "")
            if kept_doi or kept_url:
                ex_doi, ex_url = ex.get("doi", ""), ex.get("url", "")
                ex_ids = {normalize_id(x) for x in [ex_doi, ex_url] if normalize_id(x)}
                gt_ids = {normalize_id(x) for x in [kept_doi, kept_url] if normalize_id(x)}
                hit = bool(ex_ids & gt_ids) or any(
                    a and b and (a in b or b in a) for a in ex_ids for b in gt_ids
                )
                per_field_total["identifier"] += 1
                if hit:
                    per_field_match["identifier"] += 1
            # Other fields
            for bib_f, ex_f in BIB_TO_EXT.items():
                if bib_f in ("doi", "url"):
                    continue
                if bib_f not in kept:
                    continue
                # If bib has multiple venue-like fields, only the first one counted
                if ex_f == "venue" and per_field_total.get("__venue_seen") == gt["citation_id"]:
                    continue
                per_field_total[ex_f] += 1
                if field_match(ex_f, kept[bib_f], ex):
                    per_field_match[ex_f] += 1
                if ex_f == "venue":
                    per_field_total["__venue_seen"] = gt["citation_id"]

        row = {"style": style, "paper": paper,
               "n_extracted": len(extracted),
               "n_ground_truth": len(records),
               "matched_pairs": matched}
        for f in fields:
            m, t = per_field_match[f], per_field_total[f]
            row[f"{f}_acc"] = f"{m / t:.3f}" if t else "n/a"
        csv_rows.append(row)

        b = agg[style]
        b["papers"] += 1
        b["n_extracted"] += len(extracted)
        b["n_ground_truth"] += len(records)
        b["matched_pairs"] += matched
        for f in fields:
            b[f]["match"] += per_field_match[f]
            b[f]["total"] += per_field_total[f]

    # CSV
    csv_path = OUT_ROOT / f"scoring_clean{TAG}.csv"
    cols = ["style", "paper", "n_extracted", "n_ground_truth", "matched_pairs"] + [f"{f}_acc" for f in fields]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in csv_rows: w.writerow(r)
    print(f"wrote {csv_path}")

    # JSON summary
    summary = {}
    for style, b in agg.items():
        s = {"papers": b["papers"],
             "n_extracted": b["n_extracted"],
             "n_ground_truth": b["n_ground_truth"],
             "matched_pairs": b["matched_pairs"],
             "recall": b["matched_pairs"] / b["n_ground_truth"] if b["n_ground_truth"] else 0,
             "fields": {}}
        for f in fields:
            m, t = b[f]["match"], b[f]["total"]
            s["fields"][f] = {"match": m, "total": t,
                              "accuracy": (m / t) if t else None}
        summary[style] = s
    sum_path = OUT_ROOT / f"scoring_clean_summary{TAG}.json"
    sum_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {sum_path}\n")

    # Pretty table
    print(f"{'style':<10} {'papers':>6} {'matched/gt':>13} {'recall':>8}  " +
          "  ".join(f"{f[:8]:>8}" for f in fields))
    for style in ("acm", "splncs04", "iclr", "ieee", "plain"):
        s = summary.get(style)
        if not s: continue
        fa = "  ".join(
            f"{(s['fields'][f]['accuracy'] or 0)*100:>8.1f}" for f in fields)
        print(f"{style:<10} {s['papers']:>6} {s['matched_pairs']:>5}/{s['n_ground_truth']:<5} "
              f"{s['recall']*100:>7.1f}%  {fa}")


if __name__ == "__main__":
    main()
