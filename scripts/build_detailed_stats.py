"""Build a per-citation detailed stats table for P1/P3/H1-H6 using the current
(cleaned) datasets + latest rule_based eval results.

For each citation in the current dataset:
  - Look up matching eval result (by citation_id first, then by normalized title + year)
  - Record expected subtype, predicted label, predicted taxonomy

Outputs:
  - results/detailed_stats.csv          (one row per citation)
  - results/detailed_stats_summary.json (aggregate counts)
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from packages.core.normalize import normalize_title

DATA_DIR = Path("data/synthetic_data/v2")
RESULTS_DIR = Path("results")

TARGETS = ["R1", "R2", "R3", "R1_plus", "R2_plus", "R3_plus",
           "P1", "P3", "H1", "H2", "H3", "H4", "H5", "H6"]

# meta-label → pipeline-verdict label
META_TO_VERDICT = {
    "REAL": "VALID",
    "HALLUCINATED": "FAKE_REFERENCE",
    "POTENTIAL_HALLUCINATED": "POTENTIAL_REFERENCE",
}


def load_eval_index(subtype: str):
    """Return (by_id, by_title_year) indexes of eval results for `subtype`."""
    p = RESULTS_DIR / f"eval_{subtype}_rule_based.json"
    if not p.exists():
        return {}, {}
    d = json.loads(p.read_text())
    by_id = {}
    by_ty = {}
    for r in d.get("results", []):
        cid = r.get("citation_id", "")
        by_id[cid] = r
        ic = r.get("input_citation") or {}
        key = (normalize_title(ic.get("title", "")), ic.get("year"))
        by_ty[key] = r
    return by_id, by_ty


def main() -> None:
    meta = json.loads((DATA_DIR / "meta.json").read_text())

    rows: list[dict] = []
    per_sub = defaultdict(lambda: {"total": 0, "label_correct": 0,
                                     "tax_correct": 0, "has_eval": 0})

    for sub in TARGETS:
        ds_path = DATA_DIR / f"{sub}.json"
        if not ds_path.exists():
            continue
        entries = json.loads(ds_path.read_text())
        by_id, by_ty = load_eval_index(sub)

        for e in entries:
            cid = e["citation_id"]
            title = e.get("title", "")
            year = e.get("year")
            m = meta.get(cid) or {}
            expected_subtype = m.get("subtype", sub)
            expected_label_raw = m.get("label", "")
            expected_label = META_TO_VERDICT.get(expected_label_raw, expected_label_raw)
            # R*_plus entries have no meta; they are all Real by construction.
            if not expected_label and sub in ("R1_plus", "R2_plus", "R3_plus"):
                expected_label = "VALID"
            # R*_plus subtypes map to R1/R2/R3 for taxonomy comparison.
            subtype_for_tax = expected_subtype.replace("_plus", "")

            # 1) direct id lookup; 2) title+year fallback
            ev = by_id.get(cid)
            matched_by = "id"
            if ev is None:
                ev = by_ty.get((normalize_title(title), year))
                matched_by = "title_year" if ev else "none"

            predicted_label = (ev or {}).get("predicted", "")
            predicted_tax = (ev or {}).get("taxonomy") or []
            match_label = (predicted_label == expected_label) if ev else None
            match_tax = (subtype_for_tax in predicted_tax) if ev else None

            rows.append({
                "citation_id": cid,
                "subtype": expected_subtype,
                "expected_label": expected_label,
                "predicted_label": predicted_label,
                "predicted_taxonomy": "|".join(predicted_tax),
                "match_label": "" if match_label is None else ("1" if match_label else "0"),
                "match_taxonomy": "" if match_tax is None else ("1" if match_tax else "0"),
                "matched_by": matched_by,
                "title": (title or "")[:100],
                "year": year or "",
            })

            bucket = per_sub[expected_subtype]
            bucket["total"] += 1
            if ev is not None:
                bucket["has_eval"] += 1
                if match_label: bucket["label_correct"] += 1
                if match_tax: bucket["tax_correct"] += 1

    # Write CSV
    out_csv = RESULTS_DIR / "detailed_stats.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        cols = ["citation_id","subtype","expected_label","predicted_label",
                "predicted_taxonomy","match_label","match_taxonomy","matched_by",
                "title","year"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote CSV: {out_csv} ({len(rows)} rows)")

    # Summary JSON
    summary = {}
    for sub in sorted(per_sub):
        b = per_sub[sub]
        summary[sub] = {
            "total": b["total"],
            "has_eval": b["has_eval"],
            "label_correct": b["label_correct"],
            "tax_correct": b["tax_correct"],
            "label_acc": b["label_correct"]/b["has_eval"] if b["has_eval"] else None,
            "tax_acc": b["tax_correct"]/b["has_eval"] if b["has_eval"] else None,
        }
    out_json = RESULTS_DIR / "detailed_stats_summary.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote summary: {out_json}")

    # Pretty table
    print()
    print(f"{'subtype':<8} {'total':>6} {'has_eval':>9} {'label_acc':>10} {'tax_acc':>10} {'label_err':>10} {'tax_err':>10}")
    print("-"*66)
    for sub in sorted(summary):
        s = summary[sub]
        la = f"{s['label_acc']*100:.1f}%" if s['label_acc'] is not None else "-"
        ta = f"{s['tax_acc']*100:.1f}%" if s['tax_acc'] is not None else "-"
        print(f"{sub:<8} {s['total']:>6} {s['has_eval']:>9} {la:>10} {ta:>10} "
              f"{s['has_eval']-s['label_correct']:>10} {s['has_eval']-s['tax_correct']:>10}")

    # Missing-eval listing
    missing = [r for r in rows if r["matched_by"] == "none"]
    if missing:
        print(f"\nCitations missing eval: {len(missing)}")
        for r in missing[:10]:
            print(f"  {r['citation_id']}  {r['title'][:60]}")


if __name__ == "__main__":
    main()
