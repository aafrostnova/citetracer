"""Audit every dataset file for mutation consistency against its meta spec.

Reports per-file:
  - total count
  - issue buckets (with citation_id examples)
  - key stats (e.g. identifier-masked pct for R*)

Does NOT modify any file.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from packages.core.normalize import normalize_title

DATA_DIR = Path("data/synthetic_data/v2")
HARVEST_PATH = Path("data/bib/harvested_seeds_clean.json")

IDENT_FIELDS = ("url", "doi", "arxiv_id")


def load_json(p: Path):
    return json.loads(p.read_text())


def is_initials_author(a: str) -> bool:
    """True if author starts with an initial (X.) or a lowercase particle.
    Accepts both 'S. W. Zamir' (all initials) and 'S. Waqas Zamir'
    (first-initial-only) — both are valid real-world abbreviation conventions.
    """
    s = (a or "").strip()
    if not s:
        return False
    parts = s.split()
    if len(parts) < 2:
        return False
    particles = {"de","van","von","der","den","du","el","la","le","da","di","bin","ben","des","del","dos","do","y"}
    first = parts[0]
    if not re.match(r"^[A-Z]\.$", first) and first.lower() not in particles:
        return False
    return True


def audit_real(name: str, entries: list[dict], meta: dict, expect_mutation: str,
               harvest_idx: dict | None = None) -> dict:
    """Audit R1/R2/R3/R*_plus files."""
    issues = defaultdict(list)
    n_masked_all = 0

    for e in entries:
        cid = e["citation_id"]
        url, doi, ax = [e.get(f, "") for f in IDENT_FIELDS]
        all_empty = not (url or doi or ax)
        if all_empty:
            n_masked_all += 1
        # Partial masking check: only all-three-empty or all-preserved
        # (partial is allowed if original already had some empty)
        # So we only flag if this file was supposed to have "together masking"

        # Get the source/seed for comparison
        if cid in meta:
            seed = (meta[cid].get("seed") or {})
        elif harvest_idx is not None:
            seed = harvest_idx.get(
                (normalize_title(e.get("title","")), e.get("year")), {}
            ) or {}
        else:
            seed = {}

        # Audit by mutation type
        if expect_mutation == "none":
            # R1: everything except identifiers should match seed
            for f in ("title", "authors", "venue", "year", "pages", "volume",
                      "publisher", "location"):
                if e.get(f) != seed.get(f):
                    issues[f"R1_{f}_differs_from_seed"].append(cid)
                    break
        elif expect_mutation == "format_variant":
            # R2: title lowercased, authors initials
            if e.get("title") and e.get("title") != e.get("title", "").lower():
                issues["R2_title_not_lowercase"].append(cid)
            for a in e.get("authors") or []:
                if not is_initials_author(a):
                    issues["R2_author_not_initials"].append((cid, a))
                    break
        elif expect_mutation == "et_al_abbreviation":
            authors = e.get("authors") or []
            last = (authors[-1] if authors else "").strip().lower()
            if last not in {"et al.", "et al", "others", "and others"}:
                issues["R3_missing_et_al"].append((cid, authors))

    return {
        "total": len(entries),
        "masked_all_three": n_masked_all,
        "issues": dict(issues),
    }


def audit_p1(entries: list[dict], meta: dict) -> dict:
    """P1: author name variant (exactly one author mutated to a known variant form)."""
    issues = defaultdict(list)
    for e in entries:
        cid = e["citation_id"]
        m = meta.get(cid, {})
        seed = m.get("seed", {}) or {}
        # authors should differ from seed
        if e.get("authors") == seed.get("authors"):
            issues["P1_authors_equal_seed"].append(cid)
        # All non-author, non-identifier fields MUST equal seed (real bugs).
        # Identifier fields (url/doi/arxiv_id) are allowed to be masked to "".
        for f in ("title", "venue", "year", "pages", "volume",
                  "publisher", "location"):
            if e.get(f, "") != seed.get(f, ""):
                issues[f"P1_{f}_differs_from_seed"].append(cid)
                break
        for f in ("url", "doi", "arxiv_id"):
            cv = e.get(f, "") or ""
            sv = seed.get(f, "") or ""
            if cv and cv != sv:  # only flag if cite has a DIFFERENT non-empty value
                issues[f"P1_{f}_contradicts_seed"].append(cid)
                break
    return {"total": len(entries), "issues": dict(issues)}


def audit_p3(entries: list[dict], meta: dict) -> dict:
    """P3: ONLY the fields in meta.changed_fields should differ from seed."""
    issues = defaultdict(list)
    peripheral = ("pages", "volume", "publisher", "location")
    for e in entries:
        cid = e["citation_id"]
        m = meta.get(cid, {})
        seed = m.get("seed", {}) or {}
        changed = set(m.get("changed_fields") or [])
        # Fabricated fields should differ
        for f in changed:
            if str(e.get(f, "")).strip() == str(seed.get(f, "")).strip():
                issues["P3_fabricated_field_equal_seed"].append((cid, f))
                break
        # Non-peripheral fields should equal seed
        for f in ("title", "authors", "venue", "year", "doi", "arxiv_id"):
            if e.get(f, "") != seed.get(f, ""):
                issues[f"P3_{f}_differs_from_seed"].append(cid)
                break
        # Non-changed peripheral fields should equal seed
        for f in peripheral:
            if f not in changed and str(e.get(f, "")).strip() != str(seed.get(f, "")).strip():
                issues[f"P3_unchanged_{f}_differs"].append(cid)
                break
    return {"total": len(entries), "issues": dict(issues)}


def audit_h(name: str, entries: list[dict], meta: dict) -> dict:
    """H1-H6: changed_fields should differ from seed, unchanged fields should match."""
    issues = defaultdict(list)
    for e in entries:
        cid = e["citation_id"]
        m = meta.get(cid, {})
        seed = m.get("seed", {}) or {}
        changed = set(m.get("changed_fields") or [])
        if not changed:
            issues[f"{name}_no_changed_fields_in_meta"].append(cid)
            continue
        # At least one "changed field" should actually differ
        any_differs = False
        for f in changed:
            if e.get(f, "") != seed.get(f, ""):
                any_differs = True
                break
        if not any_differs:
            issues[f"{name}_no_field_actually_changed"].append((cid, sorted(changed)))
    return {"total": len(entries), "issues": dict(issues)}


def main() -> None:
    meta = load_json(DATA_DIR / "meta.json")
    harvest_idx = {
        (normalize_title(e.get("title","")), e.get("year")): e
        for e in load_json(HARVEST_PATH)
    }

    report: dict = {}

    for name, mut in (("R1", "none"), ("R2", "format_variant"), ("R3", "et_al_abbreviation")):
        entries = load_json(DATA_DIR / f"{name}.json")
        report[name] = audit_real(name, entries, meta, mut)

    for name, mut in (("R1_plus", "none"), ("R2_plus", "format_variant"), ("R3_plus", "et_al_abbreviation")):
        entries = load_json(DATA_DIR / f"{name}.json")
        report[name] = audit_real(name, entries, meta, mut, harvest_idx)

    report["P1"] = audit_p1(load_json(DATA_DIR / "P1.json"), meta)
    report["P3"] = audit_p3(load_json(DATA_DIR / "P3.json"), meta)
    for h in ("H1","H2","H3","H4","H5","H6"):
        report[h] = audit_h(h, load_json(DATA_DIR / f"{h}.json"), meta)

    # Print summary table
    print()
    print(f"{'Dataset':<10} {'Total':>6} {'Issues':>8}  Details")
    print("-" * 80)
    for name, r in report.items():
        n_iss = sum(len(v) for v in r.get("issues", {}).values())
        detail = ""
        if "masked_all_three" in r:
            detail = f"masked_all_three={r['masked_all_three']} ({100*r['masked_all_three']/r['total']:.1f}%)"
        print(f"{name:<10} {r['total']:>6} {n_iss:>8}  {detail}")

    # Print issues detail
    print("\n--- Issue details ---")
    for name, r in report.items():
        if r.get("issues"):
            print(f"\n{name}:")
            for k, v in r["issues"].items():
                print(f"  [{k}] ({len(v)} cases)")
                for x in v[:5]:
                    print(f"    {x}")
                if len(v) > 5:
                    print(f"    ... {len(v)-5} more")

    # Persist a JSON report
    out = Path("results/audit_report.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=list))
    print(f"\nFull report: {out}")


if __name__ == "__main__":
    main()
