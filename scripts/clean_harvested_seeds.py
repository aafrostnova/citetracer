"""Clean the raw harvested seed citations.

Filters:
- Drop records with no authors.
- Drop records with no venue.
- Deduplicate by normalized title (keep the richest variant per group).

Input:  data/bib/harvested_seeds.json
Output: data/bib/harvested_seeds_clean.json (default) + stats to stdout
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


_FIELD_KEYS = ("title", "authors", "venue", "year", "doi", "arxiv_id",
               "url", "volume", "pages", "publisher", "location")


def _richness_score(rec: dict[str, Any]) -> int:
    score = 0
    for k in _FIELD_KEYS:
        v = rec.get(k)
        if isinstance(v, list):
            if v:
                score += 1
        elif v not in (None, "", 0):
            score += 1
    return score


def _norm_title(t: str) -> str:
    t = (t or "").lower()
    t = re.sub(r"[^\w\s]", " ", t)      # strip punctuation
    t = re.sub(r"\s+", " ", t).strip()  # collapse whitespace
    return t[:120]                      # prefix to tolerate trailing subtitle drift


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/bib/harvested_seeds.json")
    parser.add_argument("--output", default="data/bib/harvested_seeds_clean.json")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    data: list[dict] = json.loads(in_path.read_text())
    initial = len(data)

    # Filter 1: drop no-authors
    step1 = [r for r in data if r.get("authors")]
    dropped_no_authors = initial - len(step1)

    # Filter 2: drop no-venue
    step2 = [r for r in step1 if (r.get("venue") or "").strip()]
    dropped_no_venue = len(step1) - len(step2)

    # Filter 3: dedupe by normalized title, keep richest
    by_title: dict[str, dict] = {}
    for r in step2:
        key = _norm_title(r.get("title", ""))
        if not key:
            continue
        existing = by_title.get(key)
        if existing is None or _richness_score(r) > _richness_score(existing):
            by_title[key] = r
    step3 = list(by_title.values())
    dropped_dup_title = len(step2) - len(step3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(step3, indent=2, ensure_ascii=False))

    # Report
    print(f"Input:                     {initial}")
    print(f"  dropped (no authors):    {dropped_no_authors}")
    print(f"  dropped (no venue):      {dropped_no_venue}")
    print(f"  dropped (duplicate title): {dropped_dup_title}")
    print(f"Output (clean):            {len(step3)}")
    print(f"Written to:                {out_path}")


if __name__ == "__main__":
    main()
