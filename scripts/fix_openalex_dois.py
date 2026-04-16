"""Repair OpenAlex-sourced DOIs that were truncated by a prior harvest bug.

Bug: harvest_seed_citations.py used `doi.rsplit("/", 1)[-1]` to strip the
"https://doi.org/" prefix, which also cut off the DOI registrant prefix
(e.g. "10.1109/tpami.2021.3126648" became "tpami.2021.3126648").

Fix: re-query the OpenAlex API using each seed's `url` field
(https://openalex.org/Wxxxx) and overwrite `doi` with the correctly-parsed
value.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests


def _parse_openalex_doi(work: dict) -> str:
    raw = str(work.get("doi") or "")
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
            break
    return raw.lower()


def fix_seeds(
    input_path: Path,
    output_path: Path,
    sleep_s: float,
) -> tuple[int, int, int]:
    seeds = json.loads(input_path.read_text())
    total = len(seeds)
    broken_idx: list[int] = []
    for i, s in enumerate(seeds):
        if (s.get("_source") or "").lower() != "openalex":
            continue
        doi = (s.get("doi") or "").strip()
        if doi and not doi.startswith("10."):
            broken_idx.append(i)

    print(f"[fix] total seeds: {total}", flush=True)
    print(f"[fix] broken OpenAlex DOIs: {len(broken_idx)}", flush=True)

    fixed = 0
    failed = 0
    for j, idx in enumerate(broken_idx, 1):
        s = seeds[idx]
        url = (s.get("url") or "").strip()
        if not url.startswith("https://openalex.org/"):
            print(f"  [{j}/{len(broken_idx)}] skip: no openalex url", flush=True)
            failed += 1
            continue
        work_id = url.rsplit("/", 1)[-1]
        api_url = f"https://api.openalex.org/works/{work_id}"
        try:
            resp = requests.get(api_url, timeout=20)
            resp.raise_for_status()
            work = resp.json()
        except Exception as exc:
            print(f"  [{j}/{len(broken_idx)}] fetch fail {work_id}: {exc}", flush=True)
            failed += 1
            continue

        new_doi = _parse_openalex_doi(work)
        if new_doi and new_doi.startswith("10."):
            old = s.get("doi", "")
            s["doi"] = new_doi
            fixed += 1
            print(f"  [{j}/{len(broken_idx)}] {work_id}: {old!r} -> {new_doi!r}", flush=True)
        else:
            print(f"  [{j}/{len(broken_idx)}] {work_id}: no valid DOI in API response", flush=True)
            failed += 1

        if sleep_s > 0:
            time.sleep(sleep_s)

    output_path.write_text(json.dumps(seeds, indent=2, ensure_ascii=False))
    return total, fixed, failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/bib/harvested_seeds_clean.json",
        help="Seed file to repair (default: data/bib/harvested_seeds_clean.json)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: overwrite input)",
    )
    parser.add_argument("--sleep", type=float, default=0.1, help="Seconds between API calls")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path
    if not input_path.exists():
        print(f"[fix] input not found: {input_path}", file=sys.stderr)
        return 1

    total, fixed, failed = fix_seeds(input_path, output_path, args.sleep)
    print(
        f"[fix] done. total={total} fixed={fixed} failed={failed} "
        f"-> {output_path}",
        flush=True,
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
