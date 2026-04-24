"""Exhaustively generate PDFs from all R/P/H citations, split evenly across
the 4 conference styles. After compiling, inspect each PDF's main.log for
obvious issues (unicode missing, overfull hbox) and delete offending entries
from the dataset. Optionally regenerate the affected PDFs.

Usage:
    python scripts/gen_exhaustive_pdfs.py
    python scripts/gen_exhaustive_pdfs.py --dry-run         # don't delete
    python scripts/gen_exhaustive_pdfs.py --no-regenerate   # don't regen after delete
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.generate_pdf_samples import (
    STYLES, build_main_tex, distribute_cites, generate_bib_file, _coarse_label,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data/synthetic_data/v2"
BST_DIR = PROJECT_ROOT / "data/bst"

DATASET_FILES = [
    "R1.json", "R2.json", "R3.json",
    "R1_plus.json", "R2_plus.json", "R3_plus.json",
    "P1.json", "P3.json",
    "H1.json", "H2.json", "H3.json", "H4.json", "H5.json", "H6.json",
]


def load_all_citations() -> list[dict]:
    out = []
    for fn in DATASET_FILES:
        p = DATA_DIR / fn
        if p.exists():
            out.extend(json.loads(p.read_text()))
    return out


def compile_paper(paper_dir: Path) -> tuple[bool, str]:
    """Compile a paper directory. Returns (success, main.log contents)."""
    result = subprocess.run(
        ["bash", "build.sh"],
        cwd=str(paper_dir),
        capture_output=True,
        text=True,
        timeout=120,
    )
    log_path = paper_dir / "main.log"
    log = log_path.read_text(errors="replace") if log_path.exists() else ""
    pdf_ok = (paper_dir / "main.pdf").exists()
    return pdf_ok, log


# ---------------------------------------------------------------------------
# Log inspection — map problems back to BibTeX keys
# ---------------------------------------------------------------------------

MISSING_CHAR_RE = re.compile(r"Missing character: There is no (?P<char>\S) .*! ", re.MULTILINE)
OVERFULL_RE = re.compile(r"Overfull \\hbox \((?P<badness>\d+\.?\d*)pt too wide\)")


def _scan_bbl_for_char_key(bbl_text: str, ch: str) -> set[str]:
    """Find bibentry keys in main.bbl whose content contains char `ch`."""
    # bbl entries look like "\bibitem{KEY}\n<entry body>\n\n\bibitem{KEY2}..."
    # Split on \bibitem
    keys_with_char: set[str] = set()
    pattern = re.compile(r"\\bibitem(?:\[.*?\])?\{(?P<key>[^}]+)\}(?P<body>.*?)(?=\\bibitem|$)", re.DOTALL)
    for m in pattern.finditer(bbl_text):
        if ch in m.group("body"):
            keys_with_char.add(m.group("key"))
    return keys_with_char


def _scan_bib_for_long_urls(bib_text: str, max_len: int = 100) -> set[str]:
    """Find bib keys whose `url` or `doi` field exceeds max_len (likely
    overfull hbox culprit)."""
    bad: set[str] = set()
    entry_re = re.compile(r"@\w+\{(?P<key>[^,]+),(?P<body>.*?)\n\}", re.DOTALL)
    field_re = re.compile(r"^\s*(?:url|doi|eprint)\s*=\s*\{(?P<val>.*?)\}\s*,?\s*$", re.MULTILINE)
    for m in entry_re.finditer(bib_text):
        for fm in field_re.finditer(m.group("body")):
            if len(fm.group("val")) > max_len:
                bad.add(m.group("key"))
                break
    return bad


def inspect_paper(paper_dir: Path) -> dict:
    """Return a dict of issues found + candidate bib keys to remove."""
    log = (paper_dir / "main.log").read_text(errors="replace") if (paper_dir / "main.log").exists() else ""
    bbl = (paper_dir / "main.bbl").read_text(errors="replace") if (paper_dir / "main.bbl").exists() else ""
    bib = (paper_dir / "refs.bib").read_text(errors="replace") if (paper_dir / "refs.bib").exists() else ""

    # 1. Missing characters (unicode not in font)
    missing_chars = set(m.group("char") for m in MISSING_CHAR_RE.finditer(log))
    char_problem_keys: set[str] = set()
    for ch in missing_chars:
        char_problem_keys.update(_scan_bbl_for_char_key(bbl, ch))

    # 2. Overfull hbox (line too wide) — only severe ones, > 15pt too wide
    overfull_bad_pts = [float(m.group("badness")) for m in OVERFULL_RE.finditer(log)]
    overfull_severe = [pt for pt in overfull_bad_pts if pt > 15.0]
    overfull_keys: set[str] = set()
    if overfull_severe:
        overfull_keys.update(_scan_bib_for_long_urls(bib, max_len=100))

    return {
        "paper_dir": str(paper_dir.relative_to(PROJECT_ROOT)),
        "missing_chars": sorted(missing_chars),
        "char_problem_keys": sorted(char_problem_keys),
        "overfull_count": len(overfull_bad_pts),
        "overfull_severe": len(overfull_severe),
        "overfull_keys": sorted(overfull_keys),
        "pdf_exists": (paper_dir / "main.pdf").exists(),
    }


# ---------------------------------------------------------------------------
# Exhaustive distribution
# ---------------------------------------------------------------------------

def distribute_exhaustive(
    citations: list[dict],
    styles: list[str],
    citations_per_paper: int = 50,
    seed: int = 42,
) -> dict[str, list[list[dict]]]:
    """Shuffle all citations once, split across styles evenly, then group
    each style's slice into papers of `citations_per_paper` each.

    Returns {style: [paper1_citations, paper2_citations, ...]}
    """
    rng = random.Random(seed)
    shuffled = citations.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_styles = len(styles)
    slices: dict[str, list[dict]] = {}
    for i, s in enumerate(styles):
        lo = (i * n) // n_styles
        hi = ((i + 1) * n) // n_styles
        slices[s] = shuffled[lo:hi]

    out: dict[str, list[list[dict]]] = {}
    for s, slice_cits in slices.items():
        papers: list[list[dict]] = []
        for i in range(0, len(slice_cits), citations_per_paper):
            papers.append(slice_cits[i: i + citations_per_paper])
        out[s] = papers
    return out


# ---------------------------------------------------------------------------
# Build one paper
# ---------------------------------------------------------------------------

def build_paper_from_citations(
    style_name: str, paper_id: int, out_root: Path, citations: list[dict],
) -> Path:
    style = STYLES[style_name]
    out_dir = out_root / style_name / f"paper_{paper_id:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "refs.bib").write_text(generate_bib_file(citations))
    body = distribute_cites(citations, style["cite_cmd"])
    (out_dir / "main.tex").write_text(build_main_tex(style, body))
    shutil.copy(BST_DIR / style["bst_file"], out_dir / style["bst_file"])
    (out_dir / "build.sh").write_text("""#!/bin/bash
set -e
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || true
bibtex main > /dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
""")
    (out_dir / "build.sh").chmod(0o755)

    manifest = {
        "style": style_name,
        "paper_id": paper_id,
        "n_citations": len(citations),
        "citations": [
            {
                "citation_id": c["citation_id"],
                "coarse_label": _coarse_label(c["citation_id"]),
                "title": (c.get("title") or "")[:120],
            } for c in citations
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return out_dir


# ---------------------------------------------------------------------------
# Deletion from dataset files
# ---------------------------------------------------------------------------

def delete_from_dataset(bad_ids: set[str]) -> dict[str, int]:
    """Remove entries whose citation_id is in bad_ids from all dataset files
    and meta.json and _all_test.json. Returns per-file delete counts."""
    removed = {}
    for fn in DATASET_FILES:
        p = DATA_DIR / fn
        if not p.exists():
            continue
        data = json.loads(p.read_text())
        before = len(data)
        data = [c for c in data if c.get("citation_id") not in bad_ids]
        if len(data) != before:
            p.write_text(json.dumps(data, ensure_ascii=False, indent=2))
            removed[fn] = before - len(data)

    meta_p = DATA_DIR / "meta.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
        before = len(meta)
        for cid in bad_ids:
            meta.pop(cid, None)
        if len(meta) != before:
            meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
            removed["meta.json"] = before - len(meta)

    all_p = DATA_DIR / "_all_test.json"
    if all_p.exists():
        all_t = json.loads(all_p.read_text())
        before = len(all_t)
        all_t = [c for c in all_t if c.get("citation_id") not in bad_ids]
        if len(all_t) != before:
            all_p.write_text(json.dumps(all_t, ensure_ascii=False, indent=2))
            removed["_all_test.json"] = before - len(all_t)
    return removed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-root", default=str(PROJECT_ROOT / "pdf_samples_full"))
    parser.add_argument("--citations-per-paper", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Don't delete or regenerate")
    parser.add_argument("--no-regenerate", action="store_true", help="Don't regenerate affected PDFs after delete")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    citations = load_all_citations()
    styles = list(STYLES.keys())
    print(f"Loaded {len(citations)} citations across {len(DATASET_FILES)} files")
    print(f"Splitting across {len(styles)} styles: {styles}")

    distribution = distribute_exhaustive(citations, styles, args.citations_per_paper, args.seed)
    for s, papers in distribution.items():
        print(f"  {s}: {len(papers)} papers covering {sum(len(p) for p in papers)} citations")

    # Build + compile all papers
    all_reports: list[dict] = []
    for s, papers in distribution.items():
        print(f"\n=== Style: {s} ({len(papers)} papers) ===")
        for i, pap in enumerate(papers, start=1):
            d = build_paper_from_citations(s, i, out_root, pap)
            ok, _log = compile_paper(d)
            rep = inspect_paper(d)
            status = "ok" if ok and not rep["char_problem_keys"] and not rep["overfull_keys"] else "issues"
            print(f"  paper_{i:03d} ({len(pap)} cites) [{status}]  "
                  f"missing_chars={rep['missing_chars']}  "
                  f"overfull_severe={rep['overfull_severe']}  "
                  f"problem_keys={len(rep['char_problem_keys']) + len(rep['overfull_keys'])}")
            all_reports.append(rep)

    # Collect all problem keys
    problem_keys: set[str] = set()
    by_paper: dict[str, set[str]] = {}
    for rep in all_reports:
        keys = set(rep["char_problem_keys"]) | set(rep["overfull_keys"])
        if keys:
            problem_keys.update(keys)
            by_paper[rep["paper_dir"]] = keys

    print(f"\n=== Inspection summary ===")
    print(f"Papers with issues: {len(by_paper)}/{len(all_reports)}")
    print(f"Unique problem bib keys: {len(problem_keys)}")
    if problem_keys:
        # Group by label
        by_label = Counter(_coarse_label(k) for k in problem_keys)
        by_subset = Counter(k[:2] for k in problem_keys)
        print(f"  by coarse label: {dict(by_label)}")
        print(f"  by subset: {dict(by_subset)}")

    if not problem_keys:
        print("\nNo issues detected.")
        return

    if args.dry_run:
        print("\n[dry-run] Would delete these keys. Not modifying files.")
        print(sorted(problem_keys))
        return

    # Delete
    print("\n=== Deleting problem entries ===")
    removed = delete_from_dataset(problem_keys)
    for fn, n in removed.items():
        print(f"  {fn}: -{n}")

    if args.no_regenerate:
        return

    # Regenerate affected PDFs (cleanly — re-run the whole pipeline against the
    # reduced dataset would re-shuffle. Simpler: for each affected paper, drop
    # the bad keys from its citation list and recompile.)
    print("\n=== Regenerating affected PDFs ===")
    n_regen = 0
    for rep in all_reports:
        paper_dir = PROJECT_ROOT / rep["paper_dir"]
        bad = (set(rep["char_problem_keys"]) | set(rep["overfull_keys"])) & problem_keys
        if not bad:
            continue
        manifest = json.loads((paper_dir / "manifest.json").read_text())
        kept_ids = [c["citation_id"] for c in manifest["citations"] if c["citation_id"] not in bad]
        kept_cits = [c for c in citations if c["citation_id"] in kept_ids]
        if len(kept_cits) != len(kept_ids):
            print(f"  WARN: {paper_dir.name}: some kept ids missing from dataset")
        paper_id = int(paper_dir.name.split("_")[-1])
        build_paper_from_citations(paper_dir.parent.name, paper_id, out_root, kept_cits)
        ok, _ = compile_paper(paper_dir)
        print(f"  {rep['paper_dir']} regenerated with {len(kept_cits)} citations [{'ok' if ok else 'FAIL'}]")
        n_regen += 1
    print(f"\nRegenerated {n_regen} papers.")

    # Final summary
    print(f"\n=== FINAL ===")
    print(f"Total PDFs generated: {len(all_reports)}")
    print(f"Bib keys deleted from dataset: {len(problem_keys)}")
    print(f"Per-subset deletions:")
    for subset, n in sorted(Counter(k[:2] for k in problem_keys).items()):
        print(f"  {subset}: {n}")


if __name__ == "__main__":
    main()
