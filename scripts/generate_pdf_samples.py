"""Generate LaTeX+BibTeX PDFs from the synthetic citation dataset.

Each generated document contains ~50 mixed-label citations (Real + Potential
+ Hallucinated), embedded in a plausible paper body introducing the dataset.

Per PDF output layout:
    pdf_samples/<style>/paper_<NNN>/
        refs.bib           BibTeX entries (key = citation_id)
        main.tex           LaTeX source
        <style>.bst        Conference bib style
        build.sh           pdflatex + bibtex + pdflatex x2
        manifest.json      citation_id → expected_label

Usage:
    python scripts/generate_pdf_samples.py \
        --style acm \
        --num-papers 5 \
        --out-root pdf_samples

Styles available: acm, splncs04, iclr, ieee
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data/synthetic_data/v2"
BST_DIR = PROJECT_ROOT / "data/bst"

# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------

STYLES = {
    "acm": {
        "bst_file": "ACM-Reference-Format.bst",
        "bst_name": "ACM-Reference-Format",
        "docclass": r"\documentclass[twocolumn,10pt]{article}",
        "preamble": r"""\usepackage{cmap}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{times}
\usepackage[numbers,sort&compress]{natbib}
\usepackage{hyperref}
\usepackage{url}
% Allow hyphenation inside \url{...} so long DOIs don't overflow columns.
\usepackage{xurl}
\usepackage[margin=0.75in]{geometry}
\setlength{\columnsep}{20pt}
\sloppy""",
        "cite_cmd": r"\citep",
    },
    "splncs04": {
        "bst_file": "splncs04.bst",
        "bst_name": "splncs04",
        "docclass": r"\documentclass{article}",
        "preamble": r"""\usepackage{cmap}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{times}
\usepackage[numbers,sort&compress]{natbib}
\usepackage{hyperref}
\usepackage{url}
\usepackage{xurl}
\usepackage[margin=1in]{geometry}
\sloppy""",
        "cite_cmd": r"\citep",
    },
    "iclr": {
        "bst_file": "iclr.bst",
        "bst_name": "iclr",
        "docclass": r"\documentclass{article}",
        "preamble": r"""\usepackage{cmap}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{times}
\usepackage{natbib}
\usepackage{hyperref}
\usepackage{url}
\usepackage{xurl}
\usepackage[margin=1in]{geometry}
\sloppy""",
        "cite_cmd": r"\citep",
    },
    "ieee": {
        "bst_file": "ieeenat_fullname.bst",
        "bst_name": "ieeenat_fullname",
        "docclass": r"\documentclass[twocolumn,10pt]{article}",
        "preamble": r"""\usepackage{cmap}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{times}
\usepackage{natbib}
\usepackage{hyperref}
\usepackage{url}
\usepackage{xurl}
\usepackage[margin=0.75in]{geometry}
\setlength{\columnsep}{20pt}
\sloppy""",
        "cite_cmd": r"\citep",
    },
    "plain": {
        "bst_file": "plainnat.bst",
        "bst_name": "plainnat",
        "docclass": r"\documentclass{article}",
        "preamble": r"""\usepackage{cmap}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{times}
\usepackage{natbib}
\usepackage{hyperref}
\usepackage{url}
\usepackage{xurl}
\usepackage[margin=1in]{geometry}
\sloppy""",
        "cite_cmd": r"\citep",
    },
}

# ---------------------------------------------------------------------------
# Data loading & sampling
# ---------------------------------------------------------------------------

# Files to pull from, grouped by coarse label.
REAL_FILES = ["R1.json", "R2.json", "R3.json"]
POTENTIAL_FILES = ["P1.json", "P3.json"]
HALLUCINATED_FILES = ["H1.json", "H2.json", "H3.json", "H4.json", "H5.json", "H6.json"]


def _load_pool(file_list: list[str]) -> list[dict]:
    pool = []
    for fn in file_list:
        p = DATA_DIR / fn
        if p.exists():
            pool.extend(json.loads(p.read_text()))
    return pool


def _coarse_label(cid: str) -> str:
    if cid.startswith("R"):
        return "REAL"
    if cid.startswith("P"):
        return "POTENTIAL"
    if cid.startswith("H"):
        return "HALLUCINATED"
    return "?"


def sample_citations(
    n: int = 50,
    ratio: dict[str, float] | None = None,
    seed: int | None = None,
) -> list[dict]:
    """Sample `n` citations with target ratio of REAL / POTENTIAL / HALLUCINATED."""
    ratio = ratio or {"REAL": 0.70, "POTENTIAL": 0.10, "HALLUCINATED": 0.20}
    rng = random.Random(seed)

    n_real = int(round(n * ratio["REAL"]))
    n_pot = int(round(n * ratio["POTENTIAL"]))
    n_hal = n - n_real - n_pot

    real_pool = _load_pool(REAL_FILES)
    pot_pool = _load_pool(POTENTIAL_FILES)
    hal_pool = _load_pool(HALLUCINATED_FILES)

    sampled = (
        rng.sample(real_pool, min(n_real, len(real_pool)))
        + rng.sample(pot_pool, min(n_pot, len(pot_pool)))
        + rng.sample(hal_pool, min(n_hal, len(hal_pool)))
    )
    rng.shuffle(sampled)
    return sampled


# ---------------------------------------------------------------------------
# BibTeX generation
# ---------------------------------------------------------------------------

def _escape_bib(value: str) -> str:
    """Escape characters that BibTeX gets unhappy about."""
    if not value:
        return ""
    s = str(value)
    # Braces inside the value must be balanced; simplest: escape with backslash
    s = s.replace("\\", r"\\")
    # For LaTeX special chars inside {}, &, %, $, # need escaping
    for ch in ("&", "%", "$", "#", "_"):
        s = s.replace(ch, f"\\{ch}")
    return s


def _guess_entry_type(citation: dict) -> str:
    venue = (citation.get("venue") or "").lower()
    pages = citation.get("pages") or ""
    arxiv = citation.get("arxiv_id") or ""
    if "arxiv" in venue or "corr" in venue:
        return "@misc"
    if arxiv and not pages and not venue:
        return "@misc"
    if "journal" in venue or "transactions" in venue or "magazine" in venue:
        return "@article"
    if venue or "proceedings" in venue or "conference" in venue:
        return "@inproceedings"
    return "@misc"


def to_bibtex_entry(citation: dict) -> str:
    cid = citation["citation_id"]
    entry_type = _guess_entry_type(citation)

    fields: list[str] = []

    def add(key: str, val):
        if val is None or val == "" or val == []:
            return
        escaped = _escape_bib(val) if not isinstance(val, (int, float)) else str(val)
        fields.append(f"  {key} = {{{escaped}}}")

    add("title", citation.get("title"))
    authors = citation.get("authors") or []
    if authors:
        # BibTeX uses literal "and others" as the "et al." marker. Normalize
        # common truncation markers in the citation author list to that form
        # so we don't emit "X and Jiayi Lin and and others" (double "and").
        _TRUNC = {"others", "and others", "et al", "et al.", "etc", "etc.", "..."}
        cleaned = []
        has_trunc = False
        for a in authors:
            s = (a or "").strip()
            if not s:
                continue
            if s.lower() in _TRUNC:
                has_trunc = True
                continue
            cleaned.append(s)
        author_str = " and ".join(cleaned)
        if has_trunc:
            author_str = f"{author_str} and others" if author_str else "others"
        add("author", author_str)

    venue = citation.get("venue") or ""
    if venue:
        if entry_type == "@inproceedings":
            add("booktitle", venue)
        elif entry_type == "@article":
            add("journal", venue)
        else:
            add("howpublished", venue)

    add("year", citation.get("year"))
    add("volume", citation.get("volume"))
    add("pages", citation.get("pages"))
    add("publisher", citation.get("publisher"))
    add("address", citation.get("location"))
    add("doi", citation.get("doi"))

    arxiv = citation.get("arxiv_id") or ""
    if arxiv:
        add("eprint", arxiv)
        add("archivePrefix", "arXiv")

    add("url", citation.get("url"))

    return f"{entry_type}{{{cid},\n" + ",\n".join(fields) + "\n}"


def generate_bib_file(citations: list[dict]) -> str:
    return "\n\n".join(to_bibtex_entry(c) for c in citations) + "\n"


# ---------------------------------------------------------------------------
# LaTeX body with distributed \cite positions
# ---------------------------------------------------------------------------

BODY_SECTIONS = [
    ("Introduction", [
        "Large language models have been widely adopted for academic writing assistance, "
        "but they frequently generate references that do not correspond to any real paper. "
        "Prior studies documenting this phenomenon {CITES}.",
        "A related line of work evaluates the downstream impact of fabricated citations on "
        "reader trust and scientific reproducibility {CITES}.",
        "Our work sits at the intersection of citation verification and controlled synthetic "
        "data generation. We complement recent benchmarks for retrieval-augmented factuality "
        "evaluation {CITES}.",
    ]),
    ("Related Work", [
        "Several systems verify citations by matching against structured bibliographic "
        "databases such as DBLP, CrossRef, and Semantic Scholar {CITES}.",
        "Other pipelines rely on web search or retrieval-augmented generation to reconstruct "
        "the intended reference {CITES}.",
        "Hallucination detection in general text generation is discussed in {CITES}, and "
        "fine-grained taxonomies of citation errors appear in {CITES}.",
    ]),
    ("Dataset Construction", [
        "Our benchmark is built by seeding from authentic paper metadata and applying "
        "controlled mutations to produce hallucinated variants. Seed papers are gathered "
        "from a diverse set of venues {CITES}.",
        "We distinguish three coarse categories — Real, Potential, and Hallucinated — each "
        "with fine-grained subtypes, broadly following conventions introduced by prior "
        "citation-error taxonomies {CITES}.",
        "To evaluate pipelines in realistic conditions, we also include references where the "
        "year or venue has drifted between preprint and conference versions {CITES}.",
    ]),
    ("Evaluation Protocol", [
        "We benchmark several representative verification pipelines, including rule-based "
        "field matchers, retrieval-first systems, and end-to-end LLM verifiers {CITES}.",
        "Baselines include string-matching approaches similar to {CITES} and more recent "
        "cascading-agent designs {CITES}.",
        "Metrics follow the conventions established in prior citation-verification studies "
        "{CITES}.",
    ]),
    ("Discussion", [
        "Understanding where each pipeline breaks down is crucial for deployment; our "
        "taxonomy aids in attributing failures to specific field-level errors {CITES}.",
        "Finally, we argue that dataset-level hallucination studies are complementary to "
        "model-level probes of factuality {CITES}.",
    ]),
]


def distribute_cites(citations: list[dict], cite_cmd: str) -> str:
    """Return LaTeX body text with `\cite{...}` commands distributing all
    citation keys across the paragraphs. Extra keys are appended to the last
    paragraph so none are dropped (LaTeX warns on uncited keys otherwise).
    """
    keys = [c["citation_id"] for c in citations]
    total = len(keys)

    # Count {CITES} placeholders
    slots: list[tuple[int, int]] = []  # (section_idx, para_idx)
    for si, (_sec, paras) in enumerate(BODY_SECTIONS):
        for pi, para in enumerate(paras):
            for _ in range(para.count("{CITES}")):
                slots.append((si, pi, paras[pi].count("{CITES}")))
    # Rebuild as flat (si, pi) list, order-preserving
    slot_positions: list[tuple[int, int]] = []
    for si, (_sec, paras) in enumerate(BODY_SECTIONS):
        for pi, para in enumerate(paras):
            for _ in range(para.count("{CITES}")):
                slot_positions.append((si, pi))

    n_slots = len(slot_positions)
    if n_slots == 0:
        raise RuntimeError("BODY_SECTIONS has no {CITES} placeholders.")

    # Assign keys round-robin to slots (balanced).
    # A paragraph with N {CITES} placeholders appears N times in slot_positions
    # with the same (si, pi) tuple. Store each slot's batch as its own list so
    # later slots don't overwrite earlier ones for the same paragraph.
    chunks: dict[tuple[int, int], list[list[str]]] = {}
    cursor = 0
    base = total // n_slots
    extra = total % n_slots
    for i, sp in enumerate(slot_positions):
        take = base + (1 if i < extra else 0)
        chunks.setdefault(sp, []).append(keys[cursor: cursor + take])
        cursor += take
    # Safety: shouldn't happen
    if cursor < total:
        chunks[slot_positions[-1]][-1].extend(keys[cursor:])

    # Build body text. Each (si, pi) has a list of batches — one per {CITES}
    # occurrence in source order. Pop from front so occurrences align.
    out_parts: list[str] = []
    for si, (section_title, paras) in enumerate(BODY_SECTIONS):
        out_parts.append(f"\\section{{{section_title}}}")
        for pi, para_tmpl in enumerate(paras):
            para = para_tmpl
            n_occ = para.count("{CITES}")
            batches = chunks.get((si, pi), [])
            for occ_idx in range(n_occ):
                this_keys = batches[occ_idx] if occ_idx < len(batches) else []
                cite_block = f"{cite_cmd}{{{', '.join(this_keys)}}}" if this_keys else ""
                para = para.replace("{CITES}", cite_block, 1)
            out_parts.append(para)
        out_parts.append("")  # blank line between sections

    return "\n\n".join(out_parts)


# ---------------------------------------------------------------------------
# Document template
# ---------------------------------------------------------------------------

TITLE = "A Synthetic Benchmark for Citation Hallucination Detection"
AUTHORS_BLOCK = r"""
\author{Anonymous Authors}
\date{}
"""

ABSTRACT = (
    "We introduce a controlled synthetic benchmark for citation hallucination "
    "detection in academic writing. The dataset contains mixed real, potentially "
    "hallucinated, and fully hallucinated citations spanning multiple venues, "
    "with fine-grained subtype annotations. We describe the construction pipeline, "
    "outline evaluation protocols, and report baseline results from several "
    "representative verification systems."
)


def build_main_tex(style: dict, body: str) -> str:
    return textwrap.dedent(rf"""
    {style['docclass']}
    {style['preamble']}

    \title{{{TITLE}}}
    {AUTHORS_BLOCK.strip()}

    \begin{{document}}
    \maketitle

    \begin{{abstract}}
    {ABSTRACT}
    \end{{abstract}}

    {body}

    \nocite{{*}}
    \bibliographystyle{{{style['bst_name']}}}
    \bibliography{{refs}}
    \end{{document}}
    """).strip() + "\n"


BUILD_SH = """#!/bin/bash
set -e
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode main.tex || true
bibtex main || true
pdflatex -interaction=nonstopmode main.tex || true
pdflatex -interaction=nonstopmode main.tex
echo "Built main.pdf"
"""


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_paper(
    style_name: str,
    paper_id: int,
    out_root: Path,
    n_citations: int = 50,
    seed: int | None = None,
    ratio: dict[str, float] | None = None,
) -> Path:
    if style_name not in STYLES:
        raise ValueError(f"unknown style {style_name!r}; choose from {sorted(STYLES)}")
    style = STYLES[style_name]

    out_dir = out_root / style_name / f"paper_{paper_id:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    citations = sample_citations(n=n_citations, ratio=ratio, seed=seed)

    # refs.bib
    (out_dir / "refs.bib").write_text(generate_bib_file(citations))

    # Body + main.tex
    body = distribute_cites(citations, style["cite_cmd"])
    (out_dir / "main.tex").write_text(build_main_tex(style, body))

    # Copy bst
    shutil.copy(BST_DIR / style["bst_file"], out_dir / style["bst_file"])

    # build.sh
    (out_dir / "build.sh").write_text(BUILD_SH)
    (out_dir / "build.sh").chmod(0o755)

    # manifest.json
    manifest = {
        "style": style_name,
        "paper_id": paper_id,
        "n_citations": len(citations),
        "seed": seed,
        "ratio": ratio or {"REAL": 0.70, "POTENTIAL": 0.10, "HALLUCINATED": 0.20},
        "citations": [
            {
                "citation_id": c["citation_id"],
                "coarse_label": _coarse_label(c["citation_id"]),
                "title": (c.get("title") or "")[:120],
            }
            for c in citations
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    print(f"  built {out_dir.relative_to(out_root)}  ({len(citations)} citations)")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--style",
        choices=list(STYLES.keys()) + ["all"],
        default="all",
        help="Conference bib style (or 'all' for all styles).",
    )
    parser.add_argument("--num-papers", type=int, default=1, help="Papers per style.")
    parser.add_argument("--n-citations", type=int, default=50, help="Citations per paper.")
    parser.add_argument(
        "--out-root",
        default=str(PROJECT_ROOT / "pdf_samples"),
        help="Output directory root.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Base random seed (per-paper seed = seed + paper_id).",
    )
    parser.add_argument(
        "--ratio-real", type=float, default=0.70, help="Fraction of REAL citations.",
    )
    parser.add_argument(
        "--ratio-potential", type=float, default=0.10, help="Fraction of POTENTIAL.",
    )
    parser.add_argument(
        "--ratio-hallucinated", type=float, default=0.20, help="Fraction of HALLUCINATED.",
    )
    args = parser.parse_args()

    styles = [args.style] if args.style != "all" else list(STYLES.keys())
    ratio = {
        "REAL": args.ratio_real,
        "POTENTIAL": args.ratio_potential,
        "HALLUCINATED": args.ratio_hallucinated,
    }

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for style in styles:
        print(f"Style: {style}")
        for pid in range(1, args.num_papers + 1):
            seed = None if args.seed is None else args.seed + pid
            build_paper(
                style_name=style,
                paper_id=pid,
                out_root=out_root,
                n_citations=args.n_citations,
                seed=seed,
                ratio=ratio,
            )

    print(f"\nDone. Output in {out_root}")
    print("To build PDFs:")
    print(f"  for d in {out_root}/*/*/; do (cd $d && bash build.sh); done")


if __name__ == "__main__":
    main()
