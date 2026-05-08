"""Extreme-relaxed verification mode (env-var gate only).

The actual verification is the standard cascading 3-agent flow in
`CitationVerifier.verify_citation`; extreme mode only changes the input
shape: every reference field except `title` and `authors` is masked to
empty before the cascade runs, so downstream agents only judge title and
author mismatches (H1 / H2 + their P-counterparts). H3 / H4 / H5 / H6
naturally cannot fire because their fields are `reference_missing`.

Activate via:
  CITATION_CHECKER_EXTREME_RELAXED=1

The web-search query builder (packages/connectors/google_search.py) reads
the same env var to narrow its query to title + authors only, matching
the verifier's narrowed scope.
"""
from __future__ import annotations

import os


def is_enabled() -> bool:
    return os.getenv("CITATION_CHECKER_EXTREME_RELAXED", "0").strip() in {
        "1", "true", "yes", "on",
    }


_SMALL_DIFF_CHAR_TOLERANCE = 2


def _norm(s) -> str:
    return " ".join(str(s or "").lower().split())


def _is_et_al(name) -> bool:
    n = _norm(name).replace(".", "")
    return n in {"et al", "et al.", "et. al", "et. al."} or n.endswith(" et al")


def filter_small_char_diff(verdict):
    """Promote a non-VALID verdict to VALID/R3 when the only differences
    between citation and matched_candidate (on title + paired authors) are
    within `_SMALL_DIFF_CHAR_TOLERANCE` characters per field.

    Why: in extreme mode the cascade only judges title + authors. OCR
    artifacts and 1 / 2-character typos (missing diacritic, dropped letter,
    transposed pair) regularly cause spurious H1 / H2 / P1 even when the
    citation is semantically the same paper as the candidate. Treat sub
    3-char drifts as cosmetic and pass them.

    Pairing rule for authors: positional. If citation contains an "et al"
    sentinel, only the explicitly listed prefix is compared (the citation
    deliberately did not enumerate the rest, so length mismatch is
    expected).
    """
    from .models import VerdictLabel
    from .adjudicate import _edit_distance

    if verdict.verdict == VerdictLabel.VALID:
        return verdict
    cite = verdict.reference_snapshot or {}
    cand = verdict.matched_candidate or {}
    if not cand:
        return verdict

    title_diff = _edit_distance(
        _norm(cite.get("title", "")),
        _norm(cand.get("title", "")),
        cap=_SMALL_DIFF_CHAR_TOLERANCE + 1,
    )
    if title_diff > _SMALL_DIFF_CHAR_TOLERANCE:
        return verdict

    cite_authors = [a for a in (cite.get("authors") or []) if not _is_et_al(a)]
    cand_authors = list(cand.get("authors") or [])
    n = min(len(cite_authors), len(cand_authors))
    max_author_diff = 0
    for i in range(n):
        d = _edit_distance(
            _norm(cite_authors[i]),
            _norm(cand_authors[i]),
            cap=_SMALL_DIFF_CHAR_TOLERANCE + 1,
        )
        if d > max_author_diff:
            max_author_diff = d
    if max_author_diff > _SMALL_DIFF_CHAR_TOLERANCE:
        return verdict
    # If citation enumerated MORE authors than candidate offers (and no et al
    # sentinel), candidate is missing real authors -> not a small-diff case.
    if len(cite_authors) > len(cand_authors):
        return verdict

    orig_label = getattr(verdict.verdict, "value", str(verdict.verdict))
    orig_tax = list(verdict.taxonomy_subtype or [])
    verdict.verdict = VerdictLabel.VALID
    verdict.taxonomy_subtype = ["R3"]
    verdict.adjudication_reason = (
        f"extreme small-diff override: title_diff={title_diff} char(s), "
        f"max_author_diff={max_author_diff} char(s) "
        f"(both <= {_SMALL_DIFF_CHAR_TOLERANCE}). Promoted from "
        f"{orig_label}/{orig_tax} to VALID/R3. Original reason: "
        f"{(verdict.adjudication_reason or '')[:200]}"
    )
    return verdict


def apply_filters(verdict):
    """Run all extreme-mode post-cascade filters in the right order:
    small-diff promotion first (may flip to VALID), then taxonomy stripping
    (only meaningful for FAKE / POTENTIAL paths)."""
    verdict = filter_small_char_diff(verdict)
    verdict = filter_taxonomy(verdict)
    return verdict


def filter_taxonomy(verdict):
    """Post-process a verdict so its FAKE taxonomy only contains H1 / H2.

    The cascading agents see candidates with venue / year / DOI populated and
    occasionally emit H3 / H4 / H5 / H6 even though the citation fields were
    masked to empty (so field_status reports `reference_missing`). The LLM
    sometimes ignores that signal and fires those tags off the candidate
    side. In extreme mode those errors are out of scope; strip them.

    If a FAKE verdict ends up with no H tags after filtering (e.g. the only
    complaint was H4), demote to POTENTIAL/P1 — the title + author fields we
    actually care about were never flagged.
    """
    from .models import VerdictLabel
    keep_h = {"H1", "H2"}
    tax = list(verdict.taxonomy_subtype or [])
    filtered = [t for t in tax if t in keep_h or not t.startswith("H")]
    if filtered == tax:
        return verdict

    verdict.taxonomy_subtype = filtered
    if verdict.verdict != VerdictLabel.FAKE_REFERENCE:
        return verdict

    has_h = any(t in keep_h for t in filtered)
    if not has_h:
        # FAKE was driven entirely by stripped tags (H3/H4/H5/H6). Title and
        # authors were not flagged → demote to POTENTIAL/P1.
        verdict.verdict = VerdictLabel.POTENTIAL_REFERENCE
        verdict.taxonomy_subtype = ["P1"]
        verdict.adjudication_reason = (
            "extreme: original FAKE verdict relied only on H3/H4/H5/H6 "
            "(out-of-scope fields). Title and authors were not flagged → "
            f"demoted to POTENTIAL/P1. Original reason: "
            f"{(verdict.adjudication_reason or '')[:200]}"
        )
    else:
        verdict.adjudication_reason = (
            f"[extreme: stripped H3/H4/H5/H6 from taxonomy; out of scope] "
            f"{verdict.adjudication_reason or ''}"
        )
    return verdict
