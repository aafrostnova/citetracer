"""Deterministic, no-LLM alternatives to the Bedrock agents.

RuleBasedValidAgent consumes the authoritative signals already produced by
the rule comparator (title/year/doi/arxiv_id/pages/volume/publisher/location
status) and by the FieldClassifier (authors / venue / publisher variant_type).
It maps those signals to {is_valid, taxonomy, hint, issues, reason} with a
fixed decision table — no LLM call.

Scope decision: only title / year / venue / authors count as "core" for the
hard hallucinated gate. DOI and arxiv_id mismatches are tracked as issues but
do NOT on their own force a hallucinated verdict — these identifiers are
frequently stale or wrong in citations even when the paper is real.
"""
from __future__ import annotations

from typing import Any

from .models import (
    AuthorVariantResult,
    CandidateMatch,
    CitationRecord,
    DownstreamAgentResult,
    FieldComparisonResult,
    ValidAgentResult,
    VerdictLabel,
)


_HARD_HALLUCINATED_ISSUES = {
    "title_mismatch",
    "year_mismatch",
    "venue_mismatch",
    "venue_candidate_missing",
    "year_candidate_missing",
    "author_mismatch",
    "author_count_mismatch",
    "author_reordered",
}

_SOFT_POTENTIAL_ISSUES = {
    "author_name_variant",
    "pages_candidate_missing",
    "volume_candidate_missing",
    "publisher_candidate_missing",
    "location_candidate_missing",
}


def _status(fs: dict, name: str) -> str:
    entry = fs.get(name) if isinstance(fs, dict) else None
    return (entry or {}).get("status", "") if isinstance(entry, dict) else ""


def _variant(fs: dict, name: str, key: str) -> str:
    entry = fs.get(name) if isinstance(fs, dict) else None
    return (entry or {}).get(key, "") if isinstance(entry, dict) else ""


class RuleBasedValidAgent:
    """Drop-in replacement for BedrockValidAgent — pure rule dispatch."""

    def evaluate(
        self,
        citation: CitationRecord,
        candidate: CandidateMatch,
        comparison: FieldComparisonResult,
        author_analysis: AuthorVariantResult | None = None,
    ) -> ValidAgentResult:
        fs = comparison.field_status if isinstance(comparison, FieldComparisonResult) else {}
        fs = fs or {}

        # Pull per-field status (post-FieldClassifier upgrades)
        title_s     = _status(fs, "title")
        year_s      = _status(fs, "year")
        venue_s     = _status(fs, "venue")
        doi_s       = _status(fs, "doi")
        arxiv_s     = _status(fs, "arxiv_id")
        pages_s     = _status(fs, "pages")
        volume_s    = _status(fs, "volume")
        publisher_s = _status(fs, "publisher")
        location_s  = _status(fs, "location")
        author_s    = _status(fs, "authors")

        # Author variant from classifier (falls back to caller-supplied analysis)
        author_variant = (
            _variant(fs, "authors", "author_variant_type")
            or (author_analysis.overall if author_analysis else "")
        )
        has_et_al = bool(
            fs.get("authors", {}).get("has_et_al", False)
            if isinstance(fs.get("authors"), dict) else False
        )

        issues: list[str] = []

        # --- Core hard-mismatch gate (title / year / venue only) -----------
        if title_s == "mismatch":
            issues.append("title_mismatch")
        if year_s == "mismatch":
            # Year must match exactly; any diff is a hard mismatch.
            issues.append("year_mismatch")
        if venue_s == "mismatch":
            issues.append("venue_mismatch")

        # --- Authors -------------------------------------------------------
        # Classifier verdict is authoritative when present. exact / r2_initial
        # both mean "effectively equal" — no issue regardless of the raw
        # rule-level status=mismatch.
        if author_variant in ("exact", "r2_initial"):
            pass  # no author issue
        elif author_variant == "h2_error":
            if author_s == "count_mismatch":
                issues.append("author_count_mismatch")
            elif author_s == "reordered_match":
                issues.append("author_reordered")
            else:
                issues.append("author_mismatch")
        elif author_variant == "p1_variant":
            issues.append("author_name_variant")
        else:
            # Classifier silent / unavailable → fall back to raw status.
            if author_s == "count_mismatch":
                issues.append("author_count_mismatch")
            elif author_s == "reordered_match":
                issues.append("author_reordered")
            elif author_s == "mismatch":
                issues.append("author_mismatch")
            elif author_s == "partial_overlap":
                issues.append("author_name_variant")

        # --- Core candidate_missing (venue / year) — hard -----------------
        if venue_s == "candidate_missing":
            issues.append("venue_candidate_missing")
        if year_s == "candidate_missing":
            issues.append("year_candidate_missing")

        # --- DOI / arxiv_id — tracked but NOT in the hard gate -----------
        if doi_s == "mismatch":
            issues.append("doi_mismatch")
        if arxiv_s == "mismatch":
            issues.append("arxiv_id_mismatch")
        if doi_s == "candidate_missing":
            issues.append("doi_candidate_missing")
        if arxiv_s == "candidate_missing":
            issues.append("arxiv_id_candidate_missing")

        # --- H6-class candidate_missing (soft → P3) ----------------------
        if pages_s == "candidate_missing":
            issues.append("pages_candidate_missing")
        if volume_s == "candidate_missing":
            issues.append("volume_candidate_missing")
        if publisher_s == "candidate_missing":
            issues.append("publisher_candidate_missing")
        if location_s == "candidate_missing":
            issues.append("location_candidate_missing")

        # --- H6-class mismatch (pages/volume/publisher/location) ---------
        # Treat as hard: wrong volume is genuinely wrong info.
        if pages_s == "mismatch":
            issues.append("pages_mismatch")
        if volume_s == "mismatch":
            issues.append("volume_mismatch")
        if publisher_s == "mismatch":
            issues.append("publisher_mismatch")
        if location_s == "mismatch":
            issues.append("location_mismatch")

        issue_set = set(issues)
        has_hard = bool(issue_set & _HARD_HALLUCINATED_ISSUES) or any(
            i in issue_set for i in (
                "pages_mismatch", "volume_mismatch",
                "publisher_mismatch", "location_mismatch",
            )
        )
        has_soft = bool(issue_set & _SOFT_POTENTIAL_ISSUES)
        has_id_issue = any(
            i in issue_set for i in (
                "doi_mismatch", "arxiv_id_mismatch",
                "doi_candidate_missing", "arxiv_id_candidate_missing",
            )
        )

        if has_hard:
            return ValidAgentResult(
                is_valid=False,
                taxonomy=[],
                hint="likely_hallucinated",
                issues=issues,
                reason=f"RuleBased: hard mismatch. Issues: {sorted(issues)}",
                field_result=comparison,
                candidate=candidate,
            )

        if has_soft or has_id_issue:
            # DOI/arxiv_id only → route as likely_hallucinated so HallucinatedAgent
            # can decide H5 vs explain-away. Soft-only (P1 variant / H6 candidate_missing)
            # → likely_potential.
            hint = "likely_hallucinated" if (has_id_issue and not has_soft) else "likely_potential"
            return ValidAgentResult(
                is_valid=False,
                taxonomy=[],
                hint=hint,
                issues=issues,
                reason=f"RuleBased: soft discrepancy. Issues: {sorted(issues)}",
                field_result=comparison,
                candidate=candidate,
            )

        # --- VALID path: pick taxonomy -----------------------------------
        venue_variant     = _variant(fs, "venue", "venue_variant_type")
        publisher_variant = _variant(fs, "publisher", "publisher_variant_type")
        acronym_match = venue_variant == "alias" or publisher_variant == "alias"

        if has_et_al:
            taxonomy = ["R3"]
            reason = "RuleBased: et al. abbreviation, listed fields match."
        elif author_variant == "r2_initial":
            taxonomy = ["R2"]
            reason = "RuleBased: single-letter author initial variant."
        elif acronym_match:
            taxonomy = ["R2"]
            reason = "RuleBased: venue/publisher acronym variant, other fields match."
        else:
            taxonomy = ["R1"]
            reason = "RuleBased: all fields match exactly."

        return ValidAgentResult(
            is_valid=True,
            taxonomy=taxonomy,
            hint="",
            issues=[],
            reason=reason,
            field_result=comparison,
            candidate=candidate,
        )


class RuleBasedHallucinatedAgent:
    """Deterministic HallucinatedAgent — maps field_status + FieldClassifier
    verdicts to H1..H6 taxonomy codes, no LLM call.

    H1: title mismatch
    H2: authors h2_error (count / reorder / fabrication)
    H3: venue different (after FieldClassifier judgment)
    H4: year mismatch or year candidate_missing
    H5: DOI or arxiv_id mismatch
    H6: pages / volume / publisher / location mismatch
    """

    def evaluate(
        self,
        citation: CitationRecord,
        candidate: CandidateMatch,
        comparison: FieldComparisonResult,
        valid_hint: ValidAgentResult,
    ) -> DownstreamAgentResult:
        fs = comparison.field_status if isinstance(comparison, FieldComparisonResult) else {}
        fs = fs or {}

        taxonomy: list[str] = []
        reasons: list[str] = []

        def _add(code: str, why: str) -> None:
            if code not in taxonomy:
                taxonomy.append(code)
            reasons.append(why)

        # H1 — title: mismatch OR candidate_missing (no candidate found the
        # title at all → cannot verify the paper exists → fabricated).
        title_s = _status(fs, "title")
        if title_s == "mismatch":
            _add("H1", "title mismatch")
        elif title_s == "candidate_missing":
            _add("H1", "title candidate_missing (no candidate returned a title)")

        # H2 — authors (trust classifier verdict when present)
        author_variant = _variant(fs, "authors", "author_variant_type")
        author_s = _status(fs, "authors")
        if author_variant == "h2_error":
            _add("H2", f"authors h2_error ({author_s or 'no raw status'})")
        elif author_variant in ("exact", "r2_initial", "p1_variant"):
            pass  # classifier says authors OK / soft variant; not H2
        elif author_s in ("mismatch", "count_mismatch", "reordered_match"):
            _add("H2", f"authors raw {author_s} (no classifier verdict)")

        # H3 — venue fabrication:
        #   (a) classifier says different AND status is mismatch, OR
        #   (b) candidate returned no venue at all (candidate_missing) — we
        #       can't verify the venue exists, treat as H3 like we do H1
        #       for title candidate_missing.
        venue_s = _status(fs, "venue")
        venue_variant = _variant(fs, "venue", "venue_variant_type")
        if venue_variant == "different" and venue_s == "mismatch":
            _add("H3", "venue different")
        elif venue_s == "candidate_missing":
            _add("H3", "venue candidate_missing (no candidate returned a venue)")

        # H4 — year
        year_s = _status(fs, "year")
        if year_s == "mismatch":
            _add("H4", "year mismatch")
        elif year_s == "candidate_missing":
            _add("H4", "year candidate_missing")

        # H5 — DOI / arxiv_id mismatch
        if _status(fs, "doi") == "mismatch":
            _add("H5", "doi mismatch")
        if _status(fs, "arxiv_id") == "mismatch":
            _add("H5", "arxiv_id mismatch")

        # H6 — peripheral mismatch (explicit contradiction, not candidate_missing)
        for f in ("pages", "volume", "publisher", "location"):
            if _status(fs, f) == "mismatch":
                _add("H6", f"{f} mismatch")
                break  # one H6 entry is enough

        # Default: no specific signal → H1 (complete fabrication)
        if not taxonomy:
            taxonomy = ["H1"]
            reasons = ["no specific signal; defaulting to H1 (likely fabricated)"]

        return DownstreamAgentResult(
            label=VerdictLabel.FAKE_REFERENCE,
            taxonomy=taxonomy,
            reason=f"RuleBased: {'; '.join(reasons)}",
        )
