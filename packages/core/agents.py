"""Multi-agent citation verification system.

Agents:
  1. FieldComparatorAgent — rule-based field comparison (no LLM)
  2. EvidenceGatherer — existing SecondaryVerifier (no LLM)
  3. StructuredJudgeAgent — LLM for structured source disambiguation (POTENTIAL/FAKE only)
  4. ExistenceJudgeAgent — LLM for web search evidence (VALID/POTENTIAL/FAKE)
  5. VoteAggregator — rule-based voting with conservative caps
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .models import (
    CandidateMatch,
    CitationRecord,
    CitationVerdict,
    DiscrepancyEvidence,
    ExtractionQuality,
    FieldComparisonResult,
    LLMJudgment,
    VerdictLabel,
    candidate_match_to_dict,
    canonical_verdict_label,
)


# ---------------------------------------------------------------------------
# Agent Protocols
# ---------------------------------------------------------------------------

class StructuredJudge(Protocol):
    """LLM judge for structured source candidates. Can only return POTENTIAL or FAKE."""
    def judge(
        self,
        citation: CitationRecord,
        candidate: CandidateMatch,
        comparison: FieldComparisonResult,
        evidence: list[DiscrepancyEvidence],
    ) -> LLMJudgment: ...


class ExistenceJudge(Protocol):
    """LLM judge for web search candidates. Can return VALID/POTENTIAL/FAKE."""
    def judge(
        self,
        citation: CitationRecord,
        candidate: CandidateMatch,
        source_paper_title: str,
    ) -> LLMJudgment: ...


# ---------------------------------------------------------------------------
# Agent 1: FieldComparator
# ---------------------------------------------------------------------------

_WEB_CONNECTORS = {"google_search", "web_search"}


def compare_fields(
    citation: CitationRecord,
    candidate: CandidateMatch,
    comparison_func,  # _build_comparison from adjudicate
    is_direct_valid_func,  # _is_direct_valid from adjudicate
    has_soft_discrepancies_func,  # _has_soft_discrepancies from adjudicate
) -> FieldComparisonResult:
    """Agent 1: Rule-based field comparison. Returns signal, not verdict."""
    conflicts = list(candidate.conflicts)
    comparison = comparison_func(citation, candidate, conflicts)
    field_status = comparison["field_status"]

    has_et_al = field_status.get("authors", {}).get("has_et_al", False)

    # Signal 1: Direct valid (all core fields match)
    if is_direct_valid_func(field_status, conflicts):
        return FieldComparisonResult(
            field_status=field_status,
            signal="direct_valid",
            has_et_al=has_et_al,
            comparison=comparison,
            conflicts=conflicts,
        )

    # Signal 2: Hard mismatch
    hard_fields = []
    title_status = field_status.get("title", {}).get("status", "")
    year_status = field_status.get("year", {}).get("status", "")
    author_status = field_status.get("authors", {}).get("status", "")

    if title_status == "mismatch":
        hard_fields.append("title:mismatch")
    if year_status == "mismatch":
        hard_fields.append("year:mismatch")
    if author_status == "count_mismatch":
        hard_fields.append("authors:count_mismatch")

    if hard_fields:
        return FieldComparisonResult(
            field_status=field_status,
            signal="hard_mismatch",
            hard_mismatch_fields=hard_fields,
            has_et_al=has_et_al,
            comparison=comparison,
            conflicts=conflicts,
        )

    # Signal 3: Soft discrepancy
    soft_fields = []
    if has_soft_discrepancies_func(field_status, conflicts):
        for fname in ("authors", "venue", "doi", "arxiv_id", "pages", "publisher", "location"):
            status = field_status.get(fname, {}).get("status", "")
            if status in ("mismatch", "partial_overlap", "candidate_missing", "reference_missing"):
                soft_fields.append(f"{fname}:{status}")

    if soft_fields:
        return FieldComparisonResult(
            field_status=field_status,
            signal="soft_discrepancy",
            soft_discrepancy_fields=soft_fields,
            has_et_al=has_et_al,
            comparison=comparison,
            conflicts=conflicts,
        )

    # Signal 4: No clear match
    return FieldComparisonResult(
        field_status=field_status,
        signal="no_evidence",
        has_et_al=has_et_al,
        comparison=comparison,
        conflicts=conflicts,
    )


# ---------------------------------------------------------------------------
# Agent 5: VoteAggregator
# ---------------------------------------------------------------------------

def _classify_taxonomy_from_fields(
    signal: str,
    field_status: dict[str, dict[str, Any]],
    has_et_al: bool,
    connector: str,
) -> list[str]:
    """Rule-based taxonomy classification from field comparison. Returns list of ALL detected issues."""
    if signal == "direct_valid":
        if has_et_al:
            return ["R3"]
        if connector in _WEB_CONNECTORS:
            return ["R4"]
        return ["R1"]

    if signal == "hard_mismatch":
        issues = []
        title_s = field_status.get("title", {}).get("status", "")
        year_s = field_status.get("year", {}).get("status", "")
        author_s = field_status.get("authors", {}).get("status", "")
        if title_s == "mismatch":
            issues.append("H2")
        if author_s == "count_mismatch":
            issues.append("H3")
        if year_s == "mismatch":
            issues.append("H5")
        return issues or ["H1"]

    if signal == "soft_discrepancy":
        issues = []
        author_s = field_status.get("authors", {}).get("status", "")
        venue_s = field_status.get("venue", {}).get("status", "")
        doi_s = field_status.get("doi", {}).get("status", "")
        pages_s = field_status.get("pages", {}).get("status", "")
        volume_s = field_status.get("volume", {}).get("status", "")
        publisher_s = field_status.get("publisher", {}).get("status", "")
        if author_s in ("mismatch", "partial_overlap", "reordered_match", "count_mismatch"):
            issues.append("H3")
        if venue_s == "mismatch":
            issues.append("H4")
        if doi_s == "mismatch":
            issues.append("H6")
        if pages_s == "mismatch" or volume_s == "mismatch" or publisher_s == "mismatch":
            issues.append("H7")
        return issues or ["H1"]

    return []


def aggregate_verdict(
    citation: CitationRecord,
    candidate: CandidateMatch,
    field_result: FieldComparisonResult,
    evidence: list[DiscrepancyEvidence],
    structured_judgment: LLMJudgment | None,
    existence_judgment: LLMJudgment | None,
    evidence_sources: list[str],
) -> CitationVerdict:
    """Agent 5: Aggregate all agent outputs into final verdict."""

    is_web = candidate.connector in _WEB_CONNECTORS
    fs = field_result.field_status

    # --- Rule 2: Direct valid → VALID ---
    if field_result.signal == "direct_valid":
        taxonomy = _classify_taxonomy_from_fields(
            "direct_valid", fs, field_result.has_et_al, candidate.connector
        )
        return _build_verdict(
            citation, candidate, VerdictLabel.VALID, taxonomy,
            f"All core fields match ({field_result.signal}).",
            field_result, evidence, evidence_sources,
        )

    # --- Rule 3: Hard mismatch on structured source → FAKE ---
    if field_result.signal == "hard_mismatch" and not is_web:
        taxonomy = _classify_taxonomy_from_fields(
            "hard_mismatch", fs, field_result.has_et_al, candidate.connector
        )
        reason = f"Hard field mismatch: {field_result.hard_mismatch_fields}. LLM cannot override."
        if structured_judgment:
            # Merge LLM detected issues into taxonomy
            taxonomy = list(set(taxonomy + structured_judgment.taxonomy))
            reason += f" LLM: {structured_judgment.note}"
        return _build_verdict(
            citation, candidate, VerdictLabel.FAKE_REFERENCE, taxonomy,
            reason, field_result, evidence, evidence_sources,
        )

    # --- Rule 4: Soft discrepancy → consult LLM ---
    if field_result.signal == "soft_discrepancy" and not is_web:
        all_explained = evidence and all(e.evidence_found for e in evidence)

        if structured_judgment:
            llm_verdict = structured_judgment.verdict
            llm_taxonomy = structured_judgment.taxonomy
            llm_note = structured_judgment.note
            llm_confidence = structured_judgment.confidence
        else:
            llm_verdict = VerdictLabel.FAKE_REFERENCE
            llm_taxonomy = []
            llm_note = "No LLM available."
            llm_confidence = 0.0

        # Rule 4a: All discrepancies explained → POTENTIAL
        if all_explained:
            verdict = VerdictLabel.POTENTIAL_REFERENCE
            if llm_verdict == VerdictLabel.FAKE_REFERENCE:
                verdict = VerdictLabel.FAKE_REFERENCE
            taxonomy = llm_taxonomy or ["P2"]
            reason = f"All discrepancies explained. Agent3: {llm_verdict.value}. {llm_note}"

        # Rule 4b: Some unexplained → default FAKE, LLM can upgrade to POTENTIAL
        else:
            verdict = VerdictLabel.FAKE_REFERENCE
            if llm_verdict == VerdictLabel.POTENTIAL_REFERENCE and llm_confidence >= 0.5:
                verdict = VerdictLabel.POTENTIAL_REFERENCE
            taxonomy = llm_taxonomy or []
            reason = f"Some discrepancies unexplained. Agent3: {llm_verdict.value}. {llm_note}"

        return _build_verdict(
            citation, candidate, verdict, taxonomy,
            reason, field_result, evidence, evidence_sources,
        )

    # --- Rule 5: Web search candidate ---
    if is_web:
        if existence_judgment:
            llm_verdict = existence_judgment.verdict
            llm_taxonomy = existence_judgment.taxonomy
            llm_note = existence_judgment.note
            llm_confidence = existence_judgment.confidence
        else:
            llm_verdict = VerdictLabel.POTENTIAL_REFERENCE
            llm_taxonomy = ["P3"]
            llm_note = "No LLM available."
            llm_confidence = 0.0

        # Web search VALID must still pass basic field checks against the citation.
        # The web candidate may have found the real paper but the citation's
        # year/venue/DOI may be mutated — web search alone can't catch that.
        if llm_verdict == VerdictLabel.VALID:
            web_issues = []
            # Check year if both present
            cand_year = candidate.year
            cite_year = citation.year
            if cite_year and cand_year and cite_year != cand_year:
                web_issues.append("H5")
            # Check venue if citation has one and candidate has one
            cite_venue = (citation.venue or "").strip().lower()
            cand_venue = (candidate.venue or "").strip().lower()
            if cite_venue and cand_venue and cite_venue != cand_venue:
                from .normalize import normalize_venue
                if normalize_venue(citation.venue) != normalize_venue(candidate.venue):
                    web_issues.append("H4")
            # If web found issues that contradict the citation → HALLUCINATED
            if web_issues:
                llm_verdict = VerdictLabel.FAKE_REFERENCE
                llm_taxonomy = web_issues
                llm_note = f"Web found paper but citation fields don't match: {web_issues}. {llm_note}"

        if llm_verdict == VerdictLabel.VALID:
            verdict = VerdictLabel.VALID
            taxonomy = llm_taxonomy or ["R4"]
        elif llm_verdict == VerdictLabel.POTENTIAL_REFERENCE:
            verdict = VerdictLabel.POTENTIAL_REFERENCE
            taxonomy = llm_taxonomy or ["P3"]
        else:
            verdict = VerdictLabel.FAKE_REFERENCE
            taxonomy = llm_taxonomy or ["P3"]

        # Confidence threshold
        if llm_confidence < 0.5 and verdict == VerdictLabel.VALID:
            verdict = VerdictLabel.POTENTIAL_REFERENCE
        if llm_confidence < 0.3 and verdict == VerdictLabel.POTENTIAL_REFERENCE:
            verdict = VerdictLabel.FAKE_REFERENCE

        reason = f"Web search. Agent4: {llm_verdict.value} (conf={llm_confidence:.2f}). {llm_note}"
        return _build_verdict(
            citation, candidate, verdict, taxonomy,
            reason, field_result, evidence, evidence_sources,
        )

    # --- Rule fallback ---
    taxonomy: list[str] = []
    if structured_judgment:
        taxonomy = structured_judgment.taxonomy
    return _build_verdict(
        citation, candidate, VerdictLabel.FAKE_REFERENCE, taxonomy or ["H1"],
        "No positive evidence of match.",
        field_result, evidence, evidence_sources,
    )


def _build_verdict(
    citation: CitationRecord,
    candidate: CandidateMatch,
    verdict: VerdictLabel,
    taxonomy: list[str],
    reason: str,
    field_result: FieldComparisonResult,
    evidence: list[DiscrepancyEvidence],
    evidence_sources: list[str],
) -> CitationVerdict:
    return CitationVerdict(
        citation_id=citation.citation_id,
        verdict=verdict,
        evidence_sources=evidence_sources,
        conflicts=field_result.conflicts,
        adjudication_reason=reason,
        matched_candidate=candidate_match_to_dict(candidate),
        reference_snapshot=field_result.comparison.get("reference", {}),
        comparison=field_result.comparison,
        taxonomy_subtype=taxonomy,
        needs_human_review=(verdict != VerdictLabel.VALID),
        secondary_evidence=evidence,
    )


# ---------------------------------------------------------------------------
# Multi-agent adjudication orchestrator
# ---------------------------------------------------------------------------

def multi_agent_adjudicate(
    citation: CitationRecord,
    candidates: list[CandidateMatch],
    evidence_traces: list[Any],
    extraction_quality: ExtractionQuality = ExtractionQuality.UNKNOWN,
    structured_judge: StructuredJudge | None = None,
    existence_judge: ExistenceJudge | None = None,
    secondary_verifier: Any = None,  # SecondaryVerifierProtocol
    source_paper_title: str = "",
    # Pass through comparison functions from adjudicate module
    build_comparison_func=None,
    is_direct_valid_func=None,
    has_soft_discrepancies_func=None,
) -> CitationVerdict:
    """Orchestrate multi-agent verification for one citation across all candidates."""
    from .adjudicate import (
        _build_comparison,
        _is_direct_valid,
        _has_soft_discrepancies,
        _reference_snapshot,
    )

    build_comp = build_comparison_func or _build_comparison
    is_valid = is_direct_valid_func or _is_direct_valid
    has_soft = has_soft_discrepancies_func or _has_soft_discrepancies

    evidence_sources = sorted({
        t.connector for t in evidence_traces if hasattr(t, 'connector') and t.error is None
    })

    # No candidates
    if not candidates:
        source_errors = [t for t in evidence_traces if hasattr(t, 'error') and t.error]
        if source_errors and len(source_errors) == len(evidence_traces):
            return CitationVerdict(
                citation_id=citation.citation_id,
                verdict=VerdictLabel.INSUFFICIENT_EVIDENCE,
                evidence_sources=evidence_sources,
                conflicts=["no_candidate"],
                adjudication_reason="All sources failed or timed out.",
                taxonomy_subtype=[],
                needs_human_review=True,
            )
        return CitationVerdict(
            citation_id=citation.citation_id,
            verdict=VerdictLabel.FAKE_REFERENCE,
            evidence_sources=evidence_sources,
            conflicts=["no_candidate"],
            adjudication_reason="No matching paper found in any database.",
            reference_snapshot={
                "title": citation.title,
                "authors": list(citation.authors),
                "venue": citation.venue,
                "year": citation.year,
            },
            taxonomy_subtype=["H1"],
            needs_human_review=True,
        )

    # Track best verdict across candidates
    best_verdict: CitationVerdict | None = None
    has_structured_hard_mismatch = ""  # "" | "hard" | "soft"
    all_evaluations: list[dict[str, Any]] = []

    priority = {
        VerdictLabel.VALID: 4,
        VerdictLabel.POTENTIAL_REFERENCE: 3,
        VerdictLabel.FAKE_REFERENCE: 2,
        VerdictLabel.INSUFFICIENT_EVIDENCE: 1,
    }

    for candidate in candidates:
        is_web = candidate.connector in _WEB_CONNECTORS

        # Agent 1: Field comparison
        field_result = compare_fields(
            citation, candidate, build_comp, is_valid, has_soft,
        )

        # Agent 2: Evidence gathering (only for soft discrepancies on structured sources)
        sec_evidence: list[DiscrepancyEvidence] = []
        if field_result.signal == "soft_discrepancy" and not is_web and secondary_verifier:
            try:
                sec_evidence = secondary_verifier.gather_evidence(
                    citation, candidate, field_result.field_status, field_result.conflicts,
                )
            except Exception:
                sec_evidence = []

        # Agent 3: Structured judge (only for non-web with discrepancies)
        s_judgment = None
        if not is_web and field_result.signal in ("soft_discrepancy", "hard_mismatch", "no_evidence"):
            if structured_judge:
                try:
                    s_judgment = structured_judge.judge(
                        citation, candidate, field_result, sec_evidence,
                    )
                    # ENFORCE: structured judge cannot return VALID
                    if s_judgment.verdict == VerdictLabel.VALID:
                        s_judgment = LLMJudgment(
                            verdict=VerdictLabel.POTENTIAL_REFERENCE,
                            taxonomy=s_judgment.taxonomy,
                            confidence=s_judgment.confidence,
                            note=f"Capped from VALID to POTENTIAL. {s_judgment.note}",
                            raw_response=s_judgment.raw_response,
                        )
                except Exception as exc:
                    s_judgment = LLMJudgment(
                        verdict=VerdictLabel.FAKE_REFERENCE,
                        note=f"LLM error: {exc}",
                    )

        # Agent 4: Existence judge (only for web candidates)
        e_judgment = None
        if is_web and existence_judge:
            try:
                e_judgment = existence_judge.judge(
                    citation, candidate, source_paper_title,
                )
            except Exception as exc:
                e_judgment = LLMJudgment(
                    verdict=VerdictLabel.FAKE_REFERENCE,
                    note=f"LLM error: {exc}",
                )

        # Agent 5: Aggregate
        current = aggregate_verdict(
            citation, candidate, field_result, sec_evidence,
            s_judgment, e_judgment, evidence_sources,
        )

        # Track structured mismatches for cross-candidate capping
        if not is_web and field_result.signal == "hard_mismatch":
            has_structured_hard_mismatch = "hard"  # title/year/count mismatch
        elif (not is_web and current.verdict in (VerdictLabel.FAKE_REFERENCE, VerdictLabel.POTENTIAL_REFERENCE)
                and has_structured_hard_mismatch != "hard"):
            has_structured_hard_mismatch = "soft"  # paper found but fields don't fully match

        # Cap web search when structured source found issues
        if is_web and has_structured_hard_mismatch and current.verdict == VerdictLabel.VALID:
            if has_structured_hard_mismatch == "hard":
                cap_verdict = VerdictLabel.FAKE_REFERENCE
                cap_taxonomy = current.taxonomy_subtype or ["H2"]
                cap_reason = "Structured DB found hard mismatch; web search overridden to FAKE."
            else:
                cap_verdict = VerdictLabel.POTENTIAL_REFERENCE
                cap_taxonomy = current.taxonomy_subtype or ["P2"]
                cap_reason = "Structured DB found soft mismatch; web search capped to POTENTIAL."

            current = CitationVerdict(
                citation_id=citation.citation_id,
                verdict=cap_verdict,
                evidence_sources=evidence_sources,
                conflicts=["structured_mismatch_caps_web"],
                adjudication_reason=f"{cap_reason} {current.adjudication_reason}",
                matched_candidate=current.matched_candidate,
                reference_snapshot=current.reference_snapshot,
                comparison=current.comparison,
                taxonomy_subtype=cap_taxonomy,
                needs_human_review=True,
                secondary_evidence=sec_evidence,
            )

        # Record evaluation
        all_evaluations.append({
            "connector": candidate.connector,
            "candidate": candidate_match_to_dict(candidate),
            "verdict": current.verdict.value,
            "reason": current.adjudication_reason[:300],
            "taxonomy": current.taxonomy_subtype,
            "comparison": field_result.comparison,
        })

        # VALID from structured source → return immediately
        if current.verdict == VerdictLabel.VALID and not is_web:
            current.candidate_evaluations = all_evaluations
            return current

        # Track best
        if best_verdict is None:
            best_verdict = current
        elif priority.get(current.verdict, 0) > priority.get(best_verdict.verdict, 0):
            best_verdict = current

    if best_verdict is not None:
        best_verdict.candidate_evaluations = all_evaluations
        return best_verdict

    # Fallback
    return CitationVerdict(
        citation_id=citation.citation_id,
        verdict=VerdictLabel.FAKE_REFERENCE,
        evidence_sources=evidence_sources,
        conflicts=[],
        adjudication_reason="No candidate provided sufficient evidence.",
        taxonomy_subtype="H1",
        candidate_evaluations=all_evaluations,
        needs_human_review=True,
    )
