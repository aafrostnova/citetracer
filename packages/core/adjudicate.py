from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Protocol

from .models import (
    CandidateMatch,
    CitationRecord,
    CitationVerdict,
    EvidenceTrace,
    ExtractionQuality,
    VerdictLabel,
    candidate_match_to_dict,
    canonical_verdict_label,
)


class LLMResolver(Protocol):
    def review(
        self,
        citation: CitationRecord,
        candidates: list[CandidateMatch],
        conflicts: list[str],
        proposed_verdict: VerdictLabel,
    ) -> dict[str, Any] | None:
        """
        Optional LLM second-opinion for ambiguous/fake references.
        Expected keys: label_override, confidence_override, note.
        """


@dataclass
class AdjudicationPolicy:
    valid_threshold: float = 0.85
    potential_threshold: float = 0.55
    fake_threshold: float = 0.30
    review_confidence_threshold: float = 0.72
    max_sources_with_identifier: int = 2
    max_sources_without_identifier: int = 4


def _apply_extraction_penalty(confidence: float, extraction_quality: ExtractionQuality) -> float:
    if extraction_quality == ExtractionQuality.HIGH:
        return confidence
    if extraction_quality == ExtractionQuality.MEDIUM:
        return max(0.0, confidence - 0.05)
    if extraction_quality == ExtractionQuality.LOW:
        return max(0.0, confidence - 0.15)
    return confidence


def adjudicate(
    citation: CitationRecord,
    candidates: list[CandidateMatch],
    evidence: list[EvidenceTrace],
    extraction_quality: ExtractionQuality = ExtractionQuality.UNKNOWN,
    llm_resolver: LLMResolver | None = None,
    policy: AdjudicationPolicy | None = None,
) -> CitationVerdict:
    policy = policy or AdjudicationPolicy()
    evidence_sources = sorted({trace.connector for trace in evidence if trace.error is None})
    source_errors = [trace for trace in evidence if trace.error]

    if not candidates:
        reason = "No high-confidence candidate returned by available sources."
        if source_errors and len(source_errors) == len(evidence):
            reason = "All sources failed or timed out, unable to establish ground truth."
        confidence = 0.25 if source_errors else 0.35
        confidence = _apply_extraction_penalty(confidence, extraction_quality)
        return CitationVerdict(
            citation_id=citation.citation_id,
            verdict=VerdictLabel.INSUFFICIENT_EVIDENCE,
            confidence=confidence,
            evidence_sources=evidence_sources,
            conflicts=["no_candidate"],
            adjudication_reason=reason,
            matched_candidate=None,
            needs_human_review=True,
            extraction_quality=extraction_quality,
        )

    best = candidates[0]
    conflicts = list(best.conflicts)
    confidence = best.score
    confidence = _apply_extraction_penalty(confidence, extraction_quality)

    hard_conflicts = {"doi_mismatch", "arxiv_id_mismatch", "title_mismatch", "author_mismatch", "venue_mismatch"}
    has_hard_conflict = any(conflict in hard_conflicts for conflict in conflicts)
    llm_note = ""

    if best.score >= policy.valid_threshold and not any(conf in conflicts for conf in ("doi_mismatch", "arxiv_id_mismatch")):
        verdict = VerdictLabel.VALID
        reason = "Strong metadata match across title/authors/year and no hard identifier conflicts."
    elif best.score >= policy.potential_threshold:
        verdict = VerdictLabel.POTENTIAL_REFERENCE
        reason = "Closest match exists but has metadata ambiguities/inconsistencies."
    elif best.score <= policy.fake_threshold and has_hard_conflict:
        verdict = VerdictLabel.FAKE_REFERENCE
        reason = "Closest candidate is weak and contains major metadata mismatches."
    else:
        verdict = VerdictLabel.INSUFFICIENT_EVIDENCE
        reason = "Evidence is inconclusive and requires manual review."

    if llm_resolver and verdict in {VerdictLabel.POTENTIAL_REFERENCE, VerdictLabel.FAKE_REFERENCE}:
        try:
            review = llm_resolver.review(citation, candidates[:3], conflicts, verdict) or {}
        except Exception as exc:  # noqa: BLE001 - reviewer failure should not crash adjudication
            review = {"note": f"LLM review failed: {exc}"}
        override = review.get("label_override")
        if override:
            verdict = canonical_verdict_label(override)
        if review.get("confidence_override") is not None:
            try:
                confidence = max(0.0, min(1.0, float(review["confidence_override"])))
            except (TypeError, ValueError):
                pass
        llm_note = str(review.get("note", "")).strip()

    if llm_note:
        reason = f"{reason} LLM disambiguation: {llm_note}"

    needs_human_review = verdict != VerdictLabel.VALID or confidence < policy.review_confidence_threshold

    return CitationVerdict(
        citation_id=citation.citation_id,
        verdict=verdict,
        confidence=confidence,
        evidence_sources=evidence_sources,
        conflicts=conflicts,
        adjudication_reason=reason,
        matched_candidate=candidate_match_to_dict(best),
        needs_human_review=needs_human_review,
        extraction_quality=extraction_quality,
    )
