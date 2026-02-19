from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .models import CandidateMatch, CitationRecord, CitationVerdict, EvidenceTrace, ExtractionQuality, VerdictLabel, candidate_match_to_dict


class LLMResolver(Protocol):
    def resolve(self, citation: CitationRecord, candidates: list[CandidateMatch], conflicts: list[str]) -> str:
        """Return a short disambiguation note used in adjudication reason."""


@dataclass
class AdjudicationPolicy:
    valid_threshold: float = 0.85
    flawed_threshold: float = 0.55
    hallucination_threshold: float = 0.30
    review_confidence_threshold: float = 0.72


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

    if llm_resolver and policy.flawed_threshold <= best.score < policy.valid_threshold:
        llm_note = llm_resolver.resolve(citation, candidates[:3], conflicts)
    else:
        llm_note = ""

    if best.score >= policy.valid_threshold and not any(conf in conflicts for conf in ("doi_mismatch", "arxiv_id_mismatch")):
        verdict = VerdictLabel.VALID
        reason = "Strong metadata match across title/authors/year and no hard identifier conflicts."
    elif best.score >= policy.flawed_threshold:
        verdict = VerdictLabel.FLAWED_CITATION
        reason = "Closest match exists but has metadata inconsistencies."
    elif best.score <= policy.hallucination_threshold and ("title_mismatch" in conflicts or "author_mismatch" in conflicts):
        verdict = VerdictLabel.SUSPECTED_HALLUCINATION
        reason = "Closest candidate is weak and contains major title/author mismatches."
    else:
        verdict = VerdictLabel.INSUFFICIENT_EVIDENCE
        reason = "Evidence is inconclusive and requires manual review."

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
