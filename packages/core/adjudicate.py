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
from .normalize import normalize_arxiv_id, normalize_title


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


def _norm_text(value: Any) -> str:
    text = " ".join(str(value or "").strip().lower().split())
    return text.strip(" .,:;!?\"'()[]{}")


def _norm_title(value: Any) -> str:
    return normalize_title(str(value or ""))


def _norm_identifier(value: Any) -> str:
    return _norm_text(value).rstrip(".,;")


def _norm_arxiv_id(value: Any) -> str:
    return normalize_arxiv_id(str(value or ""))


def _norm_url(value: Any) -> str:
    return _norm_identifier(value).rstrip("/")


def _to_author_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, list):
        return len(value) == 0
    return False


def _compare_scalar(reference: Any, candidate: Any, normalizer) -> dict[str, Any]:
    if _is_missing(reference) and _is_missing(candidate):
        status = "both_missing"
    elif _is_missing(reference):
        status = "reference_missing"
    elif _is_missing(candidate):
        status = "candidate_missing"
    else:
        status = "match" if normalizer(reference) == normalizer(candidate) else "mismatch"
    return {
        "status": status,
        "reference": reference,
        "candidate": candidate,
    }


def _compare_year(reference: Any, candidate: Any, candidate_version_years: list[Any] | None = None) -> dict[str, Any]:
    version_years: list[int] = []
    for item in candidate_version_years or []:
        try:
            year_value = int(str(item).strip())
        except (TypeError, ValueError):
            continue
        if year_value not in version_years:
            version_years.append(year_value)

    if _is_missing(reference) and _is_missing(candidate) and not version_years:
        status = "both_missing"
    elif _is_missing(reference):
        status = "reference_missing"
    elif _is_missing(candidate) and not version_years:
        status = "candidate_missing"
    else:
        reference_text = str(reference)
        candidate_matches = str(candidate) == reference_text if not _is_missing(candidate) else False
        version_matches = reference_text in {str(year) for year in version_years}
        status = "match" if candidate_matches or version_matches else "mismatch"

    return {
        "status": status,
        "reference": reference,
        "candidate": candidate,
        "candidate_version_years": version_years,
    }


def _compare_authors(reference: list[str], candidate: list[str]) -> dict[str, Any]:
    ref = _to_author_list(reference)
    cand = _to_author_list(candidate)
    ref_norm = [_norm_text(name) for name in ref]
    cand_norm = [_norm_text(name) for name in cand]

    if not ref_norm and not cand_norm:
        status = "both_missing"
        overlap = 0
    elif not ref_norm:
        status = "reference_missing"
        overlap = 0
    elif not cand_norm:
        status = "candidate_missing"
        overlap = 0
    elif ref_norm == cand_norm:
        status = "match"
        overlap = len(ref_norm)
    elif set(ref_norm) == set(cand_norm):
        status = "reordered_match"
        overlap = len(ref_norm)
    else:
        overlap_set = set(ref_norm).intersection(set(cand_norm))
        overlap = len(overlap_set)
        status = "partial_overlap" if overlap > 0 else "mismatch"

    return {
        "status": status,
        "reference": ref,
        "candidate": cand,
        "overlap_count": overlap,
        "reference_count": len(ref),
        "candidate_count": len(cand),
    }


def _reference_snapshot(citation: CitationRecord) -> dict[str, Any]:
    return {
        "title": citation.title,
        "authors": list(citation.authors),
        "venue": citation.venue,
        "year": citation.year,
        "doi": citation.doi,
        "arxiv_id": citation.arxiv_id,
        "url": citation.url,
    }


def _candidate_snapshot(candidate: CandidateMatch | None) -> dict[str, Any]:
    if candidate is None:
        return {}
    return {
        "connector": candidate.connector,
        "score": round(float(candidate.score), 4),
        "title": candidate.title,
        "authors": list(candidate.authors),
        "venue": candidate.venue,
        "year": candidate.year,
        "doi": candidate.doi,
        "arxiv_id": candidate.arxiv_id,
        "url": candidate.url,
    }


def _build_comparison(
    citation: CitationRecord,
    candidate: CandidateMatch | None,
    conflicts: list[str],
) -> dict[str, Any]:
    reference = _reference_snapshot(citation)
    candidate_data = _candidate_snapshot(candidate)
    candidate_version_years = []
    if candidate is not None and isinstance(candidate.raw_record, dict):
        raw_years = candidate.raw_record.get("version_years")
        if isinstance(raw_years, list):
            candidate_version_years = raw_years
    candidate_core = {
        "title": candidate_data.get("title", ""),
        "authors": candidate_data.get("authors", []),
        "venue": candidate_data.get("venue", ""),
        "year": candidate_data.get("year"),
        "doi": candidate_data.get("doi", ""),
        "arxiv_id": candidate_data.get("arxiv_id", ""),
        "url": candidate_data.get("url", ""),
    }

    field_status = {
        "title": _compare_scalar(reference["title"], candidate_core["title"], _norm_title),
        "authors": _compare_authors(reference["authors"], candidate_core["authors"]),
        "venue": _compare_scalar(reference["venue"], candidate_core["venue"], _norm_text),
        "year": _compare_year(reference["year"], candidate_core["year"], candidate_version_years),
        "doi": _compare_scalar(reference["doi"], candidate_core["doi"], _norm_identifier),
        "arxiv_id": _compare_scalar(reference["arxiv_id"], candidate_core["arxiv_id"], _norm_arxiv_id),
        "url": _compare_scalar(reference["url"], candidate_core["url"], _norm_url),
    }

    summary_pairs = [f"{name}:{meta['status']}" for name, meta in field_status.items()]
    return {
        "connector": candidate_data.get("connector", ""),
        "score": candidate_data.get("score"),
        "conflicts": list(conflicts),
        "reference": reference,
        "candidate": candidate_core,
        "field_status": field_status,
        "summary": ", ".join(summary_pairs),
    }


def _status_in(meta: dict[str, Any], accepted: set[str]) -> bool:
    return str(meta.get("status", "") or "") in accepted


def _requires_llm_recheck(field_status: dict[str, dict[str, Any]], conflicts: list[str]) -> bool:
    title_status = field_status["title"]["status"]
    year_status = field_status["year"]["status"]
    if title_status == "mismatch" or year_status == "mismatch":
        return False

    review_statuses = {
        "mismatch",
        "partial_overlap",
        "candidate_missing",
        "reference_missing",
    }
    for field_name in ("authors", "venue", "doi", "arxiv_id"):
        if _status_in(field_status[field_name], review_statuses):
            return True

    review_conflicts = {
        "author_mismatch",
        "venue_mismatch",
        "doi_mismatch",
        "arxiv_id_mismatch",
    }
    return any(conflict in review_conflicts for conflict in conflicts)


def _is_direct_valid(field_status: dict[str, dict[str, Any]], conflicts: list[str]) -> bool:
    if conflicts:
        return False
    if field_status["title"]["status"] != "match":
        return False
    if field_status["year"]["status"] != "match":
        return False
    if not _status_in(field_status["authors"], {"match", "both_missing"}):
        return False
    if not _status_in(field_status["venue"], {"match", "both_missing", "reference_missing", "candidate_missing"}):
        return False
    if not _status_in(field_status["doi"], {"match", "both_missing", "reference_missing", "candidate_missing"}):
        return False
    if not _status_in(field_status["arxiv_id"], {"match", "both_missing", "reference_missing", "candidate_missing"}):
        return False
    return True


def _evaluate_candidate(
    citation: CitationRecord,
    candidate: CandidateMatch,
    extraction_quality: ExtractionQuality,
    policy: AdjudicationPolicy,
) -> tuple[VerdictLabel, float, str, list[str], dict[str, Any], bool]:
    conflicts = list(candidate.conflicts)
    confidence = _apply_extraction_penalty(candidate.score, extraction_quality)
    comparison = _build_comparison(citation, candidate, conflicts)
    field_status = comparison["field_status"]
    should_run_llm_recheck = False

    if _is_direct_valid(field_status, conflicts):
        verdict = VerdictLabel.VALID
        confidence = max(confidence, 0.99)
        reason = "All core metadata fields match, so the citation is treated as a direct valid match."
    elif field_status["title"]["status"] == "mismatch" or field_status["year"]["status"] == "mismatch":
        verdict = VerdictLabel.FAKE_REFERENCE
        confidence = max(confidence, 0.9)
        reason = "Title or year mismatches the current candidate, so the reference is treated as fake for this candidate."
        should_run_llm_recheck = True
    elif _requires_llm_recheck(field_status, conflicts):
        verdict = VerdictLabel.POTENTIAL_REFERENCE
        reason = "Author, venue, or identifier metadata conflicts require LLM recheck."
        should_run_llm_recheck = True
    elif candidate.score >= policy.potential_threshold:
        verdict = VerdictLabel.POTENTIAL_REFERENCE
        reason = "Current candidate is plausible but still ambiguous."
        should_run_llm_recheck = True
    else:
        verdict = VerdictLabel.INSUFFICIENT_EVIDENCE
        reason = "Current candidate is inconclusive."
        should_run_llm_recheck = True

    return verdict, confidence, reason, conflicts, comparison, should_run_llm_recheck


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
            reference_snapshot=_reference_snapshot(citation),
            comparison=_build_comparison(citation, None, ["no_candidate"]),
            candidate_evaluations=[],
            llm_recheck_reason="",
            needs_human_review=True,
            extraction_quality=extraction_quality,
        )

    preferred_fallback: CitationVerdict | None = None
    candidate_evaluations: list[dict[str, Any]] = []

    for candidate in candidates:

        verdict, confidence, reason, conflicts, comparison, should_run_llm_recheck = _evaluate_candidate(
            citation=citation,
            candidate=candidate,
            extraction_quality=extraction_quality,
            policy=policy,
        )
        llm_note = ""
        import pdb
        pdb.set_trace()
        if llm_resolver and should_run_llm_recheck:
            try:
                review = llm_resolver.review(citation, [candidate], conflicts, verdict) or {}
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

        candidate_evaluations.append(
            {
                "connector": candidate.connector,
                "candidate": candidate_match_to_dict(candidate),
                "verdict": canonical_verdict_label(verdict).value,
                "reason": reason,
                "llm_recheck_reason": llm_note,
                "comparison": comparison,
            }
        )

        current = CitationVerdict(
            citation_id=citation.citation_id,
            verdict=verdict,
            confidence=confidence,
            evidence_sources=evidence_sources,
            conflicts=conflicts,
            adjudication_reason=reason,
            matched_candidate=candidate_match_to_dict(candidate),
            reference_snapshot=_reference_snapshot(citation),
            comparison=comparison,
            candidate_evaluations=list(candidate_evaluations),
            llm_recheck_reason=llm_note,
            needs_human_review=(verdict != VerdictLabel.VALID or confidence < policy.review_confidence_threshold),
            extraction_quality=extraction_quality,
        )

        if verdict == VerdictLabel.VALID:
            return current

        if preferred_fallback is None:
            preferred_fallback = current
            continue

        priority = {
            VerdictLabel.POTENTIAL_REFERENCE: 3,
            VerdictLabel.FAKE_REFERENCE: 2,
            VerdictLabel.INSUFFICIENT_EVIDENCE: 1,
        }
        if priority.get(verdict, 0) > priority.get(preferred_fallback.verdict, 0):
            preferred_fallback = current

    if preferred_fallback is not None:
        return preferred_fallback

    best = candidates[0]
    return CitationVerdict(
        citation_id=citation.citation_id,
        verdict=VerdictLabel.INSUFFICIENT_EVIDENCE,
        confidence=_apply_extraction_penalty(best.score, extraction_quality),
        evidence_sources=evidence_sources,
        conflicts=list(best.conflicts),
        adjudication_reason="Evidence is inconclusive and requires manual review.",
        matched_candidate=candidate_match_to_dict(best),
        reference_snapshot=_reference_snapshot(citation),
        comparison=_build_comparison(citation, best, list(best.conflicts)),
        candidate_evaluations=list(candidate_evaluations),
        llm_recheck_reason="",
        needs_human_review=True,
        extraction_quality=extraction_quality,
    )
