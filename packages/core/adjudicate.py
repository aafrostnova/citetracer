from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Protocol

from .models import (
    CandidateMatch,
    CitationRecord,
    CitationVerdict,
    DiscrepancyEvidence,
    EvidenceTrace,
    ExtractionQuality,
    VerdictLabel,
    candidate_match_to_dict,
    canonical_verdict_label,
)
from .normalize import normalize_arxiv_id, normalize_title, normalize_venue, similarity


class LLMResolver(Protocol):
    def review(
        self,
        citation: CitationRecord,
        candidates: list[CandidateMatch],
        conflicts: list[str],
        proposed_verdict: VerdictLabel,
        secondary_evidence: list[DiscrepancyEvidence] | None = None,
    ) -> dict[str, Any] | None:
        """
        Optional LLM second-opinion for ambiguous/fake references.
        Expected keys: label_override, note.
        """


class SecondaryVerifierProtocol(Protocol):
    def gather_evidence(
        self,
        citation: CitationRecord,
        candidate: CandidateMatch,
        field_status: dict[str, dict[str, Any]],
        conflicts: list[str],
    ) -> list[DiscrepancyEvidence]: ...


@dataclass
class AdjudicationPolicy:
    pass


def _attempt_upgrade(
    secondary_evidence: list[DiscrepancyEvidence],
) -> tuple[VerdictLabel, str]:
    """Upgrade from FAKE to POTENTIAL only if ALL discrepancies have positive evidence."""
    if not secondary_evidence:
        return VerdictLabel.FAKE_REFERENCE, "No secondary evidence gathered."

    unexplained = [ev for ev in secondary_evidence if not ev.evidence_found]
    if unexplained:
        fields = ", ".join(ev.field for ev in unexplained)
        details = "; ".join(f"{ev.field}: {ev.explanation}" for ev in unexplained)
        return (
            VerdictLabel.FAKE_REFERENCE,
            f"Discrepancies in [{fields}] not explained by evidence. {details}",
        )

    explained = "; ".join(f"{ev.field}: {ev.explanation}" for ev in secondary_evidence)
    return (
        VerdictLabel.POTENTIAL_REFERENCE,
        f"All discrepancies explained by positive evidence. {explained}",
    )


def _norm_text(value: Any) -> str:
    text = " ".join(str(value or "").strip().lower().split())
    return text.strip(" .,:;!?\"'()[]{}")


def _norm_title(value: Any) -> str:
    return normalize_title(str(value or ""))


def _norm_identifier(value: Any) -> str:
    return _norm_text(value).rstrip(".,;")


def _norm_arxiv_id(value: Any) -> str:
    return normalize_arxiv_id(str(value or ""))


def _norm_venue(value: Any) -> str:
    return normalize_venue(str(value or ""))


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


def _compare_venue(reference: Any, candidate: Any) -> dict[str, Any]:
    """Venue comparison with containment and similarity fallback."""
    if _is_missing(reference) and _is_missing(candidate):
        status = "both_missing"
    elif _is_missing(reference):
        status = "reference_missing"
    elif _is_missing(candidate):
        status = "candidate_missing"
    else:
        ref_norm = _norm_venue(reference)
        cand_norm = _norm_venue(candidate)
        if ref_norm == cand_norm:
            status = "match"
        elif ref_norm and cand_norm and (cand_norm in ref_norm or ref_norm in cand_norm):
            # One is a substring of the other (e.g. "coling 2010 posters" in longer string)
            status = "match"
        elif ref_norm and cand_norm and similarity(ref_norm, cand_norm) >= 0.7:
            # High similarity (handles minor abbreviation differences)
            status = "match"
        else:
            status = "mismatch"
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


_ET_AL_MARKERS = {"others", "et al", "et al.", "and others", "and others."}


def _is_et_al_marker(name: str) -> bool:
    return _norm_text(name) in _ET_AL_MARKERS


def _compare_authors(reference: list[str], candidate: list[str]) -> dict[str, Any]:
    ref = _to_author_list(reference)
    cand = _to_author_list(candidate)

    # Detect "et al." / "Others" in reference and filter them out
    has_et_al = any(_is_et_al_marker(name) for name in ref)
    ref_actual = [name for name in ref if not _is_et_al_marker(name)]

    ref_norm = [_norm_text(name) for name in ref_actual]
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
        if overlap > 0:
            status = "partial_overlap"
        else:
            # Try initial matching (e.g., "s bubeck" vs "bubeck sebastien")
            from .normalize import author_tokens
            initial_matches = 0
            for r_name in ref_norm:
                r_tokens = author_tokens(r_name)
                for c_name in cand_norm:
                    c_tokens = author_tokens(c_name)
                    # Last name must match; first name can be initial
                    if r_tokens and c_tokens and r_tokens & c_tokens:
                        initial_matches += 1
                        break
            overlap = initial_matches
            status = "partial_overlap" if initial_matches > 0 else "mismatch"

        # When citation uses "et al." and all listed authors match a prefix of candidate,
        # treat as match (citation intentionally omitted remaining authors)
        if has_et_al and overlap == len(ref_norm) and len(cand_norm) >= len(ref_norm):
            status = "match"

    return {
        "status": status,
        "reference": ref,
        "candidate": cand,
        "overlap_count": overlap,
        "reference_count": len(ref_actual),
        "candidate_count": len(cand),
        "has_et_al": has_et_al,
    }


def _reference_snapshot(citation: CitationRecord) -> dict[str, Any]:
    return {
        "title": citation.title,
        "authors": list(citation.authors),
        "venue": citation.venue,
        "year": citation.year,
        "pages": citation.pages,
        "publisher": citation.publisher,
        "location": citation.location,
        "doi": citation.doi,
        "arxiv_id": citation.arxiv_id,
        "url": citation.url,
    }


def _candidate_snapshot(candidate: CandidateMatch | None) -> dict[str, Any]:
    if candidate is None:
        return {}
    return {
        "connector": candidate.connector,
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
    # Extract pages/publisher from candidate raw_record (url_direct cite blocks provide these)
    cand_raw = candidate.raw_record if candidate is not None and isinstance(candidate.raw_record, dict) else {}
    candidate_core = {
        "title": candidate_data.get("title", ""),
        "authors": candidate_data.get("authors", []),
        "venue": candidate_data.get("venue", ""),
        "year": candidate_data.get("year"),
        "pages": str(cand_raw.get("pages", "") or ""),
        "publisher": str(cand_raw.get("publisher", "") or ""),
        "location": str(cand_raw.get("location", "") or ""),
        "doi": candidate_data.get("doi", ""),
        "arxiv_id": candidate_data.get("arxiv_id", ""),
        "url": candidate_data.get("url", ""),
    }

    field_status = {
        "title": _compare_scalar(reference["title"], candidate_core["title"], _norm_title),
        "authors": _compare_authors(reference["authors"], candidate_core["authors"]),
        "venue": _compare_venue(reference["venue"], candidate_core["venue"]),
        "year": _compare_year(reference["year"], candidate_core["year"], candidate_version_years),
        "pages": _compare_scalar(reference.get("pages", ""), candidate_core["pages"], _norm_text),
        "publisher": _compare_scalar(reference.get("publisher", ""), candidate_core["publisher"], _norm_text),
        "location": _compare_scalar(reference.get("location", ""), candidate_core["location"], _norm_text),
        "doi": _compare_scalar(reference["doi"], candidate_core["doi"], _norm_identifier),
        "arxiv_id": _compare_scalar(reference["arxiv_id"], candidate_core["arxiv_id"], _norm_arxiv_id),
        "url": _compare_scalar(reference["url"], candidate_core["url"], _norm_url),
    }

    summary_pairs = [f"{name}:{meta['status']}" for name, meta in field_status.items()]
    return {
        "connector": candidate_data.get("connector", ""),
        "conflicts": list(conflicts),
        "reference": reference,
        "candidate": candidate_core,
        "field_status": field_status,
        "summary": ", ".join(summary_pairs),
    }


def _status_in(meta: dict[str, Any], accepted: set[str]) -> bool:
    return str(meta.get("status", "") or "") in accepted


def _has_soft_discrepancies(field_status: dict[str, dict[str, Any]], conflicts: list[str]) -> bool:
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
    for field_name in ("authors", "venue", "doi", "arxiv_id", "pages", "publisher", "location"):
        if field_name in field_status and _status_in(field_status[field_name], review_statuses):
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
    # Core fields: title + authors + year must match
    if field_status["title"]["status"] != "match":
        return False
    if field_status["year"]["status"] != "match":
        return False
    if not _status_in(field_status["authors"], {"match", "reordered_match"}):
        return False
    # Optional fields: venue/doi/arxiv_id — if citation wrote it, must be correct
    for f in ("venue", "doi", "arxiv_id"):
        if not _status_in(field_status[f], {"match", "both_missing", "reference_missing", "candidate_missing"}):
            return False
    # Extended fields: pages/publisher/location — if citation has them, they must match
    # (candidate_missing is OK: citation provided them but this connector didn't return them)
    for f in ("pages", "publisher", "location"):
        if f in field_status:
            if not _status_in(field_status[f], {"match", "both_missing", "reference_missing", "candidate_missing"}):
                return False
    return True


def _candidate_has_no_structured_data(field_status: dict[str, dict[str, Any]]) -> bool:
    """True when candidate returned no useful structured fields (e.g. raw web_search snippet)."""
    return all(
        field_status.get(f, {}).get("status") == "candidate_missing"
        for f in ("title", "authors", "year")
    )


def _evaluate_candidate(
    citation: CitationRecord,
    candidate: CandidateMatch,
    policy: AdjudicationPolicy,
) -> tuple[VerdictLabel, str, list[str], dict[str, Any], bool, bool]:
    """Returns (verdict, reason, conflicts, comparison, should_run_llm_recheck, needs_secondary_verification)."""
    conflicts = list(candidate.conflicts)
    comparison = _build_comparison(citation, candidate, conflicts)
    field_status = comparison["field_status"]
    should_run_llm_recheck = False
    needs_secondary_verification = False

    if _is_direct_valid(field_status, conflicts):
        verdict = VerdictLabel.VALID
        reason = "All core metadata fields match, so the citation is treated as a direct valid match."
    elif field_status["title"]["status"] == "mismatch" or field_status["year"]["status"] == "mismatch":
        verdict = VerdictLabel.FAKE_REFERENCE
        reason = "Title or year mismatches the current candidate."
        should_run_llm_recheck = True
    elif _has_soft_discrepancies(field_status, conflicts):
        verdict = VerdictLabel.FAKE_REFERENCE
        reason = "Field discrepancies default to fake; secondary evidence required to upgrade."
        should_run_llm_recheck = True
        needs_secondary_verification = True
    else:
        verdict = VerdictLabel.FAKE_REFERENCE
        reason = "No positive evidence of match."
        should_run_llm_recheck = True

    return verdict, reason, conflicts, comparison, should_run_llm_recheck, needs_secondary_verification


def adjudicate(
    citation: CitationRecord,
    candidates: list[CandidateMatch],
    evidence: list[EvidenceTrace],
    extraction_quality: ExtractionQuality = ExtractionQuality.UNKNOWN,
    llm_resolver: LLMResolver | None = None,
    policy: AdjudicationPolicy | None = None,
    secondary_verifier: SecondaryVerifierProtocol | None = None,
) -> CitationVerdict:
    policy = policy or AdjudicationPolicy()
    evidence_sources = sorted({trace.connector for trace in evidence if trace.error is None})
    source_errors = [trace for trace in evidence if trace.error]

    if not candidates:
        if source_errors and len(source_errors) == len(evidence):
            verdict_label = VerdictLabel.INSUFFICIENT_EVIDENCE
            reason = "All sources failed or timed out, unable to establish ground truth."
        else:
            verdict_label = VerdictLabel.FAKE_REFERENCE
            reason = "No matching paper found in any database."
        return CitationVerdict(
            citation_id=citation.citation_id,
            verdict=verdict_label,
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

        verdict, reason, conflicts, comparison, should_run_llm_recheck, needs_secondary = _evaluate_candidate(
            citation=citation,
            candidate=candidate,
            policy=policy,
        )

        # When candidate has no structured data (web_search with empty fields),
        # skip secondary verification (nothing to compare) and go straight to LLM.
        # LLM verdict is capped at POTENTIAL_REFERENCE for such candidates.
        no_structured = _candidate_has_no_structured_data(comparison["field_status"])

        sec_evidence: list[DiscrepancyEvidence] = []
        if needs_secondary and secondary_verifier and not no_structured:
            try:
                sec_evidence = secondary_verifier.gather_evidence(
                    citation, candidate, comparison["field_status"], conflicts,
                )
            except Exception:
                sec_evidence = []
            upgraded_verdict, upgrade_reason = _attempt_upgrade(sec_evidence)
            verdict = upgraded_verdict
            reason = upgrade_reason

        # LLM resolver with secondary evidence context
        llm_note = ""
        if llm_resolver and should_run_llm_recheck:
            try:
                review = llm_resolver.review(
                    citation, [candidate], conflicts, verdict,
                    secondary_evidence=sec_evidence or None,
                ) or {}
            except Exception as exc:  # noqa: BLE001 - reviewer failure should not crash adjudication
                review = {"note": f"LLM review failed: {exc}"}
            override = review.get("label_override")
            if override:
                verdict = canonical_verdict_label(override)
            llm_note = str(review.get("note", "")).strip()

        # If candidate has no structured data and no LLM override happened, stay FAKE
        if no_structured and not llm_note:
            verdict = VerdictLabel.FAKE_REFERENCE
            reason = "No structured candidate data and no LLM evidence."

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
                "secondary_evidence": [
                    {"field": ev.field, "type": ev.discrepancy_type,
                     "found": ev.evidence_found, "explanation": ev.explanation}
                    for ev in sec_evidence
                ],
            }
        )

        current = CitationVerdict(
            citation_id=citation.citation_id,
            verdict=verdict,
            evidence_sources=evidence_sources,
            conflicts=conflicts,
            adjudication_reason=reason,
            matched_candidate=candidate_match_to_dict(candidate),
            reference_snapshot=_reference_snapshot(citation),
            comparison=comparison,
            candidate_evaluations=list(candidate_evaluations),
            llm_recheck_reason=llm_note,
            needs_human_review=(verdict != VerdictLabel.VALID),
            extraction_quality=extraction_quality,
            secondary_evidence=sec_evidence,
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
        verdict=VerdictLabel.FAKE_REFERENCE,
        evidence_sources=evidence_sources,
        conflicts=list(best.conflicts),
        adjudication_reason="No candidate provided sufficient positive evidence.",
        matched_candidate=candidate_match_to_dict(best),
        reference_snapshot=_reference_snapshot(citation),
        comparison=_build_comparison(citation, best, list(best.conflicts)),
        candidate_evaluations=list(candidate_evaluations),
        llm_recheck_reason="",
        needs_human_review=True,
        extraction_quality=extraction_quality,
    )
