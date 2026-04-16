from __future__ import annotations

import sys
from dataclasses import dataclass, replace

from packages.connectors.base import ConnectorOrchestrator, ConnectorResult

from .adjudicate import AdjudicationPolicy, LLMResolver, SecondaryVerifierProtocol, adjudicate, canonical_verdict_label
from .agents import ExistenceJudge, StructuredJudge, multi_agent_adjudicate
from .cascading_agents import (
    ValidAgentProtocol, PotentialAgentProtocol, HallucinatedAgentProtocol,
    cascading_adjudicate,
    phase0_url_check,
    phase0_decide_verdict,
    downgrade_non_academic_fake,
)
from .matching import CandidateMatch, build_candidate_match, collect_candidates
from .models import CheckReport, CitationRecord, CitationVerdict, EvidenceTrace, ExtractionQuality, VerdictLabel
from .normalize import extract_identifier, normalize_title, similarity
from .report import build_summary, compute_risk_score


@dataclass
class VerifyConfig:
    report_version: str = "1.0"
    default_extraction_quality: ExtractionQuality = ExtractionQuality.UNKNOWN


class CitationVerifier:
    def __init__(
        self,
        orchestrator: ConnectorOrchestrator,
        policy: AdjudicationPolicy | None = None,
        llm_resolver: LLMResolver | None = None,
        config: VerifyConfig | None = None,
        secondary_verifier: SecondaryVerifierProtocol | None = None,
        structured_judge: StructuredJudge | None = None,
        existence_judge: ExistenceJudge | None = None,
        # Cascading agents
        valid_agent: ValidAgentProtocol | None = None,
        potential_agent: PotentialAgentProtocol | None = None,
        hallucinated_agent: HallucinatedAgentProtocol | None = None,
        extractor_agent: Any = None,
        verified_ref_cache: Any = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.policy = policy or AdjudicationPolicy()
        self.llm_resolver = llm_resolver
        self.config = config or VerifyConfig()
        self.secondary_verifier = secondary_verifier
        self.structured_judge = structured_judge
        self.existence_judge = existence_judge
        self.valid_agent = valid_agent
        self.potential_agent = potential_agent
        self.hallucinated_agent = hallucinated_agent
        self.extractor_agent = extractor_agent
        self.verified_ref_cache = verified_ref_cache

    _CACHEABLE_TAXONOMY = {"R1", "R3"}

    def _cache_if_valid(self, citation: CitationRecord, verdict: CitationVerdict) -> None:
        """Cache the verified reference if verdict is VALID with R1/R3."""
        if self.verified_ref_cache is None:
            return
        taxonomy = set(verdict.taxonomy_subtype or [])
        if verdict.verdict == VerdictLabel.VALID and taxonomy & self._CACHEABLE_TAXONOMY:
            from .models import citation_verdict_to_dict
            self.verified_ref_cache.set(citation, citation_verdict_to_dict(verdict))

    def _result_to_trace(self, citation: CitationRecord, result: ConnectorResult) -> EvidenceTrace:
        return EvidenceTrace(
            connector=result.connector,
            query={
                "title": citation.title,
                "year": citation.year,
                "doi": citation.doi,
                "arxiv_id": citation.arxiv_id,
                "url": citation.url,
            },
            latency_ms=result.latency_ms,
            cache_hit=result.cache_hit,
            source_health=result.source_health,
            candidates_count=len(result.records),
            error=result.error,
        )

    @staticmethod
    def _enrich_citation_from_url(citation: CitationRecord) -> CitationRecord:
        url = str(citation.url or "").strip()
        # Step 1: extract doi/arxiv_id from URL
        if url:
            extracted_doi, extracted_arxiv_id = extract_identifier(url)
        else:
            extracted_doi, extracted_arxiv_id = "", ""

        new_doi = (citation.doi or "").strip().lower() or extracted_doi
        new_arxiv_id = (citation.arxiv_id or "").strip().lower() or extracted_arxiv_id

        # Step 2: if citation has arxiv_id but no venue, infer venue="arXiv"
        # (avoids LLM HallucinatedAgent flagging empty venue as H4 for arxiv papers)
        new_venue = citation.venue
        if new_arxiv_id and not (citation.venue or "").strip():
            new_venue = "arXiv"

        # Skip rebuilding if nothing actually changed
        unchanged = (
            new_doi == (citation.doi or "").strip().lower()
            and new_arxiv_id == (citation.arxiv_id or "").strip().lower()
            and new_venue == citation.venue
        )
        if unchanged:
            return citation

        return replace(
            citation,
            doi=new_doi,
            arxiv_id=new_arxiv_id,
            venue=new_venue,
        )

    _WEB_CONNECTORS = {"google_search", "web_search"}

    # Titles that indicate a redirect/CAPTCHA/error page, not a real resolved title
    _JUNK_TITLES = {
        "redirecting", "just a moment", "not found", "404", "access denied",
        "page not found", "error", "captcha", "verify you are human",
        "please wait", "loading", "forbidden", "unauthorized",
    }

    def _check_identifier_mismatch(
        self, citation: CitationRecord, url_direct_records: list[dict],
    ) -> CitationVerdict | None:
        """When citation provides DOI/arXiv/URL and url_direct resolved them,
        check if the resolved title matches. If not, ask LLM to judge whether
        the resolved content and the citation refer to the same work."""
        if not citation.title:
            return None
        has_identifier = bool(
            (citation.doi or "").strip()
            or (citation.arxiv_id or "").strip()
            or (citation.url or "").strip()
        )
        if not has_identifier or not url_direct_records:
            return None

        citation_title_norm = normalize_title(citation.title)

        # Filter junk titles and check for match
        valid_resolved = []
        for r in url_direct_records:
            resolved_title = str(r.get("title", "") or "").strip()
            if not resolved_title or resolved_title.lower().strip() in self._JUNK_TITLES:
                continue
            resolved_title_norm = normalize_title(resolved_title)
            if not resolved_title_norm:
                continue
            valid_resolved.append(r)
            if similarity(citation_title_norm, resolved_title_norm) >= 0.75:
                return None  # Title matches, no mismatch

        # All junk → skip
        if not valid_resolved:
            return None

        # Title doesn't match — ask LLM to judge
        if self.llm_resolver:
            best = valid_resolved[0]
            resolved_info = (
                f"The citation's identifier resolved to a page with:\n"
                f"  Title: {best.get('title', '')}\n"
                f"  Authors: {best.get('authors', [])}\n"
                f"  Venue: {best.get('venue', '')}\n"
                f"  Year: {best.get('year')}\n"
                f"  URL: {best.get('_resolved_from', '')}\n"
            )
            # Build a minimal CandidateMatch for LLM review
            from .matching import build_candidate_match
            cand = build_candidate_match(citation, "url_direct", best)
            try:
                review = self.llm_resolver.review(
                    citation, [cand], ["identifier_title_mismatch"],
                    VerdictLabel.FAKE_REFERENCE,
                ) or {}
            except Exception:
                review = {}

            override = review.get("label_override")
            llm_note = str(review.get("note", "")).strip()

            if override:
                verdict = canonical_verdict_label(override)
            else:
                verdict = VerdictLabel.FAKE_REFERENCE

            reason = (
                f"Citation identifier resolved to: '{best.get('title', '')[:80]}' "
                f"(from {best.get('_resolved_from', '')}). "
            )
            if llm_note:
                reason += f"LLM review: {llm_note}"

            if verdict == VerdictLabel.VALID:
                return None  # LLM says it's the same work — continue normal flow

            return CitationVerdict(
                citation_id=citation.citation_id,
                verdict=verdict,
                evidence_sources=["url_direct"],
                conflicts=["identifier_title_mismatch"],
                adjudication_reason=reason,
                reference_snapshot={
                    "title": citation.title,
                    "authors": list(citation.authors),
                    "venue": citation.venue,
                    "year": citation.year,
                    "doi": citation.doi,
                    "arxiv_id": citation.arxiv_id,
                    "url": citation.url,
                },
                llm_recheck_reason=llm_note,
                needs_human_review=True,
                extraction_quality=ExtractionQuality.UNKNOWN,
            )

        # No LLM available — skip, let normal flow handle it
        return None


    def verify_citation(
        self,
        citation: CitationRecord,
        extraction_quality: ExtractionQuality | None = None,
        source_paper_title: str = "",
    ):
        # Check verified reference cache — re-validate cached candidate against current citation
        if self.verified_ref_cache is not None:
            cached = self.verified_ref_cache.get(citation)
            if cached is not None and cached.get("matched_candidate"):
                from .adjudicate import _build_comparison, _is_direct_valid, _has_soft_discrepancies
                from .agents import compare_fields
                from .models import citation_verdict_from_dict

                candidate = build_candidate_match(citation, "cache", cached["matched_candidate"])
                field_result = compare_fields(
                    citation, candidate,
                    _build_comparison, _is_direct_valid, _has_soft_discrepancies,
                )
                if field_result.signal == "direct_valid":
                    return citation_verdict_from_dict(cached, citation.citation_id)

        citation = self._enrich_citation_from_url(citation)
        extraction_quality = extraction_quality or self.config.default_extraction_quality

        # Phase 0: URL/DOI direct verification (BEFORE orchestrator query).
        # Resolves citation.url and citation.doi via URLDirectConnector + Tavily fallback.
        # Returns status signal AND any fetched records to merge into candidates.
        url_match_status, url_issues, url_type_triggered, phase0_records = phase0_url_check(citation)

        # Phase 0 early-exit:
        # - Scholar URL (arxiv/doi) mismatch or not_found → FAKE_REFERENCE
        # - Other URL mismatch or not_found → POTENTIAL_REFERENCE / P3
        phase0_verdict = phase0_decide_verdict(
            citation, url_match_status, url_issues, url_type_triggered,
            fetched_records=phase0_records,
        )
        if phase0_verdict is not None:
            return downgrade_non_academic_fake(citation, phase0_verdict)

        # If identifiers resolve to a clearly different paper → immediate FAKE
        # (but downgrade to P3 if the citation is non-academic)
        id_mismatch = self._check_identifier_mismatch(citation, phase0_records)
        if id_mismatch:
            return downgrade_non_academic_fake(citation, id_mismatch)

        # Build Phase 0 evaluation record for the report (always included)
        _phase0_eval: dict | None = None
        if phase0_records or url_match_status:
            _phase0_eval = {
                "connector": "url_direct (Phase 0)",
                "verdict": f"PHASE0_{url_match_status.upper()}" if url_match_status else "PHASE0_SKIPPED",
                "url_type": url_type_triggered,
                "issues": url_issues,
                "candidates": phase0_records,
                "reason": (
                    f"Phase 0: url_match_status={url_match_status}, "
                    f"url_type={url_type_triggered}, issues={url_issues}"
                ),
            }

        # Query all other connectors (url_direct is NOT in the orchestrator anymore)
        connector_results = self.orchestrator.query(citation, max_connectors=None)
        records = {result.connector: result.records for result in connector_results}

        # Add Phase 0 url_direct records as a synthetic source so they participate
        # in candidate matching alongside the structured connectors.
        if phase0_records:
            existing = records.get("url_direct", [])
            existing_urls = {str(r.get("url", "")) for r in existing}
            for r in phase0_records:
                if str(r.get("url", "")) not in existing_urls:
                    existing.append(r)
            records["url_direct"] = existing

        candidates = collect_candidates(citation, records)

        # Note: web_search candidate enrichment via url_direct is removed.
        # Phase 0 already handled the citation URL. Empty web_search candidates
        # are handled by ExtractorAgent inside cascading_adjudicate.

        evidence = [self._result_to_trace(citation, result) for result in connector_results]

        # Use cascading 3-agent flow if configured (preferred)
        if self.valid_agent:
            # Resolve tavily key for web_search URL enrichment
            import os as _os
            _tavily_key = _os.getenv("TAVILY_API_KEY", "")
            if not _tavily_key:
                try:
                    from apps.pdf_checker.config import load_pdf_checker_config as _load_cfg
                    _tavily_key = (_load_cfg().connectors.tavily_api_key or "").strip()
                except Exception:
                    pass

            verdict = cascading_adjudicate(
                citation=citation,
                candidates=candidates,
                evidence_traces=evidence,
                extraction_quality=extraction_quality,
                valid_agent=self.valid_agent,
                potential_agent=self.potential_agent,
                hallucinated_agent=self.hallucinated_agent,
                secondary_verifier=self.secondary_verifier,
                extractor_agent=self.extractor_agent,
                source_paper_title=source_paper_title,
                url_match_status=url_match_status,
                url_issues=url_issues,
                url_type_triggered=url_type_triggered,
                tavily_api_key=_tavily_key,
            )
            # Final downgrade: non-academic FAKE → P3 (R packages, GitHub, blogs)
            verdict = downgrade_non_academic_fake(citation, verdict)
            # Prepend Phase 0 evaluation to candidate_evaluations for the report
            if _phase0_eval:
                verdict.candidate_evaluations.insert(0, _phase0_eval)
            self._cache_if_valid(citation, verdict)
            return verdict

        # Use parallel multi-agent flow if judges are configured
        if self.structured_judge or self.existence_judge:
            verdict = multi_agent_adjudicate(
                citation=citation,
                candidates=candidates,
                evidence_traces=evidence,
                extraction_quality=extraction_quality,
                structured_judge=self.structured_judge,
                existence_judge=self.existence_judge,
                secondary_verifier=self.secondary_verifier,
                source_paper_title=source_paper_title,
            )
            if _phase0_eval:
                verdict.candidate_evaluations.insert(0, _phase0_eval)
            self._cache_if_valid(citation, verdict)
            return verdict

        # Fallback: legacy single-LLM adjudication
        verdict = adjudicate(
            citation=citation,
            candidates=candidates,
            evidence=evidence,
            extraction_quality=extraction_quality,
            llm_resolver=self.llm_resolver,
            policy=self.policy,
            secondary_verifier=self.secondary_verifier,
        )
        if _phase0_eval:
            verdict.candidate_evaluations.insert(0, _phase0_eval)
        self._cache_if_valid(citation, verdict)
        return verdict

    def verify_paper(
        self,
        paper_id: str,
        pipeline_type: str,
        citations: list[CitationRecord],
        extraction_quality_map: dict[str, ExtractionQuality] | None = None,
        metadata: dict | None = None,
        show_progress: bool = False,
    ) -> CheckReport:
        extraction_quality_map = extraction_quality_map or {}
        verdicts = []
        total = len(citations)
        for idx, citation in enumerate(citations, start=1):
            if show_progress:
                print(
                    f"[verify] citation {idx}/{total} start id={citation.citation_id} title={citation.title[:120]!r}",
                    flush=True,
                )
            verdict = self.verify_citation(
                citation,
                extraction_quality=extraction_quality_map.get(citation.citation_id, self.config.default_extraction_quality),
                source_paper_title=paper_id,
            )
            verdicts.append(verdict)
            if show_progress:
                print(
                    f"[verify] citation {idx}/{total} result id={citation.citation_id} "
                    f"verdict={verdict.verdict.value} reason={verdict.adjudication_reason}",
                    flush=True,
                )
            if show_progress and total > 0:
                width = 28
                filled = min(width, int(width * idx / total))
                bar = "#" * filled + "-" * (width - filled)
                sys.stdout.write(f"\r[verify] [{bar}] {idx}/{total}")
                sys.stdout.flush()
        if show_progress and total > 0:
            sys.stdout.write("\n")
            sys.stdout.flush()
        summary = build_summary(verdicts)
        risk_score = compute_risk_score(verdicts)
        requires_human_review = any(verdict.needs_human_review for verdict in verdicts)
        return CheckReport(
            report_version=self.config.report_version,
            paper_id=paper_id,
            pipeline_type=pipeline_type,
            citations=verdicts,
            summary=summary,
            risk_score=risk_score,
            requires_human_review=requires_human_review,
            metadata=metadata or {},
        )
