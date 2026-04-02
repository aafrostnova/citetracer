from __future__ import annotations

import sys
from dataclasses import dataclass, replace

from packages.connectors.base import ConnectorOrchestrator, ConnectorResult

from dataclasses import replace as _replace

from packages.connectors.base import RequestPolicy
from packages.connectors.crossref import CrossrefConnector
from packages.connectors.url_direct import URLDirectConnector

from .adjudicate import AdjudicationPolicy, LLMResolver, SecondaryVerifierProtocol, adjudicate, canonical_verdict_label
from .matching import CandidateMatch, collect_candidates
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
    ) -> None:
        self.orchestrator = orchestrator
        self.policy = policy or AdjudicationPolicy()
        self.llm_resolver = llm_resolver
        self.config = config or VerifyConfig()
        self.secondary_verifier = secondary_verifier

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
        if not url:
            return citation
        extracted_doi, extracted_arxiv_id = extract_identifier(url)
        if extracted_doi == (citation.doi or "").strip().lower() and extracted_arxiv_id == (citation.arxiv_id or "").strip().lower():
            return citation
        return replace(
            citation,
            doi=(citation.doi or "").strip().lower() or extracted_doi,
            arxiv_id=(citation.arxiv_id or "").strip().lower() or extracted_arxiv_id,
        )

    _WEB_CONNECTORS = {"google_search", "web_search"}

    @staticmethod
    def _build_resolve_urls(citation: CitationRecord, candidate_url: str) -> list[str]:
        """Build prioritized list of URLs to try for web_search enrichment.

        Priority: citation arXiv → citation URL → web_search result URL.
        (DOI is handled separately via Crossref API, not url_direct.)
        """
        urls: list[str] = []
        arxiv_id = (citation.arxiv_id or "").strip()
        if arxiv_id:
            urls.append(f"https://arxiv.org/abs/{arxiv_id}")
        citation_url = (citation.url or "").strip()
        if citation_url:
            urls.append(citation_url)
        if candidate_url and candidate_url not in urls:
            urls.append(candidate_url)
        return urls

    def _enrich_web_candidates_via_url(
        self, citation: CitationRecord, candidates: list[CandidateMatch],
    ) -> list[CandidateMatch]:
        """For web_search candidates with empty structured fields, resolve URLs via url_direct.

        Tries citation's own DOI/arXiv/URL first, then falls back to the web_search result URL.
        """
        url_direct = self._find_url_direct_connector() or URLDirectConnector()
        policy = RequestPolicy(timeout_s=10)
        enriched: list[CandidateMatch] = []
        for cand in candidates:
            if cand.connector not in self._WEB_CONNECTORS or cand.title:
                enriched.append(cand)
                continue

            resolve_urls = self._build_resolve_urls(citation, cand.url)
            for try_url in resolve_urls:
                try:
                    probe = CitationRecord(citation_id="_enrich", url=try_url)
                    results = url_direct.search(probe, policy)
                    if not results:
                        continue
                    r = results[0]
                    new_title = str(r.get("title", "") or "").strip()
                    if not new_title:
                        continue
                    new_authors = r.get("authors") or []
                    new_venue = str(r.get("venue", "") or "").strip()
                    raw_year = r.get("year")
                    new_year = int(raw_year) if raw_year is not None else None
                    new_doi = str(r.get("doi", "") or "").strip()
                    new_arxiv_id = str(r.get("arxiv_id", "") or "").strip()
                    new_pages = str(r.get("pages", "") or "").strip()
                    new_publisher = str(r.get("publisher", "") or "").strip()
                    updated_raw = dict(cand.raw_record)
                    updated_raw["url_direct_enrichment"] = r
                    updated_raw["url_direct_enrichment_source"] = try_url
                    cand = _replace(
                        cand,
                        title=new_title,
                        authors=new_authors or cand.authors,
                        venue=new_venue or cand.venue,
                        year=new_year if new_year is not None else cand.year,
                        doi=new_doi or cand.doi,
                        arxiv_id=new_arxiv_id or cand.arxiv_id,
                        raw_record=updated_raw,
                    )
                    break  # Successfully enriched, stop trying URLs
                except Exception:
                    continue
            enriched.append(cand)
        return enriched

    def _find_url_direct_connector(self) -> URLDirectConnector | None:
        """Get the url_direct connector from the orchestrator (shares its cache)."""
        for conn in getattr(self.orchestrator, "connectors", []):
            if isinstance(conn, URLDirectConnector):
                return conn
        return None

    def _resolve_doi_via_crossref(self, doi: str) -> dict | None:
        """Resolve a DOI via Crossref API (no JS rendering needed)."""
        crossref = CrossrefConnector()
        policy = RequestPolicy(timeout_s=10)
        try:
            payload = crossref._request_json(
                f"https://api.crossref.org/works/{doi}",
                {},
                policy,
            )
            item = payload.get("message", {})
            if not item:
                return None
            record = crossref._normalize_item(item)
            record["_resolved_from"] = f"crossref_api:{doi}"
            return record
        except Exception:
            return None

    def _resolve_citation_urls_direct(self, citation: CitationRecord) -> list[dict]:
        """Resolve citation's DOI/arXiv/URL as highest-priority candidates.

        DOI → Crossref API (reliable, no JS issues).
        arXiv/URL → url_direct (HTML meta tag extraction).
        """
        results: list[dict] = []
        seen_titles: set[str] = set()

        # DOI: use Crossref API directly (avoids JS redirect issues with doi.org)
        doi = (citation.doi or "").strip()
        if doi:
            record = self._resolve_doi_via_crossref(doi)
            if record:
                title = str(record.get("title", "") or "").strip().lower()
                if title and title not in seen_titles:
                    results.append(record)
                    seen_titles.add(title)

        # arXiv / URL: use url_direct
        url_direct = self._find_url_direct_connector()
        if url_direct:
            policy = RequestPolicy(timeout_s=10)
            arxiv_id = (citation.arxiv_id or "").strip()
            citation_url = (citation.url or "").strip()
            urls_to_try: list[str] = []
            if arxiv_id:
                urls_to_try.append(f"https://arxiv.org/abs/{arxiv_id}")
            if citation_url:
                urls_to_try.append(citation_url)
            for try_url in urls_to_try:
                try:
                    probe = CitationRecord(citation_id="_resolve", url=try_url)
                    fetched = url_direct.search(probe, policy)
                    for r in fetched:
                        title = str(r.get("title", "") or "").strip().lower()
                        if title and title not in seen_titles:
                            r["_resolved_from"] = try_url
                            results.append(r)
                            seen_titles.add(title)
                except Exception:
                    continue

        return results

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
        citation = self._enrich_citation_from_url(citation)
        extraction_quality = extraction_quality or self.config.default_extraction_quality

        # Step 1: url_direct is highest priority — resolve citation's own DOI/arXiv/URL
        url_direct_records = self._resolve_citation_urls_direct(citation)

        # Step 1.5: If identifiers resolve to a different paper → immediate FAKE
        id_mismatch = self._check_identifier_mismatch(citation, url_direct_records)
        if id_mismatch:
            return id_mismatch

        # Step 2: Query all other connectors
        connector_results = self.orchestrator.query(citation, max_connectors=None)
        records = {result.connector: result.records for result in connector_results}

        # Prepend url_direct results (highest priority)
        if url_direct_records:
            existing = records.get("url_direct", [])
            existing_urls = {str(r.get("url", "")) for r in existing}
            for r in url_direct_records:
                if str(r.get("url", "")) not in existing_urls:
                    existing.append(r)
            records["url_direct"] = existing

        candidates = collect_candidates(citation, records)

        # Step 3: Enrich web_search candidates that still have empty fields
        candidates = self._enrich_web_candidates_via_url(citation, candidates)

        evidence = [self._result_to_trace(citation, result) for result in connector_results]
        return adjudicate(
            citation=citation,
            candidates=candidates,
            evidence=evidence,
            extraction_quality=extraction_quality,
            llm_resolver=self.llm_resolver,
            policy=self.policy,
            secondary_verifier=self.secondary_verifier,
        )

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
