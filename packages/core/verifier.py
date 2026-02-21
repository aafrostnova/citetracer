from __future__ import annotations

import sys
from dataclasses import dataclass

from packages.connectors.base import ConnectorOrchestrator, ConnectorResult

from .adjudicate import AdjudicationPolicy, LLMResolver, adjudicate
from .matching import rank_candidates
from .models import CheckReport, CitationRecord, EvidenceTrace, ExtractionQuality
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
    ) -> None:
        self.orchestrator = orchestrator
        self.policy = policy or AdjudicationPolicy()
        self.llm_resolver = llm_resolver
        self.config = config or VerifyConfig()

    def _result_to_trace(self, citation: CitationRecord, result: ConnectorResult) -> EvidenceTrace:
        return EvidenceTrace(
            connector=result.connector,
            query={
                "title": citation.title,
                "year": citation.year,
                "doi": citation.doi,
                "arxiv_id": citation.arxiv_id,
            },
            latency_ms=result.latency_ms,
            cache_hit=result.cache_hit,
            source_health=result.source_health,
            candidates_count=len(result.records),
            error=result.error,
        )

    def verify_citation(self, citation: CitationRecord, extraction_quality: ExtractionQuality | None = None):
        extraction_quality = extraction_quality or self.config.default_extraction_quality
        connector_results = self.orchestrator.query(citation)
        records = {result.connector: result.records for result in connector_results}
        candidates = rank_candidates(citation, records)
        evidence = [self._result_to_trace(citation, result) for result in connector_results]
        return adjudicate(
            citation=citation,
            candidates=candidates,
            evidence=evidence,
            extraction_quality=extraction_quality,
            llm_resolver=self.llm_resolver,
            policy=self.policy,
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
            verdicts.append(
                self.verify_citation(
                    citation,
                    extraction_quality=extraction_quality_map.get(citation.citation_id, self.config.default_extraction_quality),
                )
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
