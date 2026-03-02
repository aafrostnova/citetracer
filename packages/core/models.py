from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class VerdictLabel(str, Enum):
    VALID = "VALID"
    POTENTIAL_REFERENCE = "POTENTIAL_REFERENCE"
    FAKE_REFERENCE = "FAKE_REFERENCE"
    # Legacy labels kept for backward compatibility with old datasets/reports.
    FLAWED_CITATION = "FLAWED_CITATION"
    SUSPECTED_HALLUCINATION = "SUSPECTED_HALLUCINATION"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class ExtractionQuality(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"


@dataclass
class CitationRecord:
    citation_id: str
    raw_text: str = ""
    title: str = ""
    authors: list[str] = field(default_factory=list)
    venue: str = ""
    year: int | None = None
    doi: str = ""
    arxiv_id: str = ""
    url: str = ""
    source_span: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    parsed_fields: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateMatch:
    connector: str
    score: float
    title: str = ""
    authors: list[str] = field(default_factory=list)
    venue: str = ""
    year: int | None = None
    doi: str = ""
    arxiv_id: str = ""
    url: str = ""
    matched_fields: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    raw_record: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceTrace:
    connector: str
    query: dict[str, Any]
    latency_ms: float
    cache_hit: bool
    source_health: float
    candidates_count: int
    error: str | None = None


@dataclass
class CitationVerdict:
    citation_id: str
    verdict: VerdictLabel
    confidence: float
    evidence_sources: list[str]
    conflicts: list[str]
    adjudication_reason: str
    matched_candidate: dict[str, Any] | None = None
    needs_human_review: bool = False
    extraction_quality: ExtractionQuality = ExtractionQuality.UNKNOWN


@dataclass
class CheckReport:
    report_version: str
    paper_id: str
    pipeline_type: str
    citations: list[CitationVerdict]
    summary: dict[str, Any]
    risk_score: float
    requires_human_review: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_version": self.report_version,
            "paper_id": self.paper_id,
            "pipeline_type": self.pipeline_type,
            "citations": [citation_verdict_to_dict(c) for c in self.citations],
            "summary": self.summary,
            "risk_score": self.risk_score,
            "requires_human_review": self.requires_human_review,
            "metadata": self.metadata,
        }


def citation_record_to_query(record: CitationRecord) -> dict[str, Any]:
    return {
        "title": record.title,
        "authors": record.authors,
        "venue": record.venue,
        "year": record.year,
        "doi": record.doi,
        "arxiv_id": record.arxiv_id,
    }


def candidate_match_to_dict(candidate: CandidateMatch) -> dict[str, Any]:
    payload = asdict(candidate)
    payload["score"] = round(float(candidate.score), 4)
    return payload


def citation_verdict_to_dict(verdict: CitationVerdict) -> dict[str, Any]:
    payload = asdict(verdict)
    payload["verdict"] = canonical_verdict_label(verdict.verdict).value
    payload["extraction_quality"] = verdict.extraction_quality.value
    payload["confidence"] = round(float(verdict.confidence), 4)
    if verdict.matched_candidate is not None:
        payload["matched_candidate"] = verdict.matched_candidate
    return payload


def canonical_verdict_label(label: VerdictLabel | str) -> VerdictLabel:
    raw = str(getattr(label, "value", label)).strip().upper()
    if raw == VerdictLabel.FLAWED_CITATION.value:
        return VerdictLabel.POTENTIAL_REFERENCE
    if raw == VerdictLabel.SUSPECTED_HALLUCINATION.value:
        return VerdictLabel.FAKE_REFERENCE
    if raw == VerdictLabel.POTENTIAL_REFERENCE.value:
        return VerdictLabel.POTENTIAL_REFERENCE
    if raw == VerdictLabel.FAKE_REFERENCE.value:
        return VerdictLabel.FAKE_REFERENCE
    if raw == VerdictLabel.VALID.value:
        return VerdictLabel.VALID
    return VerdictLabel.INSUFFICIENT_EVIDENCE
