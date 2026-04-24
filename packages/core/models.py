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
    volume: str = ""
    pages: str = ""
    publisher: str = ""
    location: str = ""
    source_span: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    parsed_fields: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateMatch:
    connector: str
    title: str = ""
    authors: list[str] = field(default_factory=list)
    venue: str = ""
    year: int | None = None
    doi: str = ""
    arxiv_id: str = ""
    url: str = ""
    volume: str = ""
    pages: str = ""
    publisher: str = ""
    matched_fields: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    raw_record: dict[str, Any] = field(default_factory=dict)


@dataclass
class FieldComparisonResult:
    """Output of the FieldComparator agent."""
    field_status: dict[str, dict[str, Any]] = field(default_factory=dict)
    signal: str = ""  # "direct_valid" | "hard_mismatch" | "soft_discrepancy" | "no_candidate"
    hard_mismatch_fields: list[str] = field(default_factory=list)
    soft_discrepancy_fields: list[str] = field(default_factory=list)
    has_et_al: bool = False
    comparison: dict[str, Any] = field(default_factory=dict)
    conflicts: list[str] = field(default_factory=list)


@dataclass
class LLMJudgment:
    """Output of an LLM judge agent."""
    verdict: VerdictLabel = VerdictLabel.FAKE_REFERENCE
    taxonomy: list[str] = field(default_factory=list)
    confidence: float = 0.0
    note: str = ""
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidAgentResult:
    """Output of the ValidAgent in cascading pipeline."""
    is_valid: bool = False
    taxonomy: list[str] = field(default_factory=list)
    hint: str = ""               # "likely_potential" | "likely_hallucinated"
    issues: list[str] = field(default_factory=list)
    reason: str = ""
    field_result: Any = None     # FieldComparisonResult
    candidate: Any = None        # CandidateMatch
    raw_llm_response: dict[str, Any] = field(default_factory=dict)


@dataclass
class DownstreamAgentResult:
    """Output of PotentialAgent or HallucinatedAgent."""
    label: VerdictLabel = VerdictLabel.FAKE_REFERENCE
    taxonomy: list[str] = field(default_factory=list)
    reason: str = ""
    escalate_to: str = ""        # "" | "hallucinated" | "potential"
    raw_llm_response: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuthorVariantResult:
    """Output of the AuthorVariantClassifier. Authoritative classification of
    the relationship between citation and candidate author lists, used by
    ValidAgent and R4 coverage logic.

    overall:
      - "exact"       — byte-identical after case normalization
      - "r2_initial"  — single-letter initial expansion only (e.g. "G. Hao" ≡ "Gao Hao")
      - "p1_variant"  — same people, but name form differs by nickname / multi-letter
                        truncation / transliteration / typo / last-first reorder.
                        P1 means same identity, BUT the names are not rule-derivably equal.
      - "h2_error"    — genuinely different people (added/deleted/reordered/fabricated).
    """
    overall: str = "exact"
    reason: str = ""
    per_pair: list[dict[str, Any]] = field(default_factory=list)
    raw_llm_response: dict[str, Any] = field(default_factory=dict)


@dataclass
class VenueEquivalenceResult:
    """Output of the VenueEquivalenceClassifier. Authoritative classification
    of whether two organization names (venue or publisher) refer to the same
    entity. Used by ValidAgent (venue/publisher judgment) and by downstream
    consumers via field_status["<field>"]["variant_type"].

    overall:
      - "exact"     — byte-identical after case/punctuation normalization
      - "alias"     — same organization, expressed differently (acronym ↔ full
                      name, proceedings prefix, dot-abbreviation, alternate
                      spelling, etc.). Treat as match.
      - "different" — genuinely different venues/publishers.
    """
    overall: str = "exact"
    reason: str = ""
    raw_llm_response: dict[str, Any] = field(default_factory=dict)


@dataclass
class FieldClassificationResult:
    """Unified output of the FieldClassifier — produces verdicts for authors,
    venue, and publisher in a single LLM call.

    Individual field results carry the full context (overall + reason) so
    downstream consumers (ValidAgent prompt, R4 coverage, field_status
    stamping) can use each verdict independently.
    """
    authors: AuthorVariantResult = field(default_factory=AuthorVariantResult)
    venue: VenueEquivalenceResult = field(default_factory=VenueEquivalenceResult)
    publisher: VenueEquivalenceResult = field(default_factory=VenueEquivalenceResult)
    raw_llm_response: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscrepancyEvidence:
    """Evidence gathered to explain a single field discrepancy during secondary verification."""
    field: str
    discrepancy_type: str
    evidence_found: bool
    explanation: str
    evidence_sources: list[str] = field(default_factory=list)
    raw_evidence: dict[str, Any] = field(default_factory=dict)


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
    evidence_sources: list[str]
    conflicts: list[str]
    adjudication_reason: str
    matched_candidate: dict[str, Any] | None = None
    reference_snapshot: dict[str, Any] = field(default_factory=dict)
    comparison: dict[str, Any] = field(default_factory=dict)
    candidate_evaluations: list[dict[str, Any]] = field(default_factory=list)
    llm_recheck_reason: str = ""
    taxonomy_subtype: list[str] = field(default_factory=list)
    needs_human_review: bool = False
    extraction_quality: ExtractionQuality = ExtractionQuality.UNKNOWN
    secondary_evidence: list[DiscrepancyEvidence] = field(default_factory=list)


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
    return asdict(candidate)


def citation_verdict_to_dict(verdict: CitationVerdict) -> dict[str, Any]:
    payload = asdict(verdict)
    payload["verdict"] = canonical_verdict_label(verdict.verdict).value
    payload["extraction_quality"] = verdict.extraction_quality.value
    if verdict.matched_candidate is not None:
        payload["matched_candidate"] = verdict.matched_candidate
    return payload


def citation_verdict_from_dict(data: dict[str, Any], citation_id: str = "") -> CitationVerdict:
    """Reconstruct a CitationVerdict from a cached dict."""
    return CitationVerdict(
        citation_id=citation_id or data.get("citation_id", ""),
        verdict=canonical_verdict_label(data.get("verdict", "VALID")),
        evidence_sources=data.get("evidence_sources", []),
        conflicts=data.get("conflicts", []),
        adjudication_reason=data.get("adjudication_reason", ""),
        matched_candidate=data.get("matched_candidate"),
        reference_snapshot=data.get("reference_snapshot", {}),
        comparison=data.get("comparison", {}),
        candidate_evaluations=data.get("candidate_evaluations", []),
        llm_recheck_reason=data.get("llm_recheck_reason", ""),
        taxonomy_subtype=data.get("taxonomy_subtype", []),
        needs_human_review=data.get("needs_human_review", False),
        extraction_quality=ExtractionQuality(data.get("extraction_quality", "UNKNOWN")),
        secondary_evidence=[],
    )


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
