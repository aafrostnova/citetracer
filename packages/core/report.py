from __future__ import annotations

from collections import Counter

from .models import CheckReport, CitationVerdict, VerdictLabel, canonical_verdict_label


def build_summary(verdicts: list[CitationVerdict]) -> dict[str, int]:
    counter = Counter(canonical_verdict_label(v.verdict).value for v in verdicts)
    return {
        "total_citations": len(verdicts),
        "valid": counter[VerdictLabel.VALID.value],
        "potential_reference": counter[VerdictLabel.POTENTIAL_REFERENCE.value],
        "fake_reference": counter[VerdictLabel.FAKE_REFERENCE.value],
        # Legacy aliases for backward compatibility.
        "flawed_citation": counter[VerdictLabel.POTENTIAL_REFERENCE.value],
        "suspected_hallucination": counter[VerdictLabel.FAKE_REFERENCE.value],
        "insufficient_evidence": counter[VerdictLabel.INSUFFICIENT_EVIDENCE.value],
        "needs_human_review": sum(1 for verdict in verdicts if verdict.needs_human_review),
    }


def compute_risk_score(verdicts: list[CitationVerdict]) -> float:
    if not verdicts:
        return 0.0
    risk = 0.0
    for verdict in verdicts:
        label = canonical_verdict_label(verdict.verdict)
        if label == VerdictLabel.FAKE_REFERENCE:
            risk += 1.0
        elif label == VerdictLabel.POTENTIAL_REFERENCE:
            risk += 0.5
        elif label == VerdictLabel.INSUFFICIENT_EVIDENCE:
            risk += 0.3
        if verdict.needs_human_review:
            risk += 0.1
    return min(1.0, risk / max(len(verdicts), 1))


def render_markdown(report: CheckReport) -> str:
    summary = report.summary
    lines = [
        f"# Citation Check Report: {report.paper_id}",
        "",
        f"- Pipeline: `{report.pipeline_type}`",
        f"- Risk score: `{report.risk_score:.3f}`",
        f"- Requires human review: `{report.requires_human_review}`",
        "",
        "## Summary",
        "",
        f"- Total citations: {summary['total_citations']}",
        f"- Valid: {summary['valid']}",
        f"- Potential reference: {summary['potential_reference']}",
        f"- Fake reference: {summary['fake_reference']}",
        f"- Insufficient evidence: {summary['insufficient_evidence']}",
        f"- Needs human review: {summary['needs_human_review']}",
        "",
        "## Citation Verdicts",
        "",
    ]

    for verdict in report.citations:
        label = canonical_verdict_label(verdict.verdict).value
        comparison_summary = ""
        if getattr(verdict, "comparison", None):
            comparison_summary = str(verdict.comparison.get("summary", "") or "").strip()
        lines.extend(
            [
                f"### {verdict.citation_id}",
                f"- Verdict: `{label}`",
                f"- Confidence: `{verdict.confidence:.3f}`",
                f"- Evidence sources: {', '.join(verdict.evidence_sources) if verdict.evidence_sources else 'none'}",
                f"- Conflicts: {', '.join(verdict.conflicts) if verdict.conflicts else 'none'}",
                f"- Field comparison: {comparison_summary or 'n/a'}",
                f"- Review required: `{verdict.needs_human_review}`",
                f"- Reason: {verdict.adjudication_reason}",
                "",
            ]
        )
    return "\n".join(lines)
