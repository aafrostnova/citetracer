from __future__ import annotations

import unittest

from packages.core.adjudicate import adjudicate
from packages.core.models import CandidateMatch, CitationRecord, EvidenceTrace, VerdictLabel


class AdjudicateTests(unittest.TestCase):
    def test_valid_verdict_for_strong_match(self) -> None:
        citation = CitationRecord(citation_id="c1", title="Attention Is All You Need", authors=["Ashish Vaswani"], year=2017)
        candidate = CandidateMatch(connector="dblp_offline", score=0.95, title="Attention Is All You Need", authors=["Ashish Vaswani"], year=2017)
        evidence = [
            EvidenceTrace(
                connector="dblp_offline",
                query={"title": citation.title},
                latency_ms=10,
                cache_hit=False,
                source_health=1.0,
                candidates_count=1,
                error=None,
            )
        ]
        verdict = adjudicate(citation, [candidate], evidence)
        self.assertEqual(verdict.verdict, VerdictLabel.VALID)

    def test_insufficient_evidence_without_candidates(self) -> None:
        citation = CitationRecord(citation_id="c2", title="Imaginary Reference")
        evidence = [
            EvidenceTrace(
                connector="dblp_offline",
                query={"title": citation.title},
                latency_ms=4,
                cache_hit=False,
                source_health=1.0,
                candidates_count=0,
                error=None,
            )
        ]
        verdict = adjudicate(citation, [], evidence)
        self.assertEqual(verdict.verdict, VerdictLabel.INSUFFICIENT_EVIDENCE)


if __name__ == "__main__":
    unittest.main()
