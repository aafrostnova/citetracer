"""Regression tests for the bugs surfaced by ccs_b_part2_errors.xlsx analysis.

Each test maps to a specific row in
``artifacts/ccs_b_part2_errors.xlsx`` and the matching plan entry in
``/home/mingzhel_umass_edu/.claude/plans/silly-crunching-quilt.md``.
"""
from __future__ import annotations

import unittest

from apps.pdf_checker.ingest.citation_parser import _extract_year
from packages.core.cascading_agents import is_non_academic_citation
from packages.core.models import CitationRecord


class ExtractYearGuardsTests(unittest.TestCase):
    """Fix 1: _extract_year must skip page-range tails and prefer parens."""

    def test_page_range_tail_not_picked_as_year(self) -> None:
        # row 7 in xlsx: "...IEEE TPAMI 40, 8 (2018), 1964-1978" — heuristic
        # used to return 1978 (the page-range tail), should return 2018.
        raw = (
            "[42] Zhuwen Li et al. 2018. Simultaneous clustering. "
            "IEEE Transactions on Pattern Analysis and Machine Intelligence "
            "40, 8 (2018), 1964-1978. doi:10.1109/TPAMI.2017.2739147"
        )
        self.assertEqual(_extract_year(raw), 2018)

    def test_date_span_not_picked_as_year(self) -> None:
        # row 185: "June 20-25, 2021" — heuristic used to return 1981 from
        # garbled OCR; with page-range scrub the only year left is 2021.
        raw = (
            "[41] Wang and Yi. 2021. Secure Yannakakis. "
            "SIGMOD '21: International Conference, San Diego, CA, USA, "
            "June 20-25, 2021. ACM."
        )
        self.assertEqual(_extract_year(raw), 2021)

    def test_parens_year_overrides_default_trailing(self) -> None:
        # Without the parens preference, default would return the LAST year
        # candidate. With the new parens preference, return the parens year.
        raw = "Foo. Bar. Journal of Studies (2018), pp. 1964-2025. arXiv:2401.99999"
        self.assertEqual(_extract_year(raw), 2018)

    def test_no_parens_no_page_range_unchanged(self) -> None:
        # Smoke control: existing behaviour preserved on simple input.
        self.assertEqual(_extract_year("Smith 2024 Foo Bar"), 2024)


class IsNonAcademicArxivExitTests(unittest.TestCase):
    """Fix 4: arxiv/DOI presence overrides web-citation pattern."""

    def test_ieee_arxiv_format_is_academic(self) -> None:
        # row 12+ in xlsx (paper2092): IEEE-style "[Online]. Available:
        # https://arxiv.org/abs/...". Used to fire web-citation pattern;
        # should now early-exit as academic.
        cite = CitationRecord(
            citation_id="c1",
            raw_text=(
                '[3] A. Miah and Y. Bi, "Title," 2409.01952, 2024. '
                "[Online]. Available: https://arxiv.org/abs/2409.01952."
            ),
        )
        self.assertFalse(is_non_academic_citation(cite))

    def test_doi_present_is_academic_even_with_accessed(self) -> None:
        cite = CitationRecord(
            citation_id="c2",
            raw_text=(
                "Foo, Bar. 2024. Title. doi:10.1109/test.2024.123. "
                "Accessed: 2024-05-01."
            ),
        )
        self.assertFalse(is_non_academic_citation(cite))

    def test_no_identifier_blog_still_non_academic(self) -> None:
        # Control: real non-academic citation still flagged.
        cite = CitationRecord(
            citation_id="c3",
            raw_text="OpenAI. ChatGPT-5. https://chat.openai.com. Accessed: 2026-01-09.",
            url="https://chat.openai.com",
        )
        self.assertTrue(is_non_academic_citation(cite))


if __name__ == "__main__":
    unittest.main()
