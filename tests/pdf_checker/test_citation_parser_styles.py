from __future__ import annotations

import unittest

from apps.pdf_checker.ingest.citation_parser import parse_reference_entry


class CitationParserStyleTests(unittest.TestCase):
    def test_lncs_colon_style(self) -> None:
        entry = (
            "[77] Xu, K., Hu, W., Leskovec, J., Jegelka, S.: How powerful are graph neural networks? "
            "arXiv preprint arXiv:1810.00826 (2018)"
        )
        record = parse_reference_entry(entry, "pdf-ref:77")
        self.assertEqual(record.title, "How powerful are graph neural networks?")
        self.assertEqual(record.authors, ["Xu, K.", "Hu, W.", "Leskovec, J.", "Jegelka, S."])
        self.assertIn("lncs_or_springer_numeric", record.parsed_fields.get("style_hint", ""))

    def test_apa_style(self) -> None:
        entry = (
            "Smith, J. A., & Doe, R. (2020). Deep learning for time series forecasting. "
            "Journal of AI Research, 10(2), 100-120."
        )
        record = parse_reference_entry(entry, "pdf-ref:1")
        self.assertEqual(record.title, "Deep learning for time series forecasting")
        self.assertEqual(record.parsed_fields.get("style_hint"), "apa")

    def test_mla_style(self) -> None:
        entry = (
            'Smith, John, and Jane Doe. "Learning on Graphs." '
            "Journal of Machine Learning, vol. 12, no. 3, 2021, pp. 12-34."
        )
        record = parse_reference_entry(entry, "pdf-ref:2")
        self.assertEqual(record.title, "Learning on Graphs.")
        self.assertEqual(record.parsed_fields.get("style_hint"), "mla_or_chicago")

    def test_chicago_style(self) -> None:
        entry = (
            'Smith, John, and Jane Doe. "Robust Optimization for Vision." '
            "IEEE Transactions on Pattern Analysis and Machine Intelligence 45, no. 2 (2023): 100-120."
        )
        record = parse_reference_entry(entry, "pdf-ref:3")
        self.assertEqual(record.title, "Robust Optimization for Vision.")
        self.assertEqual(record.parsed_fields.get("style_hint"), "mla_or_chicago")

    def test_harvard_style(self) -> None:
        entry = (
            "Smith, J. and Doe, R., 2019. Contrastive representation learning in medical imaging. "
            "Medical Image Analysis, 58, pp.1-12."
        )
        record = parse_reference_entry(entry, "pdf-ref:4")
        self.assertEqual(record.title, "Contrastive representation learning in medical imaging")
        self.assertEqual(record.parsed_fields.get("style_hint"), "harvard")

    def test_vancouver_style(self) -> None:
        entry = "[12] Smith AB, Doe CD. A practical guide to sparse autoencoders. Neural Computation. 2022;34(5):123-140."
        record = parse_reference_entry(entry, "pdf-ref:5")
        self.assertEqual(record.title, "A practical guide to sparse autoencoders")
        self.assertEqual(record.parsed_fields.get("style_hint"), "vancouver_or_numeric")


if __name__ == "__main__":
    unittest.main()
