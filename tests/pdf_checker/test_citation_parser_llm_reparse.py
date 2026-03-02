from __future__ import annotations

import unittest
from unittest.mock import patch

from apps.pdf_checker.ingest.citation_parser import parse_reference_entries


class CitationParserLLMReparseTests(unittest.TestCase):
    def test_llm_reparse_overwrites_all_core_fields_when_triggered(self) -> None:
        entry = "Schutz, P. A. Emotion in education, 2007."
        base = parse_reference_entries([entry])[0]
        self.assertEqual(base.venue, "")
        self.assertEqual(base.year, 2007)
        self.assertTrue(base.title)
        self.assertTrue(base.authors)

        with patch(
            "apps.pdf_checker.ingest.citation_parser._run_llm_reparse_on_raw",
            return_value={
                "title": "Emotion in education (Reparsed)",
                "authors": ["Schutz, P. A."],
                "venue": "Education Monographs, 2007",
                "year": 2006,
                "doi": "10.1234/edu.2006.1",
                "arxiv_id": "2401.01234",
                "url": "https://example.org/edu",
            },
        ) as mocked:
            records = parse_reference_entries(
                [entry],
                llm_reparse_config={
                    "enabled": True,
                    "model_path": "/project/pi_shiqingma_umass_edu/mingzheli/model/Qwen3-0.6B",
                    "max_new_tokens": 64,
                    "temperature": 0.0,
                },
            )

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.title, "Emotion in education (Reparsed)")
        self.assertEqual(record.authors, ["Schutz, P. A."])
        self.assertEqual(record.venue, "Education Monographs, 2007")
        self.assertEqual(record.year, 2006)
        self.assertEqual(record.doi, "10.1234/edu.2006.1")
        self.assertEqual(record.arxiv_id, "2401.01234")
        self.assertEqual(record.url, "https://example.org/edu")
        self.assertEqual(record.parsed_fields.get("llm_reparse_mode"), "overwrite_core_and_identifier_fields")
        self.assertIn("title", record.parsed_fields.get("llm_reparse_applied_fields", []))
        self.assertIn("authors", record.parsed_fields.get("llm_reparse_applied_fields", []))
        self.assertIn("venue", record.parsed_fields.get("llm_reparse_applied_fields", []))
        self.assertIn("year", record.parsed_fields.get("llm_reparse_applied_fields", []))
        self.assertIn("doi", record.parsed_fields.get("llm_reparse_applied_fields", []))
        self.assertIn("arxiv_id", record.parsed_fields.get("llm_reparse_applied_fields", []))
        self.assertIn("url", record.parsed_fields.get("llm_reparse_applied_fields", []))
        mocked.assert_called_once()

    def test_llm_reparse_skipped_when_disabled(self) -> None:
        entry = "In International Conference on Learning Representations, 2021."
        with patch("apps.pdf_checker.ingest.citation_parser._run_llm_reparse_on_raw") as mocked:
            _ = parse_reference_entries(
                [entry],
                llm_reparse_config={
                    "enabled": False,
                    "model_path": "/project/pi_shiqingma_umass_edu/mingzheli/model/Qwen3-0.6B",
                },
            )
        mocked.assert_not_called()


if __name__ == "__main__":
    unittest.main()
