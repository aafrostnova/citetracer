from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from apps.pdf_checker.run import run_pdf_check
from packages.connectors.base import ConnectorOrchestrator, SQLiteCache
from packages.connectors.dblp_offline import DblpOfflineConnector


class PDFPipelineTests(unittest.TestCase):
    def test_pdf_pipeline_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            output_path = Path(tempdir) / "pdf_report.json"
            orchestrator = ConnectorOrchestrator(
                connectors=[DblpOfflineConnector("data/cache/dblp_mirror.jsonl")],
                cache=SQLiteCache(Path(tempdir) / "cache.sqlite"),
            )

            report = run_pdf_check(
                input_pdf="data/fixtures/sample_pdf/sample.pdf",
                out_path=output_path,
                orchestrator=orchestrator,
            )

            verdicts = {item["citation_id"]: item["verdict"] for item in report["citations"]}
            self.assertEqual(verdicts["pdf-ref:1"], "VALID")
            self.assertEqual(verdicts["pdf-ref:2"], "INSUFFICIENT_EVIDENCE")
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
