from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from apps.source_checker.run import run_source_check
from packages.connectors.base import ConnectorOrchestrator, SQLiteCache
from packages.connectors.dblp_offline import DblpOfflineConnector


class SourcePipelineTests(unittest.TestCase):
    def test_source_pipeline_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            output_path = Path(tempdir) / "source_report.json"
            orchestrator = ConnectorOrchestrator(
                connectors=[DblpOfflineConnector("data/cache/dblp_mirror.jsonl")],
                cache=SQLiteCache(Path(tempdir) / "cache.sqlite"),
            )

            report = run_source_check(
                input_dir="data/fixtures/sample_source",
                out_path=output_path,
                orchestrator=orchestrator,
            )

            verdicts = {item["citation_id"]: item["verdict"] for item in report["citations"]}
            self.assertEqual(verdicts["bib:goodpaper"], "VALID")
            self.assertEqual(verdicts["bib:typoauthor"], "POTENTIAL_REFERENCE")
            self.assertEqual(verdicts["bib:fakepaper"], "INSUFFICIENT_EVIDENCE")
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
