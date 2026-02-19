from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from packages.eval.benchmarks import run_suite


class BenchmarkTests(unittest.TestCase):
    def test_run_suite_offline(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            old_value = os.getenv("CITATION_CHECKER_OFFLINE_ONLY")
            os.environ["CITATION_CHECKER_OFFLINE_ONLY"] = "1"
            try:
                payload = run_suite(
                    manifest_path="data/silver/synthetic_stress_manifest.json",
                    out_dir=Path(tempdir) / "eval",
                )
            finally:
                if old_value is None:
                    os.environ.pop("CITATION_CHECKER_OFFLINE_ONLY", None)
                else:
                    os.environ["CITATION_CHECKER_OFFLINE_ONLY"] = old_value

            self.assertEqual(payload["suite"]["suite_name"], "synthetic_stress")
            self.assertGreaterEqual(payload["suite"]["total_citations"], 1)


if __name__ == "__main__":
    unittest.main()
