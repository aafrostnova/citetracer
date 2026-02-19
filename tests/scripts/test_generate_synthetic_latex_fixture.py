from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "generate_synthetic_latex_fixture.py"
SPEC = importlib.util.spec_from_file_location("generate_synthetic_latex_fixture", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec from {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules["generate_synthetic_latex_fixture"] = MODULE
SPEC.loader.exec_module(MODULE)
run = MODULE.run


class GenerateSyntheticLatexFixtureTests(unittest.TestCase):
    def test_run_generates_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            out_dir = Path(tempdir) / "fixture"
            result = run(out_dir)

            self.assertTrue((out_dir / "main.tex").exists())
            self.assertTrue((out_dir / "refs.bib").exists())
            self.assertTrue((out_dir / "expected_labels.json").exists())
            self.assertEqual(result["citation_count"], 3)

            tex_content = (out_dir / "main.tex").read_text(encoding="utf-8")
            self.assertIn("\\cite{vaswani2017attention}", tex_content)
            self.assertIn("\\cite{quantum_unicorn_2025}", tex_content)


if __name__ == "__main__":
    unittest.main()
