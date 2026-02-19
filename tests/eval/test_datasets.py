from __future__ import annotations

import unittest

from packages.eval.datasets import load_manifest


class DatasetTests(unittest.TestCase):
    def test_load_manifest(self) -> None:
        manifest = load_manifest("data/silver/synthetic_stress_manifest.json")
        self.assertEqual(manifest.name, "synthetic_stress")
        self.assertGreaterEqual(len(manifest.items), 2)


if __name__ == "__main__":
    unittest.main()
