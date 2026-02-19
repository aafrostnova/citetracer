from __future__ import annotations

import unittest

from packages.core.normalize import author_overlap_score, normalize_author, normalize_venue


class NormalizeTests(unittest.TestCase):
    def test_name_alias_normalization(self) -> None:
        self.assertEqual(normalize_author("Mike Jordan"), "michael jordan")

    def test_venue_alias(self) -> None:
        self.assertEqual(normalize_venue("Advances in Neural Information Processing Systems"), "neurips")

    def test_author_overlap_with_initial_variation(self) -> None:
        left = ["Michael I. Jordan", "Yann LeCun"]
        right = ["M. Jordan", "Yann Lecun"]
        self.assertGreaterEqual(author_overlap_score(left, right), 0.5)


if __name__ == "__main__":
    unittest.main()
