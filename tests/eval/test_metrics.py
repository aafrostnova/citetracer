from __future__ import annotations

import unittest

from packages.eval.metrics import binary_metrics, expected_calibration_error


class MetricsTests(unittest.TestCase):
    def test_binary_metrics(self) -> None:
        labels = [True, False, True, False]
        predictions = [True, False, False, False]
        metrics = binary_metrics(labels, predictions)
        self.assertAlmostEqual(metrics.precision, 1.0)
        self.assertAlmostEqual(metrics.recall, 0.5)
        self.assertAlmostEqual(metrics.f1, 2 / 3)

    def test_expected_calibration_error(self) -> None:
        confidences = [0.9, 0.8, 0.2, 0.1]
        correctness = [True, True, False, True]
        ece = expected_calibration_error(confidences, correctness, buckets=2)
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)


if __name__ == "__main__":
    unittest.main()
