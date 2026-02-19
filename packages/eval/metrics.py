from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClassificationMetrics:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int



def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def binary_metrics(labels: list[bool], predictions: list[bool]) -> ClassificationMetrics:
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have the same length")
    pairs = zip(labels, predictions)
    tp = sum(1 for y, p in pairs if y and p)
    pairs = zip(labels, predictions)
    fp = sum(1 for y, p in pairs if not y and p)
    pairs = zip(labels, predictions)
    fn = sum(1 for y, p in pairs if y and not p)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if precision + recall else 0.0

    return ClassificationMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        fp=fp,
        fn=fn,
    )


def expected_calibration_error(confidences: list[float], correctness: list[bool], buckets: int = 10) -> float:
    if not confidences:
        return 0.0
    bucket_size = 1.0 / buckets
    ece = 0.0
    total = len(confidences)

    for bucket in range(buckets):
        lower = bucket * bucket_size
        upper = (bucket + 1) * bucket_size
        members = [
            idx
            for idx, confidence in enumerate(confidences)
            if (confidence >= lower and confidence < upper) or (bucket == buckets - 1 and confidence == 1.0)
        ]
        if not members:
            continue
        avg_conf = sum(confidences[idx] for idx in members) / len(members)
        avg_acc = sum(1.0 for idx in members if correctness[idx]) / len(members)
        ece += abs(avg_acc - avg_conf) * (len(members) / total)
    return ece
