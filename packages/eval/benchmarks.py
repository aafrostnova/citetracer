from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from apps.pdf_checker.run import run_pdf_check
from apps.source_checker.run import run_source_check

from .datasets import load_ground_truth, load_manifest
from .metrics import binary_metrics


def _canon(label: str) -> str:
    value = str(label or "").strip().upper()
    if value == "FLAWED_CITATION":
        return "POTENTIAL_REFERENCE"
    if value == "SUSPECTED_HALLUCINATION":
        return "FAKE_REFERENCE"
    return value or "FAKE_REFERENCE"


@dataclass
class SuiteResult:
    suite_name: str
    total_items: int
    total_citations: int
    hallucination_precision: float
    hallucination_recall: float
    hallucination_f1: float
    flawed_precision: float
    flawed_recall: float
    flawed_f1: float



def _run_item(pipeline: str, input_path: str, output_path: Path) -> dict:
    if pipeline.upper() == "SOURCE":
        return run_source_check(input_path, output_path)
    if pipeline.upper() == "PDF":
        return run_pdf_check(input_path, output_path)
    raise ValueError(f"Unsupported pipeline: {pipeline}")


def run_suite(manifest_path: str | Path, out_dir: str | Path) -> dict:
    manifest = load_manifest(manifest_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hall_labels: list[bool] = []
    hall_preds: list[bool] = []
    flawed_labels: list[bool] = []
    flawed_preds: list[bool] = []
    total_citations = 0

    per_item = []

    for idx, item in enumerate(manifest.items, start=1):
        report_path = out_dir / f"{item.paper_id}_{idx}.json"
        report = _run_item(item.pipeline, item.input_path, report_path)
        truth = load_ground_truth(item.ground_truth_path)

        for citation in report.get("citations", []):
            citation_id = citation["citation_id"]
            predicted = _canon(citation["verdict"])
            label = _canon(truth.get(citation_id, "FAKE_REFERENCE"))

            hall_labels.append(label == "FAKE_REFERENCE")
            hall_preds.append(predicted == "FAKE_REFERENCE")

            flawed_labels.append(label == "POTENTIAL_REFERENCE")
            flawed_preds.append(predicted == "POTENTIAL_REFERENCE")

            total_citations += 1

        per_item.append(
            {
                "paper_id": item.paper_id,
                "pipeline": item.pipeline,
                "report_path": str(report_path),
                "summary": report.get("summary", {}),
                "risk_score": report.get("risk_score", 0.0),
            }
        )

    hall_metric = binary_metrics(hall_labels, hall_preds)
    flawed_metric = binary_metrics(flawed_labels, flawed_preds)

    suite_result = SuiteResult(
        suite_name=manifest.name,
        total_items=len(manifest.items),
        total_citations=total_citations,
        hallucination_precision=hall_metric.precision,
        hallucination_recall=hall_metric.recall,
        hallucination_f1=hall_metric.f1,
        flawed_precision=flawed_metric.precision,
        flawed_recall=flawed_metric.recall,
        flawed_f1=flawed_metric.f1,
    )

    payload = {
        "suite": suite_result.__dict__,
        "per_item": per_item,
        "acceptance_gates": {
            "hallucination_precision_target": 0.92,
            "top10_alert_precision_target": 0.85,
            "source_unresolved_target_max": 0.05,
            "pdf_field_f1_target": 0.85,
        },
    }

    (out_dir / f"{manifest.name}_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
