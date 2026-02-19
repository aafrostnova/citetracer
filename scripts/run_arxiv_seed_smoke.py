from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.pdf_checker.run import run_pdf_check
from apps.source_checker.run import run_source_check



def _has_tex_and_bib(source_dir: Path) -> bool:
    has_tex = any(source_dir.rglob("*.tex"))
    has_bib = any(source_dir.rglob("*.bib"))
    return has_tex and has_bib



def run(manifest_path: Path, out_dir: Path, pipelines: str) -> dict:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = payload.get("items", [])

    out_dir.mkdir(parents=True, exist_ok=True)
    reports_pdf_dir = out_dir / "reports" / "pdf"
    reports_source_dir = out_dir / "reports" / "source"
    reports_pdf_dir.mkdir(parents=True, exist_ok=True)
    reports_source_dir.mkdir(parents=True, exist_ok=True)

    run_pdf = pipelines in {"pdf", "both"}
    run_source = pipelines in {"source", "both"}

    processed_pdf = 0
    processed_source = 0
    skipped_missing_pdf = 0
    skipped_missing_source = 0
    skipped_source_without_bibtex = 0
    pdf_risk_scores: list[float] = []
    source_risk_scores: list[float] = []
    report_paths: list[str] = []

    for item in items:
        paper_id = str(item.get("paper_id", "unknown"))

        if run_pdf:
            pdf_path = Path(str(item.get("pdf_path", "")))
            if not pdf_path.exists():
                skipped_missing_pdf += 1
            else:
                output_path = reports_pdf_dir / f"{paper_id}.json"
                report = run_pdf_check(pdf_path, output_path)
                pdf_risk_scores.append(float(report.get("risk_score", 0.0)))
                report_paths.append(str(output_path))
                processed_pdf += 1

        if run_source:
            source_dir = Path(str(item.get("source_extract_dir", "")))
            if not source_dir.exists():
                skipped_missing_source += 1
            elif not _has_tex_and_bib(source_dir):
                skipped_source_without_bibtex += 1
            else:
                output_path = reports_source_dir / f"{paper_id}.json"
                report = run_source_check(source_dir, output_path)
                source_risk_scores.append(float(report.get("risk_score", 0.0)))
                report_paths.append(str(output_path))
                processed_source += 1

    avg_pdf_risk = sum(pdf_risk_scores) / len(pdf_risk_scores) if pdf_risk_scores else 0.0
    avg_source_risk = sum(source_risk_scores) / len(source_risk_scores) if source_risk_scores else 0.0

    summary = {
        "manifest": str(manifest_path),
        "pipelines": pipelines,
        "processed_pdf": processed_pdf,
        "processed_source": processed_source,
        "skipped_missing_pdf": skipped_missing_pdf,
        "skipped_missing_source": skipped_missing_source,
        "skipped_source_without_bibtex": skipped_source_without_bibtex,
        "avg_pdf_risk_score": round(avg_pdf_risk, 4),
        "avg_source_risk_score": round(avg_source_risk, 4),
        "report_paths": report_paths,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PDF/source checkers on arXiv seed manifest outputs.")
    parser.add_argument("--manifest", required=True, help="Path to arXiv seed manifest JSON.")
    parser.add_argument("--out", default="artifacts/arxiv_smoke", help="Directory for smoke reports.")
    parser.add_argument(
        "--pipelines",
        default="pdf",
        choices=["pdf", "source", "both"],
        help="Which pipelines to execute for each manifest item.",
    )
    return parser



def main() -> None:
    args = build_arg_parser().parse_args()
    summary = run(Path(args.manifest), Path(args.out), args.pipelines)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
